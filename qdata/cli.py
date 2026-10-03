import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qdata.adjust import rebuild_adjusted_layer
from qdata.auth import (
    clear_cached_token,
    get_cached_token,
    get_credential,
    mask_secret,
    save_credential_to_file,
)
from qdata.catalog import CatalogManager
from qdata.config import get_settings
from qdata.doctor import run_doctor
from qdata.providers import get_provider, list_providers
from qdata.quality import QualityFailureError, validate_bars
from qdata.store import (
    CANONICAL_COLUMNS,
    LockHeldError,
    ManifestManager,
    normalize_timeframe,
    read_partition_parquet,
)
from qdata.sync import SyncEngine

app = typer.Typer(
    name="qdata",
    help="CLI + Python library that ingests, stores, catalogs, and serves market data.",
    no_args_is_help=True,
)
catalog_app = typer.Typer(help="Inspect and manage the data catalog.", no_args_is_help=True)
auth_app = typer.Typer(help="Manage provider authentication and credentials.", no_args_is_help=True)
rebuild_app = typer.Typer(help="Rebuild derived layers.", no_args_is_help=True)

app.add_typer(catalog_app, name="catalog")
app.add_typer(auth_app, name="auth")
app.add_typer(rebuild_app, name="rebuild")

console = Console()
err_console = Console(stderr=True)


def output(data: Any, is_json: bool, rich_renderer=None) -> None:
    """Helper to route output either to JSON or Rich formatting."""
    if is_json:
        # Custom serializer for dates/timestamps
        def json_serial(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        print(json.dumps(data, indent=2, default=json_serial))
    else:
        if rich_renderer:
            rich_renderer()
        else:
            console.print(data)


# ---------------------------------------------------------
# qdata init
# ---------------------------------------------------------
@app.command(name="init")
def init_cmd(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without creating files."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Scaffold data/ directory layout and catalog tables."""
    settings = get_settings(override_data_dir=data_dir)
    target_dirs = [
        settings.data_dir,
        settings.catalog_dir,
        settings.raw_dir,
        settings.adjusted_dir,
        settings.tokens_dir,
    ]

    result = {
        "status": "dry_run" if dry_run else "success",
        "data_dir": str(settings.data_dir),
        "scaffolded_dirs": [str(d) for d in target_dirs],
        "catalog_initialized": True,
    }

    if not dry_run:
        for d in target_dirs:
            d.mkdir(parents=True, exist_ok=True)
        catalog = CatalogManager(settings.data_dir)
        catalog.init_catalog()
        manifest = ManifestManager(settings.data_dir)
        if not manifest.manifest_file.exists():
            manifest.save({})

    def render():
        action = "[yellow][DRY RUN] Would scaffold[/yellow]" if dry_run else "[green]Scaffolded[/green]"
        console.print(Panel(f"{action} qdata storage at [bold]{settings.data_dir}[/bold]", title="qdata init"))
        table = Table(title="Storage Directories")
        table.add_column("Directory", style="cyan")
        table.add_column("Status", style="green")
        for d in target_dirs:
            table.add_row(str(d), "Planned" if dry_run else "Created / Ready")
        console.print(table)

    output(result, json_out, render)
    raise typer.Exit(code=0)


# ---------------------------------------------------------
# qdata sync
# ---------------------------------------------------------
@app.command(name="sync")
def sync_cmd(
    provider: str = typer.Option("mock", "--provider", help="Provider name (e.g. mock, upstox, fyers, amfi)."),
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Ticker symbol to sync (e.g. AAPL or 120503)."),
    timeframe: str = typer.Option("1min", "--timeframe", help="Timeframe (1min, 1d)."),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD or ISO string)."),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD or ISO string)."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate sync without writing."),
    strict_quality: bool = typer.Option(False, "--strict-quality", help="Fail if quality checks warn or fail."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Sync market data from provider, validate, and atomically merge into open partition."""
    settings = get_settings(override_data_dir=data_dir)
    engine = SyncEngine(settings.data_dir)

    # AMFI only provides daily NAV data; default 1min to 1d automatically for convenience
    actual_tf = "1d" if provider.lower() == "amfi" and timeframe == "1min" else timeframe
    symbols = [symbol.upper()] if symbol else None

    try:
        results = engine.sync(
            provider_name=provider,
            symbols=symbols,
            timeframe=actual_tf,
            start=start,
            end=end,
            dry_run=dry_run,
            strict_quality=strict_quality,
        )
    except LockHeldError as e:
        err_msg = {"error": "lock_held", "message": str(e), "exit_code": 3}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Sync Lock Error:[/bold red] {e}"))
        raise typer.Exit(code=3)
    except QualityFailureError as e:
        err_msg = {"error": "quality_fail", "message": str(e), "details": e.report.to_dict(), "exit_code": 2}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Quality Failure:[/bold red] {e}"))
        raise typer.Exit(code=2)
    except Exception as e:
        err_msg = {"error": "sync_failed", "message": str(e), "exit_code": 1}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Error during sync:[/bold red] {e}"))
        raise typer.Exit(code=1)

    # Check if any quality failed
    has_quality_fail = any(r.get("quality", {}).get("status") == "FAIL" for r in results)
    exit_code = 2 if has_quality_fail else 0

    def render():
        mode = " [yellow](DRY RUN)[/yellow]" if dry_run else ""
        table = Table(title=f"Sync Results{mode}")
        table.add_column("Symbol", style="cyan")
        table.add_column("Timeframe", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Synced Bars", justify="right")
        table.add_column("Open Partition", style="dim")
        table.add_column("Quality", style="green")

        for r in results:
            q_status = r.get("quality", {}).get("status", "N/A")
            q_color = "green" if q_status == "PASSED" else ("yellow" if q_status == "WARN" else "red")
            table.add_row(
                r.get("symbol", ""),
                r.get("timeframe", ""),
                r.get("status", ""),
                str(r.get("synced_bars", 0)),
                str(r.get("open_partition", "")),
                f"[{q_color}]{q_status}[/{q_color}]",
            )
        console.print(table)

        # Print quality warning details if any
        for r in results:
            q = r.get("quality", {})
            if q.get("status") == "WARN" and q.get("details"):
                console.print(f"\n[yellow]Quality Warning Details ({r.get('symbol')}):[/yellow]")
                for detail in q.get("details", []):
                    console.print(f"  • {detail}")

    output(results, json_out, render)
    raise typer.Exit(code=exit_code)


# ---------------------------------------------------------
# qdata catalog list
# ---------------------------------------------------------
@catalog_app.command(name="list")
def catalog_list_cmd(
    layer: Optional[str] = typer.Option(None, "--layer", help="Filter by layer ('raw' or 'adjusted')."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """List datasets registered in the catalog."""
    settings = get_settings(override_data_dir=data_dir)
    catalog = CatalogManager(settings.data_dir)
    df = catalog.list_datasets(layer=layer)

    if json_out:
        # Convert df to records
        output(df.to_dict(orient="records"), is_json=True)
        raise typer.Exit(code=0)

    table = Table(title="Catalog Datasets")
    table.add_column("Symbol", style="cyan")
    table.add_column("Timeframe", style="magenta")
    table.add_column("Layer", style="blue")
    table.add_column("Provider", style="green")
    table.add_column("Start (UTC)", style="dim")
    table.add_column("End (UTC)", style="dim")
    table.add_column("Rows", justify="right")
    table.add_column("Open Partition", style="yellow")
    table.add_column("Quality", style="bold")

    for _, row in df.iterrows():
        table.add_row(
            str(row["symbol"]),
            str(row["timeframe"]),
            str(row["layer"]),
            str(row["provider"]),
            str(row["start_ts"]),
            str(row["end_ts"]),
            str(row["row_count"]),
            str(row["open_partition"]),
            str(row["quality_status"]),
        )
    console.print(table)
    raise typer.Exit(code=0)


# ---------------------------------------------------------
# qdata catalog show <sym> <tf>
# ---------------------------------------------------------
@catalog_app.command(name="show")
def catalog_show_cmd(
    symbol: str = typer.Argument(..., help="Symbol (e.g. AAPL)"),
    timeframe: str = typer.Argument(..., help="Timeframe (e.g. 1min, 1d)"),
    layer: str = typer.Option("raw", "--layer", help="Dataset layer ('raw' or 'adjusted')"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Show detailed record and partition files for a specific dataset."""
    settings = get_settings(override_data_dir=data_dir)
    catalog = CatalogManager(settings.data_dir)
    manifest = ManifestManager(settings.data_dir)

    ds = catalog.get_dataset(symbol, timeframe, layer=layer)
    if not ds:
        err_msg = {"error": "not_found", "message": f"No dataset found for {symbol} {timeframe} ({layer})"}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Not Found:[/bold red] {err_msg['message']}"))
        raise typer.Exit(code=1)

    norm_tf = normalize_timeframe(timeframe)
    sym = symbol.upper()
    prefix = f"{layer.lower()}/{sym}/{norm_tf}/"

    manifest_files = manifest.load()
    partitions = []
    for fpath, meta in manifest_files.items():
        if fpath.startswith(prefix):
            partitions.append({
                "partition_file": fpath,
                "is_open": (fpath == ds.get("open_partition")),
                "row_count": meta.get("row_count"),
                "sha256": meta.get("sha256"),
                "size_bytes": meta.get("size_bytes"),
                "updated_at": meta.get("updated_at"),
            })

    partitions.sort(key=lambda x: x["partition_file"])
    data_res = {
        "dataset": ds,
        "partitions": partitions,
    }

    def render():
        console.print(Panel(f"Dataset: [bold cyan]{sym}[/bold cyan] ({norm_tf}, {layer})", title="Catalog Show"))
        # Meta table
        meta_table = Table.grid(padding=(0, 2))
        meta_table.add_column(style="bold")
        meta_table.add_column()
        meta_table.add_row("Provider:", str(ds.get("provider")))
        meta_table.add_row("Start TS:", str(ds.get("start_ts")))
        meta_table.add_row("End TS:", str(ds.get("end_ts")))
        meta_table.add_row("Total Rows:", str(ds.get("row_count")))
        meta_table.add_row("Open Partition:", str(ds.get("open_partition")))
        meta_table.add_row("Quality Status:", str(ds.get("quality_status")))
        console.print(meta_table)
        console.print("")

        part_table = Table(title="Partition Files")
        part_table.add_column("File", style="cyan")
        part_table.add_column("State", style="bold")
        part_table.add_column("Rows", justify="right")
        part_table.add_column("Size", justify="right")
        part_table.add_column("SHA-256", style="dim")

        for p in partitions:
            state = "[yellow]OPEN[/yellow]" if p["is_open"] else "[blue]FROZEN[/blue]"
            part_table.add_row(
                p["partition_file"],
                state,
                str(p["row_count"]),
                f"{p['size_bytes'] / 1024:.1f} KB" if p.get("size_bytes") else "N/A",
                p["sha256"][:12] + "..." if p.get("sha256") else "N/A",
            )
        console.print(part_table)

    output(data_res, json_out, render)
    raise typer.Exit(code=0)


# ---------------------------------------------------------
# qdata inspect <sym> <tf>
# ---------------------------------------------------------
@app.command(name="inspect")
def inspect_cmd(
    symbol: str = typer.Argument(..., help="Ticker symbol"),
    timeframe: str = typer.Argument(..., help="Timeframe (1min, 1d)"),
    layer: str = typer.Option("raw", "--layer", help="Layer ('raw' or 'adjusted')"),
    n_rows: int = typer.Option(5, "--n-rows", help="Number of head/tail sample rows to show"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Sample rows and calculate quality statistics for a dataset."""
    settings = get_settings(override_data_dir=data_dir)
    catalog = CatalogManager(settings.data_dir)
    ds = catalog.get_dataset(symbol, timeframe, layer=layer)

    if not ds:
        err_msg = {"error": "not_found", "message": f"No dataset found for {symbol} {timeframe}."}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Not Found:[/bold red] {err_msg['message']}"))
        raise typer.Exit(code=1)

    open_p = ds.get("open_partition")
    if not open_p or not (settings.data_dir / open_p).exists():
        err_msg = {"error": "file_missing", "message": f"Partition file {open_p} missing on disk."}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Error:[/bold red] {err_msg['message']}"))
        raise typer.Exit(code=1)

    df = read_partition_parquet(settings.data_dir / open_p)
    _, quality_rep = validate_bars(df, timeframe=timeframe, strict=False)

    head_sample = df.head(n_rows).to_dict(orient="records")
    tail_sample = df.tail(n_rows).to_dict(orient="records")

    res = {
        "symbol": symbol.upper(),
        "timeframe": normalize_timeframe(timeframe),
        "partition_inspected": open_p,
        "rows_in_partition": len(df),
        "quality": quality_rep.to_dict(),
        "sample_head": head_sample,
        "sample_tail": tail_sample,
    }

    def render():
        console.print(Panel(f"Inspect: [bold cyan]{symbol.upper()}[/bold cyan] ({open_p})", title="Data Inspection"))
        q = quality_rep
        q_color = "green" if q.status == "PASSED" else ("yellow" if q.status == "WARN" else "red")
        console.print(f"Quality Status: [{q_color} bold]{q.status}[/{q_color} bold]")
        console.print(f"Total Bars: {q.total_bars} | Valid: {q.valid_bars} | Duplicates Dropped: {q.duplicates_dropped}")
        console.print(f"OHLC Violations: {q.ohlc_violations} | Insane Prices: {q.insane_prices} | NaNs: {q.nan_values}")
        console.print(f"Gaps Detected: {q.gaps_detected} | Max Gap: {q.max_gap_seconds:.0f}s")
        if q.details:
            console.print("Details: " + "; ".join(q.details))

        console.print("\n[bold]Sample Rows (Tail):[/bold]")
        tbl = Table()
        for col in ["ts", "open", "high", "low", "close", "volume"]:
            tbl.add_column(col)
        for row in tail_sample:
            tbl.add_row(
                str(row.get("ts")),
                str(row.get("open")),
                str(row.get("high")),
                str(row.get("low")),
                str(row.get("close")),
                str(row.get("volume")),
            )
        console.print(tbl)

    output(res, json_out, render)
    code = 2 if quality_rep.status == "FAIL" else 0
    raise typer.Exit(code=code)


# ---------------------------------------------------------
# qdata verify
# ---------------------------------------------------------
@app.command(name="verify")
def verify_cmd(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Re-hash all data files vs manifest.json."""
    settings = get_settings(override_data_dir=data_dir)
    manifest = ManifestManager(settings.data_dir)
    results = manifest.verify_all()

    mismatches = [r for r in results if r["status"] != "ok"]
    overall_ok = len(mismatches) == 0

    res = {
        "status": "ok" if overall_ok else "mismatch_detected",
        "verified_files": len(results),
        "mismatches_count": len(mismatches),
        "results": results,
    }

    def render():
        table = Table(title="Manifest Verification")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Expected SHA-256", style="dim")
        table.add_column("Actual SHA-256", style="dim")
        table.add_column("Rows")

        for r in results:
            st = r["status"]
            st_style = "green" if st == "ok" else "red"
            table.add_row(
                r["file"],
                f"[{st_style}]{st.upper()}[/{st_style}]",
                str(r.get("expected_sha", ""))[:12] + "...",
                str(r.get("actual_sha", ""))[:12] + "..." if r.get("actual_sha") else "None",
                str(r.get("actual_rows", "")),
            )
        console.print(table)
        if not overall_ok:
            console.print(f"[bold red]Found {len(mismatches)} mismatches/missing files![/bold red]")

    output(res, json_out, render)
    raise typer.Exit(code=0 if overall_ok else 1)


# ---------------------------------------------------------
# qdata rebuild adjusted
# ---------------------------------------------------------
@rebuild_app.command(name="adjusted")
def rebuild_adjusted_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Ticker symbol to rebuild"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe", help="Optional timeframe filter"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without writing files"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Regenerate derived 'adjusted' layer from raw parquet files."""
    settings = get_settings(override_data_dir=data_dir)
    res = rebuild_adjusted_layer(
        symbol=symbol,
        timeframe=timeframe,
        data_dir=settings.data_dir,
        dry_run=dry_run,
    )

    def render():
        mode = " [yellow](DRY RUN)[/yellow]" if dry_run else ""
        console.print(Panel(f"Rebuild Adjusted: {symbol.upper()}{mode}", title="qdata rebuild adjusted"))
        table = Table(title="Rebuilt Files")
        table.add_column("File", style="cyan")
        for f in res.get("rebuilt_files", []):
            table.add_row(f)
        console.print(table)
        console.print(f"Total bars processed: {res.get('total_bars', 0)}")

    output(res, json_out, render)
    code = 0 if res.get("status") in ("success", "dry_run") else 1
    raise typer.Exit(code=code)


# ---------------------------------------------------------
# qdata doctor
# ---------------------------------------------------------
@app.command(name="doctor")
def doctor_cmd(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Diagnose configuration, credentials security, locks, and catalog <-> fs consistency."""
    settings = get_settings(override_data_dir=data_dir)
    doc_res = run_doctor(settings.data_dir)

    def render():
        status_color = "green" if doc_res["status"] == "PASS" else "red"
        console.print(Panel(f"System Health: [{status_color} bold]{doc_res['status']}[/{status_color} bold]", title="qdata doctor"))
        table = Table(title="Doctor Diagnostics")
        table.add_column("Check", style="cyan bold")
        table.add_column("Status", style="bold")
        table.add_column("Message")

        for c in doc_res.get("checks", []):
            st = c["status"]
            st_style = "green" if st == "PASS" else ("yellow" if st == "WARN" else "red")
            table.add_row(c["name"], f"[{st_style}]{st}[/{st_style}]", c["message"])
        console.print(table)

    output(doc_res, json_out, render)
    raise typer.Exit(code=0 if doc_res["status"] == "PASS" else 1)


# ---------------------------------------------------------
# qdata auth login|status|refresh|logout <provider>
# ---------------------------------------------------------
@auth_app.command(name="login")
def auth_login_cmd(
    provider: str = typer.Argument(..., help="Provider name (e.g. mock, upstox, fyers)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Provider API key / Client ID"),
    api_secret: Optional[str] = typer.Option(None, "--api-secret", help="Provider API secret / Secret key"),
    client_id: Optional[str] = typer.Option(None, "--client-id", help="Provider Client ID (Fyers)"),
    secret_key: Optional[str] = typer.Option(None, "--secret-key", help="Provider Secret Key (Fyers)"),
    redirect_uri: Optional[str] = typer.Option(None, "--redirect-uri", help="Provider Redirect URI (Fyers)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without saving"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Configure credentials and authenticate with provider."""
    settings = get_settings(override_data_dir=data_dir)
    prov = provider.lower()

    if prov in ("mock", "amfi"):
        # Public or mock providers authenticate without external keys
        if not dry_run:
            prov_inst = get_provider(prov, data_dir=settings.data_dir)
            tok = prov_inst.ensure_auth()
            msg = "Public AMFI provider initialized successfully (no API keys required)." if prov == "amfi" else "Authenticated successfully with mock."
            res = {"status": "success", "provider": prov, "token": tok.get("access_token")}
        else:
            msg = f"[DRY RUN] Would initialize {prov} session."
            res = {"status": "dry_run", "provider": prov, "message": msg}
        output(res, json_out, lambda: console.print(f"[green]{msg}[/green]"))
        raise typer.Exit(code=0)

    # Resolve key/secret with aliases
    resolved_key = api_key or client_id
    resolved_secret = api_secret or secret_key

    if not resolved_key:
        prompt_label = "Enter Fyers Client ID" if prov == "fyers" else f"Enter {prov} API Key"
        resolved_key = typer.prompt(prompt_label, hide_input=False if prov == "fyers" else True)
    if not resolved_secret:
        prompt_label = "Enter Fyers Secret Key" if prov == "fyers" else f"Enter {prov} API Secret"
        resolved_secret = typer.prompt(prompt_label, hide_input=True)

    if dry_run:
        res = {"status": "dry_run", "provider": prov, "message": f"[DRY RUN] Would save credentials for {prov}."}
        output(res, json_out, lambda: console.print(f"[yellow][DRY RUN] Would save credentials for {prov}.[/yellow]"))
        raise typer.Exit(code=0)

    save_credential_to_file(prov, "api_key", resolved_key)
    save_credential_to_file(prov, "api_secret", resolved_secret)
    save_credential_to_file(prov, "client_id", resolved_key)
    save_credential_to_file(prov, "secret_key", resolved_secret)
    if redirect_uri:
        save_credential_to_file(prov, "redirect_uri", redirect_uri)

    res = {"status": "success", "provider": prov, "message": f"Credentials saved securely for {prov}."}
    output(res, json_out, lambda: console.print(f"[green]Credentials securely saved for {prov}.[/green]"))
    raise typer.Exit(code=0)


@auth_app.command(name="status")
def auth_status_cmd(
    provider: str = typer.Argument(..., help="Provider name (e.g. mock, upstox, fyers, amfi)"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Show authentication status and cached token expiration."""
    settings = get_settings(override_data_dir=data_dir)
    prov = provider.lower()
    cached = get_cached_token(settings.tokens_dir, prov)
    cred_api_key = get_credential(prov, "api_key") or get_credential(prov, "client_id")

    now = datetime.now(timezone.utc).timestamp()
    has_token = cached is not None
    expires_at = cached.get("expires_at") if cached else None
    remaining_secs = max(0, int(expires_at - now)) if expires_at else 0

    res = {
        "provider": prov,
        "authenticated": has_token,
        "token_expires_at": expires_at,
        "remaining_seconds": remaining_secs,
        "has_credentials": bool(cred_api_key or prov in ("mock", "amfi")),
    }

    def render():
        table = Table(title=f"Auth Status: {prov}")
        table.add_column("Property", style="bold")
        table.add_column("Value")
        table.add_row("Provider", prov)
        table.add_row("Has Credentials", "[green]YES[/green]" if res["has_credentials"] else "[red]NO[/red]")
        table.add_row("Cached Token", "[green]VALID[/green]" if has_token else "[yellow]NONE / EXPIRED[/yellow]")
        if expires_at:
            table.add_row("Expires In", f"{remaining_secs}s")
        console.print(table)

    output(res, json_out, render)
    raise typer.Exit(code=0)


@auth_app.command(name="refresh")
def auth_refresh_cmd(
    provider: str = typer.Argument(..., help="Provider name (e.g. mock, upstox)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without refreshing"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Force session token refresh for provider."""
    settings = get_settings(override_data_dir=data_dir)
    prov = provider.lower()

    if dry_run:
        res = {"status": "dry_run", "provider": prov, "message": f"[DRY RUN] Would refresh auth token for {prov}."}
        output(res, json_out, lambda: console.print(f"[yellow][DRY RUN] Would refresh token for {prov}.[/yellow]"))
        raise typer.Exit(code=0)

    try:
        p = get_provider(prov, data_dir=settings.data_dir)
        token_info = p.refresh_auth()
    except Exception as e:
        err_msg = {"error": "refresh_failed", "provider": prov, "message": str(e)}
        output(err_msg, json_out, lambda: err_console.print(f"[bold red]Refresh failed:[/bold red] {e}"))
        raise typer.Exit(code=1)

    res = {
        "status": "success",
        "provider": prov,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": token_info.get("expires_at"),
    }
    output(res, json_out, lambda: console.print(f"[green]Successfully refreshed token for {prov}.[/green]"))
    raise typer.Exit(code=0)


@auth_app.command(name="logout")
def auth_logout_cmd(
    provider: str = typer.Argument(..., help="Provider name (e.g. mock, upstox)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without clearing"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Path to data directory."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON format."),
):
    """Clear cached session token for provider."""
    settings = get_settings(override_data_dir=data_dir)
    prov = provider.lower()

    if dry_run:
        res = {"status": "dry_run", "provider": prov, "message": f"[DRY RUN] Would logout and delete token for {prov}."}
        output(res, json_out, lambda: console.print(f"[yellow][DRY RUN] Would clear token for {prov}.[/yellow]"))
        raise typer.Exit(code=0)

    cleared = clear_cached_token(settings.tokens_dir, prov)
    res = {
        "status": "success",
        "provider": prov,
        "token_cleared": cleared,
    }
    output(res, json_out, lambda: console.print(f"[green]Logged out {prov}. Token cleared.[/green]"))
    raise typer.Exit(code=0)


def _register_provider_cli_apps(main_app: typer.Typer) -> None:
    """Dynamically discover and mount provider-specific sub-apps."""
    providers = list_providers()
    for prov_name, prov_cls in providers.items():
        if hasattr(prov_cls, "get_cli_app"):
            sub_app = prov_cls.get_cli_app()
            if sub_app is not None:
                main_app.add_typer(sub_app, name=prov_name)


_register_provider_cli_apps(app)


def main():
    app()


if __name__ == "__main__":
    main()
