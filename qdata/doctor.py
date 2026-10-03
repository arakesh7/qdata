import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdata.auth import check_credentials_file_permissions
from qdata.catalog import CatalogManager
from qdata.config import get_settings
from qdata.store import ManifestManager, SyncLock, compute_sha256, read_partition_parquet


def run_doctor(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Diagnose configuration, credentials security, locks, and catalog <-> FS consistency.
    Fails loudly on world-readable credential files or catalog discrepancies.
    """
    settings = get_settings(override_data_dir=data_dir)
    catalog = CatalogManager(settings.data_dir)
    manifest = ManifestManager(settings.data_dir)

    checks: List[Dict[str, Any]] = []
    overall_ok = True

    # 1. Config & Directory Check
    dir_info = {
        "data_dir": str(settings.data_dir),
        "data_dir_exists": settings.data_dir.exists(),
        "catalog_dir_exists": settings.catalog_dir.exists(),
        "raw_dir_exists": settings.raw_dir.exists(),
        "tokens_dir_exists": settings.tokens_dir.exists(),
    }
    checks.append({
        "name": "Directories",
        "status": "PASS" if settings.data_dir.exists() else "WARN",
        "details": dir_info,
        "message": "Data directory exists." if settings.data_dir.exists() else "Data directory not initialized. Run 'qdata init'.",
    })

    # 2. Credential Security Check
    cred_file = settings.resolved_credentials_file
    if cred_file.exists():
        is_secure, perm_msg = check_credentials_file_permissions(cred_file)
        if not is_secure:
            overall_ok = False
            checks.append({
                "name": "Credentials Security",
                "status": "FAIL",
                "details": {"file": str(cred_file), "message": perm_msg},
                "message": f"CRITICAL: Credentials file has insecure permissions! {perm_msg}",
            })
        else:
            checks.append({
                "name": "Credentials Security",
                "status": "PASS",
                "details": {"file": str(cred_file)},
                "message": "Credentials file permissions are secure.",
            })
    else:
        checks.append({
            "name": "Credentials Security",
            "status": "PASS",
            "details": {"file": str(cred_file)},
            "message": "Credentials file does not exist yet (secure by default).",
        })

    # 3. Lock Status Check
    lock_file = settings.lock_file
    if lock_file.exists():
        lock = SyncLock(lock_file)
        # Try checking if pid is running
        try:
            import json
            with open(lock_file, "r", encoding="utf-8") as f:
                lock_info = json.load(f)
            pid = int(lock_info.get("pid", -1))
            if pid > 0 and lock.is_pid_running(pid):
                checks.append({
                    "name": "Sync Lock",
                    "status": "WARN",
                    "details": lock_info,
                    "message": f"Lock is currently held by active process {pid}.",
                })
            else:
                checks.append({
                    "name": "Sync Lock",
                    "status": "WARN",
                    "details": lock_info,
                    "message": f"Stale lock found for dead process {pid}. Will be cleaned up on next sync.",
                })
        except Exception as e:
            checks.append({
                "name": "Sync Lock",
                "status": "WARN",
                "details": {"error": str(e)},
                "message": "Unreadable lock file detected.",
            })
    else:
        checks.append({
            "name": "Sync Lock",
            "status": "PASS",
            "details": {},
            "message": "No active sync lock.",
        })

    # 4. Catalog <-> Filesystem Consistency
    catalog_issues: List[str] = []
    if settings.catalog_dir.exists():
        try:
            catalog.init_catalog()
            df_ds = catalog.list_datasets()
            manifest_files = manifest.load()

            for _, row in df_ds.iterrows():
                open_p = row.get("open_partition")
                if open_p:
                    open_path = settings.data_dir / open_p
                    if not open_path.exists():
                        catalog_issues.append(f"Open partition file missing on disk: {open_p}")

            # Verify manifest vs disk
            manifest_results = manifest.verify_all()
            for res in manifest_results:
                if res["status"] == "missing":
                    catalog_issues.append(f"Manifest file missing from disk: {res['file']}")
                elif res["status"] == "mismatch":
                    catalog_issues.append(
                        f"Hash mismatch on {res['file']}: expected {res['expected_sha']}, got {res['actual_sha']}"
                    )

            # Check for orphaned files on disk not in manifest
            disk_files = []
            for search_dir in [settings.raw_dir, settings.adjusted_dir]:
                if search_dir.exists():
                    for pf in search_dir.rglob("*.parquet"):
                        disk_files.append(pf.relative_to(settings.data_dir).as_posix())

            orphans = [f for f in disk_files if f not in manifest_files]
            if orphans:
                catalog_issues.append(f"Found {len(orphans)} orphaned parquet files not in manifest: {orphans[:3]}")

        except Exception as e:
            catalog_issues.append(f"Error inspecting catalog: {e}")

    if catalog_issues:
        overall_ok = False
        checks.append({
            "name": "Catalog & Manifest Consistency",
            "status": "FAIL",
            "details": {"issues": catalog_issues},
            "message": f"Catalog consistency check failed with {len(catalog_issues)} issues.",
        })
    else:
        checks.append({
            "name": "Catalog & Manifest Consistency",
            "status": "PASS",
            "details": {},
            "message": "Catalog, manifest, and disk files are fully consistent.",
        })

    return {
        "status": "PASS" if overall_ok else "FAIL",
        "checks": checks,
    }
