# qdata

qdata is a CLI + Python library that ingests, stores, catalogs, and serves market data from multiple providers at 1-min base granularity.

## Four Rules Governing Everything

1. **Raw data is immutable** — closed partitions are frozen; updates append idempotently.
2. **Catalog is the source of truth** — code never guesses what data exists.
3. **Sync is idempotent** — running twice = same state.
4. **Partition minimally** — sync touches only the current open partition.

## Storage Layout

```plain
data/
├── catalog/
│   ├── instruments.parquet    # one row per ticker
│   ├── datasets.parquet       # one row per (symbol, timeframe, layer)
│   └── manifest.json          # SHA-256 + row_count per file
├── raw/                       # exactly what provider sent
│   └── <SYMBOL>/<timeframe>/<YYYY>/<YYYY-MM>.parquet   (1min: monthly)
│   └── <SYMBOL>/1d/<YYYY>.parquet                      (1d: yearly)
├── adjusted/                  # derived, always rebuildable from raw
└── .tokens/                   # cached session tokens, chmod 600
```

## Supported Providers

- **Mock (`mock`)**: Deterministic synthetic market data generator for tests and offline usage.
- **Fyers (`fyers`)**: Direct broker API integration for NSE/BSE equities and derivatives.
- **Upstox (`upstox`)**: Upstox API provider integration.
- **AMFI (`amfi`)**: Official Association of Mutual Funds in India daily NAV data and schemes directory feed.

### AMFI Provider Commands

```bash
# Search mutual funds by name, scheme code, or ISIN
qdata amfi search "Axis Bluechip"

# List fund houses
qdata amfi fund-houses

# Sync mutual fund daily NAV data
qdata sync --provider amfi --symbol 120503 --start 2020-01-01 --end 2024-01-01
```

