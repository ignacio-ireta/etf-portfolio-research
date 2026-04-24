# Data Dictionary

## ETF Metadata Columns

Source file:

- [data/metadata/etf_universe.csv](../data/metadata/etf_universe.csv)

Storage columns currently present in the CSV:

| Column | Type | Description |
| --- | --- | --- |
| `ticker` | string | ETF ticker symbol. |
| `name` | string | Human-readable ETF name. |
| `asset_class` | string | Broad portfolio role such as `equity`, `fixed_income`, `commodity`, `real_estate`. |
| `region` | string | Geographic scope, for example `US`, `Global`, `Emerging Markets`. |
| `sector` | string | More specific exposure bucket such as `large_cap_blend` or `broad_commodities`. |
| `currency` | string | Listing/base currency code, currently expected to be 3 letters. |
| `expense_ratio` | float | Annual expense ratio in decimal form, for example `0.0003` for 3 bps. Must be in `[0, 0.01]`. |
| `benchmark_index` | string | Reference index tracked by the ETF. |
| `is_leveraged` | boolean | Leveraged ETF flag. |
| `is_inverse` | boolean | Inverse ETF flag. |
| `inception_date` | date | ETF inception date used for validation. |
| `role` | string | Portfolio role: `core`, `satellite`, `benchmark`, or `excluded`. Universe tickers must be `core` or `satellite`; the primary benchmark ticker must be `benchmark`. |
| `notes` | string | Human notes about role or rationale. |

Pandera validation schema currently requires this strict subset:

- `ticker`
- `name`
- `asset_class`
- `region`
- `currency`
- `expense_ratio`
- `benchmark_index`
- `is_leveraged`
- `is_inverse`
- `inception_date`
- `role`

## Price Data Schema

Raw and validated price files are stored in parquet form:

- `data/raw/prices.parquet`
- `data/processed/prices_validated.parquet`
- `data/processed/prices_adjusted.parquet`

Expected structure:

- index: trading dates as `datetime64[ns]`
- columns: unique ETF tickers
- values: adjusted close prices as non-negative numeric values

Validation rules:

- no duplicate dates
- no duplicate tickers
- no entirely null ticker column
- missing-data ratio below configured threshold
- sufficient history coverage for each ticker
- no negative prices
- suspicious one-day jumps flagged
- no observations before ETF inception date
- benchmark overlap with asset date range

## Returns Schema

Returns files:

- `data/processed/returns.parquet`
- `data/processed/returns_daily.parquet`

Expected structure:

- index: trading dates as `datetime64[ns]`
- columns: unique ETF tickers
- values: simple daily returns as decimal values

Derived from validated adjusted prices using `simple_returns(...)`.

## Benchmark Schema

Configured in [configs/base.yaml](../configs/base.yaml).

Current benchmark structure:

- `benchmark.primary`: single ETF ticker used as the main benchmark
- `benchmark.secondary`: named weighted mixes whose weights must sum to `1.0`

Current example:

| Name | Definition |
| --- | --- |
| `primary` | `VT` |
| `global_60_40` | `VT: 0.60`, `BND: 0.40` |
| `simple_global_baseline` | `VTI: 0.55`, `VEA: 0.25`, `VWO: 0.10`, `BND: 0.10` |
