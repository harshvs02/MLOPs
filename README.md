# MLOps Batch Job — Rolling-Mean Trading Signal

A minimal, reproducible batch job that reads OHLCV data, computes a rolling
mean on `close`, derives a binary long/flat signal, and emits structured
metrics + logs. Built to mirror the shape of a `MetaStackerBandit`
trading-signal pipeline stage: deterministic, observable, and
container-deployable.

## What it does

1. Loads and validates `config.yaml` (`seed`, `window`, `version`).
2. Sets `numpy.random.seed(seed)` for deterministic behavior.
3. Loads and validates `data.csv` (must contain a numeric `close` column).
4. Computes `rolling_mean = close.rolling(window).mean()`.
5. Derives `signal = 1 if close > rolling_mean else 0`.
6. Writes machine-readable metrics to `metrics.json` and a detailed run
   trace to `run.log`, in both success and failure cases.

### Handling the warm-up window

The first `window - 1` rows don't have a full window of prior `close`
values, so `rolling_mean` (and therefore `signal`) is `NaN` for those rows
(`min_periods=window`, no forward/backward filling). Those rows are still
counted in `rows_processed`, but excluded from `signal_rate`, which is the
mean of `signal` over rows that have a defined value. This is logged
explicitly (`"First N row(s) fall within the warm-up window..."`).

## Repo contents

```
run.py             # the pipeline (only required deliverable code file)
generate_data.py   # optional helper that creates a synthetic fallback dataset (not used to build the current data.csv)
config.yaml         # seed / window / version
data.csv            # 10,000 rows of BTC OHLCV data (1-minute bars, 2024-01-01 to 2024-01-07)
requirements.txt
Dockerfile
README.md
metrics.json        # sample output from a successful run
run.log             # sample log from a successful run
```

`data.csv` contains real 1-minute BTC/USD OHLCV bars: `timestamp, open, high,
low, close, volume_btc, volume_usd`. The pipeline only reads `close`, so the
extra columns (`timestamp`, `volume_btc`, `volume_usd`, etc.) are ignored —
`run.py` only requires a numeric `close` column to be present.

## Run locally

```bash
python3 -m pip install -r requirements.txt

python3 run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

The final metrics JSON is also printed to stdout; detailed logging goes to
`run.log` (full detail) and stderr (INFO and above). Exit code is `0` on
success and non-zero on any handled or unhandled failure.

No paths are hard-coded anywhere in `run.py` — `--input`, `--config`,
`--output`, and `--log-file` are all required CLI arguments.

## Run with Docker

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The image bundles `data.csv` and `config.yaml` and runs the exact required
CLI command as its `CMD`:

```
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

Since the container runs with `--rm`, `metrics.json` and `run.log` are
written inside the container during the run and removed with it on exit —
that's why the final metrics JSON is also printed to stdout, so the result
is visible without needing to keep the container around. To persist those
files on the host instead, mount a volume and override the working
directory expectations, e.g.:

```bash
docker run --rm -v "$(pwd)/out:/app/out" mlops-task \
  python run.py --input data.csv --config config.yaml \
                 --output out/metrics.json --log-file out/run.log
```

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 54,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input CSV missing required column 'close'. Found columns: ['open', 'high', 'low', 'volume']"
}
```

`metrics.json` is always written, in both the success and error paths.

## Validation & error handling

`run.py` returns exit code `1` (and writes an error `metrics.json`) for:

- Missing config file, empty config file, invalid YAML, non-mapping YAML,
  or missing/invalid `seed` / `window` / `version` fields.
- Missing input file, empty input file, unparseable/malformed CSV.
- CSV with no rows, or missing/non-numeric `close` column.
- Dataset with fewer rows than the configured `window`.
- Any unexpected exception (caught as a safety net, logged with full
  traceback via `logger.exception`, still reported as clean JSON).

## Reproducibility

- `seed`, `window`, and `version` all come from `config.yaml` — nothing is
  inferred or randomized outside of the fixed seed.
- `numpy.random.seed(seed)` is set once at the start of the job.
- Given the same `data.csv` and `config.yaml`, `rows_processed`,
  `signal_rate`, `seed`, and `version` are identical across runs. Only
  `latency_ms` (wall-clock runtime) varies run to run, as expected.

## Observability

`run.log` captures, with timestamps: job start, config validation
(seed/window/version), rows loaded, rolling-mean and signal-generation
steps, the warm-up-row note, the metrics summary, job end + status, and
any validation errors or unexpected exceptions (with traceback). The same
INFO-level stream also goes to stderr, keeping stdout reserved for the
final metrics JSON.
