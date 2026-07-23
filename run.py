#!/usr/bin/env python3
"""
MLOps-style batch job: rolling-mean trading-signal pipeline.

Loads a YAML config, reads OHLCV data from a CSV, computes a rolling mean
on `close`, derives a binary long/flat signal, and writes structured
metrics (JSON) plus a detailed run log. Designed to be deterministic,
observable, and safe to run inside a container with no hard-coded paths.

Usage:
    python run.py --input data.csv --config config.yaml \
                   --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS = ("seed", "window", "version")


class JobError(Exception):
    """Raised for any handled validation or processing failure.

    Anything raised as JobError is treated as an expected, recoverable
    failure mode (bad input, bad config, etc.) and is reported cleanly
    in metrics.json rather than as a stack trace.
    """


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logging(log_file: str) -> logging.Logger:
    """Configure a logger that writes detailed logs to `log_file` and a
    concise stream to stderr (so stdout stays clean for the final JSON)."""
    logger = logging.getLogger("mlops_batch_job")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on repeated calls

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(config_path: str, logger: logging.Logger) -> dict:
    """Parse and validate the YAML config. Raises JobError on any problem."""
    path = Path(config_path)
    if not path.exists():
        raise JobError(f"Config file not found: {config_path}")
    if path.stat().st_size == 0:
        raise JobError(f"Config file is empty: {config_path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise JobError(f"Invalid YAML in config file: {e}")

    if not isinstance(config, dict):
        raise JobError(
            "Invalid config structure: expected a YAML mapping with "
            "'seed', 'window', and 'version' keys"
        )

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise JobError(f"Config missing required field(s): {', '.join(missing)}")

    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise JobError("Config field 'seed' must be an integer")
    if not isinstance(config["window"], int) or isinstance(config["window"], bool) \
            or config["window"] < 1:
        raise JobError("Config field 'window' must be a positive integer")
    if not isinstance(config["version"], str) or not config["version"].strip():
        raise JobError("Config field 'version' must be a non-empty string")

    logger.info(
        "Config loaded and validated: seed=%s, window=%s, version=%s",
        config["seed"], config["window"], config["version"],
    )
    return config


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Read and validate the input CSV. Raises JobError on any problem."""
    path = Path(input_path)
    if not path.exists():
        raise JobError(f"Input file not found: {input_path}")
    if path.stat().st_size == 0:
        raise JobError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise JobError(f"Input file has no data or columns: {input_path}")
    except pd.errors.ParserError as e:
        raise JobError(f"Invalid CSV format: {e}")
    except Exception as e:  # noqa: BLE001 - surfaced as a clean JobError
        raise JobError(f"Failed to read input CSV: {e}")

    if df.empty:
        raise JobError("Input CSV contains no rows")

    if "close" not in df.columns:
        raise JobError(
            f"Input CSV missing required column 'close'. "
            f"Found columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["close"]):
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if df["close"].isna().all():
            raise JobError("Column 'close' contains no valid numeric data")

    logger.info("Dataset loaded: %d rows, %d columns from %s",
                len(df), len(df.columns), input_path)
    logger.info("Columns present: %s", list(df.columns))
    return df


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #

def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """Compute rolling mean of `close` and derive a binary signal.

    The first (window - 1) rows do not have a full window of history, so
    their rolling mean is NaN (min_periods=window) and their signal is
    left as NaN too. These rows are excluded from signal-based metrics
    but are still counted in rows_processed.
    """
    logger.info("Computing rolling mean on 'close' with window=%d", window)
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    logger.info("Generating binary signal: signal = 1 if close > rolling_mean else 0")
    df["signal"] = np.where(df["close"] > df["rolling_mean"], 1, 0).astype(float)
    df.loc[df["rolling_mean"].isna(), "signal"] = np.nan

    warmup_rows = int(df["rolling_mean"].isna().sum())
    if warmup_rows:
        logger.info(
            "First %d row(s) fall within the warm-up window and have no "
            "rolling mean / signal (excluded from signal_rate)", warmup_rows
        )

    return df


# --------------------------------------------------------------------------- #
# Metrics I/O
# --------------------------------------------------------------------------- #

def write_metrics(output_path: str, metrics: dict, logger: logging.Logger = None) -> None:
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    if logger:
        logger.info("Metrics written to %s", output_path)


# --------------------------------------------------------------------------- #
# Main job logic
# --------------------------------------------------------------------------- #

def run(args) -> int:
    start_time = time.time()
    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Job started")
    logger.info(
        "Arguments: input=%s config=%s output=%s log_file=%s",
        args.input, args.config, args.output, args.log_file,
    )

    version = "unknown"
    try:
        config = load_config(args.config, logger)
        version = config["version"]

        np.random.seed(config["seed"])
        logger.info("Deterministic seed set: numpy.random.seed(%d)", config["seed"])

        df = load_dataset(args.input, logger)

        window = config["window"]
        if len(df) < window:
            raise JobError(
                f"Dataset has {len(df)} row(s), which is fewer than the "
                f"configured window size ({window}); cannot compute a "
                f"full rolling mean"
            )

        df = compute_signal(df, window, logger)

        valid_signals = df["signal"].dropna()
        rows_processed = len(df)
        signal_rate = float(valid_signals.mean()) if len(valid_signals) > 0 else 0.0
        latency_ms = int(round((time.time() - start_time) * 1000))

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": config["seed"],
            "status": "success",
        }

        logger.info(
            "Metrics summary: rows_processed=%d, signal_rate=%.4f, latency_ms=%dms",
            rows_processed, signal_rate, latency_ms,
        )

        write_metrics(args.output, metrics, logger)
        logger.info("Job finished: status=success")
        logger.info("=" * 70)

        print(json.dumps(metrics, indent=2))
        return 0

    except JobError as e:
        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(e),
        }
        logger.error("Validation/processing error: %s", str(e))
        write_metrics(args.output, error_metrics, logger)
        logger.info("Job finished: status=error")
        logger.info("=" * 70)
        print(json.dumps(error_metrics, indent=2))
        return 1

    except Exception as e:  # noqa: BLE001 - last-resort safety net
        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": f"Unexpected error: {e}",
        }
        logger.exception("Unexpected error during job execution")
        write_metrics(args.output, error_metrics, logger)
        logger.info("Job finished: status=error")
        logger.info("=" * 70)
        print(json.dumps(error_metrics, indent=2))
        return 1


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling-mean trading-signal pipeline"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV (OHLCV data)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
