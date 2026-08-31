"""Download and validate one official UNSW-NB15 CSV file.

This utility fails closed: it never fabricates a dataset and never replaces an
existing file until the downloaded bytes pass the schema and row-count checks.
Use the URL and checksum supplied by the dataset publisher or your institution.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv"
MAX_DOWNLOAD_BYTES = 1_000_000_000
REQUIRED_COLUMNS = {
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "proto",
    "state",
    "service",
    "label",
}
COLUMN_ALIASES = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
}


def _validate_csv(path: Path, expected_rows: int | None = None) -> int:
    """Validate the required schema and return the number of records."""

    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pandas raises several parser-specific errors
        raise ValueError(f"{path} is not a readable CSV file: {exc}") from exc

    raw_columns = {str(column).strip().lower() for column in frame.columns}
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in raw_columns and new_name in raw_columns:
            raise ValueError(
                f"CSV contains both '{old_name}' and '{new_name}'; remove the duplicate spelling."
            )
    columns = {COLUMN_ALIASES.get(column, column) for column in raw_columns}
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError("CSV is missing required UNSW-NB15 columns: " + ", ".join(missing))
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("CSV label column must contain only 0 and 1.")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"CSV has {len(frame):,} rows; expected {expected_rows:,} for the requested split."
        )
    return len(frame)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download one publisher-provided CSV and atomically install it."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        rows = _validate_csv(destination, expected_rows)
        if expected_sha256 and _sha256(destination) != expected_sha256.lower():
            raise ValueError(f"Existing file checksum does not match: {destination}")
        print(f"[OK] Validated existing file: {destination} ({rows:,} rows)")
        return destination

    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "UNSW-NB15-research-pipeline/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"Download exceeds the {MAX_DOWNLOAD_BYTES:,}-byte safety limit.")
                    temporary.write(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    try:
        digest = _sha256(temporary_path)
        if expected_sha256 and digest != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch: downloaded {digest}, expected {expected_sha256.lower()}"
            )
        rows = _validate_csv(temporary_path, expected_rows)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    if expected_sha256:
        print(f"[OK] Downloaded and verified {rows:,} rows: {destination}")
    else:
        print(f"[WARN] Downloaded {rows:,} rows without a checksum; record SHA-256: {digest}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Publisher-provided CSV URL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-sha256", help="Expected lowercase SHA-256 digest")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--force", action="store_true", help="Replace an existing validated file")
    args = parser.parse_args(argv)
    try:
        download(
            args.url,
            args.output,
            expected_sha256=args.expected_sha256,
            expected_rows=args.expected_rows,
            force=args.force,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
