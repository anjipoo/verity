# ml/preprocessing/berka_ingest.py
# Strict, explicit loading of the eight Berka tables. No pandas type inference,
# no silent coercion: any unexpected token raises an error.

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.preprocessing import berka_schema as S

PROCESSING_VERSION = "berka-ingest-0.1.0"
MIRROR_BASE_URL = "https://raw.githubusercontent.com/jlacko/berka-dataset/master/"
ORIGINAL_DESCRIPTION_URL = "http://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm"


class HashMismatch(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(raw_dir):
    # runs BEFORE any parsing; a mismatch stops the pipeline
    rows, bad = [], []
    for t in S.TABLES:
        path = Path(raw_dir) / f"{t}.asc"
        if not path.exists():
            raise FileNotFoundError(f"missing raw file: {path}")
        actual = sha256_file(path)
        ok = actual == S.EXPECTED_SHA256[t]
        rows.append({"table": t, "file": path.name, "sha256": actual, "matches_verified": ok})
        if not ok:
            bad.append(f"{t}: expected {S.EXPECTED_SHA256[t]} got {actual}")
    if bad:
        raise HashMismatch("raw file hash mismatch, stopping:\n" + "\n".join(bad))
    return pd.DataFrame(rows)


def _read_tokens(path):
    # QUOTE_NONE keeps the quotes in the data, which is the only way to tell
    # unquoted-empty (missing) from "" (explicit empty) and " " (explicit blank)
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, na_values=[],
                     quoting=csv.QUOTE_NONE, encoding="ascii")
    df.columns = [c.strip('"') for c in df.columns]
    return df


def _decode_text(s, label):
    quoted = s.str.len().ge(2) & s.str.startswith('"') & s.str.endswith('"')
    stray = (~quoted) & (s != "")
    if stray.any():
        raise ValueError(f"{label}: {int(stray.sum())} unquoted non-empty tokens, e.g. {s[stray].iloc[0]!r}")
    out = s.str[1:-1].where(quoted, s).mask(s == "", pd.NA)
    tokens = {"missing (unquoted empty)": int((s == "").sum()),
              'explicit empty ("")': int((out == "").sum()),
              'explicit blank (" ")': int((out == " ").sum())}
    return out, {k: v for k, v in tokens.items() if v}


def _decode_id_str(s, label):
    # identifiers may be quoted or unquoted in the raw files; require digits either way
    out, tok = _decode_text(s, label)
    present = out.notna() & (out != "")
    bad = present & ~out.fillna("").str.fullmatch(r"\d+")
    if bad.any():
        raise ValueError(f"{label}: non-digit identifier token {out[bad].iloc[0]!r}")
    return out, tok


def _decode_number(s, kind, placeholders, label):
    pattern = r"-?\d+" if kind == "int" else r"-?\d+(\.\d+)?"
    missing = s.isin(placeholders | {""})
    bad = (~s.str.fullmatch(pattern)) & ~missing
    if bad.any():
        raise ValueError(f"{label}: invalid {kind} token {s[bad].iloc[0]!r}")
    num = pd.to_numeric(s.mask(missing, pd.NA), errors="raise")
    num = num.astype("Int64" if kind == "int" else "Float64")
    return num, ({"placeholder/empty -> missing": int(missing.sum())} if missing.any() else {})


def _decode_date(s, fmt, label):
    # errors="raise": an unparseable date stops the load
    return pd.to_datetime(s, format=fmt, errors="raise")


def load_berka(raw_dir):
    # returns (tables, report); verify_hashes() must have been called by the caller
    tables, report = {}, {"token_encodings": {}, "placeholders": {}, "dropped": {}}
    for t in S.TABLES:
        raw = _read_tokens(Path(raw_dir) / f"{t}.asc")
        if list(raw.columns) != list(S.SCHEMA[t]):
            raise ValueError(f"{t}: unexpected columns {list(raw.columns)}")
        # privacy first: aggregate facts only, the values are never kept
        for col in S.DROPPED_COLUMNS.get(t, []):
            report["dropped"][f"{t}.{col}"] = {
                "rows": int(len(raw)),
                "six_digit_format_rows": int(raw[col].str.strip('"').str.fullmatch(r"\d{6}").sum()),
                "reason": "contains birth date and sex of a person; not needed for Verity",
            }
            raw = raw.drop(columns=col)
        out = {}
        for col, kind in S.SCHEMA[t].items():
            if col not in raw.columns:
                continue
            label = f"{t}.{col}"
            s = raw[col]
            if kind == "text":
                out[col], tok = _decode_text(s, label)
            elif kind == "id_str":
                out[col], tok = _decode_id_str(s, label)
            elif kind in ("int", "float"):
                out[col], tok = _decode_number(s, kind, S.NUMERIC_PLACEHOLDERS.get((t, col), set()), label)
                if tok:
                    report["placeholders"][label] = tok
                tok = {}
            elif kind == "date":
                out[col], tok = _decode_date(s, "%y%m%d", label), {}
            elif kind == "datetime":
                out[col], tok = _decode_date(s, "%y%m%d %H:%M:%S", label), {}
            else:
                raise ValueError(f"unknown kind {kind}")
            if tok:
                report["token_encodings"][label] = tok
        df = pd.DataFrame(out)
        if t == "district":
            df = df.rename(columns=S.DISTRICT_COLUMN_NAMES)
        for col in S.CATEGORY_COLUMNS.get(t, []):
            df[col] = df[col].astype("category")
        tables[t] = df
    return tables, report


def build_lineage(raw_dir, tables, ingest_report, download_date=None):
    # one record per source file; download_date defaults to the file modification date
    records = []
    for t in S.TABLES:
        path = Path(raw_dir) / f"{t}.asc"
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        steps = ["verify SHA-256 against verified value", "tokenise with quotes preserved",
                 "decode with explicit schema (no inference)", "parse dates strictly"]
        if t in S.DROPPED_COLUMNS:
            steps.append("drop " + ", ".join(S.DROPPED_COLUMNS[t]) + " (privacy)")
        if t == "district":
            steps.append("rename A1..A16 to documented names (official description PDF)")
        if t in S.CATEGORY_COLUMNS:
            steps.append("cast to category: " + ", ".join(S.CATEGORY_COLUMNS[t]))
        records.append({
            "dataset": "Berka / PKDD'99 Financial Dataset",
            "table": t,
            "source": "GitHub mirror jlacko/berka-dataset (copy of the PKDD'99 Discovery Challenge files)",
            "source_url": MIRROR_BASE_URL + f"{t}.asc",
            "original_description_url_reported_by_third_party": ORIGINAL_DESCRIPTION_URL,
            "license": "UNVERIFIED: originator terms not yet checked; do not publish raw files",
            "local_file": str(path),
            "download_date": download_date or mtime,
            "download_date_basis": "argument" if download_date else "file modification time",
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": int(len(tables[t])),
            "columns_raw": len(S.SCHEMA[t]),
            "columns_after_ingestion": int(tables[t].shape[1]),
            "processing_version": PROCESSING_VERSION,
            "processing_steps": steps,
        })
    return {"processing_version": PROCESSING_VERSION, "files": records,
            "documentation": S.DOCUMENTATION,
            "privacy_drops": ingest_report["dropped"]}