# scripts/data/run_berka_phase1a.py
# Phase 1, Step A: Berka ingestion, snapshot, quality report, figures, lineage.
# Run from the repository root:  python -m scripts.data.run_berka_phase1a

import argparse
import contextlib
import json
import sys
from pathlib import Path

import pandas as pd

from ml.preprocessing import berka_ingest as I
from ml.preprocessing import berka_plots as P
from ml.preprocessing import berka_quality as Q
from ml.preprocessing import berka_schema as S

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 60)


class Tee:
    # write everything printed to the console AND to the snapshot file
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def banner(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


def table_snapshot(name, df, seed):
    pk = S.PRIMARY_KEYS[name]
    banner(f"TABLE: {name}")
    mem = df.memory_usage(deep=True).sum() / 1e6
    print(f"rows: {len(df):,} | columns: {df.shape[1]} | memory: {mem:.2f} MB")
    print(f"duplicate rows: {int(df.duplicated().sum())} | primary key {pk}: "
          f"unique={bool(df[pk].is_unique)} nulls={int(df[pk].isna().sum())}")
    if name in S.DROPPED_COLUMNS:
        print(f"privacy: column(s) {S.DROPPED_COLUMNS[name]} removed at ingestion and not shown")

    schema = []
    for c in df.columns:
        s = df[c]
        first = s.dropna().iloc[0] if s.notna().any() else None
        first = first.item() if hasattr(first, "item") else first
        schema.append({"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                       "unique": int(s.nunique(dropna=True)), "example": repr(first)[:40]})
    print("\n-- schema --")
    print(pd.DataFrame(schema).to_string(index=False))

    print("\n-- first 5 --")
    print(df.head().to_string())
    print("\n-- last 5 --")
    print(df.tail().to_string())
    print(f"\n-- random 5 (seed {seed}) --")
    print(df.sample(n=min(5, len(df)), random_state=seed).to_string())

    num_cols = [c for c in df.columns
                if str(df[c].dtype) in ("Int64", "Float64") and c != pk and not c.endswith("_id")]
    if num_cols:
        print("\n-- numeric statistics (identifiers excluded) --")
        stats = df[num_cols].astype("float64").agg(["min", "max", "mean", "median", "std"]).T
        print(stats.round(3).to_string())
    date_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime")]
    for c in date_cols:
        print(f"\n-- date range {c}: {df[c].min()} .. {df[c].max()}")

    for c in df.columns:
        s = df[c]
        if str(s.dtype) in ("category", "str") and s.nunique(dropna=True) <= 30 and not c.endswith("_id"):
            vc = s.astype(object).map(P._label).value_counts()
            print(f"\n-- frequency: {c} ({len(vc)} values incl. missing/blank) --")
            print(vc.to_string())


def print_quality(tables, out_dir):
    banner("DATA-QUALITY AND RELATIONAL INTEGRITY REPORT")
    reports = {
        "01_duplicates_and_keys": Q.duplicates_and_keys(tables),
        "02_null_counts": Q.null_counts(tables),
        "03_date_checks": Q.date_checks(tables),
        "04_numeric_sanity": Q.numeric_sanity(tables),
        "05_categorical_domains": Q.categorical_domains(tables),
        "06_foreign_keys": Q.foreign_keys(tables),
        "07_cardinalities": Q.cardinalities(tables),
        "08_cross_table_dates": Q.cross_table_dates(tables),
        "09_type_direction_evidence": Q.type_direction_evidence(tables["trans"]),
        "10_trans_id_order_check": Q.trans_id_order_check(tables["trans"]),
        "11_official_row_counts": Q.official_row_counts(tables),
        "12_documentation_discrepancies": Q.documentation_discrepancies(tables),
    }
    qdir = Path(out_dir) / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    for name, df in reports.items():
        print(f"\n## {name}")
        print(df.to_string(index=False))
        df.astype(str).to_csv(qdir / f"{name}.csv", index=False)
    return reports


def main():
    p = argparse.ArgumentParser(description="Berka Phase 1A: ingestion, snapshot, quality, figures, lineage")
    p.add_argument("--raw-dir", default="data/raw/berka")
    p.add_argument("--out-dir", default="reports/berka")
    p.add_argument("--meta-dir", default="data/metadata")
    p.add_argument("--interim-dir", default="data/interim/berka")
    p.add_argument("--download-date", default=None, help="YYYY-MM-DD; defaults to file modification date")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-interim", action="store_true", help="do not write typed parquet files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.meta_dir).mkdir(parents=True, exist_ok=True)

    with open(out_dir / "snapshot.txt", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(Tee(sys.__stdout__, fh)):
        banner("STEP 0: RAW FILE INTEGRITY (SHA-256 against verified values)")
        try:
            hashes = I.verify_hashes(args.raw_dir)
        except I.HashMismatch as e:
            print(str(e))
            raise SystemExit(1)
        print(hashes.to_string(index=False))

        banner("STEP 1: STRICT TYPED IMPORT")
        tables, report = I.load_berka(args.raw_dir)
        print("token encodings preserved (unquoted-empty vs \"\" vs \" \"):")
        print(json.dumps(report["token_encodings"], indent=2))
        print("numeric placeholders converted to missing:")
        print(json.dumps(report["placeholders"], indent=2))
        print("privacy: columns dropped at ingestion (aggregate facts only):")
        print(json.dumps(report["dropped"], indent=2))

        banner("STEP 2: DATA SNAPSHOT (every table)")
        for t in S.TABLES:
            table_snapshot(t, tables[t], args.seed)

        print_quality(tables, out_dir)

        banner("STEP 3: RAW-DATA FIGURES")
        for path in P.make_all(tables, out_dir / "figures"):
            print("saved", path)

        interim = []
        if not args.no_interim:
            banner("STEP 4: TYPED INTERIM TABLES (privacy-safe, no cleaning applied)")
            idir = Path(args.interim_dir)
            idir.mkdir(parents=True, exist_ok=True)
            for t in S.TABLES:
                f = idir / f"{t}.parquet"
                tables[t].to_parquet(f, index=False)
                interim.append({"table": t, "file": str(f), "rows": len(tables[t]),
                                "columns": tables[t].shape[1], "sha256": I.sha256_file(f)})
                print("wrote", f, f"({len(tables[t]):,} rows)")

        lineage = I.build_lineage(args.raw_dir, tables, report, args.download_date)
        lineage["interim_outputs"] = interim
        lineage["stage"] = "Phase 1A: ingestion only (no cleaning, no features, no labels)"
        lpath = Path(args.meta_dir) / "berka_lineage.json"
        lpath.write_text(json.dumps(lineage, indent=2, default=str), encoding="utf-8")
        print("\nlineage manifest:", lpath)
        print("snapshot text   :", out_dir / "snapshot.txt")


if __name__ == "__main__":
    main()