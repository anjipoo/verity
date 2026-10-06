# scripts/data/run_berka_phase1b.py
# Phase 1B: cleaning + relational integration + feature engineering.
# Reads ONLY data/interim/berka/. Run from the repository root:
#   python -m scripts.data.run_berka_phase1b

import argparse
import contextlib
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing import berka_clean as C
from ml.preprocessing import berka_feature_docs as D
from ml.preprocessing import berka_features as F
from ml.preprocessing import berka_phase1b_plots as P
from ml.preprocessing import berka_validation as V

SEED = 42
TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
PROCESSING_VERSION = "berka-phase1b-0.1.0"
DEMO_CUTOFF = "1997-01-01"  # used only for the leakage tests

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)


class Tee:
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cleaning_stage(tables, in_dir, p1a_lineage):
    banner("PHASE 1B / CLEANING AND RELATIONAL INTEGRATION")
    inputs = []
    recorded = {r["table"]: r["sha256"] for r in p1a_lineage.get("interim_outputs", [])}
    for t in TABLES:
        path = Path(in_dir) / f"{t}.parquet"
        h = sha256_file(path)
        inputs.append({"table": t, "file": str(path), "rows": len(tables[t]), "sha256": h,
                       "matches_phase1a_lineage": recorded.get(t) == h})
    print("input files (Phase 1A interim Parquet):")
    print(pd.DataFrame(inputs).to_string(index=False))

    cleaned, log = C.clean_tables(tables)
    counts = {t: len(tables[t]) for t in TABLES}
    out_counts = {t: len(cleaned[t]) for t in TABLES}
    rc_df, rc_checks = V.row_count_checks(counts, out_counts)
    banner("ROW COUNTS: input vs output")
    print(rc_df.to_string(index=False))
    print("rows removed in total:", int(rc_df["rows_removed"].sum()))

    banner("TRANSFORMATION LOG (table, column, transformation, affected rows, reason)")
    print(pd.DataFrame(log).to_string(index=False))

    t = cleaned["trans"]
    banner("INVESTIGATION: transactions with amount <= 0")
    z = C.investigate_nonpositive(t)
    print(z.drop(columns=["operation"]).to_string(index=False))
    print(f"\nrows: {len(z)} | distinct amounts: {sorted(z['amount'].astype(float).unique())} | "
          f"k_symbol: {z['k_symbol'].astype(str).value_counts().to_dict()}")
    print(f"types: {z['type'].astype(str).value_counts().to_dict()} | accounts affected: {z['account_id'].nunique()}")
    print(f"identical duplicate postings (same account, date, type, k_symbol, amount, balance): "
          f"{int(z['same_values_as_another_row'].sum())} rows")
    print("finding: all are exactly 0.0 interest postings (UROK credit or SANKC. UROK penalty). The balance "
          "relation to the previous row is inconclusive because same-day order is ambiguous.")
    print("treatment: rows kept; excluded from amount statistics, percentiles and ratios; "
          "included in transaction counts, direction counts and balance features.")

    banner("PRESERVED TOKEN DISTINCTIONS")
    print("trans.k_symbol_token:\n", t["k_symbol_token"].value_counts().to_string())
    print("\ntrans.k_symbol_normalized (top):\n", t["k_symbol_normalized"].value_counts().to_string())
    print("\ntrans.operation_normalized:\n", t["operation_normalized"].value_counts().to_string())
    print("\ntrans.counterparty_status:\n", t["counterparty_status"].value_counts().to_string())
    print("\nbank_missing:", int(t["bank_missing"].sum()), "| counterparty_account_missing:",
          int(t["counterparty_account_missing"].sum()))

    banner("TRANSACTION SEMANTICS (documented mapping, counts from the data)")
    sem = t.groupby(["direction", "operation_category"], observed=True).size().rename("rows").reset_index()
    print(sem.to_string(index=False))
    print("type -> direction:", C.TYPE_DIRECTION)

    banner("DISTRICT AND ORDER CHANGES")
    print({c: int(cleaned["district"][c].sum()) for c in cleaned["district"].columns if c.endswith("_missing")})
    print("order.k_symbol_token:", cleaned["order"]["k_symbol_token"].value_counts().to_dict())

    banner("POST-CLEANING INTEGRITY")
    integ = V.integrity_checks(cleaned)
    print(pd.DataFrame(rc_checks + integ).to_string(index=False))
    return cleaned, log, inputs, rc_df, rc_checks + integ


def feature_stage(cleaned, args):
    banner("PHASE 1B / FEATURE ENGINEERING")
    trans, acc, dist = cleaned["trans"], cleaned["account"], cleaned["district"]
    af = F.compute_account_features(trans, None, acc, dist)
    cf = F.compute_client_features(af, cleaned["disp"], cleaned["client"], dist)
    ratios = F.prior_day_ratios(trans)
    print(f"account_features: {af.shape} | client_features: {cf.shape} | cutoff: none (full history, "
          f"offline description only)")
    print("\naccount feature columns (%d):" % af.shape[1])
    print(list(af.columns))
    print("\nclient feature columns (%d):" % cf.shape[1])
    print(list(cf.columns))
    print("\naccount feature summary:")
    num = af.select_dtypes("number").drop(columns=["account_id", "district_id"])
    print(num.describe(percentiles=[.5]).T.round(3).to_string())
    print("\nclient feature summary:")
    print(cf.select_dtypes("number").drop(columns=["client_id", "district_id"]).describe(percentiles=[.5]).T
          .round(3).to_string())
    print(f"\nper-transaction ratios: {len(ratios):,} positive-amount rows, "
          f"{int(ratios['amount_to_typical_ratio'].notna().sum()):,} with a defined baseline "
          f"(>= {F.MIN_PRIOR_TRANSACTIONS} prior positive transactions on earlier days)")

    banner("VALIDATION")
    checks = (V.feature_checks(af, cf, trans) + V.deviation_helper_checks() +
              V.leakage_tests(trans, acc, dist, DEMO_CUTOFF, SEED) + V.determinism_test(trans, acc, dist))
    af_cut = F.compute_account_features(trans, DEMO_CUTOFF, acc, dist)
    cf_cut = F.compute_client_features(af_cut, cleaned["disp"], cleaned["client"], dist)
    checks += [dict(c, check=f"[cutoff {DEMO_CUTOFF}] " + c["check"])
               for c in V.feature_checks(af_cut, cf_cut, trans, DEMO_CUTOFF)]
    print(pd.DataFrame(checks).to_string(index=False))
    nan = V.nan_report(af, cf)
    banner("REMAINING NaN VALUES (full-history features)")
    print(nan.to_string(index=False) if len(nan) else "none")
    return af, cf, ratios, checks, nan


def write_outputs(cleaned, af, cf, args):
    pdir = Path(args.processed_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    named = {f"{t}_clean": cleaned[t] for t in TABLES}
    named.update({"account_features": af, "client_features": cf})
    for name, df in named.items():
        path = pdir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        outputs.append({"file": str(path), "rows": len(df), "columns": df.shape[1], "sha256": sha256_file(path)})
    return outputs


def main():
    p = argparse.ArgumentParser(description="Berka Phase 1B")
    p.add_argument("--interim-dir", default="data/interim/berka")
    p.add_argument("--processed-dir", default="data/processed/berka")
    p.add_argument("--meta-dir", default="data/metadata")
    p.add_argument("--validation-dir", default="data/validation")
    p.add_argument("--report-dir", default="reports/berka")
    args = p.parse_args()
    np.random.seed(SEED)
    rdir = Path(args.report_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    for d in (args.meta_dir, args.validation_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    lineage_1a = json.loads((Path(args.meta_dir) / "berka_lineage.json").read_text())
    tables = {t: pd.read_parquet(Path(args.interim_dir) / f"{t}.parquet") for t in TABLES}

    with open(rdir / "phase1b_cleaning_report.txt", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(Tee(sys.__stdout__, fh)):
        cleaned, log, inputs, rc_df, clean_checks = cleaning_stage(tables, args.interim_dir, lineage_1a)

    with open(rdir / "phase1b_feature_report.txt", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(Tee(sys.__stdout__, fh)):
        af, cf, ratios, feat_checks, nan = feature_stage(cleaned, args)
        banner("FIGURES 06-13")
        for path in P.make_all(cleaned["trans"], af, ratios, rdir / "figures"):
            print("saved", path)
        outputs = write_outputs(cleaned, af, cf, args)
        banner("OUTPUT FILES")
        print(pd.DataFrame(outputs).to_string(index=False))

        fdict = D.build_feature_dictionary(list(af.columns), list(cf.columns))
        dpath = Path(args.meta_dir) / "berka_phase1b_feature_dictionary.json"
        dpath.write_text(json.dumps(fdict, indent=2, default=str), encoding="utf-8")

        all_checks = clean_checks + feat_checks
        failed = [c for c in all_checks if c["status"] != "PASS"]
        validation = {"processing_version": PROCESSING_VERSION, "seed": SEED, "demo_cutoff": DEMO_CUTOFF,
                      "checks_total": len(all_checks), "checks_passed": len(all_checks) - len(failed),
                      "checks_failed": len(failed), "overall": "PASS" if not failed else "FAIL",
                      "row_counts": rc_df.to_dict("records"), "checks": all_checks,
                      "remaining_nan": nan.to_dict("records")}
        vpath = Path(args.validation_dir) / "berka_phase1b_validation.json"
        vpath.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")

        lineage = {
            "stage": "Phase 1B: cleaning, relational integration, feature engineering (no labels, no model)",
            "processing_version": PROCESSING_VERSION,
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "pandas": pd.__version__, "seed": SEED,
            "phase1a_lineage_file": str(Path(args.meta_dir) / "berka_lineage.json"),
            "input_files": inputs,
            "processing_steps": ["verify interim Parquet hashes against Phase 1A lineage",
                                 "clean tables (flags and normalized columns, no rows removed)",
                                 "add transaction semantics from the documented mapping",
                                 "compute account features (cutoff-safe) and client features",
                                 "validate, plot, write Parquet"],
            "cleaning_transformations": log,
            "rows_removed": {r["table"]: r["rows_removed"] for r in rc_df.to_dict("records")},
            "output_files": outputs,
            "features_created": {"account_features": list(af.columns), "client_features": list(cf.columns)},
            "features_removed": [],
            "columns_removed_upstream": "client.birth_number was removed in Phase 1A (privacy); nothing else",
            "labels_created": "none (no fraud, default or risk label)",
            "validation": {"overall": validation["overall"], "passed": validation["checks_passed"],
                           "failed": validation["checks_failed"], "file": str(vpath)},
            "feature_dictionary": str(dpath),
        }
        lpath = Path(args.meta_dir) / "berka_phase1b_lineage.json"
        lpath.write_text(json.dumps(lineage, indent=2, default=str), encoding="utf-8")
        print(f"\nvalidation: {validation['overall']} ({validation['checks_passed']}/{validation['checks_total']})")
        print("feature dictionary:", dpath)
        print("validation report :", vpath)
        print("lineage           :", lpath)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()