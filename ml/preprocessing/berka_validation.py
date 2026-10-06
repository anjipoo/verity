# ml/preprocessing/berka_validation.py
# Automated Phase 1B checks. Each check returns {"check", "status", "detail"}.

import numpy as np
import pandas as pd

from ml.preprocessing import berka_features as F
from ml.preprocessing import berka_quality as Q

COUNT_COLUMNS = ["transaction_count", "active_days", "history_days", "amount_observation_count",
                 "nonpositive_amount_count", "credit_count", "debit_count", "withdrawal_count",
                 "deposit_count", "transfer_count", "negative_balance_count",
                 "overdraft_transaction_count", "ratio_eligible_count"]
RATIO_COLUMNS = ["overdraft_ratio", "share_above_prior_p95"]


def _c(name, passed, detail=""):
    return {"check": name, "status": "PASS" if bool(passed) else "FAIL", "detail": str(detail)}


def row_count_checks(inputs, outputs):
    rows = []
    for t, n_in in inputs.items():
        n_out = outputs[t]
        rows.append({"table": t, "input_rows": n_in, "output_rows": n_out, "rows_removed": n_in - n_out})
    df = pd.DataFrame(rows)
    return df, [_c("no rows removed by cleaning", (df["rows_removed"] == 0).all(),
                   df.set_index("table")["rows_removed"].to_dict())]


def integrity_checks(cleaned):
    fk = Q.foreign_keys(cleaned)
    dates = Q.date_checks(cleaned)
    keys = Q.duplicates_and_keys(cleaned)
    return [
        _c("foreign keys: zero orphans after cleaning", (fk["orphan_rows"] == 0).all(),
           fk.set_index("relationship")["orphan_rows"].to_dict()),
        _c("primary keys unique and non-null after cleaning", keys["pk_unique"].all()),
        _c("no impossible dates (all inside the dataset window, no NaT)",
           (dates["outside_window"] == 0).all() and (dates["nat"] == 0).all(),
           dates[["table", "column", "outside_window", "nat"]].to_dict("records")),
    ]


def feature_checks(af, cf, trans, cutoff_date=None):
    out = []
    out.append(_c("one row per account, unique account_id", af["account_id"].is_unique))
    out.append(_c("one row per client, unique client_id", cf["client_id"].is_unique))
    out.append(_c("counts >= 0", all((af[c] >= 0).all() for c in COUNT_COLUMNS)))
    out.append(_c("ratios between 0 and 1",
                  all(af[c].dropna().between(0, 1).all() for c in RATIO_COLUMNS), RATIO_COLUMNS))
    p = af.dropna(subset=["p25_transaction_amount"])
    order_ok = ((p["min_transaction_amount"] <= p["p25_transaction_amount"]) &
                (p["p25_transaction_amount"] <= p["median_transaction_amount"]) &
                (p["median_transaction_amount"] <= p["p75_transaction_amount"]) &
                (p["p75_transaction_amount"] <= p["p95_transaction_amount"]) &
                (p["p95_transaction_amount"] <= p["max_transaction_amount"])).all()
    out.append(_c("min <= p25 <= median <= p75 <= p95 <= max", order_ok, f"{len(p)} accounts checked"))
    d = af.dropna(subset=["p95_debit_amount"])
    out.append(_c("debit baselines ordered: median <= p75 <= p95",
                  ((d["median_debit_amount"] <= d["p75_debit_amount"]) &
                   (d["p75_debit_amount"] <= d["p95_debit_amount"])).all()))
    num = pd.concat([af.select_dtypes("number"), cf.select_dtypes("number")], axis=1)
    out.append(_c("no infinite values", not np.isinf(num.to_numpy(dtype="float64")).any()))
    out.append(_c("credit_count + debit_count == transaction_count",
                  (af["credit_count"] + af["debit_count"] == af["transaction_count"]).all()))
    out.append(_c("negative_balance_count == overdraft_transaction_count; has_overdraft consistent",
                  (af["negative_balance_count"] == af["overdraft_transaction_count"]).all() and
                  (af["has_overdraft"] == (af["negative_balance_count"] > 0)).all()))
    neg = af[af["minimum_balance"] < 0]
    out.append(_c("overdraft_amount == -minimum_balance where the minimum is negative",
                  np.allclose(neg["overdraft_amount"], -neg["minimum_balance"]) and (af["overdraft_amount"] >= 0).all()))
    out.append(_c("typical_transaction_amount == median_transaction_amount",
                  af["typical_transaction_amount"].fillna(-1).equals(af["median_transaction_amount"].fillna(-1))))
    out.append(_c("first_transaction_date <= last_transaction_date",
                  (af["first_transaction_date"].dropna() <= af["last_transaction_date"].dropna()).all()))
    t = F.apply_cutoff(trans, cutoff_date)
    out.append(_c("sum of account transaction_count == transactions used", af["transaction_count"].sum() == len(t),
                  f"{int(af['transaction_count'].sum()):,} vs {len(t):,}"))
    n_nonpos = int((t["amount"].astype("float64") <= 0).sum())
    out.append(_c("zero-amount rows excluded from amount features only",
                  af["nonpositive_amount_count"].sum() == n_nonpos and
                  af["amount_observation_count"].sum() == len(t) - n_nonpos,
                  f"{n_nonpos} excluded of {len(t):,}"))
    out.append(_c("client: account_count >= owner_account_count, every client linked to >= 1 account",
                  ((cf["account_count"] >= cf["owner_account_count"]).all() and (cf["account_count"] >= 1).all())))
    out.append(_c("client overdraft_account_count <= account_count",
                  (cf["overdraft_account_count"] <= cf["account_count"]).all()))
    if cutoff_date is not None:
        out.append(_c("all features end before the cutoff",
                      (af["last_transaction_date"].dropna() < pd.Timestamp(cutoff_date)).all()))
    return out


def nan_report(af, cf):
    rows = []
    for name, df in (("account_features", af), ("client_features", cf)):
        for c, n in df.isna().sum().items():
            if n:
                rows.append({"table": name, "column": c, "nan_count": int(n), "of_rows": len(df)})
    return pd.DataFrame(rows, columns=["table", "column", "nan_count", "of_rows"])


def deviation_helper_checks():
    a = F.amount_deviation_signals(120, 60, 120)
    b = F.amount_deviation_signals(120, 0, float("nan"))
    return [_c("amount_deviation_signals: 120 vs typical 60 / p95 120 -> 2.0 and 1.0",
               a == {"amount_to_typical_ratio": 2.0, "amount_to_p95_ratio": 1.0}, a),
            _c("amount_deviation_signals: zero or missing baseline -> NaN",
               all(np.isnan(v) for v in b.values()), b)]


def leakage_tests(trans, accounts, district, cutoff, seed=42):
    # 1) Corrupt every transaction ON OR AFTER the cutoff; features computed with the cutoff
    #    must be identical. 2) The same corruption must change the no-cutoff features, which
    #    proves the test is sensitive.
    rng = np.random.default_rng(seed)
    cut = pd.Timestamp(cutoff)
    mod = trans.copy()
    late = (mod["date"] >= cut).to_numpy()
    n = int(late.sum())
    mod.loc[late, "amount"] = (mod.loc[late, "amount"].astype("float64").to_numpy() * rng.uniform(1, 50, n))
    mod.loc[late, "balance"] = (mod.loc[late, "balance"].astype("float64").to_numpy() + rng.normal(0, 1e5, n))
    mod["is_overdraft"] = mod["balance"].astype("float64") < 0
    base = F.compute_account_features(trans, cutoff, accounts, district)
    after = F.compute_account_features(mod, cutoff, accounts, district)
    full = F.compute_account_features(trans, None, accounts, district)
    full_mod = F.compute_account_features(mod, None, accounts, district)
    same = True
    try:
        pd.testing.assert_frame_equal(base, after, check_exact=True)
    except AssertionError as e:
        same = False
        detail = str(e)[:200]
    sensitive = not full.drop(columns="cutoff_date").equals(full_mod.drop(columns="cutoff_date"))
    ratios = F.prior_day_ratios(trans, cutoff)
    return [
        _c(f"cutoff {cutoff}: corrupting {n:,} later transactions leaves all features unchanged", same,
           "identical" if same else detail),
        _c("test is sensitive: the same corruption changes features when no cutoff is applied", sensitive),
        _c("per-transaction ratios use only dates before the cutoff", (ratios["date"] < cut).all(),
           f"{len(ratios):,} rows"),
    ]


def determinism_test(trans, accounts, district):
    a = F.compute_account_features(trans, None, accounts, district)
    b = F.compute_account_features(trans, None, accounts, district)
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        ok = True
    except AssertionError:
        ok = False
    return [_c("feature computation is deterministic (two runs identical, seed 42 where random)", ok)]