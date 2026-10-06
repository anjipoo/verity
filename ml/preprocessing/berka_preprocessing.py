# ml/preprocessing/berka_features.py
# Behavioral features from cleaned Berka tables. Advisory signals only: they never
# decide ALLOW / REVIEW / BLOCK. All amounts are historical Czech-bank values and are
# used as relative baselines, never as INR limits.
#
# Leakage rule: compute_account_features(transactions, cutoff_date) uses ONLY rows with
# date < cutoff_date. With cutoff_date=None the full history is used (offline description
# only; never use that for a decision-time evaluation).

import numpy as np
import pandas as pd

MIN_PRIOR_TRANSACTIONS = 10  # prior positive-amount transactions needed before a ratio is defined
DISTRICT_CONTEXT = ["region", "n_inhabitants", "average_salary", "unemployment_rate_1995",
                    "unemployment_rate_1996", "entrepreneurs_per_1000"]
COUNT_FLAGS = {"credit_count": "is_credit", "debit_count": "is_debit",
               "withdrawal_count": "is_withdrawal", "deposit_count": "is_deposit",
               "transfer_count": "is_transfer"}


def apply_cutoff(transactions, cutoff_date=None):
    if cutoff_date is None:
        return transactions
    return transactions[transactions["date"] < pd.Timestamp(cutoff_date)]


def prior_day_ratios(transactions, cutoff_date=None, min_prior=MIN_PRIOR_TRANSACTIONS):
    # Per-transaction ratios against a baseline built ONLY from strictly earlier days of the
    # same account. Same-day rows are never in the baseline, so row order within a day
    # (which is ambiguous in Berka) cannot leak information.
    t = apply_cutoff(transactions, cutoff_date)
    p = t.loc[t["amount"].astype("float64") > 0, ["account_id", "date", "trans_id", "amount"]].copy()
    p["amount"] = p["amount"].astype("float64")
    p = p.sort_values(["account_id", "date", "trans_id"]).reset_index(drop=True)
    g = p.groupby("account_id")["amount"]
    p["n_through"] = g.cumcount() + 1
    p["median_through"] = g.expanding().median().reset_index(level=0, drop=True)
    p["p95_through"] = g.expanding().quantile(0.95).reset_index(level=0, drop=True)
    eod = p.groupby(["account_id", "date"], sort=False).tail(1)[
        ["account_id", "date", "n_through", "median_through", "p95_through"]].copy()
    cols = ["prior_n", "prior_median", "prior_p95"]
    eod[cols] = eod.groupby("account_id")[["n_through", "median_through", "p95_through"]].shift(1)
    p = p.merge(eod[["account_id", "date"] + cols], on=["account_id", "date"], how="left")
    ok = (p["prior_n"] >= min_prior) & (p["prior_median"] > 0) & (p["prior_p95"] > 0)
    p["amount_to_typical_ratio"] = np.where(ok, p["amount"] / p["prior_median"], np.nan)
    p["amount_to_p95_ratio"] = np.where(ok, p["amount"] / p["prior_p95"], np.nan)
    return p[["account_id", "date", "trans_id", "amount", "prior_n", "prior_median", "prior_p95",
              "amount_to_typical_ratio", "amount_to_p95_ratio"]]


def amount_deviation_signals(amount, typical_amount, p95_amount):
    # Runtime helper for later phases: how large is a requested amount relative to an
    # account's own historical baseline? Unitless; NaN when a baseline is missing or zero.
    def ratio(a, b):
        return float(a) / float(b) if b is not None and not pd.isna(b) and b > 0 else float("nan")
    return {"amount_to_typical_ratio": ratio(amount, typical_amount),
            "amount_to_p95_ratio": ratio(amount, p95_amount)}


def compute_account_features(transactions, cutoff_date=None, accounts=None, district=None):
    t = apply_cutoff(transactions, cutoff_date)
    w = t[["account_id", "date", "type", "amount", "balance", "is_credit", "is_debit",
           "is_withdrawal", "is_deposit", "is_transfer", "is_overdraft"]].copy()
    w["amount"] = w["amount"].astype("float64")
    w["balance"] = w["balance"].astype("float64")
    w["account_id"] = w["account_id"].astype("int64")
    g = w.groupby("account_id")

    f = pd.DataFrame({"transaction_count": g.size()})
    f["first_transaction_date"] = g["date"].min()
    f["last_transaction_date"] = g["date"].max()
    f["active_days"] = g["date"].nunique()
    f["history_days"] = (f["last_transaction_date"] - f["first_transaction_date"]).dt.days + 1
    f["transactions_per_active_day"] = f["transaction_count"] / f["active_days"]

    # amount features use positive amounts only (zero-value interest postings are excluded)
    pos = w[w["amount"] > 0]
    gp = pos.groupby("account_id")["amount"]
    f["amount_observation_count"] = gp.size()
    f["nonpositive_amount_count"] = w[w["amount"] <= 0].groupby("account_id").size()
    f["total_transaction_amount"] = gp.sum()
    f["mean_transaction_amount"] = gp.mean()
    f["median_transaction_amount"] = gp.median()
    f["std_transaction_amount"] = gp.std()
    f["min_transaction_amount"] = gp.min()
    f["max_transaction_amount"] = gp.max()
    for name, q in (("p25", 0.25), ("p75", 0.75), ("p95", 0.95)):
        f[f"{name}_transaction_amount"] = gp.quantile(q)
    f["typical_transaction_amount"] = f["median_transaction_amount"]  # alias, documented
    f["credit_amount_total"] = pos[pos["is_credit"]].groupby("account_id")["amount"].sum()
    f["debit_amount_total"] = pos[pos["is_debit"]].groupby("account_id")["amount"].sum()
    gd = pos[pos["is_debit"]].groupby("account_id")["amount"]
    f["median_debit_amount"] = gd.median()
    f["p75_debit_amount"] = gd.quantile(0.75)
    f["p95_debit_amount"] = gd.quantile(0.95)

    for out_name, flag in COUNT_FLAGS.items():
        f[out_name] = g[flag].sum()

    # balance features use every row, including zero-value postings (the balance is still valid)
    gb = g["balance"]
    f["mean_balance"] = gb.mean()
    f["median_balance"] = gb.median()
    f["minimum_balance"] = gb.min()
    f["maximum_balance"] = gb.max()
    f["balance_std"] = gb.std()
    f["negative_balance_count"] = g["is_overdraft"].sum()
    f["overdraft_transaction_count"] = f["negative_balance_count"]
    f["has_overdraft"] = f["negative_balance_count"] > 0
    f["overdraft_amount"] = (-w["balance"]).clip(lower=0).groupby(w["account_id"]).max()
    f["overdraft_ratio"] = f["negative_balance_count"] / f["transaction_count"]

    # deviation of each transaction from the account's prior-day baseline, summarised per account
    r = prior_day_ratios(t)
    r = r.dropna(subset=["amount_to_typical_ratio"])
    gr = r.groupby("account_id")
    f["ratio_eligible_count"] = gr.size()
    f["amount_to_typical_ratio_median"] = gr["amount_to_typical_ratio"].median()
    f["amount_to_typical_ratio_max"] = gr["amount_to_typical_ratio"].max()
    f["amount_to_p95_ratio_max"] = gr["amount_to_p95_ratio"].max()
    f["share_above_prior_p95"] = (r["amount_to_p95_ratio"] > 1).groupby(r["account_id"]).mean()

    base = accounts[["account_id", "district_id"]].copy() if accounts is not None \
        else pd.DataFrame({"account_id": f.index})
    base["account_id"] = base["account_id"].astype("int64")
    out = base.merge(f.reset_index(), on="account_id", how="left")
    zero_fill = ["transaction_count", "active_days", "history_days", "amount_observation_count",
                 "nonpositive_amount_count", "ratio_eligible_count", "negative_balance_count",
                 "overdraft_transaction_count", "overdraft_amount", "credit_amount_total",
                 "debit_amount_total", "total_transaction_amount", "overdraft_ratio",
                 *COUNT_FLAGS]
    for c in zero_fill:
        out[c] = out[c].fillna(0)
    out["has_overdraft"] = out["has_overdraft"].fillna(False).astype(bool)
    for c in ("transaction_count", "active_days", "history_days", "amount_observation_count",
              "nonpositive_amount_count", "ratio_eligible_count", "negative_balance_count",
              "overdraft_transaction_count", *COUNT_FLAGS):
        out[c] = out[c].astype("int64")
    out["cutoff_date"] = pd.Timestamp(cutoff_date) if cutoff_date is not None else pd.NaT
    out["cutoff_date"] = pd.to_datetime(out["cutoff_date"])

    if district is not None and "district_id" in out:
        d = district[["district_id"] + DISTRICT_CONTEXT].rename(
            columns={c: f"district_{c}" for c in DISTRICT_CONTEXT})
        d["district_region"] = d["district_region"].astype(object)
        out = out.merge(d, on="district_id", how="left")
    return out.sort_values("account_id").reset_index(drop=True)


def compute_client_features(account_features, disp, client, district=None):
    # client -> disp -> account -> (account features). Semantics stay separate: nothing here
    # re-reads transactions, it aggregates the already cutoff-safe account features.
    link = disp[["client_id", "account_id", "type"]].copy()
    link["client_id"] = link["client_id"].astype("int64")
    link["account_id"] = link["account_id"].astype("int64")
    link = link.merge(account_features[["account_id", "transaction_count", "total_transaction_amount",
                                        "mean_balance", "has_overdraft"]], on="account_id", how="left")
    g = link.groupby("client_id")
    f = pd.DataFrame({"account_count": g["account_id"].nunique()})
    f["owner_account_count"] = link[link["type"].astype(str) == "OWNER"].groupby("client_id")["account_id"].nunique()
    f["total_transaction_count"] = g["transaction_count"].sum()
    f["total_transaction_amount"] = g["total_transaction_amount"].sum()
    f["average_account_balance"] = g["mean_balance"].mean()
    f["overdraft_account_count"] = g["has_overdraft"].sum()

    base = client[["client_id", "district_id"]].copy()
    base["client_id"] = base["client_id"].astype("int64")
    out = base.merge(f.reset_index(), on="client_id", how="left")
    for c in ("account_count", "owner_account_count", "total_transaction_count",
              "total_transaction_amount", "overdraft_account_count"):
        out[c] = out[c].fillna(0)
    for c in ("account_count", "owner_account_count", "total_transaction_count", "overdraft_account_count"):
        out[c] = out[c].astype("int64")
    cut = account_features["cutoff_date"].iloc[0] if len(account_features) else pd.NaT
    out["cutoff_date"] = pd.to_datetime(pd.Series([cut] * len(out)))
    if district is not None:
        d = district[["district_id"] + DISTRICT_CONTEXT].copy()
        d = d.rename(columns={c: f"client_district_{c}" for c in DISTRICT_CONTEXT})
        d["client_district_region"] = d["client_district_region"].astype(object)
        out = out.merge(d, on="district_id", how="left")
    return out.sort_values("client_id").reset_index(drop=True)