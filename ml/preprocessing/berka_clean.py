# ml/preprocessing/berka_clean.py
# Phase 1B cleaning. Conservative: no rows are deleted, original columns are kept,
# and every transformation is logged with table, column, affected rows and reason.

import numpy as np
import pandas as pd

# Transaction semantics. Sources: (1) official description PDF for `operation`;
# (2) Phase 1A balance test for `type` (PRIJEM raises the balance, VYDAJ and VYBER lower it).
TYPE_DIRECTION = {"PRIJEM": "CREDIT", "VYDAJ": "DEBIT", "VYBER": "DEBIT"}
OPERATION_CATEGORY = {
    "VKLAD": "DEPOSIT",            # credit in cash
    "VYBER": "WITHDRAWAL",         # withdrawal in cash
    "VYBER KARTOU": "WITHDRAWAL",  # credit card withdrawal
    "PREVOD Z UCTU": "TRANSFER",   # collection from another bank
    "PREVOD NA UCET": "TRANSFER",  # remittance to another bank
    "NONE": "NONE",                # no operation recorded (see evidence in the log)
}


def _log(log, table, column, transformation, affected, reason):
    log.append({"table": table, "column": column, "transformation": transformation,
                "affected_rows": int(affected), "reason": reason})


def _token(series):
    # the three "empty" encodings preserved by Phase 1A: missing (NA), "" and " "
    s = series.astype(object)
    return pd.Series(np.select([s.isna(), s == "", s == " "], ["missing", "empty", "blank"],
                               default="value"), index=series.index)


def clean_trans(t, log):
    t = t.copy()
    n = len(t)
    for c in ("type", "operation", "k_symbol", "bank"):
        t[c] = t[c].astype(object)

    # k_symbol: keep the original, add the token kind and a normalized label
    tok = _token(t["k_symbol"])
    t["k_symbol_token"] = pd.Categorical(tok, categories=["value", "empty", "blank", "missing"])
    t["k_symbol_normalized"] = pd.Categorical(t["k_symbol"].where(tok == "value", "UNKNOWN"))
    _log(log, "trans", "k_symbol", "added k_symbol_token (value/empty/blank/missing) and "
         "k_symbol_normalized (non-value -> UNKNOWN); original column kept",
         (tok != "value").sum(),
         "three different upstream encodings occur with different operations; the token column "
         "preserves the distinction while the normalized column gives one feature-level label")

    # operation: blank exists only for interest credits (evidence checked here, not assumed)
    op_empty = t["operation"].isna() | (t["operation"] == "")
    interest = (t["type"] == "PRIJEM") & (t["k_symbol"] == "UROK")
    if not (op_empty == interest).all():
        raise ValueError("operation blank no longer coincides with type=PRIJEM & k_symbol=UROK")
    t["operation_normalized"] = pd.Categorical(t["operation"].where(~op_empty, "NONE"))
    _log(log, "trans", "operation", "added operation_normalized (blank -> NONE); original kept",
         op_empty.sum(), "every blank operation is a PRIJEM row with k_symbol UROK (interest "
         "credit), so it means 'no operation', not 'unknown'")

    # direction and category flags from the documented mapping
    direction = t["type"].map(TYPE_DIRECTION)
    if direction.isna().any():
        raise ValueError("unmapped transaction type")
    t["direction"] = pd.Categorical(direction)
    t["operation_category"] = pd.Categorical(t["operation_normalized"].astype(object).map(OPERATION_CATEGORY))
    t["is_credit"] = direction == "CREDIT"
    t["is_debit"] = direction == "DEBIT"
    t["is_withdrawal"] = t["operation_category"] == "WITHDRAWAL"
    t["is_deposit"] = t["operation_category"] == "DEPOSIT"
    t["is_transfer"] = t["operation_category"] == "TRANSFER"
    _log(log, "trans", "type, operation", "added direction, operation_category and is_credit/"
         "is_debit/is_withdrawal/is_deposit/is_transfer flags", n,
         "documented mapping in TYPE_DIRECTION and OPERATION_CATEGORY; type=VYBER is not in the "
         "official description and is treated as DEBIT because its balance falls by the amount")

    # counterparty fields: bank "" always comes with account "0" (placeholder), NA means absent
    bank_missing = t["bank"].isna() | (t["bank"] == "")
    acct = t["account"].astype(object)
    status = np.select([acct.isna(), acct == "0"], ["absent", "placeholder"], default="present")
    t["bank_missing"] = bank_missing
    t["counterparty_account_missing"] = pd.Series(status, index=t.index) != "present"
    t["counterparty_status"] = pd.Categorical(status, categories=["present", "placeholder", "absent"])
    _log(log, "trans", "bank", "added bank_missing (NA or empty)", bank_missing.sum(),
         "a missing counterparty bank is expected for cash and interest rows, not an error")
    _log(log, "trans", "account", "added counterparty_account_missing and counterparty_status "
         "(present/placeholder '0'/absent)", (status != "present").sum(),
         f"{int((status == 'placeholder').sum()):,} rows carry the placeholder '0' with an empty bank code, "
         f"{int((status == 'absent').sum()):,} rows are absent; both mean no counterparty, but the "
         "upstream encodings differ and are kept apart")

    # zero amounts and overdraft indicators: flagged, never removed
    t["amount_nonpositive"] = t["amount"].astype("float64") <= 0
    t["is_overdraft"] = t["balance"].astype("float64") < 0
    _log(log, "trans", "amount", "added amount_nonpositive flag; rows kept", t["amount_nonpositive"].sum(),
         "all are exactly 0.0 interest postings (see investigation); excluded from amount "
         "statistics and ratios, kept in counts and balance features")
    _log(log, "trans", "balance", "added is_overdraft (balance < 0); rows kept", t["is_overdraft"].sum(),
         "negative balances are genuine account behavior")

    for c in ("type", "operation", "k_symbol", "bank"):
        t[c] = t[c].astype("category")
    # deterministic order; trans_id is NOT chronological (Phase 1A), so sort by date first
    t = t.sort_values(["account_id", "date", "trans_id"]).reset_index(drop=True)
    _log(log, "trans", "(row order)", "sorted by account_id, date, trans_id", n,
         "trans_id is not chronological; (date, trans_id) gives a deterministic order")
    return t


def investigate_nonpositive(trans):
    # context for each non-positive amount: previous balance, duplicates, token kinds
    t = trans.sort_values(["account_id", "date", "trans_id"]).copy()
    t["previous_balance"] = t.groupby("account_id")["balance"].shift(1)
    z = t[t["amount_nonpositive"]].copy()
    key = ["account_id", "date", "type", "k_symbol", "amount", "balance"]
    z["same_values_as_another_row"] = z.duplicated(key, keep=False)
    z["balance_unchanged"] = (z["balance"].astype("float64") == z["previous_balance"].astype("float64"))
    return z[["trans_id", "account_id", "date", "type", "operation", "k_symbol", "amount", "balance",
              "previous_balance", "balance_unchanged", "same_values_as_another_row"]]


def clean_tables(tables):
    log = []
    out = {k: v.copy() for k, v in tables.items()}
    out["trans"] = clean_trans(tables["trans"], log)

    # order.k_symbol has only the blank encoding " "
    o = out["order"]
    tok = _token(o["k_symbol"])
    o["k_symbol_token"] = pd.Categorical(tok, categories=["value", "empty", "blank", "missing"])
    o["k_symbol_normalized"] = pd.Categorical(o["k_symbol"].astype(object).where(tok == "value", "UNKNOWN"))
    _log(log, "order", "k_symbol", "added k_symbol_token and k_symbol_normalized", (tok != "value").sum(),
         "same treatment as trans.k_symbol; original column kept")

    # district: explicit missing indicators, no imputation
    d = out["district"]
    for c in d.columns:
        if d[c].isna().any():
            d[f"{c}_missing"] = d[c].isna()
            _log(log, "district", c, f"added {c}_missing; value left missing (no imputation)",
                 d[c].isna().sum(), "source marks it with '?'; inventing a value would be a fabrication")

    for t in ("account", "card", "client", "disp", "loan"):
        _log(log, t, "(all)", "no transformation", 0,
             "Phase 1A found no duplicates, orphans, impossible values or missing data here; "
             "loan.status is kept descriptive only and is not turned into a label")
    return out, log