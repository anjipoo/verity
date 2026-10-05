# ml/preprocessing/berka_quality.py
# Read-only quality and integrity checks. Every function returns a DataFrame and
# changes nothing: cleaning decisions come later and must cite this evidence.

import numpy as np
import pandas as pd

from ml.preprocessing import berka_schema as S


def duplicates_and_keys(tables):
    rows = []
    for t, df in tables.items():
        pk = S.PRIMARY_KEYS[t]
        rows.append({
            "table": t, "rows": len(df), "primary_key": pk,
            "duplicate_full_rows": int(df.duplicated().sum()),
            "duplicate_pk_values": int(df[pk].duplicated().sum()),
            "null_pk_values": int(df[pk].isna().sum()),
            "pk_unique": bool(df[pk].is_unique and df[pk].notna().all()),
        })
    return pd.DataFrame(rows)


def null_counts(tables):
    rows = []
    for t, df in tables.items():
        for c in df.columns:
            n = int(df[c].isna().sum())
            if n:
                rows.append({"table": t, "column": c, "null_count": n,
                             "null_pct": round(100 * n / len(df), 3)})
    return pd.DataFrame(rows, columns=["table", "column", "null_count", "null_pct"])


def date_checks(tables):
    lo, hi = pd.Timestamp(S.DATE_WINDOW[0]), pd.Timestamp(S.DATE_WINDOW[1])
    rows = []
    for t, df in tables.items():
        for c in df.columns:
            if str(df[c].dtype).startswith("datetime"):
                out = int(((df[c] < lo) | (df[c] > hi)).sum())
                rows.append({"table": t, "column": c, "min": df[c].min().date(),
                             "max": df[c].max().date(), "nat": int(df[c].isna().sum()),
                             "outside_window": out})
    return pd.DataFrame(rows)


def numeric_sanity(tables):
    # (table, column, rule text, mask function, severity)
    rules = [
        ("trans", "amount", "amount <= 0", lambda d: d.amount <= 0, "review"),
        ("trans", "balance", "balance < 0 (overdraft)", lambda d: d.balance < 0, "informational"),
        ("loan", "amount", "amount <= 0", lambda d: d.amount <= 0, "review"),
        ("loan", "duration", "duration <= 0", lambda d: d.duration <= 0, "review"),
        ("loan", "payments", "payments <= 0", lambda d: d.payments <= 0, "review"),
        ("order", "amount", "amount <= 0", lambda d: d.amount <= 0, "review"),
        ("district", "n_inhabitants", "n_inhabitants <= 0", lambda d: d.n_inhabitants <= 0, "review"),
    ]
    rows = []
    for t, col, text, fn, sev in rules:
        mask = fn(tables[t]).fillna(False)
        rows.append({"table": t, "column": col, "rule": text, "violations": int(mask.sum()),
                     "of_rows": len(tables[t]), "severity": sev})
    return pd.DataFrame(rows)


def categorical_domains(tables):
    rows = []
    for (t, col), expected in S.EXPECTED_DOMAINS.items():
        seen = set(tables[t][col].dropna().astype(str).unique())
        rows.append({"table": t, "column": col, "distinct_values": len(seen),
                     "unexpected": sorted(seen - expected), "absent_vs_baseline": sorted(expected - seen)})
    return pd.DataFrame(rows)


def foreign_keys(tables):
    rows = []
    for ct, cc, pt, pc in S.FOREIGN_KEYS:
        child, parent = tables[ct][cc], tables[pt][pc]
        orphans = ~child.isin(parent) & child.notna()
        rows.append({"relationship": f"{ct}.{cc} -> {pt}.{pc}", "child_rows": len(child),
                     "null_fk": int(child.isna().sum()), "orphan_rows": int(orphans.sum()),
                     "parents_without_children": int((~parent.isin(child)).sum()),
                     "parent_rows": len(parent)})
    return pd.DataFrame(rows)


def cardinalities(tables):
    # children per parent, counting parents with zero children as 0
    def per_parent(child_col, parent_ids):
        counts = child_col.value_counts()
        return counts.reindex(parent_ids, fill_value=0).astype(int)

    acc = tables["account"].account_id
    disp = tables["disp"]
    cli = tables["client"].client_id
    owners = disp[disp.type.astype(str) == "OWNER"]
    specs = [
        ("disp per account (clients linked to an account)", per_parent(disp.account_id, acc)),
        ("OWNER disp per account", per_parent(owners.account_id, acc)),
        ("accounts (via disp) per client", per_parent(disp.client_id, cli)),
        ("transactions per account", per_parent(tables["trans"].account_id, acc)),
        ("loans per account", per_parent(tables["loan"].account_id, acc)),
        ("orders per account", per_parent(tables["order"].account_id, acc)),
        ("cards per disp", per_parent(tables["card"].disp_id, disp.disp_id)),
    ]
    rows = []
    for name, s in specs:
        rows.append({"relationship": name, "parents": len(s), "min": int(s.min()),
                     "median": float(s.median()), "mean": round(float(s.mean()), 2),
                     "max": int(s.max()), "parents_with_zero": int((s == 0).sum()),
                     "distribution": dict(sorted(s.value_counts().head(6).to_dict().items()))})
    return pd.DataFrame(rows)


def cross_table_dates(tables):
    # events dated before the account existed would be impossible
    acc = tables["account"].set_index("account_id").date.rename("account_date")
    rows = []
    for t, col in [("trans", "date"), ("loan", "date")]:
        m = tables[t].join(acc, on="account_id")
        rows.append({"check": f"{t}.{col} before account creation",
                     "violations": int((m[col] < m.account_date).sum()), "rows": len(m)})
    card = tables["card"].merge(tables["disp"][["disp_id", "account_id"]], on="disp_id").join(acc, on="account_id")
    rows.append({"check": "card.issued before account creation",
                 "violations": int((card.issued < card.account_date).sum()), "rows": len(card)})
    return pd.DataFrame(rows)


def type_direction_evidence(trans):
    # Does the balance move up or down by `amount` for each transaction `type`?
    # Evidence for how to read `type`, instead of assuming a credit/debit meaning.
    # Order within an account is (date, trans_id); same-day ordering is ambiguous,
    # so exact matches are a lower bound.
    s = trans[["account_id", "date", "trans_id", "type", "amount", "balance"]].copy()
    s["type"] = s["type"].astype(str)
    s = s.sort_values(["account_id", "date", "trans_id"])
    d = (s.groupby("account_id")["balance"].diff()).astype("float64")
    amt = s["amount"].astype("float64")
    ok = d.notna()
    rows = []
    for ty, idx in s[ok].groupby("type").groups.items():
        dd, aa = d.loc[idx], amt.loc[idx]
        rows.append({"type": ty, "pairs_checked": len(idx),
                     "balance_rose_by_amount_pct": round(100 * np.isclose(dd, aa, atol=0.01).mean(), 1),
                     "balance_fell_by_amount_pct": round(100 * np.isclose(dd, -aa, atol=0.01).mean(), 1)})
    return pd.DataFrame(rows)


def trans_id_order_check(trans):
    # is trans_id a usable chronological order? (affects any sequence feature later)
    s = trans.sort_values(["account_id", "trans_id"])
    diff = s.groupby("account_id")["date"].diff().dropna()
    return pd.DataFrame([{"consecutive_pairs_by_trans_id": len(diff),
                          "pairs_with_earlier_date": int((diff < pd.Timedelta(0)).sum()),
                          "share_non_decreasing_pct": round(100 * (diff >= pd.Timedelta(0)).mean(), 2)}])


def official_row_counts(tables):
    rows = []
    for t, expected in S.OFFICIAL_ROW_COUNTS.items():
        rows.append({"table": t, "loaded": len(tables[t]), "official_description": expected,
                     "match": len(tables[t]) == expected})
    return pd.DataFrame(rows)


def documentation_discrepancies(tables):
    # values present in the data but not listed in the official description, with row counts
    rows = []
    for (t, col), documented in S.DOCUMENTED_VALUES.items():
        s = tables[t][col].astype(object)
        for value, n in s.value_counts(dropna=False).items():
            shown = "<missing>" if pd.isna(value) else repr(value)
            if pd.isna(value) or value not in documented:
                rows.append({"table": t, "column": col, "value_in_data": shown, "rows": int(n),
                             "status": "not in official description"})
    d = tables["disp"].type.astype(str).value_counts()
    rows.append({"table": "disp", "column": "type", "value_in_data": "DISPONENT", "rows": int(d.get("DISPONENT", 0)),
                 "status": "description says only 'owner/user' (no literal strings given)"})
    return pd.DataFrame(rows)