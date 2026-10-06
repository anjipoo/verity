# ml/preprocessing/berka_feature_docs.py
# Feature dictionary for Phase 1B. build_feature_dictionary() fails if a generated column
# is undocumented or a documented column does not exist, so the docs cannot drift.

UNIT_AMT = "historical currency units as stored in Berka (not converted, not INR)"
CUT = "uses only transactions with date < cutoff_date; no cutoff means the full history (offline description only)"
CUT_STATIC = "static district attribute from the source file; not time-varying and not affected by the cutoff"
CUT_CLIENT = "aggregates account features, so it inherits their cutoff behavior"
CUT_RATIO = ("each transaction is compared with a baseline from strictly earlier days of the same account, "
             "and only rows before cutoff_date are used")

GROUPS = {
    "key": ("account / account", "identifier or context", CUT_STATIC),
    "activity": ("trans", "historical activity baseline -> velocity-deviation signal", CUT),
    "amount": ("trans", "historical behavioral baseline -> amount-deviation signal", CUT),
    "mix": ("trans", "historical behavior mix -> behavioral context signal", CUT),
    "balance": ("trans", "historical account behavior -> financial-context signal", CUT),
    "ratio": ("trans", "calibration of the amount-deviation signal against the account's own history", CUT_RATIO),
    "district": ("account, district", "contextual feature only; no causal claim about financial risk", CUT_STATIC),
    "client": ("client, disp, account, trans", "relationship and graph context", CUT_CLIENT),
}

# name, definition, calculation, dtype, unit, group, limitations
ACCOUNT = [
    ("account_id", "Account identifier", "from account.account_id", "int64", "id", "key", "Berka identifier, not a real account number"),
    ("district_id", "District of the account", "account.district_id", "Int64", "id", "key", ""),
    ("cutoff_date", "Cutoff used for this feature row", "argument of compute_account_features", "datetime64", "date", "key", "NaT means no cutoff was applied"),
    ("transaction_count", "Transactions in the window", "count of rows", "int64", "count", "activity", "includes zero-amount interest postings"),
    ("first_transaction_date", "Date of the earliest transaction", "min(date)", "datetime64", "date", "activity", "NaT when no transactions"),
    ("last_transaction_date", "Date of the latest transaction", "max(date)", "datetime64", "date", "activity", "NaT when no transactions"),
    ("active_days", "Distinct dates with at least one transaction", "nunique(date)", "int64", "days", "activity", ""),
    ("history_days", "Span of observed history", "(last - first).days + 1", "int64", "days", "activity", "measures observed span, not account age"),
    ("transactions_per_active_day", "Average transactions on days with activity", "transaction_count / active_days", "float64", "transactions/day", "activity", "monthly interest and fee postings raise the baseline"),
    ("amount_observation_count", "Transactions used for amount statistics", "count of rows with amount > 0", "int64", "count", "amount", "zero-amount postings excluded"),
    ("nonpositive_amount_count", "Zero-amount postings excluded from amount statistics", "count of rows with amount <= 0", "int64", "count", "amount", "all observed values are exactly 0"),
    ("total_transaction_amount", "Gross amount moved (credits plus debits)", "sum of amount where amount > 0", "float64", "currency units", "amount", "gross volume, not net flow"),
    ("mean_transaction_amount", "Mean amount", "mean of amount > 0", "float64", "currency units", "amount", "skewed by large transfers"),
    ("median_transaction_amount", "Median amount", "median of amount > 0", "float64", "currency units", "amount", "dominated by small recurring fees and interest"),
    ("std_transaction_amount", "Standard deviation of amount", "sample std (ddof=1) of amount > 0", "float64", "currency units", "amount", "NaN with fewer than 2 observations"),
    ("min_transaction_amount", "Smallest positive amount", "min of amount > 0", "float64", "currency units", "amount", ""),
    ("max_transaction_amount", "Largest amount", "max of amount > 0", "float64", "currency units", "amount", ""),
    ("p25_transaction_amount", "25th percentile of amount", "linear-interpolated quantile 0.25 of amount > 0", "float64", "currency units", "amount", ""),
    ("p75_transaction_amount", "75th percentile of amount", "linear-interpolated quantile 0.75 of amount > 0", "float64", "currency units", "amount", ""),
    ("p95_transaction_amount", "95th percentile of amount", "linear-interpolated quantile 0.95 of amount > 0", "float64", "currency units", "amount", "tail baseline; unstable on short histories"),
    ("typical_transaction_amount", "Typical amount baseline", "alias of median_transaction_amount", "float64", "currency units", "amount", "identical to the median; kept because Verity names it separately"),
    ("credit_amount_total", "Total credited amount", "sum of amount > 0 where direction is CREDIT", "float64", "currency units", "amount", "direction from type (PRIJEM)"),
    ("debit_amount_total", "Total debited amount", "sum of amount > 0 where direction is DEBIT", "float64", "currency units", "amount", "type VYDAJ and VYBER"),
    ("median_debit_amount", "Median outgoing amount", "median of debit amount > 0", "float64", "currency units", "amount", "added because agent requests are outgoing payments"),
    ("p75_debit_amount", "75th percentile of outgoing amount", "quantile 0.75 of debit amount > 0", "float64", "currency units", "amount", ""),
    ("p95_debit_amount", "95th percentile of outgoing amount", "quantile 0.95 of debit amount > 0", "float64", "currency units", "amount", ""),
    ("credit_count", "Credit transactions", "count where type is PRIJEM", "int64", "count", "mix", "mapping in berka_clean.TYPE_DIRECTION"),
    ("debit_count", "Debit transactions", "count where type is VYDAJ or VYBER", "int64", "count", "mix", "VYBER is not in the official description; balance evidence says debit"),
    ("withdrawal_count", "Withdrawals", "count where operation is VYBER or VYBER KARTOU", "int64", "count", "mix", "cash and card withdrawals"),
    ("deposit_count", "Cash deposits", "count where operation is VKLAD", "int64", "count", "mix", ""),
    ("transfer_count", "Transfers", "count where operation is PREVOD Z UCTU or PREVOD NA UCET", "int64", "count", "mix", "incoming and outgoing together"),
    ("mean_balance", "Mean balance after transaction", "mean of balance", "float64", "currency units", "balance", "one balance per transaction, not time-weighted"),
    ("median_balance", "Median balance", "median of balance", "float64", "currency units", "balance", ""),
    ("minimum_balance", "Lowest balance", "min of balance", "float64", "currency units", "balance", ""),
    ("maximum_balance", "Highest balance", "max of balance", "float64", "currency units", "balance", ""),
    ("balance_std", "Balance volatility", "sample std of balance", "float64", "currency units", "balance", ""),
    ("negative_balance_count", "Transactions leaving a negative balance", "count of balance < 0", "int64", "count", "balance", "same value as overdraft_transaction_count"),
    ("overdraft_transaction_count", "Transactions leaving an overdraft", "count of balance < 0", "int64", "count", "balance", "duplicate of negative_balance_count, requested by name"),
    ("has_overdraft", "Account was ever overdrawn in the window", "negative_balance_count > 0", "bool", "flag", "balance", ""),
    ("overdraft_amount", "Deepest overdraft", "max(-balance) over negative balances, else 0", "float64", "currency units", "balance", "depth, not total owed"),
    ("overdraft_ratio", "Share of transactions with negative balance", "negative_balance_count / transaction_count", "float64", "ratio 0-1", "balance", ""),
    ("ratio_eligible_count", "Transactions with a defined ratio", "count with at least 10 prior positive transactions on earlier days", "int64", "count", "ratio", "MIN_PRIOR_TRANSACTIONS = 10"),
    ("amount_to_typical_ratio_median", "Median of amount / prior median", "median of amount_to_typical_ratio over eligible rows", "float64", "ratio", "ratio", "NaN with too little history"),
    ("amount_to_typical_ratio_max", "Largest amount / prior median", "max of amount_to_typical_ratio", "float64", "ratio", "ratio", "driven by one transaction"),
    ("amount_to_p95_ratio_max", "Largest amount / prior p95", "max of amount_to_p95_ratio", "float64", "ratio", "ratio", ""),
    ("share_above_prior_p95", "Share of transactions above their own prior p95", "mean(amount_to_p95_ratio > 1)", "float64", "ratio 0-1", "ratio", "sanity anchor: close to 0.05 minus ties"),
    ("district_region", "Region of the account's district", "district.region", "object", "label", "district", "context only"),
    ("district_n_inhabitants", "Inhabitants of the district", "district.n_inhabitants", "Int64", "persons", "district", "context only"),
    ("district_average_salary", "Average salary in the district", "district.average_salary", "Int64", "currency units", "district", "context only"),
    ("district_unemployment_rate_1995", "Unemployment rate 1995", "district.unemployment_rate_1995", "Float64", "percent", "district", "missing for one district ('?' in the source)"),
    ("district_unemployment_rate_1996", "Unemployment rate 1996", "district.unemployment_rate_1996", "Float64", "percent", "district", "context only"),
    ("district_entrepreneurs_per_1000", "Entrepreneurs per 1000 inhabitants", "district.entrepreneurs_per_1000", "Int64", "per 1000", "district", "context only"),
]

CLIENT = [
    ("client_id", "Client identifier", "client.client_id", "int64", "id", "key", "Berka identifier"),
    ("district_id", "District of the client", "client.district_id", "Int64", "id", "key", "may differ from the account's district"),
    ("cutoff_date", "Cutoff used", "inherited from account features", "datetime64", "date", "key", "NaT means no cutoff"),
    ("account_count", "Accounts linked through disp", "distinct disp.account_id per client", "int64", "count", "client", ""),
    ("owner_account_count", "Accounts where the client is OWNER", "distinct accounts with disp.type OWNER", "int64", "count", "client", "a DISPONENT client has 0"),
    ("total_transaction_count", "Transactions on linked accounts", "sum of account transaction_count", "int64", "count", "client", "an account's totals appear for every client linked to it"),
    ("total_transaction_amount", "Gross amount on linked accounts", "sum of account total_transaction_amount", "float64", "currency units", "client", "same double-counting note"),
    ("average_account_balance", "Mean of linked accounts' mean balances", "mean of account mean_balance", "float64", "currency units", "client", "NaN with no transactions"),
    ("overdraft_account_count", "Linked accounts that were overdrawn", "sum of account has_overdraft", "int64", "count", "client", ""),
    ("client_district_region", "Region of the client's district", "district.region", "object", "label", "district", "context only"),
    ("client_district_n_inhabitants", "Inhabitants of the client's district", "district.n_inhabitants", "Int64", "persons", "district", "context only"),
    ("client_district_average_salary", "Average salary in the client's district", "district.average_salary", "Int64", "currency units", "district", "context only"),
    ("client_district_unemployment_rate_1995", "Unemployment rate 1995", "district.unemployment_rate_1995", "Float64", "percent", "district", "missing for one district"),
    ("client_district_unemployment_rate_1996", "Unemployment rate 1996", "district.unemployment_rate_1996", "Float64", "percent", "district", "context only"),
    ("client_district_entrepreneurs_per_1000", "Entrepreneurs per 1000", "district.entrepreneurs_per_1000", "Int64", "per 1000", "district", "context only"),
]

VERITY_MAPPING = [
    {"feature": "p95_transaction_amount / p95_debit_amount", "role": "historical behavioral baseline",
     "signal": "amount-deviation signal (compare a requested amount through amount_deviation_signals)"},
    {"feature": "typical_transaction_amount", "role": "historical behavioral baseline", "signal": "amount-to-typical signal"},
    {"feature": "transactions_per_active_day", "role": "historical activity baseline", "signal": "velocity-deviation signal"},
    {"feature": "overdraft_ratio, minimum_balance, overdraft_amount", "role": "historical account behavior",
     "signal": "financial-context signal"},
    {"feature": "district_* and client_district_*", "role": "context", "signal": "none; descriptive context only"},
    {"feature": "all features", "role": "advisory",
     "signal": "they inform the deterministic Risk Engine; they never produce ALLOW / REVIEW / BLOCK"},
]


def build_feature_dictionary(account_columns, client_columns):
    def expand(entries, table):
        rows = []
        for name, definition, calc, dtype, unit, group, limits in entries:
            src, use, cut = GROUPS[group]
            if name.startswith(("district_", "client_district_")):
                src = "district"
            rows.append({"feature_name": name, "output_table": table, "source_tables": src,
                         "definition": definition, "calculation": calc, "data_type": dtype,
                         "unit": UNIT_AMT if unit == "currency units" else unit,
                         "temporal_cutoff_behavior": cut, "intended_use": use,
                         "limitations": limits or "none beyond the dataset limits"})
        return rows

    for entries, cols, label in ((ACCOUNT, account_columns, "account_features"),
                                 (CLIENT, client_columns, "client_features")):
        documented = {e[0] for e in entries}
        if documented != set(cols):
            raise ValueError(f"{label}: undocumented={sorted(set(cols) - documented)} "
                             f"missing={sorted(documented - set(cols))}")
    return {
        "scope": "Berka Phase 1B features. Advisory behavioral signals for Verity; Berka is not a fraud "
                 "dataset and no label is created. Amounts are historical and are never converted to INR.",
        "conceptual_mapping_to_verity": VERITY_MAPPING,
        "features": expand(ACCOUNT, "account_features.parquet") + expand(CLIENT, "client_features.parquet"),
    }