# ml/preprocessing/berka_phase1b_plots.py
# Figures 06-13. Matplotlib only; every figure has a title and labelled axes.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORR_COLUMNS = ["transaction_count", "active_days", "history_days", "transactions_per_active_day",
                "median_transaction_amount", "p95_transaction_amount", "std_transaction_amount",
                "total_transaction_amount", "mean_balance", "balance_std", "minimum_balance",
                "overdraft_ratio", "share_above_prior_p95", "amount_to_typical_ratio_max",
                "district_average_salary", "district_unemployment_rate_1996"]


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def make_all(trans, af, ratios, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    amt = trans["amount"].astype("float64")
    pos = trans[amt > 0].copy()
    pos["amount"] = pos["amount"].astype("float64")
    cred, deb = pos[pos["is_credit"]]["amount"], pos[pos["is_debit"]]["amount"]

    # 06 amount distribution: how skewed are amounts, and do credits differ from debits?
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, hi, title in ((ax[0], None, "All positive amounts (log count axis)"),
                         (ax[1], pos["amount"].quantile(0.99), "Amounts up to the 99th percentile (linear)")):
        rng = (0, hi) if hi else (0, pos["amount"].max())
        a.hist(cred, bins=60, range=rng, alpha=0.6, label="credit")
        a.hist(deb, bins=60, range=rng, alpha=0.6, label="debit")
        a.set_title(title)
        a.set_xlabel("transaction amount (historical currency units)")
        a.set_ylabel("transactions")
        a.legend()
    ax[0].set_yscale("log")
    paths.append(_save(fig, out_dir, "06_transaction_amount_distribution.png"))

    # 07 log amount: the spike near 10 is a block of small recurring postings
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(np.log10(pos["amount"].min()), np.log10(pos["amount"].max()), 80)
    ax.hist(np.log10(cred), bins=bins, histtype="step", lw=2, label="credit")
    ax.hist(np.log10(deb), bins=bins, histtype="step", lw=2, label="debit")
    # annotate the spike with the most frequent amount found in the data, not an assumed value
    top_amount = pos["amount"].value_counts().idxmax()
    top_n = int((pos["amount"] == top_amount).sum())
    top_kind = pos.loc[pos["amount"] == top_amount, "k_symbol_normalized"].astype(str).value_counts().idxmax()
    ax.axvline(np.log10(top_amount), color="grey", ls=":", lw=1)
    ax.annotate(f"{top_n:,} rows with amount exactly {top_amount:g} (mostly k_symbol {top_kind})",
                xy=(np.log10(top_amount), ax.get_ylim()[1] * 0.95), xytext=(2.0, ax.get_ylim()[1] * 0.9),
                arrowprops={"arrowstyle": "->"})
    ax.set_title("Transaction amount on a log10 scale (zero amounts excluded)")
    ax.set_xlabel("log10(amount)")
    ax.set_ylabel("transactions")
    ax.legend()
    paths.append(_save(fig, out_dir, "07_transaction_amount_log_distribution.png"))

    # 08 balance: how often and how deep is the balance negative?
    bal = trans["balance"].astype("float64")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(bal.clip(bal.quantile(0.001), bal.quantile(0.999)), bins=80)
    ax[0].axvline(0, color="red", lw=1)
    ax[0].set_title(f"Balance after each transaction ({int((bal < 0).sum()):,} negative; 0.1%-99.9% shown)")
    ax[0].set_xlabel("balance")
    ax[0].set_ylabel("transactions")
    ax[1].hist(af["mean_balance"].dropna(), bins=50)
    ax[1].axvline(af["mean_balance"].median(), color="red", lw=1, label="median")
    ax[1].set_title("Mean balance per account")
    ax[1].set_xlabel("account mean balance")
    ax[1].set_ylabel("accounts")
    ax[1].legend()
    paths.append(_save(fig, out_dir, "08_balance_distribution.png"))

    # 09 transactions per account: how much history does each account have?
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    n = af["transaction_count"]
    ax[0].hist(n, bins=50)
    ax[0].axvline(n.median(), color="red", lw=1, label=f"median {int(n.median())}")
    ax[0].axvline(10, color="black", ls=":", lw=1, label="10 = min history for ratios")
    ax[0].set_title("Transactions per account")
    ax[0].set_xlabel("transactions")
    ax[0].set_ylabel("accounts")
    ax[0].legend()
    s = np.sort(n.to_numpy())
    ax[1].plot(s, np.arange(1, len(s) + 1) / len(s))
    ax[1].set_title("Cumulative share of accounts")
    ax[1].set_xlabel("transactions per account")
    ax[1].set_ylabel("share of accounts")
    paths.append(_save(fig, out_dir, "09_transactions_per_account.png"))

    # 10 overdraft behavior: how common, how frequent, how deep?
    od = af[af["has_overdraft"]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    counts = af["has_overdraft"].value_counts().reindex([False, True]).fillna(0)
    ax[0].bar(["no overdraft", "overdraft"], counts.values)
    ax[0].set_title("Accounts with any negative balance")
    ax[0].set_ylabel("accounts")
    ax[1].hist(od["overdraft_ratio"], bins=40)
    ax[1].set_title(f"Overdraft ratio, {len(od):,} accounts with overdraft")
    ax[1].set_xlabel("share of transactions with balance < 0")
    ax[1].set_ylabel("accounts")
    ax[2].hist(np.log10(od["overdraft_amount"].clip(lower=1)), bins=40)
    ax[2].set_title("Deepest overdraft per account")
    ax[2].set_xlabel("log10(deepest overdraft)")
    ax[2].set_ylabel("accounts")
    paths.append(_save(fig, out_dir, "10_overdraft_behavior.png"))

    # 11 amount-to-typical: does the prior-day baseline behave as intended?
    r = ratios.dropna(subset=["amount_to_typical_ratio"])
    share = float((r["amount_to_p95_ratio"] > 1).mean())
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].hist(np.log10(r["amount_to_typical_ratio"]), bins=80)
    ax[0].axvline(0, color="red", lw=1)
    ax[0].set_title("amount / prior median (each transaction vs earlier days)")
    ax[0].set_xlabel("log10(ratio), 0 = equal to typical")
    ax[0].set_ylabel("transactions")
    ax[1].hist(np.log10(r["amount_to_p95_ratio"]), bins=80)
    ax[1].axvline(0, color="red", lw=1)
    ax[1].set_title(f"amount / prior p95 ({share:.1%} of transactions above 1)")
    ax[1].set_xlabel("log10(ratio), 0 = equal to prior p95")
    ax[1].set_ylabel("transactions")
    ax[2].hist(af["share_above_prior_p95"].dropna(), bins=40)
    ax[2].set_title("Per-account share above own prior p95")
    ax[2].set_xlabel("share of the account's transactions")
    ax[2].set_ylabel("accounts")
    paths.append(_save(fig, out_dir, "11_amount_to_typical_ratio.png"))

    # 12 account behavior features: shape of the main account-level features
    specs = [("transactions_per_active_day", False, "transactions per active day"),
             ("history_days", False, "history (days)"),
             ("median_transaction_amount", True, "log10 median amount"),
             ("p95_transaction_amount", True, "log10 p95 amount"),
             ("overdraft_ratio", False, "overdraft ratio (log count axis)"),
             ("balance_std", False, "balance standard deviation")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for a, (col, log, label) in zip(ax.ravel(), specs):
        v = af[col].dropna()
        a.hist(np.log10(v.clip(lower=1e-9)) if log else v, bins=50)
        if col == "overdraft_ratio":
            a.set_yscale("log")
        a.set_title(col)
        a.set_xlabel(label)
        a.set_ylabel("accounts")
    paths.append(_save(fig, out_dir, "12_account_behavior_features.png"))

    # 13 correlations: which features carry the same information?
    c = af[CORR_COLUMNS].astype("float64").corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.columns, fontsize=8)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Account feature correlations (Spearman; alias and duplicate columns excluded)")
    paths.append(_save(fig, out_dir, "13_feature_correlations.png"))
    return paths