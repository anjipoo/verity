# ml/preprocessing/berka_plots.py
# Exploratory figures of the RAW imported Berka tables. Only fields that exist are
# plotted. Matplotlib only, one PNG per table group.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _label(v):
    # make the three "empty" encodings visible instead of hiding them
    if pd.isna(v):
        return "<missing>"
    if v == "":
        return '<"" empty>'
    if v == " ":
        return '<" " blank>'
    return str(v)


def _bar(ax, series, title, log=False, top=None, horizontal=False):
    counts = series.astype(object).map(_label).value_counts()
    if top:
        counts = counts.head(top)
    if horizontal:
        ax.barh(counts.index[::-1], counts.values[::-1])
    else:
        ax.bar(counts.index, counts.values)
        ax.tick_params(axis="x", rotation=40)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    if log:
        ax.set_yscale("log") if not horizontal else ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("rows" if not horizontal else "")


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def make_all(tables, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    acc, cli, disp, dist = tables["account"], tables["client"], tables["disp"], tables["district"]
    trans, loan, card = tables["trans"], tables["loan"], tables["card"]
    names = dist.set_index("district_id").district_name
    paths = []

    # 1. accounts
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    _bar(ax[0], acc.frequency, "Accounts by frequency (raw labels)")
    top = acc.district_id.map(names).value_counts().head(10)
    ax[1].barh(top.index[::-1], top.values[::-1])
    ax[1].set_title("Accounts: top 10 districts", fontsize=10)
    yr = acc.date.dt.year.value_counts().sort_index()
    ax[2].bar(yr.index.astype(str), yr.values)
    ax[2].set_title("Accounts opened per year", fontsize=10)
    paths.append(_save(fig, out_dir, "01_accounts.png"))

    # 2. clients (birth_number dropped, so district and disposition role are what remains)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    per_d = cli.district_id.value_counts()
    ax[0].hist(per_d.values, bins=20)
    ax[0].set_title("Clients per district (distribution over 77 districts)", fontsize=10)
    ax[0].set_xlabel("clients in district")
    _bar(ax[1], disp.type, "Disposition type")
    d_per_acc = disp.account_id.value_counts().value_counts().sort_index()
    ax[2].bar(d_per_acc.index.astype(str), d_per_acc.values)
    ax[2].set_title("Clients linked per account", fontsize=10)
    paths.append(_save(fig, out_dir, "02_clients.png"))

    # 3. transactions
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    amt = trans.amount.astype("float64")
    ax[0, 0].hist(amt, bins=60)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("Amount, raw (log count)", fontsize=10)
    ax[0, 1].hist(np.log10(amt[amt > 0]), bins=60)
    ax[0, 1].set_title(f"log10(amount), amount>0 ({int((amt <= 0).sum())} rows with amount<=0 excluded)", fontsize=9)
    _bar(ax[0, 2], trans.type, "type (raw values)")
    _bar(ax[0, 3], trans.operation, "operation")
    _bar(ax[1, 0], trans.k_symbol, "k_symbol (log scale)", log=True)
    m = trans.groupby(trans.date.dt.to_period("M")).size()
    ax[1, 1].plot(m.index.to_timestamp(), m.values)
    ax[1, 1].set_title("Transactions per month", fontsize=10)
    ax[1, 2].hist(trans.balance.astype("float64"), bins=60)
    ax[1, 2].axvline(0, color="red", lw=1)
    ax[1, 2].set_title("Balance after transaction (red = 0)", fontsize=10)
    ax[1, 3].hist(trans.account_id.value_counts().values, bins=40)
    ax[1, 3].set_title("Transactions per account", fontsize=10)
    paths.append(_save(fig, out_dir, "03_transactions.png"))

    # 4. loans
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    _bar(ax[0], loan.status, "Loan status")
    ax[1].hist(loan.amount.astype("float64"), bins=40)
    ax[1].set_title("Loan amount", fontsize=10)
    dur = loan.duration.value_counts().sort_index()
    ax[2].bar(dur.index.astype(str), dur.values)
    ax[2].set_title("Loan duration (months, as stored)", fontsize=10)
    paths.append(_save(fig, out_dir, "04_loans.png"))

    # 5. relationships: how many children per parent
    cards_per_acc = card.merge(disp[["disp_id", "account_id"]], on="disp_id").account_id.value_counts()
    cards_per_acc = cards_per_acc.reindex(acc.account_id, fill_value=0)
    loans_per_acc = loan.account_id.value_counts().reindex(acc.account_id, fill_value=0)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, s, title in [
        (ax[0], disp.account_id.value_counts().reindex(acc.account_id, fill_value=0), "clients -> account"),
        (ax[2], loans_per_acc, "loans per account"),
        (ax[3], cards_per_acc, "cards per account (via disp)"),
    ]:
        vc = s.value_counts().sort_index()
        a.bar(vc.index.astype(str), vc.values)
        a.set_title(title, fontsize=10)
        a.set_xlabel("children per account")
    ax[1].hist(trans.account_id.value_counts().values, bins=40)
    ax[1].set_title("transactions per account", fontsize=10)
    paths.append(_save(fig, out_dir, "05_relationships.png"))
    return paths