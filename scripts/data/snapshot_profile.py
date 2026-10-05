# scripts/data/snapshot_profile.py
# Dataset-agnostic import + snapshot + profiling + raw-data visualization.
# Reads a local CSV/Parquet file. Does not clean, modify, or join anything.

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to disk, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sha256_file(path):
    # chunked read so large files do not need to fit in memory
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, sep):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=sep, low_memory=False)


def safe_nunique(s):
    # unhashable values (lists/dicts) raise TypeError, so fall back to strings
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique(dropna=True))


def example_values(s, k=3):
    vals = s.dropna().head(1000).astype(str).unique()[:k]
    return " | ".join(v[:30] for v in vals)


def save_fig(fig, out_dir, name):
    fig.savefig(out_dir / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {out_dir / name}")


def build_overview(args, path, df):
    overview = {
        "dataset_name": args.name,
        "source": args.source,
        "url": args.url,
        "license": args.license,
        "download_date": args.download_date
        or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
        "target_column": args.target,
    }
    if args.target:
        counts = df[args.target].value_counts(dropna=False)
        overview["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        overview["minority_class_pct"] = round(float(counts.min() / counts.sum() * 100), 4)
    return overview


def build_schema(df):
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 3),
                "unique": safe_nunique(s),
                "examples": example_values(s),
            }
        )
    return pd.DataFrame(rows)


def plot_class_distribution(df, target, out_dir):
    counts = df[target].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_yscale("log")  # log scale keeps the minority class visible
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    pct = counts.min() / counts.sum() * 100
    ax.set_title(f"Class distribution (minority = {pct:.3f}%)")
    ax.set_xlabel(target)
    ax.set_ylabel("count (log scale)")
    save_fig(fig, out_dir, "01_class_distribution.png")


def plot_amount(df, amount_col, target, out_dir):
    amt = pd.to_numeric(df[amount_col], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(amt.dropna(), bins=60)
    axes[0].set_title(f"{amount_col}: raw")
    axes[0].set_yscale("log")
    if (amt.dropna() >= 0).all():
        if target:
            for cls, idx in df.groupby(target).groups.items():
                axes[1].hist(np.log1p(amt.loc[idx].dropna()), bins=60,
                             alpha=0.5, density=True, label=str(cls))
            axes[1].legend(title=target)
        else:
            axes[1].hist(np.log1p(amt.dropna()), bins=60)
        axes[1].set_title(f"log1p({amount_col})" + (" by class" if target else ""))
    else:
        axes[1].text(0.5, 0.5, "negative values present:\nlog1p skipped",
                     ha="center", va="center")
    save_fig(fig, out_dir, "02_amount_distribution.png")


def plot_missing(df, out_dir):
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False).head(30)
    if miss.empty:
        print("  no missing values: missing-value chart skipped")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(miss))))
    ax.barh(miss.index[::-1], miss.values[::-1])
    ax.set_xlabel("% missing")
    ax.set_title("Columns with missing values (top 30)")
    save_fig(fig, out_dir, "03_missing_values.png")


def plot_numeric(df, skip, max_hist, out_dir):
    cols = [c for c in df.select_dtypes("number").columns
            if c not in skip and safe_nunique(df[c]) > 1][:max_hist]
    if not cols:
        return
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        ax.hist(df[c].dropna(), bins=40)
        ax.set_title(c, fontsize=9)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numeric distributions (first columns, non-constant)")
    save_fig(fig, out_dir, "04_numeric_distributions.png")


def plot_categorical(df, skip, max_cat, out_dir):
    cols = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
            if c not in skip and 1 < safe_nunique(df[c])][:max_cat]
    if not cols:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        top = df[c].astype(str).value_counts().head(10)
        ax.barh(top.index[::-1], top.values[::-1])
        ax.set_title(f"{c} (top 10 of {safe_nunique(df[c])})", fontsize=9)
    for ax in axes[len(cols):]:
        ax.axis("off")
    save_fig(fig, out_dir, "05_categorical_frequencies.png")


def main():
    p = argparse.ArgumentParser(description="Snapshot and profile one raw dataset file.")
    p.add_argument("--path", required=True, help="CSV or Parquet file")
    p.add_argument("--name", required=True)
    p.add_argument("--source", default="UNVERIFIED")
    p.add_argument("--url", default="UNVERIFIED")
    p.add_argument("--license", default="UNVERIFIED")
    p.add_argument("--download-date", default=None)
    p.add_argument("--target", default=None, help="label column, if one exists")
    p.add_argument("--amount-col", default=None)
    p.add_argument("--id-col", default=None, help="column expected to be unique")
    p.add_argument("--sep", default=",")
    p.add_argument("--out", default="data/metadata/snapshots")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hist", type=int, default=12)
    p.add_argument("--max-cat", type=int, default=6)
    args = p.parse_args()

    path = Path(args.path)
    df = load(path, args.sep)
    for col in (args.target, args.amount_col, args.id_col):
        if col and col not in df.columns:
            raise SystemExit(f"Column '{col}' not found. Available: {list(df.columns)}")

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    section("1. DATASET OVERVIEW")
    overview = build_overview(args, path, df)
    for k, v in overview.items():
        print(f"{k:>22}: {v}")
    (out_dir / "snapshot.json").write_text(json.dumps(overview, indent=2, default=str))

    section("2. SCHEMA")
    schema = build_schema(df)
    print(schema.to_string(index=False))
    schema.to_csv(out_dir / "schema.csv", index=False)

    section("3. SAMPLE ROWS")
    print("--- first 5 ---")
    print(df.head())
    print("--- last 5 ---")
    print(df.tail())
    print("--- 5 random (seed fixed) ---")
    sample = df.sample(n=min(5, len(df)), random_state=args.seed)
    print(sample)
    pd.concat([df.head(), df.tail(), sample]).to_csv(out_dir / "sample_rows.csv", index=False)

    section("4. STATISTICS")
    num = df.select_dtypes("number")
    if not num.empty:
        desc = num.describe().T
        desc["median"] = num.median()
        print(desc)
        desc.to_csv(out_dir / "numeric_stats.csv")
    cat = df.select_dtypes(include=["object", "category", "bool"])
    if not cat.empty:
        card = pd.Series({c: safe_nunique(cat[c]) for c in cat.columns}, name="cardinality")
        print("\nCategorical cardinality:")
        print(card.sort_values(ascending=False))
        card.to_csv(out_dir / "categorical_cardinality.csv")

    section("5. DUPLICATES AND IDENTIFIERS")
    try:
        dup = int(df.duplicated().sum())
    except TypeError:
        dup = int(df.astype(str).duplicated().sum())
    print(f"fully duplicated rows: {dup:,}")
    if args.id_col:
        n_unique = safe_nunique(df[args.id_col])
        print(f"{args.id_col}: {n_unique:,} unique of {len(df):,} rows "
              f"(unique = {n_unique == len(df)})")

    section("6. RAW-DATA VISUALIZATIONS")
    skip = {c for c in (args.target, args.id_col) if c}
    if args.target:
        plot_class_distribution(df, args.target, out_dir)
    if args.amount_col:
        plot_amount(df, args.amount_col, args.target, out_dir)
    plot_missing(df, out_dir)
    plot_numeric(df, skip, args.max_hist, out_dir)
    plot_categorical(df, skip, args.max_cat, out_dir)

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()