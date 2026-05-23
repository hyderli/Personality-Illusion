"""
Analyze the C0 / C1 / C2 source-manipulation comparison.

Inputs (defaults are correct for the canonical experiment paths):
  C0  baseline disagree:  reasoning_extension/results/sycophancy/
                              deepseek-r1-distill-qwen-7b_bidirectional_sycophancy_items.csv
                          (filtered to persona_name='respond', uses `flipped` column)
  C1  impersonal:         reasoning_extension/results/sycophancy/
                              deepseek-r1-distill-qwen-7b_source_impersonal_sycophancy_items.csv
  C2  user_reasoned:      reasoning_extension/results/sycophancy/
                              deepseek-r1-distill-qwen-7b_source_user_reasoned_sycophancy_items.csv

Outputs:
  - Console: per-condition flip rates with 95% bootstrap CIs, pairwise z-tests,
    3-arm Pearson chi-square.
  - PNG figure: reasoning_extension/results/sycophancy/source_manipulation_flip_rates.png
"""

import argparse
import csv
import math
import random
from pathlib import Path
from collections import defaultdict

DEFAULT_BASE_DIR = "reasoning_extension/results/sycophancy"
DEFAULT_MODEL = "deepseek-r1-distill-qwen-7b"
PERSONA = "respond"


def load_flips_from_csv(path, condition_name, source_filter=None):
    """
    Return list of (dilemma_id, flipped_int) for trials with non-empty `flipped`.

    For C0 (bidirectional CSV) we filter to persona_name=='respond' and use the
    `flipped` column (which is the disagree-condition flip).

    For C1/C2 we only keep rows where source_condition matches `source_filter`.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{condition_name} CSV not found: {path}")

    flips = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("persona_name") != PERSONA:
                continue
            if source_filter is not None:
                if row.get("source_condition") != source_filter:
                    continue
            v = row.get("flipped", "")
            if v in ("True", "False"):
                flips.append((row.get("dilemma_id"), 1 if v == "True" else 0))
    return flips


def bootstrap_ci(values, n_iter=10_000, alpha=0.05, seed=0):
    if not values:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_iter)]
    hi = means[int((1 - alpha / 2) * n_iter)]
    return (lo, hi)


def two_proportion_z(a_flips, a_n, b_flips, b_n):
    """Returns (z, two-sided p-value approximation)."""
    if a_n == 0 or b_n == 0:
        return 0.0, 1.0
    p1 = a_flips / a_n
    p2 = b_flips / b_n
    p_pool = (a_flips + b_flips) / (a_n + b_n)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / a_n + 1 / b_n))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # Two-sided p via normal approximation (no scipy)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def chi_square_3x2(table):
    total = sum(sum(row) for row in table)
    row_sums = [sum(r) for r in table]
    col_sums = [sum(table[i][j] for i in range(len(table))) for j in range(len(table[0]))]
    chi2 = 0.0
    for i, row in enumerate(table):
        for j, v in enumerate(row):
            expected = row_sums[i] * col_sums[j] / total
            if expected > 0:
                chi2 += (v - expected) ** 2 / expected
    # df = (rows-1)*(cols-1) = 2 here (3x2)
    # No scipy available — provide critical-value reference instead of exact p
    return chi2


def chi2_pvalue_df2(chi2):
    """Exact survival function of chi-square with df=2: P(X>chi2) = exp(-chi2/2)."""
    return math.exp(-chi2 / 2.0)


def render_bar_chart(stats, out_path):
    """stats = list[(label, rate, lo, hi, n_flips, n_total)]"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[render_bar_chart] matplotlib not installed; skipping figure.")
        return

    labels = [s[0] for s in stats]
    rates = [s[1] * 100 for s in stats]
    err_lo = [(s[1] - s[2]) * 100 for s in stats]
    err_hi = [(s[3] - s[1]) * 100 for s in stats]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    xs = list(range(len(labels)))
    ax.bar(xs, rates, color=colors[: len(stats)],
           yerr=[err_lo, err_hi], capsize=6, edgecolor="black", linewidth=0.8)

    for x, s in zip(xs, stats):
        ax.text(x, s[1] * 100 + 1.2, f"{s[1]*100:.1f}%\n(n={s[5]})",
                ha="center", fontsize=10)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Disagree-condition flip rate (%)")
    ax.set_title("Sycophancy source-manipulation: DeepSeek-R1-Distill-Qwen-7B (respond persona)")
    ax.set_ylim(0, max(rates) * 1.25 + 5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--c0-csv", default=None,
                    help="Override path to C0 CSV (default: <base>/<model>_bidirectional_sycophancy_items.csv)")
    ap.add_argument("--c1-csv", default=None,
                    help="Override path to C1 CSV (default: <base>/<model>_source_impersonal_sycophancy_items.csv)")
    ap.add_argument("--c2-csv", default=None,
                    help="Override path to C2 CSV (default: <base>/<model>_source_user_reasoned_sycophancy_items.csv)")
    ap.add_argument("--figure", default=None,
                    help="Output PNG path (default: <base>/source_manipulation_flip_rates.png)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    c0_csv = Path(args.c0_csv) if args.c0_csv else base / f"{args.model}_bidirectional_sycophancy_items.csv"
    c1_csv = Path(args.c1_csv) if args.c1_csv else base / f"{args.model}_source_impersonal_sycophancy_items.csv"
    c2_csv = Path(args.c2_csv) if args.c2_csv else base / f"{args.model}_source_user_reasoned_sycophancy_items.csv"
    fig_path = Path(args.figure) if args.figure else base / "source_manipulation_flip_rates.png"

    print("=" * 72)
    print("Source-manipulation analysis")
    print(f"  Model:   {args.model}")
    print(f"  Persona: {PERSONA}")
    print("=" * 72)

    conditions = []  # (name, label, flips_list)

    print(f"\nLoading C0 (user_baseline) from {c0_csv}")
    c0 = load_flips_from_csv(c0_csv, "C0")
    print(f"  {len(c0)} trials")
    conditions.append(("C0\nuser_baseline", "C0_user_baseline", c0))

    if c1_csv.exists():
        print(f"\nLoading C1 (impersonal) from {c1_csv}")
        c1 = load_flips_from_csv(c1_csv, "C1", source_filter="impersonal")
        print(f"  {len(c1)} trials")
        conditions.append(("C1\nimpersonal", "C1_impersonal", c1))
    else:
        print(f"\n[skip] C1 CSV not found: {c1_csv}")

    if c2_csv.exists():
        print(f"\nLoading C2 (user_reasoned) from {c2_csv}")
        c2 = load_flips_from_csv(c2_csv, "C2", source_filter="user_reasoned")
        print(f"  {len(c2)} trials")
        conditions.append(("C2\nuser_reasoned", "C2_user_reasoned", c2))
    else:
        print(f"\n[skip] C2 CSV not found: {c2_csv}")

    # Per-condition summary
    print("\n" + "=" * 72)
    print(f"{'condition':<22} {'n':>5} {'flips':>6} {'rate':>8}    95% CI")
    print("-" * 72)
    stats = []
    for label, key, flips_data in conditions:
        n = len(flips_data)
        if n == 0:
            print(f"{key:<22} {n:>5}  (no data)")
            continue
        flip_count = sum(v for _, v in flips_data)
        rate = flip_count / n
        lo, hi = bootstrap_ci([v for _, v in flips_data])
        print(f"{key:<22} {n:>5} {flip_count:>6} {rate*100:>7.2f}%    "
              f"[{lo*100:5.2f}%, {hi*100:5.2f}%]")
        stats.append((label, rate, lo, hi, flip_count, n))

    # Pairwise z-tests
    if len(stats) >= 2:
        print("\nPairwise two-proportion z-tests:")
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                a_lab = stats[i][0].replace("\n", " ")
                b_lab = stats[j][0].replace("\n", " ")
                z, p = two_proportion_z(stats[i][4], stats[i][5],
                                        stats[j][4], stats[j][5])
                marker = ""
                if p < 0.001: marker = "  ***"
                elif p < 0.01: marker = "  **"
                elif p < 0.05: marker = "  *"
                print(f"  {a_lab:<20} vs {b_lab:<20} "
                      f"z={z:+.3f}  p={p:.4f}{marker}")

    # 3-arm chi-square
    if len(stats) == 3:
        table = [[s[4], s[5] - s[4]] for s in stats]
        chi2 = chi_square_3x2(table)
        p = chi2_pvalue_df2(chi2)
        sig = ""
        if p < 0.001: sig = " ***"
        elif p < 0.01: sig = " **"
        elif p < 0.05: sig = " *"
        print(f"\n3-arm Pearson chi-square(2) = {chi2:.4f}  p = {p:.4f}{sig}")

    # Figure
    if stats:
        render_bar_chart(stats, fig_path)

    print("\n" + "=" * 72)
    print("Interpretation cheat-sheet:")
    print("  C1 << C0  →  flips are user/social-pressure-driven (sycophancy).")
    print("  C1 ≈  C0  →  opinion content matters regardless of agent.")
    print("  C2 >  C0  →  model responds to argument quality (Bayesian updating).")
    print("  C2 ≈  C0  →  model unresponsive to information content (pure compliance).")
    print("=" * 72)


if __name__ == "__main__":
    main()
