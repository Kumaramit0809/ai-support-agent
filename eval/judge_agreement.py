"""
Validates the LLM judge (eval/llm_judge.py) against real human ratings, per the
required deliverable: "evidence of how well your judge agrees with a human."

Workflow:
  1. Take a subset (e.g. 30-50) of the golden set.
  2. A human rates the same 4-dimension rubric by hand for those rows
     (add columns human_groundedness, human_helpfulness, human_tone_fit,
     human_correctness to a CSV — see human_ratings_template.csv).
  3. Run the LLM judge on the same rows (eval/llm_judge.py).
  4. This script reports Spearman correlation and quadratic-weighted kappa
     per dimension, plus overall — both are standard for ordinal 1-5 rating
     agreement (plain accuracy is too strict for a 5-point scale; correlation
     and weighted kappa reward "close" disagreements less harshly than "far"
     ones, which is the right notion of agreement here).
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

DIMENSIONS = ["groundedness", "helpfulness", "tone_fit", "correctness"]


def compute_agreement(human_df: pd.DataFrame, judge_df: pd.DataFrame) -> dict:
    human_df = human_df.copy()
    judge_df = judge_df.copy()
    human_df["customer_tweet_id"] = human_df["customer_tweet_id"].astype(str)
    judge_df["customer_tweet_id"] = judge_df["customer_tweet_id"].astype(str)
    merged = human_df.merge(judge_df, on="customer_tweet_id", suffixes=("_human", "_judge"))
    if merged.empty:
        raise ValueError("No overlapping customer_tweet_id between human ratings and judge scores.")

    results = {}
    for dim in DIMENSIONS:
        h_col, j_col = f"human_{dim}", dim
        if h_col not in merged.columns or j_col not in merged.columns:
            continue
        h = merged[h_col].astype(float)
        j = merged[j_col].astype(float)
        rho, p = spearmanr(h, j)
        kappa = cohen_kappa_score(h.round().astype(int), j.round().astype(int), weights="quadratic")
        results[dim] = {"spearman_rho": rho, "spearman_p": p, "quadratic_weighted_kappa": kappa, "n": len(merged)}

    # overall (mean of the four dims)
    merged["human_overall"] = merged[[f"human_{d}" for d in DIMENSIONS if f"human_{d}" in merged.columns]].mean(axis=1)
    merged["judge_overall"] = merged[[d for d in DIMENSIONS if d in merged.columns]].mean(axis=1)
    rho, p = spearmanr(merged["human_overall"], merged["judge_overall"])
    results["overall"] = {"spearman_rho": rho, "spearman_p": p, "n": len(merged)}
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--human-csv", required=True, help="CSV with customer_tweet_id + human_<dimension> columns")
    ap.add_argument("--judge-csv", required=True, help="Output of eval/llm_judge.py")
    args = ap.parse_args()

    human_df = pd.read_csv(args.human_csv)
    judge_df = pd.read_csv(args.judge_csv)
    results = compute_agreement(human_df, judge_df)
    for dim, stats in results.items():
        if "quadratic_weighted_kappa" in stats:
            print(f"{dim:14s} spearman_rho={stats['spearman_rho']:.3f} (p={stats['spearman_p']:.3f})  "
                  f"qw_kappa={stats['quadratic_weighted_kappa']:.3f}  n={stats['n']}")
        else:
            print(f"{dim:14s} spearman_rho={stats['spearman_rho']:.3f} (p={stats['spearman_p']:.3f})  n={stats['n']}")
