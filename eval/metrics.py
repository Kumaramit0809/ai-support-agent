"""
Automated metrics comparing agent (or baseline) outputs against the hand-labeled
golden set:
  - Intent classification: accuracy, macro-F1, per-intent precision/recall
  - Escalation decision: precision, recall, F1 (positive class = "should escalate")
    — deliberately reported separately from accuracy, because in this task a
    false negative (should have escalated, didn't) is much more costly than a
    false positive, and accuracy alone would hide that asymmetry.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def merge_golden(golden_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    golden_df = golden_df.copy()
    pred_df = pred_df.copy()
    golden_df["customer_tweet_id"] = golden_df["customer_tweet_id"].astype(str)
    pred_df["customer_tweet_id"] = pred_df["customer_tweet_id"].astype(str)
    merged = golden_df.merge(pred_df, on="customer_tweet_id", how="inner", suffixes=("", "_pred"))
    if merged.empty:
        raise ValueError(
            "No overlap between golden set and predictions on customer_tweet_id. "
            "Make sure the golden set was sampled from the same pairs_csv / same run."
        )
    return merged


def intent_metrics(merged: pd.DataFrame) -> dict:
    y_true = merged["gold_intent_id"].astype(str)
    y_pred = merged["intent_id"].astype(str)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "n": len(merged),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def escalation_metrics(merged: pd.DataFrame) -> dict:
    y_true = merged["gold_escalate"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    y_pred = merged["escalate"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n": len(merged),
        "false_negatives": int(((y_true) & (~y_pred)).sum()),  # should have escalated, didn't -- the costly kind
        "false_positives": int(((~y_true) & (y_pred)).sum()),
    }


def summarize(golden_csv: str, pred_csv: str, label: str = "system") -> dict:
    golden_df = pd.read_csv(golden_csv, dtype=str)
    pred_df = pd.read_csv(pred_csv, dtype=str)
    merged = merge_golden(golden_df, pred_df)
    return {
        "label": label,
        "intent": intent_metrics(merged),
        "escalation": escalation_metrics(merged),
    }


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-csv", required=True)
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--label", default="system")
    args = ap.parse_args()

    result = summarize(args.golden_csv, args.pred_csv, args.label)
    print(f"=== {result['label']} ===")
    print(f"Intent accuracy: {result['intent']['accuracy']:.3f}  macro-F1: {result['intent']['macro_f1']:.3f}  (n={result['intent']['n']})")
    print(result["intent"]["classification_report"])
    esc = result["escalation"]
    print(f"Escalation precision={esc['precision']:.3f} recall={esc['recall']:.3f} f1={esc['f1']:.3f} "
          f"(FN={esc['false_negatives']}, FP={esc['false_positives']}, n={esc['n']})")
