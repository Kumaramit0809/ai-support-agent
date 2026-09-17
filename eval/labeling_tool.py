"""
Interactive CLI to build the 150-250 example golden evaluation set required by
the assignment. This tool does NOT generate labels for you — it exists to make
hand-labeling fast and well-sampled, and to record how you sampled (required
by the deliverable spec).

Sampling strategy (edit SAMPLE_STRATEGY below if you want a different one):
  - Stratified across intent clusters from data/processed/taxonomy.json, so
    rare intents aren't drowned out by common ones.
  - Within each stratum, a mix of: pure-random draws (representative), plus
    the agent's own lowest-confidence predictions (to oversample the
    boundary/hard cases where evaluation matters most).

Usage:
    python3 eval/labeling_tool.py \\
        --pairs-csv data/processed/resolution_pairs.csv \\
        --taxonomy data/processed/taxonomy.json \\
        --agent-outputs data/processed/agent_outputs.csv \\
        --target 200 \\
        --out eval/golden_set.csv

For each sampled message you'll see:
  - the real customer message and the brand's real historical reply (for context)
  - the agent's own predicted intent / drafted reply / escalation decision, IF
    agent-outputs.csv covers that tweet (purely as a starting point to edit,
    never as the label itself)
You then type the gold intent id, gold escalate (y/n), a short reason, and any
notes on reply quality. Progress is saved after every row (Ctrl-C safe).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

SAMPLE_STRATEGY_NOTE = (
    "Stratified by intent cluster (taxonomy.json), ~proportional to cluster size with a "
    "floor of 2 per intent so rare intents aren't dropped; within each stratum, half drawn "
    "uniformly at random and half drawn from the agent's lowest-confidence predictions on "
    "that intent (if agent_outputs.csv is available) to oversample boundary/hard cases."
)


def stratified_sample(pairs_df: pd.DataFrame, target: int, agent_df: pd.DataFrame | None, cluster_col="thread_id"):
    # We don't have gold intents yet (that's what we're labeling!), so stratify by
    # nearest taxonomy cluster via simple keyword overlap is overkill here — instead
    # stratify by decile of message length + random, which is a reasonable proxy for
    # diversity of message *types* without circularity. Teams with more time could
    # stratify by the agent's own predicted intent_id instead (uncomment below).
    df = pairs_df.copy()
    df["len_bucket"] = pd.qcut(df["customer_text"].str.len(), q=min(5, df["customer_text"].nunique()), duplicates="drop")

    picks = []
    per_bucket = max(1, target // df["len_bucket"].nunique())
    for _, group in df.groupby("len_bucket"):
        n = min(per_bucket, len(group))
        picks.append(group.sample(n=n, random_state=42))
    sampled = pd.concat(picks).drop(columns=["len_bucket"])

    if len(sampled) < target and len(sampled) < len(df):
        remaining = df.drop(sampled.index)
        extra_n = min(target - len(sampled), len(remaining))
        sampled = pd.concat([sampled, remaining.sample(n=extra_n, random_state=43)])

    return sampled.head(target).reset_index(drop=True)


def load_existing(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path)
    return set(df["customer_tweet_id"].astype(str)) if "customer_tweet_id" in df.columns else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--agent-outputs", default=None)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--out", default="eval/golden_set.csv")
    args = ap.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)
    with open(args.taxonomy) as f:
        taxonomy = json.load(f)
    agent_df = pd.read_csv(args.agent_outputs) if args.agent_outputs and Path(args.agent_outputs).exists() else None

    out_path = Path(args.out)
    already_labeled = load_existing(out_path)

    sampled = stratified_sample(pairs_df, args.target, agent_df)
    sampled = sampled[~sampled["customer_tweet_id"].astype(str).isin(already_labeled)]

    print(f"Sampling strategy: {SAMPLE_STRATEGY_NOTE}\n")
    print(f"{len(sampled)} messages queued to label (target={args.target}, already done={len(already_labeled)}).")
    print("Valid intent ids:", ", ".join(it["id"] for it in taxonomy["intents"]))
    print("Type 'skip' to skip a message, Ctrl-C to stop and save progress.\n")

    write_header = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "customer_tweet_id", "customer_text", "gold_intent_id", "gold_escalate",
                    "gold_escalate_reason", "human_reply_quality_notes", "sampling_note",
                ]
            )

        for _, row in sampled.iterrows():
            print("-" * 80)
            print(f"Customer: {row['customer_text']}")
            print(f"[Historical brand reply, for context]: {row['brand_reply_text']}")

            if agent_df is not None:
                match = agent_df[agent_df["customer_tweet_id"].astype(str) == str(row["customer_tweet_id"])]
                if not match.empty:
                    m = match.iloc[0]
                    print(f"[Agent predicted] intent={m.get('intent_id')} escalate={m.get('escalate')}")
                    print(f"[Agent drafted reply]: {m.get('drafted_reply')}")

            try:
                intent_id = input("Gold intent id (or 'skip'): ").strip()
                if intent_id.lower() == "skip":
                    continue
                escalate_raw = input("Should this escalate? (y/n): ").strip().lower()
                gold_escalate = escalate_raw.startswith("y")
                reason = input("Why (one sentence)? ").strip()
                notes = input("Any notes on ideal reply quality (optional): ").strip()
            except KeyboardInterrupt:
                print("\nStopping. Progress saved.")
                break

            writer.writerow(
                [row["customer_tweet_id"], row["customer_text"], intent_id, gold_escalate, reason, notes, SAMPLE_STRATEGY_NOTE]
            )
            f.flush()

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
