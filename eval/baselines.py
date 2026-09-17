"""
Two baselines to compare the LLM agent against, as required by the report:

1. TRIVIAL baseline — the floor. No understanding of the message at all:
   - intent: always the single most common intent in the taxonomy
   - reply: a single fixed template, always the same
   - escalation: always False (never escalate) — a plausible naive default
     for "just auto-handle everything," useful to show what happens without
     any escalation policy at all.

2. SIMPLE baseline — a reasonable non-LLM system a team might actually ship
   in a week:
   - intent: 1-nearest-neighbor via the same TF-IDF space used for retrieval
     (nearest historical message's intent, if intents are back-filled onto
     the corpus — see note below)
   - reply: verbatim reply from the nearest-neighbor historical resolution
     (no generation at all)
   - escalation: keyword rules only (reuses escalate.HARD_ESCALATION_PATTERNS,
     no LLM judgment call, no confidence-based branch)

Both baselines are cheap and fast (no network/API calls), so they always run
even with --offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.escalate import rule_based_check  # noqa: E402


def trivial_baseline(eval_df: pd.DataFrame, taxonomy: dict, fixed_reply: str | None = None) -> pd.DataFrame:
    # Most common intent by cluster_size in the taxonomy
    most_common = max(taxonomy["intents"], key=lambda it: it["cluster_size"])["id"]
    template = fixed_reply or "Thanks for reaching out! We're looking into this and will follow up shortly."
    rows = []
    for _, row in eval_df.iterrows():
        rows.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "intent_id": most_common,
                "drafted_reply": template,
                "escalate": False,
                "escalation_reason": "Trivial baseline never escalates.",
            }
        )
    return pd.DataFrame(rows)


def simple_baseline(eval_df: pd.DataFrame, corpus_df: pd.DataFrame, corpus_intents: list[str] | None = None) -> pd.DataFrame:
    """
    1-NN over TF-IDF space of corpus_df['customer_text']. If corpus_intents is
    given (same length/order as corpus_df, e.g. from a prior LLM labeling pass
    used only to seed this baseline), intent is copied from the nearest
    neighbor; otherwise intent_id is left as "unclassified_nn" since the
    simple baseline has no independent way to name intents.
    """
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    corpus_matrix = vec.fit_transform(corpus_df["customer_text"].astype(str))

    rows = []
    for _, row in eval_df.iterrows():
        message = str(row["customer_text"])
        q = vec.transform([message])
        sims = cosine_similarity(q, corpus_matrix)[0]
        nn_idx = int(np.argmax(sims))
        nn_row = corpus_df.iloc[nn_idx]

        hard = rule_based_check(message, prior_customer_turns=int(row.get("prior_customer_turns", 0)))
        escalate = hard is not None
        reason = hard.reason if hard else "No hard-rule keyword match; simple baseline has no other escalation logic."

        intent_id = corpus_intents[nn_idx] if corpus_intents else "unclassified_nn"

        rows.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "intent_id": intent_id,
                "drafted_reply": nn_row["brand_reply_text"],
                "escalate": escalate,
                "escalation_reason": reason,
                "nn_similarity": float(sims[nn_idx]),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--golden-csv", default=None)
    ap.add_argument("--out-trivial", default=str(ROOT / "data/processed/baseline_trivial.csv"))
    ap.add_argument("--out-simple", default=str(ROOT / "data/processed/baseline_simple.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    with open(args.taxonomy) as f:
        taxonomy = json.load(f)

    if args.golden_csv:
        golden_ids = set(pd.read_csv(args.golden_csv)["customer_tweet_id"].astype(str))
        df["customer_tweet_id"] = df["customer_tweet_id"].astype(str)
        eval_df = df[df["customer_tweet_id"].isin(golden_ids)].reset_index(drop=True)
        corpus_df = df[~df["customer_tweet_id"].isin(golden_ids)]
    else:
        eval_df = df.sample(n=min(args.sample, len(df)), random_state=42)
        corpus_df = df.drop(eval_df.index)
        if corpus_df.empty:
            corpus_df = df

    trivial_out = trivial_baseline(eval_df, taxonomy)
    simple_out = simple_baseline(eval_df, corpus_df)

    trivial_out.to_csv(args.out_trivial, index=False)
    simple_out.to_csv(args.out_simple, index=False)
    print(f"[baselines] trivial -> {args.out_trivial} ({len(trivial_out)} rows)")
    print(f"[baselines] simple  -> {args.out_simple} ({len(simple_out)} rows)")
