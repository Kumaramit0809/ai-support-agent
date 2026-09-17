"""
End-to-end agent: message -> intent -> grounded reply -> escalation decision.

Usage:
    python3 -m src.pipeline --brand AmazonHelp --sample 500

Writes data/processed/agent_outputs.csv with one row per processed message:
  customer_tweet_id, customer_text, intent_id, intent_confidence,
  drafted_reply, escalate, escalation_reason, escalation_source
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load_data import load_brand_pairs  # noqa: E402
from src.retrieve import ResolutionRetriever  # noqa: E402
from src.classify import classify_message  # noqa: E402
from src.draft_reply import draft_reply  # noqa: E402
from src.escalate import decide_escalation  # noqa: E402


def run_pipeline(
    brand: str,
    data_path: str | None,
    taxonomy_path: str,
    sample: int,
    model: str,
    seed: int,
    out_path: str,
    holdout_fraction: float = 0.2,
    english_only: bool = False,
) -> pd.DataFrame:
    pairs_df, is_real = load_brand_pairs(brand, data_path, english_only=english_only)
    if pairs_df.empty:
        raise ValueError(f"No resolution pairs found for brand={brand!r}. Check the data / brand name.")

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    intent_by_id = {it["id"]: it for it in taxonomy["intents"]}

    pairs_df = pairs_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_holdout = max(1, int(len(pairs_df) * holdout_fraction))
    eval_df = pairs_df.iloc[:n_holdout].copy()
    corpus_df = pairs_df.iloc[n_holdout:].copy()
    if corpus_df.empty:
        corpus_df = pairs_df.copy()

    if sample and sample < len(eval_df):
        eval_df = eval_df.sample(n=sample, random_state=seed)

    retriever = ResolutionRetriever(corpus_df)

    from src.llm_client import get_client

    if not os.environ.get("GROQ_API_KEY"):
        print("[pipeline] WARNING: GROQ_API_KEY not set — cannot run the LLM agent. "
              "Use eval/baselines.py for a non-LLM run, or set the key "
              "(get one free at https://aistudio.google.com/app/apikey).", file=sys.stderr)
        raise SystemExit(1)
    client = get_client()

    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="agent"):
        message = str(row["customer_text"])
        try:
            cls = classify_message(message, taxonomy, brand, model, client=client)
            intent_meta = intent_by_id.get(cls.intent_id, {"label": cls.intent_id, "description": ""})

            examples = retriever.retrieve(message, k=3, exclude_thread_id=row.get("thread_id"))
            drafted = draft_reply(
                message, brand, intent_meta["label"], intent_meta.get("description", ""), examples, model, client=client
            )

            decision = decide_escalation(
                message, brand, intent_meta["label"], cls.confidence,
                int(row.get("prior_customer_turns", 0)), model, client=client,
            )

            rows.append(
                {
                    "customer_tweet_id": row["customer_tweet_id"],
                    "customer_text": message,
                    "gold_brand_reply": row["brand_reply_text"],
                    "intent_id": cls.intent_id,
                    "intent_confidence": cls.confidence,
                    "intent_rationale": cls.rationale,
                    "drafted_reply": drafted.reply_text,
                    "grounding_examples": " ||| ".join(drafted.grounding_used),
                    "escalate": decision.escalate,
                    "escalation_reason": decision.reason,
                    "escalation_source": decision.source,
                }
            )
        except Exception as e:
            print(f"[pipeline] ERROR on tweet {row['customer_tweet_id']}: {e}", file=sys.stderr)
            rows.append(
                {
                    "customer_tweet_id": row["customer_tweet_id"],
                    "customer_text": message,
                    "gold_brand_reply": row["brand_reply_text"],
                    "intent_id": "ERROR",
                    "intent_confidence": 0.0,
                    "intent_rationale": str(e),
                    "drafted_reply": "",
                    "grounding_examples": "",
                    "escalate": True,
                    "escalation_reason": f"Pipeline error, defaulting to escalate: {e}",
                                       "escalation_source": "error",
                }
            )

        pd.DataFrame(rows).to_csv(out_path, index=False)

    out_df = pd.DataFrame(rows)
    print(f"[pipeline] wrote {len(out_df)} rows -> {out_path}  (data={'REAL' if is_real else 'FIXTURE'})")
    return out_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--taxonomy", default=str(ROOT / "data/processed/taxonomy.json"))
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data/processed/agent_outputs.csv"))
    ap.add_argument("--english-only", action="store_true", help="Drop non-English pairs (see src/load_data.py)")
    args = ap.parse_args()

    run_pipeline(args.brand, args.data_path, args.taxonomy, args.sample, args.model, args.seed, args.out, english_only=args.english_only)
