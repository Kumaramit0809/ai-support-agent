"""Run the agent on exactly the golden-set message IDs, so metrics.py can score them."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load_data import load_brand_pairs
from src.retrieve import ResolutionRetriever
from src.classify import classify_message
from src.draft_reply import draft_reply
from src.escalate import decide_escalation
from src.llm_client import get_client


def main():
    import argparse, json
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--golden-csv", default="eval/golden_set.csv")
    ap.add_argument("--taxonomy", default="data/processed/taxonomy.json")
    ap.add_argument("--model", default="groq/compound-mini")
    ap.add_argument("--out", default="data/processed/agent_outputs.csv")
    ap.add_argument("--english-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    golden_df = pd.read_csv(args.golden_csv)
    golden_ids = set(golden_df["customer_tweet_id"].astype(str))
    if args.limit:
        golden_ids = set(list(golden_ids)[:args.limit])

    pairs_df, is_real = load_brand_pairs(args.brand, None, english_only=args.english_only)
    pairs_df["customer_tweet_id"] = pairs_df["customer_tweet_id"].astype(str)

    target_df = pairs_df[pairs_df["customer_tweet_id"].isin(golden_ids)].reset_index(drop=True)
    corpus_df = pairs_df[~pairs_df["customer_tweet_id"].isin(golden_ids)]

    print(f"[eval_on_golden] {len(target_df)}/{len(golden_ids)} golden IDs found")
    if target_df.empty:
        raise SystemExit("No overlap -- check --brand/--english-only match.")

    with open(args.taxonomy) as f:
        taxonomy = json.load(f)
    intent_by_id = {it["id"]: it for it in taxonomy["intents"]}

    retriever = ResolutionRetriever(corpus_df)
    client = get_client()

    rows = []
    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="agent-on-golden"):
        message = str(row["customer_text"])
        try:
            cls = classify_message(message, taxonomy, args.brand, args.model, client=client)
            meta = intent_by_id.get(cls.intent_id, {"label": cls.intent_id, "description": ""})
            examples = retriever.retrieve(message, k=3)
            drafted = draft_reply(message, args.brand, meta["label"], meta.get("description", ""), examples, args.model, client=client)
            decision = decide_escalation(message, args.brand, meta["label"], cls.confidence, int(row.get("prior_customer_turns", 0)), args.model, client=client)
            rows.append({
                "customer_tweet_id": row["customer_tweet_id"], "customer_text": message,
                "gold_brand_reply": row["brand_reply_text"], "intent_id": cls.intent_id,
                "intent_confidence": cls.confidence, "intent_rationale": cls.rationale,
                "drafted_reply": drafted.reply_text, "grounding_examples": " ||| ".join(drafted.grounding_used),
                "escalate": decision.escalate, "escalation_reason": decision.reason, "escalation_source": decision.source,
            })
        except Exception as e:
            print(f"[eval_on_golden] ERROR on {row['customer_tweet_id']}: {e}", file=sys.stderr)
            rows.append({
                "customer_tweet_id": row["customer_tweet_id"], "customer_text": message,
                "gold_brand_reply": row["brand_reply_text"], "intent_id": "ERROR",
                "intent_confidence": 0.0, "intent_rationale": str(e), "drafted_reply": "",
                "grounding_examples": "", "escalate": True, "escalation_reason": f"error: {e}", "escalation_source": "error",
            })
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"[eval_on_golden] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()