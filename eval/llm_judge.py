"""LLM-as-judge for drafted reply quality."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.llm_client import DEFAULT_MODEL, generate_json

JUDGE_PROMPT_TEMPLATE = """You are a strict, consistent quality judge for customer support replies at {brand}.
Score the DRAFTED REPLY below on this rubric, 1-5 each (5 = excellent):

- groundedness: Does it avoid inventing facts (order status, refund amounts, policy
  claims) not supported by the customer's message or the historical context reply?
- helpfulness: Does it give the customer a clear, actionable next step or resolution?
- tone_fit: Does it match the brand's real support tone/style shown in the historical reply?
- correctness: Is it procedurally consistent with how the brand historically handles this
  (per the historical reply), even if worded differently?

Customer message:
\"\"\"{message}\"\"\"

Historical brand reply (context only, not a template to match verbatim):
\"\"\"{historical_reply}\"\"\"

DRAFTED REPLY TO SCORE:
\"\"\"{drafted_reply}\"\"\"

Respond with ONLY a JSON object:
{{"groundedness": <1-5>, "helpfulness": <1-5>, "tone_fit": <1-5>, "correctness": <1-5>, "notes": "<one short sentence>"}}
"""


@dataclass
class JudgeScore:
    groundedness: int
    helpfulness: int
    tone_fit: int
    correctness: int
    notes: str

    @property
    def overall(self) -> float:
        return (self.groundedness + self.helpfulness + self.tone_fit + self.correctness) / 4.0


def judge_reply(message: str, drafted_reply: str, historical_reply: str, brand: str, model: str = DEFAULT_MODEL, client=None) -> JudgeScore:
    prompt = JUDGE_PROMPT_TEMPLATE.format(brand=brand, message=message, historical_reply=historical_reply, drafted_reply=drafted_reply)
    data = generate_json(prompt, model=model, client=client)
    return JudgeScore(
        groundedness=int(data.get("groundedness", 0)),
        helpfulness=int(data.get("helpfulness", 0)),
        tone_fit=int(data.get("tone_fit", 0)),
        correctness=int(data.get("correctness", 0)),
        notes=data.get("notes", ""),
    )


def judge_dataframe(df: pd.DataFrame, brand: str, model: str = DEFAULT_MODEL, reply_col="drafted_reply", historical_col="gold_brand_reply", out_path=None) -> pd.DataFrame:
    from tqdm import tqdm
    from src.llm_client import get_client

    client = get_client()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="judging"):
        score = judge_reply(
            str(row["customer_text"]), str(row[reply_col]), str(row.get(historical_col, "")), brand, model, client=client
        )
        rows.append({"customer_tweet_id": row["customer_tweet_id"], **score.__dict__, "overall": score.overall})
        if out_path:
            pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-csv", required=True)
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="data/processed/judge_scores.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.outputs_csv)
    scored = judge_dataframe(df, args.brand, args.model, out_path=args.out)
    scored.to_csv(args.out, index=False)
    print(f"[llm_judge] mean overall score: {scored['overall'].mean():.2f} -> {args.out}")