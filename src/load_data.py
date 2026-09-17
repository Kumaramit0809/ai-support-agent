"""
Load twcs.csv (real Kaggle file, or the schema-matching fixture), filter to one
brand, and reconstruct (customer_message -> brand_reply) resolution pairs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REAL_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "twcs.csv"
FIXTURE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_synthetic.csv"


def _is_mostly_ascii(s: str, threshold: float = 0.85) -> bool:
    if not isinstance(s, str) or not s:
        return False
    ascii_chars = sum(1 for c in s if ord(c) < 128)
    return (ascii_chars / len(s)) > threshold


@dataclass
class ResolutionPair:
    thread_id: str
    customer_tweet_id: str
    customer_text: str
    customer_created_at: str
    brand_reply_tweet_id: str
    brand_reply_text: str
    brand_reply_created_at: str
    prior_customer_turns: int


def resolve_data_path(explicit_path: Optional[str] = None) -> tuple[Path, bool]:
    if explicit_path:
        p = Path(explicit_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Given data path does not exist: {p}")
        return p, (p != FIXTURE_DATA_PATH.resolve())
    if REAL_DATA_PATH.exists():
        return REAL_DATA_PATH, True
    if not FIXTURE_DATA_PATH.exists():
        raise FileNotFoundError("Neither the real dataset nor the fixture were found.")
    return FIXTURE_DATA_PATH, False


def load_raw(explicit_path: Optional[str] = None) -> tuple[pd.DataFrame, bool]:
    path, is_real = resolve_data_path(explicit_path)
    dtype = {
        "tweet_id": str,
        "author_id": str,
        "response_tweet_id": str,
        "in_response_to_tweet_id": str,
    }
    df = pd.read_csv(path, dtype=dtype)
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().isin(["true", "1"])
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].replace({"nan": ""}).fillna("")
    df["response_tweet_id"] = df["response_tweet_id"].replace({"nan": ""}).fillna("")
    return df, is_real


def build_resolution_pairs(df: pd.DataFrame, brand: str, english_only: bool = False) -> list[ResolutionPair]:
    by_id = df.set_index("tweet_id", drop=False)
    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand)]

    pairs: list[ResolutionPair] = []
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if not parent_id or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if bool(parent["inbound"]) is not True:
            continue

        if english_only and not _is_mostly_ascii(str(parent["text"])):
            continue

        prior_turns = 0
        cursor = parent
        seen = set()
        while cursor["in_response_to_tweet_id"] and cursor["in_response_to_tweet_id"] not in seen:
            seen.add(cursor["in_response_to_tweet_id"])
            nxt_id = cursor["in_response_to_tweet_id"]
            if nxt_id not in by_id.index:
                break
            cursor = by_id.loc[nxt_id]
            if isinstance(cursor, pd.DataFrame):
                cursor = cursor.iloc[0]
            prior_turns += 1
            if prior_turns > 20:
                break

        pairs.append(
            ResolutionPair(
                thread_id=f"{parent['tweet_id']}->{reply['tweet_id']}",
                customer_tweet_id=str(parent["tweet_id"]),
                customer_text=str(parent["text"]),
                customer_created_at=str(parent["created_at"]),
                brand_reply_tweet_id=str(reply["tweet_id"]),
                brand_reply_text=str(reply["text"]),
                brand_reply_created_at=str(reply["created_at"]),
                prior_customer_turns=prior_turns,
            )
        )
    return pairs


def pairs_to_dataframe(pairs: list[ResolutionPair]) -> pd.DataFrame:
    return pd.DataFrame([p.__dict__ for p in pairs])


def load_brand_pairs(brand: str, explicit_path: Optional[str] = None, english_only: bool = False) -> tuple[pd.DataFrame, bool]:
    df, is_real = load_raw(explicit_path)
    pairs = build_resolution_pairs(df, brand, english_only=english_only)
    return pairs_to_dataframe(pairs), is_real


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--english-only", action="store_true")
    args = ap.parse_args()

    df_pairs, is_real = load_brand_pairs(args.brand, args.data_path, english_only=args.english_only)
    print(f"[load_data] using {'REAL' if is_real else 'FIXTURE'} data")
    print(f"[load_data] brand={args.brand!r} english_only={args.english_only} -> {len(df_pairs)} resolution pairs")
    if not df_pairs.empty:
        print(df_pairs.head(3).to_string())
    if args.out:
        df_pairs.to_csv(args.out, index=False)
        print(f"[load_data] wrote {args.out}")