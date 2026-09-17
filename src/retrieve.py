"""
Retrieval over historical (customer_message -> brand_reply) resolution pairs.
Given a new incoming message, find the k most similar past customer messages
that this brand has already resolved, and return their resolutions as
grounding examples for reply drafting.

Deliberately simple (TF-IDF + cosine similarity) rather than an embedding
model/vector DB: it's fast, needs no API key, is fully inspectable, and is
also reused as-is for the "simple baseline" in eval/baselines.py (retrieval
without generation = nearest-neighbor reply).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GroundingExample:
    customer_text: str
    brand_reply_text: str
    similarity: float


class ResolutionRetriever:
    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        self._matrix = self.vectorizer.fit_transform(self.pairs_df["customer_text"].astype(str))

    def retrieve(self, query: str, k: int = 3, exclude_thread_id: str | None = None) -> list[GroundingExample]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        order = np.argsort(-sims)
        results = []
        for idx in order:
            row = self.pairs_df.iloc[idx]
            if exclude_thread_id is not None and row.get("thread_id") == exclude_thread_id:
                continue
            results.append(
                GroundingExample(
                    customer_text=str(row["customer_text"]),
                    brand_reply_text=str(row["brand_reply_text"]),
                    similarity=float(sims[idx]),
                )
            )
            if len(results) >= k:
                break
        return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    retriever = ResolutionRetriever(df)
    for ex in retriever.retrieve(args.query, k=args.k):
        print(f"[sim={ex.similarity:.3f}] Q: {ex.customer_text}")
        print(f"           A: {ex.brand_reply_text}\n")
