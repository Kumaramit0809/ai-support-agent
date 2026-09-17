"""
Derive a small intent taxonomy (target: 8-14 intents) from the brand's real
customer messages, rather than inventing one from priors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv()


def pick_k(X, k_min=6, k_max=14, seed=42):
    n = X.shape[0]
    k_max = min(k_max, max(k_min, n - 1))
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        if k >= n:
            break
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_messages(messages: list[str], k_min=6, k_max=14, seed=42):
    vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(messages)
    k = pick_k(X, k_min=k_min, k_max=k_max, seed=seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    terms = np.array(vec.get_feature_names_out())
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    clusters = []
    for c in range(k):
        idx = [i for i, l in enumerate(labels) if l == c]
        top_terms = [terms[t] for t in order_centroids[c, :8]]
        examples = [messages[i] for i in idx[:5]]
        clusters.append({"cluster": c, "size": len(idx), "top_terms": top_terms, "examples": examples})
    return clusters, labels.tolist()


NAMING_PROMPT_TEMPLATE = """You are helping design a support-intent taxonomy for {brand} customer support,
derived from real clustered customer messages.

Below are {n_clusters} clusters. For EACH cluster, give a short snake_case intent id,
a 2-4 word human label, and a one-sentence description of what the customer needs.
Merge near-duplicate clusters into the same intent id if they clearly represent the same
underlying need. Prefer intents that are actionable and mutually distinguishable.

Respond with ONLY a JSON array, one object per input cluster index (same order, same count),
each shaped like:
  {{"cluster": <int>, "id": "snake_case_id", "label": "Human Label", "description": "..."}}

Clusters:
{clusters_block}
"""


def format_clusters_block(clusters: list[dict]) -> str:
    lines = []
    for c in clusters:
        lines.append(
            f"Cluster {c['cluster']} (n={c['size']}, top terms: {', '.join(c['top_terms'])}):\n"
            + "\n".join(f"  - {ex[:180]}" for ex in c["examples"])
        )
    return "\n\n".join(lines)


def name_clusters_with_llm(brand: str, clusters: list[dict], model: str) -> dict:
    from src.llm_client import generate_json

    prompt = NAMING_PROMPT_TEMPLATE.format(
        brand=brand, n_clusters=len(clusters), clusters_block=format_clusters_block(clusters)
    )
    named = generate_json(prompt, model=model)
    by_cluster = {item["cluster"]: item for item in named}
    return by_cluster


def name_clusters_heuristic(clusters: list[dict]) -> dict:
    out = {}
    for c in clusters:
        label = " ".join(t.replace("_", " ").title() for t in c["top_terms"][:2])
        cid = "_".join(t.replace(" ", "_") for t in c["top_terms"][:2]).lower() or f"cluster_{c['cluster']}"
        out[c["cluster"]] = {
            "cluster": c["cluster"],
            "id": cid,
            "label": label or f"Cluster {c['cluster']}",
            "description": f"Messages characterized by terms: {', '.join(c['top_terms'])}.",
        }
    return out


def build_taxonomy(brand: str, pairs_csv: str, out_path: str, k_min=6, k_max=14, model=None, offline=False, cluster_sample_size=5000, seed=42):
    from src.llm_client import DEFAULT_MODEL

    model = model or DEFAULT_MODEL
    df = pd.read_csv(pairs_csv)
    messages = df["customer_text"].dropna().astype(str).tolist()
    if len(messages) < k_min + 1:
        raise ValueError(f"Not enough messages ({len(messages)}) to cluster into >= {k_min} groups.")

    if len(messages) > cluster_sample_size:
        import random
        rng = random.Random(seed)
        messages_for_clustering = rng.sample(messages, cluster_sample_size)
    else:
        messages_for_clustering = messages

    clusters, labels = cluster_messages(messages_for_clustering, k_min=k_min, k_max=k_max)

    if offline or not os.environ.get("GROQ_API_KEY"):
        named = name_clusters_heuristic(clusters)
    else:
        named = name_clusters_with_llm(brand, clusters, model)

    intents = []
    for c in clusters:
        meta = named.get(c["cluster"], {})
        intents.append(
            {
                "id": meta.get("id", f"cluster_{c['cluster']}"),
                "label": meta.get("label", f"Cluster {c['cluster']}"),
                "description": meta.get("description", ""),
                "cluster_size": c["size"],
                "top_terms": c["top_terms"],
                "example_messages": c["examples"],
            }
        )

    taxonomy = {"brand": brand, "n_messages": len(messages), "intents": intents}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    return taxonomy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--pairs-csv", default=str(ROOT / "data/processed/resolution_pairs.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/taxonomy.json"))
    ap.add_argument("--k-min", type=int, default=6)
    ap.add_argument("--k-max", type=int, default=14)
    ap.add_argument("--model", default=None, help="Defaults to src.llm_client.DEFAULT_MODEL")
    ap.add_argument("--cluster-sample", type=int, default=5000, help="Max messages used for clustering")
    ap.add_argument("--offline", action="store_true", help="Skip LLM naming, use heuristic cluster names")
    args = ap.parse_args()

    tax = build_taxonomy(args.brand, args.pairs_csv, args.out, args.k_min, args.k_max, args.model, args.offline, cluster_sample_size=args.cluster_sample)
    print(f"[build_taxonomy] {len(tax['intents'])} intents derived from {tax['n_messages']} messages -> {args.out}")
    for it in tax["intents"]:
        print(f"  - {it['id']:30s} (n={it['cluster_size']:3d})  {it['label']}")
