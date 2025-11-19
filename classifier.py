from __future__ import annotations

import json
import re
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# =====================================================
#  🔐 Groq Client
# =====================================================
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =====================================================
#  Classification Model
# =====================================================
class Classification(BaseModel):
    family: str
    category1: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


# =====================================================
#  Load Taxonomy
# =====================================================
def load_taxonomy(path: str | Path = "taxonomy.json") -> Dict[str, List[str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = {}
    for family, categories in data.items():
        uniq, seen = [], set()
        for c in categories:
            c2 = c.strip()
            if c2.lower() not in seen:
                uniq.append(c2)
                seen.add(c2.lower())
        cleaned[family.strip()] = uniq

    return cleaned


# =====================================================
#  Embeddings + Examples Index
# =====================================================
_EMBED_MODEL = None
_EX_INDEX = None

def load_examples_index(path: str = "examples_index.pkl"):
    global _EX_INDEX
    if _EX_INDEX is None:
        with open(path, "rb") as f:
            _EX_INDEX = pickle.load(f)
    return _EX_INDEX

def get_embed_model(model_name: str):
    global _EMBED_MODEL
    if _EMBED_MODEL is None or getattr(_EMBED_MODEL, "_name", None) != model_name:
        model = SentenceTransformer(model_name)
        model._name = model_name
        _EMBED_MODEL = model
    return _EMBED_MODEL

def top_k_examples(query: str, k: int = 5):
    idx = load_examples_index()
    model = get_embed_model(idx["model_name"])
    qv = model.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(idx["embeddings"], qv)
    top_idx = sims.argsort()[-k:][::-1]

    return [
        {
            "description": idx["descriptions"][i],
            "family": idx["families"][i],
            "category1": idx["categories"][i],
            "similarity": float(sims[i]),
        }
        for i in top_idx
    ]


# =====================================================
#  Retrieval-based Confidence
# =====================================================
def compute_retrieval_conf_from_row(sim_row: np.ndarray, idx) -> tuple[str, str, float]:
    """
    Given one row of similarities (item vs all examples), compute:
      - the best (family, category) pair
      - a confidence in [0,1] based on similarity + margin.
    """
    best_by_pair: dict[tuple[str, str], float] = {}

    families = idx["families"]
    categories = idx["categories"]

    for j, s in enumerate(sim_row):
        key = (families[j], categories[j])
        s = float(s)
        if key not in best_by_pair or s > best_by_pair[key]:
            best_by_pair[key] = s

    # If somehow no data, fall back
    if not best_by_pair:
        return "Unclassified", "Unclassified", 0.0

    # Best pair and its similarity
    (best_pair, pred_sim) = max(best_by_pair.items(), key=lambda kv: kv[1])
    fam, cat = best_pair

    # Next-best alternative similarity
    others = [v for k, v in best_by_pair.items() if k != best_pair]
    alt_sim = max(others) if others else 0.0

    margin = max(0.0, pred_sim - alt_sim)
    conf = 0.55 * pred_sim + 0.45 * margin
    conf = max(0.0, min(1.0, conf))  # clamp to [0,1]

    return fam, cat, conf



# =====================================================
#  Retrieval-only Classifier (Fast Path)
# =====================================================
def classify_by_retrieval_only(
    item_description: str,
    taxonomy: Dict[str, List[str]],
    min_confidence: float = 0.75,
) -> Classification | None:

    try:
        examples = top_k_examples(item_description, k=1)
        if not examples:
            return None

        best = examples[0]
        fam, cat = best["family"], best["category1"]

        conf = compute_retrieval_conf_from_row(item_description, fam, cat)
        if conf is None or conf < min_confidence:
            return None

        return Classification(
            family=fam,
            category1=cat,
            confidence=conf,
            rationale="retrieval-only match",
        )
    except:
        return None


# =====================================================
#  LLM Fallback (Single Item)
# =====================================================
def classify_with_llm(
    item_description: str,
    taxonomy: Dict[str, List[str]],
) -> Classification:

    system_prompt = (
        "You classify purchasing item descriptions.\n"
        "Choose exactly ONE Family and ONE Category from this taxonomy:\n\n"
        + "\n".join([f"- {f}: {', '.join(cats)}" for f, cats in taxonomy.items()])
        + "\n\nReturn JSON with keys: family, category1, confidence."
    )

    user_prompt = f"Item description: {item_description}\nReturn JSON only."

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content[content.find("{"):content.rfind("}")+1])
    except:
        return Classification(
            family="Unclassified",
            category1="Unclassified",
            confidence=0.0,
            rationale=f"Bad JSON: {content[:200]}",
        )

    fam = parsed.get("family", "Unclassified")
    cat = parsed.get("category1", parsed.get("category", "Unclassified"))
    conf = float(parsed.get("confidence", 0))

    return Classification(family=fam, category1=cat, confidence=conf, rationale="LLM fallback")


# =====================================================
#  Main Single-Item Classifier
# =====================================================
def classify_with_ollama(
    item_description: str,
    taxonomy: Dict[str, List[str]],
    include_rationale: bool = True,
    use_examples: bool = False,
) -> Classification:

    fast = classify_by_retrieval_only(item_description, taxonomy)
    if fast is not None:
        if not include_rationale:
            fast.rationale = ""
        return fast

    res = classify_with_llm(item_description, taxonomy)
    if not include_rationale:
        res.rationale = ""
    return res


# =====================================================
#  TRUE BATCH CLASSIFIER
# =====================================================
def classify_batch_items(
    items: List[str],
    taxonomy: Dict[str, List[str]],
    model_name: str = "llama-3.1-8b-instant",
    min_confidence: float = 0.75,
) -> List[Classification]:
    """
    TRUE batch classifier:
      1) Do a single batched embedding call for ALL items (if index available).
      2) Use nearest neighbor in examples_index.pkl for fast retrieval.
      3) Only send 'hard' / low-confidence items to the Groq LLM in batches.
      4) Guarantee every output is a Classification (no None).
    """

    n = len(items)
    results: List[Classification | None] = [None] * n
    hard_items: List[str] = []
    hard_indices: List[int] = []

    # ------------------------------------------------------------------
    # 1) Retrieval-first pass (batched) – if index/model available
    # ------------------------------------------------------------------
    try:
        idx = load_examples_index()
        model = get_embed_model(idx["model_name"])

        # Encode all items at once → one big call instead of N
        query_vecs = model.encode(items, normalize_embeddings=True)   # shape (N, D)
        emb = idx["embeddings"]                                      # shape (M, D)

        sims = np.dot(query_vecs, emb.T)                             # shape (N, M)

        for i, desc in enumerate(items):
            sim_row = sims[i]

            fam, cat, conf = compute_retrieval_conf_from_row(sim_row, idx)

            if conf >= min_confidence:
                results[i] = Classification(
                    family=fam,
                    category1=cat,
                    confidence=conf,
                    rationale="retrieval batch match",
                )
            else:
                hard_items.append(desc)
                hard_indices.append(i)

    except Exception:
        # If anything goes wrong in retrieval, just send everything to LLM
        hard_items = items[:]
        hard_indices = list(range(n))

    # ------------------------------------------------------------------
    # 2) LLM fallback for "hard" items (batched)
    # ------------------------------------------------------------------
    if hard_items:
        BATCH_SIZE = 50
        llm_results: List[Classification] = []

        for i in range(0, len(hard_items), BATCH_SIZE):
            batch = hard_items[i:i + BATCH_SIZE]

            system_prompt = (
                "You classify purchasing item descriptions.\n"
                "Choose exactly ONE Family and ONE Category from this taxonomy:\n\n"
                + "\n".join([f"- {f}: {', '.join(cats)}" for f, cats in taxonomy.items()])
                + "\n\nReturn a JSON list, one object per line item, "
                  "with keys: family, category1, confidence."
            )

            items_text = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])

            user_prompt = (
                "Classify EACH line item below.\n\n"
                f"{items_text}\n\n"
                "Return ONLY a JSON list of objects, in the same order."
            )

            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )

            raw = resp.choices[0].message.content

            try:
                start = raw.find("[")
                end = raw.rfind("]")
                parsed_list = json.loads(raw[start:end+1])
            except Exception:
                # If parsing fails, mark everything in this chunk as Unclassified
                parsed_list = [
                    {"family": "Unclassified", "category1": "Unclassified", "confidence": 0.0}
                    for _ in batch
                ]

            for obj in parsed_list:
                fam = obj.get("family", "Unclassified")
                cat = obj.get("category1", obj.get("category", "Unclassified"))
                conf = float(obj.get("confidence", 0.0))
                llm_results.append(
                    Classification(
                        family=fam,
                        category1=cat,
                        confidence=conf,
                        rationale="LLM batch fallback",
                    )
                )

        # Merge LLM results back into their original positions
        for idx_i, cls in zip(hard_indices, llm_results):
            results[idx_i] = cls

    # ------------------------------------------------------------------
    # 3) Final safety – no None left behind
    # ------------------------------------------------------------------
    final_results: List[Classification] = []
    for i, r in enumerate(results):
        if r is None:
            final_results.append(
                Classification(
                    family="Unclassified",
                    category1="Unclassified",
                    confidence=0.0,
                    rationale="fallback: missing result",
                )
            )
        else:
            final_results.append(r)

    return final_results







