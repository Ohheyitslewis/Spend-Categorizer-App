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
def retrieval_confidence_for(query: str, family: str, category: str):
    try:
        idx = load_examples_index()
        model = get_embed_model(idx["model_name"])
        qv = model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(idx["embeddings"], qv)

        best_by_pair = {}
        for i, s in enumerate(sims):
            key = (idx["families"][i], idx["categories"][i])
            best_by_pair[key] = max(best_by_pair.get(key, -1), s)

        pred = best_by_pair.get((family, category), 0)
        alt = max([v for k, v in best_by_pair.items() if k != (family, category)], default=0)

        margin = max(0.0, pred - alt)
        confidence = 0.55 * pred + 0.45 * margin
        return float(max(0, min(1, confidence)))
    except Exception:
        return None


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

        conf = retrieval_confidence_for(item_description, fam, cat)
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
    use_examples: bool = False,
) -> List[Classification]:

    retrieval_results = []
    hard_items, hard_indices = [], []

    for i, item in enumerate(items):
        r = classify_by_retrieval_only(item, taxonomy)
        if r is None:
            retrieval_results.append(None)
            hard_items.append(item)
            hard_indices.append(i)
        else:
            retrieval_results.append(r)

    if not hard_items:
        return retrieval_results

    BATCH_SIZE = 50
    llm_results = []

    for i in range(0, len(hard_items), BATCH_SIZE):
        batch = hard_items[i:i + BATCH_SIZE]

        system_prompt = (
            "You classify purchasing item descriptions.\n"
            "Choose exactly ONE Family and ONE Category from this taxonomy:\n\n"
            + "\n".join([f"- {f}: {', '.join(cats)}" for f, cats in taxonomy.items()])
            + "\n\nReturn a JSON list, one object per item."
        )

        items_text = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])

        user_prompt = (
            f"Classify EACH line item:\n\n{items_text}\n\n"
            "Return ONLY a JSON list of objects."
        )

        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        raw = resp.choices[0].message.content
        try:
            parsed_list = json.loads(raw[raw.find("["):raw.rfind("]")+1])
        except:
            parsed_list = [{"family": "Unclassified", "category1": "Unclassified", "confidence": 0.0} for _ in batch]

        for obj in parsed_list:
            llm_results.append(
                Classification(
                    family=obj.get("family", "Unclassified"),
                    category1=obj.get("category1", obj.get("category", "Unclassified")),
                    confidence=float(obj.get("confidence", 0)),
                    rationale="",
                )
            )

    llm_i = 0
    for idx in hard_indices:
        retrieval_results[idx] = llm_results[llm_i]
        llm_i += 1

    return retrieval_results




