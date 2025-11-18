from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError
from ollama import Client
import difflib
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re  # NEW

def _normalize_confidence(val):
    """
    Accepts 0.87, 87, '87%', '0.87', 'confidence: 87%' etc.
    Returns a float in [0, 1] or None if not parsable.
    """
    try:
        if isinstance(val, (int, float)):
            v = float(val)
        else:
            s = str(val).strip()
            m = re.search(r"(\d+(\.\d+)?)", s)
            if not m:
                return None
            v = float(m.group(1))
            if "%" in s or v > 1.0:  # treat as percentage
                v /= 100.0
        return max(0.0, min(1.0, v))
    except Exception:
        return None

def retrieval_confidence_for(query: str, pred_family: str, pred_category: str):
    """
    Uses your examples_index.pkl to compute a confidence from similarity:
      - pred = best similarity among examples labeled with the predicted (family, category)
      - alt  = best similarity among *other* (family, category) pairs
      - margin = max(0, pred - alt)
      - confidence = 0.55 * pred + 0.45 * margin    (clamped 0..1)
    Returns float in [0,1] or None if index/model isn't available.
    """
    try:
        idx = load_examples_index()  # uses your existing helper
        model = get_embed_model(idx["model_name"])
        qv = model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(idx["embeddings"], qv).astype(float)

        # best similarity per (family, category)
        best_by_pair = {}
        for i, s in enumerate(sims):
            key = (idx["families"][i], idx["categories"][i])
            if key not in best_by_pair or s > best_by_pair[key]:
                best_by_pair[key] = s

        pred = best_by_pair.get((pred_family, pred_category), 0.0)
        others = [s for k, s in best_by_pair.items() if k != (pred_family, pred_category)]
        alt = max(others) if others else 0.0
        margin = max(0.0, pred - alt)
        conf = 0.55 * pred + 0.45 * margin
        return max(0.0, min(1.0, conf))
    except Exception:
        return None


class Classification(BaseModel):
    family: str
    category1: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def load_taxonomy(path: str | Path = "taxonomy.json") -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {p.resolve()}. Make sure taxonomy.json is in your project folder."
        )

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned: Dict[str, List[str]] = {}
    if not isinstance(data, dict):
        raise ValueError("taxonomy.json must contain an object mapping Family → list of Categories.")

    for family, categories in data.items():
        if not isinstance(family, str):
            raise ValueError("Each Family name must be a string.")
        if not isinstance(categories, list):
            raise ValueError(f"The value for Family '{family}' must be a list of Category strings.")

        unique = []
        seen = set()
        for c in categories:
            if not isinstance(c, str):
                raise ValueError(f"Every Category under '{family}' must be a string.")
            c2 = c.strip()
            if c2 and c2.lower() not in seen:
                seen.add(c2.lower())
                unique.append(c2)
        if not unique:
            raise ValueError(f"Family '{family}' has no valid Categories.")
        cleaned[family.strip()] = unique

    return cleaned


def build_system_prompt(taxonomy: Dict[str, List[str]]) -> str:
    lines = []
    lines.append("You are a strict classifier for purchasing line item descriptions.")
    lines.append("Choose exactly ONE Family and exactly ONE Category from the allowed taxonomy below.")
    lines.append("If uncertain, choose the single best option and explain why in the rationale.")
    lines.append("Return ONLY valid JSON with keys: family, category1, confidence, rationale.")
    lines.append("")
    lines.append("Allowed taxonomy (Family: Categories):")

    for family in sorted(taxonomy.keys(), key=lambda s: s.lower()):
        cats = ", ".join(taxonomy[family])
        lines.append(f"- {family}: {cats}")

    return "\n".join(lines)


def build_user_prompt(item_description: str) -> str:
    return (
        f"Item description: {item_description}\n"
        "Return JSON now with keys: family, category1, confidence, rationale."
    )
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
        _EMBED_MODEL = SentenceTransformer(model_name)
        _EMBED_MODEL._name = model_name
    return _EMBED_MODEL

def top_k_examples(query: str, k: int = 5):
    idx = load_examples_index()
    model = get_embed_model(idx["model_name"])
    qv = model.encode([query], normalize_embeddings=True)
    sims = np.dot(idx["embeddings"], qv[0])  # cosine similarity
    top_idx = sims.argsort()[-k:][::-1]
    examples = []
    for i in top_idx:
        examples.append({
            "description": idx["descriptions"][i],
            "family": idx["families"][i],
            "category1": idx["categories"][i],
            "similarity": float(sims[i])
        })
    return examples

def format_examples_for_prompt(examples):
    lines = ["Here are labeled examples most similar to the item description. Use these only as guidance:"]
    for j, ex in enumerate(examples, 1):
        lines.append(
            f"Example {j}:\n"
            f"  Description: {ex['description']}\n"
            f"  Family: {ex['family']}\n"
            f"  Category: {ex['category1']}"
        )
    return "\n".join(lines)

def _best_match(name: str, choices: list[str], cutoff: float = 0.6) -> str | None:
    if not name or not choices:
        return None
    matches = difflib.get_close_matches(name, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def normalize_prediction(pred: Classification, taxonomy: Dict[str, List[str]]) -> Classification:
    """Coerce model output to the closest valid Family and Category from taxonomy."""
    # 1) Best matching Family
    families = list(taxonomy.keys())
    fam_best = _best_match(pred.family, families)
    if not fam_best:
        # try to infer family from category if family was way off
        all_pairs = [(f, c) for f, cats in taxonomy.items() for c in cats]
        all_cats = [c for _, c in all_pairs]
        cat_guess = _best_match(pred.category1, all_cats)
        if cat_guess:
            for f, c in all_pairs:
                if c == cat_guess:
                    fam_best = f
                    break
    if not fam_best:
        return Classification(
            family="Unclassified",
            category1="Unclassified",
            confidence=pred.confidence,
            rationale=pred.rationale + " | Normalization failed to match any Family."
        )

    # 2) Best matching Category within the chosen Family
    cats_for_family = taxonomy[fam_best]
    cat_best = _best_match(pred.category1, cats_for_family)
    if not cat_best:
        # as a fallback, pick the closest category across all, then snap family to that
        all_pairs = [(f, c) for f, cats in taxonomy.items() for c in cats]
        all_cats = [c for _, c in all_pairs]
        cat_any = _best_match(pred.category1, all_cats)
        if cat_any:
            for f, c in all_pairs:
                if c == cat_any:
                    fam_best, cat_best = f, c
                    break
        else:
            # last resort: choose the first category for this family
            cat_best = cats_for_family[0]

    # 3) Return a normalized prediction (bump confidence a little since we corrected it)
    new_conf = min(1.0, max(pred.confidence, 0.6))
    new_rat = pred.rationale + " | Normalized to known taxonomy."
    return Classification(family=fam_best, category1=cat_best, confidence=new_conf, rationale=new_rat)


def classify_with_ollama(
    item_description: str,
    taxonomy: Dict[str, List[str]],
    model_name: str = "llama3",
    include_rationale: bool = True,
) -> Classification:
    system_prompt = build_system_prompt(taxonomy)

    # Retrieve similar labeled examples (top 4)
    examples_text = "No close examples found; rely on taxonomy."
    try:
        examples = top_k_examples(item_description, k=4)  # ↓ fewer examples
        if examples:
            # Fast path: snap to nearest if extremely similar
            best = examples[0]
            if best.get("similarity", 0.0) >= 0.92:
                raw = Classification(
                    family=best["family"],
                    category1=best["category1"],
                    confidence=0.95,
                    rationale="Nearest-neighbor match."
                )
                try:
                    return normalize_prediction(raw, taxonomy)
                except Exception:
                    return raw

            # Otherwise include top examples (threshold optional)
            filtered = [e for e in examples if e.get("similarity", 0.0) >= 0.60] or examples
            examples_text = format_examples_for_prompt(filtered)
    except Exception:
        pass

    # Build user prompt; optionally omit rationale to save tokens
    if include_rationale:
        want_keys = "family, category1, confidence, rationale"
        confidence_rule = "For confidence, return a NUMBER between 0 and 1 (e.g., 0.73)."
    else:
        want_keys = "family, category1, confidence"
        confidence_rule = "For confidence, return a NUMBER between 0 and 1 (e.g., 0.73). Do not include a rationale."

    user_prompt = (
        f"Item description: {item_description}\n"
        f"{examples_text}\n"
        f"Return strict JSON with keys exactly: {want_keys}. {confidence_rule}"
    )

    client = Client()  # defaults to http://localhost:11434
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1}
    )

    content = response["message"]["content"]
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return Classification(
            family="Unclassified",
            category1="Unclassified",
            confidence=0.0,
            rationale=f"Model did not return JSON: {content[:200]}"
        )

    import json as _json
    try:
        parsed = _json.loads(content[start:end+1])
    except Exception as e:
        return Classification(
            family="Unclassified",
            category1="Unclassified",
            confidence=0.0,
            rationale=f"JSON parse error: {e}; content: {content[:200]}"
        )

    fam = parsed.get("family") or parsed.get("Family") or "Unclassified"
    cat = parsed.get("category1") or parsed.get("category") or parsed.get("Category") or "Unclassified"
    conf = parsed.get("confidence") or parsed.get("Confidence") or 0.0
    try:
        conf = float(conf)
        if conf > 1.0:  # tolerate percentage
            conf = conf / 100.0
    except Exception:
        conf = 0.0

    rat = parsed.get("rationale") or parsed.get("Rationale") or ""
    if not include_rationale:
        rat = ""  # explicitly omit rationale in batch mode

    raw = Classification(family=fam, category1=cat, confidence=conf, rationale=rat)
    try:
        return normalize_prediction(raw, taxonomy)
    except Exception:
        return raw



if __name__ == "__main__":
    # Only runs when you execute: py classifier.py
    taxonomy = load_taxonomy("taxonomy.json")
    print(f"Loaded taxonomy with {len(taxonomy)} Families.")

    system_prompt = build_system_prompt(taxonomy)
    preview_lines = system_prompt.splitlines()[:20]
    print("\n=== System prompt preview (first 20 lines) ===")
    print("\n".join(preview_lines))
    print("=== End of preview ===\n")

    description_example = "3M nitrile gloves, size large, 100 count"
    result = classify_with_ollama(description_example, taxonomy, model_name="llama3")
    print("Classification output:")
    print(result.model_dump())

