# ui.py — streamlined + aligned with new classifier architecture
import io
import os
import csv
from datetime import datetime

import pandas as pd
import streamlit as st

from classifier import (
    load_taxonomy,
    classify_with_ollama,       # ✅ correct single-item function
    classify_batch_items,
    Classification,
)

# Optional: similar examples (not required)
try:
    from classifier import top_k_examples
    HAS_EXAMPLES = True
except Exception:
    HAS_EXAMPLES = False


# --------------------------- PAGE STYLE ---------------------------
st.set_page_config(page_title="Item Classifier", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #0f172a;
  --panel: #111827;
  --card: #0b1220;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --brand: #22d3ee;
  --brand2: #60a5fa;
  --success: #34d399;
  --warn: #fbbf24;
  --danger: #fb7185;
  --chip: #1f2937;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
}
.block-container { max-width: 1100px; padding-top: 2.2rem; }
.app-title {
  font-size: 2.1rem;
  font-weight: 700;
  margin-bottom: 0.3rem;
  background: linear-gradient(90deg, var(--brand), var(--brand2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.panel {
  background: var(--panel);
  border-radius: 18px;
  padding: 1.1rem 1.1rem 0.8rem;
  margin-bottom: 1rem;
  border: 1px solid rgba(255,255,255,0.08);
}
.card {
  background: var(--card);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin: 0.25rem 0 0.75rem;
  border: 1px solid rgba(255,255,255,0.08);
}
.chip {
  display: inline-block;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  background: var(--chip);
  margin-right: 0.4rem;
}
.divider {
  height: 1px;
  background: rgba(255,255,255,0.08);
  margin: 0.75rem 0 0.8rem;
}
textarea, .stTextInput>div>div>input {
  background: #0b1220 !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------- HELPERS ---------------------------
TRAINING_CSV = "training_examples.csv"
CORRECTIONS_CSV = "corrections.csv"

def append_row_csv(path: str, row: dict, columns: list[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            w.writeheader()
        w.writerow(row)

@st.cache_resource
def get_taxonomy():
    return load_taxonomy("taxonomy.json")

taxonomy = get_taxonomy()


# --------------------------- HEADER ---------------------------
st.markdown('<div class="app-title">🧭 Item Description → Family & Category</div>', unsafe_allow_html=True)


# --------------------------- SETTINGS ---------------------------
with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns([1.2, 1])
    with c1:
        model_name = st.text_input("Groq model", value="llama-3.1-8b-instant")
    with c2:
        conf_floor = st.slider("Confidence threshold ≥", 0.0, 1.0, 0.60, 0.05)

    reload_tax = st.button("🔄 Reload taxonomy.json")
    if reload_tax:
        get_taxonomy.clear()
        st.success("Reloaded taxonomy.json")
        st.rerun()


# --------------------------- TABS ---------------------------
tab_single, tab_batch = st.tabs(["🔹 Single item", "📦 Batch"])


# =================================================================
# SINGLE ITEM
# =================================================================
with tab_single:

    desc = st.text_area(
        "Item description",
        height=120,
        placeholder="Example: 3M nitrile gloves, size large, 100 count"
    )

    if st.button("✨ Classify", type="primary"):
        if not desc.strip():
            st.warning("Please enter a description.")
        else:
            with st.spinner("Classifying…"):
                res: Classification = classify_with_ollama(
                    desc.strip(),
                    taxonomy,
                    include_rationale=True,
                    use_examples=False,
                )

            # Output
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Result")
            st.markdown(f"**Family**  \n<span class='chip'>{res.family}</span>", unsafe_allow_html=True)
            st.markdown(f"**Category**  \n<span class='chip'>{res.category1}</span>", unsafe_allow_html=True)

            pct = int(round(res.confidence * 100))
            st.progress(pct, text=f"{pct}%")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.write(res.rationale)
            st.markdown("</div>", unsafe_allow_html=True)

    # Optional similar examples viewer
    if HAS_EXAMPLES and desc.strip():
        try:
            examples = top_k_examples(desc.strip(), k=5)
            if examples:
                with st.expander("🔎 See similar examples used for retrieval"):
                    for ex in examples:
                        st.markdown(
                            f"""
<div class="card">
  <div>Similarity: {ex['similarity']:.2f}</div>
  <strong>{ex['description']}</strong><br>
  <span class="chip">{ex['family']}</span> <span class="chip">{ex['category1']}</span>
</div>
""",
                            unsafe_allow_html=True,
                        )
        except Exception:
            pass


# =================================================================
# BATCH MODE
# =================================================================
with tab_batch:

    multi = st.text_area(
        "One item per line",
        height=200,
        placeholder="3M nitrile gloves\n30x30x30 boxes\nSchedule 80 PVC elbow",
    )

    if st.button("🚀 Classify all", use_container_width=True):
        lines = [ln.strip() for ln in multi.splitlines() if ln.strip()]

        if not lines:
            st.warning("Enter at least one line.")
        else:
            with st.spinner("Batch classifying…"):
                results = classify_batch_items(
                    lines,
                    taxonomy,
                )

            df = pd.DataFrame([
                {
                    "description": text,
                    "family": res.family,
                    "category1": res.category1,
                    "confidence": res.confidence,
                }
                for text, res in zip(lines, results)
            ])

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Results")
            st.dataframe(df, use_container_width