# ui.py (aesthetic version with feedback loop + optional examples viewer)
import io
import os
import csv
from datetime import datetime

import pandas as pd
import streamlit as st

from classifier import (
    load_taxonomy,
    classify_with_ollama,
    Classification,
)

# Optional: similar-examples viewer (safe to skip if not present)
HAS_EXAMPLES = False
try:
    from classifier import top_k_examples
    HAS_EXAMPLES = True
except Exception:
    HAS_EXAMPLES = False

# ---------- Page + Styles ----------
st.set_page_config(page_title="Item Classifier", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #0f172a;            /* slate-900 */
  --panel: #111827;         /* gray-900 */
  --card: #0b1220;          /* deep slate */
  --text: #e5e7eb;          /* gray-200 */
  --muted: #9ca3af;         /* gray-400 */
  --brand: #22d3ee;         /* cyan-400 */
  --brand2: #60a5fa;        /* blue-400 */
  --success: #34d399;       /* emerald-400 */
  --warn: #fbbf24;          /* amber-400 */
  --danger: #fb7185;        /* rose-400 */
  --chip: #1f2937;          /* gray-800 */
}

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 10% 0%, rgba(34,211,238,0.08), transparent 40%),
              radial-gradient(1200px 800px at 90% 20%, rgba(96,165,250,0.06), transparent 45%),
              var(--bg);
  color: var(--text);
}
.block-container { max-width: 1100px; padding-top: 2.2rem; }

h1, h2, h3, h4 {
  letter-spacing: 0.2px;
}

.app-title {
  font-size: 2.1rem;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 0.25rem;
  background: linear-gradient(90deg, var(--brand), var(--brand2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtle {
  color: var(--muted);
  font-size: 0.95rem;
  margin-bottom: 1rem;
}

.panel {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 1.1rem 1.1rem 0.6rem;
  margin-bottom: 1rem;
}

.card {
  background: var(--card);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin: 0.25rem 0 0.75rem;
  box-shadow: 0 0 0 / 0 transparent, 0 18px 50px -22px rgba(0,0,0,0.55);
}

.badge {
  display: inline-block;
  padding: 0.25rem 0.55rem;
  font-size: 0.8rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  color: var(--text);
  background: #0b1324;
  margin-right: 0.4rem;
}

.chip {
  display: inline-block;
  padding: 0.35rem 0.65rem;
  font-size: 0.9rem;
  border-radius: 999px;
  background: var(--chip);
  border: 1px solid rgba(255,255,255,0.08);
  margin-right: 0.4rem;
}

.divider {
  height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.12), rgba(255,255,255,0));
  margin: 0.75rem 0 0.8rem;
}

.small {
  color: var(--muted);
  font-size: 0.85rem;
}

.ok { color: var(--success); }
.warn { color: var(--warn); }
.danger { color: var(--danger); }

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

# ---------- Helpers ----------
TRAINING_CSV = "training_examples.csv"
CORRECTIONS_CSV = "corrections.csv"

def append_row_csv(path: str, row: dict, columns: list[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            w.writeheader()
        w.writerow(row)

def rebuild_examples_index(training_csv: str = TRAINING_CSV, out_pkl: str = "examples_index.pkl"):
    try:
        import numpy as np  # noqa
        import pickle
        from sentence_transformers import SentenceTransformer

        df = pd.read_csv(training_csv, dtype=str).fillna("")
        if df.empty:
            return False, "Training CSV is empty; nothing to index."

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        texts = df["description"].astype(str).tolist()
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True)
        payload = {
            "model_name": model_name,
            "embeddings": emb,
            "descriptions": df["description"].astype(str).tolist(),
            "families": df["family"].astype(str).tolist(),
            "categories": df["category1"].astype(str).tolist(),
        }
        with open(out_pkl, "wb") as f:
            pickle.dump(payload, f)
        return True, f"Index rebuilt with {len(df)} example(s)."
    except Exception as e:
        return False, f"Index not rebuilt (optional): {e}"

@st.cache_resource
def get_taxonomy():
    return load_taxonomy("taxonomy.json")

taxonomy = get_taxonomy()

# ---------- Header ----------
st.markdown('<div class="app-title">🧭 Item Description → Family & Category</div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="subtle">Classify any item description into your <span class="badge">taxonomy.json</span>.
Add feedback to improve over time.{"  "}
<span class="badge">{'Similar examples enabled' if HAS_EXAMPLES and os.path.exists('examples_index.pkl') else 'Examples index not loaded'}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Settings Panel ----------
with st.container():
    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3 = st.columns([1.2, 1, 1.3])
        with c1:
            model_name = st.text_input("Model name", value="llama3", help="Your Ollama model name")
        with c2:
            conf_floor = st.slider("Good/confident ≥", 0.0, 1.0, 0.6, 0.05)
        with c3:
            auto_rebuild = st.checkbox(
                "Rebuild examples index after feedback",
                value=True,
                help="Requires sentence-transformers. Uncheck if you prefer to rebuild later."
            )

        # Reload taxonomy button
        reload_tax = st.button("🔄 Reload taxonomy.json")
        if reload_tax:
            get_taxonomy.clear()
            st.success("Reloaded taxonomy.json")
            st.rerun()

# ---------- Tabs ----------
tab_single, tab_batch = st.tabs(["🔹 Single item", "📦 Batch (many items)"])

# ========== Single ==========
with tab_single:
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_desc" not in st.session_state:
        st.session_state.last_desc = ""

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    desc = st.text_area(
        "Item description",
        height=120,
        placeholder="Example: 3M nitrile gloves, size large, 100 count"
    )
    cA, cB, cC = st.columns([0.6, 0.4, 0.6])
    with cA:
        go = st.button("✨ Classify", type="primary", use_container_width=True)
    with cB:
        clear = st.button("🧹 Clear", use_container_width=True)
    with cC:
        show_examples = st.checkbox("Show similar examples (if available)", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.last_pred = None
        st.session_state.last_desc = ""
        st.rerun()

    if go:
        if not desc.strip():
            st.warning("Please enter an item description.")
        else:
            with st.spinner("Thinking…"):
                try:
                    res: Classification = classify_with_ollama(desc.strip(), taxonomy, model_name=model_name)
                    st.session_state.last_desc = desc.strip()
                    st.session_state.last_pred = res
                except Exception as e:
                    st.error(f"Error while classifying: {e}")

    if st.session_state.last_pred:
        res: Classification = st.session_state.last_pred

        # Result Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**Family**  \n<span class='chip'>{res.family}</span>", unsafe_allow_html=True)
            st.markdown(f"**Category**  \n<span class='chip'>{res.category1}</span>", unsafe_allow_html=True)
        with c2:
            pct = int(round(res.confidence * 100))
            st.markdown("**Confidence**")
            st.progress(pct if 0 <= pct <= 100 else 0, text=f"{pct}%")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**Rationale**")
        st.write(res.rationale)
        st.markdown("</div>", unsafe_allow_html=True)

        # Optional: show top examples if index exists and user wants it
        if show_examples and HAS_EXAMPLES and os.path.exists("examples_index.pkl"):
            try:
                examples = top_k_examples(st.session_state.last_desc, k=5)
                if examples:
                    with st.expander("🔎 Similar labeled examples that influenced this decision"):
                        for ex in examples:
                            st.markdown(
                                f"""
<div class="card">
  <div class="small">Similarity ~ {ex['similarity']:.2f}</div>
  <div><strong>Description</strong><br>{ex['description']}</div>
  <div class="divider"></div>
  <div><span class="chip">{ex['family']}</span> <span class="chip">{ex['category1']}</span></div>
</div>
""",
                                unsafe_allow_html=True,
                            )
            except Exception:
                pass

        # Feedback Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Was this correct?")
        cY, cN = st.columns([1, 1])

        if cY.button("✅ Yes — add to training", use_container_width=True):
            append_row_csv(
                TRAINING_CSV,
                {"description": st.session_state.last_desc, "family": res.family, "category1": res.category1},
                ["description", "family", "category1"],
            )
            msg = "Saved to training_examples.csv."
            if auto_rebuild:
                ok, info = rebuild_examples_index()
                msg += f" {info}"
            st.success(msg)

        if cN.button("❌ No — correct it", use_container_width=True):
            st.session_state.show_fix = True
        else:
            st.session_state.show_fix = st.session_state.get("show_fix", False)

        # ----- Full taxonomy correction (ANY Family + Category) -----
        if st.session_state.show_fix:
            with st.form("correction_form", clear_on_submit=True):
                # Build a flat list of ALL (Family, Category) pairs
                pairs = [(f, c) for f, cats in taxonomy.items() for c in cats]
                labels = [f"{f} ⟶ {c}" for (f, c) in pairs]

                # Preselect to the model's predicted pair if present
                try:
                    default_idx = labels.index(f"{res.family} ⟶ {res.category1}")
                except ValueError:
                    default_idx = 0

                sel = st.selectbox("Correct Family + Category", labels, index=default_idx)
                family, category1 = pairs[labels.index(sel)]

                submitted = st.form_submit_button("💾 Save correction")
                if submitted:
                    append_row_csv(
                        CORRECTIONS_CSV,
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "description": st.session_state.last_desc,
                            "pred_family": res.family,
                            "pred_category1": res.category1,
                            "pred_confidence": f"{res.confidence:.4f}",
                            "correct_family": family,
                            "correct_category1": category1,
                        },
                        ["timestamp", "description", "pred_family", "pred_category1", "pred_confidence",
                         "correct_family", "correct_category1"],
                    )
                    append_row_csv(
                        TRAINING_CSV,
                        {"description": st.session_state.last_desc, "family": family, "category1": category1},
                        ["description", "family", "category1"],
                    )
                    msg = "Correction saved and added to training_examples.csv."
                    if auto_rebuild:
                        ok, info = rebuild_examples_index()
                        msg += f" {info}"
                    st.success(msg)
        st.markdown("</div>", unsafe_allow_html=True)

# ========== Batch ==========
with tab_batch:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    multi = st.text_area(
        "One description per line",
        height=200,
        placeholder="3M nitrile gloves, size large, 100 count\n30x30x30 double wall corrugated cartons\n1.5 inch schedule 80 PVC elbow",
    )
    run_batch = st.button("🚀 Classify all", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_batch:
        lines = [ln.strip() for ln in multi.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one non-empty line.")
        else:
            rows = []
            with st.spinner("Classifying…"):
                try:
                    for i, text in enumerate(lines, 1):
                        res: Classification = classify_with_ollama(
                            text,
                            taxonomy,
                            model_name=model_name,
                            include_rationale=False,   # ← omit rationale in batch
                        )
                        rows.append({
                            "description": text,
                            "family": res.family,
                            "category1": res.category1,
                            "confidence": res.confidence,
                        })
                        if i % 10 == 0:
                            st.caption(f"Processed {i} items…")

                    # Show table without rationale
                    df = pd.DataFrame(rows, columns=["description", "family", "category1", "confidence"])
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Results")
                    st.dataframe(df, use_container_width=True)

                    # Download CSV (no rationale)
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv_buf.getvalue(),
                        file_name="items_classified.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # Flag low-confidence items
                    low = df[df["confidence"] < conf_floor]
                    if not low.empty:
                        st.warning(f"{len(low)} item(s) below confidence {conf_floor:.2f} — consider manual review.")
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error while classifying batch: {e}")



