# ui.py — TruPointe Spend Categorizer (branded, animated, polished)
import io
import os
import csv
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt

from classifier import (
    load_taxonomy,
    classify_with_ollama,
    classify_batch_items,
    Classification,
)

# Optional: similar examples viewer
try:
    from classifier import top_k_examples
    HAS_EXAMPLES = True
except Exception:
    HAS_EXAMPLES = False

# --------------------------- PAGE STYLE ---------------------------
st.set_page_config(page_title="TruPointe Spend Categorizer", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
:root {
  --primary: #1c2a51;   /* TruPointe blue */
  --accent: #e2551c;    /* TruPointe orange */
  --bg: #050816;
  --panel: #0b1020;
  --card: #070a16;
  --text: #f9fafb;
  --muted: #d1d5db;
  --chip: #111827;
}

/* App background */
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 0% 0%, rgba(226,85,28,0.20), transparent 55%),
    radial-gradient(circle at 100% 0%, rgba(28,42,81,0.60), transparent 55%),
    var(--bg);
  color: var(--text);
}

/* Main container */
.block-container {
  max-width: 1150px;
  padding-top: 2.4rem;
}

/* Hero animation */
@keyframes heroFadeDown {
  0%   { opacity: 0; transform: translateY(-16px); }
  100% { opacity: 1; transform: translateY(0); }
}

.hero {
  animation: heroFadeDown 0.9s ease-out forwards;
}

/* Panels & cards */
.panel {
  background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(15,23,42,0.85));
  border-radius: 18px;
  padding: 1.1rem 1.25rem 0.9rem;
  margin-bottom: 1.1rem;
  border: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow: 0 18px 45px -25px rgba(0,0,0,0.8);
}

.card {
  background: var(--card);
  border-radius: 16px;
  padding: 1.0rem 1.15rem;
  margin: 0.3rem 0 0.9rem;
  border: 1px solid rgba(148, 163, 184, 0.28);
}

/* Title + tagline */
.app-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  margin-bottom: 0.15rem;
  background: linear-gradient(110deg, var(--accent), #f97316, #facc15);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.app-subtitle {
  font-size: 0.95rem;
  color: var(--muted);
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.1rem 0.55rem;
  font-size: 0.78rem;
  border-radius: 999px;
  background: rgba(15,23,42,0.9);
  border: 1px solid rgba(148,163,184,0.45);
  color: var(--muted);
}

/* Chips & divider */
.chip {
  display: inline-block;
  padding: 0.30rem 0.7rem;
  font-size: 0.9rem;
  border-radius: 999px;
  background: var(--chip);
  border: 1px solid rgba(148,163,184,0.4);
  margin-right: 0.4rem;
}

.divider {
  height: 1px;
  background: linear-gradient(90deg, rgba(15,23,42,0), rgba(148,163,184,0.7), rgba(15,23,42,0));
  margin: 0.75rem 0 0.8rem;
}

/* Make labels + helper text white / bright */
label, .stRadio label, .stFileUploader label, .stCheckbox label {
  color: var(--text) !important;
}

/* Tabs styling + white labels */
[data-testid="stTabs"] button {
  border-radius: 999px !important;
  padding: 0.35rem 1.0rem !important;
}
[data-testid="stTabs"] button p {
  color: var(--text) !important;
  font-size: 0.9rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  background: linear-gradient(120deg, var(--accent), #f97316);
}

/* Text areas & inputs */
textarea, .stTextInput > div > div > input {
  background: #020617 !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(148,163,184,0.6) !important;
}

/* File uploader text color */
[data-testid="stFileUploader"] div {
  color: var(--text) !important;
}

/* Small muted text */
.small {
  font-size: 0.8rem;
  color: var(--muted);
}

/* Fade-in for tab sections */
@keyframes tabFadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.tab-section {
  animation: tabFadeIn 0.4s ease-out;
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

# --------------------------- HERO / HEADER ---------------------------
with st.container():
    left, right = st.columns([1.6, 1.1])

    with left:
        st.markdown(
            """
<div class="hero">
  <div class="app-title">TruPointe Spend Categorizer</div>
  <div class="app-subtitle">
    The fastest way to turn raw item descriptions into clean, consistent
    <strong>Family & Category</strong> mappings — powered by AI and your own training data.
  </div>
  <div style="margin-top: 0.7rem;">
    <span class="badge">⚡ Retrieval + LLM hybrid</span>
    <span class="badge">📊 Built for high-volume spend data</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Quick summary (this session)**")
        st.write("• Single-item & batch classification")
        st.write("• Optional Excel/CSV upload")
        st.write("• Family & Category distribution chart")
        st.markdown(
            "<div class='small'>Tip: Use <strong>Batch</strong> or <strong>Upload</strong> for large lists.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- SETTINGS ---------------------------
with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns([1.2, 1])
    with c1:
        model_name = st.text_input("Groq model", value="llama-3.1-8b-instant")
    with c2:
        conf_floor = st.slider("Flag items below confidence", 0.0, 1.0, 0.60, 0.05)

    reload_tax = st.button("🔄 Reload taxonomy.json")
    if reload_tax:
        get_taxonomy.clear()
        st.success("Reloaded taxonomy.json")
        st.rerun()

# --------------------------- TABS ---------------------------
tab_single, tab_batch, tab_upload = st.tabs(["🔹 Single item", "📦 Batch", "📁 Upload file"])

# =================================================================
# SINGLE ITEM
# =================================================================
with tab_single:
    st.markdown('<div class="tab-section">', unsafe_allow_html=True)

    desc = st.text_area(
        "Item description",
        height=120,
        placeholder="Example: 3M nitrile gloves, size large, 100 count",
    )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        go = st.button("✨ Classify", type="primary", use_container_width=True)

    if go:
        if not desc.strip():
            st.warning("Please enter a description.")
        else:
            with st.spinner("Classifying…"):
                res: Classification = classify_with_ollama(
                    desc.strip(),
                    taxonomy,
                    include_rationale=True,
                    use_examples=False,  # retrieval handles similarity; keep prompts lean
                )

            # Result card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Result")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(
                    f"**Family**  \n<span class='chip'>{res.family}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Category**  \n<span class='chip'>{res.category1}</span>",
                    unsafe_allow_html=True,
                )
            with c2:
                pct = int(round(res.confidence * 100))
                st.markdown("**Confidence**")
                st.progress(max(0, min(100, pct)), text=f"{pct}%")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**Why this classification?**")
            st.write(res.rationale or "No rationale available.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Optional similar examples
    if HAS_EXAMPLES and desc.strip():
        try:
            examples = top_k_examples(desc.strip(), k=5)
            if examples:
                with st.expander("🔎 See similar examples from your training data"):
                    for ex in examples:
                        st.markdown(
                            f"""
<div class="card">
  <div class="small">Similarity ≈ {ex['similarity']:.2f}</div>
  <div><strong>Description</strong><br>{ex['description']}</div>
  <div class="divider"></div>
  <div><span class="chip">{ex['family']}</span> <span class="chip">{ex['category1']}</span></div>
</div>
""",
                            unsafe_allow_html=True,
                        )
        except Exception:
            pass

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================================
# BATCH MODE (TEXT AREA)
# =================================================================
with tab_batch:
    st.markdown('<div class="tab-section">', unsafe_allow_html=True)

    multi = st.text_area(
        "One item description per line",
        height=220,
        placeholder=(
            "3M nitrile gloves, size large, 100 count\n"
            "30x30x30 double wall corrugated cartons\n"
            "1.5 inch schedule 80 PVC elbow"
        ),
    )

    run_batch = st.button("🚀 Classify all", use_container_width=True)

    if run_batch:
        lines = [ln.strip() for ln in multi.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one non-empty line.")
        else:
            with st.spinner("Batch classifying…"):
                results = classify_batch_items(
                    lines,
                    taxonomy,
                    model_name=model_name,
                    min_confidence=conf_floor,
                )

            df = pd.DataFrame(
                [
                    {
                        "description": text,
                        "family": res.family,
                        "category1": res.category1,
                        "confidence": res.confidence,
                    }
                    for text, res in zip(lines, results)
                ]
            )

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv_buf.getvalue(),
                file_name="items_classified_batch.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Confidence warning
            low = df[df["confidence"] < conf_floor]
            if not low.empty:
                st.warning(f"{len(low)} items below confidence {conf_floor:.2f} — review recommended.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================================
# UPLOAD MODE (EXCEL / CSV)
# =================================================================
with tab_upload:
    st.markdown('<div class="tab-section">', unsafe_allow_html=True)

    st.markdown(
        "Upload an **Excel or CSV** file containing an item description column. "
        "We'll classify every row and give you a summary by Family & Category."
    )

    uploaded = st.file_uploader(
        "Upload Excel (.xlsx) or CSV",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        # Read file
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if df_in.empty:
                st.warning("The uploaded file appears to be empty.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("**Preview of your data (first 10 rows):**")
                st.dataframe(df_in.head(10), use_container_width=True)

                cols = list(df_in.columns)
                desc_col = st.selectbox("Which column contains the item description?", cols)

                go_upload = st.button("🚀 Classify file", use_container_width=True)

                if go_upload:
                    texts = df_in[desc_col].astype(str).fillna("").tolist()
                    with st.spinner("Classifying uploaded rows…"):
                        results = classify_batch_items(
                            texts,
                            taxonomy,
                            model_name=model_name,
                            min_confidence=conf_floor,
                        )

                    df_res = pd.DataFrame(
                        [
                            {
                                "description": txt,
                                "family": res.family,
                                "category1": res.category1,
                                "confidence": res.confidence,
                            }
                            for txt, res in zip(texts, results)
                        ]
                    )

                    full = pd.concat([df_in.reset_index(drop=True), df_res.reset_index(drop=True)], axis=1)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Classified dataset")
                    st.dataframe(full.head(50), use_container_width=True)

                    # Download full classified file
                    out_buf = io.StringIO()
                    full.to_csv(out_buf, index=False)
                    st.download_button(
                        "⬇️ Download full classified CSV",
                        out_buf.getvalue(),
                        file_name="items_classified_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # -------- Distribution charts --------
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown("### 📊 Distribution")

                    # Family distribution
                    fam_counts = (
                        full["family"]
                        .fillna("Unclassified")
                        .value_counts()
                        .reset_index()
                        .rename(columns={"index": "family", "family": "count"})
                    )
                    fam_chart = (
                        alt.Chart(fam_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("family:N", sort="-y", title="Family"),
                            y=alt.Y("count:Q", title="Count"),
                            tooltip=["family", "count"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(fam_chart, use_container_width=True)

                    # Family+Category distribution
                    combo = (
                        full.assign(combo=lambda d: d["family"].fillna("Unclassified")
                                              + " → "
                                              + d["category1"].fillna("Unclassified"))
                        ["combo"]
                        .value_counts()
                        .reset_index()
                        .rename(columns={"index": "combo", "combo": "count"})
                    )

                    combo_chart = (
                        alt.Chart(combo)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y("combo:N", sort="-x", title="Family → Category"),
                            tooltip=["combo", "count"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(combo_chart, use_container_width=True)

                    # Low-confidence summary
                    low_upload = full[full["confidence"] < conf_floor]
                    if not low_upload.empty:
                        st.warning(
                            f"{len(low_upload)} rows are below confidence {conf_floor:.2f}. "
                            "You may want to review or correct those classifications."
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

