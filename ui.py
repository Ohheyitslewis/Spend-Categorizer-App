# ui.py — SpendPilot AI: hero card + single tab bar + batch upload + charts
import io
import os
import csv
from datetime import datetime

import pandas as pd
import streamlit as st

from classifier import (
    load_taxonomy,
    classify_with_ollama,
    classify_batch_items,
    Classification,
)

# Optional: similar examples
try:
    from classifier import top_k_examples
    HAS_EXAMPLES = True
except Exception:
    HAS_EXAMPLES = False

# ============================================================
#  PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="SpendPilot AI",
    page_icon="🧭",
    layout="wide",
)

PRIMARY_BG = "#1c2a51"   # your dark blue
ACCENT = "#e2551c"       # your orange

st.markdown(
    f"""
<style>
/* --------- Global layout & background --------- */
header {{visibility: hidden;}}
footer {{visibility: hidden;}}

.block-container {{
    padding-top: 2rem !important;
    max-width: 1200px;
}}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: radial-gradient(1200px 800px at 10% 0%, rgba(226,85,28,0.16), transparent 45%),
                radial-gradient(1200px 800px at 90% 20%, rgba(255,255,255,0.05), transparent 55%),
                {PRIMARY_BG};
    color: #f9fafb;
}}

/* --------- Hero card --------- */
.hero-wrapper {{
    display: flex;
    justify-content: center;
    margin-bottom: 1.8rem;
}}

.hero-card {{
    background: #ffffff;
    color: #111827;
    border-radius: 22px;
    padding: 1.6rem 2.2rem 1.4rem;
    box-shadow: 0 24px 60px rgba(15,23,42,0.35);
    max-width: 900px;
    width: 100%;
}}

.hero-eyebrow {{
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.3rem;
}}

.hero-title {{
    font-size: 2.1rem;
    font-weight: 800;
    color: {ACCENT};
    margin-bottom: 0.3rem;
}}

.hero-sub {{
    font-size: 0.98rem;
    color: #374151;
    margin-bottom: 0.8rem;
}}

.hero-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}}

.hero-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.6rem;
    font-size: 0.8rem;
    border-radius: 999px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    color: #111827;
}}

.hero-chip span.icon {{
    font-size: 0.8rem;
}}

/* --------- Main panel --------- */
.main-panel {{
    margin-top: 1.5rem;
    background: rgba(15,23,42,0.96);
    border-radius: 24px;
    padding: 1.6rem 1.7rem 1.4rem;
    box-shadow: 0 22px 60px rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.45);
}}

/* --------- Tabs (single bar only) --------- */
[data-baseweb="tab-list"] {{
    gap: 0.4rem;
}}

button[role="tab"] {{
    border-radius: 999px !important;
    padding: 0.45rem 1.2rem !important;
    font-size: 0.9rem !important;
    border: 1px solid rgba(148,163,184,0.6) !important;
    background: transparent !important;
    color: #e5e7eb !important;
}}

button[role="tab"][aria-selected="true"] {{
    background: {ACCENT} !important;
    color: #ffffff !important;
    border-color: {ACCENT} !important;
}}

.stTabs [data-baseweb="tab"] p {{
    color: #e5e7eb !important;
}}

/* --------- Inputs & text --------- */
textarea, .stTextInput>div>div>input {{
    background: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.7) !important;
}}

label, .stRadio label, .stFileUploader label {{
    color: #f9fafb !important;
}}

.small-label {{
    font-size: 0.86rem;
    color: #cbd5f5;
    margin-bottom: 0.2rem;
}}

/* --------- Buttons --------- */
.stButton>button:first-child {{
    background: {ACCENT} !important;
    color: white !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.4rem 1.4rem !important;
}}

/* Dataframe container */
[data-testid="stDataFrame"] {{
    background: #020617;
    border-radius: 14px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  HERO
# ============================================================
st.markdown(
    """
<div class="hero-wrapper">
  <div class="hero-card">
    <div class="hero-eyebrow">SPEND CLASSIFICATION • FP&A</div>
    <div class="hero-title">SpendPilot AI</div>
    <div class="hero-sub">
      The AI copilot for spend classification — upload raw line items and get clean, normalized
      Families & Categories in seconds.
    </div>
    <div class="hero-chips">
      <div class="hero-chip"><span class="icon">✅</span><span>Purpose-built for purchasing &amp; FP&amp;A</span></div>
      <div class="hero-chip"><span class="icon">🧠</span><span>LLM + similarity-based retrieval</span></div>
      <div class="hero-chip"><span class="icon">📊</span><span>Instant distribution by Family &amp; Category</span></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  TAXONOMY
# ============================================================
@st.cache_resource
def get_taxonomy():
    return load_taxonomy("taxonomy.json")

taxonomy = get_taxonomy()

# ============================================================
#  MAIN PANEL WITH TABS
# ============================================================
st.markdown("<div class='main-panel'>", unsafe_allow_html=True)

tabs = st.tabs(["Single item", "Batch (paste / upload)"])

# --------------------------- SINGLE ITEM ---------------------------
with tabs[0]:
    st.markdown("### Classify a single line item")

    desc = st.text_area(
        "Item description",
        height=110,
        placeholder="Example: 3M nitrile gloves, size large, 100 count",
    )

    classify_button = st.button("✨ Classify item")

    if classify_button:
        if not desc.strip():
            st.warning("Please enter an item description.")
        else:
            with st.spinner("Classifying…"):
                res: Classification = classify_with_ollama(
                    desc.strip(),
                    taxonomy,
                    include_rationale=True,
                )

            st.markdown("---")
            st.markdown("#### Result")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Family**  \n{res.family}")
                st.markdown(f"**Category**  \n{res.category1}")
            with col2:
                pct = int(round(res.confidence * 100))
                st.markdown("**Confidence**")
                st.progress(pct if 0 <= pct <= 100 else 0, text=f"{pct}%")

            st.markdown("**Rationale**")
            st.write(res.rationale)

    # Optional similar examples
    if HAS_EXAMPLES and desc.strip():
        try:
            examples = top_k_examples(desc.strip(), k=5)
            if examples:
                with st.expander("🔎 View similar examples used in retrieval"):
                    for ex in examples:
                        st.markdown(
                            f"""
**Similarity:** {ex['similarity']:.2f}  
**Description:** {ex['description']}  
**Family:** {ex['family']}  
**Category:** {ex['category1']}  

---
"""
                        )
        except Exception:
            pass

# --------------------------- BATCH MODE ---------------------------
with tabs[1]:
    st.markdown("### Classify many items at once")

    mode = st.radio(
        "How would you like to provide items?",
        ["📄 Upload Excel/CSV", "✏️ Paste lines"],
        horizontal=True,
    )

    desc_list: list[str] = []

    if mode.startswith("📄"):
        uploaded_file = st.file_uploader(
            "Upload a file (.xlsx, .xls, or .csv) with a text column of item descriptions",
            type=["xlsx", "xls", "csv"],
        )

        df_input = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)

                if df_input.empty:
                    st.warning("Uploaded file is empty.")
                else:
                    st.markdown("<div class='small-label'>Preview</div>", unsafe_allow_html=True)
                    st.dataframe(df_input.head(), use_container_width=True)

                    desc_col = st.selectbox(
                        "Select the column that contains item descriptions",
                        options=list(df_input.columns),
                    )

                    if desc_col:
                        series = df_input[desc_col].astype(str).fillna("")
                        desc_list = [s.strip() for s in series.tolist() if s.strip()]
            except Exception as e:
                st.error(f"Could not read file: {e}")

    else:
        multi = st.text_area(
            "One item per line",
            height=200,
            placeholder="3M nitrile gloves\n30x30x30 double wall corrugated carton\n1.5\" Schedule 80 PVC elbow",
        )
        if multi:
            desc_list = [ln.strip() for ln in multi.splitlines() if ln.strip()]

    run_batch = st.button("🚀 Classify all items", use_container_width=True)

    if run_batch:
        if not desc_list:
            st.warning("No valid item descriptions found. Check your file/column or pasted lines.")
        else:
            with st.spinner(f"Classifying {len(desc_list)} items…"):
                results = classify_batch_items(
                    desc_list,
                    taxonomy,
                    model_name="llama-3.1-8b-instant",
                )

            df = pd.DataFrame(
                [
                    {
                        "description": text,
                        "family": res.family,
                        "category1": res.category1,
                        "confidence": res.confidence,
                    }
                    for text, res in zip(desc_list, results)
                ]
            )

            st.markdown("#### Results")
            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download classified CSV",
                csv_buf.getvalue(),
                file_name="items_classified.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Distribution charts
            st.markdown("---")
            st.markdown("#### Distribution overview")

            if not df.empty:
                family_counts = df["family"].value_counts().reset_index()
                family_counts.columns = ["family", "count"]

                st.markdown("**By Family**")
                st.bar_chart(
                    data=family_counts.set_index("family")["count"],
                    use_container_width=True,
                )

                df["family_category"] = df["family"] + " → " + df["category1"]
                fc_counts = df["family_category"].value_counts().reset_index()
                fc_counts.columns = ["family_category", "count"]

                st.markdown("**By Family + Category**")
                st.bar_chart(
                    data=fc_counts.set_index("family_category")["count"],
                    use_container_width=True,
                )

st.markdown("</div>", unsafe_allow_html=True)




