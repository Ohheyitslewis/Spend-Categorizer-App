# ui.py — SpendPilot AI polished UI (hero card + pill mode switch + upload + charts)
import io
import os
import csv

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
#  PAGE CONFIG & BRANDING
# ============================================================
st.set_page_config(
    page_title="SpendPilot AI",
    page_icon="🧭",
    layout="wide",
)

PRIMARY_BG = "#1c2a51"   # your dark blue
ACCENT = "#e2551c"       # your orange
CONF_FLOOR = 0.60        # low-confidence warning threshold
MODEL_NAME = "llama-3.1-8b-instant"

# ============================================================
#  GLOBAL STYLES
# ============================================================
st.markdown(
    f"""
<style>
/* Hide default header/footer */
header {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* Layout */
.block-container {{
    padding-top: 1.8rem !important;
    max-width: 1200px;
}}

/* Background & base text */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:
        radial-gradient(1000px 700px at 10% 0%, rgba(226,85,28,0.15), transparent 55%),
        radial-gradient(1200px 800px at 90% 20%, rgba(255,255,255,0.06), transparent 55%),
        {PRIMARY_BG};
    color: #f9fafb;
}}

/* Smooth fade-in animations */
@keyframes fadeInDown {{
  from {{ opacity: 0; transform: translateY(-16px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeInUp {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

/* HERO WRAPPER + CARD */
.hero-wrapper {{
    display: flex;
    justify-content: center;
    margin-bottom: 1.4rem;
}}

.hero-card {{
    background: #ffffff;
    color: #111827;
    border-radius: 20px;
    padding: 1.4rem 2.2rem 1.6rem;
    max-width: 780px;
    width: 100%;
    box-shadow: 0 22px 60px rgba(15,23,42,0.55);
    border: 1px solid rgba(15,23,42,0.08);
    animation: fadeInDown 0.6s ease-out;
}}

.hero-title-main {{
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 0.25rem;
    background: linear-gradient(90deg, {ACCENT}, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-sub {{
    font-size: 0.98rem;
    color: #4b5563;
    margin-bottom: 0.8rem;
}}

.hero-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.2rem;
}}

.hero-tag {{
    font-size: 0.78rem;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #e5e7eb;
}}

/* MAIN CARD */
.main-card {{
    background: rgba(15,23,42,0.96);
    border-radius: 22px;
    padding: 1.6rem 1.6rem 1.4rem;
    border: 1px solid rgba(148,163,184,0.55);
    box-shadow: 0 20px 55px rgba(15,23,42,0.8);
    animation: fadeInUp 0.6s ease-out;
    margin-bottom: 2.0rem;
}}

/* MODE PILLS */
.mode-bar {{
    display: flex;
    justify-content: center;
    margin-bottom: 1.4rem;
}}

.mode-pill-group {{
    display: inline-flex;
    background: rgba(15,23,42,0.9);
    padding: 0.25rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.7);
}}

.mode-pill {{
    padding: 0.45rem 1.1rem;
    font-size: 0.92rem;
    border-radius: 999px;
    cursor: pointer;
    border: none;
    background: transparent;
    color: #e5e7eb;
    transition: all 0.18s ease-out;
    white-space: nowrap;
}}

.mode-pill.active {{
    background: linear-gradient(135deg, {ACCENT}, #fb923c);
    color: #111827;
    font-weight: 600;
    box-shadow: 0 10px 25px rgba(0,0,0,0.45);
}}

/* Section titles */
.section-title {{
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}}

/* Inputs */
textarea, .stTextInput>div>div>input {{
    background: rgba(15,23,42,0.98) !important;
    color: #f9fafb !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.7) !important;
}}

textarea::placeholder {{
    color: #9ca3af !important;
}}

/* Buttons */
.stButton>button {{
    border-radius: 999px !important;
    border: none !important;
    background: {ACCENT} !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.4rem !important;
}}

.stDownloadButton>button {{
    border-radius: 999px !important;
    border: 1px solid rgba(148,163,184,0.7) !important;
    background: transparent !important;
    color: #f9fafb !important;
    font-weight: 500 !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    background: rgba(15,23,42,0.96);
    border-radius: 14px;
}}

small, .stCaption, .stMarkdown p {{
    color: #e5e7eb;
}}
</style>
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
#  HERO SECTION
# ============================================================
st.markdown(
    """
<div class="hero-wrapper">
  <div class="hero-card">
    <div class="hero-title-main">SpendPilot AI</div>
    <div class="hero-sub">
      The AI copilot for spend classification — upload raw line items and get clean,
      normalized Families & Categories in seconds.
    </div>
    <div class="hero-tags">
      <span class="hero-tag">✅ Purpose-built for purchasing & FP&amp;A</span>
      <span class="hero-tag">🧠 LLM + similarity-based retrieval</span>
      <span class="hero-tag">📊 Instant distribution by Family & Category</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  MODE STATE (Single vs Batch)
# ============================================================
if "sp_mode" not in st.session_state:
    st.session_state.sp_mode = "single"

def set_mode(new_mode: str):
    st.session_state.sp_mode = new_mode

# ============================================================
#  MAIN CARD
# ============================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# MODE PILLS
col_mode = st.columns([1])[0]
with col_mode:
    st.markdown(
        f"""
<div class="mode-bar">
  <div class="mode-pill-group">
    <button class="mode-pill {'active' if st.session_state.sp_mode == 'single' else ''}"
            onclick="window.parent.postMessage({{'type': 'spendpilot-set-mode', 'mode': 'single'}}, '*')">
      Single item
    </button>
    <button class="mode-pill {'active' if st.session_state.sp_mode == 'batch' else ''}"
            onclick="window.parent.postMessage({{'type': 'spendpilot-set-mode', 'mode': 'batch'}}, '*')">
      Batch (paste / upload)
    </button>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# NOTE:
# Streamlit doesn't natively support JS event wiring in Python,
# so we mirror the pill state with simple buttons underneath.

c1, c2 = st.columns(2)
with c1:
    if st.button("Single item", key="mode_single_btn"):
        set_mode("single")
with c2:
    if st.button("Batch (paste / upload)", key="mode_batch_btn"):
        set_mode("batch")

st.write("")  # spacer

mode = st.session_state.sp_mode

# ============================================================
#  SINGLE ITEM MODE
# ============================================================
if mode == "single":
    st.markdown('<div class="section-title">Classify a single line item</div>', unsafe_allow_html=True)

    desc = st.text_area(
        "Item description",
        height=120,
        placeholder="Example: 3M nitrile gloves, size large, 100 count",
        key="single_desc",
    )

    classify_single = st.button("✨ Classify item", key="single_go")

    if classify_single:
        if not desc.strip():
            st.warning("Please enter an item description.")
        else:
            with st.spinner("Classifying…"):
                res: Classification = classify_with_ollama(
                    desc.strip(),
                    taxonomy,
                    include_rationale=True,
                )

            # Result
            st.markdown("---")
            st.markdown("#### Result")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Family:** {res.family}")
                st.write(f"**Category:** {res.category1}")
            with col_b:
                pct = int(round(res.confidence * 100))
                st.write("**Confidence**")
                st.progress(pct if 0 <= pct <= 100 else 0, text=f"{pct}%")

            st.markdown("**Rationale**")
            st.write(res.rationale)

    # Optional similar examples
    if HAS_EXAMPLES and desc.strip():
        try:
            examples = top_k_examples(desc.strip(), k=5)
            if examples:
                with st.expander("🔎 View similar labeled examples"):
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

# ============================================================
#  BATCH MODE
# ============================================================
else:
    st.markdown('<div class="section-title">Classify many items at once</div>', unsafe_allow_html=True)

    mode_choice = st.radio(
        "Input method",
        ["📄 Upload Excel/CSV", "✏️ Paste lines"],
        horizontal=True,
        key="batch_input_mode",
    )

    desc_list: list[str] = []
    df_input = None

    if mode_choice.startswith("📄"):
        uploaded_file = st.file_uploader(
            "Upload .xlsx, .xls, or .csv with a text column for item descriptions",
            type=["xlsx", "xls", "csv"],
            key="batch_file",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)

                if df_input.empty:
                    st.warning("Uploaded file is empty.")
                else:
                    st.write("Preview:")
                    st.dataframe(df_input.head(), use_container_width=True)

                    desc_col = st.selectbox(
                        "Select the description column",
                        options=list(df_input.columns),
                        key="batch_desc_col",
                    )

                    if desc_col:
                        desc_series = df_input[desc_col].astype(str).fillna("")
                        desc_list = [d.strip() for d in desc_series.tolist() if d.strip()]

            except Exception as e:
                st.error(f"Could not read file: {e}")

    else:
        multi = st.text_area(
            "One item per line",
            height=220,
            placeholder="3M nitrile gloves, size large, 100 count\n30x30x30 double wall carton\n1.5” Schedule 80 PVC elbow",
            key="batch_paste",
        )
        if multi:
            desc_list = [ln.strip() for ln in multi.splitlines() if ln.strip()]

    run_batch = st.button("🚀 Classify all items", use_container_width=True, key="batch_go")

    if run_batch:
        if not desc_list:
            st.warning("No valid item descriptions found. Check your file/column or pasted lines.")
        else:
            with st.spinner(f"Classifying {len(desc_list)} items…"):
                results = classify_batch_items(
                    desc_list,
                    taxonomy,
                    model_name=MODEL_NAME,
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

            # Results table
            st.markdown("### Results")
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
            if not df.empty:
                st.markdown("---")
                st.markdown("### Distribution overview")

                # Family distribution
                family_counts = df["family"].value_counts().reset_index()
                family_counts.columns = ["family", "count"]
                st.markdown("**By Family**")
                st.bar_chart(
                    data=family_counts.set_index("family")["count"],
                    use_container_width=True,
                )

                # Family + Category distribution
                df["family_category"] = df["family"] + " → " + df["category1"]
                fc_counts = df["family_category"].value_counts().reset_index()
                fc_counts.columns = ["family_category", "count"]
                st.markdown("**By Family + Category**")
                st.bar_chart(
                    data=fc_counts.set_index("family_category")["count"],
                    use_container_width=True,
                )

                # Low confidence flag
                low = df[df["confidence"] < CONF_FLOOR]
                if not low.empty:
                    st.warning(
                        f"{len(low)} item(s) below confidence threshold {CONF_FLOOR:.2f} — consider manual review."
                    )

# Close main-card div
st.markdown("</div>", unsafe_allow_html=True)




