# ui.py — SpendPilot AI with branding, file upload, and distribution charts
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
#  PAGE SETUP (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="SpendPilot AI",
    page_icon="🧭",
    layout="wide",
)

# Brand colors
PRIMARY_BG = "#1c2a51"   # your dark blue
ACCENT = "#e2551c"       # your orange


# ============================================================
#  SIMPLE USERNAME + PASSWORD LOGIN
# ============================================================
def check_credentials():
    """
    Simple username/password auth using st.secrets.

    Expected structure in .streamlit/secrets.toml:

    [users]
    lewis = "password1"
    manager = "password2"
    """

    # Initialize auth state once
    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
        st.session_state["username"] = None

    # Already authenticated in this session?
    if st.session_state["auth_ok"]:
        return True

    st.markdown("### 🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_col, _ = st.columns([1, 3])
    with login_col:
        login_clicked = st.button("Log in")

    if login_clicked:
        # Load users from secrets
        try:
            users = st.secrets["users"]
        except Exception:
            st.error("No [users] section found in secrets.toml. Please configure credentials.")
            st.stop()

        # users is a mapping: username -> password
        expected_password = users.get(username)

        if expected_password is not None and password == expected_password:
            st.session_state["auth_ok"] = True
            st.session_state["username"] = username
            st.success("Logged in successfully.")
            st.rerun()  # clear the login form and render the app
        else:
            st.error("Invalid username or password.")

    # If still not authenticated, stop rendering the rest of the app
    if not st.session_state["auth_ok"]:
        st.stop()

    return True


# Call auth gate before rendering the main app UI
check_credentials()


# ============================================================
#  GLOBAL STYLE
# ============================================================
st.markdown(f"""
<style>
/* Hide Streamlit default header/footer */
header {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Layout */
.block-container {{
    padding-top: 2rem !important;
    max-width: 1200px;
}}

/* Background & font */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: radial-gradient(1200px 800px at 10% 0%, rgba(226,85,28,0.10), transparent 45%),
                radial-gradient(1200px 800px at 90% 20%, rgba(255,255,255,0.05), transparent 50%),
                {PRIMARY_BG};
    color: #e5e7eb;
}}

/* Hero title */
.hero-title {{
    font-size: 3rem;
    text-align: center;
    font-weight: 800;
    margin-top: 0.2em;
    background: linear-gradient(90deg, {ACCENT}, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

/* Hero subtitle — now white */
.hero-sub {{
    text-align: center;
    font-size: 1.1rem;
    margin-top: 0.4em;
    color: #ffffff;
}}

/* Section card (kept if you want to reuse later) */
.section-card {{
    background: rgba(15,23,42,0.9);
    border-radius: 20px;
    padding: 1.7rem 2rem;
    border: 1px solid rgba(148,163,184,0.45);
    margin-top: 1.5rem;
    box-shadow: 0 18px 45px rgba(15,23,42,0.7);
}}

/* Tabs */
[data-baseweb="tab-list"] {{
    gap: 0.75rem;
    justify-content: center;
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
    overflow: visible !important;   /* allow hover lift without clipping */
}}

button[role="tab"] {{
    border-radius: 999px !important;
    padding: 0.35rem 1.4rem !important;
    background: rgba(15,23,42,0.55) !important;
    border: 1px solid rgba(148,163,184,0.8) !important;
    box-shadow: 0 8px 18px rgba(15,23,42,0.65);
    cursor: pointer;
    transition:
        background 0.15s ease,
        transform 0.1s ease,
        box-shadow 0.15s ease,
        border-color 0.15s ease;
}}

button[role="tab"]:hover {{
    background: rgba(15,23,42,0.85) !important;
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(15,23,42,0.75);
}}

button[role="tab"][aria-selected="true"] {{
    background: {ACCENT} !important;
    border-color: {ACCENT} !important;
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.55);
}}

button[role="tab"] > div[data-testid="stMarkdownContainer"] p {{
    color: #e5e7eb !important;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0 !important;
}}

button[role="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] p {{
    color: #0f172a !important;
}}

/* Inputs */
textarea, .stTextInput>div>div>input {{
    background: rgba(15,23,42,0.95) !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.6) !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: rgba(15,23,42,0.95);
    border-radius: 14px;
}}

/* Accent buttons */
.stButton>button:first-child {{
    background: {ACCENT} !important;
    color: white !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 600 !important;
}}

/* Make all labels and small UI text white */
label,
.stRadio label,
.stSelectbox label,
.stFileUploader label,
.stSlider label,
.st-expanderHeader,
.stRadio > label,
.stCheckbox > label {{
    color: #ffffff !important;
}}

/* Markdown containers that streamlit uses for text in many widgets */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span {{
    color: #ffffff !important;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HERO SECTION
# ============================================================
st.markdown("<div class='hero-title'>SpendPilot AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>AI-powered line item classification — turn raw spend data into clean, actionable categories in seconds.</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
#  TAXONOMY
# ============================================================
@st.cache_resource
def get_taxonomy():
    return load_taxonomy("taxonomy.json")

taxonomy = get_taxonomy()


# ============================================================
#  SETTINGS
# ============================================================
with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns([1.2, 1])
    with c1:
        model_name = st.text_input("Groq model", value="llama-3.1-8b-instant")
    with c2:
        conf_floor = st.slider("Confidence threshold ≥", 0.0, 1.0, 0.60, 0.05)

    if st.button("🔄 Reload taxonomy.json"):
        get_taxonomy.clear()
        st.success("Reloaded taxonomy.json")
        st.rerun()


# ============================================================
#  TABS
# ============================================================
tab_single, tab_batch = st.tabs(["🔹 Single Item", "📦 Batch (Paste or Upload)"])


# ============================================================
#  SINGLE ITEM TAB
# ============================================================
with tab_single:
    st.markdown("#### Classify a single line item")
    desc = st.text_area(
        "Item Description",
        height=120,
        placeholder="Example: 3M nitrile gloves, size large, 100 count",
    )

    classify_button = st.button("✨ Classify Item")

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
            st.markdown("### Result")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Family:** {res.family}")
                st.write(f"**Category:** {res.category1}")
            with col2:
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
                with st.expander("🔎 View similar examples used in retrieval"):
                    for ex in examples:
                        st.markdown(
                            f"""
**Similarity:** {ex['similarity']:.2f}  
**Description:** {ex['description']}  
**Family:** {ex['family']}  
**Category:** {ex['category1']}  
"""
                        )
        except Exception:
            pass


# ============================================================
#  BATCH TAB (PASTE OR FILE UPLOAD)
# ============================================================
with tab_batch:
    st.markdown("#### Classify many items at once")

    mode = st.radio(
        "Choose how to provide your items:",
        ["📄 Upload Excel/CSV", "✏️ Paste lines"],
        horizontal=True,
    )

    desc_list = []

    if mode.startswith("📄"):
        uploaded_file = st.file_uploader(
            "Upload a file (.xlsx, .xls, or .csv) with at least one text column for item descriptions",
            type=["xlsx", "xls", "csv"],
        )

        df_input = None
        desc_col = None

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)

                if df_input.empty:
                    st.warning("Uploaded file is empty.")
                else:
                    st.write("Preview of your data:")
                    st.dataframe(df_input.head(), use_container_width=True)

                    desc_col = st.selectbox(
                        "Select the column that contains item descriptions",
                        options=list(df_input.columns),
                    )

                    if desc_col:
                        desc_series = df_input[desc_col].astype(str).fillna("")
                        desc_list = [d.strip() for d in desc_series.tolist() if d.strip()]

            except Exception as e:
                st.error(f"Could not read file: {e}")

    else:
        multi = st.text_area(
            "One item per line",
            height=200,
            placeholder="3M nitrile gloves\n30x30x30 double wall carton\n1.5” Schedule 80 PVC elbow",
        )
        if multi:
            desc_list = [ln.strip() for ln in multi.splitlines() if ln.strip()]

    run_batch = st.button("🚀 Classify All Items", use_container_width=True)

    if run_batch:
        if not desc_list:
            st.warning("No valid item descriptions found. Check your file/column or pasted lines.")
        else:
            with st.spinner(f"Classifying {len(desc_list)} items…"):
                results = classify_batch_items(
                    desc_list,
                    taxonomy,
                    model_name=model_name,
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

            # ====================================================
            #  DISTRIBUTION CHARTS
            # ====================================================
            st.markdown("---")
            st.markdown("### Distribution Overview")

            # Family distribution
            if not df.empty:
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
                low = df[df["confidence"] < conf_floor]
                if not low.empty:
                    st.warning(f"{len(low)} item(s) below confidence threshold {conf_floor:.2f} — consider manual review.")







