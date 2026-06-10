import streamlit as st
import os
import time
import random

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vaanu · English → Telugu Translator",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

    /* ── Root ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0C0E14;
        color: #E8E6DF;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #0F1119;
        border-right: 1px solid #1E2130;
    }

    /* ── Hero strip ── */
    .hero {
        background: linear-gradient(135deg, #1A1F35 0%, #0C1428 60%, #0C0E14 100%);
        border: 1px solid #1E2A4A;
        border-radius: 18px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, #4F6EF744 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #7DA7FF 0%, #B8D0FF 55%, #E8C9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.05rem;
        color: #7E8BAA;
        margin: 0;
    }
    .badge {
        display: inline-block;
        background: #1A2545;
        border: 1px solid #2D3D6A;
        color: #7DA7FF;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 1rem;
    }

    /* ── Translation cards ── */
    .card {
        background: #111420;
        border: 1px solid #1E2440;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
    }
    .card-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #4A5578;
        margin-bottom: 0.6rem;
    }
    .card-result {
        font-family: 'Syne', sans-serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: #C8D8FF;
        line-height: 1.35;
        min-height: 2.2rem;
        word-break: break-word;
    }
    .card-result.source {
        color: #E8E6DF;
    }

    /* ── Glow ring on result card ── */
    .result-card {
        background: #101525;
        border: 1px solid #2D3D6A;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 0 0 1px #4F6EF720, 0 8px 32px #4F6EF714;
    }

    /* ── Translate button ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3B5BDB 0%, #5C7CFA 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.04em;
        padding: 0.65rem 2.5rem;
        cursor: pointer;
        transition: opacity 0.2s;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.88;
    }

    /* ── Text area ── */
    textarea {
        background: #0E1220 !important;
        color: #E8E6DF !important;
        border: 1px solid #1E2A4A !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
    }
    textarea:focus {
        border-color: #3B5BDB !important;
        box-shadow: 0 0 0 2px #3B5BDB33 !important;
    }

    /* ── Sidebar ── */
    .sidebar-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: #7DA7FF;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .pair-item {
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
        padding: 0.55rem 0;
        border-bottom: 1px solid #1A1F30;
        font-size: 0.84rem;
        line-height: 1.4;
    }
    .pair-en { color: #C5D0EE; flex: 1; }
    .pair-te { color: #7DA7FF; flex: 1; text-align: right; }
    .arrow   { color: #2D3D6A; flex-shrink: 0; margin-top: 1px; }

    /* ── Stats row ── */
    .stat-box {
        background: #111420;
        border: 1px solid #1E2440;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-num {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #7DA7FF;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #4A5578;
        margin-top: 2px;
    }

    /* ── Misc ── */
    .divider { border: none; border-top: 1px solid #1E2130; margin: 1.2rem 0; }
    .tip { font-size: 0.8rem; color: #4A5578; font-style: italic; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Lazy-load model ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    from transformer_analysis import load_saved_model
    return load_saved_model()

# ── Check model exists ───────────────────────────────────────────────────────
model_ready = os.path.exists("saved_model/en_te_transformer")

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="badge">🌿 Transformer · English → Telugu</div>
      <h1 class="hero-title">Vaanu</h1>
      <p class="hero-sub">A seq-to-seq Transformer trained from scratch on curated English → Telugu pairs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Stats row ────────────────────────────────────────────────────────────────
from transformer_analysis import (
    get_training_pairs, EPOCHS, VOCAB_SIZE,
    EMBED_DIM, NUM_HEADS, DENSE_DIM,
)

pairs = get_training_pairs()

c1, c2, c3, c4 = st.columns(4)
for col, (num, label) in zip(
    [c1, c2, c3, c4],
    [
        (len(pairs),    "Training pairs"),
        (VOCAB_SIZE,    "Vocab size"),
        (EMBED_DIM,     "Embedding dim"),
        (f"{NUM_HEADS}×{DENSE_DIM}", "Heads × FFN"),
    ],
):
    col.markdown(
        f'<div class="stat-box"><div class="stat-num">{num}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Main translation UI ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-label">English sentence</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        placeholder="Type an English sentence…",
        height=120,
        label_visibility="collapsed",
    )

    # Quick-fill suggestions
    suggestions = random.sample([e for e, _ in pairs], k=min(4, len(pairs)))
    st.markdown('<p class="tip">Try one of these →</p>', unsafe_allow_html=True)
    sg_cols = st.columns(2)
    for idx, sug in enumerate(suggestions):
        if sg_cols[idx % 2].button(sug, key=f"sug_{idx}"):
            user_input = sug

    st.markdown("")
    translate_btn = st.button("✦  Translate", use_container_width=True)

with col_right:
    st.markdown('<div class="card-label">Telugu translation</div>', unsafe_allow_html=True)

    if not model_ready:
        st.warning(
            "⚠️  No trained model found.  \n"
            "Run `python transformer_analysis.py` to train and save the model, then refresh.",
            icon="🛠️",
        )
    elif translate_btn and user_input.strip():
        with st.spinner("Translating…"):
            model, sv, tv = get_model()
            result = None
            try:
                from transformer_analysis import translate
                result = translate(user_input.strip(), model, sv, tv)
            except Exception as e:
                st.error(f"Translation error: {e}")

        if result:
            st.markdown(
                f'<div class="result-card">'
                f'  <div class="card-label">Result</div>'
                f'  <div class="card-result">{result}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.success("Translation complete", icon="✅")
    else:
        st.markdown(
            '<div class="result-card">'
            '  <div class="card-label">Result</div>'
            '  <div class="card-result" style="color:#2D3D6A;">— awaiting input —</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Architecture diagram ─────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

with st.expander("⬡  Model architecture", expanded=False):
    a1, a2, a3 = st.columns(3)
    for col, title, items in zip(
        [a1, a2, a3],
        ["Encoder", "Decoder", "Training"],
        [
            [
                f"Embedding dim : {EMBED_DIM}",
                f"FFN hidden    : {DENSE_DIM}",
                f"Attention heads: {NUM_HEADS}",
                "Layer norm + Dropout",
                "Positional embedding",
            ],
            [
                "Masked self-attention",
                "Cross-attention → Encoder",
                f"FFN hidden    : {DENSE_DIM}",
                "Layer norm + Dropout",
                f"Vocab out     : {VOCAB_SIZE}",
            ],
            [
                f"Max epochs    : {EPOCHS}",
                "Optimizer     : Adam",
                "LR scheduler  : ReduceLR",
                "Early stopping: 40 patience",
                "Loss          : Sparse CE",
            ],
        ],
    ):
        col.markdown(f"**{title}**")
        for it in items:
            col.markdown(f"- `{it}`")

# ── Sidebar: dataset browser ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📖 Dataset</div>', unsafe_allow_html=True)
    search = st.text_input("Filter", placeholder="Search pairs…", label_visibility="collapsed")

    filtered = [
        (e, t) for e, t in pairs
        if search.lower() in e.lower() or search.lower() in t.lower()
    ] if search else pairs

    st.caption(f"{len(filtered)} pair{'s' if len(filtered) != 1 else ''} shown")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    for eng, tel in filtered[:60]:
        st.markdown(
            f'<div class="pair-item">'
            f'  <span class="pair-en">{eng}</span>'
            f'  <span class="arrow">→</span>'
            f'  <span class="pair-te">{tel}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if len(filtered) > 60:
        st.caption(f"… and {len(filtered) - 60} more")