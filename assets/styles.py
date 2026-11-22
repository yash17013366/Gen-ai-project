"""
DocuMind – warm cream-and-brown theme.
Inject once at the top of app.py:
    from assets.styles import inject_css
    inject_css()
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Full CSS — structured by section, stable data-testid selectors throughout
# ---------------------------------------------------------------------------
CSS = """
/* ── 0. GOOGLE FONT ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

/* ── 1. GLOBAL RESET & TYPOGRAPHY ───────────────────────────────────────── */
html, body,
[class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
button, input, textarea, select {
    font-family: 'Poppins', Inter, sans-serif !important;
    color: #5F574C;
}

/* ── 2. PAGE BACKGROUND ──────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-color: #F5F1EA !important;
}
/* Body behind the app */
body, .stApp {
    background-color: #F5F1EA !important;
}

/* ── 3. TOP HEADER BAR (make cream / transparent) ───────────────────────── */
[data-testid="stHeader"] {
    background-color: #F5F1EA !important;
    border-bottom: 1px solid #E8E1D2 !important;
}
/* Deploy button & three-dot menu – keep them, just recolour */
[data-testid="stHeader"] button,
[data-testid="stHeader"] [data-testid="baseButton-header"] {
    color: #8B6B47 !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stHeader"] button:hover {
    color: #6F5236 !important;
    background: #F0E9DA !important;
    border-radius: 8px !important;
}

/* ── 4. MAIN CONTENT CONTAINER ───────────────────────────────────────────── */
.block-container {
    max-width: 1060px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    background-color: #F5F1EA !important;
}

/* ── 5. HEADINGS ─────────────────────────────────────────────────────────── */
h1, h2, h3, h4,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    font-family: 'Poppins', Inter, sans-serif !important;
    color: #1F1B16 !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
}

/* ── 6. DOCUMIND TITLE + SUBTITLE + PILL ─────────────────────────────────── */
/* Title */
.documind-title {
    color: #1F1B16 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.75rem !important;   /* ~44px */
    line-height: 1.15 !important;
}
/* Scale the emoji/icon beside the title proportionally */
.documind-title > img,
.documind-title [data-testid="stImage"],
.documind-title span[role="img"] {
    font-size: 2.75rem !important;
    vertical-align: middle !important;
}
/* Subtitle / tagline */
.documind-subtitle {
    color: #5F574C !important;
    font-family: 'Poppins', Inter, sans-serif !important;
}
/* Green evidence pill → warm cream pill */
.documind-tagline {
    background: #F0E9DA !important;
    border: 1px solid #E8E1D2 !important;
    color: #6F5236 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}
/* The leading dot inside the pill */
.documind-tagline::first-letter {
    color: #8B6B47 !important;
}

/* ── 7. FILE UPLOADER ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #FDFBF6 !important;
    border: 1.5px dashed #CDBF9F !important;
    border-radius: 12px !important;
    padding: 0.4rem 0.6rem !important;
}
/* Inner grey drop zone */
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section {
    background: #F5F1EA !important;
    border-radius: 10px !important;
    border: none !important;
}
/* "Upload" button inside the uploader */
[data-testid="stFileUploader"] button {
    background: #FDFBF6 !important;
    border: 1px solid #8B6B47 !important;
    color: #8B6B47 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #F0E9DA !important;
    border-color: #6F5236 !important;
    color: #6F5236 !important;
}
/* "200MB per file • PDF" label */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploader"] span[class*="caption"] {
    color: #8C8374 !important;
}
/* Uploader label ("Add your PDFs") */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderLabel"] {
    color: #1F1B16 !important;
    font-weight: 500 !important;
}

/* ── 8. BUTTONS (all) ────────────────────────────────────────────────────── */
/* Default secondary button */
.stButton > button {
    background: #FDFBF6 !important;
    color: #8B6B47 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
    box-shadow: none !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.stButton > button:hover {
    background: #F0E9DA !important;
    border-color: #8B6B47 !important;
    color: #6F5236 !important;
}
/* Primary button ("Process documents") */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: #8B6B47 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(139,107,71,0.18) !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: #6F5236 !important;
    box-shadow: 0 2px 10px rgba(111,82,54,0.22) !important;
}
/* Disabled primary */
.stButton > button[kind="primary"]:disabled {
    background: #C8B99E !important;
    color: #FDFBF6 !important;
    box-shadow: none !important;
}

/* ── 9. DIVIDER ──────────────────────────────────────────────────────────── */
hr, [data-testid="stDivider"] {
    border-color: #E8E1D2 !important;
}

/* ── 10. ALERT / INFO / SUCCESS / WARNING / ERROR BOXES ─────────────────── */
/* st.info  → warm callout */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stInfo"] {
    background: #EDE6D8 !important;
    border: 1px solid #DDD3BE !important;
    border-radius: 10px !important;
    color: #5F574C !important;
}
div[data-testid="stInfo"] svg,
div[data-testid="stInfo"] [data-testid="stNotificationContentInfo"] {
    color: #8B6B47 !important;
    fill: #8B6B47 !important;
}
/* st.success → muted green tint */
div[data-testid="stSuccess"],
[data-testid="stAlert"][kind="success"] {
    background: #EBF3E8 !important;
    border: 1px solid #C5DCBE !important;
    border-radius: 10px !important;
    color: #3D5C37 !important;
}
div[data-testid="stSuccess"] svg { color: #6B8F5E !important; fill: #6B8F5E !important; }
/* st.warning → sandy tint */
div[data-testid="stWarning"],
[data-testid="stAlert"][kind="warning"] {
    background: #F5EDD8 !important;
    border: 1px solid #DCCA9A !important;
    border-radius: 10px !important;
    color: #6B5520 !important;
}
div[data-testid="stWarning"] svg { color: #A07C2B !important; fill: #A07C2B !important; }
/* st.error → terracotta tint */
div[data-testid="stError"],
[data-testid="stAlert"][kind="error"] {
    background: #F8EDEA !important;
    border: 1px solid #E5BEB7 !important;
    border-radius: 10px !important;
    color: #7A2E22 !important;
}
div[data-testid="stError"] svg { color: #B4533F !important; fill: #B4533F !important; }
/* Generic stAlert wrapper */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ── 11. CHAT INPUT ──────────────────────────────────────────────────────── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background: #FDFBF6 !important;
    color: #1F1B16 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
    caret-color: #8B6B47 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8C8374 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #8B6B47 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(139,107,71,0.15) !important;
}
/* Send button inside chat input */
[data-testid="stChatInput"] button {
    color: #8B6B47 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] button:hover {
    color: #6F5236 !important;
}

/* ── 12. CHAT MESSAGES ───────────────────────────────────────────────────── */
/* Assistant message */
[data-testid="stChatMessage"][data-testid*="assistant"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(80,60,30,0.05) !important;
    margin-bottom: 0.9rem !important;
}
/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #F0E9DA !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(80,60,30,0.05) !important;
    margin-bottom: 0.9rem !important;
}
/* Fallback: all chat messages (without :has support) */
[data-testid="stChatMessage"] {
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    background: #FDFBF6 !important;
    margin-bottom: 0.9rem !important;
    box-shadow: 0 2px 8px rgba(80,60,30,0.05) !important;
}

/* ── 13. EXPANDERS (source citations) ────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(80,60,30,0.04) !important;
}
[data-testid="stExpander"] summary {
    color: #1F1B16 !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #8B6B47 !important;
}
[data-testid="stExpander"] svg {
    color: #8B6B47 !important;
    fill: #8B6B47 !important;
}

/* ── 14. METRIC TILES ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    padding: 0.5rem 0.7rem !important;   /* tighter, proportional to smaller text */
    box-shadow: 0 2px 8px rgba(80,60,30,0.04) !important;
}
[data-testid="stMetricLabel"] {
    color: #8C8374 !important;
    font-size: 0.72rem !important;   /* ~11-12px, down 30% from ~16px */
    font-weight: 600 !important;
    line-height: 1.3 !important;
}
[data-testid="stMetricValue"] {
    color: #1F1B16 !important;
    font-weight: 700 !important;
    font-size: 1.9rem !important;    /* ~30-31px, down 30% from ~44px */
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] { color: #6B8F5E !important; }

/* ── 15. VERTICAL BLOCK BORDER WRAPPER (st.container with border) ─────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(80,60,30,0.05) !important;
}

/* ── 16. STATUS WIDGET (st.status) ──────────────────────────────────────── */
[data-testid="stStatusWidget"],
[data-testid="stStatusContainer"] {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 12px !important;
    color: #5F574C !important;
}
[data-testid="stStatusWidget"] svg { color: #8B6B47 !important; fill: #8B6B47 !important; }

/* ── 17. SPINNER ─────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: #8B6B47 !important;
}
/* Generic spinner ring */
div[role="status"] > div { border-top-color: #8B6B47 !important; }

/* ── 18. PROGRESS BAR ────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: #8B6B47 !important;
}
[data-testid="stProgress"] > div {
    background: #E8E1D2 !important;
}

/* ── 19. RADIO & CHECKBOX ────────────────────────────────────────────────── */
[data-testid="stCheckbox"] input:checked + div,
input[type="checkbox"]:checked {
    accent-color: #8B6B47 !important;
}
[data-testid="stRadio"] input:checked + div {
    accent-color: #8B6B47 !important;
}
/* Fallback accent for all form elements */
input, textarea, select {
    accent-color: #8B6B47 !important;
}

/* ── 20. SELECT BOX / DROPDOWN ───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 10px !important;
    color: #1F1B16 !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #8B6B47 !important;
    box-shadow: 0 0 0 2px rgba(139,107,71,0.15) !important;
}

/* ── 21. TEXT INPUT ──────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #FDFBF6 !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 10px !important;
    color: #1F1B16 !important;
    caret-color: #8B6B47 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #8B6B47 !important;
    box-shadow: 0 0 0 2px rgba(139,107,71,0.15) !important;
    outline: none !important;
}

/* ── 22. SIDEBAR ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #EFE9DC !important;
    border-right: 1px solid #E8E1D2 !important;
}
[data-testid="stSidebar"] * {
    color: #1F1B16 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #8C8374 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1F1B16 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #FDFBF6 !important;
    color: #8B6B47 !important;
    border: 1px solid #CDBF9F !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #F0E9DA !important;
    border-color: #8B6B47 !important;
}
/* Sidebar collapse arrow */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    color: #8B6B47 !important;
}

/* ── 23. CAPTION & MUTED TEXT ────────────────────────────────────────────── */
.stCaption,
[data-testid="stCaptionContainer"],
small {
    color: #8C8374 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
}

/* ── 24. LINKS ───────────────────────────────────────────────────────────── */
a, a:visited {
    color: #8B6B47 !important;
    text-decoration: underline !important;
}
a:hover { color: #6F5236 !important; }

/* ── 25. FOCUS RINGS (remove blue, replace with brown) ───────────────────── */
*:focus-visible {
    outline: 2px solid #8B6B47 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 3px rgba(139,107,71,0.18) !important;
}

/* ── 26. CUSTOM CLASSES (file chip, verified badge, evidence heading) ─────── */
.file-chip {
    color: #6F5236 !important;
    background: #F0E9DA !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
.verified-badge {
    color: #6B8F5E !important;
    background: #EBF3E8 !important;
    border-radius: 999px !important;
    padding: 0.25rem 0.55rem !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
.evidence-heading {
    color: #1F1B16 !important;
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
    margin-top: 0.9rem !important;
}
.suggestion-label {
    color: #8C8374 !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    font-family: 'Poppins', Inter, sans-serif !important;
}

/* ── 27. SUGGESTION BUTTONS (horizontal block) ───────────────────────────── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #FDFBF6 !important;
    color: #5F574C !important;
    border: 1px solid #E8E1D2 !important;
    border-radius: 10px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    min-height: 50px !important;
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.2 !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #F0E9DA !important;
    border-color: #CDBF9F !important;
    color: #8B6B47 !important;
}

/* ── 28. FOOTER ("Made with Streamlit") ──────────────────────────────────── */
footer, [data-testid="stFooter"] {
    color: #8C8374 !important;
    background: transparent !important;
    font-family: 'Poppins', Inter, sans-serif !important;
    font-size: 0.75rem !important;
}
footer a { color: #8B6B47 !important; }

/* ── 29. RESPONSIVE ──────────────────────────────────────────────────────── */
@media (max-width: 700px) {
    .block-container { padding: 1.5rem 1rem 2rem !important; }
}
"""


def inject_css() -> None:
    """Call once at the top of app.py to apply the DocuMind warm theme."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
