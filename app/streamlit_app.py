import streamlit as st
from PIL import Image
import os, sys, pathlib, io, datetime
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Full screen ── */
.block-container {
    max-width: 100% !important;
    padding: 0 2.5rem !important;
}

/* ── Background ── */
.stApp {
    background: #0b0f1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(124,58,237,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(6,182,212,0.12) 0%, transparent 60%);
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 52px 20px 36px;
}
.hero-icon {
    font-size: 3.2rem;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 10px;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}

/* ── Step label ── */
.step-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 10px;
}

/* ── Verdict ── */
.verdict-wrap {
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.verdict-pneumonia {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.35);
}
.verdict-normal {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.06));
    border: 1px solid rgba(52,211,153,0.35);
}
.verdict-tag {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.verdict-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 6px;
}
.verdict-conf-text { color: #94a3b8; font-size: 0.88rem; }

/* ── Confidence bar ── */
.cbar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 8px;
    margin: 12px 0 6px;
    overflow: hidden;
}
.cbar-red   { height:8px; border-radius:99px; background: linear-gradient(90deg,#ef4444,#f87171); }
.cbar-green { height:8px; border-radius:99px; background: linear-gradient(90deg,#10b981,#34d399); }

/* ── Image captions ── */
.img-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    text-align: center;
    margin-bottom: 8px;
}

/* ── Explanation ── */
.explain-card {
    background: rgba(167,139,250,0.05);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 20px;
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.85;
}
.explain-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 12px;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
}

/* ── Disclaimer ── */
.disclaimer-bar {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    color: #92400e;
    color: #fbbf24;
    font-size: 0.8rem;
    line-height: 1.65;
    margin-top: 28px;
    opacity: 0.8;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 28px 0 !important; }

/* Streamlit image captions */
[data-testid="caption"] { color: #475569 !important; text-align: center; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def heatmap_region(arr: np.ndarray) -> str:
    h, w = arr.shape[:2]
    heat = arr.mean(axis=2).astype(float)
    tv, bv = heat[:h//2, :].mean(), heat[h//2:, :].mean()
    lv, rv = heat[:, :w//2].mean(), heat[:, w//2:].mean()
    dv, dh = abs(tv - bv), abs(lv - rv)
    if dv < 5 and dh < 5:
        return "bilateral lung fields"
    if dv > dh:
        return ("upper" if tv > bv else "lower") + " lung fields"
    return ("left" if lv > rv else "right") + "-sided lung field"


def build_explanation(prediction: str, confidence: float, arr: np.ndarray) -> str:
    region = heatmap_region(arr)
    pct = f"{confidence:.0%}"
    if prediction == "Pneumonia":
        level = "high" if confidence > 0.85 else "moderate"
        return (
            f"The model predicted **Pneumonia** with **{pct} confidence** ({level} certainty). "
            f"The Grad-CAM heatmap highlights the **{region}** as the primary area of concern. "
            f"Pneumonia typically appears on X-rays as increased opacity or whitening caused by "
            f"fluid, pus, or inflammatory tissue filling the air sacs (alveoli). "
            f"The model detected density patterns in that region consistent with such changes. "
            f"Similar appearances can also occur with consolidation, pleural effusion, or atelectasis - "
            f"a clinical evaluation is essential for an accurate diagnosis."
        )
    return (
        f"The model predicted **Normal** with **{pct} confidence**. "
        f"The Grad-CAM heatmap shows low, diffuse activation across the lung fields - no specific "
        f"region triggered a strong pneumonia signal. "
        f"Clear lung fields (appearing dark on X-ray) with no areas of consolidation or increased "
        f"opacity are consistent with healthy lung tissue. "
        f"The mild activations in the **{region}** are within expected variation and did not reach "
        f"a threshold the model associates with pathology."
    )


def build_pdf(image, arr, prediction, confidence, explanation):
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(109, 40, 217)
            self.rect(0, 0, 210, 20, 'F')
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 4)
            self.cell(0, 12, "PneumoScan AI - Diagnostic Report", ln=True)
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(160, 160, 160)
            self.cell(0, 8, "FOR EDUCATIONAL & RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS", align="C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=16)

    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.set_xy(10, 24)
    pdf.cell(0, 6, f"Generated: {now}", ln=True)
    pdf.ln(3)

    color = (185, 28, 28) if prediction == "Pneumonia" else (4, 120, 87)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "PREDICTION RESULT", ln=True)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*color)
    icon = "[!]" if prediction == "Pneumonia" else "[OK]"
    pdf.cell(0, 13, f"{icon}  {prediction}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Confidence: {confidence:.1%}", ln=True)
    pdf.ln(5)

    def save_tmp(img, name):
        p = f"/tmp/{name}.jpg"
        img.save(p, "JPEG", quality=90)
        return p

    orig_p = save_tmp(image.resize((300, 300)), "orig")
    heat_p = save_tmp(Image.fromarray(arr).resize((300, 300)), "heat")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(95, 6, "Original X-ray", align="C")
    pdf.cell(10, 6, "")
    pdf.cell(95, 6, "Grad-CAM Heatmap", align="C", ln=True)
    y = pdf.get_y()
    pdf.image(orig_p, x=10,  y=y, w=90, h=90)
    pdf.image(heat_p, x=110, y=y, w=90, h=90)
    pdf.set_y(y + 94)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "WHY DID THE MODEL PREDICT THIS?", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, _safe(explanation.replace("**", "")))
    pdf.ln(5)

    pdf.set_fill_color(254, 243, 199)
    pdf.set_draw_color(253, 230, 138)
    pdf.set_font("Helvetica", "BI", 9)
    pdf.set_text_color(120, 53, 15)
    pdf.multi_cell(0, 6,
        "DISCLAIMER: This report is AI-generated for educational/research purposes only. "
        "It is NOT a substitute for professional medical diagnosis or clinical decision-making. "
        "Consult a qualified healthcare professional for any medical concerns.",
        border=1, fill=True)


    return bytes(pdf.output())


def _safe(text: str) -> str:
    """Strip characters unsupported by Helvetica for PDF output."""
    return text.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"')


# ── Model check ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/pneumonia_model.pth"
if not os.path.exists(MODEL_PATH):
    st.error("No trained model found at `models/pneumonia_model.pth`.")
    st.stop()

from app.inference import predict_with_gradcam  # noqa: E402

# ── Hero ───────────────────────────────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(logo_path):
    logo_col, _ = st.columns([1, 5])
    with logo_col:
        st.image(logo_path, width=100)

st.markdown("""
<div class="hero">
    <div class="hero-title">PneumoScan AI</div>
    <div class="hero-sub">Upload a chest X-ray · Get an AI prediction · Download a full report</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="step-label">Upload X-ray Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:70px 0; color:#334155;">
        <div style="font-size:2.8rem; margin-bottom:14px; opacity:0.4;">📡</div>
        <div style="font-size:1rem; font-weight:500; color:#475569;">Awaiting X-ray upload</div>
        <div style="font-size:0.82rem; margin-top:6px; color:#334155;">PNG · JPG · JPEG</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run inference ──────────────────────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Running analysis..."):
    prediction, confidence, heatmap = predict_with_gradcam(image)

arr          = np.array(heatmap) if not isinstance(heatmap, np.ndarray) else heatmap
explanation  = build_explanation(prediction, confidence, arr)
is_pneumonia = prediction == "Pneumonia"
pct          = f"{confidence:.1%}"
bar_cls      = "cbar-red" if is_pneumonia else "cbar-green"
bar_w        = f"{confidence*100:.1f}%"
v_cls        = "verdict-pneumonia" if is_pneumonia else "verdict-normal"
v_color      = "#f87171" if is_pneumonia else "#34d399"
icon         = "⚠️" if is_pneumonia else "✅"

# ── Verdict ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="verdict-wrap {v_cls}">
    <div class="verdict-tag" style="color:{v_color};">AI Prediction</div>
    <div class="verdict-name" style="color:{v_color};">{icon}&nbsp; {prediction}</div>
    <div class="verdict-conf-text">Confidence score: <strong style="color:#e2e8f0;">{pct}</strong></div>
    <div class="cbar-track">
        <div class="{bar_cls}" style="width:{bar_w};"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Images ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown('<p class="img-label">Original X-ray</p>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
with c2:
    st.markdown('<p class="img-label">Grad-CAM — Model Attention</p>', unsafe_allow_html=True)
    st.image(heatmap, use_container_width=True)
    st.caption("Red / yellow = high attention  ·  Blue / green = low attention")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Explanation ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="explain-card">
    <div class="explain-title">Why did the model predict this?</div>
    {explanation.replace("**", "<strong>").replace("</strong><strong>", "").replace("**", "</strong>")}
</div>
""", unsafe_allow_html=True)

# Properly render bold markdown
st.markdown('<div style="display:none">', unsafe_allow_html=True)
st.markdown(explanation)  # renders bold correctly via streamlit markdown
st.markdown('</div>', unsafe_allow_html=True)

# ── PDF Download ───────────────────────────────────────────────────────────────
st.markdown('<p class="step-label">Download Full Report</p>', unsafe_allow_html=True)
pdf_bytes = build_pdf(image, arr, prediction, confidence, explanation)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    label="⬇  Download PDF Report",
    data=pdf_bytes,
    file_name=f"pneumoscan_{ts}.pdf",
    mime="application/pdf",
)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-bar">
    ⚠️ <strong>Disclaimer:</strong> This application is for <strong>educational and research purposes only</strong>.
    It is not a certified medical device and must not be used for clinical diagnosis,
    treatment planning, or patient triage. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)
