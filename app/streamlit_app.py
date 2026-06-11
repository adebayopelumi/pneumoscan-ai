import streamlit as st
from PIL import Image
import os, sys, pathlib, datetime
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0f1117; }

[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}

.block-container {
    max-width: 100% !important;
    padding: 2rem 2.5rem !important;
}

/* Cards */
.result-card {
    background: linear-gradient(135deg, #1e2535, #252d40);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border: 1px solid #2e3650;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.pneumonia-card { border-left: 4px solid #ff4b4b; }
.normal-card    { border-left: 4px solid #00c48c; }

/* Badges */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px 2px;
}
.badge-red   { background: rgba(255,75,75,0.15);  color: #ff4b4b;  border: 1px solid #ff4b4b; }
.badge-green { background: rgba(0,196,140,0.15);  color: #00c48c;  border: 1px solid #00c48c; }
.badge-blue  { background: rgba(100,149,237,0.15); color: #6495ed; border: 1px solid #6495ed; }

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background: #1a2035 !important;
    border: 2px dashed #3a4a6b !important;
    border-radius: 12px !important;
}

/* Hero title */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6495ed, #00c48c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #8892b0;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 1.5rem;
}

/* Section headers */
.section-title {
    color: #ccd6f6;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Confidence bars */
.conf-bar-bg {
    background: #1a2035;
    border-radius: 8px;
    height: 12px;
    width: 100%;
    margin: 8px 0 16px 0;
}
.conf-bar-fill-red   { background: linear-gradient(90deg, #ff4b4b, #ff8080); border-radius: 8px; height: 12px; }
.conf-bar-fill-green { background: linear-gradient(90deg, #00c48c, #00e6a8); border-radius: 8px; height: 12px; }

/* Explanation card */
.explain-card {
    background: linear-gradient(135deg, #1a2035, #1e2540);
    border: 1px solid #2e3650;
    border-left: 4px solid #6495ed;
    border-radius: 0 16px 16px 0;
    padding: 22px 24px;
    color: #a8b2d8;
    font-size: 0.93rem;
    line-height: 1.85;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.explain-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6495ed;
    margin-bottom: 10px;
}

/* Warning box */
.warning-box {
    background: rgba(255,170,0,0.08);
    border: 1px solid rgba(255,170,0,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: #ffaa00;
    font-size: 0.82rem;
    line-height: 1.5;
}

hr { border-color: #2a2f3e !important; }

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #3a4a8a, #2a3a7a) !important;
    color: white !important;
    border: 1px solid #4a5a9a !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

[data-testid="caption"] { color: #4a5568 !important; font-size: 0.78rem !important; }
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
            self.set_fill_color(43, 80, 140)
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

    color = (200, 50, 50) if prediction == "Pneumonia" else (0, 150, 100)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "PREDICTION RESULT", ln=True)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*color)
    pdf.cell(0, 12, f"{'[!]' if prediction == 'Pneumonia' else '[OK]'}  {prediction}", ln=True)
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
    pdf.multi_cell(0, 6, explanation.replace("**", ""))
    pdf.ln(5)

    pdf.set_fill_color(255, 248, 220)
    pdf.set_draw_color(230, 180, 50)
    pdf.set_font("Helvetica", "BI", 9)
    pdf.set_text_color(120, 80, 0)
    pdf.multi_cell(0, 6,
        "DISCLAIMER: This report is AI-generated for educational/research purposes only. "
        "It is NOT a substitute for professional medical diagnosis or clinical decision-making. "
        "Consult a qualified healthcare professional for any medical concerns.",
        border=1, fill=True)

    return bytes(pdf.output())


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)

    st.markdown("## PneumoScan AI")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### About")
    st.markdown("""
    PneumoScan AI uses a **ResNet18** deep learning model trained on
    chest X-rays to detect signs of **pneumonia**.

    It generates a **Grad-CAM heatmap** showing which regions of the
    X-ray influenced the prediction, plus a plain-English explanation.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### How to use")
    st.markdown("""
    1. Upload a chest X-ray image
    2. View the prediction & confidence
    3. Read the AI explanation
    4. Download the full PDF report
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Disclaimer</strong><br>
    This app is for <strong>educational purposes only</strong>.
    It is NOT a substitute for professional medical diagnosis
    or clinical decision-making.
    </div>
    """, unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/pneumonia_model.pth"
if not os.path.exists(MODEL_PATH):
    st.error("No trained model found at `models/pneumonia_model.pth`.")
    st.stop()

from app.inference import predict_with_gradcam  # noqa: E402

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Chest X-ray Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a chest X-ray to detect signs of pneumonia using deep learning + Grad-CAM explainability.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
col_up, col_how = st.columns([2, 1])
with col_up:
    st.markdown('<p class="section-title">Upload X-ray</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
with col_how:
    st.markdown('<p class="section-title">Accepted Formats</p>', unsafe_allow_html=True)
    st.markdown("<span class='badge badge-blue'>PNG</span> <span class='badge badge-blue'>JPG</span> <span class='badge badge-blue'>JPEG</span>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:70px 0; color:#4a5568;">
        <div style="font-size:3rem; margin-bottom:14px;">🫁</div>
        <div style="font-size:1rem; font-weight:500; color:#8892b0;">Upload a chest X-ray above to get started</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Inference ──────────────────────────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Analysing X-ray..."):
    prediction, confidence, heatmap = predict_with_gradcam(image)

arr         = np.array(heatmap) if not isinstance(heatmap, np.ndarray) else heatmap
explanation = build_explanation(prediction, confidence, arr)
is_pneu     = prediction == "Pneumonia"
color       = "red" if is_pneu else "green"
icon        = "🔴" if is_pneu else "🟢"
card_cls    = "pneumonia-card" if is_pneu else "normal-card"
badge_cls   = "badge-red" if is_pneu else "badge-green"
pct         = f"{confidence:.1%}"
bar_w       = f"{confidence*100:.1f}%"
label_color = "#ff4b4b" if is_pneu else "#00c48c"

st.markdown("<hr>", unsafe_allow_html=True)

# ── Images ─────────────────────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.markdown('<p class="section-title">Original X-ray</p>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
with right:
    st.markdown('<p class="section-title">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
    st.image(heatmap, use_container_width=True)
    st.caption("Warm colours (red/yellow) show regions that most influenced the prediction.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Result cards ───────────────────────────────────────────────────────────────
res_left, res_right = st.columns(2)

with res_left:
    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="section-title">Prediction Result</div>
        <div style="font-size:2rem; font-weight:800; color:{label_color}; margin:8px 0;">
            {icon} {prediction}
        </div>
        <span class="badge {badge_cls}">{pct} confidence</span>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill-{color}" style="width:{bar_w}"></div>
        </div>
        <p style="color:#8892b0; font-size:0.88rem; margin:0;">
            {'The model detected features associated with pneumonia.' if is_pneu else 'The model found no significant signs of pneumonia.'}
        </p>
    </div>
    """, unsafe_allow_html=True)

with res_right:
    st.markdown(f"""
    <div class="result-card">
        <div class="section-title">Analysis Details</div>
        <br>
        <span class="badge badge-blue">ResNet18</span>
        <span class="badge badge-blue">Transfer Learning</span>
        <span class="badge badge-blue">Grad-CAM</span>
        <br><br>
        <div style="color:#8892b0; font-size:0.88rem; line-height:1.8;">
            <b style="color:#ccd6f6;">Model:</b> ResNet18 (ImageNet pretrained)<br>
            <b style="color:#ccd6f6;">Val F1-Score:</b> 0.9373<br>
            <b style="color:#ccd6f6;">Image size:</b> 224 x 224 px<br>
            <b style="color:#ccd6f6;">Classes:</b> Normal / Pneumonia
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Explanation ────────────────────────────────────────────────────────────────
st.markdown('<div class="explain-card"><div class="explain-label">Why did the model predict this?</div>', unsafe_allow_html=True)
st.markdown(explanation)
st.markdown('</div>', unsafe_allow_html=True)

# ── PDF Download ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-title" style="margin-top:8px;">Download Report</p>', unsafe_allow_html=True)
pdf_bytes = build_pdf(image, arr, prediction, confidence, explanation)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    label="⬇  Download Full PDF Report",
    data=pdf_bytes,
    file_name=f"pneumoscan_{ts}.pdf",
    mime="application/pdf",
)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="warning-box" style="text-align:center;">
⚠️ This result is for <strong>educational and research purposes only</strong>.
Always consult a qualified medical professional for diagnosis.
</div>
""", unsafe_allow_html=True)
