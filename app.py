# app.py
import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import datetime
from feature_extractor import extract_features_from_array
from classifier import load_classifier, predict
from disease_info import get_disease_info, get_severity_color

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2d6a4f; }
    .top-header {
        border-bottom: 2px solid #2d6a4f;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-healthy {
        background-color: #d8f3dc;
        border-left: 5px solid #2d6a4f;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }
    .result-disease {
        background-color: #ffe8e8;
        border-left: 5px solid #e63946;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }
    .result-text { font-size: 1.5rem; font-weight: bold; color: #1b1b1b; }
    .confidence-text { font-size: 1rem; color: #555; margin-top: 0.3rem; }
    .info-box {
        background-color: #eaf4ee;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #b7e4c7;
        color: #1b4332;
        font-size: 0.95rem;
    }
    .disease-card {
        background-color: #fff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #ddd;
        margin-top: 1rem;
        color: #1b1b1b !important;
    }
    .disease-card p {
        color: #1b1b1b !important;
        font-size: 0.95rem;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf = load_classifier("models/classifier.pkl")
    with open("models/selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    with open("models/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return clf, selected_features, class_names

# ── PDF REPORT ───────────────────────────────────────────────
def generate_pdf(predicted_class, confidence, top5_labels, top5_values, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Plant Disease Detection Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Detection Result", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Predicted Disease : {predicted_class.replace('_', ' ')}", ln=True)
    pdf.cell(0, 8, f"Confidence        : {confidence:.1f}%", ln=True)
    pdf.cell(0, 8, f"Severity          : {info['severity']}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Description", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, info['description'])
    pdf.ln(3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Symptoms", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, info['symptoms'])
    pdf.ln(3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Treatment Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, tip in enumerate(info['treatment'], 1):
        pdf.cell(0, 8, f"{i}. {tip}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Top 5 Predictions", ln=True)
    pdf.set_font("Arial", "", 11)
    for label, val in zip(top5_labels, top5_values):
        pdf.cell(0, 8, f"  {label.replace('_', ' ')}: {val:.1f}%", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 8, "JAIN University - ISE Dept | Plant Disease Detector 2026", align="C")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Plant Disease Detector")
    st.caption("AI-powered leaf analysis")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
1. 📸 Upload a leaf image
2. 🔬 112 features extracted
3. 🧬 GA picks best 50 features
4. 🌲 Random Forest predicts
    """)

    st.markdown("---")
    st.markdown("### Model Stats")
    st.metric("Accuracy", "94.23%")
    st.metric("GA Features", "50 / 112")
    st.metric("Classes", "15")
    st.metric("Training Images", "20,638")

    st.markdown("---")
    st.markdown("### 🌱 Supported Plants")
    st.markdown("🫑 Pepper &nbsp; 🥔 Potato &nbsp; 🍅 Tomato")

    st.markdown("---")
    st.markdown("### 🧬 GA Fitness Over Generations")
    gen = list(range(21))
    max_fit = [0.903,0.908,0.907,0.913,0.913,0.913,0.916,
               0.918,0.919,0.919,0.921,0.921,0.921,0.924,
               0.924,0.924,0.924,0.924,0.925,0.925,0.925]
    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f9f9f9')
    ax.plot(gen, max_fit, color='#2d6a4f', linewidth=2)
    ax.fill_between(gen, max_fit, alpha=0.15, color='#2d6a4f')
    ax.set_xlabel("Generation", color='#555', fontsize=8)
    ax.set_ylabel("Accuracy", color='#555', fontsize=8)
    ax.set_title("Actual training run", fontsize=7, color='#888', style='italic')
    ax.tick_params(colors='#555', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#ddd')
    st.pyplot(fig)
    plt.close()

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="top-header">', unsafe_allow_html=True)
st.title("🌿 Plant Disease Detector")
st.caption("Genetic Algorithm + Random Forest | JAIN University — ISE Dept | 2026")
st.markdown('</div>', unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Image", "⚖️ Compare Two Leaves"])

# ════════════════════════════════════════════════════════════
# TAB 1 — SINGLE IMAGE
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload a Leaf Image")
    use_camera = st.checkbox("📷 Use Camera instead")

    if use_camera:
        uploaded_file = st.camera_input("Take a photo of the leaf")
    else:
        uploaded_file = st.file_uploader(
            "JPG, JPEG or PNG — clear close-up of a single leaf",
            type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            st.markdown("#### Image Info")
            green = img_array[:, :, 1]
            red   = img_array[:, :, 0]
            blue  = img_array[:, :, 2]
            green_pixels = np.sum((green > 60) & (green > red) & (green > blue))
            green_ratio  = green_pixels / img_array.size

            st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
            st.write(f"**Green ratio:** {green_ratio:.3f}")

            if green_ratio < 0.03:
                st.error("❌ This doesn't look like a plant/leaf image!")
                st.stop()
            else:
                st.success("✅ Plant/leaf detected!")

            st.markdown('<div class="info-box">💡 For best results, upload a <b>close-up</b> photo of a single leaf in good lighting.</div>', unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🔍 Detect Disease", use_container_width=True):
            if not os.path.exists("models/classifier.pkl"):
                st.error("⚠️ Model not found! Please run train.py first.")
                st.stop()

            with st.spinner("🧬 Running GA feature selection & analyzing leaf..."):
                try:
                    clf, selected_features, class_names = load_models()
                    features = extract_features_from_array(img_array)
                    predicted_class, confidence, proba = predict(
                        clf, features, selected_features, class_names
                    )
                    info = get_disease_info(predicted_class)

                    st.markdown("### 🎯 Result")

                    if "healthy" in predicted_class.lower():
                        st.markdown(f"""
                        <div class="result-healthy">
                            <div class="result-text">✅ {predicted_class.replace("_", " ")}</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-disease">
                            <div class="result-text">⚠️ {predicted_class.replace("_", " ")}</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── DISEASE INFO ──────────────────────────
                    if not info:
                        info = {
                            'severity': 'None',
                            'description': 'This leaf appears healthy with no signs of disease.',
                            'symptoms': 'No symptoms detected.',
                            'treatment': ['Continue regular watering and fertilisation.', 'Monitor periodically for early signs of disease.']
                        }
                    severity_color = get_severity_color(info['severity'])
                    st.markdown(f"""
                    <div class="disease-card">
                        <span class="severity-badge" style="background-color:{severity_color}">
                            Severity: {info['severity']}
                        </span>
                        <p><b>📋 Description:</b> {info['description']}</p>
                        <p><b>🔍 Symptoms:</b> {info['symptoms']}</p>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("#### 💊 Treatment Recommendations")
                    for tip in info['treatment']:
                        st.markdown(f"- {tip}")

                    # ── TOP 5 CHART ───────────────────────────
                    st.markdown("#### 📊 Top 5 Predictions")
                    top5_idx = np.argsort(proba)[::-1][:5]
                    top5_labels = [class_names[i] for i in top5_idx]
                    top5_values = [proba[i] * 100 for i in top5_idx]

                    fig2, ax2 = plt.subplots(figsize=(7, 3))
                    fig2.patch.set_facecolor('#f9f9f9')
                    ax2.set_facecolor('#f9f9f9')
                    labels = [class_names[i].replace("_", " ") for i in top5_idx]
                    colors = ['#2d6a4f' if 'healthy' in class_names[i].lower()
                              else '#e63946' for i in top5_idx]
                    ax2.barh(labels[::-1], top5_values[::-1], color=colors[::-1], height=0.5)
                    ax2.set_xlabel("Confidence %", color='#555', fontsize=9)
                    ax2.tick_params(colors='#333', labelsize=8)
                    for spine in ax2.spines.values():
                        spine.set_edgecolor('#ddd')
                    st.pyplot(fig2)
                    plt.close()

                    st.info(f"🧬 GA selected **{len(selected_features)} out of 112 features** for this prediction.")

                    # ── PDF DOWNLOAD ──────────────────────────
                    st.markdown("#### 📄 Download Report")
                    pdf_path = generate_pdf(
                        predicted_class, confidence,
                        top5_labels, top5_values, info
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=f,
                            file_name=f"plant_disease_report_{predicted_class}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════
# TAB 2 — COMPARE TWO LEAVES
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Compare Two Leaf Images")
    st.caption("Upload two leaf images to compare their disease predictions side by side.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🌿 Leaf A")
        file_a = st.file_uploader("Upload Leaf A", type=["jpg","jpeg","png"], key="leafA")

    with col_b:
        st.markdown("#### 🌿 Leaf B")
        file_b = st.file_uploader("Upload Leaf B", type=["jpg","jpeg","png"], key="leafB")

    if file_a and file_b:
        if st.button("⚖️ Compare Both Leaves", use_container_width=True):
            if not os.path.exists("models/classifier.pkl"):
                st.error("⚠️ Model not found! Please run train.py first.")
                st.stop()
            clf, selected_features, class_names = load_models()

            results = []
            for label, f in [("Leaf A", file_a), ("Leaf B", file_b)]:
                img = Image.open(f).convert("RGB")
                arr = np.array(img)
                feat = extract_features_from_array(arr)
                pred, conf, proba = predict(clf, feat, selected_features, class_names)
                results.append((label, img, pred, conf))

            col1, col2 = st.columns(2)
            for col, (label, img, pred, conf) in zip([col1, col2], results):
                with col:
                    st.image(img, caption=label, width=250)
                    if "healthy" in pred.lower():
                        st.success(f"✅ {pred.replace('_',' ')}")
                    else:
                        st.error(f"⚠️ {pred.replace('_',' ')}")
                    st.metric("Confidence", f"{conf:.1f}%")
                    disease_info = get_disease_info(pred)
                    if disease_info:
                        st.caption(f"Severity: {disease_info['severity']}")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Plant Disease Detector &nbsp;·&nbsp; DEAP + Scikit-learn + Streamlit &nbsp;·&nbsp; 
    <a href="https://github.com/srilakshmi21206/plant-disease-ea" target="_blank" style="color:#2d6a4f">GitHub</a>
    &nbsp;·&nbsp; Srilakshmi.k &nbsp;·&nbsp; JAIN University 2026
</div>
""", unsafe_allow_html=True)