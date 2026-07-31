import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from streamlit_cropper import st_cropper
from fpdf import FPDF
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
import io
import math
import os
from PIL import Image

# --- 1. THEME & FONT CONFIGURATION (Black, Orange, Times New Roman) ---
st.set_page_config(page_title="Latent Visual Spectra", layout="wide")

st.markdown("""
    <style>
    /* Main Background and Font */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: black !important;
        color: #FF8C00 !important; /* Forensic Orange */
        font-family: "Times New Roman", Times, serif !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #FF8C00;
    }
    /* Headers and Text */
    h1, h2, h3, p, span, label {
        color: #FF8C00 !important;
        font-family: "Times New Roman", Times, serif !important;
    }
    /* Force 12pt font on labels and text */
    p, label, .stSelectbox, .stTextInput {
        font-size: 12pt !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #FF8C00 !important;
        color: black !important;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: white !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FF8C00 !important;
        border-bottom-color: #FF8C00 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session States
if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. HEADER & LOGO ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    else:
        st.info("Logo not found. Upload 'logo.png' to GitHub.")
with col_title:
    st.title("SpectraPrint AI: Professional Forensic Suite")

# --- 3. ENGINES ---
def classify_minutiae(skel, x, y):
    block = skel[y-1:y+2, x-1:x+2].flatten().astype(float) / 255
    cn = 0.5 * np.sum(np.abs(block - np.roll(block, 1)))
    if cn == 1: return "Ending Ridge"
    if cn == 3: return "Bifurcation"
    if cn == 4: return "Crossover"
    return "Specialty"

def process_forensic_gold(cropped_img):
    img = np.array(cropped_img.convert('L'))
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(smooth)
    inverted = cv2.bitwise_not(enhanced) 
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    feats = []
    inventory = {"Ending Ridge": 0, "Bifurcation": 0, "Crossover": 0}
    rows, cols = skel.shape
    char_map = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    temp_points = []
    for i in range(15, rows-15): 
        for j in range(15, cols-15):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                cn = 0.5 * np.sum(np.abs(block.flatten().astype(float)/255 - np.roll(block.flatten().astype(float)/255, 1)))
                if cn in [1, 3, 4]: temp_points.append((j, i))

    for p in temp_points:
        if all(math.hypot(p[0]-f['x'], p[1]-f['y']) > 15 for f in feats):
            m_type = classify_minutiae(skel, p[0], p[1])
            feats.append({'x': p[0], 'y': p[1], 'type': m_type})
            if m_type in inventory: inventory[m_type] += 1
            cv2.circle(char_map, (p[0], p[1]), 6, (0, 0, 255), -1)
                    
    return {"enhanced": enhanced, "inverted": inverted, "char_map": char_map, 
            "feats": feats, "area": np.sum(binary > 0), "inventory": inventory}

# --- 4. SIDEBAR: FULL FORENSIC REGISTRATION ---
st.sidebar.header("📋 Case Registration")
case_id = st.sidebar.text_input("Case ID", "FBI-MASTER-01")
examiner = st.sidebar.text_input("Lead Examiner")

st.sidebar.subheader("Substrate & Collection")
surface_type = st.sidebar.selectbox("Surface Type", 
    ["Glass", "Smooth Plastic", "Metal", "Paper/Cardboard", "Finished Wood", "Tape/Adhesive"])
collection_method = st.sidebar.selectbox("Collection Method", 
    ["Black Powder", "Magnetic Powder", "Fluorescent Powder", "Ninhydrin", "Cyanoacrylate", "DFO", "Indanedione", "Amido Black"])

classification = st.sidebar.selectbox("Pattern Classification", ["Loop - Ulnar", "Loop - Radial", "Whorl", "Arch", "Tented Arch"])
uv_abs = st.sidebar.number_input("UV Absorbance Value", value=1.0)

# --- 5. MAIN APP TABS ---
tabs = st.tabs(["📸 Acquisition", "📊 Analysis", "📈 Temporal Decay", "📄 Report"])

with tabs[0]:
    st.subheader("Interactive Manual Crop")
    up_file = st.file_uploader("Upload Fingerprint", type=['png','jpg','jpeg'])
    
    if up_file:
        img_pil = Image.open(up_file)
        # Manual Cropping Box
        cropped_img = st_cropper(img_pil, realtime_update=True, box_color='#FF8C00', aspect_ratio=None)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("LOCK BASELINE (DAY 0)"):
                st.session_state.baseline_data = process_forensic_gold(cropped_img)
        with c2:
            if st.button("LOCK CURRENT (AGED)"):
                st.session_state.current_data = process_forensic_gold(cropped_img)

    if st.session_state.baseline_data:
        st.divider()
        st.subheader("Set 1: Human Comparison (Inverted)")
        col_a, col_b = st.columns(2)
        col_a.image(st.session_state.baseline_data['inverted'], caption="Baseline Inverted")
        if st.session_state.current_data:
            col_b.image(st.session_state.current_data['inverted'], caption="Current Inverted")

        st.subheader("Set 2: Forensic Characterization (Inside Print)")
        col_c, col_d = st.columns(2)
        col_c.image(st.session_state.baseline_data['char_map'], caption="Baseline Characterized")
        if st.session_state.current_data:
            col_d.image(st.session_state.current_data['char_map'], caption="Current Characterized")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        retention = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        area_rem = (float(c['area']) / float(b['area'])) * 100
        fqs = (retention * 0.5) + (area_rem * 0.3) + (uv_abs * 20)
        
        st.header("Forensic Analysis Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Minutiae Retention", f"{retention:.1f}%")
        m2.metric("Area Remaining", f"{area_rem:.1f}%")
        m3.metric("FQS Score", f"{fqs:.1f}/100")
        
        st.subheader("Forensic Inventory")
        st.table(pd.DataFrame(list(c['inventory'].items()), columns=["Characteristic", "Count"]))

        if st.button("Save Entry to Research Log"):
            st.session_state.research_log.append({
                "Date": datetime.datetime.now().strftime("%m/%d %H:%M"), "FQS": round(fqs, 2)})
            st.toast("Data point saved to decay curve!")

with tabs[2]:
    st.header("Research Degradation Graph")
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="FQS", markers=True)
        fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='#FF8C00')
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Save log entries in the Analysis tab to generate graph.")

with tabs[3]:
    st.header("Export Master Forensic Report")
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ BUILD PDF REPORT"):
            plt.figure(figsize=(5,3)); plt.style.use('dark_background')
            if st.session_state.research_log:
                plt.plot(pd.DataFrame(st.session_state.research_log)['Date'], pd.DataFrame(st.session_state.research_log)['FQS'], color='orange', marker='o')
            plt.savefig("chart.png"); plt.close()

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Times", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI FORENSIC REPORT", ln=True, align='C')
            pdf.set_font("Times", '', 12)
            pdf.ln(5)
            pdf.cell(200, 7, f"Case: {case_id} | Examiner: {examiner}", ln=True)
            pdf.cell(200, 7, f"Surface: {surface_type} | Method: {collection_method}", ln=True)
            pdf.cell(200, 7, f"Classification: {classification} | FQS: {fqs:.2f}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, "DEGRADATION TREND ANALYSIS:", ln=True)
            pdf.image("chart.png", x=50, w=110)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📩 DOWNLOAD PDF", pdf_bytes, f"Report_{case_id}.pdf")
            if os.path.exists("chart.png"): os.remove("chart.png")