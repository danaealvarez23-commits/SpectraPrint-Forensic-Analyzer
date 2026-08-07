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
from PIL import Image, ImageOps, ImageDraw

# --- 1. SYSTEM CONFIGURATION & EXECUTIVE THEME (Arial 12pt, Black & White) ---
st.set_page_config(page_title="SpectraPrint AI", layout="wide")

st.markdown("""
    <style>
    /* Main App Background */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: Arial, Helvetica, sans-serif !important;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #444444;
    }
    /* Force Arial 12pt on all text and labels */
    p, label, span, div, .stSelectbox, .stTextInput, .stNumberInput {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 12pt !important;
        color: #FFFFFF !important;
    }
    h1, h2, h3 { color: #FFFFFF !important; font-family: Arial !important; }
    
    /* BUTTON FIX: Solid White with Black Text */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: 1px solid #FFFFFF !important;
        width: 100%;
        border-radius: 4px;
        padding: 10px;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab"] { color: #888888 !important; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom: 2px solid #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session Memory
if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. HEADER & LOGO ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
with col_title:
    st.title("SpectraPrint AI: Executive Forensic Suite")

# --- 3. FORENSIC ENGINES ---

def apply_oval_mask(pil_img):
    """Turns the manual rectangular crop into a clean Oval."""
    size = pil_img.size
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    # Create white background
    white_bg = Image.new("L", size, 255)
    white_bg.paste(pil_img.convert('L'), (0, 0), mask)
    return white_bg

def process_forensic_pro(pil_img):
    # 1. Oval Masking
    oval_img = apply_oval_mask(pil_img)
    img_np = np.array(oval_img)
    
    # 2. Enhancement (CLAHE + Denoise)
    smooth = cv2.bilateralFilter(img_np, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(smooth)
    inverted = cv2.bitwise_not(enhanced) # SET 1
    
    # 3. Skeleton & Taxonomy
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # SET 2: Markers Only Map
    char_map = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    feats = []
    inventory = {"Ending": 0, "Bifurcation": 0, "Dot/Island": 0}
    
    rows, cols = skel.shape
    for i in range(15, rows-15): 
        for j in range(15, cols-15):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                cn = 0.5 * np.sum(np.abs(block.flatten().astype(float)/255 - np.roll(block.flatten().astype(float)/255, 1)))
                if cn in [0, 1, 3]:
                    if all(math.hypot(j-f[0], i-f[1]) > 15 for f in feats):
                        feats.append((j, i, cn))
                        if cn == 1: inventory["Ending"] += 1
                        elif cn == 3: inventory["Bifurcation"] += 1
                        else: inventory["Dot/Island"] += 1
                        # Draw Red Circle (Taxonomy Mark)
                        cv2.circle(char_map, (j, i), 6, (0, 0, 255), 1)

    return {"inverted": inverted, "char_map": char_map, "feats": feats, "area": np.sum(binary > 0), "inventory": inventory}

# --- 4. SIDEBAR: CASE REGISTRATION ---
st.sidebar.header("📋 Case Registration")
case_id = st.sidebar.text_input("Registration ID", "FOR-2024-01")
examiner = st.sidebar.text_input("Lead Examiner")

st.sidebar.subheader("Forensic Data")
pattern = st.sidebar.selectbox("Fingerprint Pattern", ["Loop - Ulnar", "Loop - Radial", "Whorl", "Arch", "Tented Arch"])
collection = st.sidebar.selectbox("Collection Method", ["Powder", "Ninhydrin", "Cyanoacrylate", "DFO", "Silver Nitrate"])
uv_val = st.sidebar.number_input("UV Absorbance Value", value=1.0)

# --- 5. MAIN INTERFACE ---
tabs = st.tabs(["📸 Acquisition", "📊 Analysis", "📈 Temporal Log", "📄 Master Report"])

with tabs[0]:
    st.subheader("Dual-Phase Fingerprint Acquisition")
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.write("**Phase 1: Baseline (Day 0)**")
        b_file = st.file_uploader("Upload Baseline Image", type=['png','jpg','jpeg'], key="b_up")
        if b_file:
            img_b = Image.open(b_file)
            st.write("Crop the baseline print (Oval auto-apply):")
            cropped_b = st_cropper(img_b, realtime_update=True, box_color='#FFFFFF', key="crop_b")
            if st.button("PROCESS BASELINE"):
                st.session_state.baseline_data = process_forensic_pro(cropped_b)
                st.success("Baseline Locked.")

    with col_r:
        st.write("**Phase 2: Current (Aged)**")
        c_file = st.file_uploader("Upload Current Sample", type=['png','jpg','jpeg'], key="c_up")
        if c_file:
            img_c = Image.open(c_file)
            st.write("Crop the current print (Oval auto-apply):")
            cropped_c = st_cropper(img_c, realtime_update=True, box_color='#FFFFFF', key="crop_c")
            if st.button("PROCESS CURRENT"):
                st.session_state.current_data = process_forensic_pro(cropped_c)
                st.success("Current Sample Locked.")

    if st.session_state.baseline_data:
        st.divider()
        st.subheader("Set 1: Human-View Inverted Comparison")
        r1 = st.columns(2)
        r1[0].image(st.session_state.baseline_data['inverted'], caption="Day 0: Inverted")
        if st.session_state.current_data:
            r1[1].image(st.session_state.current_data['inverted'], caption="Current: Inverted")

        st.subheader("Set 2: Characterization Map (Marks Only)")
        r2 = st.columns(2)
        r2[0].image(st.session_state.baseline_data['char_map'], caption="Day 0: Taxonomy Marks")
        if st.session_state.current_data:
            r2[1].image(st.session_state.current_data['char_map'], caption="Current: Taxonomy Marks")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        retention = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        area_rem = (float(c['area']) / float(b['area'])) * 100
        fqs = (retention * 0.6) + (area_rem * 0.4)
        
        st.header("Forensic Findings Summary")
        st.write(f"**Identified Pattern:** {pattern}")
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Minutiae Retention", f"{retention:.1f}%")
        k2.metric("Ridge Area Remaining", f"{area_rem:.1f}%")
        k3.metric("FQS Quality Score", f"{fqs:.1f}/100")
        
        st.subheader("Detected Characteristics Inventory")
        st.table(pd.DataFrame(list(c['inventory'].items()), columns=["Ridge Type", "Count"]))

        if st.button("SAVE ENTRY TO RESEARCH LOG"):
            st.session_state.research_log.append({"Date": datetime.datetime.now().strftime("%m/%d %H:%M"), "FQS": round(fqs, 2)})
            st.toast("Entry saved to decay graph.")
    else: st.warning("Please upload and process both images in Tab 1.")

with tabs[2]:
    st.header("Degradation Decay Analysis")
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="FQS", markers=True, template="plotly_dark")
        fig.update_layout(font_family="Arial", font_size=12, paper_bgcolor="black", plot_bgcolor="black")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No data logged yet.")

with tabs[3]:
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ GENERATE MASTER PDF REPORT"):
            plt.figure(figsize=(5,3)); plt.style.use('dark_background')
            if st.session_state.research_log:
                plt.plot(pd.DataFrame(st.session_state.research_log)['Date'], pd.DataFrame(st.session_state.research_log)['FQS'], color='white', marker='o')
            plt.savefig("decay_chart.png"); plt.close()

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI: MASTER FORENSIC REPORT", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 7, f"Case ID: {case_id} | Examiner: {examiner}", ln=True)
            pdf.cell(200, 7, f"Method: {collection} | Pattern: {pattern}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, "QUANTITATIVE SUMMARY", ln=True)
            pdf.cell(200, 7, f" - FQS Score: {fqs:.2f}/100", ln=True)
            pdf.cell(200, 7, f" - Minutiae Retention: {retention:.1f}%", ln=True)
            pdf.ln(10)
            pdf.image("decay_chart.png", x=50, w=110)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📩 DOWNLOAD FINAL REPORT", pdf_bytes, f"Report_{case_id}.pdf")