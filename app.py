import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage import exposure
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="SpectraPrint AI - Professional", layout="wide")

# This "Session State" allows the app to remember the Baseline print while you switch tabs
if 'baseline_data' not in st.session_state:
    st.session_state.baseline_data = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# --- 2. THE ANALYTICS ENGINE ---
def process_fingerprint(raw_img):
    # Enhancement Pipeline
    # 1. CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(raw_img)
    # 2. Denoising
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # 3. Adaptive Thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Auto-Inversion
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    # 4. Skeletonization
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # Feature Extraction (Minutiae)
    feats = []
    rows, cols = skel.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                block_sum = np.sum(block) / 255
                if block_sum == 2: feats.append((j, i, "Ending"))
                elif block_sum == 4: feats.append((j, i, "Bifurcation"))
                
    return {"enhanced": enhanced, "binary": binary, "skeleton": skel, "features": feats, "raw": raw_img}

# --- 3. SIDEBAR: CASE REGISTRATION ---
st.sidebar.header("📋 Case Registration")
case_id = st.sidebar.text_input("Case ID", "2024-EXP-001")
fp_id = st.sidebar.text_input("Fingerprint ID", "Right Index")
examiner = st.sidebar.text_input("Examiner", "Enter Name")
col_date = st.sidebar.date_input("Collection Date")
surface = st.sidebar.selectbox("Surface Type", ["Glass", "Metal", "Plastic", "Paper"])
temp = st.sidebar.slider("Temperature (°C)", 0, 50, 22)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 45)
notes = st.sidebar.text_area("Notes")

# --- 4. MAIN APP TABS ---
st.title("🔬 SpectraPrint AI: Forensic Analysis")
tab1, tab2, tab3 = st.tabs(["📸 Acquisition & Enhancement", "📊 Comparison Analysis", "📄 Forensic Report"])

# TAB 1: ACQUISITION
with tab1:
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Baseline (Day 0)")
        base_file = st.file_uploader("Upload Baseline", type=['png', 'jpg', 'jpeg'], key="u1")
        if base_file:
            img = cv2.imdecode(np.asarray(bytearray(base_file.read()), dtype=np.uint8), 0)
            if st.button("Process Baseline"):
                st.session_state.baseline_data = process_fingerprint(img)
                st.success("Baseline Processed!")

    with col_r:
        st.subheader("Current Sample")
        curr_file = st.file_uploader("Upload Degraded Print", type=['png', 'jpg', 'jpeg'], key="u2")
        # Added Camera Option
        cam_file = st.camera_input("OR Capture with Camera")
        
        active_curr = curr_file if curr_file else cam_file
        if active_curr:
            img = cv2.imdecode(np.asarray(bytearray(active_curr.read()), dtype=np.uint8), 0)
            if st.button("Process Current Sample"):
                st.session_state.current_data = process_fingerprint(img)
                st.success("Current Sample Processed!")

    if st.session_state.baseline_data:
        st.divider()
        st.write("### Preview of Features Detected")
        v1, v2 = st.columns(2)
        v1.image(st.session_state.baseline_data['skeleton'], caption="Baseline Skeleton")
        if st.session_state.current_data:
            v2.image(st.session_state.current_data['skeleton'], caption="Current Skeleton")

# TAB 2: COMPARISON
with tab2:
    if st.session_state.baseline_data and st.session_state.current_data:
        b = st.session_state.baseline_data
        c = st.session_state.current_data
        
        # Calculations
        coverage = (np.sum(c['binary'] > 0) / np.sum(b['binary'] > 0)) * 100
        retention = (len(c['features']) / len(b['features'])) * 100
        preservation = (np.sum(c['skeleton'] > 0) / np.sum(b['skeleton'] > 0)) * 100
        
        # FQS Score
        fqs = (0.3 * coverage) + (0.4 * retention) + (0.3 * preservation)
        
        # Metrics UI
        st.header("Quantitative Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Coverage %", f"{coverage:.1f}%")
        m2.metric("Minutiae Retention", f"{retention:.1f}%")
        m3.metric("Ridge Preservation", f"{preservation:.1f}%")
        m4.metric("FQS SCORE", f"{fqs:.1f}/100", delta=f"{fqs-100:.1f}")
        
        # Diff Map
        st.subheader("Degradation Difference Map")
        # Resize baseline to match current for subtraction
        b_bin_resized = cv2.resize(b['binary'], (c['binary'].shape[1], c['binary'].shape[0]))
        diff = cv2.absdiff(b_bin_resized, c['binary'])
        st.image(diff, caption="White areas = Missing ridge data", width=500)
    else:
        st.warning("Please process both images in the 'Acquisition' tab first.")

# TAB 3: REPORT
with tab3:
    st.header("Automated Report Generation")
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("Create PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SpectraPrint AI: Forensic Analysis Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Case ID: {case_id}", ln=True)
            pdf.cell(200, 10, f"Fingerprint: {fp_id}", ln=True)
            pdf.cell(200, 10, f"Final Quality Score: {fqs:.2f}/100", ln=True)
            pdf.cell(200, 10, f"Surface: {surface} | Temp: {temp}C", ln=True)
            pdf.output("Forensic_Report.pdf")
            st.success("Report generated! Check your folder for 'Forensic_Report.pdf'")
    else:
        st.info("Complete analysis to unlock reporting.")