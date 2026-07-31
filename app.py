import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from fpdf import FPDF
import plotly.express as px
import datetime
import io
import math

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="SpectraPrint AI - Forensic Pro", layout="wide")

if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. FORENSIC ENGINES ---

def get_orientation(img, x, y):
    """Calculates the local ridge orientation (the 'tail' direction)."""
    if y < 5 or y > img.shape[0]-5 or x < 5 or x > img.shape[1]-5:
        return 0
    roi = img[y-5:y+5, x-5:x+5]
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    angle = math.atan2(np.sum(gy), np.sum(gx))
    return angle + math.pi/2

def process_forensic(raw_img, crop_val):
    # 1. ROI Cropping
    h, w = raw_img.shape
    c = int(crop_val * 0.01 * min(h, w))
    if c > 0: raw_img = raw_img[c:h-c, c:w-c]
    
    # 2. Enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(raw_img)
    inverted = cv2.bitwise_not(enhanced) 
    
    # 3. Binarization & Skeletonization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # 4. AFIS-Style Characterization
    feats = []
    rows, cols = skel.shape
    char_map = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    for i in range(2, rows-2):
        for j in range(2, cols-2):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                block_sum = np.sum(block) / 255
                if block_sum == 2 or block_sum == 4:
                    angle = get_orientation(enhanced, j, i)
                    feats.append((j, i, angle))
                    cv2.circle(char_map, (j, i), 6, (0, 0, 255), 1)
                    x2 = int(j + 12 * math.cos(angle))
                    y2 = int(i + 12 * math.sin(angle))
                    cv2.line(char_map, (j, i), (x2, y2), (0, 0, 255), 1)
                    
    return {
        "enhanced": enhanced, "inverted": inverted, 
        "skel": skel, "char_map": char_map, 
        "feats": feats, "area": np.sum(binary > 0),
        "binary": binary
    }

# --- 3. SIDEBAR: CASE REGISTRATION ---
st.sidebar.title("👮 Case Registration")
case_id = st.sidebar.text_input("Case Number", "FBI-2024-001")
examiner = st.sidebar.text_input("Lead Examiner", "Enter Name")
pattern_type = st.sidebar.selectbox("Ridge Pattern", ["Loop (Ulnar/Radial)", "Whorl (Plain/Double)", "Arch (Plain/Tented)", "Unknown"])
collection = st.sidebar.selectbox("Method", ["Black Powder", "Magnetic Powder", "Ninhydrin", "Cyanoacrylate"])
uv_input = st.sidebar.number_input("UV Signal (Absorbance)", value=1.0)
crop_factor = st.sidebar.slider("Background Noise Removal (%)", 0, 40, 10)

# --- 4. MAIN APP INTERFACE ---
st.title("🔬 SpectraPrint AI: Forensic Characterization Suite")

tabs = st.tabs(["📸 Image Sets", "📊 Degradation Analysis", "📈 Temporal Log", "📄 PDF Master Report"])

with tabs[0]:
    st.header("Dual-Set Acquisition")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Day 0: Baseline")
        b_file = st.file_uploader("Set Baseline Image", type=['png','jpg','jpeg'], key="b")
        if b_file and st.button("Process & Characterize Day 0"):
            img = cv2.imdecode(np.asarray(bytearray(b_file.read()), dtype=np.uint8), 0)
            st.session_state.baseline_data = process_forensic(img, crop_factor)
            st.success("Baseline Analyzed.")
            
    with c2:
        st.subheader("Current: Degraded")
        c_file = st.file_uploader("Set Current Image", type=['png','jpg','jpeg'], key="c")
        if c_file and st.button("Process & Characterize Current"):
            img = cv2.imdecode(np.asarray(bytearray(c_file.read()), dtype=np.uint8), 0)
            st.session_state.current_data = process_forensic(img, crop_factor)
            st.success("Current Sample Analyzed.")

    if st.session_state.baseline_data:
        st.divider()
        st.subheader("Visual Analysis Sets")
        row1 = st.columns(2)
        row1[0].image(st.session_state.baseline_data['inverted'], caption="Set A: Baseline (High-Contrast Inverted)")
        row1[1].image(st.session_state.baseline_data['char_map'], caption="Set B: Baseline (AFIS Characterization)")
        
        if st.session_state.current_data:
            row2 = st.columns(2)
            row2[0].image(st.session_state.current_data['inverted'], caption="Set A: Current (High-Contrast Inverted)")
            row2[1].image(st.session_state.current_data['char_map'], caption="Set B: Current (AFIS Characterization)")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        
        area_pct = (float(c['area']) / float(b['area'])) * 100
        minutiae_ret = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        fqs = (area_pct * 0.3) + (minutiae_ret * 0.5) + (uv_input * 10)
        
        st.header("Forensic Comparison Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Print Area Collected", f"{area_pct:.1f}%")
        k2.metric("Minutiae Retention", f"{minutiae_ret:.1f}%")
        k3.metric("Final Quality (FQS)", f"{fqs:.1f}/100")
        
        if st.button("💾 Save to Research Database"):
            st.session_state.research_log.append({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "FQS": round(fqs, 2), "Retention": round(minutiae_ret, 2), "Area": round(area_pct, 2)
            })
            st.toast("Data points stored.")
    else:
        st.warning("Please upload both image sets in Tab 1.")

with tabs[2]:
    st.header("Degradation Curve Log")
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="FQS", title="Degradation Trend Line", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.table(df)

with tabs[3]:
    st.header("Generate Master Forensic Report")
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ Build PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI: MASTER FORENSIC REPORT", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, f"Case ID: {case_id} | Lead Examiner: {examiner}", ln=True)
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 7, f"Pattern: {pattern_type} | Method: {collection}", ln=True)
            pdf.cell(200, 7, f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 8, f" - Area Collected (% of Baseline): {area_pct:.2f}%", ln=True)
            pdf.cell(200, 8, f" - Minutiae Retention: {minutiae_ret:.2f}%", ln=True)
            pdf.cell(200, 8, f" - FINAL QUALITY SCORE (FQS): {fqs:.2f}", ln=True)

            pdf_out = pdf.output(dest='S').encode('latin-1')
            st.download_button(label="📩 Download Completed PDF", data=pdf_out, file_name=f"SpectraPrint_Report_{case_id}.pdf", mime="application/pdf")
    else:
        st.info("Complete analysis in Tab 1 and 2 first.")