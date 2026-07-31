import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from fpdf import FPDF
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
import io
import math

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="SpectraPrint AI - NBIS Forensic Pro", layout="wide")

if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. NIST/NBIS STYLE ENGINES ---

def get_ridge_mask(img):
    """Creates a mask to only detect minutiae INSIDE the fingerprint area."""
    # Calculate local variance to find ridge regions
    kernel = np.ones((15,15), np.uint8)
    # Standard deviation filter to find areas with high ridge activity
    mean, std = cv2.meanStdDev(img)
    _, mask = cv2.threshold(cv2.GaussianBlur(img, (15,15), 0), mean[0][0], 255, cv2.THRESH_BINARY_INV)
    # Clean the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def get_orientation(img, x, y):
    if y < 10 or y > img.shape[0]-10 or x < 10 or x > img.shape[1]-10: return 0
    roi = img[y-5:y+5, x-5:x+5]
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    return math.atan2(np.sum(gy), np.sum(gx)) + math.pi/2

def process_forensic_nbis(raw_img, crop_val):
    # ROI Crop
    h, w = raw_img.shape
    c = int(crop_val * 0.01 * min(h, w))
    if c > 0: raw_img = raw_img[c:h-c, c:w-c]
    
    # 1. Image Enhancement (FBI/NIST standard)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(raw_img)
    inverted = cv2.bitwise_not(enhanced) 
    
    # 2. Segmentation (Finding the fingerprint boundary)
    mask = get_ridge_mask(enhanced)
    
    # 3. Binarization & Skeletonization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # 4. Filtered Minutiae Extraction (ONLY inside mask)
    feats = []
    rows, cols = skel.shape
    # Set 2: Characterization Map (Drawn ON TOP of enhanced print)
    char_map = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    temp_feats = []
    for i in range(15, rows-15): 
        for j in range(15, cols-15):
            # CHECK: Only look where the ridge mask is white
            if skel[i,j] == 255 and mask[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                cn = 0.5 * np.sum(np.abs(block.flatten().astype(float)/255 - np.roll(block.flatten().astype(float)/255, 1)))
                if cn == 1 or cn == 3:
                    temp_feats.append((j, i))

    # Strict distance filter for clean AFIS look
    filtered_points = []
    for p in temp_feats:
        if all(math.hypot(p[0]-f[0], p[1]-f[1]) > 12 for f in filtered_points):
            filtered_points.append(p)
            angle = get_orientation(enhanced, p[0], p[1])
            feats.append((p[0], p[1], angle))
            # Draw Red Circles + Tails (NBIS STYLE)
            cv2.circle(char_map, (p[0], p[1]), 5, (0, 0, 255), 1)
            x2 = int(p[0] + 12 * math.cos(angle))
            y2 = int(p[1] + 12 * math.sin(angle))
            cv2.line(char_map, (p[0], p[1]), (x2, y2), (0, 0, 255), 1)
                    
    return {"enhanced": enhanced, "inverted": inverted, "char_map": char_map, "feats": feats, "area": np.sum(binary > 0)}

# --- 3. SIDEBAR ---
st.sidebar.title("👮 NIST/NBIS Case Registration")
case_id = st.sidebar.text_input("Case ID", "FBI-001")
examiner = st.sidebar.text_input("Lead Examiner")
classification = st.sidebar.selectbox("Classification", ["Loop - Ulnar", "Loop - Radial", "Whorl", "Arch"])
uv_abs = st.sidebar.number_input("UV Absorbance", value=1.0)
crop_factor = st.sidebar.slider("DPI Edge Crop %", 0, 30, 5)

# --- 4. MAIN APP ---
st.title("🔬 SpectraPrint AI: NIST/NBIS Analysis Suite")

tabs = st.tabs(["📸 Acquisition", "📊 Analysis", "📈 Temporal Log", "📄 Master PDF Report"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Day 0: Baseline (Probe)")
        b_file = st.file_uploader("Upload Day 0", type=['png','jpg','jpeg'], key="b")
        if b_file and st.button("Process Day 0"):
            img = cv2.imdecode(np.asarray(bytearray(b_file.read()), dtype=np.uint8), 0)
            st.session_state.baseline_data = process_forensic_nbis(img, crop_factor)
    with c2:
        st.subheader("Current Sample (Candidate)")
        c_file = st.file_uploader("Upload Current", type=['png','jpg','jpeg'], key="c")
        if c_file and st.button("Analyze Current"):
            img = cv2.imdecode(np.asarray(bytearray(c_file.read()), dtype=np.uint8), 0)
            st.session_state.current_data = process_forensic_nbis(img, crop_factor)

    if st.session_state.baseline_data:
        st.divider()
        # --- SET 1: COLORS INVERTED ---
        st.subheader("Set 1: High-Contrast Inverted Comparison")
        row1 = st.columns(2)
        row1[0].image(st.session_state.baseline_data['inverted'], caption="Day 0: Inverted")
        if st.session_state.current_data:
            row1[1].image(st.session_state.current_data['inverted'], caption="Current: Inverted")

        # --- SET 2: CHARACTERIZATION INSIDE PRINT ---
        st.subheader("Set 2: NIST/NBIS Characterization (Minutiae Tails)")
        row2 = st.columns(2)
        row2[0].image(st.session_state.baseline_data['char_map'], caption="Day 0: Characterized")
        if st.session_state.current_data:
            row2[1].image(st.session_state.current_data['char_map'], caption="Current: Characterized")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        score = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        area_rem = (float(c['area']) / float(b['area'])) * 100
        
        st.header("Forensic Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Minutiae Retention", f"{score:.1f}%")
        k2.metric("Area Recovery", f"{area_rem:.1f}%")
        k3.metric("FQS Score", f"{ (score*0.6) + (area_rem*0.4):.1f}")
        
        st.code("""
// NIST MINDTCT Algorithm logic:
FingerprintTemplate probe = new FingerprintTemplate(day0_img);
FingerprintTemplate candidate = new FingerprintTemplate(current_img);
double score = matcher.Match(candidate);
        """, language="java")

        if st.button("Save to Log"):
            st.session_state.research_log.append({
                "Date": datetime.datetime.now().strftime("%m/%d %H:%M"),
                "Score": round(score, 2), "Area": round(area_rem, 2)
            })
    else: st.warning("Upload images in Tab 1")

with tabs[2]:
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="Score", title="Degradation Decay Curve", markers=True)
        st.plotly_chart(fig)
    else: st.info("Save results to see the graph.")

with tabs[3]:
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ Export Final PDF Report"):
            # Render graph for PDF
            plt.figure(figsize=(5,3))
            plt.plot(df['Date'], df['Score'], marker='o', color='red')
            plt.title("Fingerprint Decay Chart")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI: MASTER FORENSIC REPORT", ln=True, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 7, f"Case: {case_id} | examiner: {examiner} | Class: {classification}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 7, "DEGRADATION DECAY CURVE:", ln=True)
            pdf.image(buf, x=50, w=110)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📩 Download PDF", pdf_bytes, "Forensic_Report.pdf")