import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from fpdf import FPDF
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
import io
import math
import os

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Latent Vision Spectra", layout="wide")

if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. FORENSIC CHARACTERIZATION ENGINE ---

def get_ridge_mask(img):
    kernel = np.ones((15,15), np.uint8)
    mean, _ = cv2.meanStdDev(img)
    _, mask = cv2.threshold(cv2.GaussianBlur(img, (15,15), 0), mean[0][0], 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def classify_minutiae(skel, x, y):
    """Classifies the type of ridge characteristic based on pixel connectivity."""
    block = skel[y-1:y+2, x-1:x+2].flatten().astype(float) / 255
    cn = 0.5 * np.sum(np.abs(block - np.roll(block, 1)))
    
    if cn == 1: return "Ending Ridge"
    if cn == 3: return "Bifurcation"
    if cn == 4: return "Crossover/Bridge"
    if cn == 0: return "Dot/Island"
    return "Specialty"

def process_forensic_expert(raw_img, crop_box):
    # 1. Precision Cropping (Top, Bottom, Left, Right)
    t, b, l, r = crop_box
    h, w = raw_img.shape
    raw_img = raw_img[int(h*t/100):int(h*b/100), int(w*l/100):int(w*r/100)]
    
    # 2. Enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(raw_img)
    inverted = cv2.bitwise_not(enhanced) 
    mask = get_ridge_mask(enhanced)
    
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # 3. Forensic Taxonomy Extraction
    feats = []
    found_types = set()
    rows, cols = skel.shape
    char_map = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    for i in range(15, rows-15): 
        for j in range(15, cols-15):
            if skel[i,j] == 255 and mask[i,j] == 255:
                # Basic Crossing Number Logic
                block = skel[i-1:i+2, j-1:j+2]
                cn = 0.5 * np.sum(np.abs(block.flatten().astype(float)/255 - np.roll(block.flatten().astype(float)/255, 1)))
                
                if cn in [1, 3, 4]:
                    # Filter distance
                    if all(math.hypot(j-f['x'], i-f['y']) > 15 for f in feats):
                        m_type = classify_minutiae(skel, j, i)
                        feats.append({'x': j, 'y': i, 'type': m_type})
                        found_types.add(m_type)
                        # Draw Red Marker (Like Pic 1)
                        cv2.circle(char_map, (j, i), 6, (0, 0, 255), -1) # Solid Red Circle
                        
    return {"enhanced": enhanced, "inverted": inverted, "char_map": char_map, 
            "feats": feats, "area": np.sum(binary > 0), "taxonomy": list(found_types)}

# --- 3. SIDEBAR: CASE & PRECISION CROP ---
st.sidebar.title("Forensic Registration")
case_id = st.sidebar.text_input("Case ID", "FBI-MASTER-001")
examiner = st.sidebar.text_input("Lead Examiner")
classification = st.sidebar.selectbox("Classification", ["Loop - Ulnar", "Loop - Radial", "Whorl", "Arch", "Tented Arch"])

st.sidebar.subheader("Precision Crop Controls")
crop_top = st.sidebar.slider("Top (%)", 0, 50, 0)
crop_bot = st.sidebar.slider("Bottom (%)", 50, 100, 100)
crop_left = st.sidebar.slider("Left (%)", 0, 50, 0)
crop_right = st.sidebar.slider("Right (%)", 50, 100, 100)
crop_box = (crop_top, crop_bot, crop_left, crop_right)

uv_abs = st.sidebar.number_input("UV Absorbance Signal", value=1.0)

# --- 4. MAIN APP ---
st.title("Latent Vision Spectra")
tabs = st.tabs(["📸 Acquisition & Crop", "📊 Forensic Analysis", "📈 Temporal Decay", "📄 Master Report"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Day 0: Baseline")
        b_file = st.file_uploader("Upload Day 0", type=['png','jpg','jpeg'], key="b")
        if b_file and st.button("Process Day 0 with Crop"):
            img = cv2.imdecode(np.asarray(bytearray(b_file.read()), dtype=np.uint8), 0)
            st.session_state.baseline_data = process_forensic_expert(img, crop_box)
            st.success("Baseline Locked with Precision Crop")
    with c2:
        st.subheader("Current Sample")
        c_file = st.file_uploader("Upload Current", type=['png','jpg', 'jpeg'], key="c")
        if c_file and st.button("Analyze Current with Crop"):
            img = cv2.imdecode(np.asarray(bytearray(c_file.read()), dtype=np.uint8), 0)
            st.session_state.current_data = process_forensic_expert(img, crop_box)
            st.success("Current Sample Processed with Precision Crop")

    if st.session_state.baseline_data:
        st.divider()
        st.subheader("Set 1: Comparison (High-Contrast Inverted)")
        r1 = st.columns(2)
        r1[0].image(st.session_state.baseline_data['inverted'], caption="Baseline Inverted")
        if st.session_state.current_data:
            r1[1].image(st.session_state.current_data['inverted'], caption="Current Inverted")

        st.subheader("Set 2: Characterization (Forensic Markers)")
        r2 = st.columns(2)
        r2[0].image(st.session_state.baseline_data['char_map'], caption="Baseline Features")
        if st.session_state.current_data:
            r2[1].image(st.session_state.current_data['char_map'], caption="Current Features")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        retention = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        area_rem = (float(c['area']) / float(b['area'])) * 100
        fqs = (retention * 0.5) + (area_rem * 0.3) + (uv_abs * 20)
        
        st.header("Quantitative Evidence Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Taxonomy Retention", f"{retention:.1f}%")
        k2.metric("Surface Area Remaining", f"{area_rem:.1f}%")
        k3.metric("FQS Quality Score", f"{fqs:.1f}/100")
        
        st.subheader("Forensic Inventory (Detected Characteristics)")
        st.write(", ".join(c['taxonomy']))

        if st.button("Save Entry to Research Log"):
            st.session_state.research_log.append({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "FQS": round(fqs, 2), "Inventory": ", ".join(c['taxonomy'])
            })
    else: st.warning("Please process images in Acquisition Tab.")

with tabs[2]:
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="FQS", title="Degradation Decay Curve (FQS Score)", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Save log entries to see the graph.")

with tabs[3]:
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ Generate Final Master Report"):
            # Create Graph for PDF
            plt.figure(figsize=(5,3))
            if st.session_state.research_log:
                df_log = pd.DataFrame(st.session_state.research_log)
                plt.plot(df_log['Date'], df_log['FQS'], marker='o', color='red')
            plt.title("Fingerprint Decay Curve")
            chart_path = "temp_chart.png"
            plt.savefig(chart_path)
            plt.close()

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI: MASTER FORENSIC REPORT", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font("Arial", '', 11)
            pdf.cell(200, 7, f"Case ID: {case_id} | examiner: {examiner}", ln=True)
            pdf.cell(200, 7, f"Classification: {classification} | Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "1. QUANTITATIVE ANALYSIS", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.cell(200, 7, f" - Final Quality Score (FQS): {fqs:.2f}", ln=True)
            pdf.cell(200, 7, f" - Minutiae Retention: {retention:.2f}%", ln=True)
            pdf.cell(200, 7, f" - Ridge Surface Area Recovery: {area_rem:.2f}%", ln=True)
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "2. FORENSIC INVENTORY", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 7, f"The algorithm successfully identified the following ridge characteristics: {', '.join(c['taxonomy'])}")
            
            pdf.ln(5)
            pdf.cell(200, 10, "3. DEGRADATION DECAY GRAPH", ln=True)
            pdf.image(chart_path, x=50, w=110)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📩 Download Forensic Report", pdf_bytes, f"Report_{case_id}.pdf")
            if os.path.exists(chart_path): os.remove(chart_path)