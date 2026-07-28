import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from fpdf import FPDF
import plotly.express as px
import datetime
import io

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="SpectraPrint AI - Ultra Forensic", layout="wide")

if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. ENGINES (Alignment, ROI, Analysis) ---
def align_images(base_img, curr_img):
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(base_img, None)
    kp2, des2 = orb.detectAndCompute(curr_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    if len(matches) < 10: return curr_img 
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(curr_img, M, (base_img.shape[1], base_img.shape[0]))

def process_fingerprint(raw_img, crop_val):
    h, w = raw_img.shape
    c = int(crop_val * 0.01 * min(h, w))
    if c > 0: raw_img = raw_img[c:h-c, c:w-c]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(raw_img)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    feats = []
    rows, cols = skel.shape
    disp = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                block_sum = np.sum(block) / 255
                if block_sum == 2: 
                    feats.append((j, i, "Ending"))
                    cv2.circle(disp, (j, i), 5, (0, 255, 0), 1)
                elif block_sum == 4: 
                    feats.append((j, i, "Bifurcation"))
                    cv2.circle(disp, (j, i), 5, (255, 255, 0), 1)
    return {"enhanced": enhanced, "binary": binary, "skeleton": skel, "display": disp, "feats": feats, "raw": raw_img}

# --- 3. SIDEBAR: RESEARCH PARAMETERS ---
st.sidebar.title("📋 Case Registration")
case_id = st.sidebar.text_input("Case ID", "SAMPLE-01")
fp_id = st.sidebar.text_input("Fingerprint ID", "Index Finger")
examiner = st.sidebar.text_input("Examiner", "Enter Name")

st.sidebar.subheader("Substrate & Collection")
# NEW FIELDS REQUESTED
surf_type = st.sidebar.selectbox("Surface Type", [
    "Non-Porous (Glass/Metal/Plastic)", 
    "Porous (Paper/Cardboard)", 
    "Semi-Porous", 
    "Human Skin", 
    "Textile"
])
collection_method = st.sidebar.selectbox("Collection/Development Method", [
    "Powder Dusting (Black/Magnetic)", 
    "Ninhydrin", 
    "Cyanoacrylate Fuming (Superglue)", 
    "DFO / Indanedione", 
    "Vacuum Metal Deposition",
    "Silver Nitrate",
    "Physical Developer"
])
surf_area = st.sidebar.number_input("Total Surface Area (cm²)", value=10.0, step=0.1)

st.sidebar.subheader("Analysis Parameters")
uv_val = st.sidebar.number_input("UV Absorbance Value", value=1.0, step=0.1)
crop_factor = st.sidebar.slider("Crop Background Noise (%)", 0, 30, 5)

# --- 4. MAIN APP TABS ---
st.title("🔬 SpectraPrint AI: Ultra-Forensic Suite")
tabs = st.tabs(["📸 Acquisition", "📊 Analysis", "📈 Temporal Log", "📄 Report"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Baseline (Day 0)")
        b_file = st.file_uploader("Upload Baseline", type=['png','jpg','jpeg'], key="b_up")
        if b_file and st.button("Set Baseline"):
            img = cv2.imdecode(np.asarray(bytearray(b_file.read()), dtype=np.uint8), 0)
            st.session_state.baseline_data = process_fingerprint(img, crop_factor)
            st.success("Baseline Fixed!")
    with c2:
        st.subheader("Current (Degraded)")
        c_file = st.file_uploader("Upload Degraded", type=['png','jpg','jpeg'], key="c_up")
        if c_file and st.button("Analyze Current"):
            img = cv2.imdecode(np.asarray(bytearray(c_file.read()), dtype=np.uint8), 0)
            if st.session_state.baseline_data:
                img = align_images(st.session_state.baseline_data['raw'], img)
            st.session_state.current_data = process_fingerprint(img, crop_factor)
            st.success("Current Processed & Aligned!")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        cov = (np.sum(c['binary'] > 0) / np.sum(b['binary'] > 0)) * 100
        ret = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        pres = (np.sum(c['skeleton'] > 0) / np.sum(b['skeleton'] > 0)) * 100
        fqs = (0.2*cov) + (0.4*ret) + (0.4*pres) + (uv_val * 2)

        st.header("Quantitative Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Coverage %", f"{cov:.1f}%")
        m2.metric("Minutiae Retention", f"{ret:.1f}%")
        m3.metric("Ridge Preservation", f"{pres:.1f}%")
        m4.metric("FQS SCORE", f"{fqs:.1f}/100")
        
        if st.button("💾 Save to Research History"):
            log_entry = {"Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "FQS": round(fqs, 2), "Retention": round(ret, 2)}
            st.session_state.research_log.append(log_entry)
            st.toast("Data Logged!")

        st.image(c['display'], caption="Characterization Points Map")
    else: st.warning("Upload images in Acquisition tab.")

with tabs[2]:
    st.header("Research Decay Graph")
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="FQS", title="Quality Decay Curve", markers=True)
        st.plotly_chart(fig)
    else: st.info("Save results in Analysis tab to see graph.")

with tabs[3]:
    st.header("Forensic Export")
    if st.session_state.baseline_data and st.session_state.current_data:
        # UPDATED PDF REPORT LOGIC
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "SpectraPrint AI Forensic Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 7, f"Case ID: {case_id} | Examiner: {examiner}", ln=True)
        pdf.cell(200, 7, f"Collection Method: {collection_method}", ln=True)
        pdf.cell(200, 7, f"Surface Type: {surf_type} | Area: {surf_area} cm2", ln=True)
        pdf.cell(200, 7, f"Final Quality Score (FQS): {fqs:.2f}", ln=True)
        pdf.cell(200, 7, f"UV Signal Strength: {uv_val}", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📩 Download PDF Forensic Report", pdf_bytes, f"Report_{case_id}.pdf")
        
        if st.session_state.research_log:
            csv = pd.DataFrame(st.session_state.research_log).to_csv(index=False)
            st.download_button("📊 Export Full Research Log (Excel/CSV)", csv, "research_history.csv")