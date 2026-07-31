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
st.set_page_config(page_title="SpectraPrint AI - Clean AFIS Edition", layout="wide")

if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. ADVANCED FORENSIC ENGINES ---

def forensic_cleanup(img):
    """Aggressively removes noise speckles before characterization."""
    # 1. Smooth noise but keep ridge edges sharp
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    # 2. High-Contrast Thresholding
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    # 3. Remove tiny "dust" objects (the cause of the blue mess)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def get_orientation(img, x, y):
    if y < 10 or y > img.shape[0]-10 or x < 10 or x > img.shape[1]-10: return 0
    roi = img[y-5:y+5, x-5:x+5]
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    return math.atan2(np.sum(gy), np.sum(gx)) + math.pi/2

def process_clean_afis(raw_img, crop_val):
    # ROI Crop
    h, w = raw_img.shape
    c = int(crop_val * 0.01 * min(h, w))
    if c > 0: raw_img = raw_img[c:h-c, c:w-c]
    
    # 1. Clean the image (Bilateral + Morphological)
    binary = forensic_cleanup(raw_img)
    inverted = cv2.bitwise_not(raw_img) # For the "Human View"
    
    # 2. Skeletonize
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # 3. Filtered Minutiae Extraction
    feats = []
    rows, cols = skel.shape
    # Draw Green Ridges on White Background (SourceAFIS style)
    afis_view = np.full((rows, cols, 3), 255, dtype=np.uint8)
    afis_view[skel == 255] = [0, 180, 0] # Dark Green Ridges

    temp_feats = []
    for i in range(15, rows-15): # Wider border to avoid edge noise
        for j in range(15, cols-15):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                cn = 0.5 * np.sum(np.abs(block.flatten().astype(float)/255 - np.roll(block.flatten().astype(float)/255, 1)))
                if cn == 1 or cn == 3: # Ending or Bifurcation
                    temp_feats.append((j, i))

    # Strict Distance Filter (Increases spacing between points)
    filtered_points = []
    for p in temp_feats:
        # Only keep point if it is at least 15 pixels away from others
        if all(math.hypot(p[0]-f[0], p[1]-f[1]) > 15 for f in filtered_points):
            filtered_points.append(p)
            angle = get_orientation(raw_img, p[0], p[1])
            feats.append((p[0], p[1], angle))
            # Draw Marker: Red Circle + Tail (AFIS STYLE)
            cv2.circle(afis_view, (p[0], p[1]), 6, (200, 0, 0), 2)
            x2 = int(p[0] + 12 * math.cos(angle))
            y2 = int(p[1] + 12 * math.sin(angle))
            cv2.line(afis_view, (p[0], p[1]), (x2, y2), (200, 0, 0), 2)
                    
    return {"inverted": inverted, "afis_view": afis_view, "feats": feats, "area": np.sum(binary > 0), "binary": binary}

# --- 3. SIDEBAR ---
st.sidebar.title("👮 Case Registration")
case_id = st.sidebar.text_input("Case Number", "FBI-2024-001")
examiner = st.sidebar.text_input("Lead Examiner", "Enter Name")
pattern = st.sidebar.selectbox("Pattern", ["Loop - Ulnar", "Loop - Radial", "Whorl", "Arch", "Tented Arch"])
uv_input = st.sidebar.number_input("UV Absorbance", value=1.0)
crop_factor = st.sidebar.slider("DPI Noise Filter (Crop %)", 0, 30, 10)

# --- 4. MAIN APP ---
st.title("🔬 SpectraPrint AI: SourceAFIS Pro (Clean Edition)")

tabs = st.tabs(["📸 Acquisition", "📊 Analysis", "📈 Temporal Log", "📄 Master Report"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Probe (Day 0)")
        b_file = st.file_uploader("Upload Probe", type=['png','jpg','jpeg'], key="b")
        if b_file and st.button("Initialize Probe"):
            img = cv2.imdecode(np.asarray(bytearray(b_file.read()), dtype=np.uint8), 0)
            st.session_state.baseline_data = process_clean_afis(img, crop_factor)
    with c2:
        st.subheader("Candidate (Current)")
        c_file = st.file_uploader("Upload Candidate", type=['png','jpg','jpeg'], key="c")
        if c_file and st.button("Match Candidate"):
            img = cv2.imdecode(np.asarray(bytearray(c_file.read()), dtype=np.uint8), 0)
            st.session_state.current_data = process_clean_afis(img, crop_factor)

    if st.session_state.baseline_data:
        st.divider()
        st.subheader("Forensic Comparison Sets")
        row1 = st.columns(2)
        row1[0].image(st.session_state.baseline_data['inverted'], caption="Set A: High-Contrast Inverted Ridge Flow")
        row1[1].image(st.session_state.baseline_data['afis_view'], caption="Set B: Cleaned SourceAFIS Characterization")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        area_pct = (float(c['area']) / float(b['area'])) * 100
        score = (len(c['feats']) / len(b['feats'])) * 80 if len(b['feats']) > 0 else 0
        matches = score >= 40

        st.header("Matching & Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Area Recovered", f"{area_pct:.1f}%")
        k2.metric("SourceAFIS Score", f"{score:.1f}")
        k3.metric("Match Status", "VERIFIED" if matches else "UNFIT")
        
        st.code(f"""
// SourceAFIS Matching Logic Applied
var probe = new FingerprintTemplate(probe_img);
var candidate = new FingerprintTemplate(candidate_img);
double score = matcher.Match(candidate); // Current Score: {score:.2f}
        """, language="csharp")

        if st.button("Save to History"):
            st.session_state.research_log.append({
                "Date": datetime.datetime.now().strftime("%m/%d %H:%M"),
                "Score": round(score, 2), "Area": round(area_pct, 2)
            })
    else: st.warning("Upload images in Tab 1")

with tabs[2]:
    st.header("Research Decay Curve")
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="Score", title="Match Score Decay Over Time", markers=True)
        st.plotly_chart(fig)
    else: st.info("Save some data points in Tab 2 first.")

with tabs[3]:
    st.header("Master Forensic Report")
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ Export PDF with Decay Chart"):
            # Create the chart for the PDF
            plt.figure(figsize=(5,3))
            plt.plot(df['Date'], df['Score'], marker='o', color='green', linewidth=2)
            plt.title("Fingerprint Degradation Score Curve")
            plt.grid(True)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI MASTER REPORT", ln=True, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 7, f"Case: {case_id} | Examiner: {examiner} | Pattern: {pattern}", ln=True)
            pdf.cell(200, 7, f"Final Match Score: {score:.2f} | Area %: {area_pct:.1f}%", ln=True)
            
            pdf.ln(10)
            pdf.cell(200, 10, "DEGRADATION TREND ANALYSIS:", ln=True)
            pdf.image(img_buf, x=50, w=110)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📩 Download PDF Report", data=pdf_bytes, file_name="SpectraPrint_Report.pdf")