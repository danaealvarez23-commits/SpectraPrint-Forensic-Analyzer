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

# --- 1. FORMAL THEME (Black, White, Times New Roman) ---
st.set_page_config(page_title="SpectraPrint AI", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: "Times New Roman", Times, serif !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #444444;
    }
    h1, h2, h3, p, span, label, .stSelectbox, .stTextInput, .stNumberInput {
        color: #FFFFFF !important;
        font-family: "Times New Roman", Times, serif !important;
        font-size: 12pt !important;
    }
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 0px;
        font-weight: bold;
        font-family: "Times New Roman", Times, serif !important;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session States
if 'baseline_data' not in st.session_state: st.session_state.baseline_data = None
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'research_log' not in st.session_state: st.session_state.research_log = []

# --- 2. LOGO INTEGRATION ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    else:
        st.info("Upload 'logo.png' to GitHub.")
with col_title:
    st.title("SpectraPrint AI: Forensic Analysis Suite")

# --- 3. FORENSIC ENGINES ---

def apply_oval_mask(pil_img):
    """Converts a rectangular crop into a clean oval crop."""
    size = pil_img.size
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    output = ImageOps.fit(pil_img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    # Convert to White background for processing
    background = Image.new("L", size, 255)
    background.paste(output, (0, 0), output)
    return background

def identify_pattern(img):
    """Heuristic to guess the Fingerprint Pattern."""
    blur = cv2.GaussianBlur(img, (25, 25), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # This is a proxy for flow complexity
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 50: return "Whorl"
    if len(contours) > 20: return "Loop"
    return "Arch"

def classify_taxonomy(skel, x, y):
    """Classifies characteristics based on the provided reference chart."""
    block = skel[y-1:y+2, x-1:x+2].flatten().astype(float) / 255
    cn = 0.5 * np.sum(np.abs(block - np.roll(block, 1)))
    
    if cn == 1: return "Ending Ridge"
    if cn == 3: return "Bifurcation"
    if cn == 4: return "Crossover/Bridge"
    if cn == 0: return "Dot/Island"
    return "Specialty"

def process_forensic_executive(pil_img):
    # Apply the Oval Mask first
    oval_img = apply_oval_mask(pil_img)
    img_np = np.array(oval_img)
    
    # Enhancement
    smooth = cv2.bilateralFilter(img_np, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(smooth)
    inverted = cv2.bitwise_not(enhanced) 
    
    # Segmentation & Skeleton
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)
    skel = (skeletonize(binary//255) * 255).astype(np.uint8)
    
    # Pattern Recognition
    pattern_guess = identify_pattern(enhanced)
    
    # Feature Mapping
    feats = []
    inventory = {"Ending Ridge": 0, "Bifurcation": 0, "Bridge": 0, "Dot/Island": 0, "Core/Delta": 0}
    rows, cols = skel.shape
    char_map = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    temp_points = []
    for i in range(20, rows-20): 
        for j in range(20, cols-20):
            if skel[i,j] == 255:
                block = skel[i-1:i+2, j-1:j+2]
                cn = 0.5 * np.sum(np.abs(block.flatten().astype(float)/255 - np.roll(block.flatten().astype(float)/255, 1)))
                if cn in [0, 1, 3, 4]: temp_points.append((j, i))

    for p in temp_points:
        if all(math.hypot(p[0]-f['x'], p[1]-f['y']) > 15 for f in feats):
            m_type = classify_taxonomy(skel, p[0], p[1])
            feats.append({'x': p[0], 'y': p[1], 'type': m_type})
            if m_type in inventory: inventory[m_type] += 1
            # Solid Red Circles for Characterization (Set 2)
            cv2.circle(char_map, (p[0], p[1]), 6, (0, 0, 255), -1)
                    
    return {"enhanced": enhanced, "inverted": inverted, "char_map": char_map, 
            "feats": feats, "area": np.sum(binary > 0), "inventory": inventory, "pattern": pattern_guess}

# --- 4. SIDEBAR ---
st.sidebar.header("📋 Case Registration")
case_id = st.sidebar.text_input("Case Registration ID", "FORENSIC-2024-001")
examiner = st.sidebar.text_input("Chief Examiner")
manual_method = st.sidebar.selectbox("Collection Method", ["Powder", "Ninhydrin", "Cyanoacrylate", "Other"])
uv_val = st.sidebar.number_input("UV Absorbance Value", value=1.0)

# --- 5. MAIN APP ---
tabs = st.tabs(["📸 Acquisition & Oval Crop", "📊 Forensic Comparison", "📈 Decay Analysis", "📄 Export Report"])

with tabs[0]:
    st.subheader("I. Acquisition: Manual Oval Selection")
    up_file = st.file_uploader("Upload Fingerprint", type=['png','jpg','jpeg'])
    if up_file:
        img_pil = Image.open(up_file).convert('L')
        # Use the cropper to get the bounds, our code will turn it into an oval
        st.info("Select the fingerprint area. The app will automatically convert this to an Oval crop.")
        cropped_img = st_cropper(img_pil, realtime_update=True, box_color='#FFFFFF', aspect_ratio=None)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("LOCK BASELINE (DAY 0)"):
                st.session_state.baseline_data = process_forensic_executive(cropped_img)
        with c2:
            if st.button("LOCK CURRENT (AGED)"):
                st.session_state.current_data = process_forensic_executive(cropped_img)

    if st.session_state.baseline_data:
        st.divider()
        # SET 1: COLORS INVERTED
        st.subheader("Set 1: Human Comparison (Colors Inverted)")
        col_a, col_b = st.columns(2)
        col_a.image(st.session_state.baseline_data['inverted'], caption="Baseline Inverted")
        if st.session_state.current_data:
            col_b.image(st.session_state.current_data['inverted'], caption="Current Inverted")

        # SET 2: CHARACTERIZATION ONLY
        st.subheader("Set 2: Forensic Characterization (Markers Only)")
        col_c, col_d = st.columns(2)
        col_c.image(st.session_state.baseline_data['char_map'], caption="Baseline Characterization")
        if st.session_state.current_data:
            col_d.image(st.session_state.current_data['char_map'], caption="Current Characterization")

with tabs[1]:
    if st.session_state.baseline_data and st.session_state.current_data:
        b, c = st.session_state.baseline_data, st.session_state.current_data
        retention = (len(c['feats']) / len(b['feats'])) * 100 if len(b['feats']) > 0 else 0
        area_rem = (float(c['area']) / float(b['area'])) * 100
        
        st.header("II. Forensic Summary")
        st.write(f"**Identified Pattern:** {c['pattern']}")
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Minutiae Retention", f"{retention:.1f}%")
        k2.metric("Area Remaining", f"{area_rem:.1f}%")
        k3.metric("FQS Score", f"{(retention*0.6)+(area_rem*0.4):.1f}/100")
        
        st.subheader("Taxonomy Inventory")
        st.table(pd.DataFrame(list(c['inventory'].items()), columns=["Characteristic", "Count"]))

        if st.button("Save Data to Research Log"):
            st.session_state.research_log.append({"Date": datetime.datetime.now().strftime("%m/%d %H:%M"), "FQS": round((retention*0.6)+(area_rem*0.4), 2)})
            st.toast("Data points recorded.")
    else: st.warning("Acquire images in Tab 1.")

with tabs[2]:
    if st.session_state.research_log:
        df = pd.DataFrame(st.session_state.research_log)
        fig = px.line(df, x="Date", y="FQS", markers=True, template="plotly_dark")
        fig.update_layout(font_family="Times New Roman", font_size=12)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No research data found.")

with tabs[3]:
    if st.session_state.baseline_data and st.session_state.current_data:
        if st.button("🛠️ EXPORT MASTER PDF REPORT"):
            # Prepare Chart
            plt.figure(figsize=(5,3)); plt.style.use('dark_background')
            if st.session_state.research_log:
                plt.plot(pd.DataFrame(st.session_state.research_log)['Date'], pd.DataFrame(st.session_state.research_log)['FQS'], color='white', marker='s')
            plt.title("Degradation Decay Curve", fontname="Times New Roman")
            plt.savefig("chart.png"); plt.close()

            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Times", 'B', 16)
            pdf.cell(200, 10, "SPECTRAPRINT AI: MASTER FORENSIC REPORT", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font("Times", '', 12)
            pdf.cell(200, 10, f"Case ID: {case_id} | Examiner: {examiner}", ln=True)
            pdf.cell(200, 10, f"Method: {manual_method} | Pattern ID: {c['pattern']}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, "QUANTITATIVE FINDINGS:", ln=True)
            pdf.cell(200, 7, f" - Minutiae Retention: {retention:.2f}%", ln=True)
            pdf.cell(200, 7, f" - Ridge Surface Area: {area_rem:.2f}%", ln=True)
            pdf.ln(5)
            pdf.image("chart.png", x=50, w=110)
            
            # Temporary Image files for the report
            cv2.imwrite("base_inv.png", st.session_state.baseline_data['inverted'])
            cv2.imwrite("curr_char.png", st.session_state.current_data['char_map'])
            pdf.add_page()
            pdf.cell(200, 10, "VISUAL EVIDENCE LOG:", ln=True)
            pdf.image("base_inv.png", x=10, w=90)
            pdf.image("curr_char.png", x=110, w=90, y=pdf.get_y()-90)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📩 DOWNLOAD REPORT", pdf_bytes, f"Forensic_Report_{case_id}.pdf")