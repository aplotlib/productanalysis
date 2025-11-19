import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import re
from datetime import datetime
import collections

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Product Lifecycle Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ROBUST IMPORT FOR AI ---
# This prevents the app from crashing immediately if the library isn't installed
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.error("⚠️ Module 'google.generativeai' not found. AI features will be disabled. Please ensure 'google-generativeai' is in your requirements.txt.")

# --- API SETUP ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 6px; height: 3em; font-weight: 600; width: 100%; }
    .report-box { border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; background: white; margin-bottom: 15px; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #fff; border-radius: 4px 4px 0px 0px; border: 1px solid #ddd; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: #e8f0fe; color: #1a73e8; border-bottom: 2px solid #1a73e8; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- AI ENGINE ---
class AIClient:
    def __init__(self):
        self.enabled = bool(api_key) and AI_AVAILABLE
        if self.enabled:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
                self.vision = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            except Exception as e:
                st.error(f"AI Configuration Error: {e}")
                self.enabled = False

    def generate(self, prompt, temperature=0.7):
        if not self.enabled: 
            if not AI_AVAILABLE: return "⚠️ AI Library Missing"
            return "⚠️ AI Key Missing"
        try:
            return self.model.generate_content(prompt, generation_config={"temperature": temperature}).text
        except Exception as e:
            return f"AI Error: {e}"

    def analyze_image(self, image, prompt):
        if not self.enabled:
            if not AI_AVAILABLE: return "⚠️ AI Library Missing"
            return "⚠️ AI Key Missing"
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Vision Error: {e}"

if 'ai' not in st.session_state:
    st.session_state.ai = AIClient()

# --- UTILITIES ---
def clean_text_for_export(text):
    """Removes markdown artifacts for raw text export."""
    if not text: return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'#+\s', '', text)              # Headers
    text = re.sub(r'__', '', text)                # Underscores
    text = re.sub(r'`', '', text)                 # Code ticks
    return text.strip()

def smart_odoo_parser(file):
    """Intelligently finds the header row in poorly structured Odoo CSV/Excel exports."""
    try:
        content = file.getvalue().decode("utf-8", errors='replace')
        lines = content.split('\n')
        
        keywords = ['id', 'date', 'product', 'sku', 'qty', 'quantity', 'status', 'name', 'reference', 'priority', 'stage', 'ticket']
        best_idx = 0
        max_score = 0
        
        for i, line in enumerate(lines[:20]):
            score = sum(1 for k in keywords if k in line.lower())
            if score > max_score:
                max_score = score
                best_idx = i
        
        file.seek(0)
        df = pd.read_csv(file, header=best_idx)
        df = df.dropna(how='all')
        return df, None
    except Exception as e:
        return None, str(e)

# --- MODULES ---

def render_dashboard():
    st.title("📊 Product Lifecycle Intelligence")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h3>4.2</h3><p>Avg Star Rating</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h3>12%</h3><p>Return Rate</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h3>5</h3><p>Open CAPAs</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h3>2</h3><p>Draft Plans</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("⚡ Active Workflows")
    col1, col2, col3 = st.columns(3)
    if col1.button("🛠️ New Product Plan"):
        st.session_state.nav = "Quality Planning"
        st.rerun()
    if col2.button("📢 Analyze Feedback"):
        st.session_state.nav = "Market Intelligence"
        st.rerun()
    if col3.button("🛡️ Log New CAPA"):
        st.session_state.nav = "CAPA Manager"
        st.rerun()

    st.markdown("### 📈 Trends")
    data = pd.DataFrame({
        "Date": pd.date_range(start="2025-01-01", periods=8, freq="W"),
        "Complaints": [12, 15, 8, 20, 10, 5, 12, 8],
        "CAPAs Closed": [2, 1, 4, 0, 3, 2, 1, 3]
    })
    fig = px.line(data, x="Date", y=["Complaints", "CAPAs Closed"], markers=True)
    st.plotly_chart(fig, use_container_width=True)

def render_quality_planning():
    st.title("🛠️ Quality Project Planner")
    
    defaults = {"qp_name": "", "qp_risk": "Class I", "qp_scope": "", "qp_regs": "", "qp_testing": "", "qp_vendor": "", "qp_path": ""}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    with st.expander("📝 Project Context", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        st.session_state.qp_name = c1.text_input("Project Name", st.session_state.qp_name)
        st.session_state.qp_risk = c2.selectbox("Risk Level", ["Class I", "Class II", "Class III"])
        mkts = c3.multiselect("Markets", ["USA", "EU", "UK", "Canada"])

    col_edit, col_view = st.columns([1.2, 1])
    sections = [
        ("scope", "Scope", "Deliverables & Boundaries"),
        ("regs", "Regulatory", "ISO/FDA Standards"),
        ("testing", "Testing Plan", "Verification & Validation"),
        ("vendor", "Vendor Controls", "Audits & IQC"),
        ("path", "Critical Path", "Timeline Milestones")
    ]
    
    with col_edit:
        st.subheader("Drafting")
        with st.form("qp_form"):
            locks = {}
            for code, title, hint in sections:
                st.markdown(f"**{title}**")
                locks[code] = st.checkbox(f"🔒 Lock {title}", key=f"lock_{code}")
                st.session_state[f"qp_{code}"] = st.text_area(hint, value=st.session_state[f"qp_{code}"], key=f"input_{code}", height=100)
                st.markdown("---")
            if st.form_submit_button("✨ Generate / Optimize"):
                with st.spinner("AI Working..."):
                    context = f"Project: {st.session_state.qp_name}, Risk: {st.session_state.qp_risk}, Markets: {mkts}.\n"
                    for code, _, _ in sections:
                        if st.session_state[f"qp_{code}"]: context += f"User Input ({code}): {st.session_state[f'qp_{code}']}\n"
                    
                    for code, title, _ in sections:
                        if not locks[code]:
                            val = st.session_state[f"qp_{code}"]
                            prompt = f"Write/Optimize section '{title}' for Quality Plan. Context: {context}. Current: '{val}'. No markdown."
                            st.session_state[f"qp_{code}"] = st.session_state.ai.generate(prompt)
                            time.sleep(0.5)
                    st.rerun()

    with col_view:
        st.subheader("Preview")
        full_doc = f"QUALITY PLAN: {st.session_state.qp_name}\nDATE: {datetime.now().date()}\n\n"
        for code, title, _ in sections:
            full_doc += f"{title.upper()}\n{'-'*len(title)}\n{clean_text_for_export(st.session_state.get(f'qp_{code}',''))}\n\n"
        st.text_area("Output", full_doc, height=600)
        st.download_button("📥 Download .txt", full_doc, file_name="Plan.txt")

def render_market_intel():
    st.title("🌐 Market Intelligence")
    t1, t2 = st.tabs(["Amazon VoC", "Odoo Data"])
    
    with t1:
        st.markdown("Upload screenshots for analysis.")
        img_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        if img_file:
            img = Image.open(img_file)
            st.image(img, width=400)
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    res = st.session_state.ai.analyze_image(img, "Analyze sentiment, defects, and key data points.")
                    st.session_state.voc_result = res
        
        if 'voc_result' in st.session_state:
            st.write(st.session_state.voc_result)
            q = st.text_input("Ask question about data:")
            if q: st.info(st.session_state.ai.generate(f"Context: {st.session_state.voc_result}. Q: {q}"))
            if st.button("Create CAPA"):
                st.session_state.capa_prefill = st.session_state.voc_result
                st.session_state.nav = "CAPA Manager"
                st.rerun()

    with t2:
        st.markdown("Upload Odoo CSV/XLSX (Auto-cleaning enabled).")
        f = st.file_uploader("Odoo File", type=['csv', 'xlsx'])
        if f:
            df, err = smart_odoo_parser(f) if f.name.endswith('.csv') else (pd.read_excel(f), None)
            if err: st.error(err)
            elif df is not None:
                st.dataframe(df.head())
                if st.button("AI Insights"):
                    st.write(st.session_state.ai.generate(f"Analyze this supply chain data: {df.head(20).to_string()}"))

def render_capa():
    st.title("🛡️ CAPA Manager")
    desc = st.session_state.get("capa_prefill", "")
    if desc: 
        st.info("Prefilled from VoC Analysis")
        del st.session_state.capa_prefill
    
    tabs = st.tabs(["Intake", "RCA", "Action", "Close"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.text_input("ID", f"CAPA-{int(time.time())}")
        c1.selectbox("Source", ["Amazon", "Internal", "Supplier"])
        c2.selectbox("Owner", ["Quality", "Ops"])
        st.text_area("Description", value=desc)
    
    with tabs[1]:
        if st.radio("Tool", ["5 Whys", "Fishbone"]) == "5 Whys":
            w1 = st.text_input("Why 1")
            w2 = st.text_input("Why 2")
            if w1 and w2 and st.button("AI Suggest Root Cause"):
                st.write(st.session_state.ai.generate(f"5 Whys: {w1}, {w2}... Root cause?"))

    with tabs[2]:
        st.checkbox("Correction")
        st.checkbox("Corrective Action")
        st.text_area("Plan")

    with tabs[3]:
        if st.button("Close CAPA"): st.success("Closed")

# --- MAIN ---
def main():
    if 'nav' not in st.session_state: st.session_state.nav = "Dashboard"
    with st.sidebar:
        st.title("PLI System")
        st.session_state.nav = st.radio("Menu", ["Dashboard", "Quality Planning", "Market Intelligence", "CAPA Manager"], index=["Dashboard", "Quality Planning", "Market Intelligence", "CAPA Manager"].index(st.session_state.nav))
    
    if st.session_state.nav == "Dashboard": render_dashboard()
    elif st.session_state.nav == "Quality Planning": render_quality_planning()
    elif st.session_state.nav == "Market Intelligence": render_market_intel()
    elif st.session_state.nav == "CAPA Manager": render_capa()

if __name__ == "__main__":
    main()
