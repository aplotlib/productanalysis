import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import re
import os
from datetime import datetime

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="O.R.I.O.N. | VIVE Health",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VIVE BRAND THEME (STRICT ADHERENCE) ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    /* Montserrat (Simulates Redzone for Headers) & Open Sans (Body) */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Open+Sans:wght@400;600&display=swap');

    /* --- COLOR PALETTE ---
       Primary: #00C6D7 (Vive Teal)
       Secondary: #0B1E3D (Vive Deep Navy)
       Accent: #FFFFFF (White)
       Success: #00E676
       Warning: #FFEA00
       Error: #FF1744
    */

    /* GLOBAL RESET */
    .stApp {
        background-color: #0B1E3D !important; /* Vive Navy */
        /* Subtle Starfield Texture Overlay */
        background-image: 
            radial-gradient(white, rgba(255,255,255,.1) 1px, transparent 20px),
            radial-gradient(white, rgba(255,255,255,.05) 1px, transparent 20px);
        background-size: 350px 350px, 200px 200px; 
        background-position: 0 0, 40px 60px;
        color: #ffffff !important;
        font-family: 'Open Sans', sans-serif !important;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 2px; /* Matches Redzone extended feel */
        color: #ffffff !important;
        margin-bottom: 10px;
    }
    
    h1 {
        font-size: 3rem !important;
        background: linear-gradient(90deg, #00C6D7 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.1rem;
        color: #00C6D7; /* Vive Teal */
        letter-spacing: 1.5px;
        margin-bottom: 40px;
        border-bottom: 1px solid rgba(0, 198, 215, 0.3);
        padding-bottom: 10px;
        display: inline-block;
    }

    p, li, label, div, span {
        font-family: 'Open Sans', sans-serif !important;
        color: #e2e8f0; /* Off-white for body text comfort */
    }

    /* CONTAINERS (High Contrast Cards) */
    .stContainer, div[data-testid="metric-container"], .report-box {
        background-color: #132448 !important; /* Slightly lighter Navy */
        border: 1px solid rgba(0, 198, 215, 0.2); /* Teal Border */
        border-radius: 4px; /* Sharper corners for professional look */
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        padding: 24px;
    }

    /* METRICS */
    div[data-testid="metric-container"] {
        border-left: 4px solid #00C6D7;
        background: linear-gradient(145deg, #132448, #0e1b38);
    }
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif !important;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: white !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050E1F !important; /* Darkest Navy */
        border-right: 1px solid #1e293b;
    }
    
    /* BUTTONS (Vive Brand Primary) */
    .stButton>button {
        background: #00C6D7 !important; /* Brand Teal */
        color: #050E1F !important; /* Navy Text */
        border: none;
        border-radius: 4px;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        height: 3.5em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 0 #0097a7; /* 3D Effect */
    }
    .stButton>button:hover {
        background: #ffffff !important;
        color: #00C6D7 !important;
        box-shadow: 0 0 20px rgba(0, 198, 215, 0.6);
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: translateY(2px);
        box-shadow: none;
    }

    /* FORM INPUTS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea {
        background-color: #1B3B6F !important; /* Lighter Navy Input */
        color: white !important;
        border: 1px solid #475569;
        border-radius: 4px;
        font-family: 'Open Sans', sans-serif;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #00C6D7 !important;
        box-shadow: 0 0 0 1px #00C6D7;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #1e293b; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8 !important;
        border: none;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        color: #00C6D7 !important;
        border-bottom: 3px solid #00C6D7;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. INTELLIGENCE ENGINE (ROBUST INIT) ---
class IntelligenceEngine:
    def __init__(self):
        self.available = False
        self.model = None
        self.vision = None
        self._initialize_ai_clients()

    def _initialize_ai_clients(self):
        """Initialize AI clients from Streamlit secrets or Env"""
        try:
            import google.generativeai as genai
            self.genai = genai
            
            api_key = None
            # Priority: Secrets > Env
            if 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
            elif 'GOOGLE_API_KEY' in st.secrets:
                api_key = st.secrets['GOOGLE_API_KEY']
            
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY")
            
            if api_key:
                self.configure(api_key)
            else:
                self.available = False
                
        except Exception as e:
            st.sidebar.error(f"System Error: {e}")
            self.available = False

    def configure(self, key):
        try:
            self.genai.configure(api_key=key)
            self.model = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.available = True
        except Exception:
            self.available = False

    def generate(self, prompt):
        if not self.available: return "⚠️ Intelligence Offline."
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Analysis Error: {e}"

    def analyze_vision(self, image, prompt):
        if not self.available: return "⚠️ Intelligence Offline."
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Vision Error: {e}"

if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()

# --- 3. DATA PARSER ---
class DataParser:
    @staticmethod
    def normalize_columns(df):
        col_map = {
            'date': 'Date', 'created on': 'Date', 'order date': 'Date',
            'product': 'Product', 'sku': 'Product', 'product title': 'Product',
            'qty': 'Qty', 'quantity': 'Qty', 'returns': 'Returns', 'return qty': 'Returns',
            'sales': 'Sales', 'sold': 'Sales', 'order qty': 'Sales',
            'reason': 'Reason', 'return reason': 'Reason', 'disposition': 'Reason',
            'ticket': 'Ticket', 'ticket id': 'Ticket', 'stage': 'Stage', 'status': 'Stage'
        }
        df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
        return df

    @staticmethod
    def process_file(file):
        try:
            content = file.getvalue().decode("utf-8", errors='replace')
            lines = content.split('\n')
            keywords = ['product', 'sku', 'date', 'qty', 'sales', 'return', 'ticket', 'stage']
            best_idx = 0
            max_score = 0
            for i, line in enumerate(lines[:20]):
                score = sum(1 for k in keywords if k in line.lower())
                if score > max_score:
                    max_score = score
                    best_idx = i
            
            file.seek(0)
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=best_idx)
            else:
                df = pd.read_excel(file, header=best_idx)
            
            df = DataParser.normalize_columns(df)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')
            return df, None
        except Exception as e:
            return None, str(e)

# --- 4. UI MODULES ---

def render_dashboard():
    st.markdown("<h1>O.R.I.O.N.</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>OPERATIONAL REVIEW & INTELLIGENCE OPTIMIZATION NETWORK</div>", unsafe_allow_html=True)
    
    if 'orion_data' not in st.session_state: st.session_state.orion_data = None
    
    if st.session_state.orion_data is None:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info("📡 **SYSTEM STANDBY - AWAITING DATA STREAM**")
            st.markdown("Initialize Command Center by uploading operational data.")
            f = st.file_uploader("Upload Source (CSV/XLSX)", type=['csv', 'xlsx'])
            if f:
                df, err = DataParser.process_file(f)
                if err: st.error(err)
                else:
                    st.session_state.orion_data = df
                    st.success("SIGNAL LOCKED")
                    time.sleep(1)
                    st.rerun()
        with c2:
            st.markdown("### SYSTEM CAPABILITIES")
            st.markdown("• **Auto-Detect:** Returns / Sales / Tickets")
            st.markdown("• **Analysis:** Trends / Seasonality")
            st.markdown("• **Core:** AI Root Cause Hypothesis")
    else:
        df = st.session_state.orion_data
        sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        returns = df['Returns'].sum() if 'Returns' in df.columns else 0
        rate = (returns/sales*100) if sales > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("DATA RECORDS", f"{len(df):,}")
        m2.metric("SALES VOL", f"{sales:,.0f}" if sales else "--")
        m3.metric("RETURNS", f"{returns:,.0f}" if returns else "--")
        m4.metric("RETURN RATE", f"{rate:.2f}%" if sales else "N/A")
        
        st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📉 TREND ANALYSIS")
            if 'Date' in df.columns:
                df['Month'] = df['Date'].dt.to_period('M').astype(str)
                if 'Sales' in df.columns:
                    monthly = df.groupby('Month')[['Sales', 'Returns']].sum().reset_index()
                    monthly['Rate'] = (monthly['Returns'] / monthly['Sales']) * 100
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=monthly['Month'], y=monthly['Sales'], name='Sales', marker_color='#1B3B6F'))
                    fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Rate'], name='Return Rate', yaxis='y2', line=dict(color='#00C6D7', width=3)))
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis2=dict(title="Rate %", overlaying='y', side='right'), legend=dict(orientation="h", y=1.1),
                        font=dict(family="Open Sans")
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("### 🧠 AI ANALYST")
            if st.button("RUN DIAGNOSTICS", type="primary"):
                with st.spinner("Processing..."):
                    prompt = f"Analyze this data summary: {df.describe().to_string()}. Identify anomalies and suggest root causes."
                    st.info(st.session_state.ai.generate(prompt))
            
            if st.button("RESET SIGNAL"):
                st.session_state.orion_data = None
                st.rerun()

def render_voc():
    st.markdown("<h1>VISION INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>VISUAL DATA EXTRACTION MODULE</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📥 INPUT STREAM")
        img = st.file_uploader("Upload Screenshot", type=['png', 'jpg'])
        if img:
            st.image(img, caption="Target Asset", use_column_width=True)
            if st.button("INITIATE SCAN", type="primary"):
                with st.spinner("ANALYZING PIXEL DATA..."):
                    prompt = "Analyze screenshot. Extract: 1. Metrics (Stars, Rates) 2. Top Defects 3. Sentiment. 4. CAPA Recommendation."
                    st.session_state.voc_res = st.session_state.ai.analyze_vision(Image.open(img), prompt)
    
    with c2:
        st.markdown("### 📝 DECODED INTELLIGENCE")
        if 'voc_res' in st.session_state:
            st.success("SCAN COMPLETE")
            st.markdown(st.session_state.voc_res)
            st.divider()
            if st.button("🛡️ ELEVATE TO CAPA"):
                st.session_state.capa_prefill = st.session_state.voc_res
                st.session_state.nav = "CAPA"
                st.rerun()

def render_plan():
    st.markdown("<h1>STRATEGIC PLANNER</h1>", unsafe_allow_html=True)
    
    if 'qp_risk' not in st.session_state: st.session_state.qp_risk = "Class I"
    c1, c2 = st.columns(2)
    name = c1.text_input("PROJECT SKU / NAME", placeholder="EX: MOBILITY X1")
    risk = c2.selectbox("RISK CLASS", ["Class I", "Class II", "Class III"])
    st.session_state.qp_risk = risk
    st.divider()
    
    c_edit, c_view = st.columns([1.3, 1])
    sections = {"scope": "Scope & Objectives", "regs": "Regulatory Strategy", "test": "Validation Plan", "vend": "Vendor Controls"}
    
    with c_edit:
        st.markdown("### 🛠️ MODULES")
        locks = {}
        for key, label in sections.items():
            with st.expander(label, expanded=True):
                locks[key] = st.checkbox(f"LOCK SECTION", key=f"lock_{key}")
                st.session_state[f"qp_{key}"] = st.text_area("CONTENT", key=f"txt_{key}", height=100, label_visibility="collapsed")
        
        if st.button("✨ AI: COMPLIANCE OPTIMIZER", type="primary"):
            with st.spinner("OPTIMIZING..."):
                ctx = f"Project: {name}, Risk: {risk}"
                for k, label in sections.items():
                    if not locks[k]:
                        p = f"Write Quality Plan section '{label}'. Context: {ctx}. User Draft: {st.session_state[f'qp_{k}']}. No Markdown."
                        st.session_state[f"qp_{k}"] = st.session_state.ai.generate(p)
                        time.sleep(0.5)
                st.rerun()

    with c_view:
        st.markdown("### 📄 SUMMARY")
        full = f"STRATEGY: {name.upper()}\nRISK: {risk}\nDATE: {datetime.now().date()}\n\n"
        for k, label in sections.items():
            full += f"{label.upper()}\n{'-'*len(label)}\n{st.session_state.get(f'qp_{k}', '')}\n\n"
        st.text_area("PREVIEW", full, height=600)
        st.download_button("📥 DOWNLOAD", full, file_name="Strategy.txt")

def render_supply():
    st.markdown("<h1>SUPPLY CHAIN VALIDATOR</h1>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["SINGLE STREAM", "EFFECTIVENESS LOOP"])
    
    with t1:
        f = st.file_uploader("Upload Odoo Data", type=['csv', 'xlsx'], key="sc_up")
        if f:
            df, err = DataParser.process_file(f)
            if err: st.error(err)
            else:
                st.dataframe(df.head())
                if st.button("RISK AUDIT"):
                    st.info(st.session_state.ai.generate(f"Audit Supply Chain Data: {df.head(15).to_string()}"))

    with t2:
        st.markdown("### 🔄 BEFORE / AFTER VALIDATION")
        c1, c2 = st.columns(2)
        f1 = c1.file_uploader("BASELINE (BEFORE)", type=['csv', 'xlsx'], key="f1")
        f2 = c2.file_uploader("CURRENT (AFTER)", type=['csv', 'xlsx'], key="f2")
        if f1 and f2:
            df1, _ = DataParser.process_file(f1)
            df2, _ = DataParser.process_file(f2)
            if df1 is not None and df2 is not None:
                r1 = (df1['Returns'].sum() / df1['Sales'].sum() * 100) if 'Sales' in df1.columns else 0
                r2 = (df2['Returns'].sum() / df2['Sales'].sum() * 100) if 'Sales' in df2.columns else 0
                diff = r2 - r1
                st.markdown("#### RESULT")
                m1, m2, m3 = st.columns(3)
                m1.metric("BASELINE", f"{r1:.2f}%")
                m2.metric("CURRENT", f"{r2:.2f}%")
                m3.metric("CHANGE", f"{diff:.2f}%", delta_color="inverse")

def render_capa():
    st.markdown("<h1>CAPA MANAGER</h1>", unsafe_allow_html=True)
    prefill = st.session_state.get("capa_prefill", "")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.text_input("ID", f"CAPA-{int(time.time())}", disabled=True)
        st.text_area("DESCRIPTION", value=prefill, height=150)
    with c2:
        st.select_slider("SEVERITY", ["Minor", "Major", "Critical"])
        st.selectbox("OWNER", ["Quality", "Product", "Ops"])
        if st.button("🤖 AI RCA"):
            st.write(st.session_state.ai.generate("Suggest 3 Root Causes for generic defect."))

# --- 5. MAIN CONTROLLER ---
def main():
    with st.sidebar:
        st.title("O.R.I.O.N.")
        st.caption("VIVE HEALTH v6.0 | INTERNAL")
        
        if st.session_state.ai.available:
            st.success("🟢 ONLINE")
        else:
            st.error("🔴 OFFLINE")
            k = st.text_input("ACCESS KEY", type="password")
            if k: 
                st.session_state.ai.configure(k)
                st.rerun()
        
        st.markdown("---")
        if 'nav' not in st.session_state: st.session_state.nav = "DASHBOARD"
        
        opts = {"DASHBOARD": "📊", "VISION INTEL": "👁️", "STRATEGY": "📝", "SUPPLY CHAIN": "📦", "CAPA": "🛡️"}
        for label, icon in opts.items():
            if st.button(f"{icon}  {label}", use_container_width=True):
                st.session_state.nav = label
                st.rerun()

    if st.session_state.nav == "DASHBOARD": render_dashboard()
    elif st.session_state.nav == "VISION INTEL": render_voc()
    elif st.session_state.nav == "STRATEGY": render_plan()
    elif st.session_state.nav == "SUPPLY CHAIN": render_supply()
    elif st.session_state.nav == "CAPA": render_capa()

if __name__ == "__main__":
    main()
