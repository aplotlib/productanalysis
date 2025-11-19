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

# --- 1. SYSTEM CONFIGURATION & OPTIMIZATION ---
st.set_page_config(
    page_title="O.R.I.O.N. v6.3 | VIVE Health",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VIVE BRAND THEME (High Performance CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Open+Sans:wght@400;600&display=swap');

    /* GLOBAL PERFORMANCE & THEME */
    .stApp {
        background-color: #0B1E3D !important;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.08) 1px, transparent 20px),
            radial-gradient(white, rgba(255,255,255,.05) 1px, transparent 20px);
        background-size: 350px 350px, 200px 200px; 
        background-position: 0 0, 40px 60px;
        color: #ffffff !important;
        font-family: 'Open Sans', sans-serif;
    }

    /* HEADERS (Montserrat Redzone Style) */
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #ffffff !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #00C6D7 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
    }

    /* COMPONENT STYLING */
    .stContainer, div[data-testid="metric-container"], .report-box {
        background-color: #132448 !important;
        border: 1px solid rgba(0, 198, 215, 0.25);
        border-radius: 6px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        padding: 20px;
    }
    
    /* METRICS */
    div[data-testid="metric-container"] {
        border-left: 4px solid #00C6D7;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: #ffffff;
    }
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important;
        font-weight: 700;
        font-family: 'Montserrat', sans-serif !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 800;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050E1F !important;
        border-right: 1px solid #1e293b;
    }
    
    /* BUTTONS (High Speed Interaction) */
    .stButton>button {
        background: #00C6D7 !important;
        color: #050E1F !important;
        border: none;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        border-radius: 4px;
        height: 3.2em;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: #ffffff !important;
        color: #00C6D7 !important;
        box-shadow: 0 0 15px rgba(0, 198, 215, 0.5);
    }

    /* INPUTS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea {
        background-color: #1B3B6F !important;
        color: white !important;
        border: 1px solid #475569;
    }
    
    /* FOOTER STYLE */
    .footer-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.75rem;
        color: #5d6d8a;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. CACHED INTELLIGENCE ENGINE (SPEED OPTIMIZED) ---
class IntelligenceEngine:
    def __init__(self):
        self.available = False
        self.model = None
        self.vision = None
        self._initialize_clients()

    def _initialize_clients(self):
        # Check for Manual Key Override First (User Priority)
        if 'user_api_key' in st.session_state and st.session_state.user_api_key:
            self.configure(st.session_state.user_api_key)
            return

        # Auto-Discovery logic
        try:
            api_key = None
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
        except Exception:
            self.available = False

    def configure(self, key):
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.available = True
        except Exception as e:
            print(f"AI Init Error: {e}")
            self.available = False

    def generate(self, prompt):
        if not self.available: return "⚠️ AI Offline. Please enter Key in Sidebar."
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Error: {e}"

    def analyze_vision(self, image, prompt):
        if not self.available: return "⚠️ AI Offline. Please enter Key in Sidebar."
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Error: {e}"

# Initialize AI only once per run
if 'ai' not in st.session_state:
    st.session_state.ai = IntelligenceEngine()

# --- 3. CACHED DATA PARSER (SPEED OPTIMIZED) ---
class DataParser:
    @staticmethod
    @st.cache_data(show_spinner=False) # CACHING ENABLED FOR SPEED
    def process_file(file_content, file_name):
        """
        Processes file content. Arguments are passed as bytes/strings to allow caching.
        """
        try:
            # Convert bytes back to IO for reading
            file_io = io.BytesIO(file_content)
            
            # Read structure
            content_str = file_content.decode("utf-8", errors='replace')
            lines = content_str.split('\n')
            
            # Smart Header Detection
            keywords = ['product', 'sku', 'date', 'qty', 'sales', 'return', 'ticket', 'stage']
            best_idx = 0
            max_score = 0
            for i, line in enumerate(lines[:25]):
                score = sum(1 for k in keywords if k in line.lower())
                if score > max_score:
                    max_score = score
                    best_idx = i
            
            # Load Data
            file_io.seek(0)
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_io, header=best_idx)
            else:
                df = pd.read_excel(file_io, header=best_idx)
            
            # Normalize
            col_map = {
                'date': 'Date', 'created on': 'Date', 'order date': 'Date',
                'product': 'Product', 'sku': 'Product', 'product title': 'Product',
                'qty': 'Qty', 'quantity': 'Qty', 'returns': 'Returns', 'return qty': 'Returns',
                'sales': 'Sales', 'sold': 'Sales', 'order qty': 'Sales',
                'reason': 'Reason', 'return reason': 'Reason', 'disposition': 'Reason',
                'ticket': 'Ticket', 'ticket id': 'Ticket', 'stage': 'Stage', 'status': 'Stage'
            }
            df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')
            
            return df, None
        except Exception as e:
            return None, str(e)

# --- 4. UI MODULES ---

def render_dashboard():
    st.markdown("<h1>O.R.I.O.N. <span style='font-size:1rem; color:#00C6D7'>v6.3</span></h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#00C6D7; letter-spacing:2px; margin-bottom:20px; font-family:Montserrat'>OPERATIONAL REVIEW & INTELLIGENCE OPTIMIZATION NETWORK</div>", unsafe_allow_html=True)
    
    if 'orion_data' not in st.session_state: st.session_state.orion_data = None
    
    if st.session_state.orion_data is None:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.info("📡 **SYSTEM READY**")
            f = st.file_uploader("UPLOAD OPERATIONAL DATA (CSV/XLSX)", type=['csv', 'xlsx'])
            if f:
                # Pass bytes to cached function
                bytes_data = f.getvalue()
                with st.spinner("PARSING & INDEXING..."):
                    df, err = DataParser.process_file(bytes_data, f.name)
                    if err: st.error(err)
                    else:
                        st.session_state.orion_data = df
                        st.success("SIGNAL LOCKED")
                        time.sleep(0.5)
                        st.rerun()
        with c2:
            st.markdown("### CAPABILITIES")
            st.markdown("""
            * **Instant Parsing:** Odoo / Amazon / Shopify
            * **Auto-Trend:** Sales vs Returns
            * **Neural:** Root Cause Analysis
            """)
    else:
        df = st.session_state.orion_data
        sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        returns = df['Returns'].sum() if 'Returns' in df.columns else 0
        rate = (returns/sales*100) if sales > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RECORDS", f"{len(df):,}")
        m2.metric("SALES VOL", f"{sales:,.0f}" if sales else "--")
        m3.metric("RETURNS", f"{returns:,.0f}" if returns else "--")
        m4.metric("RETURN RATE", f"{rate:.2f}%" if sales else "N/A")
        
        st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📉 PERFORMANCE TREND")
            if 'Date' in df.columns:
                df['Month'] = df['Date'].dt.to_period('M').astype(str)
                if 'Sales' in df.columns:
                    monthly = df.groupby('Month')[['Sales', 'Returns']].sum().reset_index()
                    monthly['Rate'] = (monthly['Returns'] / monthly['Sales']) * 100
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=monthly['Month'], y=monthly['Sales'], name='Sales', marker_color='#1B3B6F'))
                    fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Rate'], name='Rate %', yaxis='y2', line=dict(color='#00C6D7', width=3)))
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis2=dict(overlaying='y', side='right'), legend=dict(orientation="h", y=1.1),
                        margin=dict(l=20, r=20, t=20, b=20), height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("### 🧠 NEURAL ANALYST")
            if st.button("RUN HYPOTHESIS", type="primary"):
                with st.spinner("ANALYZING PATTERNS..."):
                    summ = df.describe().to_string()
                    prompt = f"Analyze this data summary: {summ}. Find anomalies in Return Rate and suggest 1 Root Cause."
                    st.info(st.session_state.ai.generate(prompt))
            
            if st.button("RESET SIGNAL"):
                st.session_state.orion_data = None
                st.rerun()

def render_voc():
    st.markdown("<h1>VISION INTELLIGENCE</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📥 VISUAL INPUT")
        img = st.file_uploader("Upload Screenshot", type=['png', 'jpg'])
        if img:
            st.image(img, caption="Target", use_column_width=True)
            if st.button("INITIATE SCAN", type="primary"):
                with st.spinner("PROCESSING PIXELS..."):
                    prompt = "Extract: 1. Metrics 2. Top Defects 3. Sentiment 4. Recommendation."
                    st.session_state.voc_res = st.session_state.ai.analyze_vision(Image.open(img), prompt)
    
    with c2:
        st.markdown("### 📝 DECODED DATA")
        if 'voc_res' in st.session_state:
            st.success("SCAN COMPLETE")
            st.markdown(st.session_state.voc_res)
            st.divider()
            if st.button("🛡️ CREATE CAPA"):
                st.session_state.capa_prefill = st.session_state.voc_res
                st.session_state.nav = "CAPA"
                st.rerun()

def render_plan():
    st.markdown("<h1>STRATEGIC PLANNER</h1>", unsafe_allow_html=True)
    
    if 'qp_risk' not in st.session_state: st.session_state.qp_risk = "Class I"
    c1, c2 = st.columns(2)
    name = c1.text_input("PROJECT SKU", placeholder="EX: MOBILITY X1")
    risk = c2.selectbox("RISK CLASS", ["Class I", "Class II", "Class III"])
    st.session_state.qp_risk = risk
    st.divider()
    
    c_edit, c_view = st.columns([1.3, 1])
    sections = {"scope": "Scope", "regs": "Regulatory", "test": "Validation", "vend": "Vendor Controls"}
    
    with c_edit:
        st.markdown("### 🛠️ DRAFTING")
        locks = {}
        for key, label in sections.items():
            with st.expander(label, expanded=True):
                locks[key] = st.checkbox(f"LOCK", key=f"lock_{key}")
                st.session_state[f"qp_{key}"] = st.text_area("CONTENT", key=f"txt_{key}", height=100, label_visibility="collapsed")
        
        if st.button("✨ AI: OPTIMIZE", type="primary"):
            with st.spinner("OPTIMIZING..."):
                ctx = f"Project: {name}, Risk: {risk}"
                for k, label in sections.items():
                    if not locks[k]:
                        p = f"Write Quality Plan section '{label}'. Context: {ctx}. User Draft: {st.session_state[f'qp_{k}']}. No Markdown."
                        st.session_state[f"qp_{k}"] = st.session_state.ai.generate(p)
                st.rerun()

    with c_view:
        st.markdown("### 📄 PREVIEW")
        full = f"STRATEGY: {name.upper()}\nRISK: {risk}\nDATE: {datetime.now().date()}\n\n"
        for k, label in sections.items():
            full += f"{label.upper()}\n{'-'*len(label)}\n{st.session_state.get(f'qp_{k}', '')}\n\n"
        st.text_area("DOC", full, height=600)
        st.download_button("📥 DOWNLOAD", full, file_name="Strategy.txt")

def render_supply():
    st.markdown("<h1>SUPPLY CHAIN VALIDATOR</h1>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["AUDIT", "EFFECTIVENESS"])
    
    with t1:
        f = st.file_uploader("Upload Data", type=['csv', 'xlsx'], key="sc_up")
        if f:
            bytes_data = f.getvalue()
            df, err = DataParser.process_file(bytes_data, f.name)
            if err: st.error(err)
            else:
                st.dataframe(df.head())
                if st.button("AI AUDIT"):
                    st.info(st.session_state.ai.generate(f"Audit for risk: {df.head(15).to_string()}"))

    with t2:
        st.markdown("### 🔄 COMPARE")
        c1, c2 = st.columns(2)
        f1 = c1.file_uploader("BASELINE", type=['csv', 'xlsx'], key="f1")
        f2 = c2.file_uploader("CURRENT", type=['csv', 'xlsx'], key="f2")
        if f1 and f2:
            df1, _ = DataParser.process_file(f1.getvalue(), f1.name)
            df2, _ = DataParser.process_file(f2.getvalue(), f2.name)
            if df1 is not None and df2 is not None:
                r1 = (df1['Returns'].sum() / df1['Sales'].sum() * 100) if 'Sales' in df1.columns else 0
                r2 = (df2['Returns'].sum() / df2['Sales'].sum() * 100) if 'Sales' in df2.columns else 0
                diff = r2 - r1
                st.markdown("#### DELTA")
                m1, m2, m3 = st.columns(3)
                m1.metric("OLD", f"{r1:.2f}%")
                m2.metric("NEW", f"{r2:.2f}%")
                m3.metric("DIFF", f"{diff:.2f}%", delta_color="inverse")

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
            st.write(st.session_state.ai.generate("Root Cause Analysis suggestions."))

# --- 5. MAIN CONTROLLER ---
def main():
    with st.sidebar:
        st.title("O.R.I.O.N.")
        st.caption("VIVE HEALTH v6.3")
        
        # STATUS INDICATOR
        if st.session_state.ai.available:
            st.success("🟢 NEURAL ENGINE ONLINE")
        else:
            st.error("🔴 NEURAL ENGINE OFFLINE")
            key_input = st.text_input("ENTER API KEY", type="password", help="Gemini/Google Key")
            if key_input:
                st.session_state.user_api_key = key_input
                st.session_state.ai.configure(key_input)
                st.rerun()
        
        st.markdown("---")
        
        if 'nav' not in st.session_state: st.session_state.nav = "DASHBOARD"
        
        opts = {"DASHBOARD": "📊", "VISION INTEL": "👁️", "STRATEGY": "📝", "SUPPLY CHAIN": "📦", "CAPA": "🛡️"}
        for label, icon in opts.items():
            if st.button(f"{icon}  {label}", use_container_width=True):
                st.session_state.nav = label
                st.rerun()

        # --- REQUESTED FOOTER ---
        st.markdown("---")
        st.markdown("""
        <div class='footer-text'>
        built by alex popoff 11/19/2025<br>
        recent build v.6.3<br>
        gemini vibe coded beta test
        </div>
        """, unsafe_allow_html=True)

    # NAVIGATION
    if st.session_state.nav == "DASHBOARD": render_dashboard()
    elif st.session_state.nav == "VISION INTEL": render_voc()
    elif st.session_state.nav == "STRATEGY": render_plan()
    elif st.session_state.nav == "SUPPLY CHAIN": render_supply()
    elif st.session_state.nav == "CAPA": render_capa()

if __name__ == "__main__":
    main()
