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

# --- O.R.I.O.N. THEME (High Contrast / Leadership Ready) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');

    /* CORE PALETTE: VIVE TEAL (#00C6D7), DEEP NAVY (#020408), WHITE */
    
    /* BACKGROUND: Subtle Moving Stars */
    @keyframes move-twink-back {
        from {background-position:0 0;}
        to {background-position:-10000px 5000px;}
    }
    .stApp {
        background-color: #020408 !important;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px);
        background-size: 550px 550px, 350px 350px, 250px 250px; 
        color: #ffffff !important;
    }

    /* TYPOGRAPHY: Montserrat - Crisp & Readable */
    h1, h2, h3, h4, p, div, span, li, label {
        font-family: 'Montserrat', sans-serif !important;
        color: white !important;
    }
    
    h1 {
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: linear-gradient(90deg, #00C6D7 0%, #ffffff 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8 !important;
        letter-spacing: 1px;
        margin-bottom: 30px;
    }

    /* CARDS: High Contrast - No Transparency Issues */
    .stContainer, .metric-box, .report-box, div[data-testid="metric-container"] {
        background-color: #0B1221 !important; /* Solid Dark Navy */
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }

    /* METRICS: Big & Bold */
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: white !important;
    }

    /* SIDEBAR: Minimalist */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* BUTTONS: Action Oriented */
    .stButton>button {
        background: #00C6D7 !important;
        color: #000000 !important;
        border: none;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 4px;
        height: 3.5em;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: white !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 198, 215, 0.4);
    }

    /* INPUTS: Readable Dark Mode */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea {
        background-color: #161f30 !important;
        color: white !important;
        border: 1px solid #334155;
        border-radius: 4px;
    }
    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. AI & INTELLIGENCE LAYER (ROBUST INITIALIZATION) ---
class IntelligenceEngine:
    def __init__(self):
        self.available = False
        self.model = None
        self.vision = None
        self._initialize_ai_clients()

    def _initialize_ai_clients(self):
        """Initialize AI clients from Streamlit secrets (Requested Pattern)"""
        try:
            import google.generativeai as genai
            self.genai = genai
            
            api_key = None
            # Check Streamlit Secrets for preferred keys
            if 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
            elif 'GOOGLE_API_KEY' in st.secrets:
                api_key = st.secrets['GOOGLE_API_KEY']
            
            # Fallback to Environment Variable
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY")
            
            if api_key:
                self.configure(api_key)
            else:
                # Fail silently here, UI will show "Offline" status
                self.available = False
                
        except Exception as e:
            st.error(f"Error initializing AI clients: {e}")
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
        if not self.available: return "⚠️ Neural Engine Offline. Check API Key."
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Error: {e}"

    def analyze_vision(self, image, prompt):
        if not self.available: return "⚠️ Neural Engine Offline."
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Error: {e}"

if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()

# --- 3. DATA PARSER (ODOO SPECIALIST) ---
class DataParser:
    @staticmethod
    def normalize_columns(df):
        """Standardizes messy Odoo/Excel headers to VIVE Internal Names."""
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
        """
        Smart-scans the file to find the actual header row, 
        skipping metadata often found in Odoo exports.
        """
        try:
            # Read raw bytes to sniff structure
            content = file.getvalue().decode("utf-8", errors='replace')
            lines = content.split('\n')
            
            # Scoring system to find the best header row
            keywords = ['product', 'sku', 'date', 'qty', 'sales', 'return', 'ticket', 'stage']
            best_idx = 0
            max_score = 0
            
            for i, line in enumerate(lines[:20]):
                score = sum(1 for k in keywords if k in line.lower())
                if score > max_score:
                    max_score = score
                    best_idx = i
            
            # Parse
            file.seek(0)
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=best_idx)
            else:
                df = pd.read_excel(file, header=best_idx)
            
            df = DataParser.normalize_columns(df)
            
            # Type Handling
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']) # Drop rows without dates if date column exists
                df = df.sort_values('Date')
                
            return df, None
        except Exception as e:
            return None, str(e)

# --- 4. UI SECTIONS ---

def render_dashboard():
    st.markdown("<h1>O.R.I.O.N.</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Operational Review & Intelligence Optimization Network</div>", unsafe_allow_html=True)
    
    # --- STATE: DATA HOLDER ---
    if 'orion_data' not in st.session_state: st.session_state.orion_data = None
    
    # --- A: EMPTY STATE (WAITING FOR SIGNAL) ---
    if st.session_state.orion_data is None:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info("📡 **Awaiting Data Signal**")
            st.markdown("Upload Raw Odoo Exports or Sales/Return Logs to activate the Command Center.")
            
            f = st.file_uploader("Select Data Source (CSV/XLSX)", type=['csv', 'xlsx'])
            if f:
                df, err = DataParser.process_file(f)
                if err:
                    st.error(f"Parsing Error: {err}")
                else:
                    st.session_state.orion_data = df
                    st.success(f"Signal Acquired: {len(df)} Records")
                    time.sleep(1)
                    st.rerun()
        with c2:
            st.markdown("### Capabilities")
            st.markdown("""
            * **Auto-Detection:** Returns, Sales, Tickets
            * **Seasonality:** Monthly Trend Analysis
            * **Reason Shifts:** Defect Pattern Recognition
            * **AI Analyst:** Root Cause Hypothesis
            """)

    # --- B: ACTIVE COMMAND CENTER ---
    else:
        df = st.session_state.orion_data
        
        # Metric Logic
        sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        returns = df['Returns'].sum() if 'Returns' in df.columns else 0
        rate = (returns/sales*100) if sales > 0 else 0
        
        # 1. TOP LEVEL METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records Parsed", f"{len(df):,}")
        m2.metric("Total Volume", f"{sales:,.0f}" if sales else "--")
        m3.metric("Returns Detected", f"{returns:,.0f}" if returns else "--")
        m4.metric("Return Rate", f"{rate:.2f}%" if sales else "N/A")
        
        st.markdown("---")
        
        # 2. VISUAL INTELLIGENCE
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("### 📉 Trend Analysis")
            if 'Date' in df.columns:
                # Monthly Aggregation
                df['Month'] = df['Date'].dt.to_period('M').astype(str)
                monthly = df.groupby('Month')[['Sales', 'Returns']].sum().reset_index() if 'Sales' in df.columns else df.groupby('Month').size().reset_index(name='Count')
                
                fig = go.Figure()
                
                if 'Sales' in df.columns:
                    monthly['Rate'] = (monthly['Returns'] / monthly['Sales']) * 100
                    fig.add_trace(go.Bar(x=monthly['Month'], y=monthly['Sales'], name='Sales', marker_color='#1e293b'))
                    fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Rate'], name='Return Rate %', yaxis='y2', line=dict(color='#00C6D7', width=4)))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=400,
                        yaxis2=dict(title="Rate %", overlaying='y', side='right'),
                        legend=dict(orientation="h", y=1.1)
                    )
                else:
                    # Generic Time Series
                    col = 'Count' if 'Count' in monthly.columns else df.columns[0]
                    fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly[col], line=dict(color='#00C6D7')))
                    
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No Date column detected for Time Series analysis.")

        with c2:
            st.markdown("### 🧠 AI Context Analyst")
            if st.button("RUN HYPOTHESIS ENGINE", type="primary"):
                with st.spinner("Analyzing seasonality and defect vectors..."):
                    summ = df.describe(include='all').to_string()
                    prompt = f"""
                    You are the O.R.I.O.N. AI. Analyze this product performance data.
                    Summary: {summ}
                    
                    1. Detect shifts in patterns (Seasonality? Spikes?).
                    2. Hypothesize a Root Cause (e.g. "Did vendor specs change in Q3?").
                    3. Suggest 1 actionable move for Leadership.
                    """
                    analysis = st.session_state.ai.generate(prompt)
                    st.info(analysis)
            
            st.markdown("### Actions")
            if st.button("🗑️ Reset Signal"):
                st.session_state.orion_data = None
                st.rerun()

def render_voc():
    st.markdown("<h1>Vision Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Screenshot & Review Parser</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📥 Upload Evidence")
        st.caption("Amazon Dashboards, Customer Photos, Return Reports")
        img = st.file_uploader("Select Image", type=['png', 'jpg'])
        
        if img:
            st.image(img, use_column_width=True)
            if st.button("SCAN & EXTRACT", type="primary"):
                with st.spinner("Extracting Text & Metrics..."):
                    prompt = """
                    Analyze this screenshot for Quality Leadership.
                    1. Extract key metrics (Star Rating, NCX, Return Rate).
                    2. Identify the top 3 specific defect keywords.
                    3. Summarize the 'Voice of Customer' sentiment.
                    """
                    st.session_state.voc_res = st.session_state.ai.analyze_vision(Image.open(img), prompt)
    
    with c2:
        st.markdown("### 📝 Extraction Results")
        if 'voc_res' in st.session_state:
            st.success("Intelligence Extracted")
            st.markdown(st.session_state.voc_res)
            
            st.divider()
            if st.button("🛡️ Initiate CAPA from Findings"):
                st.session_state.capa_prefill = st.session_state.voc_res
                st.session_state.nav = "CAPA"
                st.rerun()
        else:
            st.info("Waiting for scan...")

def render_plan():
    st.markdown("<h1>Strategic Planner</h1>", unsafe_allow_html=True)
    
    # State Init
    if 'qp_risk' not in st.session_state: st.session_state.qp_risk = "Class I"
    
    with st.container():
        c1, c2 = st.columns(2)
        name = c1.text_input("Project / SKU Name", placeholder="Ex: Bariatric Rollator Gen 2")
        risk = c2.selectbox("Risk Classification", ["Class I", "Class II", "Class III"])
        st.session_state.qp_risk = risk
    
    st.divider()
    
    c_edit, c_view = st.columns([1.3, 1])
    
    sections = {
        "scope": "Scope & Objectives",
        "regs": "Regulatory Strategy",
        "test": "Validation Plan",
        "vend": "Vendor Controls"
    }
    
    with c_edit:
        st.markdown("### 🛠️ Strategy Modules")
        locks = {}
        for key, label in sections.items():
            with st.expander(label, expanded=True):
                lock = st.checkbox(f"Lock {label}", key=f"lock_{key}")
                locks[key] = lock
                st.session_state[f"qp_{key}"] = st.text_area("Draft", key=f"txt_{key}", height=100, label_visibility="collapsed")
        
        if st.button("✨ AI: COMPLIANCE OPTIMIZATION", type="primary"):
            with st.spinner("Optimizing against ISO/FDA standards..."):
                ctx = f"Project: {name}, Risk: {risk}"
                for k, label in sections.items():
                    if not locks[k]:
                        draft = st.session_state[f"qp_{k}"]
                        p = f"Write professional Quality Plan section '{label}'. Context: {ctx}. User Draft: {draft}. No Markdown."
                        st.session_state[f"qp_{k}"] = st.session_state.ai.generate(p)
                        time.sleep(0.5)
                st.rerun()

    with c_view:
        st.markdown("### 📄 Executive Summary")
        full = f"QUALITY STRATEGY: {name.upper()}\nRISK: {risk}\nDATE: {datetime.now().date()}\n\n"
        for k, label in sections.items():
            full += f"{label.upper()}\n{'-'*len(label)}\n{st.session_state.get(f'qp_{k}', '')}\n\n"
        st.text_area("Preview", full, height=600)
        st.download_button("📥 Download Strategy", full, file_name="Strategy.txt")

def render_supply():
    st.markdown("<h1>Supply Chain Validator</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Single Stream Analysis", "⚖️ Compare / Effectiveness Loop"])
    
    with tab1:
        f = st.file_uploader("Upload Odoo Export", type=['csv', 'xlsx'], key="sc_up")
        if f:
            df, err = DataParser.process_file(f)
            if err: st.error(err)
            else:
                st.success(f"Loaded {len(df)} rows")
                st.dataframe(df.head())
                if st.button("Run Risk Audit"):
                    st.info(st.session_state.ai.generate(f"Audit this supply chain data for risks (stockouts/overstock): {df.head(15).to_string()}"))

    with tab2:
        st.markdown("### 🔄 Effectiveness Check")
        st.markdown("Upload **Before** and **After** datasets to validate Quality Initiatives.")
        
        c1, c2 = st.columns(2)
        f1 = c1.file_uploader("Baseline (Before)", type=['csv', 'xlsx'], key="f1")
        f2 = c2.file_uploader("Current (After)", type=['csv', 'xlsx'], key="f2")
        
        if f1 and f2:
            df1, _ = DataParser.process_file(f1)
            df2, _ = DataParser.process_file(f2)
            
            if df1 is not None and df2 is not None:
                # Simple Return Rate Comparison Logic
                r1 = (df1['Returns'].sum() / df1['Sales'].sum() * 100) if 'Sales' in df1.columns else 0
                r2 = (df2['Returns'].sum() / df2['Sales'].sum() * 100) if 'Sales' in df2.columns else 0
                
                diff = r2 - r1
                color = "#00ff00" if diff < 0 else "#ff0000"
                
                st.markdown("#### Impact Result")
                m1, m2, m3 = st.columns(3)
                m1.metric("Baseline Rate", f"{r1:.2f}%")
                m2.metric("Current Rate", f"{r2:.2f}%")
                m3.metric("Net Change", f"{diff:.2f}%", delta_color="inverse")
                
                if diff < 0:
                    st.success("✅ IMPROVEMENT CONFIRMED: Quality Initiative Effective.")
                else:
                    st.error("⚠️ NO IMPROVEMENT: Re-evaluate Strategy.")

def render_capa():
    st.markdown("<h1>CAPA Manager</h1>", unsafe_allow_html=True)
    
    prefill = st.session_state.get("capa_prefill", "")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Incident Details")
        st.text_input("CAPA ID", f"CAPA-{int(time.time())}", disabled=True)
        st.text_area("Description / Non-Conformance", value=prefill, height=150)
        
    with c2:
        st.markdown("### Risk & Owner")
        st.select_slider("Severity", ["Minor", "Major", "Critical"])
        st.selectbox("Owner", ["Quality", "Product", "Ops"])
        if st.button("🤖 AI RCA Assist"):
            st.write(st.session_state.ai.generate("Give me 3 potential root causes for a generic product defect."))

# --- 5. MAIN ---
def main():
    with st.sidebar:
        st.markdown("## O.R.I.O.N.")
        st.caption("v5.1 | LEADERSHIP BUILD")
        
        # CHECK FOR API KEY & STATUS
        if st.session_state.ai.available:
            st.success("✅ AI Neural Engine Connected")
        else:
            st.error("⚠️ AI Offline")
            k = st.text_input("API Key", type="password", help="Enter GEMINI_API_KEY here if not in secrets.")
            if k: 
                st.session_state.ai.configure(k)
                st.rerun()

        st.markdown("---")
        
        if 'nav' not in st.session_state: st.session_state.nav = "Dashboard"
        
        opts = {
            "Dashboard": "📊",
            "Vision Intel": "👁️",
            "Strategy Plan": "📝",
            "Supply Chain": "📦",
            "CAPA": "🛡️"
        }
        
        for label, icon in opts.items():
            if st.button(f"{icon}  {label}", use_container_width=True):
                st.session_state.nav = label
                st.rerun()

    if st.session_state.nav == "Dashboard": render_dashboard()
    elif st.session_state.nav == "Vision Intel": render_voc()
    elif st.session_state.nav == "Strategy Plan": render_plan()
    elif st.session_state.nav == "Supply Chain": render_supply()
    elif st.session_state.nav == "CAPA": render_capa()

if __name__ == "__main__":
    main()
