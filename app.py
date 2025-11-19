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
    page_title="VIVE Health | Product Lifecycle Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VIVE "NORTH STAR" THEME (HIGH CONTRAST) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');

    /* DEEP SPACE BACKGROUND */
    .stApp {
        background-color: #020408 !important;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px);
        background-size: 550px 550px, 350px 350px, 250px 250px; 
        color: #ffffff !important;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, h4, p, div, span {
        font-family: 'Montserrat', sans-serif !important;
        color: white !important;
    }
    h1 {
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00C6D7 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* CONTAINER STYLING */
    .stContainer, div[data-testid="metric-container"], .report-box {
        background-color: rgba(13, 23, 33, 0.95) !important;
        border: 1px solid #1e293b;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        padding: 20px;
    }

    /* METRICS */
    div[data-testid="metric-container"] {
        border-left: 4px solid #00C6D7;
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 20px rgba(0, 198, 215, 0.3);
    }
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #333;
    }
    
    /* BUTTONS */
    .stButton>button {
        background: #00C6D7 !important;
        color: #000000 !important;
        border: none;
        font-weight: 800 !important;
        text-transform: uppercase;
        border-radius: 4px;
        height: 3em;
    }
    .stButton>button:hover {
        background: #ffffff !important;
        box-shadow: 0 0 15px #00C6D7;
    }

    /* INPUTS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea {
        background-color: #0f172a !important;
        color: white !important;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE ENGINE (AI & LOGGING) ---
if 'system_log' not in st.session_state:
    st.session_state.system_log = []

def log_event(action, icon="🔵"):
    """Logs real user actions to the 'Live Activity' feed."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.system_log.insert(0, {"time": timestamp, "action": action, "icon": icon})

class AIHandler:
    def __init__(self):
        self.available = False
        self.model = None
        self.vision = None
        
        try:
            import google.generativeai as genai
            self.genai = genai
            # Key Priority: Streamlit Secrets > Env Var > Manual
            key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if key: self.configure(key)
        except ImportError:
            pass

    def configure(self, key):
        try:
            self.genai.configure(api_key=key)
            self.model = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.available = True
        except Exception:
            self.available = False

    def generate(self, prompt):
        if not self.available: return "⚠️ AI Offline"
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"AI Error: {e}"

    def analyze_vision(self, image, prompt):
        if not self.available: return "⚠️ AI Offline"
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Vision Error: {e}"

if 'ai' not in st.session_state: st.session_state.ai = AIHandler()

# --- 3. ANALYTICS ENGINE ---
class PerformanceAnalyzer:
    @staticmethod
    def clean_cols(df):
        """Normalizes column names to standard internal IDs."""
        col_map = {
            'date': 'Date', 'created on': 'Date', 'return date': 'Date',
            'product': 'Product', 'sku': 'Product', 'item': 'Product',
            'qty': 'Qty', 'quantity': 'Qty', 'returns': 'Returns', 'return qty': 'Returns',
            'sales': 'Sales', 'sold': 'Sales', 'order qty': 'Sales',
            'reason': 'Reason', 'return reason': 'Reason', 'disposition': 'Reason'
        }
        df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
        return df

    @staticmethod
    def process_performance_file(file):
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            df = PerformanceAnalyzer.clean_cols(df)
            
            # Ensure Date parsing
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.sort_values('Date')

            # Logic: If Sales and Returns columns exist, calculate Rate
            if 'Sales' in df.columns and 'Returns' in df.columns:
                df['Return Rate'] = (df['Returns'] / df['Sales']) * 100
            
            return df, None
        except Exception as e:
            return None, str(e)

# --- 4. UI MODULES ---

def render_dashboard():
    st.markdown("<h1>Executive Command Center</h1>", unsafe_allow_html=True)
    
    # -- STATE MANAGEMENT FOR DATA --
    if 'dash_data' not in st.session_state: st.session_state.dash_data = None
    
    # -- SECTION 1: DATA INGESTION (NO FAKE DATA) --
    if st.session_state.dash_data is None:
        with st.container():
            st.markdown("### 📡 System Ready. Awaiting Data Stream.")
            st.info("Upload Sales & Return Reports (Excel/CSV) to initialize the Command Center visualization.")
            
            up_file = st.file_uploader("Upload Performance Data", type=['csv', 'xlsx'])
            if up_file:
                df, err = PerformanceAnalyzer.process_performance_file(up_file)
                if err:
                    st.error(f"Ingestion Failed: {err}")
                else:
                    st.session_state.dash_data = df
                    log_event(f"Data Ingested: {up_file.name} ({len(df)} records)", "💾")
                    st.success("Data Stream Active")
                    st.rerun()

    # -- SECTION 2: LIVE DASHBOARD (ONLY IF DATA EXISTS) --
    else:
        df = st.session_state.dash_data
        
        # -- METRICS CALCULATION --
        # Dynamic metrics based on uploaded data
        total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        total_returns = df['Returns'].sum() if 'Returns' in df.columns else 0
        avg_rate = (total_returns / total_sales * 100) if total_sales > 0 else 0
        
        # Render Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sales Volume", f"{total_sales:,.0f}")
        m2.metric("Total Returns", f"{total_returns:,.0f}")
        m3.metric("Avg Return Rate", f"{avg_rate:.2f}%")
        m4.metric("Data Points", len(df))
        
        st.markdown("---")
        
        # -- VISUALIZATION & INSIGHTS --
        c1, c2 = st.columns([2.5, 1])
        
        with c1:
            st.markdown("### 📉 Performance & Seasonality Analysis")
            
            if 'Date' in df.columns and 'Return Rate' in df.columns:
                # Time Series Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['Return Rate'], 
                    mode='lines+markers', 
                    name='Return Rate %',
                    line=dict(color='#FF0055', width=3)
                ))
                if 'Sales' in df.columns:
                     fig.add_trace(go.Bar(
                        x=df['Date'], y=df['Sales'], 
                        name='Sales Volume', 
                        yaxis='y2',
                        marker_color='rgba(0, 198, 215, 0.2)'
                    ))
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                    yaxis=dict(title="Return Rate %", gridcolor="#333"),
                    yaxis2=dict(title="Sales Volume", overlaying='y', side='right'),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Reason Analysis (If 'Reason' column exists)
            if 'Reason' in df.columns:
                st.markdown("#### Return Reason Shifts")
                reason_counts = df.groupby('Reason')['Returns'].sum().reset_index()
                fig2 = px.bar(reason_counts, x='Reason', y='Returns', color='Returns', color_continuous_scale=['#00C6D7', '#FF0055'])
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown("### 🧠 AI Strategic Analyst")
            if st.button("Analyze Context & Trends", type="primary"):
                with st.spinner("AI analyzing seasonality, spikes, and vendor correlations..."):
                    # Serialize data for AI
                    summary = df.describe().to_string()
                    head = df.head(10).to_string()
                    prompt = f"""
                    You are a Senior Quality Analyst. Analyze this Sales/Return data.
                    Data Summary: {summary}
                    First 10 Rows: {head}
                    
                    1. Identify any seasonality or spikes in Return Rate.
                    2. Hypothesize why (e.g., "Did a vendor change material in Q3?").
                    3. Suggest immediate actions.
                    """
                    analysis = st.session_state.ai.generate(prompt)
                    st.info(analysis)
                    log_event("AI Strategic Analysis Generated", "🧠")
            
            st.divider()
            st.markdown("### 🛰️ Live Session Log")
            for event in st.session_state.system_log:
                st.markdown(f"`{event['time']}` {event['icon']} {event['action']}")
                
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.dash_data = None
                st.rerun()

def render_quality_planning():
    st.markdown("<h1>Quality Strategy Planner</h1>", unsafe_allow_html=True)
    
    keys = ["qp_name", "qp_risk", "qp_mkts", "qp_scope", "qp_regs", "qp_test", "qp_vend", "qp_path"]
    for k in keys: 
        if k not in st.session_state: st.session_state[k] = ""
    
    with st.container():
        c1, c2 = st.columns(2)
        st.session_state.qp_name = c1.text_input("Project Name / SKU", st.session_state.qp_name)
        st.session_state.qp_risk = c2.selectbox("Risk Classification", ["Class I", "Class II", "Class III"])
        st.session_state.qp_mkts = st.multiselect("Markets", ["USA", "EU", "UK", "CAN"], default=["USA"])

    st.divider()
    
    c_left, c_right = st.columns([1.2, 1])
    
    sections = [
        ("scope", "Scope", "Objectives & Deliverables"),
        ("regs", "Regulatory", "ISO 13485, FDA, MDR"),
        ("test", "Validation", "Testing Requirements"),
        ("vend", "Supply Chain", "Vendor Controls"),
        ("path", "Critical Path", "Timeline")
    ]
    
    with c_left:
        st.markdown("### 📝 AI-Assisted Drafting")
        locks = {}
        for code, title, hint in sections:
            with st.expander(title, expanded=True):
                l, t = st.columns([0.2, 1])
                locks[code] = l.checkbox("Lock", key=f"lock_{code}")
                st.session_state[f"qp_{code}"] = t.text_area("Content", st.session_state[f"qp_{code}"], height=100, placeholder=hint, label_visibility="collapsed")
        
        if st.button("✨ Generate / Optimize Plan", type="primary"):
            with st.spinner("AI Optimizing..."):
                ctx = f"Project: {st.session_state.qp_name}. Risk: {st.session_state.qp_risk}."
                for c, t, _ in sections:
                    if not locks[c]:
                        p = f"Write professional Quality Plan section '{t}'. Context: {ctx}. User draft: {st.session_state[f'qp_{c}']}. No Markdown."
                        st.session_state[f"qp_{c}"] = st.session_state.ai.generate(p)
                        time.sleep(0.5)
                log_event(f"Plan Generated: {st.session_state.qp_name}", "📄")
                st.success("Plan Updated")
                st.rerun()

    with c_right:
        st.markdown("### 📄 Preview")
        full = f"QUALITY PLAN: {st.session_state.qp_name.upper()}\nDATE: {datetime.now().date()}\nRISK: {st.session_state.qp_risk}\n\n"
        for c, t, _ in sections:
            full += f"{t.upper()}\n{'-'*len(t)}\n{st.session_state[f'qp_{c}']}\n\n"
        st.text_area("Final Doc", full, height=600)
        st.download_button("📥 Download .txt", full, file_name="plan.txt")

def render_market_intel():
    st.markdown("<h1>Market Intelligence (VoC)</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["👁️ Vision Analysis", "📊 Data Parser"])
    
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.info("Upload Amazon Review Dashboard / Return Report Screenshot")
            img_file = st.file_uploader("Source Image", type=['png', 'jpg'])
            if img_file:
                img = Image.open(img_file)
                st.image(img, caption="Target", use_column_width=True)
                if st.button("🔍 Analyze Vision", type="primary"):
                    with st.spinner("Extracting Intelligence..."):
                        prompt = """
                        Analyze this screenshot. 
                        1. Extract Statistics (Star Rating, Return %, etc).
                        2. Identify Top 3 Defects.
                        3. Sentiment Analysis.
                        4. Recommended CAPA Actions.
                        """
                        st.session_state.voc_res = st.session_state.ai.analyze_vision(img, prompt)
                        log_event("Vision Analysis Complete", "👁️")
        with c2:
            if 'voc_res' in st.session_state:
                st.markdown("### 🤖 Intelligence Report")
                st.write(st.session_state.voc_res)
                if st.button("🚨 Create CAPA"):
                    st.session_state.capa_prefill = st.session_state.voc_res
                    st.session_state.nav = "CAPA Manager"
                    st.rerun()
    
    with tab2:
        st.markdown("### 📊 Upload Review Data")
        f = st.file_uploader("Upload Reviews (CSV)", type=['csv'])
        if f:
            df = pd.read_csv(f)
            st.dataframe(df.head())
            if st.button("Analyze Sentiment"):
                st.info(st.session_state.ai.generate(f"Analyze sentiment of these reviews: {df.head(10).to_string()}"))

def render_supply_chain():
    st.markdown("<h1>Supply Chain Analytics</h1>", unsafe_allow_html=True)
    st.info("Smart Parser for Odoo Exports (Inventory/Helpdesk)")
    
    f = st.file_uploader("Upload Odoo Export", type=['csv', 'xlsx'])
    if f:
        df, err = PerformanceAnalyzer.process_performance_file(f)
        if err:
            st.error(err)
        else:
            st.success(f"Loaded {len(df)} rows")
            st.dataframe(df.head())
            
            if st.button("Generate Insights"):
                st.info(st.session_state.ai.generate(f"Analyze Supply Chain Data: {df.head(10).to_string()}"))

def render_capa():
    st.markdown("<h1>CAPA Manager</h1>", unsafe_allow_html=True)
    
    prefill = st.session_state.get("capa_prefill", "")
    if prefill:
        st.success("Using VoC Intelligence Data")
        del st.session_state.capa_prefill
        
    tabs = st.tabs(["Intake", "RCA", "Action", "Close"])
    
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.text_input("ID", f"CAPA-{int(time.time())}", disabled=True)
        c2.select_slider("Risk", ["Low", "Medium", "High", "Critical"])
        st.text_area("Issue Description", value=prefill, height=150)
        
    with tabs[1]:
        w1 = st.text_input("1. Why?")
        if st.button("🤖 AI RCA Coach"):
            if w1: st.write(st.session_state.ai.generate(f"Root cause for: {w1}"))
            
    with tabs[3]:
        if st.button("Close CAPA", type="primary"):
            log_event("CAPA Closed", "✅")
            st.balloons()

# --- 5. MAIN CONTROLLER ---
def main():
    with st.sidebar:
        st.title("PLI SYSTEM")
        st.caption("VIVE HEALTH v4.0")
        
        if not st.session_state.ai.available:
            st.error("AI Offline")
            k = st.text_input("API Key", type="password")
            if k: 
                st.session_state.ai.configure(k)
                st.rerun()
        else:
            st.success("AI Online")

        st.markdown("---")
        
        menu = ["Dashboard", "Quality Planning", "Market Intelligence", "Supply Chain", "CAPA Manager"]
        if 'nav' not in st.session_state: st.session_state.nav = "Dashboard"
        
        for item in menu:
            if st.button(item, use_container_width=True):
                st.session_state.nav = item
                st.rerun()

    if st.session_state.nav == "Dashboard": render_dashboard()
    elif st.session_state.nav == "Quality Planning": render_quality_planning()
    elif st.session_state.nav == "Market Intelligence": render_market_intel()
    elif st.session_state.nav == "Supply Chain": render_supply_chain()
    elif st.session_state.nav == "CAPA Manager": render_capa()

if __name__ == "__main__":
    main()
