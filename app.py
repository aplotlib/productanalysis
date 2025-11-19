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
    page_title="O.R.I.O.N. v7.0 | VIVE Health",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VIVE BRAND THEME (EXECUTIVE DARK MODE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&family=Open+Sans:wght@400;600&display=swap');

    /* GLOBAL APP RESET */
    .stApp {
        background-color: #0B1E3D !important; /* Vive Navy */
        color: #FFFFFF !important;
        font-family: 'Open Sans', sans-serif;
    }

    /* HEADERS (Redzone Style) */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #FFFFFF !important;
    }
    h1 {
        font-size: 2.5rem !important;
        color: #00C6D7 !important; /* Vive Teal */
        margin-bottom: 0px;
    }
    .subtitle {
        color: #94A3B8;
        font-family: 'Montserrat', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }

    /* CARDS & CONTAINERS */
    .stContainer, div[data-testid="metric-container"], .report-box {
        background-color: #132448 !important;
        border: 1px solid #1E3A5F;
        border-radius: 6px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* METRICS */
    div[data-testid="metric-container"] {
        border-left: 4px solid #00C6D7;
        background-color: #0F2042 !important;
    }
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important;
        font-weight: 700;
        font-size: 0.85rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 700;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050E1F !important;
        border-right: 1px solid #1E3A5F;
    }

    /* BUTTONS (Primary Action) */
    .stButton>button {
        background-color: #00C6D7 !important;
        color: #050E1F !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        border: none;
        border-radius: 4px;
        height: 3.2em;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #FFFFFF !important;
        color: #00C6D7 !important;
        box-shadow: 0 0 15px rgba(0, 198, 215, 0.4);
    }

    /* INPUTS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea {
        background-color: #1B3B6F !important;
        color: white !important;
        border: 1px solid #475569;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8;
        border: none;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00C6D7;
        border-bottom: 2px solid #00C6D7;
    }
    
    /* FOOTER */
    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 0.7rem;
        font-family: 'Montserrat', sans-serif;
        margin-top: 40px;
        border-top: 1px solid #1E3A5F;
        padding-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ROBUST INTELLIGENCE ENGINE (Requested Logic) ---
class IntelligenceEngine:
    def __init__(self):
        self.client = None
        self.provider = None
        self.available = False
        self._initialize_ai_clients()

    def _initialize_ai_clients(self):
        """Initialize AI clients from Streamlit secrets (Robust Pattern)"""
        try:
            # 1. Try Google Gemini (Visual capabilities)
            if 'GEMINI_API_KEY' in st.secrets:
                self._configure_gemini(st.secrets['GEMINI_API_KEY'])
            elif 'GOOGLE_API_KEY' in st.secrets:
                self._configure_gemini(st.secrets['GOOGLE_API_KEY'])
            
            # 2. Try OpenAI (Text capabilities fallback)
            if not self.available and 'OPENAI_API_KEY' in st.secrets:
                self._configure_openai(st.secrets['OPENAI_API_KEY'])
            
            # 3. Try Env Vars
            if not self.available:
                key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if key:
                    if key.startswith("sk-"): self._configure_openai(key)
                    else: self._configure_gemini(key)

            # 4. Manual Override (Session State)
            if not self.available and st.session_state.get("manual_key"):
                key = st.session_state.manual_key
                if key.startswith("sk-"): self._configure_openai(key)
                else: self._configure_gemini(key)

        except Exception as e:
            st.session_state.ai_error = str(e)
            self.available = False

    def _configure_gemini(self, key):
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self.client = genai
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.provider = "Gemini"
            self.available = True
        except:
            pass

    def _configure_openai(self, key):
        try:
            import openai
            self.client = openai.OpenAI(api_key=key)
            self.provider = "OpenAI"
            self.available = True
        except:
            pass

    def generate(self, prompt):
        if not self.available: return "⚠️ AI Offline. Check API configuration."
        try:
            if self.provider == "Gemini":
                return self.model.generate_content(prompt).text
            elif self.provider == "OpenAI":
                response = self.client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def analyze_vision(self, image, prompt):
        if not self.available: return "⚠️ AI Offline."
        try:
            if self.provider == "Gemini":
                return self.vision.generate_content([prompt, image]).text
            else:
                return "Vision analysis requires Gemini API."
        except Exception as e:
            return f"Error: {e}"

# Initialize Singleton
if 'ai' not in st.session_state:
    st.session_state.ai = IntelligenceEngine()

# --- 3. CACHED DATA ENGINE ---
class DataEngine:
    @staticmethod
    @st.cache_data
    def process_file(file_content, filename):
        """Parses generic sales/return reports."""
        try:
            io_file = io.BytesIO(file_content)
            # Try simple read
            try:
                df = pd.read_csv(io_file) if filename.endswith('.csv') else pd.read_excel(io_file)
            except:
                # Advanced read: Scan for header
                io_file.seek(0)
                content = file_content.decode('utf-8', errors='replace')
                lines = content.splitlines()
                best_idx = 0
                max_score = 0
                keywords = ['sku', 'product', 'date', 'qty', 'sales', 'return']
                for i, line in enumerate(lines[:20]):
                    score = sum(1 for k in keywords if k in line.lower())
                    if score > max_score:
                        max_score = score
                        best_idx = i
                io_file.seek(0)
                df = pd.read_csv(io_file, header=best_idx) if filename.endswith('.csv') else pd.read_excel(io_file, header=best_idx)

            # Normalize Columns
            col_map = {
                'created on': 'Date', 'date': 'Date',
                'product': 'Product', 'sku': 'Product', 'item': 'Product',
                'sales': 'Sales', 'sold': 'Sales', 'order qty': 'Sales',
                'returns': 'Returns', 'return qty': 'Returns', 'qty returned': 'Returns',
                'reason': 'Reason', 'disposition': 'Reason',
                'ticket': 'Ticket ID'
            }
            df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
            
            # Type enforcement
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')

            return df
        except Exception:
            return pd.DataFrame()

# --- 4. MODULES ---

def render_dashboard():
    st.markdown("<h1>O.R.I.O.N.</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>OPERATIONAL REVIEW & INTELLIGENCE OPTIMIZATION NETWORK</div>", unsafe_allow_html=True)
    
    # Data State
    if 'data' not in st.session_state: st.session_state.data = None

    if st.session_state.data is None:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.info("📡 **AWAITING SIGNAL**")
            f = st.file_uploader("Upload Quality/Sales Report (CSV/XLSX)", type=['csv', 'xlsx'])
            if f:
                df = DataEngine.process_file(f.getvalue(), f.name)
                if not df.empty:
                    st.session_state.data = df
                    st.success("SIGNAL LOCKED")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Could not parse file.")
        with c2:
            st.markdown("### SYSTEM READY")
            st.markdown("• **Trend Analysis:** Auto-detects monthly return rates.")
            st.markdown("• **Root Cause:** AI hypothesis engine active.")
            st.markdown("• **Vision:** Screenshot parser online.")
    else:
        df = st.session_state.data
        
        # Calc Metrics
        sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        returns = df['Returns'].sum() if 'Returns' in df.columns else 0
        rate = (returns/sales*100) if sales > 0 else 0
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RECORDS", len(df))
        m2.metric("SALES VOL", f"{sales:,.0f}")
        m3.metric("RETURNS", f"{returns:,.0f}")
        m4.metric("RETURN RATE", f"{rate:.2f}%")
        
        st.markdown("---")
        
        # Visuals
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📉 QUALITY TREND")
            if 'Date' in df.columns and 'Sales' in df.columns:
                df['Month'] = df['Date'].dt.to_period('M').astype(str)
                monthly = df.groupby('Month')[['Sales', 'Returns']].sum().reset_index()
                monthly['Rate'] = (monthly['Returns'] / monthly['Sales']) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=monthly['Month'], y=monthly['Sales'], name="Sales", marker_color='#1B3B6F'))
                fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Rate'], name="Rate %", yaxis="y2", line=dict(color='#00C6D7', width=3)))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis2=dict(overlaying='y', side='right'), legend=dict(orientation="h", y=1.1), height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Upload file with Date/Sales/Returns columns for trend analysis.")

        with c2:
            st.markdown("### 🧠 AI ANALYST")
            if st.button("RUN HYPOTHESIS", type="primary"):
                with st.spinner("Analyzing..."):
                    summ = df.describe().to_string()
                    prompt = f"Analyze this quality data summary. Identify 1 major anomaly in Return Rate and suggest a root cause (e.g. Vendor change? Seasonality?). Data: {summ}"
                    st.info(st.session_state.ai.generate(prompt))
            
            if st.button("CLEAR DATA"):
                st.session_state.data = None
                st.rerun()

def render_voc():
    st.markdown("<h1>VISION INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>CUSTOMER SENTIMENT & DEFECT EXTRACTION</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📥 SOURCE")
        img = st.file_uploader("Upload Dashboard Screenshot", type=['png', 'jpg'])
        if img:
            st.image(img, caption="Context", use_column_width=True)
            if st.button("INITIATE SCAN", type="primary"):
                with st.spinner("EXTRACTING..."):
                    prompt = "Analyze this screenshot. 1. Extract Statistics. 2. Identify Top 3 Product Defects. 3. Determine Sentiment. 4. Suggest CAPA."
                    st.session_state.voc_res = st.session_state.ai.analyze_vision(Image.open(img), prompt)
    
    with c2:
        st.markdown("### 📝 INTELLIGENCE")
        if 'voc_res' in st.session_state:
            st.markdown(st.session_state.voc_res)
            st.divider()
            if st.button("🛡️ ELEVATE TO CAPA"):
                st.session_state.capa_prefill = st.session_state.voc_res
                st.session_state.nav = "CAPA MANAGER"
                st.rerun()

def render_plan():
    st.markdown("<h1>STRATEGIC PLANNER</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    name = c1.text_input("PROJECT NAME", placeholder="Ex: Mobility X1 Gen2")
    risk = c2.selectbox("RISK CLASS", ["Class I", "Class II", "Class III"])
    
    st.divider()
    
    c_edit, c_view = st.columns([1.3, 1])
    sections = {"scope": "Scope", "regs": "Regulatory", "test": "Validation", "vend": "Vendor Controls"}
    
    with c_edit:
        locks = {}
        for k, label in sections.items():
            with st.expander(label, expanded=True):
                locks[k] = st.checkbox(f"LOCK", key=f"l_{k}")
                st.session_state[f"qp_{k}"] = st.text_area("CONTENT", key=f"t_{k}", height=100, label_visibility="collapsed")
        
        if st.button("✨ AI: OPTIMIZE PLAN", type="primary"):
            with st.spinner("Processing..."):
                ctx = f"Project: {name}, Risk: {risk}"
                for k, label in sections.items():
                    if not locks[k]:
                        p = f"Write professional Quality Plan section '{label}'. Context: {ctx}. User Draft: {st.session_state[f'qp_{k}']}. No Markdown."
                        st.session_state[f"qp_{k}"] = st.session_state.ai.generate(p)
                st.rerun()

    with c_view:
        st.markdown("### 📄 PREVIEW")
        full = f"QUALITY STRATEGY: {name.upper()}\nRISK: {risk}\nDATE: {datetime.now().date()}\n\n"
        for k, label in sections.items():
            full += f"{label.upper()}\n{'-'*len(label)}\n{st.session_state.get(f'qp_{k}', '')}\n\n"
        st.text_area("DOC", full, height=600)
        st.download_button("📥 DOWNLOAD", full, file_name="Strategy.txt")

def render_capa():
    st.markdown("<h1>CAPA MANAGER</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>CORRECTIVE & PREVENTIVE ACTION SUITE</div>", unsafe_allow_html=True)

    if 'capa_id' not in st.session_state: st.session_state.capa_id = f"CAPA-{int(time.time())}"
    prefill = st.session_state.get("capa_prefill", "")

    tabs = st.tabs(["1. INTAKE", "2. RISK", "3. INVESTIGATION", "4. ACTION", "5. VERIFICATION", "6. COST"])

    # 1. INTAKE
    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.text_input("CAPA ID", st.session_state.capa_id, disabled=True)
            st.text_input("ISSUE TITLE", placeholder="Short description of non-conformance")
            st.text_area("DETAILED DESCRIPTION", value=prefill, height=150)
        with c2:
            st.selectbox("SOURCE", ["Customer Complaint", "Internal Audit", "Supplier NCR", "Regulatory"])
            st.selectbox("OWNER", ["Quality", "Ops", "Product"])
            st.date_input("DATE OPENED")

    # 2. RISK
    with tabs[1]:
        st.markdown("### FMEA RISK ASSESSMENT")
        r1, r2 = st.columns(2)
        sev = r1.select_slider("SEVERITY (S)", options=[1, 2, 3, 4, 5])
        occ = r2.select_slider("OCCURRENCE (O)", options=[1, 2, 3, 4, 5])
        rpn = sev * occ
        
        color = "green" if rpn <= 6 else ("orange" if rpn <= 12 else "red")
        st.markdown(f"#### RPN SCORE: :{color}[{rpn}]")
        
        if rpn > 10:
            st.warning("⚠️ HIGH RISK DETECTED. CONTAINMENT REQUIRED.")
            st.text_area("CONTAINMENT ACTION", placeholder="Immediate fix...")
        else:
            st.success("✅ RISK ACCEPTABLE")

    # 3. INVESTIGATION (RCA)
    with tabs[2]:
        st.markdown("### ROOT CAUSE ANALYSIS")
        method = st.radio("TOOL", ["5 Whys", "Fishbone"])
        
        if method == "5 Whys":
            c_w, c_ai = st.columns([2, 1])
            with c_w:
                w1 = st.text_input("1. WHY?")
                w2 = st.text_input("2. WHY?")
                w3 = st.text_input("3. WHY?")
                w4 = st.text_input("4. WHY?")
                w5 = st.text_input("5. WHY (ROOT CAUSE)?")
            with c_ai:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🤖 AI COACH"):
                    if w1: st.info(st.session_state.ai.generate(f"Suggest root cause chain based on: {w1}"))
        else:
            c1, c2 = st.columns(2)
            c1.text_area("MAN / MATERIAL / MACHINE")
            c2.text_area("METHOD / MEASUREMENT / ENV")

    # 4. ACTION
    with tabs[3]:
        st.markdown("### ACTION PLAN")
        st.text_input("CORRECTIVE ACTION (Long Term)")
        st.text_input("PREVENTIVE ACTION (Systemic)")
        st.date_input("DUE DATE")
        st.checkbox("Requires SOP Update?")
        st.checkbox("Requires Training?")

    # 5. VERIFICATION
    with tabs[4]:
        st.markdown("### EFFECTIVENESS CHECK")
        st.radio("METHOD", ["Data Trend", "Audit", "Re-Test"])
        st.text_area("EVIDENCE")
        if st.checkbox("EFFECTIVENESS VERIFIED?"):
            st.success("READY FOR CLOSURE")
            if st.button("CLOSE CAPA", type="primary"):
                st.balloons()

    # 6. COST
    with tabs[5]:
        st.markdown("### COST OF QUALITY (CoQ)")
        c1, c2 = st.columns(2)
        c1.number_input("SCRAP ($)", 0.0)
        c1.number_input("REWORK ($)", 0.0)
        c2.number_input("SHIPPING ($)", 0.0)
        c2.number_input("LABOR HOURS", 0.0)
        st.metric("TOTAL IMPACT", "$0.00")

# --- 5. MAIN CONTROLLER ---
def main():
    with st.sidebar:
        st.title("O.R.I.O.N.")
        st.caption("VIVE HEALTH v7.0 | EXEC BUILD")
        
        # AI Status
        if st.session_state.ai.available:
            st.success(f"🟢 ONLINE ({st.session_state.ai.provider})")
        else:
            st.error("🔴 OFFLINE")
            k = st.text_input("MANUAL KEY", type="password")
            if k:
                st.session_state.manual_key = k
                st.session_state.ai._initialize_ai_clients()
                st.rerun()
        
        st.markdown("---")
        
        if 'nav' not in st.session_state: st.session_state.nav = "DASHBOARD"
        
        opts = {
            "DASHBOARD": "📊", 
            "VISION INTEL": "👁️", 
            "STRATEGY": "📝", 
            "CAPA MANAGER": "🛡️"
        }
        
        for label, icon in opts.items():
            if st.button(f"{icon}  {label}", use_container_width=True):
                st.session_state.nav = label
                st.rerun()
        
        st.markdown("""
        <div class='footer-text'>
        built by alex popoff 11/19/2025<br>
        recent build v.6.3<br>
        gemini vibe coded beta test
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.nav == "DASHBOARD": render_dashboard()
    elif st.session_state.nav == "VISION INTEL": render_voc()
    elif st.session_state.nav == "STRATEGY": render_plan()
    elif st.session_state.nav == "CAPA MANAGER": render_capa()

if __name__ == "__main__":
    main()
