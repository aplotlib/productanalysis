import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import re
import os
import random
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="VIVE Health | Product Lifecycle Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VIVE BRAND & HIGH CONTRAST THEME ---
st.markdown("""
    <style>
    /* Montserrat Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');

    /* FORCE DARK MODE COMPATIBILITY */
    /* We force specific colors to override Chrome Dark Mode inversions */
    
    .stApp {
        background-color: #020408 !important;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px);
        background-size: 550px 550px, 350px 350px, 250px 250px; 
        color: #ffffff !important;
    }

    /* TYPOGRAPHY - FORCED WHITE */
    h1, h2, h3, h4, h5, h6, p, span, li, div {
        color: #ffffff !important;
        font-family: 'Montserrat', sans-serif !important;
    }
    
    h1, h2, h3 {
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 800 !important;
    }
    
    /* ACCENT TEXT */
    .accent-text {
        color: #00C6D7 !important;
    }

    /* HIGH CONTRAST CONTAINERS (Replaces Transparent Glass) */
    .stContainer, div[data-testid="metric-container"], .report-box, .element-container {
        background-color: rgba(13, 23, 33, 0.95) !important; /* Almost solid dark navy */
        border: 1px solid #1e293b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.8);
    }
    
    /* METRICS STYLING */
    div[data-testid="metric-container"] {
        border-left: 4px solid #00C6D7;
    }
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important; /* Teal Label */
        font-size: 0.9rem;
        opacity: 1 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.0rem;
        text-shadow: 0 0 10px rgba(0,0,0,0.5);
    }

    /* SIDEBAR - SOLID BLACK */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #333;
    }
    section[data-testid="stSidebar"] h1 {
        color: #00C6D7 !important;
    }
    
    /* INTERACTIVE ELEMENTS */
    .stButton>button {
        background: #00C6D7 !important;
        color: #000000 !important; /* Black text on Teal button for contrast */
        border: none;
        border-radius: 6px;
        font-weight: 800 !important;
        text-transform: uppercase;
        height: 3.2em;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #ffffff !important;
        box-shadow: 0 0 15px #00C6D7;
    }

    /* INPUT FIELDS - SOLID BACKGROUNDS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea, .stDateInput input {
        background-color: #1a2230 !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    
    /* TABLES */
    div[data-testid="stDataFrame"] {
        background-color: #1a2230;
        border-radius: 5px;
        padding: 5px;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8 !important;
        border: none;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00C6D7;
        color: #000000 !important; /* Active Tab Black Text */
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ROBUST AI HANDLER ---
class AIHandler:
    def __init__(self):
        self.available = False
        self.model = None
        self.vision = None
        self.api_key = None

        # 1. Try Imports
        try:
            import google.generativeai as genai
            self.genai = genai
            
            # 2. Try finding API Key in multiple locations
            # Priority: Streamlit Secrets -> Environment Variable -> Manual Input (handled in UI)
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                pass
            
            if not self.api_key:
                self.api_key = os.environ.get("GEMINI_API_KEY")

            # 3. Configure if key exists
            if self.api_key:
                self.configure_ai(self.api_key)
                
        except ImportError:
            st.error("⚠️ Google Generative AI library not installed.")

    def configure_ai(self, key):
        try:
            self.genai.configure(api_key=key)
            self.model = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.available = True
        except Exception as e:
            st.error(f"AI Config Failed: {e}")
            self.available = False

    def generate_text(self, prompt, temperature=0.7):
        if not self.available: return "⚠️ AI Offline. Please enter API Key in Sidebar."
        try:
            return self.model.generate_content(prompt, generation_config={"temperature": temperature}).text
        except Exception as e:
            return f"AI Generation Error: {e}"

    def analyze_image(self, image, prompt):
        if not self.available: return "⚠️ AI Offline. Please enter API Key in Sidebar."
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Vision Error: {e}"

# Initialize AI State
if 'ai' not in st.session_state:
    st.session_state.ai = AIHandler()

# --- 3. DATA INTELLIGENCE MODULE ---
class DataProcessor:
    @staticmethod
    def normalize_columns(df):
        """Maps varied user headers to standard VIVE system names."""
        col_map = {
            'product title': 'Product', 'product name': 'Product', 'sku': 'SKU', 'reference': 'SKU',
            'qty': 'Qty', 'quantity': 'Qty', 'on hand': 'On Hand', 'forecast': 'Forecast',
            'ticket': 'Ticket ID', 'subject': 'Issue', 'create date': 'Date', 'created on': 'Date',
            'priority': 'Priority', 'stage': 'Status', 'daily rate': 'Daily Sales'
        }
        df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
        return df

    @staticmethod
    def parse_odoo_file(uploaded_file):
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors='replace')
            lines = content.split('\n')
            
            strong_signals = ['product title', 'created on', 'ticket ids sequence', 'total to buy']
            
            header_idx = 0
            max_score = 0
            
            for i, line in enumerate(lines[:30]):
                line_lower = line.lower()
                score = sum(2 for s in strong_signals if s in line_lower)
                if score > max_score:
                    max_score = score
                    header_idx = i
            
            uploaded_file.seek(0)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=header_idx)
            else:
                df = pd.read_excel(uploaded_file, header=header_idx)
            
            df = df.dropna(how='all')
            df = DataProcessor.normalize_columns(df)
            
            cols = [str(c).lower() for c in df.columns]
            data_type = "Unknown"
            if 'ticket id' in df.columns or 'status' in df.columns:
                data_type = "Helpdesk"
            elif 'on hand' in df.columns or 'daily sales' in df.columns:
                data_type = "Inventory"
                
            return df, data_type, None
            
        except Exception as e:
            return None, "Error", str(e)

    @staticmethod
    def calculate_inventory_metrics(df):
        if 'On Hand' in df.columns and 'Daily Sales' in df.columns:
            df['Weeks of Supply'] = df.apply(lambda x: (x['On Hand'] / x['Daily Sales'] / 7) if x['Daily Sales'] > 0 else 999, axis=1)
            df['Risk Level'] = df['Weeks of Supply'].apply(lambda x: 'Critical Low' if x < 2 else ('Overstock' if x > 26 else 'Healthy'))
        return df

    @staticmethod
    def clean_text_for_export(text):
        if not text: return ""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'#+\s', '', text)
        return text.strip()

# --- 4. UI MODULES ---

def render_dashboard():
    st.markdown("<h1>Executive Command Center</h1>", unsafe_allow_html=True)
    st.caption(f"LOGGED IN: {datetime.now().strftime('%d-%b-%Y %H:%M')} | VIVE QUALITY SYSTEMS")
    
    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Star Rating", "4.2 ⭐", "+0.1 MoM")
    m2.metric("Return Rate (30d)", "8.4%", "-1.2% MoM")
    m3.metric("Active CAPAs", "4", "1 Critical")
    m4.metric("Inv. Health", "92%", "Stable")
    
    st.markdown("---")
    
    c1, c2 = st.columns([2.2, 1])
    
    with c1:
        st.markdown("### 📉 Enterprise Quality Trends")
        dates = pd.date_range(start="2025-09-01", periods=12, freq="W")
        df_chart = pd.DataFrame({
            "Date": dates,
            "Complaints": [15, 12, 18, 10, 8, 14, 9, 5, 7, 6, 4, 5],
            "Returns": [25, 22, 30, 20, 18, 22, 15, 12, 14, 10, 9, 8],
            "Sales Vol": [100, 110, 105, 120, 125, 130, 135, 140, 145, 150, 155, 160]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['Returns'], mode='lines+markers', name='Returns', line=dict(color='#FF0055', width=3)))
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['Complaints'], mode='lines+markers', name='Complaints', line=dict(color='#FFAA00', width=3)))
        fig.add_trace(go.Bar(x=df_chart['Date'], y=df_chart['Sales Vol'], name='Sales Volume', marker_color='rgba(0, 198, 215, 0.2)'))
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 🛰️ Live Activity")
        activities = [
            ("10:42 AM", "CAPA-2025-004 Review", "🟢"),
            ("10:38 AM", "Odoo Feed (1.2k rows)", "🔵"),
            ("10:15 AM", "Alert: SKU MOB1025", "🔴"),
            ("09:55 AM", "Plan: 'Mobility X'", "⚪"),
            ("09:30 AM", "Amazon VoC Parse", "🔵")
        ]
        for time_str, action, icon in activities:
            st.markdown(f"""
            <div style="background: #0f172a; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #00C6D7;">
                <small style="color: #00C6D7;">{time_str}</small><br>
                <span style="color: white; font-weight:600;">{icon} {action}</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        if st.button("🚀 Quick Launch: Plan", use_container_width=True):
            st.session_state.nav = "Quality Planning"
            st.rerun()

def render_quality_planning():
    st.markdown("# 🛠️ Quality Strategy")
    
    keys = ["qp_name", "qp_risk", "qp_scope", "qp_regs", "qp_testing", "qp_vendor", "qp_path"]
    for k in keys:
        if k not in st.session_state: st.session_state[k] = ""
    if not st.session_state.qp_risk: st.session_state.qp_risk = "Class I"

    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        st.session_state.qp_name = c1.text_input("Project Name", st.session_state.qp_name)
        st.session_state.qp_risk = c2.selectbox("Risk Class", ["Class I", "Class II", "Class III"], index=0)
        mkts = c3.multiselect("Markets", ["USA", "EU", "UK", "CAN"], default=["USA"])

    st.divider()

    col_left, col_right = st.columns([1.3, 1])
    sections = [
        ("scope", "Scope", "Boundaries & Deliverables"),
        ("regs", "Regulatory", "ISO 13485 / FDA / MDR"),
        ("testing", "Validation", "Mechanical & Bio Tests"),
        ("vendor", "Vendor Controls", "IQC & Audits"),
        ("path", "Critical Path", "Milestones")
    ]

    with col_left:
        st.markdown("### 📝 Drafting")
        locks = {}
        for code, title, hint in sections:
            with st.expander(f"{title}", expanded=True):
                c_lock, c_txt = st.columns([0.3, 1])
                with c_lock:
                    locks[code] = st.checkbox(f"Lock", key=f"lock_{code}")
                with c_txt:
                    st.session_state[f"qp_{code}"] = st.text_area("Content", value=st.session_state[f"qp_{code}"], placeholder=hint, height=100, key=f"input_{code}", label_visibility="collapsed")
        
        if st.button("✨ AI: Generate Plan", type="primary", use_container_width=True):
            with st.spinner("AI Processing..."):
                context = f"Project: {st.session_state.qp_name}. Risk: {st.session_state.qp_risk}. Markets: {mkts}.\n"
                for code, title, _ in sections:
                    if st.session_state[f"qp_{code}"]: context += f"User: {st.session_state[f'qp_{code}']}\n"
                for code, title, _ in sections:
                    if not locks[code]:
                        prompt = f"Write Quality Plan section '{title}'. Context: {context}. No markdown."
                        st.session_state[f"qp_{code}"] = st.session_state.ai.generate_text(prompt)
                        time.sleep(0.5)
                st.success("Done!")
                st.rerun()

    with col_right:
        st.markdown("### 📄 Preview")
        full_doc = f"QUALITY PLAN: {st.session_state.qp_name.upper()}\nDATE: {datetime.now().date()}\nRISK: {st.session_state.qp_risk}\n\n"
        for code, title, _ in sections:
            full_doc += f"{title.upper()}\n{'-'*len(title)}\n{DataProcessor.clean_text_for_export(st.session_state[f'qp_{code}'])}\n\n"
        st.text_area("Output", full_doc, height=600)
        st.download_button("📥 Download .TXT", full_doc, file_name=f"{st.session_state.qp_name}.txt", use_container_width=True)

def render_market_intel():
    st.markdown("# 🌐 Market Intelligence")
    st.caption("VISION & SENTIMENT ANALYSIS")
    
    tab1, tab2 = st.tabs(["👁️ Vision", "📊 Data"])
    
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            img_file = st.file_uploader("Upload Screenshot", type=['png', 'jpg'])
            if img_file:
                img = Image.open(img_file)
                st.image(img, caption="Target", use_column_width=True)
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Scanning..."):
                        st.session_state.voc_analysis = st.session_state.ai.analyze_image(img, "Extract metrics (Stars, Return %), Top 3 Defects, Sentiment, and Action Plan.")
                        st.session_state.voc_context = st.session_state.voc_analysis
        with c2:
            if 'voc_analysis' in st.session_state:
                st.markdown("### 🤖 Analysis")
                st.write(st.session_state.voc_analysis)
                st.divider()
                if st.button("🚨 Create CAPA"):
                    st.session_state.capa_prefill = st.session_state.voc_analysis
                    st.session_state.nav = "CAPA Manager"
                    st.rerun()
                if st.button("💬 Chat"): st.session_state.chat_mode = True

            if st.session_state.get("chat_mode"):
                q = st.text_input("Question:")
                if q: st.info(st.session_state.ai.generate_text(f"Context: {st.session_state.voc_context}. Q: {q}"))

    with tab2:
        st.markdown("### 📊 Structured Data")
        if 'voc_analysis' in st.session_state:
            df_sim = pd.DataFrame({"Defect": ["Broken", "Missing", "Comfort"], "Count": [12, 8, 5]})
            fig = px.bar(df_sim, x="Defect", y="Count", color="Count", color_continuous_scale=["#00C6D7", "#FF0055"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Analyze image first.")

def render_supply_chain():
    st.markdown("# 📦 Supply Chain")
    f = st.file_uploader("Upload Odoo File", type=['csv', 'xlsx'])
    
    if f:
        with st.spinner("Processing..."):
            df, dtype, err = DataProcessor.parse_odoo_file(f)
        
        if err: st.error(f"Error: {err}")
        else:
            st.success(f"Type: **{dtype}** ({len(df)} rows)")
            
            if dtype == "Inventory":
                df = DataProcessor.calculate_inventory_metrics(df)
                c1, c2 = st.columns([2, 1])
                with c1:
                    if 'Risk Level' in df.columns:
                        fig = px.pie(df, names='Risk Level', title="Stock Health", hole=0.5, color_discrete_sequence=["#FF0055", "#00C6D7", "#FFAA00"])
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if st.button("AI Insight"):
                        st.info(st.session_state.ai.generate_text(f"Analyze: {df.head(10).to_string()}"))

            elif dtype == "Helpdesk":
                if 'Status' in df.columns:
                    fig = px.bar(df['Status'].value_counts(), title="Tickets by Status", color_discrete_sequence=["#00C6D7"])
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw Data"):
                st.dataframe(df)

def render_capa():
    st.markdown("# 🛡️ CAPA Manager")
    desc = st.session_state.get("capa_prefill", "")
    if desc: 
        st.info("Prefilled from VoC")
        del st.session_state.capa_prefill

    tabs = st.tabs(["Intake", "RCA", "Action", "Close"])
    
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.text_input("CAPA ID", f"CAPA-{int(time.time())}", disabled=True)
        c2.select_slider("Risk", options=["Minor", "Major", "Critical"])
        st.text_area("Description", value=desc)

    with tabs[1]:
        w1 = st.text_input("1. Why?")
        if st.button("🤖 AI Coach"):
            if w1: st.write(st.session_state.ai.generate_text(f"Root cause for: {w1}"))
            
    with tabs[3]:
        if st.button("Close CAPA", type="primary"):
            st.balloons()

# --- 5. MAIN CONTROLLER ---
def main():
    with st.sidebar:
        st.title("PLI SYSTEM")
        st.caption("VIVE HEALTH v3.1")
        
        # AI KEY FALLBACK INPUT
        if not st.session_state.ai.available:
            st.error("🔴 AI Offline")
            key_input = st.text_input("Enter Google Gemini API Key:", type="password")
            if key_input:
                st.session_state.ai.configure_ai(key_input)
                st.rerun()
        else:
            st.success("🟢 AI Online")

        st.markdown("---")
        
        menu = ["Dashboard", "Quality Planning", "Market Intelligence", "Supply Chain", "CAPA Manager"]
        icons = ["📊", "🛠️", "🌐", "📦", "🛡️"]
        
        if 'nav' not in st.session_state: st.session_state.nav = "Dashboard"
        
        for i, item in enumerate(menu):
            if st.button(f"{icons[i]}  {item}", key=item, use_container_width=True):
                st.session_state.nav = item
                st.rerun()

    if st.session_state.nav == "Dashboard": render_dashboard()
    elif st.session_state.nav == "Quality Planning": render_quality_planning()
    elif st.session_state.nav == "Market Intelligence": render_market_intel()
    elif st.session_state.nav == "Supply Chain": render_supply_chain()
    elif st.session_state.nav == "CAPA Manager": render_capa()

if __name__ == "__main__":
    main()
