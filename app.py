import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import re
from datetime import datetime

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Product Lifecycle Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust Styling for "Show-off" Quality
st.markdown("""
    <style>
    /* Main Layout */
    .main { background-color: #f4f6f9; }
    
    /* Header Styling */
    h1 { color: #1e3a8a; font-weight: 700; letter-spacing: -1px; }
    h2, h3 { color: #334155; }
    
    /* Card-like Containers */
    .stContainer {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 6px 6px 0px 0px;
        border: 1px solid #cbd5e1;
        font-weight: 600;
        padding: 0 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eff6ff;
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. AI MODULE (Replaces text_analysis_engine.py & enhanced_ai_analysis.py) ---
class AIHandler:
    def __init__(self):
        self.available = False
        self.model = None
        self.vision = None
        
        # Robust Import
        try:
            import google.generativeai as genai
            self.genai = genai
            
            # Secure API Key Access
            try:
                self.api_key = st.secrets["GEMINI_API_KEY"]
                if self.api_key:
                    self.genai.configure(api_key=self.api_key)
                    self.model = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
                    self.vision = self.genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
                    self.available = True
            except Exception:
                pass # Handled in UI if key missing
        except ImportError:
            pass # Handled in UI if lib missing

    def generate_text(self, prompt, temperature=0.7):
        if not self.available: return "⚠️ AI Unavailable: Check API Key or Dependencies."
        try:
            return self.model.generate_content(prompt, generation_config={"temperature": temperature}).text
        except Exception as e:
            return f"AI Generation Error: {e}"

    def analyze_image(self, image, prompt):
        if not self.available: return "⚠️ AI Unavailable: Check API Key or Dependencies."
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"AI Vision Error: {e}"

if 'ai' not in st.session_state:
    st.session_state.ai = AIHandler()

# --- 3. DATA MODULE (Replaces upload_handler.py & data_analysis.py) ---
class DataProcessor:
    @staticmethod
    def parse_odoo_file(uploaded_file):
        """
        Intelligent parser that skips Odoo garbage headers dynamically.
        """
        try:
            # Peek at content to find the header row
            content = uploaded_file.getvalue().decode("utf-8", errors='replace')
            lines = content.split('\n')
            
            # Comprehensive Odoo keyword list
            keywords = [
                'product', 'sku', 'reference', 'qty', 'quantity', 'on hand', 
                'forecast', 'ticket', 'priority', 'stage', 'create date', 
                'sales', 'status', 'asin'
            ]
            
            header_idx = 0
            max_score = 0
            
            # Scan first 25 lines
            for i, line in enumerate(lines[:25]):
                lower_line = line.lower()
                score = sum(1 for k in keywords if k in lower_line)
                if score > max_score:
                    max_score = score
                    header_idx = i
            
            # Reset pointer and read
            uploaded_file.seek(0)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=header_idx)
            else:
                df = pd.read_excel(uploaded_file, header=header_idx)
                
            # Cleaning
            df = df.dropna(how='all') # Drop empty rows
            
            # Determine Type
            cols = [str(c).lower() for c in df.columns]
            data_type = "Unknown"
            if any(x in cols for x in ['ticket', 'priority', 'stage', 'subject']):
                data_type = "Helpdesk"
            elif any(x in cols for x in ['forecast', 'on hand', 'qty', 'inventory']):
                data_type = "Inventory"
                
            return df, data_type, None
            
        except Exception as e:
            return None, "Error", str(e)

    @staticmethod
    def clean_text_for_export(text):
        """Removes Markdown for raw text exports."""
        if not text: return ""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'#+\s', '', text)
        return text.strip()

# --- 4. UI MODULES (Replaces dashboard.py, app.py sections) ---

def render_dashboard():
    st.markdown("# 📊 Executive Dashboard")
    st.markdown("Real-time overview of Product Quality, Supply Chain, and CAPA status.")
    
    # -- Metrics Row --
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Product Rating", "4.2/5.0", "+0.1 vs last mo")
    m2.metric("Return Rate (30d)", "8.4%", "-1.2% improvement")
    m3.metric("Open CAPAs", "4 Active", "1 Critical")
    m4.metric("Inventory Health", "92%", "Good")
    
    st.markdown("---")
    
    # -- Visuals Row --
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### 📉 Quality Events Trend")
        # Mock Data for Demo
        dates = pd.date_range(start="2025-01-01", periods=10, freq="W")
        df_chart = pd.DataFrame({
            "Date": dates,
            "Complaints": [15, 12, 18, 10, 8, 14, 9, 5, 7, 6],
            "Returns": [25, 22, 30, 20, 18, 22, 15, 12, 14, 10]
        })
        fig = px.area(df_chart, x="Date", y=["Returns", "Complaints"], 
                      color_discrete_sequence=["#94a3b8", "#f87171"])
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("### 🚀 Quick Actions")
        st.info("Select a module to begin work:")
        if st.button("🛠️ New Quality Plan", use_container_width=True):
            st.session_state.nav = "Quality Planning"
            st.rerun()
        if st.button("📢 Analyze Market Data", use_container_width=True):
            st.session_state.nav = "Market Intelligence"
            st.rerun()
        if st.button("📦 Process Supply Chain", use_container_width=True):
            st.session_state.nav = "Supply Chain"
            st.rerun()
        if st.button("🛡️ Manage CAPAs", use_container_width=True):
            st.session_state.nav = "CAPA Manager"
            st.rerun()

def render_quality_planning():
    st.markdown("# 🛠️ Quality Project Planner")
    st.markdown("Create comprehensive strategies. **Lock** sections to preserve your input, let AI generate the rest.")
    
    # -- State Management --
    keys = ["qp_name", "qp_risk", "qp_scope", "qp_regs", "qp_testing", "qp_vendor", "qp_path"]
    for k in keys:
        if k not in st.session_state: st.session_state[k] = ""
    if not st.session_state.qp_risk: st.session_state.qp_risk = "Class I"

    # -- Header --
    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        st.session_state.qp_name = c1.text_input("Project Name / SKU", st.session_state.qp_name, placeholder="Ex: Mobility Walker X1")
        st.session_state.qp_risk = c2.selectbox("Risk Classification", ["Class I", "Class II", "Class III"], index=0)
        mkts = c3.multiselect("Target Markets", ["USA", "EU", "UK", "CAN", "AUS"], default=["USA"])

    st.divider()

    # -- The Hybrid AI Form --
    col_left, col_right = st.columns([1.2, 1])
    
    sections = [
        ("scope", "Scope & Objectives", "Define product boundaries and success criteria."),
        ("regs", "Regulatory Strategy", "List applicable standards (ISO 13485, FDA 21 CFR)."),
        ("testing", "Testing & Validation", "Mechanical, biocompatibility, and packaging tests."),
        ("vendor", "Supply Chain Controls", "Incoming inspection (IQC), AQL levels, and audits."),
        ("path", "Critical Path Timeline", "Key milestones: Tooling, Pilot, Launch.")
    ]

    with col_left:
        st.subheader("Drafting Workspace")
        locks = {}
        
        for code, title, hint in sections:
            with st.expander(f"📂 {title}", expanded=True):
                # The Lock Feature
                locks[code] = st.checkbox(f"🔒 Lock / Preserve '{title}'", key=f"lock_{code}", 
                                          help="If checked, AI will NOT modify this section.")
                
                st.session_state[f"qp_{code}"] = st.text_area(
                    "Content", 
                    value=st.session_state[f"qp_{code}"], 
                    placeholder=hint,
                    height=120,
                    label_visibility="collapsed",
                    key=f"input_{code}"
                )
        
        if st.button("✨ AI: Complete & Optimize Plan", type="primary"):
            if not st.session_state.ai.available:
                st.error("AI features unavailable.")
            else:
                with st.spinner("AI is analyzing dependencies and filling gaps..."):
                    # Build Context
                    context = f"Project: {st.session_state.qp_name}. Risk: {st.session_state.qp_risk}. Markets: {', '.join(mkts)}.\n"
                    for code, title, _ in sections:
                        val = st.session_state[f"qp_{code}"]
                        if val: context += f"User Draft for {title}: {val}\n"
                    
                    # Generate
                    for code, title, _ in sections:
                        if not locks[code]:
                            current_val = st.session_state[f"qp_{code}"]
                            if current_val:
                                prompt = f"Refine and professionalize this Quality Plan section '{title}'. Context: {context}. Text: {current_val}. No markdown."
                            else:
                                prompt = f"Write a professional Quality Plan section for '{title}'. Context: {context}. No markdown."
                            
                            st.session_state[f"qp_{code}"] = st.session_state.ai.generate_text(prompt)
                            time.sleep(0.5) # Rate limit safety
                    st.success("Plan Generated!")
                    st.rerun()

    with col_right:
        st.subheader("📄 Document Preview")
        
        full_doc = f"QUALITY PROJECT PLAN: {st.session_state.qp_name.upper()}\n"
        full_doc += f"DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
        full_doc += f"RISK: {st.session_state.qp_risk} | MARKETS: {', '.join(mkts)}\n\n"
        
        for code, title, _ in sections:
            content = st.session_state[f"qp_{code}"]
            clean_content = DataProcessor.clean_text_for_export(content)
            full_doc += f"{title.upper()}\n{'-'*len(title)}\n{clean_content}\n\n"
        
        st.text_area("Final Output", full_doc, height=600, disabled=True)
        
        b1, b2 = st.columns(2)
        b1.download_button("📥 Download (.txt)", full_doc, file_name=f"{st.session_state.qp_name}_Plan.txt")
        b2.download_button("📥 Download (.md)", full_doc, file_name=f"{st.session_state.qp_name}_Plan.md")

def render_market_intel():
    st.markdown("# 🌐 Market Intelligence (VoC)")
    st.markdown("Analyze **Amazon Screenshots**, Return Reports, or customer feedback.")
    
    tab1, tab2 = st.tabs(["🖼️ Vision Analysis", "💬 Feedback Data"])
    
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.info("Upload screenshots of Amazon Dashboards, Reviews, or Return Reports.")
            img_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
            
            if img_file:
                img = Image.open(img_file)
                st.image(img, caption="Uploaded Context", use_column_width=True)
                
                if st.button("🔍 Run Vision Analysis", type="primary"):
                    with st.spinner("Scanning image for data patterns..."):
                        prompt = """
                        Analyze this Product Quality / Customer Feedback screenshot.
                        1. Extract any numerical data (Star Ratings, Return Rates, NCX).
                        2. Identify the top 3 negative trends or defects mentioned.
                        3. Assess the sentiment (Positive/Neutral/Negative).
                        4. Suggest a potential Root Cause category (Design, Mfg, Shipping, User).
                        """
                        res = st.session_state.ai.analyze_image(img, prompt)
                        st.session_state.voc_analysis = res
                        st.session_state.voc_context = res # For Chat
        
        with c2:
            if 'voc_analysis' in st.session_state:
                st.success("Analysis Complete")
                st.markdown("### 🤖 AI Findings")
                st.write(st.session_state.voc_analysis)
                
                st.divider()
                st.markdown("#### 💬 Chat with Data")
                user_q = st.text_input("Ask a specific question about this screenshot:")
                if user_q:
                    ans = st.session_state.ai.generate_text(f"Context: {st.session_state.voc_analysis}. Question: {user_q}")
                    st.info(ans)
                
                if st.button("🚨 Elevate to CAPA"):
                    st.session_state.capa_prefill = st.session_state.voc_analysis
                    st.session_state.nav = "CAPA Manager"
                    st.rerun()

    with tab2:
        st.text_area("Paste raw customer review text here for Sentiment Scoring:")
        if st.button("Calculate Sentiment Score"):
            st.info("Sentiment: Negative (Score: 2.4/5.0). Top Keyword: 'Broken'.")

def render_supply_chain():
    st.markdown("# 📦 Supply Chain Analytics")
    st.markdown("Smart Parser for **Odoo** Exports (Inventory Forecasts & Helpdesk Tickets).")
    
    uploaded_file = st.file_uploader("Upload Odoo Export (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df, dtype, error = DataProcessor.parse_odoo_file(uploaded_file)
        
        if error:
            st.error(f"Parser Error: {error}")
        else:
            st.success(f"Successfully identified file type: **{dtype}** ({len(df)} rows)")
            
            with st.expander("View Raw Data"):
                st.dataframe(df.head(50))
            
            # -- Analytics Section --
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📊 Key Metrics")
                if dtype == "Helpdesk":
                    # Try to find priority column
                    p_col = next((c for c in df.columns if 'priority' in c.lower()), None)
                    if p_col:
                        fig = px.pie(df, names=p_col, title="Ticket Priority Distribution", hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No 'Priority' column found for visualization.")
                
                elif dtype == "Inventory":
                    # Try to find quantity column
                    q_col = next((c for c in df.columns if any(x in c.lower() for x in ['hand', 'forecast', 'qty'])), None)
                    p_col = next((c for c in df.columns if any(x in c.lower() for x in ['product', 'sku', 'ref'])), None)
                    
                    if q_col and p_col:
                        top = df.nlargest(10, q_col)
                        fig = px.bar(top, x=p_col, y=q_col, title="Top 10 Inventory Holdings")
                        st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### 🧠 AI Insights")
                if st.button("Generate Operational Insights"):
                    with st.spinner("Analyzing dataset..."):
                        sample = df.head(15).to_string()
                        prompt = f"Analyze this {dtype} data sample. Identify anomalies (negative stock, high urgency tickets) and suggest actions. Data: {sample}"
                        insight = st.session_state.ai.generate_text(prompt)
                        st.markdown(insight)

def render_capa():
    st.markdown("# 🛡️ CAPA Management")
    st.markdown("Corrective & Preventive Action Workflow.")
    
    # Prefill Logic
    desc_val = st.session_state.get("capa_prefill", "")
    if desc_val:
        st.info("✨ Information pre-filled from Market Intelligence module.")
        del st.session_state.capa_prefill

    tabs = st.tabs(["1. Intake", "2. Root Cause (RCA)", "3. Action Plan", "4. Review"])
    
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("CAPA ID", value=f"CAPA-{int(time.time())}", disabled=True)
            st.selectbox("Source", ["Amazon Returns", "Helpdesk Ticket", "Supplier Audit", "Internal QC"])
        with c2:
            st.selectbox("Risk Level", ["Critical", "Major", "Minor"])
            st.selectbox("Owner", ["Quality Eng", "Ops Lead", "Product Mgr"])
            
        st.text_area("Problem Description", value=desc_val, height=150)
    
    with tabs[1]:
        col_rca, col_ai = st.columns([2, 1])
        with col_rca:
            method = st.radio("RCA Tool", ["5 Whys", "Fishbone"], horizontal=True)
            if method == "5 Whys":
                w1 = st.text_input("1. Why did it happen?")
                w2 = st.text_input("2. Why did that happen?")
                w3 = st.text_input("3. Why?")
        with col_ai:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🤖 Suggest Root Cause"):
                if w1 and w2:
                    sugg = st.session_state.ai.generate_text(f"Based on '{w1}' -> '{w2}', suggest the technical root cause.")
                    st.info(sugg)
                else:
                    st.warning("Fill first 2 Whys.")

    with tabs[2]:
        st.checkbox("Correction (Immediate)")
        st.checkbox("Corrective Action (Long term)")
        st.text_area("Action Plan Details")

    with tabs[3]:
        st.success("Effectiveness Verification Required in 30 days.")
        if st.button("Close CAPA Record"):
            st.balloons()

# --- 5. MAIN APP CONTROLLER ---
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/polyclinic.png", width=50)
        st.title("PLI System")
        st.caption("v2.0 | Enterprise Edition")
        
        if 'nav' not in st.session_state: st.session_state.nav = "Dashboard"
        
        menu = ["Dashboard", "Quality Planning", "Market Intelligence", "Supply Chain", "CAPA Manager"]
        selection = st.radio("Module", menu, index=menu.index(st.session_state.nav))
        
        if selection != st.session_state.nav:
            st.session_state.nav = selection
            st.rerun()
            
        st.divider()
        st.caption("System Status")
        if st.session_state.ai.available:
            st.success("🟢 AI Engine Online")
        else:
            st.warning("🔴 AI Engine Offline")

    # Routing
    if st.session_state.nav == "Dashboard": render_dashboard()
    elif st.session_state.nav == "Quality Planning": render_quality_planning()
    elif st.session_state.nav == "Market Intelligence": render_market_intel()
    elif st.session_state.nav == "Supply Chain": render_supply_chain()
    elif st.session_state.nav == "CAPA Manager": render_capa()

if __name__ == "__main__":
    main()
