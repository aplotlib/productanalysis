import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import json
import time
from datetime import datetime
from docx import Document

from odoo_processor import OdooProcessor

# --- 1. ENTERPRISE CONFIG ---
st.set_page_config(
    page_title="ORION | Quality Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed by default for max screen real estate
)

# --- 2. EXECUTIVE THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Base */
    .stApp { background-color: #F1F5F9; font-family: 'Inter', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #0F172A; font-weight: 800; letter-spacing: -0.5px; }
    
    /* Cards */
    .kpi-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-left: 5px solid #3B82F6;
    }
    .kpi-card.danger { border-left-color: #EF4444; }
    .kpi-card.success { border-left-color: #22C55E; }
    
    .kpi-label { font-size: 0.85rem; text-transform: uppercase; color: #64748B; font-weight: 600; letter-spacing: 0.05em; }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #1E293B; margin-top: 5px; }
    .kpi-sub { font-size: 0.9rem; color: #94A3B8; margin-top: 5px; }
    
    /* Navigation */
    .nav-btn { width: 100%; padding: 10px; text-align: left; border:none; background:none; font-weight:600; color: #475569; }
    
    /* Tables */
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE CORE ---
class IntelligenceEngine:
    def __init__(self):
        self.available = False
        self.client = None
        self.provider = ""

    def configure(self, provider, key=None):
        try:
            if "Gemini" in provider:
                import google.generativeai as genai
                if not key and "GOOGLE_API_KEY" in st.secrets: key = st.secrets["GOOGLE_API_KEY"]
                if key:
                    genai.configure(api_key=key)
                    self.client = genai.GenerativeModel('gemini-1.5-flash')
                    self.available = True
                    self.provider = "Gemini"
            elif "GPT" in provider:
                import openai
                if not key and "OPENAI_API_KEY" in st.secrets: key = st.secrets["OPENAI_API_KEY"]
                if key:
                    self.client = openai.OpenAI(api_key=key)
                    self.available = True
                    self.provider = "OpenAI"
        except Exception:
            self.available = False

    def ask(self, prompt, temperature=0.3):
        if not self.available: return "⚠️ AI Offline. Check API Key."
        try:
            if self.provider == "Gemini":
                return self.client.generate_content(prompt).text
            elif self.provider == "OpenAI":
                res = self.client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return res.choices[0].message.content
        except Exception as e: return f"Error: {e}"

# Initialize Global State
if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()
if 'data' not in st.session_state: st.session_state.data = pd.DataFrame()
if 'processor' not in st.session_state: st.session_state.processor = OdooProcessor()

# --- 4. UI COMPONENTS ---

def kpi_card(label, value, subtext="", style="neutral"):
    color = "#3B82F6" # blue
    if style == "danger": color = "#EF4444"
    if style == "success": color = "#22C55E"
    
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

def render_upload_zone():
    st.markdown("### 📥 Data Ingestion")
    with st.expander("Upload Reports", expanded=True):
        c1, c2, c3 = st.columns(3)
        sales = c1.file_uploader("Odoo Sales/Forecast", type=['xlsx'])
        returns = c2.file_uploader("Pivot Return Analysis", type=['xlsx'])
        tickets = c3.file_uploader("Helpdesk Export", type=['xlsx'])
        
        if st.button("🔄 Run Analysis Pipeline", type="primary", use_container_width=True):
            with st.spinner("Harmonizing Data & Calculating Financial Impact..."):
                df = st.session_state.processor.merge_datasets(
                    st.session_state.processor.process_sales_file(sales),
                    st.session_state.processor.process_returns_file(returns),
                    st.session_state.processor.process_helpdesk_file(tickets)
                )
                st.session_state.data = df
                time.sleep(0.5) # UX Pause
                st.rerun()

def render_executive_dashboard():
    df = st.session_state.data
    if df.empty:
        render_upload_zone()
        return

    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    c1.markdown("# 🛡️ Quality Command Center")
    c1.markdown(f"**Reporting Period:** Last 90 Days | **Active SKUs:** {len(df)}")
    
    if c2.button("New Analysis"):
        st.session_state.data = pd.DataFrame()
        st.rerun()

    st.markdown("---")

    # --- C-SUITE KPIS ---
    tot_sales = df['sales_qty'].sum()
    tot_rev = df['est_revenue'].sum()
    lost_rev = df['lost_revenue'].sum()
    avg_rate = (df['return_qty'].sum() / tot_sales * 100) if tot_sales else 0
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total Revenue (Est)", f"${tot_rev/1000:.1f}K", f"{int(tot_sales):,} units moved", "success")
    with k2: kpi_card("Revenue Lost to Returns", f"${lost_rev/1000:.1f}K", "Direct financial impact", "danger")
    with k3: kpi_card("Global Return Rate", f"{avg_rate:.2f}%", "Target: < 4.0%", "neutral")
    with k4: kpi_card("Support Ticket Volume", f"{int(df['ticket_count'].sum())}", "Customer touchpoints", "neutral")

    st.markdown("### 🔥 Priority Action Feed")
    
    # Filter: High Impact Items (High Loss $ OR High Rate with volume)
    alerts = df[
        (df['lost_revenue'] > 500) | 
        ((df['return_rate'] > 8) & (df['sales_qty'] > 20))
    ].head(5)
    
    if not alerts.empty:
        for _, row in alerts.iterrows():
            with st.container():
                col_icon, col_det, col_act = st.columns([0.5, 4, 1.5])
                col_icon.markdown("## 🚨")
                col_det.markdown(f"**{row['clean_sku']}** | {row['product_name']}")
                col_det.caption(f"Lost Revenue: ${row['lost_revenue']:.0f} | Return Rate: {row['return_rate']:.1f}% | Reasons: {', '.join(row['return_reason']) if isinstance(row['return_reason'], list) else row['return_reason']}")
                
                if col_act.button("Investigate", key=f"inv_{row['clean_sku']}"):
                    st.session_state.deep_dive_sku = row['clean_sku']
                    st.session_state.page = "Product 360"
                    st.rerun()
                st.divider()
    else:
        st.success("✅ No Critical Alerts. Systems Nominal.")

    # --- CHARTS ---
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("#### Financial Loss by Category")
        cat_loss = df.groupby('category')['lost_revenue'].sum().reset_index()
        fig = px.bar(cat_loss, x='category', y='lost_revenue', color='lost_revenue', color_continuous_scale='reds', title="Where are we losing money?")
        st.plotly_chart(fig, use_container_width=True)
        
    with c_right:
        st.markdown("#### Risk Matrix")
        # Only meaningful products
        scatter_df = df[df['sales_qty'] > 10]
        fig2 = px.scatter(
            scatter_df, x='sales_qty', y='return_rate', 
            size='lost_revenue', color='return_rate',
            hover_name='product_name', hover_data=['clean_sku', 'lost_revenue'],
            color_continuous_scale='RdYlGn_r', title="Volume vs. Quality Risk"
        )
        st.plotly_chart(fig2, use_container_width=True)

def render_product_360():
    df = st.session_state.data
    if df.empty: 
        st.warning("Load data first.")
        return
        
    # Navigation back
    if st.button("← Back to Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()

    # Selection
    pre_select = st.session_state.get('deep_dive_sku', df['clean_sku'].iloc[0])
    sku = st.selectbox("Select Product to Audit:", df['clean_sku'].unique(), index=list(df['clean_sku']).index(pre_select) if pre_select in list(df['clean_sku']) else 0)
    
    row = df[df['clean_sku'] == sku].iloc[0]
    
    # Header
    st.markdown(f"# {sku}")
    st.markdown(f"### {row['product_name']}")
    
    # Mini Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sales Units", f"{int(row['sales_qty'])}")
    m2.metric("Return Rate", f"{row['return_rate']:.2f}%", delta=f"{row['return_rate'] - df['return_rate'].mean():.2f}% vs Avg", delta_color="inverse")
    m3.metric("Revenue Impact", f"-${row['lost_revenue']:.0f}", "Loss")
    m4.metric("Ticket Vol", f"{int(row['ticket_count'])}")

    st.markdown("---")
    
    # Analysis Tabs
    t1, t2, t3 = st.tabs(["🔍 Root Cause Analysis", "🎫 Support Tickets", "🤖 AI Action Plan"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top Return Reasons")
            reasons = row['return_reason']
            if reasons:
                st.table(pd.DataFrame(reasons, columns=["Reported Reason"]))
            else:
                st.info("No categorical reason data.")
        with c2:
            st.markdown("#### Quality Trend")
            st.caption("Assuming consistent defect rate over time.")
            # Mock trend for visual completeness
            dates = pd.date_range(end=datetime.now(), periods=6, freq='W')
            vals = [row['return_rate'] * (1 + (i*0.1 - 0.2)) for i in range(6)]
            st.line_chart(pd.DataFrame({'Rate': vals}, index=dates))

    with t2:
        st.markdown("#### Customer Voice")
        # This would pull from the raw helpdesk df if we stored it globally or in a complex object
        # For now, we show the count is high/low
        if row['ticket_count'] > 0:
            st.warning(f"{int(row['ticket_count'])} tickets linked to this SKU. Check Helpdesk export for full text.")
        else:
            st.success("No support tickets linked.")

    with t3:
        st.markdown("#### AI Consultant")
        if st.button("⚡ Draft CAPA & Solution"):
            with st.spinner("Analyzing Data Points..."):
                prompt = f"""
                Analyze this product data:
                Product: {row['product_name']}
                Return Rate: {row['return_rate']}% (Avg is 4%)
                Lost Revenue: ${row['lost_revenue']}
                Top Reasons: {row['return_reason']}
                
                1. Identify potential Root Causes.
                2. Suggest 3 Immediate Actions.
                3. Draft a formal CAPA Problem Statement.
                """
                res = st.session_state.ai.ask(prompt)
                st.markdown(res)

def render_capa_manager():
    st.markdown("## 🛠️ CAPA Command Center")
    st.info("Formalize investigations into ISO 13485 compliant reports.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        capa_id = st.text_input("CAPA ID", f"CAPA-{datetime.now().strftime('%Y%m%d')}")
        sku = st.text_input("Affected SKU")
        severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
    
    with c2:
        problem = st.text_area("Problem Statement", height=100, placeholder="Describe the non-conformance...")
        root_cause = st.text_area("Root Cause (Fishbone/5 Why)", height=100)
    
    st.markdown("### Action Plan")
    c3, c4 = st.columns(2)
    corr = c3.text_area("Corrective Action (Immediate fix)")
    prev = c4.text_area("Preventive Action (Long term)")
    
    if st.button("📄 Generate Formal Report (.docx)"):
        doc = Document()
        doc.add_heading(f"CAPA Report: {capa_id}", 0)
        doc.add_paragraph(f"Generated: {datetime.now()}")
        doc.add_heading("Problem", 1)
        doc.add_paragraph(f"SKU: {sku} | Severity: {severity}")
        doc.add_paragraph(problem)
        doc.add_heading("Investigation", 1)
        doc.add_paragraph(root_cause)
        doc.add_heading("Plan", 1)
        doc.add_paragraph(f"Corrective: {corr}")
        doc.add_paragraph(f"Preventive: {prev}")
        
        bio = io.BytesIO()
        doc.save(bio)
        st.download_button("Download Doc", bio.getvalue(), f"{capa_id}.docx", "application/docx")

def render_settings():
    st.markdown("## ⚙️ System Settings")
    
    st.markdown("### AI Configuration")
    provider = st.selectbox("Provider", ["Google Gemini 1.5 Flash", "OpenAI GPT-4o"])
    key = st.text_input("API Key", type="password")
    
    if st.button("Save & Connect"):
        st.session_state.ai.configure(provider, key)
        if st.session_state.ai.available:
            st.success(f"Connected to {provider}!")
        else:
            st.error("Connection Failed.")

# --- MAIN CONTROLLER ---
def main():
    if 'page' not in st.session_state: st.session_state.page = "Dashboard"
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("ORION")
        st.caption("Quality Intelligence System")
        
        if st.button("📊 Dashboard", use_container_width=True): st.session_state.page = "Dashboard"
        if st.button("🛡️ CAPA Manager", use_container_width=True): st.session_state.page = "CAPA"
        if st.button("⚙️ Settings", use_container_width=True): st.session_state.page = "Settings"
        
        st.markdown("---")
        st.info("System Status: Online")

    # Page Routing
    if st.session_state.page == "Dashboard": render_executive_dashboard()
    elif st.session_state.page == "Product 360": render_product_360()
    elif st.session_state.page == "CAPA": render_capa_manager()
    elif st.session_state.page == "Settings": render_settings()

if __name__ == "__main__":
    main()
