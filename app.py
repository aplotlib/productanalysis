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

# Try to import the processor; warn if missing to prevent crash on load
try:
    from odoo_processor import OdooProcessor
except ImportError:
    st.error("⚠️ Critical: `odoo_processor.py` is missing. Please create this file.")
    st.stop()

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="ORION | Operational Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ENTERPRISE CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }
    
    /* Metric Cards */
    .metric-card {
        background: white; border-radius: 8px; padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    .metric-label { color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #0F172A; font-size: 2rem; font-weight: 800; margin-top: 8px; }
    .metric-delta { font-size: 0.9rem; margin-top: 4px; font-weight: 500; }
    .positive { color: #16A34A; }
    .negative { color: #DC2626; }
    
    /* Alert Strip */
    .alert-strip {
        background-color: #FEF2F2; border-left: 4px solid #EF4444;
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
    }
    
    /* Global Headers */
    h1, h2, h3 { color: #0F172A; font-weight: 700; letter-spacing: -0.02em; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE CORE (AI) ---
class IntelligenceEngine:
    def __init__(self):
        self.available = False
        self.client = None
        self.provider = ""

    def configure(self, provider, key=None):
        """Configures the AI Client based on user selection."""
        try:
            self.available = False # Reset first
            if "Gemini" in provider:
                import google.generativeai as genai
                # Auto-detect key from secrets if not provided manually
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
        except Exception as e:
            st.error(f"AI Configuration Failed: {str(e)}")
            self.available = False

    def ask(self, prompt):
        """Sends a prompt to the configured AI."""
        if not self.available: return "⚠️ AI unavailable. Please configure API Key in Settings."
        try:
            if self.provider == "Gemini":
                return self.client.generate_content(prompt).text
            elif self.provider == "OpenAI":
                res = self.client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role": "user", "content": prompt}]
                )
                return res.choices[0].message.content
        except Exception as e: return f"Analysis Error: {e}"

# --- STATE MANAGEMENT & SELF-HEALING ---
if 'data' not in st.session_state: st.session_state.data = pd.DataFrame()
if 'processor' not in st.session_state: st.session_state.processor = OdooProcessor()

# HOTFIX: Detect if session has old 'IntelligenceEngine' object and reset it
if 'ai' not in st.session_state or not hasattr(st.session_state.ai, 'configure'):
    st.session_state.ai = IntelligenceEngine()
    # Try auto-connecting if secrets exist
    if "GOOGLE_API_KEY" in st.secrets: st.session_state.ai.configure("Gemini")
    elif "OPENAI_API_KEY" in st.secrets: st.session_state.ai.configure("GPT")

# --- 4. HELPER COMPONENTS ---

def metric_card(label, value, delta=None, is_bad=False):
    """Visual Component for KPIs."""
    delta_html = ""
    if delta:
        color_class = "negative" if is_bad else "positive"
        delta_html = f'<div class="metric-delta {color_class}">{delta}</div>'
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# --- 5. PAGE RENDERERS ---

def render_upload():
    st.markdown("### 📥 Data Pipeline")
    st.markdown("Upload your raw Odoo exports to generate the intelligence report.")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        sales = c1.file_uploader("Sales Forecast / Orders", type=['xlsx'], help="Export from Odoo Sales > Orders")
        returns = c2.file_uploader("Returns Pivot Report", type=['xlsx'], help="Export from Odoo Inventory > Reporting > Returns")
        tickets = c3.file_uploader("Helpdesk Export", type=['xlsx'], help="Export from Odoo Helpdesk > Tickets")
        
        if st.button("Run Analysis Pipeline", type="primary", use_container_width=True):
            if not sales and not returns:
                st.warning("⚠️ Minimum Requirement: Sales or Returns data.")
                return
                
            with st.status("Processing Data...", expanded=True) as status:
                st.write("Ingesting files...")
                try:
                    # Process files
                    s_df = st.session_state.processor.process_sales_file(sales) if sales else pd.DataFrame()
                    r_df = st.session_state.processor.process_returns_file(returns) if returns else pd.DataFrame()
                    h_df = st.session_state.processor.process_helpdesk_file(tickets) if tickets else pd.DataFrame()
                    
                    st.write("Harmonizing SKUs & Calculating Financials...")
                    df = st.session_state.processor.merge_datasets(s_df, r_df, h_df)
                    
                    st.session_state.data = df
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Pipeline Error: {str(e)}")
                    st.stop()

def render_dashboard():
    df = st.session_state.data
    if df.empty:
        render_upload()
        return

    # Header
    c1, c2 = st.columns([3, 1])
    c1.title("Operational Intelligence")
    if c2.button("New Analysis", use_container_width=True):
        st.session_state.data = pd.DataFrame()
        st.rerun()

    # KPIs
    tot_sales = df['sales_qty'].sum()
    tot_rev = df['est_revenue'].sum()
    lost_rev = df['lost_revenue'].sum()
    
    # Avoid division by zero
    global_rate = (df['return_qty'].sum() / tot_sales * 100) if tot_sales > 0 else 0.0
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Est. Revenue", f"${tot_rev/1000:.1f}K")
    with k2: metric_card("Financial Loss (Returns)", f"${lost_rev/1000:.1f}K", "Direct Impact", is_bad=True)
    with k3: metric_card("Global Return Rate", f"{global_rate:.2f}%", "Target: < 4.0%", is_bad=(global_rate > 4))
    with k4: metric_card("Active SKUs", f"{len(df)}")

    st.divider()

    # Critical Alerts
    st.subheader("🔥 Critical Alerts")
    # Logic: Loss > $500 OR Rate > 8% (min vol 20)
    alerts = df[
        (df['lost_revenue'] > 500) | 
        ((df['return_rate'] > 8) & (df['sales_qty'] > 20))
    ].head(5)

    if not alerts.empty:
        for _, row in alerts.iterrows():
            with st.container():
                c_det, c_stat, c_btn = st.columns([3, 2, 1])
                
                # Details
                c_det.markdown(f"**{row['clean_sku']}**")
                c_det.caption(str(row['product_name'])[:60] + "..." if len(str(row['product_name'])) > 60 else str(row['product_name']))
                
                # Stats
                reasons = ", ".join(row['return_reason']) if isinstance(row['return_reason'], list) else str(row['return_reason'])
                c_stat.markdown(f"Loss: **${row['lost_revenue']:.0f}** | Rate: **{row['return_rate']:.1f}%**")
                c_stat.caption(f"Top Issues: {reasons[:50]}...")
                
                # Action
                if c_btn.button("Inspect", key=f"btn_{row['clean_sku']}", use_container_width=True):
                    st.session_state.deep_dive_sku = row['clean_sku']
                    st.session_state.page = "Product 360"
                    st.rerun()
            st.divider()
    else:
        st.success("✅ No critical anomalies detected.")

    # Visuals
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Financial Impact by Category**")
        if 'category' in df.columns:
            cat_df = df.groupby('category')['lost_revenue'].sum().reset_index()
            fig = px.bar(cat_df, x='category', y='lost_revenue', color='lost_revenue', color_continuous_scale='reds')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with c_right:
        st.markdown("**Volume vs. Quality Risk**")
        scatter_df = df[df['sales_qty'] > 10] # Filter low volume noise
        if not scatter_df.empty:
            fig2 = px.scatter(
                scatter_df, x='sales_qty', y='return_rate', 
                size='lost_revenue', color='return_rate',
                hover_name='clean_sku', color_continuous_scale='RdYlGn_r'
            )
            fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)

def render_product_360():
    df = st.session_state.data
    if df.empty: return

    if st.button("← Back to Overview"):
        st.session_state.page = "Dashboard"
        st.rerun()

    # SKU Selector
    pre_select = st.session_state.get('deep_dive_sku', df['clean_sku'].iloc[0])
    # Ensure pre-select exists in current data
    if pre_select not in df['clean_sku'].values:
        pre_select = df['clean_sku'].iloc[0]
    
    sku = st.selectbox("Select Product", df['clean_sku'].unique(), index=list(df['clean_sku']).index(pre_select))
    row = df[df['clean_sku'] == sku].iloc[0]

    # Header
    st.title(f"{sku}")
    st.markdown(f"**{row['product_name']}**")

    # Stats Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Sales Units", int(row['sales_qty']))
    with c2: metric_card("Return Rate", f"{row['return_rate']:.2f}%", is_bad=row['return_rate']>5)
    with c3: metric_card("Financial Loss", f"${row['lost_revenue']:.0f}", is_bad=True)
    with c4: metric_card("Support Tickets", int(row['ticket_count']))

    st.divider()

    # Deep Dive
    c_data, c_ai = st.columns([1, 1])
    
    with c_data:
        st.subheader("📊 Data Profile")
        st.markdown("**Reported Return Reasons**")
        reasons = row['return_reason']
        
        if isinstance(reasons, list) and len(reasons) > 0:
            st.table(pd.DataFrame(reasons, columns=["Reason Code"]))
        elif isinstance(reasons, str) and reasons != "Unspecified":
            st.info(reasons)
        else:
            st.caption("No specific return codes found.")
            
        if row['ticket_count'] > 0:
            st.warning(f"⚠️ {int(row['ticket_count'])} support tickets linked. Cross-reference Helpdesk export.")

    with c_ai:
        st.subheader("🤖 AI Consultant")
        st.markdown("Generate root cause analysis and action plan.")
        
        if st.button("Run Analysis"):
            with st.spinner("Analyzing patterns..."):
                reasons_str = str(reasons)
                prompt = f"""
                Analyze this product:
                SKU: {sku}
                Name: {row['product_name']}
                Return Rate: {row['return_rate']}% (Industry avg 4%)
                Loss: ${row['lost_revenue']}
                Top Reasons: {reasons_str}
                
                1. Provide a hypothesis for the root cause.
                2. Suggest 3 actionable manufacturing or documentation improvements.
                3. Estimate potential savings if fixed.
                """
                res = st.session_state.ai.ask(prompt)
                st.info(res)

def render_capa():
    st.title("CAPA Generator")
    st.markdown("Create ISO 13485 compliant investigation reports.")
    
    c1, c2 = st.columns(2)
    sku_val = c1.text_input("Affected SKU")
    problem = c2.text_area("Problem Statement")
    
    st.markdown("**Investigation**")
    root_cause = st.text_area("Root Cause (5 Whys / Fishbone)")
    
    c3, c4 = st.columns(2)
    corr = c3.text_area("Corrective Action (Immediate)")
    prev = c4.text_area("Preventive Action (Long-term)")
    
    if st.button("Generate Report (.docx)"):
        doc = Document()
        doc.add_heading(f"CAPA Report: {sku_val}", 0)
        doc.add_paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        doc.add_heading("1. Issue Description", level=1)
        doc.add_paragraph(f"Product: {sku_val}")
        doc.add_paragraph(problem)
        
        doc.add_heading("2. Investigation", level=1)
        doc.add_paragraph(root_cause)
        
        doc.add_heading("3. Action Plan", level=1)
        doc.add_paragraph(f"Corrective: {corr}")
        doc.add_paragraph(f"Preventive: {prev}")
        
        bio = io.BytesIO()
        doc.save(bio)
        st.download_button("Download Report", bio.getvalue(), f"CAPA_{sku_val}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

def render_settings():
    st.title("Settings")
    st.info("Configure AI provider for automated insights.")
    
    p = st.selectbox("AI Provider", ["Google Gemini 1.5 Flash", "OpenAI GPT-4o"])
    k = st.text_input("API Key", type="password")
    
    if st.button("Connect"):
        st.session_state.ai.configure(p, k)
        if st.session_state.ai.available:
            st.success(f"Connected to {p} successfully.")
        else:
            st.error("Connection failed. Check API key.")

# --- MAIN ROUTER ---
def main():
    if 'page' not in st.session_state: st.session_state.page = "Dashboard"
    
    with st.sidebar:
        st.title("ORION")
        st.caption("Operational Intelligence")
        st.markdown("---")
        if st.button("📊 Dashboard", use_container_width=True): st.session_state.page = "Dashboard"
        if st.button("🛡️ CAPA Tools", use_container_width=True): st.session_state.page = "CAPA"
        if st.button("⚙️ Settings", use_container_width=True): st.session_state.page = "Settings"

    if st.session_state.page == "Dashboard": render_dashboard()
    elif st.session_state.page == "Product 360": render_product_360()
    elif st.session_state.page == "CAPA": render_capa()
    elif st.session_state.page == "Settings": render_settings()

if __name__ == "__main__":
    main()
