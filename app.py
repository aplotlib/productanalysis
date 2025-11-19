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

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="ORION | Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }
    
    /* Custom Metric Cards */
    .metric-card {
        background: white; border-radius: 8px; padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E2E8F0;
    }
    .metric-label { color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #0F172A; font-size: 2rem; font-weight: 800; margin-top: 8px; }
    .metric-delta { font-size: 0.9rem; margin-top: 4px; }
    .positive { color: #16A34A; }
    .negative { color: #DC2626; }
    
    /* Headers */
    h1, h2, h3 { color: #0F172A; font-weight: 700; letter-spacing: -0.02em; }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. AI ENGINE ---
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

    def ask(self, prompt):
        if not self.available: return "⚠️ AI unavailable. Check API Key in Settings."
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

# State Management
if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()
if 'data' not in st.session_state: st.session_state.data = pd.DataFrame()
if 'processor' not in st.session_state: st.session_state.processor = OdooProcessor()

# --- 4. COMPONENTS ---

def metric_card(label, value, delta=None, is_bad=False):
    """Custom HTML metric card for better visual impact."""
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

def render_upload():
    st.markdown("### 📥 Data Pipeline")
    with st.container():
        st.info("Upload Odoo exports to generate intelligence report.")
        c1, c2, c3 = st.columns(3)
        sales = c1.file_uploader("Sales Forecast / Orders", type=['xlsx'])
        returns = c2.file_uploader("Returns Pivot Report", type=['xlsx'])
        tickets = c3.file_uploader("Helpdesk Export", type=['xlsx'])
        
        if st.button("Run Analysis", type="primary", use_container_width=True):
            if not sales and not returns:
                st.error("Sales or Returns data required.")
                return
                
            with st.spinner("Processing..."):
                df = st.session_state.processor.merge_datasets(
                    st.session_state.processor.process_sales_file(sales),
                    st.session_state.processor.process_returns_file(returns),
                    st.session_state.processor.process_helpdesk_file(tickets)
                )
                st.session_state.data = df
                st.rerun()

def render_dashboard():
    df = st.session_state.data
    if df.empty:
        render_upload()
        return

    # --- TOP LEVEL STATS ---
    c1, c2 = st.columns([3, 1])
    c1.title("Operational Intelligence")
    if c2.button("Reset Analysis"):
        st.session_state.data = pd.DataFrame()
        st.rerun()

    tot_sales = df['sales_qty'].sum()
    tot_rev = df['est_revenue'].sum()
    lost_rev = df['lost_revenue'].sum()
    global_rate = (df['return_qty'].sum() / tot_sales * 100) if tot_sales else 0
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Est. Revenue", f"${tot_rev/1000:.1f}K")
    with k2: metric_card("Financial Loss (Returns)", f"${lost_rev/1000:.1f}K", "Direct Impact", is_bad=True)
    with k3: metric_card("Global Return Rate", f"{global_rate:.2f}%", "Target: < 4.0%", is_bad=(global_rate > 4))
    with k4: metric_card("Active SKUs", f"{len(df)}")

    st.divider()

    # --- CRITICAL ALERTS ---
    st.subheader("🔥 Critical Alerts")
    # Logic: Products losing > $500 or with > 8% return rate (min vol 20)
    alerts = df[
        (df['lost_revenue'] > 500) | 
        ((df['return_rate'] > 8) & (df['sales_qty'] > 20))
    ].head(5)

    if not alerts.empty:
        for _, row in alerts.iterrows():
            with st.container():
                c_det, c_stat, c_btn = st.columns([3, 2, 1])
                c_det.markdown(f"**{row['clean_sku']}**")
                c_det.caption(row['product_name'])
                
                reasons = ", ".join(row['return_reason']) if isinstance(row['return_reason'], list) else str(row['return_reason'])
                c_stat.markdown(f"Loss: **${row['lost_revenue']:.0f}** | Rate: **{row['return_rate']:.1f}%**")
                c_stat.caption(f"Top Issues: {reasons[:50]}...")
                
                if c_btn.button("Inspect", key=f"btn_{row['clean_sku']}", use_container_width=True):
                    st.session_state.deep_dive_sku = row['clean_sku']
                    st.session_state.page = "Product 360"
                    st.rerun()
            st.divider()
    else:
        st.success("No critical performance alerts detected.")

    # --- MACRO VIEW ---
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Financial Impact by Category**")
        cat_df = df.groupby('category')['lost_revenue'].sum().reset_index()
        fig = px.bar(cat_df, x='category', y='lost_revenue', color='lost_revenue', color_continuous_scale='reds')
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with c_right:
        st.markdown("**Volume vs. Quality Risk**")
        scatter_df = df[df['sales_qty'] > 10] # Filter noise
        fig2 = px.scatter(
            scatter_df, x='sales_qty', y='return_rate', 
            size='lost_revenue', color='return_rate',
            hover_name='clean_sku', color_continuous_scale='RdYlGn_r'
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)

def render_product_360():
    df = st.session_state.data
    if df.empty: return

    if st.button("← Back to Overview"):
        st.session_state.page = "Dashboard"
        st.rerun()

    # Selection Logic
    pre_select = st.session_state.get('deep_dive_sku', df['clean_sku'].iloc[0])
    if pre_select not in list(df['clean_sku']): pre_select = df['clean_sku'].iloc[0]
    
    sku = st.selectbox("Select Product", df['clean_sku'].unique(), index=list(df['clean_sku']).index(pre_select))
    row = df[df['clean_sku'] == sku].iloc[0]

    st.title(f"{sku}")
    st.markdown(f"**{row['product_name']}**")

    # Product Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Sales Units", int(row['sales_qty']))
    with c2: metric_card("Return Rate", f"{row['return_rate']:.2f}%", is_bad=row['return_rate']>5)
    with c3: metric_card("Financial Loss", f"${row['lost_revenue']:.0f}", is_bad=True)
    with c4: metric_card("Support Tickets", int(row['ticket_count']))

    st.markdown("### Diagnosis")
    
    # Layout: Reasons + AI Analysis
    c_data, c_ai = st.columns([1, 1])
    
    with c_data:
        st.markdown("**Reported Return Reasons**")
        reasons = row['return_reason']
        if reasons:
            # Simple frequency visualization if we had raw data, for now just list
            for i, r in enumerate(reasons, 1):
                st.markdown(f"{i}. {r}")
        else:
            st.info("No categorical reason data available.")
            
        if row['ticket_count'] > 0:
            st.warning(f"⚠️ {int(row['ticket_count'])} associated support tickets found.")

    with c_ai:
        st.markdown("**AI Root Cause Analysis**")
        if st.button("Generate Analysis"):
            with st.spinner("Analyzing patterns..."):
                prompt = f"""
                Analyze product: {row['product_name']} (SKU: {sku})
                Metrics: {row['return_rate']}% Return Rate (Avg 4%), ${row['lost_revenue']} Loss.
                Top Reasons: {reasons}
                
                1. Hypothesize the Root Cause based on the product type and reasons.
                2. Suggest 3 specific manufacturing or content improvements.
                """
                res = st.session_state.ai.ask(prompt)
                st.info(res)

def render_capa():
    st.title("CAPA Generator")
    st.markdown("Create ISO 13485 compliant investigation reports.")
    
    c1, c2 = st.columns(2)
    sku = c1.text_input("Affected SKU")
    problem = c2.text_area("Problem Statement")
    
    st.markdown("**Investigation**")
    root_cause = st.text_area("Root Cause (5 Whys)")
    
    c3, c4 = st.columns(2)
    corr = c3.text_area("Corrective Action")
    prev = c4.text_area("Preventive Action")
    
    if st.button("Generate Report (.docx)"):
        doc = Document()
        doc.add_heading(f"CAPA Report", 0)
        doc.add_paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        doc.add_heading("1. Issue Description", level=1)
        doc.add_paragraph(f"Product: {sku}")
        doc.add_paragraph(problem)
        doc.add_heading("2. Investigation", level=1)
        doc.add_paragraph(root_cause)
        doc.add_heading("3. Action Plan", level=1)
        doc.add_paragraph(f"Corrective: {corr}")
        doc.add_paragraph(f"Preventive: {prev}")
        
        bio = io.BytesIO()
        doc.save(bio)
        st.download_button("Download Report", bio.getvalue(), f"CAPA_{sku}.docx", "application/docx")

def render_settings():
    st.title("Settings")
    st.info("Configure AI for automated insights.")
    
    p = st.selectbox("AI Provider", ["Google Gemini 1.5 Flash", "OpenAI GPT-4o"])
    k = st.text_input("API Key", type="password")
    
    if st.button("Connect"):
        st.session_state.ai.configure(p, k)
        if st.session_state.ai.available:
            st.success("Connected successfully.")
        else:
            st.error("Connection failed.")

# --- MAIN ROUTER ---
def main():
    if 'page' not in st.session_state: st.session_state.page = "Dashboard"
    
    # Sidebar
    with st.sidebar:
        st.title("ORION")
        if st.button("📊 Overview", use_container_width=True): st.session_state.page = "Dashboard"
        if st.button("🛡️ CAPA Tools", use_container_width=True): st.session_state.page = "CAPA"
        if st.button("⚙️ Settings", use_container_width=True): st.session_state.page = "Settings"
        st.markdown("---")
        st.caption("v2.1.0 | Operational Intelligence")

    # Routing
    if st.session_state.page == "Dashboard": render_dashboard()
    elif st.session_state.page == "Product 360": render_product_360()
    elif st.session_state.page == "CAPA": render_capa()
    elif st.session_state.page == "Settings": render_settings()

if __name__ == "__main__":
    main()
