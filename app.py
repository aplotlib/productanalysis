import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import datetime
import json

# --- CUSTOM MODULES ---
from odoo_processor import OdooProcessor
from return_processor import ReturnReportProcessor
from enhanced_ai_analysis import IntelligenceEngine

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="ORION | Product Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL UI THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    :root { --primary: #0F172A; --accent: #2563EB; --bg: #F8FAFC; }
    .stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }
    
    /* Card Styling */
    .metric-card {
        background: white; border: 1px solid #E2E8F0; 
        border-radius: 10px; padding: 20px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-value { font-size: 24px; font-weight: 800; color: var(--primary); }
    .metric-label { font-size: 14px; color: #64748B; font-weight: 500; }
    
    /* Headers */
    h1, h2, h3 { color: var(--primary); font-weight: 700; letter-spacing: -0.5px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INITIALIZATION ---
if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()
if 'master_data' not in st.session_state: st.session_state.master_data = pd.DataFrame()

# Initialize Processors
odoo_proc = OdooProcessor()
return_proc = ReturnReportProcessor()

# --- 4. SIDEBAR ---
def render_sidebar():
    with st.sidebar:
        st.title("🧬 ORION v2.1")
        st.caption("Product Intelligence System")
        st.markdown("---")
        
        # AI Configuration
        st.subheader("🧠 Intelligence Core")
        model_choice = st.selectbox("AI Model", 
            ["Google Gemini 1.5 Flash", "Google Gemini 1.5 Pro", "OpenAI GPT-4o"], 
            index=0
        )
        
        manual_key = None
        if "Gemini" in model_choice and "GEMINI_API_KEY" not in st.secrets:
            manual_key = st.text_input("API Key", type="password", help="Enter Google AI Key")
        elif "GPT" in model_choice and "OPENAI_API_KEY" not in st.secrets:
            manual_key = st.text_input("API Key", type="password")
            
        st.session_state.ai.configure_client(model_choice, manual_key)
        
        if st.session_state.ai.available:
            st.success(f"● {model_choice} Online")
        else:
            st.error("● AI Offline")
            
        st.markdown("---")
        return st.radio("Navigation", ["Dashboard", "Data Ingestion", "Vision Analysis", "CAPA Manager"], label_visibility="collapsed")

# --- 5. MODULES ---

def render_ingestion():
    st.header("📂 Data Ingestion")
    st.markdown("Upload your raw Odoo exports and Return reports. The system will auto-clean and link them.")
    
    uploaded_files = st.file_uploader("Drop Files (Inventory, Helpdesk, Returns)", accept_multiple_files=True)
    
    if st.button("Process Files", type="primary"):
        if not uploaded_files:
            st.warning("Please upload files first.")
            return

        with st.spinner("Processing & Linking Data..."):
            inv_df = pd.DataFrame()
            help_df = pd.DataFrame()
            ret_df = pd.DataFrame()
            
            # 1. Load Inventory (Master Key)
            for f in uploaded_files:
                if "Inventory" in f.name:
                    inv_df = odoo_proc.load_inventory_master(f.getvalue())
                    st.toast(f"Inventory: {len(inv_df)} records", icon="✅")
            
            # 2. Load Others
            for f in uploaded_files:
                try:
                    if "Inventory" in f.name: continue
                    
                    if "Helpdesk" in f.name:
                        help_df = odoo_proc.process_helpdesk(f.getvalue())
                        st.toast(f"Tickets: {len(help_df)} loaded", icon="🎫")
                    
                    elif "Return" in f.name or "Pivot" in f.name:
                        ret_df = return_proc.process(f.getvalue())
                        st.toast(f"Returns: {len(ret_df)} loaded", icon="📉")
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

            # 3. Merge
            if not inv_df.empty:
                merged = odoo_proc.merge_data(inv_df, help_df, ret_df)
                st.session_state.master_data = merged
                st.success(f"Successfully merged {len(merged)} Product SKUs.")
            else:
                st.error("Inventory Forecast file is required as the Master Key.")

def render_dashboard():
    st.header("📊 Executive Dashboard")
    
    df = st.session_state.master_data
    if df.empty:
        st.info("No data loaded. Go to 'Data Ingestion' to begin.")
        return

    # 1. Top Metrics
    total_returns = df['Total Returns'].sum() if 'Total Returns' in df else 0
    total_tickets = df['Ticket Count'].sum() if 'Ticket Count' in df else 0
    risky_products = len(df[(df.get('Total Returns', 0) > 5) | (df.get('Ticket Count', 0) > 5)])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total SKUs", len(df))
    c2.metric("Total Returns", int(total_returns))
    c3.metric("Support Tickets", int(total_tickets))
    c4.metric("High Risk Items", risky_products, delta_color="inverse")
    
    st.markdown("---")

    # 2. High Risk Analysis
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader("Top Products by Issue Volume")
        if 'Total Returns' in df.columns and 'Ticket Count' in df.columns:
            df['Combined Risk'] = df['Total Returns'] + df['Ticket Count']
            top_risk = df.sort_values('Combined Risk', ascending=False).head(10)
            
            fig = px.bar(
                top_risk, 
                x='Product SKU', 
                y=['Total Returns', 'Ticket Count'],
                title="Returns vs Tickets (Top 10)",
                barmode='group',
                color_discrete_sequence=['#EF4444', '#3B82F6']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insight Button
            if st.button("🤖 Analyze Trends"):
                with st.spinner("AI Analyzing..."):
                    data_summary = top_risk[['Product SKU', 'Total Returns', 'Ticket Count', 'Product Title']].to_string()
                    insight = st.session_state.ai.generate(
                        f"Analyze this product risk data. Identify patterns in returns vs tickets. Suggest 3 focus areas.\n{data_summary}"
                    )
                    st.info(insight)

    with c_right:
        st.subheader("Watchlist")
        st.dataframe(
            df[['Product SKU', 'Total Returns', 'Ticket Count']].sort_values('Total Returns', ascending=False).head(15),
            hide_index=True,
            use_container_width=True
        )

def render_vision():
    st.header("👁️ Vision Diagnostics")
    st.markdown("Upload photos of defects or products for AI analysis.")
    
    img_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if img_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img_file, caption="Uploaded Artifact", use_column_width=True)
        
        with col2:
            prompt = st.text_area("Instruction", "Analyze this product image. Describe any visible defects, wear, or quality issues. Assess severity (Low/Medium/High).")
            if st.button("Analyze Artifact"):
                with st.spinner("Vision Engine Processing..."):
                    image = Image.open(img_file)
                    result = st.session_state.ai.analyze_image(image, prompt)
                    st.markdown("### Findings")
                    st.write(result)
                    
                    # Integration: Send to CAPA
                    if st.button("Create CAPA from Findings"):
                        st.session_state.capa_prefill = {'desc': result, 'sku': 'FROM_IMAGE'}
                        st.success("Findings sent to CAPA Manager.")

def render_capa():
    st.header("🛡️ CAPA Manager")
    st.caption("Corrective And Preventive Action Workflow")
    
    # Initialize Session Data
    if 'capa_data' not in st.session_state: 
        st.session_state.capa_data = {
            'id': f"CAPA-{datetime.now().strftime('%y%m%d')}-{datetime.now().second}",
            'status': 'Draft'
        }
    
    data = st.session_state.capa_data
    
    # Check for prefill from Vision
    if 'capa_prefill' in st.session_state:
        data['desc'] = st.session_state.capa_prefill.get('desc')
        data['sku'] = st.session_state.capa_prefill.get('sku')
        del st.session_state.capa_prefill

    t1, t2 = st.tabs(["📝 Investigation", "📋 Formal Report"])
    
    with t1:
        c1, c2 = st.columns(2)
        data['id'] = c1.text_input("Reference ID", data['id'])
        data['sku'] = c2.text_input("Product SKU", data.get('sku', ''))
        data['desc'] = st.text_area("Problem Description", data.get('desc', ''), height=150)
        
        if st.button("✨ Auto-Draft Investigation"):
            with st.spinner("AI Consultant is drafting report..."):
                draft = st.session_state.ai.generate_capa_draft({'product': data['sku'], 'issue': data['desc']})
                if draft:
                    data.update(draft)
                    st.success("Draft Generated Successfully!")
                    st.rerun()

    with t2:
        if data.get('root_cause_analysis'):
            st.markdown(f"### CAPA Report: {data['id']}")
            st.markdown("---")
            st.markdown(f"**Root Cause Analysis:**\n{data.get('root_cause_analysis')}")
            st.markdown(f"**Immediate Action:**\n{data.get('immediate_action')}")
            st.markdown(f"**Corrective Action:**\n{data.get('corrective_action')}")
            st.markdown(f"**Effectiveness Check:**\n{data.get('effectiveness_check')}")
            
            st.download_button(
                "Download JSON", 
                data=json.dumps(data, indent=2), 
                file_name=f"{data['id']}.json",
                mime="application/json"
            )
        else:
            st.info("Complete the Investigation tab to generate the formal report.")

# --- MAIN EXECUTION ---
def main():
    nav = render_sidebar()
    
    if nav == "Dashboard": render_dashboard()
    elif nav == "Data Ingestion": render_ingestion()
    elif nav == "Vision Analysis": render_vision()
    elif nav == "CAPA Manager": render_capa()

if __name__ == "__main__":
    main()
