import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import graphviz
from PIL import Image
import io
import re
import os
import json
import numpy as np
from datetime import datetime, timedelta
from docx import Document 
from docx.shared import Inches

# Import our deep logic processor
from odoo_processor import OdooProcessor

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="ORION | Medical Device Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. UI THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root { --primary: #0F172A; --accent: #2563EB; --bg-light: #F8FAFC; --card-bg: #FFFFFF; --border: #E2E8F0; }
    .stApp { background-color: var(--bg-light); font-family: 'Inter', sans-serif; color: #1E293B; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: var(--primary); letter-spacing: -0.5px; margin: 0; }
    .sub-header { font-size: 1.1rem; color: #64748B; font-weight: 500; margin-top: 0.25rem; }
    .feature-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    div[data-testid="metric-container"] { background-color: white; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: white; border-radius: 6px; border: 1px solid #E2E8F0; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: var(--accent); color: white; border-color: var(--accent); }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE ENGINE (AI Logic) ---
class IntelligenceEngine:
    def __init__(self):
        self.client = None
        self.provider = None
        self.model_name = None
        self.available = False
        self.connection_error = None

    def _get_key(self, names, manual_key=None):
        if manual_key: return manual_key
        for name in names:
            if hasattr(st, "secrets") and name in st.secrets: return st.secrets[name]
            if os.environ.get(name): return os.environ.get(name)
        return None

    def configure_client(self, provider_choice, manual_key_input=None):
        self.available = False
        self.connection_error = None
        
        try:
            if "Gemini" in provider_choice:
                import google.generativeai as genai
                api_key = self._get_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], manual_key_input)
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai
                    self.provider = "Google Gemini"
                    self.model_name = "gemini-1.5-pro" 
                    self.available = True
                else:
                    self.connection_error = "Missing Google API Key"

            elif "GPT" in provider_choice:
                import openai
                api_key = self._get_key(["OPENAI_API_KEY"], manual_key_input)
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.provider = "OpenAI"
                    self.model_name = "gpt-4o"
                    self.available = True
                else:
                    self.connection_error = "Missing OpenAI API Key"
                    
        except Exception as e:
            self.connection_error = f"Connection Failed: {str(e)}"
            self.available = False

    def generate(self, prompt, temperature=0.3, json_mode=False):
        if not self.available: return None if json_mode else f"⚠️ AI Offline: {self.connection_error}"
        try:
            if "Gemini" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                config = {'temperature': temperature}
                if json_mode: config['response_mime_type'] = 'application/json'
                response = model.generate_content(prompt, generation_config=config)
                return response.text
            elif "OpenAI" in self.provider:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                if json_mode: kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)}"

    def analyze_image(self, image, prompt):
        if not self.available: return f"AI Offline: {self.connection_error}"
        try:
            if "Gemini" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content([prompt, image])
                return response.text
            elif "OpenAI" in self.provider:
                return "Vision requires Gemini 1.5 Pro (Multimodal)."
        except Exception as e: return f"Vision Error: {e}"

    def generate_capa_draft(self, context):
        prompt = f"""
        Act as a Quality Engineer (ISO 13485). Create a CAPA investigation for:
        Product: {context.get('product')}
        Issue: {context.get('issue')}
        Return Stats: {context.get('stats', 'N/A')}
        
        Return a JSON object with these keys: 
        issue_description, root_cause_analysis, corrective_action, preventive_action, effectiveness_plan.
        """
        res = self.generate(prompt, json_mode=True)
        try:
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            return json.loads(res)
        except: return None

# Initialize Session State
if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()
if 'master_data' not in st.session_state: st.session_state.master_data = pd.DataFrame()
if 'helpdesk_data' not in st.session_state: st.session_state.helpdesk_data = pd.DataFrame()
if 'processor' not in st.session_state: st.session_state.processor = OdooProcessor()

# --- 4. EXPORT UTILS ---
def create_word_doc(capa_data):
    doc = Document()
    doc.add_heading(f"CAPA Report: {capa_data.get('id', 'New')}", 0)
    doc.add_heading('1. Initiation', level=1)
    doc.add_paragraph(f"Product SKU: {capa_data.get('sku', 'N/A')}")
    doc.add_paragraph(f"Issue Description: {capa_data.get('desc', 'N/A')}")
    doc.add_heading('2. Investigation', level=1)
    doc.add_paragraph(capa_data.get('root_cause_analysis', 'Pending.'))
    if 'fishbone' in capa_data and capa_data['fishbone']:
        doc.add_heading('Fishbone Analysis:', level=2)
        for cat, cause in capa_data['fishbone']:
            doc.add_paragraph(f"{cat}: {cause}", style='List Bullet')
    doc.add_heading('3. Action Plan', level=1)
    doc.add_paragraph(f"Corrective: {capa_data.get('corrective_action', 'Pending')}")
    doc.add_paragraph(f"Preventive: {capa_data.get('preventive_action', 'Pending')}")
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- 5. UI MODULES ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### System Controls")
        model_choice = st.selectbox("AI Provider", ["Google Gemini 1.5 Pro", "OpenAI GPT-4o"], index=0)
        manual_key = None
        
        # Key input logic
        if ("Gemini" in model_choice and "GOOGLE_API_KEY" not in st.secrets) or \
           ("GPT" in model_choice and "OPENAI_API_KEY" not in st.secrets):
            manual_key = st.text_input("API Key", type="password")
        
        st.session_state.ai.configure_client(model_choice, manual_key)
        
        if st.session_state.ai.available:
            st.markdown(f"<span style='color:green'>● {model_choice} Online</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'>● Offline</span>", unsafe_allow_html=True)

        st.markdown("---")
        return st.radio("Navigation", [
            "📤 Data Central", 
            "📊 Dashboard", 
            "🔍 SKU Deep Dive", 
            "🎫 Helpdesk Insights", 
            "🛠️ CAPA Manager", 
            "👁️ Vision", 
            "📝 Strategy"
        ])

def render_data_central():
    st.markdown("<h1 class='main-header'>Data Central</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Ingest and harmonize Odoo ecosystem data.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. Sales/Forecast")
        sales_file = st.file_uploader("Upload Odoo Sales (.xlsx)", type=['xlsx'], key='sales')
    with c2:
        st.markdown("### 2. Returns Report")
        returns_file = st.file_uploader("Upload Pivot Report (.xlsx)", type=['xlsx'], key='ret')
    with c3:
        st.markdown("### 3. Helpdesk")
        helpdesk_file = st.file_uploader("Upload Tickets (.xlsx)", type=['xlsx'], key='hd')

    if st.button("🚀 Process & Harmonize Data", type="primary", use_container_width=True):
        if not sales_file and not returns_file:
            st.error("Please upload at least Sales or Returns data.")
            return

        with st.status("Running Analysis Pipeline...", expanded=True) as status:
            processor = st.session_state.processor
            
            st.write("Parsing Sales Data...")
            sales_df = processor.process_sales_file(sales_file) if sales_file else pd.DataFrame()
            
            st.write("Parsing Return Data...")
            returns_df = processor.process_returns_file(returns_file) if returns_file else pd.DataFrame()
            
            st.write("Parsing Helpdesk Data...")
            helpdesk_df = processor.process_helpdesk_file(helpdesk_file) if helpdesk_file else pd.DataFrame()
            st.session_state.helpdesk_data = helpdesk_df
            
            st.write("Executing XLOOKUP Merges & Categorization...")
            master_df = processor.merge_datasets(sales_df, returns_df, helpdesk_df)
            st.session_state.master_data = master_df
            
            status.update(label="Pipeline Complete!", state="complete", expanded=False)
            
        st.success(f"Analysis Ready! Processed {len(master_df)} unique SKUs.")
        st.dataframe(master_df.head(), use_container_width=True)

def render_dashboard():
    st.markdown("<h1 class='main-header'>Executive Dashboard</h1>", unsafe_allow_html=True)
    df = st.session_state.master_data
    
    if df.empty:
        st.warning("No data loaded. Please visit 'Data Central' first.")
        return

    # Metrics
    total_sales = df['sales_qty'].sum()
    total_returns = df['return_qty'].sum()
    avg_rate = (total_returns / total_sales * 100) if total_sales > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales Units", f"{int(total_sales):,}")
    m2.metric("Total Returns", f"{int(total_returns):,}")
    m3.metric("Avg Return Rate", f"{avg_rate:.2f}%")
    m4.metric("Active SKUs", f"{len(df):,}")

    st.markdown("---")
    
    # Top Offenders
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Risk Matrix: Volume vs Return Rate")
        # Filter for meaningful data (at least 10 sales)
        chart_df = df[df['sales_qty'] > 10].copy()
        if not chart_df.empty:
            fig = px.scatter(
                chart_df, 
                x='sales_qty', y='return_rate', 
                size='return_qty', color='return_rate',
                hover_name='product_name', hover_data=['clean_sku'],
                color_continuous_scale='RdYlGn_r',
                labels={'sales_qty': 'Sales Volume', 'return_rate': 'Return Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Watchlist (>8% Rate)")
        watchlist = df[(df['return_rate'] > 8) & (df['sales_qty'] > 20)].sort_values('return_rate', ascending=False).head(10)
        st.dataframe(watchlist[['clean_sku', 'return_rate']], hide_index=True, use_container_width=True)

def render_sku_deep_dive():
    st.markdown("<h1 class='main-header'>SKU Deep Dive</h1>", unsafe_allow_html=True)
    df = st.session_state.master_data
    if df.empty: return

    # Selector
    sku_list = df['clean_sku'].unique()
    selected_sku = st.selectbox("Select SKU", sku_list)
    
    prod = df[df['clean_sku'] == selected_sku].iloc[0]
    
    # Product Card
    st.markdown(f"## {prod['product_name']}")
    st.caption(f"SKU: {prod['clean_sku']} | Category: {prod['Category']}")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Sales", int(prod['sales_qty']))
    k2.metric("Returns", int(prod['return_qty']))
    k3.metric("Return Rate", f"{prod['return_rate']:.2f}%")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Return Reasons")
        reasons = prod['return_reason'] # List from OdooProcessor
        if isinstance(reasons, list) and reasons:
            st.write(reasons)
        else:
            st.info("No specific return reasons found.")

    with c2:
        st.markdown("### Support Tickets")
        count = prod.get('ticket_count', 0)
        st.metric("Ticket Volume", int(count))
        
        if count > 0:
            # Show ticket extracts if available
            hd = st.session_state.helpdesk_data
            if not hd.empty:
                tickets = hd[hd['clean_sku'] == selected_sku]
                for t in tickets['ticket_text'].head(3):
                    st.text(f"• {t[:100]}...")
    
    # Direct Link to CAPA
    if st.button("⚡ Escalate to CAPA"):
        st.session_state.capa_prefill = {
            'sku': prod['clean_sku'],
            'desc': f"High return rate ({prod['return_rate']:.2f}%) observed. Top reasons: {prod['return_reason']}",
            'stats': f"Sales: {prod['sales_qty']}, Returns: {prod['return_qty']}"
        }
        st.success("Data sent to CAPA Manager. Navigate there to continue.")

def render_helpdesk_insights():
    st.markdown("<h1 class='main-header'>Helpdesk Insights</h1>", unsafe_allow_html=True)
    hd = st.session_state.helpdesk_data
    if hd.empty:
        st.warning("No Helpdesk data loaded.")
        return

    st.markdown("### Recurring Keywords")
    from collections import Counter
    all_text = " ".join(hd['ticket_text'].astype(str).tolist()).lower()
    words = re.findall(r'\w+', all_text)
    # Basic stopwords
    stop = {'the', 'to', 'and', 'of', 'a', 'in', 'is', 'it', 'for', 'my', 'on', 'with', 'ticket', 'order', 'please', 'hi', 'hello'}
    filtered = [w for w in words if w not in stop and len(w) > 3]
    common = Counter(filtered).most_common(15)
    
    chart_df = pd.DataFrame(common, columns=['Term', 'Count'])
    fig = px.bar(chart_df, x='Count', y='Term', orientation='h', title="Common Terms in Support Tickets")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Recent Ticket Feed")
    st.dataframe(hd[['clean_sku', 'ticket_text']].head(20), use_container_width=True)

def render_capa_manager():
    st.markdown("<h1 class='main-header'>CAPA Manager</h1>", unsafe_allow_html=True)
    
    # Initialize CAPA State
    if 'capa_data' not in st.session_state:
        st.session_state.capa_data = {'id': f"CAPA-{datetime.now().strftime('%y%m%d')}", 'risks': pd.DataFrame([{"Failure Mode": "Ex", "RPN": 0}])}
    
    data = st.session_state.capa_data
    
    # Prefill from Deep Dive
    if 'capa_prefill' in st.session_state:
        data.update(st.session_state.capa_prefill)
        del st.session_state.capa_prefill

    t1, t2, t3 = st.tabs(["📝 Intake & Investigation", "🛡️ Risk Analysis (FMEA)", "📤 Actions & Report"])

    with t1:
        c1, c2 = st.columns(2)
        data['id'] = c1.text_input("CAPA ID", data['id'])
        data['sku'] = c2.text_input("Product SKU", data.get('sku',''))
        data['desc'] = st.text_area("Problem Description", data.get('desc',''))
        
        if st.button("✨ AI Auto-Draft Investigation"):
            with st.spinner("Consulting AI Quality Engineer..."):
                draft = st.session_state.ai.generate_capa_draft({'product': data['sku'], 'issue': data['desc'], 'stats': data.get('stats', '')})
                if draft:
                    data.update(draft)
                    st.rerun()
        
        st.subheader("Root Cause Analysis")
        data['root_cause_analysis'] = st.text_area("5 Whys / Technical Analysis", data.get('root_cause_analysis', ''), height=150)

    with t2:
        st.subheader("FMEA Risk Table")
        edited = st.data_editor(data['risks'], num_rows="dynamic", use_container_width=True, key='fmea_edit')
        data['risks'] = edited

    with t3:
        st.subheader("Action Plan")
        data['corrective_action'] = st.text_area("Corrective Action", data.get('corrective_action', ''))
        data['preventive_action'] = st.text_area("Preventive Action", data.get('preventive_action', ''))
        
        st.markdown("---")
        doc = create_word_doc(data)
        st.download_button("📄 Download Word Report", data=doc, file_name=f"{data['id']}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

def render_vision():
    st.markdown("<h1 class='main-header'>Vision Diagnostics</h1>", unsafe_allow_html=True)
    st.info("Upload photos of defective returns for AI damage analysis.")
    
    img_file = st.file_uploader("Upload Product Photo", type=['jpg','png'])
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=400)
        
        if st.button("🔍 Analyze Defect"):
            res = st.session_state.ai.analyze_image(image, "Identify the defect, potential root cause, and severity.")
            st.markdown(f"### AI Analysis\n{res}")
            if st.button("Add to CAPA"):
                st.session_state.capa_prefill = {'desc': f"Vision Analysis Result: {res}"}
                st.success("Added to CAPA buffer.")

def render_strategy():
    st.markdown("<h1 class='main-header'>Strategy Architect</h1>", unsafe_allow_html=True)
    st.markdown("Generate SOPs and Work Instructions based on your data insights.")
    
    with st.form("strat_form"):
        topic = st.text_input("Topic (e.g., Battery Handling Procedure)")
        context = st.text_area("Context / Requirements")
        if st.form_submit_button("Generate SOP"):
            res = st.session_state.ai.generate(f"Write a Medical Device SOP for: {topic}. Context: {context}")
            st.markdown(res)

# --- 6. MAIN ROUTING ---
def main():
    nav = render_sidebar()
    
    if nav == "📤 Data Central": render_data_central()
    elif nav == "📊 Dashboard": render_dashboard()
    elif nav == "🔍 SKU Deep Dive": render_sku_deep_dive()
    elif nav == "🎫 Helpdesk Insights": render_helpdesk_insights()
    elif nav == "🛠️ CAPA Manager": render_capa_manager()
    elif nav == "👁️ Vision": render_vision()
    elif nav == "📝 Strategy": render_strategy()

if __name__ == "__main__":
    main()
