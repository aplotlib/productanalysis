import streamlit as st
import pandas as pd
import plotly.express as px
import graphviz
from PIL import Image
import io
import re
import os
import json
import numpy as np
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="ORION | VIVE Health",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL UI THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #0F172A; 
        --accent: #2563EB;
        --bg-light: #F8FAFC;
        --card-bg: #FFFFFF;
        --border: #E2E8F0;
    }

    .stApp { background-color: var(--bg-light); font-family: 'Inter', sans-serif; color: #1E293B; }

    /* HEADER */
    .brand-header { padding-bottom: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
    .brand-title { font-size: 2.2rem; font-weight: 800; color: var(--primary); letter-spacing: -0.5px; margin: 0; }
    .brand-subtitle { font-size: 0.9rem; color: #64748B; font-weight: 500; margin-top: 0.25rem; }

    /* CARDS */
    .feature-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: white; border-radius: 6px; border: 1px solid #E2E8F0; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: var(--accent); color: white; border-color: var(--accent); }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE ENGINE (Gemini/Vertex + OpenAI) ---
class IntelligenceEngine:
    def __init__(self):
        self.client = None
        self.provider = None
        self.model_name = None
        self.available = False
        self.manual_key = None # For runtime key entry

    def _get_key(self, names):
        # Priority: Manual Entry -> Secrets -> Environment
        if self.manual_key: return self.manual_key
        for name in names:
            if hasattr(st, "secrets") and name in st.secrets: return st.secrets[name]
            if os.environ.get(name): return os.environ.get(name)
        return None

    def configure_client(self, provider_choice, manual_key_input=None):
        self.available = False
        self.manual_key = manual_key_input
        
        try:
            if "Gemini" in provider_choice:
                api_key = self._get_key(["GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api_key"])
                if api_key:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    self.client = genai
                    self.provider = "Google Gemini"
                    # Map selection to model ID
                    if "Flash" in provider_choice: self.model_name = "gemini-1.5-flash"
                    else: self.model_name = "gemini-1.5-pro"
                    self.available = True
                else:
                    self.available = False 

            elif "GPT" in provider_choice:
                api_key = self._get_key(["OPENAI_API_KEY", "openai_api_key"])
                if api_key:
                    import openai
                    self.client = openai.OpenAI(api_key=api_key)
                    self.provider = "OpenAI"
                    if "4o" in provider_choice: self.model_name = "gpt-4o"
                    else: self.model_name = "gpt-4o-mini"
                    self.available = True
                else:
                    self.available = False
        except Exception as e:
            st.error(f"Connection Error: {e}")
            self.available = False

    def generate(self, prompt, temperature=0.3, json_mode=False):
        if not self.available: return None if json_mode else "⚠️ AI Offline. Please check API Key."
        
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
            return f"Generation Error: {str(e)}"

    def analyze_image(self, image, prompt):
        if not self.available: return "AI Offline"
        try:
            if "Gemini" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content([prompt, image])
                return response.text
            elif "OpenAI" in self.provider:
                # Fallback for OpenAI Vision if needed
                return "Please use Gemini 1.5 Pro/Flash for best vision analysis."
        except Exception as e: return f"Vision Error: {e}"

if 'ai' not in st.session_state:
    st.session_state.ai = IntelligenceEngine()

# --- 4. DATA HANDLER (Robust Odoo & Return Logic) ---
class DataHandler:
    @staticmethod
    def clean_sku(sku):
        """Smart Parent SKU extraction (MOB1027BLK -> MOB1027)"""
        if pd.isna(sku): return "Unknown"
        sku = str(sku).upper().strip()
        match = re.match(r"^([A-Z]+[0-9]+)", sku)
        return match.group(1) if match else sku

    @staticmethod
    def load_and_merge(files, period_days=90):
        master_data = {} # {Parent_SKU: {sales: 0, returns: 0, issues: []}}
        warnings = []
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        for file in files:
            try:
                # Intelligent Header Hunt
                raw = pd.read_excel(file, header=None)
                header_idx = 0
                for i in range(min(20, len(raw))):
                    row_str = raw.iloc[i].astype(str).str.lower().tolist()
                    # Look for row with at least 2 recognized keywords
                    if sum(1 for k in ['sku', 'product', 'asin', 'order', 'date', 'qty', 'sales'] if any(k in str(rs) for rs in row_str)) >= 2:
                        header_idx = i
                        break
                
                df = pd.read_excel(file, header=header_idx)
                df.columns = [str(c).strip() for c in df.columns]
                
                # Detect File Type
                fname = file.name.lower()
                is_odoo = "odoo" in fname or "forecast" in fname
                is_returns = "return" in fname or "pivot" in fname
                
                sku_col = next((c for c in df.columns if any(x in c.lower() for x in ['sku', 'product', 'default code'])), None)
                if not sku_col: continue

                # --- ODOO SALES LOGIC ---
                if is_odoo:
                    # Find Sales Column: prioritized check
                    sales_col = None
                    candidates = [c for c in df.columns if any(x in c.lower() for x in ['sales', 'qty', 'quantity', 'unnamed'])]
                    
                    # Filter for numeric candidates
                    numeric_candidates = []
                    for c in candidates:
                        if pd.to_numeric(df[c], errors='coerce').sum() > 0:
                            numeric_candidates.append(c)
                    
                    # Pick best candidate (prefer one with 'sales' in name, else take last numeric)
                    if numeric_candidates:
                        sales_col = next((c for c in numeric_candidates if 'sales' in c.lower()), numeric_candidates[-1])
                    
                    if sales_col:
                        for _, row in df.iterrows():
                            parent = DataHandler.clean_sku(row[sku_col])
                            qty = pd.to_numeric(row[sales_col], errors='coerce') or 0
                            if parent not in master_data: master_data[parent] = {'sales':0, 'returns':0, 'issues':[]}
                            master_data[parent]['sales'] += qty

                # --- RETURN REPORT LOGIC ---
                elif is_returns:
                    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                    reason_col = next((c for c in df.columns if any(x in c.lower() for x in ['reason', 'comment', 'review'])), None)
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        # Strict 1:1 Date Filter
                        df = df[df[date_col] >= cutoff_date]
                    else:
                        warnings.append(f"⚠️ {file.name}: No date column found. Filtering disabled.")
                        
                    for _, row in df.iterrows():
                        parent = DataHandler.clean_sku(row[sku_col])
                        if parent not in master_data: master_data[parent] = {'sales':0, 'returns':0, 'issues':[]}
                        master_data[parent]['returns'] += 1
                        if reason_col and pd.notna(row[reason_col]):
                            master_data[parent]['issues'].append(str(row[reason_col]))

            except Exception as e:
                warnings.append(f"Error {file.name}: {str(e)}")

        # Flatten
        rows = []
        for sku, data in master_data.items():
            rate = (data['returns'] / data['sales'] * 100) if data['sales'] > 0 else 0
            rows.append({
                'Parent_SKU': sku,
                'Sales': data['sales'],
                'Returns': data['returns'],
                'Return_Rate': rate,
                'Issues': len(data['issues'])
            })
        
        # FIX: Ensure columns exist even if empty
        if not rows:
            return pd.DataFrame(columns=['Parent_SKU', 'Sales', 'Returns', 'Return_Rate', 'Issues']), warnings
            
        return pd.DataFrame(rows), warnings

# --- 5. MODULES ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### System Controls")
        
        # 1. AI Configuration
        st.markdown("**AI Configuration**")
        model_choice = st.selectbox(
            "Select Provider", 
            ["Google Gemini 1.5 Flash", "Google Gemini 1.5 Pro", "OpenAI GPT-4o", "OpenAI GPT-4o Mini"],
            index=0
        )
        
        # Check for Key (Logic: Check Secrets -> If Missing, Show Input)
        has_secret = False
        if "Gemini" in model_choice:
            has_secret = "GOOGLE_API_KEY" in st.secrets or "GEMINI_API_KEY" in st.secrets
        elif "GPT" in model_choice:
            has_secret = "OPENAI_API_KEY" in st.secrets
            
        manual_key = None
        if not has_secret:
            st.warning(f"No API key found for {model_choice.split()[0]}.")
            manual_key = st.text_input("Enter API Key", type="password")
        
        # Initialize AI
        st.session_state.ai.configure_client(model_choice, manual_key)
        
        if st.session_state.ai.available:
            st.markdown(f"<span style='color:green'>● {model_choice} Connected</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. Period Settings
        st.markdown("**Reporting Period**")
        st.caption("Select the window for Sales & Returns (1:1)")
        days = st.selectbox("Input report period", [30, 60, 90, 180, 365], index=2)
        st.session_state.period_days = days
        
        st.markdown("---")
        nav = st.radio("Navigation", ["Dashboard", "Data Ingestion", "Vision", "CAPA Manager", "Strategy"], label_visibility="collapsed")
        return nav

def render_ingestion():
    st.markdown("### Data Ingestion")
    st.markdown(f"""
    <div class='feature-card'>
        <b>Processing Rules:</b><br>
        1. <b>Odoo Forecasts:</b> System scans for 'Sales' or 'Qty' columns.<br>
        2. <b>Return Reports:</b> Filtered to the last <b>{st.session_state.period_days} days</b> for accuracy.<br>
        3. <b>Grouping:</b> Data is aggregated by Parent SKU (e.g. MOB1027).
    </div>
    """, unsafe_allow_html=True)
    
    files = st.file_uploader("Upload Odoo & Return Files", accept_multiple_files=True)
    if files and st.button("Run Analysis"):
        df, warns = DataHandler.load_and_merge(files, st.session_state.period_days)
        st.session_state.master_data = df
        for w in warns: st.warning(w)
        st.success(f"Processed {len(df)} SKU Families.")
        
    if 'master_data' in st.session_state:
        # FIX: Check if data exists before sorting
        if not st.session_state.master_data.empty:
            st.dataframe(st.session_state.master_data.sort_values('Return_Rate', ascending=False), use_container_width=True)
        else:
            st.info("Files processed but no matching records found. Check column names or date ranges.")

def render_capa():
    st.markdown("### CAPA Manager")
    
    if 'capa_data' not in st.session_state:
        st.session_state.capa_data = {
            'id': f"CAPA-{datetime.now().strftime('%y%m%d')}",
            'risks': pd.DataFrame(columns=['Failure Mode', 'Effect', 'Sev', 'Occ', 'Det', 'RPN'])
        }
        
    data = st.session_state.capa_data
    
    # Pre-fill hook
    if 'capa_prefill' in st.session_state:
        data['desc'] = st.session_state.capa_prefill
        del st.session_state.capa_prefill

    tab1, tab2, tab3, tab4 = st.tabs(["1. Intake", "2. Investigation (Fishbone)", "3. Risk (FMEA)", "4. Action"])

    with tab1:
        c1, c2 = st.columns(2)
        data['id'] = c1.text_input("CAPA ID", data['id'])
        data['sku'] = c2.text_input("Product SKU")
        data['desc'] = st.text_area("Issue Description", value=data.get('desc', ''), height=100)
        
        if st.button("✨ Auto-Draft with AI"):
            with st.spinner("Drafting investigation plan..."):
                prompt = f"Draft a CAPA investigation for {data.get('sku', 'Unknown Product')}. Issue: {data.get('desc', 'Not provided')}. JSON format."
                res = st.session_state.ai.generate(prompt, json_mode=True)
                if res: st.info("Draft generated (simulated populate)")

    with tab2:
        st.markdown("#### Root Cause Analysis (Fishbone)")
        
        c_fish1, c_fish2 = st.columns([1, 3])
        with c_fish1:
            cause_cats = ["Man", "Machine", "Material", "Method", "Measurement", "Environment"]
            selected_cat = st.selectbox("Add Cause Category", cause_cats)
            new_cause = st.text_input("Cause Detail")
            if st.button("Add Branch"):
                if 'fishbone' not in data: data['fishbone'] = []
                data['fishbone'].append((selected_cat, new_cause))
        
        with c_fish2:
            # Visual Fishbone using Graphviz
            if data.get('fishbone'):
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR')
                graph.node('Problem', (data.get('desc') or "Problem")[:20]+"...")
                
                for cat, cause in data['fishbone']:
                    graph.node(cat, cat, shape='box')
                    graph.edge(cat, 'Problem')
                    graph.node(cause, cause, shape='plain')
                    graph.edge(cause, cat)
                    
                st.graphviz_chart(graph)
            else:
                st.info("Add causes to generate Fishbone Diagram.")

    with tab3:
        st.markdown("#### Failure Mode & Effects Analysis (FMEA)")
        
        # Initialize FMEA dataframe if empty
        if data['risks'].empty:
            data['risks'] = pd.DataFrame([
                {"Failure Mode": "Example Mode", "Effect": "Customer Returns", "Sev": 3, "Occ": 2, "Det": 4, "RPN": 24}
            ])

        # Interactive Editor
        edited_df = st.data_editor(
            data['risks'],
            num_rows="dynamic",
            column_config={
                "Sev": st.column_config.NumberColumn("Sev (1-10)", min_value=1, max_value=10),
                "Occ": st.column_config.NumberColumn("Occ (1-10)", min_value=1, max_value=10),
                "Det": st.column_config.NumberColumn("Det (1-10)", min_value=1, max_value=10),
                "RPN": st.column_config.NumberColumn("RPN", disabled=True),
            },
            use_container_width=True
        )
        
        # Auto-Calculate RPN
        edited_df['RPN'] = edited_df['Sev'] * edited_df['Occ'] * edited_df['Det']
        data['risks'] = edited_df
        
        # High Risk Highlight
        high_risk = edited_df[edited_df['RPN'] > 40]
        if not high_risk.empty:
            st.error(f"⚠️ {len(high_risk)} High Risk Items detected (RPN > 40)")

    with tab4:
        st.text_area("Corrective Action Plan")
        st.text_area("Verification of Effectiveness")
        st.button("Save Record", type="primary")

def render_vision():
    st.markdown("### Vision Diagnostics")
    img_file = st.file_uploader("Upload Defect Image", type=['png', 'jpg'])
    
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Artifact", width=400)
        
        if st.button("Run AI Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                res = st.session_state.ai.analyze_image(img, "Analyze this defect. Identify product, defect type, and likely root cause.")
                st.session_state.vision_res = res
    
    if 'vision_res' in st.session_state:
        st.markdown(f"<div class='feature-card'>{st.session_state.vision_res}</div>", unsafe_allow_html=True)
        if st.button("Escalate to CAPA"):
            st.session_state.capa_prefill = st.session_state.vision_res
            st.success("Sent to CAPA Intake")

def render_dashboard():
    st.markdown("### Dashboard")
    if 'master_data' in st.session_state and not st.session_state.master_data.empty:
        df = st.session_state.master_data
        c1, c2, c3 = st.columns(3)
        c1.metric("Active SKUs", len(df))
        c2.metric("Avg Return Rate", f"{df['Return_Rate'].mean():.1f}%")
        c3.metric("Total Returns", int(df['Returns'].sum()))
        
        fig = px.bar(df.head(10), x='Parent_SKU', y='Return_Rate', color='Return_Rate', title="Top Return Rates")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please load data in 'Data Ingestion' tab.")

def render_strategy():
    st.markdown("### Strategy & Compliance")
    st.info("ISO 13485 Document Generator")
    # Placeholder for document gen logic
    doc_type = st.selectbox("Document Type", ["Quality Manual", "SOP", "Work Instruction"])
    if st.button("Generate Template"):
        with st.spinner("Generating..."):
            res = st.session_state.ai.generate(f"Write a {doc_type} template for ISO 13485")
            st.markdown(res)

# --- MAIN ---
def main():
    nav = render_sidebar()
    if nav == "Dashboard": render_dashboard()
    elif nav == "Data Ingestion": render_ingestion()
    elif nav == "CAPA Manager": render_capa()
    elif nav == "Vision": render_vision()
    elif nav == "Strategy": render_strategy()

if __name__ == "__main__":
    main()
