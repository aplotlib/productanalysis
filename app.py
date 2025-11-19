import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

    /* CARDS & CONTAINERS */
    .feature-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    div[data-testid="metric-container"] { background-color: white; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: white; border-radius: 6px; border: 1px solid #E2E8F0; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: var(--accent); color: white; border-color: var(--accent); }

    /* ALERTS */
    .disclaimer-box { font-size: 0.8rem; color: #B45309; background-color: #FFFBEB; padding: 8px; border-radius: 4px; border: 1px solid #FCD34D; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE ENGINE (Dual Provider: Gemini/Vertex + OpenAI) ---
class IntelligenceEngine:
    def __init__(self):
        self.client = None
        self.provider = None
        self.model_name = None
        self.available = False
        self.clients = {} 
        self._initialize_all_clients()

    def _get_key(self, names):
        for name in names:
            if hasattr(st, "secrets") and name in st.secrets: return st.secrets[name]
            if os.environ.get(name): return os.environ.get(name)
        return None

    def _initialize_all_clients(self):
        # OpenAI
        openai_key = self._get_key(["OPENAI_API_KEY", "openai_api_key"])
        if openai_key:
            try:
                import openai
                self.clients['openai'] = openai.OpenAI(api_key=openai_key)
            except: pass
        # Google Vertex / Gemini
        google_key = self._get_key(["GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api_key"])
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                self.clients['google'] = genai
            except: pass

    def set_active_model(self, choice):
        self.available = False
        if "Gemini" in choice or "Vertex" in choice:
            if 'google' in self.clients:
                self.provider = "Google Vertex/Gemini"
                self.client = self.clients['google']
                if "Pro" in choice: self.model_name = "gemini-1.5-pro"
                elif "Flash" in choice: self.model_name = "gemini-1.5-flash"
                else: self.model_name = "gemini-1.5-flash"
                self.available = True
        elif "GPT" in choice:
            if 'openai' in self.clients:
                self.provider = "OpenAI"
                self.client = self.clients['openai']
                if "4o" in choice: self.model_name = "gpt-4o"
                else: self.model_name = "gpt-4o-mini"
                self.available = True

    def generate(self, prompt, temperature=0.3, json_mode=False):
        if not self.available: return None if json_mode else "⚠️ AI Offline. Check API Keys."
        
        try:
            if "Google" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                # Gemini JSON mode handling via prompt engineering mostly, or generation_config
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
            return f"Error: {str(e)}"

    def analyze_image(self, image, prompt):
        if not self.available: return "AI Offline"
        try:
            if "Google" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content([prompt, image])
                return response.text
            elif "OpenAI" in self.provider:
                return "Vision analysis currently optimized for Gemini models in this version."
        except Exception as e: return f"Vision Error: {e}"

    def generate_capa_draft(self, context_data):
        """Specific method for CAPA auto-filling"""
        prompt = f"""
        You are a Quality Assurance Expert (ISO 13485). Based on the following issue context, generate a JSON object for a CAPA form.
        
        Context:
        Product: {context_data.get('product', 'N/A')}
        Issue Description: {context_data.get('issue', 'N/A')}
        Root Cause Hint: {context_data.get('cause_hint', 'Investigation needed')}
        
        Return JSON with exactly these keys:
        - issue_description (Professional technical description)
        - containment_action (Immediate fix)
        - root_cause_analysis (Probable root causes using 5-Whys style)
        - corrective_action (Long term fix)
        - preventive_action (Systemic fix)
        - effectiveness_plan (How to verify)
        """
        res = self.generate(prompt, temperature=0.4, json_mode=True)
        try:
            # Clean up Markdown code blocks if present
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            return json.loads(res)
        except:
            return None

if 'ai' not in st.session_state or not hasattr(st.session_state.ai, 'clients'):
    st.session_state.ai = IntelligenceEngine()

# --- 4. ADVANCED DATA HANDLER ---
class DataHandler:
    
    @staticmethod
    def clean_sku(sku):
        """Strip suffixes to find Parent SKU (e.g., MOB1027BLK -> MOB1027)"""
        if pd.isna(sku): return "Unknown"
        sku = str(sku).upper().strip()
        match = re.match(r"^([A-Z]+[0-9]+)", sku)
        return match.group(1) if match else sku

    @staticmethod
    def load_and_merge(files, period_days=90):
        """
        Powerful merge logic:
        1. Parses Odoo Forecasts (Sales) & Return Reports.
        2. Filters Returns to exactly the last X days (1:1 comparison).
        3. Groups by Parent SKU.
        """
        master_data = {} # {Parent_SKU: {sales: 0, returns: 0, issues: []}}
        warnings = []
        
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        for file in files:
            try:
                # Intelligent Header Search
                raw = pd.read_excel(file, header=None)
                header_idx = 0
                for i in range(min(20, len(raw))):
                    row_str = raw.iloc[i].astype(str).str.lower().to_list()
                    if sum(1 for k in ['sku', 'product', 'asin', 'order', 'date'] if any(k in rs for rs in row_str)) >= 2:
                        header_idx = i
                        break
                
                df = pd.read_excel(file, header=header_idx)
                df.columns = [str(c).strip() for c in df.columns]
                
                # Detect Type
                fname = file.name.lower()
                is_odoo = "odoo" in fname or "forecast" in fname
                is_returns = "return" in fname or "pivot" in fname
                
                # Column Mapping
                sku_col = next((c for c in df.columns if 'sku' in c.lower() or 'product' in c.lower() or 'default code' in c.lower()), None)
                
                if not sku_col: continue

                # --- ODOO LOGIC ---
                if is_odoo:
                    # Find the dynamic sales column (often Unnamed or specific to period)
                    sales_col = None
                    for c in df.columns:
                        # Look for numeric columns that might be sales
                        if "unnamed" in c.lower() or "qty" in c.lower() or "sales" in c.lower():
                             if pd.to_numeric(df[c], errors='coerce').sum() > 0:
                                 sales_col = c
                                 # If we find a likely candidate, stick with it (simple heuristic)
                                 if "sales" in c.lower(): break 
                    
                    if sales_col:
                        for _, row in df.iterrows():
                            parent = DataHandler.clean_sku(row[sku_col])
                            qty = pd.to_numeric(row[sales_col], errors='coerce') or 0
                            if parent not in master_data: master_data[parent] = {'sales':0, 'returns':0, 'issues':[]}
                            master_data[parent]['sales'] += qty

                # --- RETURN REPORT LOGIC ---
                elif is_returns:
                    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                    reason_col = next((c for c in df.columns if 'reason' in c.lower() or 'comment' in c.lower()), None)
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        # 1:1 Filtering
                        df = df[df[date_col] >= cutoff_date]
                    else:
                        warnings.append(f"⚠️ No date column in {file.name}. Comparison may be inaccurate.")
                        
                    for _, row in df.iterrows():
                        parent = DataHandler.clean_sku(row[sku_col])
                        if parent not in master_data: master_data[parent] = {'sales':0, 'returns':0, 'issues':[]}
                        master_data[parent]['returns'] += 1
                        if reason_col and pd.notna(row[reason_col]):
                            master_data[parent]['issues'].append(str(row[reason_col]))

            except Exception as e:
                warnings.append(f"Error processing {file.name}: {str(e)}")

        # Convert to DataFrame
        rows = []
        for sku, data in master_data.items():
            rows.append({
                'Parent_SKU': sku,
                'Sales': data['sales'],
                'Returns': data['returns'],
                'Return_Rate': (data['returns'] / data['sales'] * 100) if data['sales'] > 0 else 0,
                'Issues_List': data['issues']
            })
            
        return pd.DataFrame(rows), warnings

# --- 5. UI SECTIONS ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### System Controls")
        
        # AI Selector
        st.markdown("**AI Intelligence Engine**")
        model_options = []
        if 'google' in st.session_state.ai.clients:
            model_options.extend(["Google Vertex (Gemini 1.5 Pro)", "Google Vertex (Gemini 1.5 Flash)"])
        if 'openai' in st.session_state.ai.clients:
            model_options.extend(["OpenAI GPT-4o", "OpenAI GPT-4o Mini"])
            
        if not model_options:
            st.error("No API Keys Found")
            st.caption("Check .streamlit/secrets.toml")
        else:
            # Default to Gemini Pro
            idx = 0
            for i, m in enumerate(model_options):
                if "Gemini 1.5 Pro" in m: idx = i
            
            choice = st.selectbox("Active Model", model_options, index=idx, label_visibility="collapsed")
            st.session_state.ai.set_active_model(choice)
            
            # Status
            color = "#DBEAFE" if "Google" in choice else "#DCFCE7"
            text = "#1E40AF" if "Google" in choice else "#166534"
            st.markdown(f"""
            <div style='background-color:{color}; color:{text}; padding:8px; border-radius:6px; font-size:0.8rem; font-weight:600;'>
                ● {choice} Online
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Reporting Period
        st.markdown("**Analysis Period**")
        st.caption("Aligns Sales & Returns 1:1")
        days = st.selectbox("Time Window", [30, 60, 90, 180, 365], index=2)
        st.session_state.period_days = days
        
        st.markdown("---")
        nav = st.radio("Navigation", ["Dashboard", "Data Ingestion", "Vision", "CAPA Manager", "Strategy"], label_visibility="collapsed")
        return nav

def render_header(title):
    st.markdown(f"""
    <div class="brand-header">
        <div class="brand-title">ORION</div>
        <div class="brand-subtitle">Operational Review & Intelligence Optimization Network</div>
    </div>
    <h3>{title}</h3>
    """, unsafe_allow_html=True)

def render_ingestion():
    render_header("Data Ingestion")
    st.markdown("""
    <div class="feature-card">
        <b>Upload Reports</b><br>
        Upload Odoo Sales Forecasts and Pivot Return Reports. ORION will:
        <ul>
            <li>Detect Parent SKUs automatically.</li>
            <li>Filter returns to the selected <b>{days}-day</b> window.</li>
            <li>Calculate true return rates.</li>
        </ul>
    </div>
    """.format(days=st.session_state.get('period_days', 90)), unsafe_allow_html=True)
    
    files = st.file_uploader("Drop Files Here", accept_multiple_files=True)
    if files:
        if st.button("Process Files", type="primary"):
            with st.spinner("Analyzing data structures..."):
                df, warns = DataHandler.load_and_merge(files, st.session_state.period_days)
                st.session_state.master_data = df
                
                if warns:
                    for w in warns: st.warning(w)
                
                st.success(f"Successfully processed {len(df)} Parent SKU families.")

    if 'master_data' in st.session_state:
        st.markdown("#### Analysis Preview")
        st.dataframe(st.session_state.master_data.sort_values('Return_Rate', ascending=False).head(10), use_container_width=True)

def render_dashboard():
    render_header("Executive Dashboard")
    if 'master_data' not in st.session_state:
        st.info("No data loaded. Please go to Data Ingestion.")
        return
        
    df = st.session_state.master_data
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active SKUs", len(df))
    c2.metric("Total Returns", int(df['Returns'].sum()))
    avg_rate = df['Return_Rate'].mean()
    c3.metric("Avg Return Rate", f"{avg_rate:.2f}%")
    c4.metric("High Risk Items", len(df[df['Return_Rate'] > 5.0]))
    
    st.markdown("---")
    
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("#### Top Offenders (Return Rate > 3%)")
        high_risk = df[df['Return_Rate'] > 3.0].sort_values('Return_Rate', ascending=False).head(10)
        fig = px.bar(high_risk, x='Parent_SKU', y='Return_Rate', color='Return_Rate', title="Return Rate by Family")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("#### Watchlist")
        st.dataframe(high_risk[['Parent_SKU', 'Return_Rate', 'Returns']], use_container_width=True)

def render_vision():
    render_header("Vision Diagnostics")
    c1, c2 = st.columns([1,1])
    
    with c1:
        st.markdown("#### Artifact Analysis")
        img = st.file_uploader("Upload Image", type=['jpg','png'])
        if img:
            st.image(img, caption="Uploaded Artifact", use_column_width=True)
            if st.button("Run Vision Analysis", type="primary"):
                with st.spinner("Scanning..."):
                    res = st.session_state.ai.analyze_image(Image.open(img), "Analyze this defect. Describe the failure mode, potential root cause, and severity.")
                    st.session_state.vision_result = res
    
    with c2:
        if 'vision_result' in st.session_state:
            st.markdown("#### AI Findings")
            st.markdown(f"<div class='feature-card'>{st.session_state.vision_result}</div>", unsafe_allow_html=True)
            
            st.markdown("### Actions")
            if st.button("🚨 Escalate to CAPA"):
                # Pre-fill CAPA state
                st.session_state.capa_prefill = {
                    'issue': st.session_state.vision_result,
                    'source': 'Vision Diagnostic'
                }
                st.success("Findings sent to CAPA Manager. Navigate there to complete.")

def render_capa():
    render_header("CAPA Manager")
    
    # Init CAPA State
    if 'capa_data' not in st.session_state:
        st.session_state.capa_data = {}
    
    # Check for prefills from other modules
    if 'capa_prefill' in st.session_state:
        st.session_state.capa_data['issue_description'] = st.session_state.capa_prefill.get('issue', '')
        st.session_state.capa_data['source'] = st.session_state.capa_prefill.get('source', 'Manual')
        del st.session_state.capa_prefill # Clear after use
        
    data = st.session_state.capa_data

    # --- AI AUTO-FILL BUTTON ---
    c_ai, _ = st.columns([1, 2])
    with c_ai:
        if st.button("✨ Auto-Draft with AI", help="Generates a draft for all fields based on the Issue Description"):
            if not data.get('issue_description'):
                st.error("Please enter an Issue Description first.")
            else:
                with st.spinner(f"Drafting via {st.session_state.ai.provider}..."):
                    draft = st.session_state.ai.generate_capa_draft({
                        'product': data.get('product_sku', 'Unknown'),
                        'issue': data.get('issue_description')
                    })
                    if draft:
                        # Update state with draft
                        data.update(draft)
                        st.success("Draft Generated!")
                        st.rerun()

    # --- WORKFLOW TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["1. Initiation", "2. Investigation & Risk", "3. Action Plan", "4. Verification"])

    with tab1:
        st.markdown("#### Event Identification")
        c1, c2 = st.columns(2)
        data['capa_id'] = c1.text_input("CAPA ID", value=data.get('capa_id', f"CAPA-{datetime.now().strftime('%y%m%d')}-001"))
        data['product_sku'] = c2.text_input("Product SKU", value=data.get('product_sku', ''))
        
        data['source'] = st.selectbox("Source", ["Customer Complaint", "Internal Audit", "Vision Diagnostic", "Supplier"], index=0)
        data['issue_description'] = st.text_area("Issue Description", value=data.get('issue_description', ''), height=150)

    with tab2:
        st.markdown("#### Risk Assessment (FMEA)")
        c1, c2, c3 = st.columns(3)
        sev = c1.slider("Severity (S)", 1, 5, value=3, help="1=Minor, 5=Critical")
        occ = c2.slider("Occurrence (O)", 1, 5, value=3, help="1=Rare, 5=Frequent")
        rpn = sev * occ
        
        color = "red" if rpn >= 15 else "orange" if rpn >= 8 else "green"
        c3.markdown(f"### RPN: <span style='color:{color}'>{rpn}</span>", unsafe_allow_html=True)
        
        st.markdown("#### Root Cause Analysis")
        data['root_cause_analysis'] = st.text_area("Root Cause (5 Whys / Fishbone)", value=data.get('root_cause_analysis', ''), height=150)
        data['containment_action'] = st.text_area("Immediate Containment", value=data.get('containment_action', ''), height=100)

    with tab3:
        st.markdown("#### Correction & Prevention")
        data['corrective_action'] = st.text_area("Corrective Action (Long Term)", value=data.get('corrective_action', ''), height=150)
        data['preventive_action'] = st.text_area("Preventive Action (Systemic)", value=data.get('preventive_action', ''), height=150)
        
        c1, c2 = st.columns(2)
        c1.date_input("Implementation Due Date")
        c2.text_input("Owner")

    with tab4:
        st.markdown("#### Effectiveness Check")
        data['effectiveness_plan'] = st.text_area("Verification Plan", value=data.get('effectiveness_plan', ''), height=100)
        
        if st.button("💾 Save CAPA Record", type="primary"):
            st.success(f"CAPA {data['capa_id']} Saved Successfully.")
            # In a real app, save to DB here

def render_strategy():
    render_header("Strategy & Compliance")
    st.info("ISO 13485 Document Generator")
    # Placeholder for document gen logic
    doc_type = st.selectbox("Document Type", ["Quality Manual", "SOP", "Work Instruction"])
    if st.button("Generate Template"):
        st.markdown(st.session_state.ai.generate(f"Write a {doc_type} template for ISO 13485"))

# --- MAIN APP ---
def main():
    nav = render_sidebar()
    if nav == "Dashboard": render_dashboard()
    elif nav == "Data Ingestion": render_ingestion()
    elif nav == "Vision": render_vision()
    elif nav == "CAPA Manager": render_capa()
    elif nav == "Strategy": render_strategy()

if __name__ == "__main__":
    main()
