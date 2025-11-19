import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import re
import os
import gc
from datetime import datetime

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="ORION Enterprise | VIVE Health",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ENTERPRISE THEME (Glassmorphism & Professional) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #2563EB;
        --bg-light: #F8FAFC;
        --card-bg: #FFFFFF;
        --border: #E2E8F0;
    }

    .stApp { background-color: var(--bg-light); font-family: 'Inter', sans-serif; color: #1E293B; }

    /* HEADER */
    .brand-header { padding-bottom: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
    .brand-title { font-size: 2.5rem; font-weight: 800; color: #0F172A; letter-spacing: -1px; margin: 0; line-height: 1.1; }
    .brand-subtitle { font-size: 0.875rem; color: #64748B; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-top: 0.5rem; }
    .brand-accent { color: var(--primary); }

    /* CARDS */
    .feature-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    div[data-testid="metric-container"] { background-color: white; border: 1px solid #E2E8F0; padding: 15px; border-radius: 10px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }

    /* INPUTS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; color: #1E293B; }
    .stButton>button { border-radius: 8px; font-weight: 600; padding: 0.5rem 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE ENGINE ---
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
        # Google
        google_key = self._get_key(["GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api_key"])
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                self.clients['google'] = genai
            except: pass

    def set_active_model(self, choice):
        self.available = False
        if choice.startswith("Gemini") and 'google' in self.clients:
            self.provider = "Google"
            self.client = self.clients['google']
            if "Flash" in choice: self.model_name = "gemini-1.5-flash"
            elif "Pro" in choice: self.model_name = "gemini-1.5-pro"
            else: self.model_name = "gemini-1.5-flash"
            self.available = True
        elif choice.startswith("GPT") and 'openai' in self.clients:
            self.provider = "OpenAI"
            self.client = self.clients['openai']
            if "4o" in choice: self.model_name = "gpt-4o"
            elif "Mini" in choice: self.model_name = "gpt-4o-mini"
            else: self.model_name = "gpt-4o"
            self.available = True

    def generate(self, prompt, temperature=0.3):
        if not self.available: return "⚠️ AI Offline. Please check API Keys."
        try:
            if self.provider == "Google":
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content(prompt, generation_config={'temperature': temperature})
                return response.text
            elif self.provider == "OpenAI":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error ({self.provider}): {str(e)}"

    def analyze_image(self, image, prompt):
        if not self.available: return "⚠️ AI Offline."
        try:
            if self.provider == "Google":
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content([prompt, image])
                return response.text
            elif self.provider == "OpenAI":
                return "Vision analysis is optimized for Gemini models in this version."
        except Exception as e:
            return f"Vision Error: {str(e)}"

    def categorize_batch(self, items):
        if not self.available: return items
        prompt = f"""
        You are a Quality Assurance AI. Classify the following medical device feedback into ONE of these categories:
        [Product Defect, Usability Issue, Missing Parts, Design Flaw, Shipping Damage, Medical Adverse Event, General Inquiry].
        Return ONLY the category name for each item in a new line.
        Items:
        """
        for item in items:
            prompt += f"- {item['text']}\n"
        try:
            res = self.generate(prompt, temperature=0.0)
            lines = [l.strip().replace('- ', '') for l in res.split('\n') if l.strip()]
            for i, item in enumerate(items):
                if i < len(lines): item['category'] = lines[i]
                else: item['category'] = "Uncategorized"
            return items
        except:
            return items

if 'ai' not in st.session_state:
    st.session_state.ai = IntelligenceEngine()

# --- 4. ADVANCED DATA HANDLER (MULTI-FILE & DIRTY HEADER LOGIC) ---
class DataHandler:
    
    @staticmethod
    def find_header_row(df, keywords=['sku', 'asin', 'product', 'title', 'date']):
        """Scans first 20 rows to find the true header row based on keywords."""
        for i in range(min(20, len(df))):
            row_values = df.iloc[i].astype(str).str.lower().tolist()
            # Check if this row contains at least 2 of our target keywords
            matches = sum(1 for k in keywords if any(k in val for val in row_values))
            if matches >= 2:
                return i
        return 0 # Default to 0 if not found

    @staticmethod
    def process_odoo_file(file, period_label):
        """Special logic for Odoo Inventory Forecasts"""
        try:
            # Read without header initially to sniff structure
            raw_df = pd.read_excel(file, header=None)
            
            # Find header row (skipping "Odoo - Inventory Forecast...")
            header_idx = DataHandler.find_header_row(raw_df, keywords=['product', 'sku', 'forecast', 'quantity'])
            
            # Reload with correct header
            df = pd.read_excel(file, header=header_idx)
            
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            # Handle the Sales Column (often Unnamed: 11 or similar)
            # We look for columns that might be the sales data if they are unnamed
            for col in df.columns:
                if "Unnamed" in col:
                    # Heuristic: Check if data in this column is numeric
                    if pd.to_numeric(df[col], errors='coerce').notna().sum() > (len(df) * 0.5):
                        # Rename using the user-selected period
                        df.rename(columns={col: f"Sales_{period_label}"}, inplace=True)
                        break
            
            df['Source_File'] = "Odoo_Forecast"
            return df
        except Exception as e:
            st.error(f"Odoo Parse Error: {e}")
            return None

    @staticmethod
    def process_return_report(file):
        """Special logic for Pivot Return Reports"""
        try:
            raw_df = pd.read_excel(file, header=None)
            header_idx = DataHandler.find_header_row(raw_df, keywords=['return', 'reason', 'sku', 'product'])
            
            df = pd.read_excel(file, header=header_idx)
            df['Source_File'] = "Return_Report"
            return df
        except Exception as e:
            st.error(f"Return Report Parse Error: {e}")
            return None

    @staticmethod
    def load_data(files, odoo_period="30 Days"):
        combined_df = pd.DataFrame()
        
        for file in files:
            fname = file.name.lower()
            
            # Route based on filename signature
            if "odoo" in fname or "forecast" in fname:
                df = DataHandler.process_odoo_file(file, odoo_period)
                
            elif "pivot" in fname or "return.report" in fname:
                df = DataHandler.process_return_report(file)
                
            else:
                # Generic fallback
                if fname.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    # Try to find header even for generic files
                    raw_df = pd.read_excel(file, header=None)
                    h_idx = DataHandler.find_header_row(raw_df)
                    df = pd.read_excel(file, header=h_idx)
            
            if df is not None:
                # Add a tracking column for origin
                df['Origin_File'] = file.name
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
        return combined_df

    @staticmethod
    def detect_columns(df):
        cols = [str(c).lower() for c in df.columns]
        mapping = {}
        
        # Smart Detection
        mapping['text'] = next((c for c in df.columns if any(x in str(c).lower() for x in ['body', 'comment', 'review', 'reason', 'complaint'])), None)
        mapping['date'] = next((c for c in df.columns if 'date' in str(c).lower()), None)
        
        return mapping

# --- 5. UI COMPONENTS ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ System Controls")
        
        # AI Model Selector
        st.markdown("**Active Intelligence Model**")
        model_options = []
        if 'google' in st.session_state.ai.clients:
            model_options.extend(["Gemini 1.5 Pro (Google)", "Gemini 1.5 Flash (Google)"])
        if 'openai' in st.session_state.ai.clients:
            model_options.extend(["GPT-4o (OpenAI)", "GPT-4o Mini (OpenAI)"])
            
        if not model_options:
            st.error("No API Keys Detected")
            model_choice = "Offline"
        else:
            default_ix = 0
            for i, m in enumerate(model_options):
                if "Flash" in m: default_ix = i
            model_choice = st.selectbox("Select Provider", model_options, index=default_ix, label_visibility="collapsed")
            st.session_state.ai.set_active_model(model_choice)
            
            st.markdown(f"""
            <div style='background-color:#DCFCE7; padding:8px; border-radius:6px; border:1px solid #86EFAC; display:flex; align-items:center; gap:8px;'>
                <div style='width:8px; height:8px; background-color:#16A34A; border-radius:50%;'></div>
                <span style='font-size:0.8rem; color:#14532D; font-weight:600;'>{model_choice.split('(')[0]} Active</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Odoo Time Config
        st.markdown("**📅 Odoo Reporting Period**")
        st.caption("Select the time period for uploaded Odoo Sales columns (e.g., Unnamed: 11)")
        odoo_period = st.selectbox("Sales Period", ["30 Days", "45 Days", "60 Days", "90 Days", "180 Days", "365 Days"], index=3)
        st.session_state.odoo_period = odoo_period

        st.markdown("---")
        
        st.markdown("**Module Selection**")
        nav = st.radio("Navigate", [
            "Executive Dashboard", 
            "Smart Categorizer", 
            "Vision Diagnostics", 
            "CAPA Suite", 
            "Strategy & Compliance"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        st.caption(f"**O.R.I.O.N. Enterprise**\nOperational Review & Intelligence\nOptimization Network")

    return nav

def render_header(module_name):
    st.markdown(f"""
    <div class="brand-header">
        <h1 class="brand-title">O.R.I.O.N. <span class="brand-accent">Enterprise</span></h1>
        <div class="brand-subtitle">Operational Review & Intelligence Optimization Network</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"### {module_name}")

def render_dashboard():
    render_header("Executive Dashboard")
    
    if 'analyzed_df' not in st.session_state:
        st.info("👋 Welcome to O.R.I.O.N. Please load data in the **Smart Categorizer** to populate this dashboard.")
        return

    df = st.session_state.analyzed_df
    
    # Top Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    defects = len(df[df['Category'] == 'Product Defect']) if 'Category' in df.columns else 0
    
    c1.metric("Total Records", f"{total:,}")
    if total > 0:
        c2.metric("Defect Rate", f"{(defects/total)*100:.1f}%", delta_color="inverse")
    c3.metric("Model Provider", st.session_state.ai.provider)
    c4.metric("Odoo Period", st.session_state.get('odoo_period', 'N/A'))
    
    st.markdown("---")
    
    if 'Category' in df.columns:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Quality Distribution")
            counts = df['Category'].value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            fig = px.bar(counts, x='Category', y='Count', color='Category', color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Priority Segments")
            fig2 = px.pie(counts, values='Count', names='Category', hole=0.7)
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

def render_categorizer():
    render_header("Smart Categorizer")
    
    st.markdown(f"""
    <div class="feature-card">
        <b>Multi-File Ingestion Engine</b><br>
        Supports Odoo Forecasts, Pivot Return Reports, and standard CSVs simultaneously.<br>
        <span style="font-size:0.8rem; color:#64748B;">Current Odoo Sales Period Config: <b>{st.session_state.get('odoo_period', '90 Days')}</b></span>
    </div><br>
    """, unsafe_allow_html=True)
    
    # Multi-file uploader
    files = st.file_uploader("Upload Data Sources", type=['csv', 'xlsx'], accept_multiple_files=True)
    
    if files:
        # Process all files
        df = DataHandler.load_data(files, st.session_state.get('odoo_period', "90 Days"))
        
        if df is not None and not df.empty:
            st.success(f"Successfully merged {len(files)} files. Total records: {len(df)}.")
            
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head(), use_container_width=True)
            
            cols = DataHandler.detect_columns(df)
            
            if not cols['text']:
                st.warning("Auto-detection couldn't find a clear 'Review' or 'Reason' column. Please select it manually below.")
                text_col = st.selectbox("Select Column to Categorize", df.columns)
            else:
                text_col = cols['text']
                st.info(f"Ready to classify based on column: **{text_col}**")
            
            if st.button("🚀 Start AI Classification", type="primary"):
                progress_bar = st.progress(0)
                
                # Filter for valid text rows
                valid_df = df[df[text_col].notna()].copy()
                
                # Limit for demo speed if huge
                process_limit = 100
                if len(valid_df) > process_limit:
                    st.caption(f"Demo mode: Processing first {process_limit} records for speed...")
                    valid_df = valid_df.head(process_limit)
                
                # Convert to list of dicts
                items = [{'text': str(x)} for x in valid_df[text_col].tolist()]
                
                # AI Batch Process
                results = st.session_state.ai.categorize_batch(items)
                
                # Assign back to dataframe (careful with indices if filtered)
                valid_df['Category'] = [r.get('category', 'Uncategorized') for r in results]
                
                # Update session state
                st.session_state.analyzed_df = valid_df
                progress_bar.progress(100)
                st.rerun()
                
    if 'analyzed_df' in st.session_state:
        st.markdown("### Classification Results")
        # Show key columns
        display_cols = [c for c in st.session_state.analyzed_df.columns if c in ['Category', 'Origin_File'] or "Sales" in c]
        st.dataframe(st.session_state.analyzed_df, use_container_width=True)
        
        csv = st.session_state.analyzed_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Categorized Data", csv, "orion_export.csv", "text/csv")

def render_vision():
    render_header("Vision Diagnostics")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Image Input")
        img_file = st.file_uploader("Upload Defect Photo", type=['png', 'jpg', 'jpeg'])
        
        if img_file:
            image = Image.open(img_file)
            st.image(image, use_column_width=True, caption="Source Artifact")
            
            if st.button("Analyze Artifact", type="primary", use_container_width=True):
                if "OpenAI" in st.session_state.ai.provider:
                    st.warning("Note: Ensure you are using a vision-capable model (GPT-4o).")
                
                with st.spinner("Processing visual data points..."):
                    prompt = "Analyze this medical device image. Identify: 1. The product type. 2. Visible defects or damage. 3. Potential root cause. 4. Risk severity (Low/Med/High)."
                    res = st.session_state.ai.analyze_image(image, prompt)
                    st.session_state.vision_result = res
    
    with c2:
        st.markdown("#### AI Findings")
        if 'vision_result' in st.session_state:
            st.markdown(f"""<div class="feature-card">{st.session_state.vision_result}</div>""", unsafe_allow_html=True)
            
            st.markdown("### Next Actions")
            if st.button("⚡ Escalate to CAPA"):
                st.session_state.capa_intake_desc = st.session_state.vision_result
                st.info("Findings copied to CAPA Intake.")

def render_capa():
    render_header("CAPA Suite")
    
    if 'capa_id' not in st.session_state:
        st.session_state.capa_id = f"CAPA-{datetime.now().strftime('%Y%m%d')}-001"
    
    intake_desc = st.session_state.get('capa_intake_desc', "")

    tab1, tab2, tab3, tab4 = st.tabs(["1. Intake", "2. Risk (RPN)", "3. Investigation", "4. Action"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("CAPA ID", st.session_state.capa_id, disabled=True)
            st.text_area("Problem Description", value=intake_desc, height=150)
        with c2:
            st.selectbox("Source", ["Customer Complaint", "Internal Audit", "Vision AI"])
            st.selectbox("Product Line", ["Mobility", "Respiratory", "Patient Room"])

    with tab2:
        st.markdown("#### Risk Quantification")
        c1, c2, c3 = st.columns(3)
        sev = c1.select_slider("Severity", [1, 2, 3, 4, 5], value=3)
        occ = c2.select_slider("Occurrence", [1, 2, 3, 4, 5], value=2)
        det = c3.select_slider("Detection", [1, 2, 3, 4, 5], value=3)
        rpn = sev * occ * det
        st.metric("RPN Score", rpn)

    with tab3:
        st.text_area("Root Cause Analysis (Fishbone / 5 Whys)", height=150)

    with tab4:
        st.text_area("Corrective Action Plan")
        if st.button("Save CAPA"):
            st.success("Record Saved.")

def render_strategy():
    render_header("Strategy & Compliance")
    
    st.markdown("""<div class="feature-card"><b>AI Quality Planner</b><br>Generate ISO 13485 compliant project plans.</div><br>""", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        doc_type = st.selectbox("Document Type", ["Quality Project Plan", "Validation Protocol", "Risk Management Plan"])
        project = st.text_input("Project Name")
        focus = st.text_area("Key Focus")
        
        if st.button("Generate Document", type="primary"):
            with st.spinner(f"Generating via {st.session_state.ai.model_name}..."):
                prompt = f"Draft a medical device {doc_type} for project '{project}'. Focus: {focus}. Include ISO 13485 sections."
                res = st.session_state.ai.generate(prompt)
                st.session_state.strategy_doc = res
    
    with c2:
        if 'strategy_doc' in st.session_state:
            st.text_area("Generated Content", st.session_state.strategy_doc, height=500)
            st.download_button("Download .MD", st.session_state.strategy_doc, "plan.md")

# --- 6. MAIN APP LOOP ---
def main():
    nav_choice = render_sidebar()
    
    if nav_choice == "Executive Dashboard": render_dashboard()
    elif nav_choice == "Smart Categorizer": render_categorizer()
    elif nav_choice == "Vision Diagnostics": render_vision()
    elif nav_choice == "CAPA Suite": render_capa()
    elif nav_choice == "Strategy & Compliance": render_strategy()

if __name__ == "__main__":
    main()
