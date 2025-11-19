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
    .brand-header { padding-bottom: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
    .brand-title { font-size: 2.2rem; font-weight: 800; color: var(--primary); letter-spacing: -0.5px; margin: 0; }
    .brand-subtitle { font-size: 0.9rem; color: #64748B; font-weight: 500; margin-top: 0.25rem; }
    .feature-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    
    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: white; border-radius: 6px; border: 1px solid #E2E8F0; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: var(--accent); color: white; border-color: var(--accent); }
    </style>
""", unsafe_allow_html=True)

# --- 3. INTELLIGENCE ENGINE ---
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
                try:
                    import google.generativeai as genai
                except ImportError:
                    self.connection_error = "Missing Library: google-generativeai"
                    return

                api_key = self._get_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], manual_key_input)
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai
                    self.provider = "Google Gemini"
                    self.model_name = "gemini-1.5-flash" if "Flash" in provider_choice else "gemini-1.5-pro"
                    self.available = True
                else:
                    self.connection_error = "Missing Google API Key"

            elif "GPT" in provider_choice:
                try:
                    import openai
                except ImportError:
                    self.connection_error = "Missing Library: openai"
                    return

                api_key = self._get_key(["OPENAI_API_KEY"], manual_key_input)
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.provider = "OpenAI"
                    self.model_name = "gpt-4o" if "4o" in provider_choice else "gpt-4o-mini"
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
                kwargs = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
                if json_mode: kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
        except Exception as e: return f"Generation Error: {str(e)}"

    def analyze_image(self, image, prompt):
        if not self.available: return f"AI Offline: {self.connection_error}"
        try:
            if "Gemini" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content([prompt, image])
                return response.text
            elif "OpenAI" in self.provider:
                return "Switch to Gemini for vision analysis."
        except Exception as e: return f"Vision Error: {e}"

    def generate_capa_draft(self, context):
        prompt = f"Create a CAPA investigation JSON for {context.get('product')}. Issue: {context.get('issue')}. Include keys: issue_description, root_cause_analysis, immediate_action, corrective_action, effectiveness_check."
        res = self.generate(prompt, json_mode=True)
        try:
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            return json.loads(res)
        except: return None

if 'ai' not in st.session_state: st.session_state.ai = IntelligenceEngine()

# --- 4. DATA HANDLER (Enhanced Grouping & Merging) ---
class DataHandler:
    @staticmethod
    def clean_sku(sku):
        """Extract Parent SKU (MOB1027BLK -> MOB1027)"""
        if pd.isna(sku): return "UNKNOWN"
        sku = str(sku).upper().strip()
        # Regex: Start with Letters, then Numbers. Capture that group.
        match = re.match(r"^([A-Z]+[0-9]+)", sku)
        return match.group(1) if match else sku

    @staticmethod
    def get_category(sku):
        """Extract Category (MOB1027 -> MOB)"""
        if pd.isna(sku): return "Other"
        sku = str(sku).upper().strip()
        match = re.match(r"^([A-Z]+)", sku)
        return match.group(1) if match else "Other"

    @staticmethod
    def load_and_merge(files, period_days=90, manual_data=None):
        """
        Merges Odoo (Sales), Return Reports (Returns), and Manual Data.
        Returns a unified DataFrame keyed by Variant SKU.
        """
        warnings = []
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Temp storage
        sales_records = [] # {'SKU': ..., 'Sales': ...}
        return_records = [] # {'SKU': ..., 'Returns': 1, 'Issue': ...}

        # 1. Process Uploaded Files
        for file in files:
            try:
                fname = file.name.lower()
                raw = pd.read_excel(file, header=None)
                
                # Header Hunt
                header_idx = 0
                for i in range(min(20, len(raw))):
                    row_str = raw.iloc[i].astype(str).str.lower().tolist()
                    if sum(1 for k in ['sku', 'product', 'asin', 'date', 'qty', 'sales'] if any(k in str(rs) for rs in row_str)) >= 2:
                        header_idx = i
                        break
                
                df = pd.read_excel(file, header=header_idx)
                df.columns = [str(c).strip() for c in df.columns]
                
                sku_col = next((c for c in df.columns if any(x in c.lower() for x in ['sku', 'product', 'default code'])), None)
                if not sku_col: continue

                # Odoo Sales Logic
                if "odoo" in fname or "forecast" in fname:
                    sales_col = None
                    candidates = [c for c in df.columns if any(x in c.lower() for x in ['sales', 'qty', 'quantity', 'unnamed'])]
                    numeric_candidates = [c for c in candidates if pd.to_numeric(df[c], errors='coerce').sum() > 0]
                    if numeric_candidates: sales_col = numeric_candidates[-1]
                    
                    if sales_col:
                        for _, row in df.iterrows():
                            sku = str(row[sku_col]).strip().upper()
                            qty = pd.to_numeric(row[sales_col], errors='coerce') or 0
                            sales_records.append({'SKU': sku, 'Sales': qty})

                # Return Report Logic
                elif "return" in fname or "pivot" in fname:
                    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                    reason_col = next((c for c in df.columns if any(x in c.lower() for x in ['reason', 'comment'])), None)
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df[df[date_col] >= cutoff_date] # 1:1 Filter
                    else:
                        warnings.append(f"⚠️ {file.name}: No date column. All returns included.")

                    for _, row in df.iterrows():
                        sku = str(row[sku_col]).strip().upper()
                        issue = str(row[reason_col]) if reason_col and pd.notna(row[reason_col]) else None
                        return_records.append({'SKU': sku, 'Returns': 1, 'Issue': issue})

            except Exception as e:
                warnings.append(f"Error {file.name}: {str(e)}")

        # 2. Process Manual Data
        if manual_data is not None and not manual_data.empty:
            for _, row in manual_data.iterrows():
                sku = str(row.get('SKU', '')).strip().upper()
                if sku:
                    if 'Sales' in row and row['Sales'] > 0:
                        sales_records.append({'SKU': sku, 'Sales': row['Sales']})
                    if 'Returns' in row and row['Returns'] > 0:
                        # Manual returns added as aggregate counts
                        for _ in range(int(row['Returns'])):
                            return_records.append({'SKU': sku, 'Returns': 1, 'Issue': 'Manual Entry'})

        # 3. Merge & Aggregate
        sales_df = pd.DataFrame(sales_records)
        if not sales_df.empty:
            sales_df = sales_df.groupby('SKU')['Sales'].sum().reset_index()
        else:
            sales_df = pd.DataFrame(columns=['SKU', 'Sales'])

        returns_df = pd.DataFrame(return_records)
        if not returns_df.empty:
            # Count returns per SKU
            returns_agg = returns_df.groupby('SKU')['Returns'].sum().reset_index()
            # Aggregate issues
            issues_agg = returns_df.groupby('SKU')['Issue'].apply(lambda x: [i for i in x if i]).reset_index()
            returns_final = pd.merge(returns_agg, issues_agg, on='SKU')
        else:
            returns_final = pd.DataFrame(columns=['SKU', 'Returns', 'Issue'])

        # Full Outer Join to keep all data
        master = pd.merge(sales_df, returns_final, on='SKU', how='outer').fillna(0)
        
        # Add Metadata
        if not master.empty:
            master['Returns'] = master['Returns'].astype(int)
            master['Sales'] = master['Sales'].astype(int)
            master['Parent_SKU'] = master['SKU'].apply(DataHandler.clean_sku)
            master['Category'] = master['SKU'].apply(DataHandler.get_category)
            master['Return_Rate'] = (master['Returns'] / master['Sales'] * 100).fillna(0)
            # Cap return rate for visualization if sales are 0 but returns exist
            master['Return_Rate'] = master.apply(lambda x: 100.0 if x['Sales'] == 0 and x['Returns'] > 0 else x['Return_Rate'], axis=1)
            # Format Issues List
            master['Issues'] = master['Issue'].apply(lambda x: x if isinstance(x, list) else [])
            master = master.drop(columns=['Issue'])

        return master, warnings

# --- 5. UI MODULES ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### System Controls")
        
        # AI Config
        st.markdown("**AI Engine**")
        model_choice = st.selectbox("Provider", 
            ["Google Gemini 1.5 Flash", "Google Gemini 1.5 Pro", "OpenAI GPT-4o", "OpenAI GPT-4o Mini"], index=0)
        
        manual_key = None
        needs_key = ("Gemini" in model_choice and "GOOGLE_API_KEY" not in st.secrets) or \
                    ("GPT" in model_choice and "OPENAI_API_KEY" not in st.secrets)
        
        if needs_key:
            st.warning(f"No secret found for {model_choice[:10]}...")
            manual_key = st.text_input("Enter API Key", type="password")
        
        st.session_state.ai.configure_client(model_choice, manual_key)
        
        if st.session_state.ai.available:
            st.markdown(f"<span style='color:green'>● {model_choice} Online</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'>● Offline</span>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Filtering & Scope
        st.markdown("**Analysis Scope**")
        st.caption("1:1 Date Alignment")
        st.session_state.period_days = st.selectbox("Time Window", [30, 60, 90, 180, 365], index=2)

        # Dynamic Filters
        if 'master_data' in st.session_state and not st.session_state.master_data.empty:
            df = st.session_state.master_data
            st.markdown("---")
            st.markdown("**Product Filters**")
            
            # Level 1: Category
            cats = ["All"] + sorted(df['Category'].unique().tolist())
            sel_cat = st.selectbox("Category", cats)
            
            # Level 2: Parent SKU (Filtered)
            if sel_cat != "All":
                df = df[df['Category'] == sel_cat]
            
            parents = ["All"] + sorted(df['Parent_SKU'].unique().tolist())
            sel_parent = st.selectbox("Parent Family", parents)
            
            # Level 3: Variant SKU (Filtered)
            if sel_parent != "All":
                df = df[df['Parent_SKU'] == sel_parent]
            
            variants = ["All"] + sorted(df['SKU'].unique().tolist())
            sel_variant = st.selectbox("Specific Product", variants)
            
            # Apply Final Filter to View
            if sel_variant != "All":
                df = df[df['SKU'] == sel_variant]
            
            st.session_state.filtered_data = df
            st.caption(f"Viewing {len(df)} records")

        st.markdown("---")
        return st.radio("Navigation", ["Dashboard", "Data Ingestion", "Vision", "CAPA Manager"], label_visibility="collapsed")

def render_ingestion():
    st.markdown("### Data Ingestion")
    
    t1, t2 = st.tabs(["File Upload", "Manual Entry"])
    
    with t1:
        st.markdown(f"""
        <div class='feature-card'>
            <b>Auto-Parsing Rules:</b><br>
            • <b>Sales:</b> Extracts from Odoo 'Qty'/'Sales' columns.<br>
            • <b>Returns:</b> Filters to last <b>{st.session_state.period_days} days</b>.<br>
            • <b>Merging:</b> Aligns on SKU, aggregates to Parent/Category.
        </div>
        """, unsafe_allow_html=True)
        
        files = st.file_uploader("Upload Odoo & Return Files", accept_multiple_files=True)
        
    with t2:
        st.info("Enter specific sales/returns adjustments manually below.")
        if 'manual_entry_df' not in st.session_state:
            st.session_state.manual_entry_df = pd.DataFrame(
                [{"SKU": "EXAMPLE123", "Sales": 100, "Returns": 5}],
                columns=["SKU", "Sales", "Returns"]
            )
        
        manual_data = st.data_editor(
            st.session_state.manual_entry_df,
            num_rows="dynamic",
            column_config={
                "SKU": st.column_config.TextColumn("Product SKU", required=True),
                "Sales": st.column_config.NumberColumn("Sales Qty", min_value=0),
                "Returns": st.column_config.NumberColumn("Returns Qty", min_value=0),
            },
            use_container_width=True
        )
        st.session_state.manual_entry_df = manual_data

    if st.button("Process All Data", type="primary"):
        with st.spinner("Merging datasets..."):
            # Pass both files and manual data to handler
            df, warns = DataHandler.load_and_merge(
                files if files else [], 
                st.session_state.period_days, 
                manual_data
            )
            st.session_state.master_data = df
            st.session_state.filtered_data = df # Reset filter
            
            for w in warns: st.warning(w)
            st.success(f"Processed {len(df)} SKUs successfully.")

def render_dashboard():
    st.markdown("### Executive Dashboard")
    
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data.empty:
        st.info("No data loaded or filter is too restrictive. Go to Data Ingestion.")
        return
        
    df = st.session_state.filtered_data
    
    # 1. Aggregate Metrics (weighted averages/sums)
    total_sales = int(df['Sales'].sum())
    total_returns = int(df['Returns'].sum())
    # Calculate global return rate for the current selection
    avg_rate = (total_returns / total_sales * 100) if total_sales > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"{total_sales:,}")
    c2.metric("Total Returns", f"{total_returns:,}")
    c3.metric("Avg Return Rate", f"{avg_rate:.2f}%")
    c4.metric("Active SKUs", len(df))
    
    st.markdown("---")
    
    # 2. Sales & Returns by Category/Parent
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Group by Parent SKU for cleaner chart
        chart_df = df.groupby('Parent_SKU')[['Sales', 'Returns']].sum().reset_index()
        chart_df['Rate'] = (chart_df['Returns'] / chart_df['Sales'] * 100).fillna(0)
        chart_df = chart_df.sort_values('Rate', ascending=False).head(15)
        
        fig = px.bar(chart_df, x='Parent_SKU', y=['Sales', 'Returns'], 
                     title="Sales vs Returns (Top 15 by Rate)", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("#### Detailed Data")
        st.dataframe(
            df[['SKU', 'Sales', 'Returns', 'Return_Rate']].sort_values('Return_Rate', ascending=False),
            use_container_width=True,
            hide_index=True
        )

def render_capa():
    st.markdown("### CAPA Manager")
    if 'capa_data' not in st.session_state: st.session_state.capa_data = {'id': f"CAPA-{datetime.now().strftime('%y%m%d')}", 'risks': pd.DataFrame([{'Mode': 'Failure', 'Sev': 3, 'Occ': 2, 'Det': 5}])}
    data = st.session_state.capa_data
    
    if 'capa_prefill' in st.session_state:
        data['desc'] = st.session_state.capa_prefill.get('desc')
        del st.session_state.capa_prefill

    t1, t2, t3 = st.tabs(["Intake", "Investigation", "Risk (FMEA)"])
    
    with t1:
        c1, c2 = st.columns(2)
        data['id'] = c1.text_input("ID", data['id'])
        data['sku'] = c2.text_input("SKU", data.get('sku',''))
        data['desc'] = st.text_area("Issue", data.get('desc',''))
        
        if st.button("✨ Auto-Draft"):
            with st.spinner("AI Drafting..."):
                draft = st.session_state.ai.generate_capa_draft({'product':data['sku'], 'issue':data['desc']})
                if draft: 
                    data.update(draft)
                    st.success("Drafted!")
                    st.rerun()
                    
    with t2:
        st.markdown("**Root Cause Analysis (Fishbone)**")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            cat = st.selectbox("Category", ["Man", "Machine", "Material", "Method"])
            cause = st.text_input("Cause")
            if st.button("Add Cause"):
                if 'fishbone' not in data: data['fishbone'] = []
                data['fishbone'].append((cat, cause))
        with col_b:
            if data.get('fishbone'):
                g = graphviz.Digraph()
                g.attr(rankdir='LR')
                g.node('Problem', 'Issue')
                for c, r in data['fishbone']:
                    g.edge(c, 'Problem')
                    g.edge(r, c)
                st.graphviz_chart(g)

    with t3:
        st.markdown("**Risk Analysis**")
        edited = st.data_editor(data['risks'], num_rows="dynamic", use_container_width=True)
        data['risks'] = edited

def render_vision():
    st.markdown("### Vision Diagnostics")
    img = st.file_uploader("Upload Image", type=['png','jpg'])
    if img:
        st.image(img, width=300)
        if st.button("Analyze"):
            res = st.session_state.ai.analyze_image(Image.open(img), "Analyze defect severity.")
            st.markdown(f"<div class='feature-card'>{res}</div>", unsafe_allow_html=True)
            if st.button("Escalate to CAPA"):
                st.session_state.capa_prefill = {'desc': res}
                st.success("Sent to CAPA")

# --- MAIN ---
def main():
    nav = render_sidebar()
    if nav == "Dashboard": render_dashboard()
    elif nav == "Data Ingestion": render_ingestion()
    elif nav == "CAPA Manager": render_capa()
    elif nav == "Vision": render_vision()

if __name__ == "__main__":
    main()
