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
    page_title="O.R.I.O.N. v8.0 | VIVE Health",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VIVE BRAND THEME (MINIMALIST PRODUCTION) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&family=Open+Sans:wght@400;600&display=swap');

    /* GLOBAL APP RESET */
    .stApp {
        background-color: #0B1E3D !important; /* Vive Navy */
        color: #FFFFFF !important;
        font-family: 'Open Sans', sans-serif;
    }

    /* HEADERS (Redzone Style) */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #FFFFFF !important;
    }
    h1 {
        font-size: 2.5rem !important;
        color: #00C6D7 !important; /* Vive Teal */
        margin-bottom: 0px;
    }
    .subtitle {
        color: #94A3B8;
        font-family: 'Montserrat', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }

    /* CARDS & CONTAINERS */
    .stContainer, div[data-testid="metric-container"], .report-box, .info-box {
        background-color: #132448 !important;
        border: 1px solid #1E3A5F;
        border-radius: 6px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* METRICS */
    div[data-testid="metric-container"] {
        border-left: 4px solid #00C6D7;
        background-color: #0F2042 !important;
    }
    div[data-testid="metric-container"] label {
        color: #00C6D7 !important;
        font-weight: 700;
        font-size: 0.85rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 700;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050E1F !important;
        border-right: 1px solid #1E3A5F;
    }

    /* BUTTONS (Primary Action) */
    .stButton>button {
        background-color: #00C6D7 !important;
        color: #050E1F !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        border: none;
        border-radius: 4px;
        height: 3.2em;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #FFFFFF !important;
        color: #00C6D7 !important;
        box-shadow: 0 0 15px rgba(0, 198, 215, 0.4);
    }

    /* INPUTS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] div, .stTextArea textarea {
        background-color: #1B3B6F !important;
        color: white !important;
        border: 1px solid #475569;
    }
    
    /* PROGRESS BAR */
    .stProgress > div > div {
        background-color: #00C6D7 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. INTELLIGENCE ENGINE (ROBUST) ---
class IntelligenceEngine:
    def __init__(self):
        self.client = None
        self.provider = None
        self.available = False
        self._initialize_ai_clients()

    def _initialize_ai_clients(self):
        """Initialize AI clients from Streamlit secrets (Robust Pattern)"""
        try:
            # 1. Try Google Gemini (Visual capabilities)
            if 'GEMINI_API_KEY' in st.secrets:
                self._configure_gemini(st.secrets['GEMINI_API_KEY'])
            elif 'GOOGLE_API_KEY' in st.secrets:
                self._configure_gemini(st.secrets['GOOGLE_API_KEY'])
            
            # 2. Try OpenAI (Text capabilities fallback)
            if not self.available and 'OPENAI_API_KEY' in st.secrets:
                self._configure_openai(st.secrets['OPENAI_API_KEY'])
            
            # 3. Try Env Vars
            if not self.available:
                key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if key:
                    if key.startswith("sk-"): self._configure_openai(key)
                    else: self._configure_gemini(key)

            # 4. Manual Override (Session State)
            if not self.available and st.session_state.get("manual_key"):
                key = st.session_state.manual_key
                if key.startswith("sk-"): self._configure_openai(key)
                else: self._configure_gemini(key)

        except Exception as e:
            st.session_state.ai_error = str(e)
            self.available = False

    def _configure_gemini(self, key):
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self.client = genai
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.provider = "Gemini"
            self.available = True
        except:
            pass

    def _configure_openai(self, key):
        try:
            import openai
            self.client = openai.OpenAI(api_key=key)
            self.provider = "OpenAI"
            self.available = True
        except:
            pass

    def generate(self, prompt):
        if not self.available: return "⚠️ AI Offline. Check API configuration."
        try:
            if self.provider == "Gemini":
                return self.model.generate_content(prompt).text
            elif self.provider == "OpenAI":
                response = self.client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def analyze_vision(self, image, prompt):
        if not self.available: return "⚠️ AI Offline."
        try:
            if self.provider == "Gemini":
                return self.vision.generate_content([prompt, image]).text
            else:
                return "Vision analysis requires Gemini API."
        except Exception as e:
            return f"Error: {e}"

    def categorize_batch(self, batch_data):
        """Optimized batch processor for Return Categorizer"""
        if not self.available: return batch_data
        
        # Construct a batch prompt
        prompt = "Categorize the following medical device complaints into: [Product Defects, Performance Issues, Missing Components, Design Issues, Stability Issues, Medical Concerns, Other]. Return ONLY the category name for each line.\n\n"
        for item in batch_data:
            prompt += f"- {item['complaint']}\n"
            
        try:
            response = self.generate(prompt)
            categories = response.strip().split('\n')
            
            # Map results back to batch
            for i, item in enumerate(batch_data):
                if i < len(categories):
                    # Clean formatting bullets if AI adds them
                    cat = categories[i].replace('- ', '').strip()
                    item['category'] = cat
                else:
                    item['category'] = "Uncategorized"
            return batch_data
        except Exception as e:
            print(f"Batch Error: {e}")
            for item in batch_data: item['category'] = "Error"
            return batch_data

# Initialize Singleton
if 'ai' not in st.session_state:
    st.session_state.ai = IntelligenceEngine()

# --- 3. DATA ENGINE (PRODUCTION GRADE) ---
class DataEngine:
    @staticmethod
    def process_file_preserve_structure(file_content, filename):
        """Reads file as string/object to preserve structure (00123 SKU issue)"""
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content), dtype=str)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file_content), dtype=str)
            elif filename.endswith('.txt'):
                df = pd.read_csv(io.BytesIO(file_content), sep='\t', dtype=str)
            else:
                return None, None
            
            # Map Columns (Column I = Complaint, K = Category)
            col_map = {}
            cols = df.columns.tolist()
            
            # Logic for user's specific template
            if len(cols) >= 9: # At least up to I
                col_map['complaint'] = cols[8] # Column I (0-index 8)
                col_map['sku'] = cols[1] if len(cols) > 1 else None # Column B
                
                # Ensure K exists
                if len(cols) > 10:
                    col_map['category'] = cols[10]
                else:
                    # Add columns if needed
                    while len(df.columns) < 11:
                        df[f'Col_{len(df.columns)}'] = ''
                    col_map['category'] = df.columns[10]
                    
            return df, col_map
        except Exception as e:
            return None, None

    @staticmethod
    def export_data(df):
        output = io.BytesIO()
        # Try Excel for best formatting
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Categorized')
        except:
            # Fallback CSV
            df.to_csv(output, index=False)
        return output.getvalue()

# --- 4. MODULES ---

def render_dashboard():
    st.markdown("<h1>O.R.I.O.N.</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>OPERATIONAL REVIEW & INTELLIGENCE OPTIMIZATION NETWORK</div>", unsafe_allow_html=True)
    
    # Simple Executive Summary of the Categories if they exist
    if 'categorized_data' in st.session_state and st.session_state.categorized_data is not None:
        df = st.session_state.categorized_data
        col_map = st.session_state.column_mapping
        cat_col = col_map.get('category')
        
        if cat_col:
            counts = df[cat_col].value_counts()
            total = len(df)
            quality_issues = sum(1 for c in df[cat_col] if "Defect" in str(c) or "Quality" in str(c))
            
            m1, m2, m3 = st.columns(3)
            m1.metric("TOTAL PROCESSED", total)
            m2.metric("QUALITY FLAGS", quality_issues)
            m3.metric("DEFECT RATE", f"{(quality_issues/total*100):.1f}%")
            
            st.markdown("### 📉 CATEGORY BREAKDOWN")
            st.bar_chart(counts)
    else:
        st.info("👋 Welcome to O.R.I.O.N. v8.0. Go to **CATEGORIZER** to process return data.")

def render_categorizer():
    st.markdown("<h1>RETURN CATEGORIZER</h1>", unsafe_allow_html=True)
    st.caption("PRODUCTION ENGINE v16.1 | COLUMN I → COLUMN K")
    
    # File Upload
    f = st.file_uploader("Upload Return Report (Excel/CSV)", type=['xlsx', 'csv', 'txt'])
    
    if f:
        if 'original_file' not in st.session_state or st.session_state.original_file != f.name:
            df, col_map = DataEngine.process_file_preserve_structure(f.getvalue(), f.name)
            if df is not None:
                st.session_state.raw_df = df
                st.session_state.col_map = col_map
                st.session_state.original_file = f.name
                st.session_state.processing_complete = False
    
    if 'raw_df' in st.session_state:
        df = st.session_state.raw_df
        col_map = st.session_state.col_map
        
        # Stats
        c_col = col_map.get('complaint')
        valid = df[df[c_col].notna() & (df[c_col] != '')].shape[0]
        
        st.markdown(f"""
        <div class='info-box'>
            <b>FILE LOCKED:</b> {st.session_state.original_file}<br>
            <b>TOTAL ROWS:</b> {len(df):,}<br>
            <b>VALID COMPLAINTS (COL I):</b> {valid:,}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"🚀 CATEGORIZE {valid:,} ROWS", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            # Batch Processing Logic
            batch_size = 50
            complaint_col_idx = df.columns.get_loc(col_map['complaint'])
            cat_col_idx = df.columns.get_loc(col_map['category'])
            
            valid_indices = df[df[col_map['complaint']].notna()].index.tolist()
            total_batches = (len(valid_indices) + batch_size - 1) // batch_size
            
            processed_count = 0
            
            for i in range(0, len(valid_indices), batch_size):
                batch_idxs = valid_indices[i:i+batch_size]
                
                # Prepare batch
                batch_payload = []
                for idx in batch_idxs:
                    txt = str(df.iat[idx, complaint_col_idx])
                    batch_payload.append({'index': idx, 'complaint': txt})
                
                # AI Call
                results = st.session_state.ai.categorize_batch(batch_payload)
                
                # Update DF
                for item in results:
                    df.iat[item['index'], cat_col_idx] = item.get('category', 'Other')
                
                processed_count += len(batch_idxs)
                progress.progress(processed_count / len(valid_indices))
                status.text(f"Processed {processed_count}/{len(valid_indices)}...")
                
                # Garbage Collection
                if i % 500 == 0: gc.collect()
            
            st.session_state.categorized_data = df
            st.session_state.processing_complete = True
            st.success("✅ CATEGORIZATION COMPLETE")
            st.rerun()

    # Download Section
    if st.session_state.get('processing_complete'):
        st.markdown("---")
        st.markdown("### 📥 AUTO-DOWNLOAD")
        
        data = DataEngine.export_data(st.session_state.categorized_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        st.download_button(
            label="DOWNLOAD PROCESSED FILE",
            data=data,
            file_name=f"Categorized_Returns_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
        st.caption("File includes all original columns + AI Categories in Column K.")

def render_voc():
    st.markdown("<h1>VISION INTELLIGENCE</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        img = st.file_uploader("Upload Screenshot", type=['png', 'jpg'])
        if img:
            st.image(img, use_column_width=True)
            if st.button("SCAN", type="primary"):
                res = st.session_state.ai.analyze_vision(Image.open(img), "Extract metrics, defects, and sentiment.")
                st.session_state.voc_res = res
    with c2:
        if 'voc_res' in st.session_state:
            st.markdown(st.session_state.voc_res)
            if st.button("INITIATE CAPA"):
                st.session_state.capa_prefill = st.session_state.voc_res
                st.session_state.nav = "CAPA MANAGER"
                st.rerun()

def render_plan():
    st.markdown("<h1>STRATEGY PLANNER</h1>", unsafe_allow_html=True)
    st.info("AI-Assisted Quality Planning Module")
    c1, c2 = st.columns(2)
    st.session_state.qp_name = c1.text_input("Project Name")
    st.session_state.qp_risk = c2.selectbox("Risk", ["Class I", "Class II"])
    
    with st.expander("SCOPE & OBJECTIVES", expanded=True):
        st.text_area("Draft Content", height=100)
        if st.button("AI OPTIMIZE"):
            st.write(st.session_state.ai.generate("Write a quality plan scope"))

def render_capa():
    st.markdown("<h1>CAPA MANAGER</h1>", unsafe_allow_html=True)
    st.caption("CORRECTIVE & PREVENTIVE ACTION SUITE")
    
    if 'capa_id' not in st.session_state: st.session_state.capa_id = f"CAPA-{int(time.time())}"
    prefill = st.session_state.get("capa_prefill", "")

    tabs = st.tabs(["1. INTAKE", "2. RISK", "3. INVESTIGATION", "4. ACTION", "5. VERIFICATION", "6. COST"])

    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.text_input("CAPA ID", st.session_state.capa_id, disabled=True)
            st.text_area("DESCRIPTION", value=prefill, height=150)
        with c2:
            st.selectbox("SOURCE", ["Customer", "Audit", "NCR"])
            st.selectbox("OWNER", ["QA", "Ops", "Product"])

    with tabs[1]:
        st.markdown("### RISK MATRIX")
        r1, r2 = st.columns(2)
        sev = r1.select_slider("SEVERITY", options=[1, 2, 3, 4, 5])
        occ = r2.select_slider("OCCURRENCE", options=[1, 2, 3, 4, 5])
        st.metric("RPN", sev * occ)

    with tabs[2]:
        st.markdown("### RCA")
        w1 = st.text_input("1. WHY?")
        if st.button("AI COACH"):
            st.info(st.session_state.ai.generate(f"Root cause for: {w1}"))

    with tabs[3]:
        st.text_input("CORRECTIVE ACTION")
        st.date_input("DUE DATE")

    with tabs[4]:
        st.text_area("EVIDENCE OF EFFECTIVENESS")
        if st.button("CLOSE CAPA"): st.balloons()

    with tabs[5]:
        st.number_input("COST OF QUALITY ($)", 0.0)

# --- 5. MAIN CONTROLLER ---
def main():
    with st.sidebar:
        st.title("O.R.I.O.N.")
        st.caption("VIVE HEALTH v8.0 | EXEC BUILD")
        
        # AI Status
        if st.session_state.ai.available:
            st.success(f"🟢 ONLINE ({st.session_state.ai.provider})")
        else:
            st.error("🔴 OFFLINE")
            if st.session_state.get("ai_error"):
                st.caption(st.session_state.ai_error)
            k = st.text_input("MANUAL KEY", type="password")
            if k:
                st.session_state.manual_key = k
                st.session_state.ai._initialize_ai_clients()
                st.rerun()
        
        st.markdown("---")
        
        if 'nav' not in st.session_state: st.session_state.nav = "DASHBOARD"
        
        opts = {
            "DASHBOARD": "📊", 
            "CATEGORIZER": "📂",
            "VISION INTEL": "👁️", 
            "STRATEGY": "📝", 
            "CAPA MANAGER": "🛡️"
        }
        
        for label, icon in opts.items():
            if st.button(f"{icon}  {label}", use_container_width=True):
                st.session_state.nav = label
                st.rerun()
        
        st.markdown("""
        <div style='text-align:center; color:#5d6d8a; font-size:0.8rem; margin-top:30px;'>
        built by alex popoff 11/19/2025<br>
        v.8.0 gemini vibe coded beta
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.nav == "DASHBOARD": render_dashboard()
    elif st.session_state.nav == "CATEGORIZER": render_categorizer()
    elif st.session_state.nav == "VISION INTEL": render_voc()
    elif st.session_state.nav == "STRATEGY": render_plan()
    elif st.session_state.nav == "CAPA MANAGER": render_capa()

if __name__ == "__main__":
    main()
