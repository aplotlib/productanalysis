import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
import io
import time
import re
from datetime import datetime
import collections

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Product Lifecycle Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API SETUP ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""  # Handler for local/missing key

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 6px; height: 3em; font-weight: 600; width: 100%; }
    .report-box { border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; background: white; margin-bottom: 15px; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #fff; border-radius: 4px 4px 0px 0px; border: 1px solid #ddd; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: #e8f0fe; color: #1a73e8; border-bottom: 2px solid #1a73e8; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- AI ENGINE ---
class AIClient:
    def __init__(self):
        self.enabled = bool(api_key)
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            self.vision = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

    def generate(self, prompt, temperature=0.7):
        if not self.enabled: return "⚠️ AI Key Missing"
        try:
            return self.model.generate_content(prompt, generation_config={"temperature": temperature}).text
        except Exception as e:
            return f"AI Error: {e}"

    def analyze_image(self, image, prompt):
        if not self.enabled: return "⚠️ AI Key Missing"
        try:
            return self.vision.generate_content([prompt, image]).text
        except Exception as e:
            return f"Vision Error: {e}"

if 'ai' not in st.session_state:
    st.session_state.ai = AIClient()

# --- UTILITIES ---
def clean_text_for_export(text):
    """Removes markdown artifacts for raw text export."""
    if not text: return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'#+\s', '', text)              # Headers
    text = re.sub(r'__', '', text)                # Underscores
    text = re.sub(r'`', '', text)                 # Code ticks
    return text.strip()

def smart_odoo_parser(file):
    """
    Intelligently finds the header row in poorly structured Odoo CSV/Excel exports.
    """
    try:
        content = file.getvalue().decode("utf-8", errors='replace')
        lines = content.split('\n')
        
        # Odoo often has metadata in the first few lines. We look for the header.
        # Keywords commonly found in Odoo headers
        keywords = ['id', 'date', 'product', 'sku', 'qty', 'quantity', 'status', 'name', 'reference', 'priority', 'stage', 'ticket']
        
        best_idx = 0
        max_score = 0
        
        # Scan first 20 lines
        for i, line in enumerate(lines[:20]):
            score = sum(1 for k in keywords if k in line.lower())
            if score > max_score:
                max_score = score
                best_idx = i
        
        # Reload with correct header
        file.seek(0)
        df = pd.read_csv(file, header=best_idx)
        
        # Cleanup: Drop rows that are entirely empty or just summary lines
        df = df.dropna(how='all')
        # Filter out rows where 'Product' or key ID fields are empty if possible
        return df, None
    except Exception as e:
        return None, str(e)

# --- MODULE 1: DASHBOARD ---
def render_dashboard():
    st.title("📊 Product Lifecycle Intelligence")
    
    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h3>4.2</h3><p>Avg Star Rating</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>12%</h3><p>Return Rate (Last 30d)</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>5</h3><p>Open CAPAs</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h3>2</h3><p>Draft Quality Plans</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Launch
    st.subheader("⚡ Active Workflows")
    col1, col2, col3 = st.columns(3)
    if col1.button("🛠️ New Product Plan"):
        st.session_state.nav = "Quality Planning"
        st.rerun()
    if col2.button("📢 Analyze Feedback"):
        st.session_state.nav = "Market Intelligence"
        st.rerun()
    if col3.button("🛡️ Log New CAPA"):
        st.session_state.nav = "CAPA Manager"
        st.rerun()

    # Activity Chart
    st.markdown("### 📈 Quality Event Trends")
    data = pd.DataFrame({
        "Date": pd.date_range(start="2025-01-01", periods=8, freq="W"),
        "Complaints": [12, 15, 8, 20, 10, 5, 12, 8],
        "CAPAs Closed": [2, 1, 4, 0, 3, 2, 1, 3]
    })
    fig = px.line(data, x="Date", y=["Complaints", "CAPAs Closed"], markers=True, color_discrete_sequence=["#e74c3c", "#2ecc71"])
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: QUALITY PLANNING ---
def render_quality_planning():
    st.title("🛠️ Quality Project Planner")
    st.markdown("Define quality strategy from concept to launch. **Lock** sections you write manually; let AI fill the rest.")

    # Data State
    defaults = {
        "qp_name": "", "qp_risk": "Class I", "qp_scope": "", "qp_regs": "", 
        "qp_testing": "", "qp_vendor": "", "qp_path": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # 1. Project Context
    with st.expander("📝 Project Context", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        st.session_state.qp_name = c1.text_input("Project Name/SKU", st.session_state.qp_name)
        st.session_state.qp_risk = c2.selectbox("Risk Level", ["Class I (Low)", "Class II (Med)", "Class III (High)"], index=0)
        mkts = c3.multiselect("Target Markets", ["USA (FDA)", "EU (MDR)", "UK", "Canada"])

    # 2. Planning Sections
    col_edit, col_view = st.columns([1.2, 1])
    
    sections = [
        ("scope", "Scope & Objectives", "Define boundaries, deliverables, and success criteria."),
        ("regs", "Regulatory Strategy", "Applicable standards (ISO 13485, FDA 21 CFR, ASTM)."),
        ("testing", "Testing & Validation Plan", "Mechanical, Chemical, User Testing requirements."),
        ("vendor", "Supply Chain & Vendor Controls", "IQC, AQL levels, Supplier Audits."),
        ("path", "Critical Path & Timeline", "Key milestones: Proto, Tooling, Pilot, Launch.")
    ]
    
    generated_content = {}

    with col_edit:
        st.subheader("Drafting")
        with st.form("qp_form"):
            locks = {}
            for code, title, hint in sections:
                st.markdown(f"**{title}**")
                # Lock Checkbox
                locks[code] = st.checkbox(f"🔒 Lock/Preserve {title}", key=f"lock_{code}", help="Check to keep your text exactly as is. Uncheck to let AI optimize or generate.")
                
                # Text Area
                st.session_state[f"qp_{code}"] = st.text_area(
                    hint, 
                    value=st.session_state[f"qp_{code}"], 
                    height=100,
                    key=f"input_{code}",
                    label_visibility="collapsed"
                )
                st.markdown("---")
            
            gen_btn = st.form_submit_button("✨ Generate / Optimize Plan")

    if gen_btn:
        with st.spinner("AI is analyzing dependencies and filling gaps..."):
            # Build Context
            context_str = f"Project: {st.session_state.qp_name}. Risk: {st.session_state.qp_risk}. Markets: {', '.join(mkts)}.\n"
            
            # Add User Input to Context
            for code, title, _ in sections:
                val = st.session_state[f"qp_{code}"]
                if val:
                    context_str += f"User Input for {title}: {val}\n"

            # Generate
            for code, title, _ in sections:
                current_val = st.session_state[f"qp_{code}"]
                is_locked = locks[code]

                if is_locked:
                    # Keep exact
                    generated_content[code] = current_val
                else:
                    # Generate or Optimize
                    if not current_val:
                        prompt = f"Write a professional '{title}' section for a {st.session_state.qp_risk} medical/consumer device project named {st.session_state.qp_name}. Context: {context_str}. Do NOT use markdown formatting like bold or headers."
                    else:
                        prompt = f"Optimize this text for a Quality Plan (make it professional, clear, and compliant): '{current_val}'. Context: {context_str}. Do NOT use markdown formatting."
                    
                    # Call AI
                    resp = st.session_state.ai.generate(prompt)
                    st.session_state[f"qp_{code}"] = resp
                    generated_content[code] = resp
                    time.sleep(0.5) # Rate limit buffer
            
            st.success("Plan Updated!")
            st.rerun()

    # 3. Preview & Export
    with col_view:
        st.subheader("📄 Live Preview (Clean)")
        
        full_doc = f"PROJECT QUALITY PLAN: {st.session_state.qp_name.upper()}\n"
        full_doc += f"DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
        full_doc += f"RISK LEVEL: {st.session_state.qp_risk} | MARKETS: {', '.join(mkts)}\n\n"
        
        for code, title, _ in sections:
            content = st.session_state.get(f"qp_{code}", "")
            clean_content = clean_text_for_export(content)
            full_doc += f"{title.upper()}\n{'-'*len(title)}\n{clean_content}\n\n"
        
        st.text_area("Final Document", value=full_doc, height=600, disabled=True)
        
        c_ex1, c_ex2 = st.columns(2)
        c_ex1.download_button("📥 Download .txt (Raw)", full_doc, file_name=f"{st.session_state.qp_name}_Plan.txt")
        c_ex2.download_button("📥 Download .md (Markdown)", full_doc, file_name=f"{st.session_state.qp_name}_Plan.md")

# --- MODULE 3: MARKET INTELLIGENCE ---
def render_market_intel():
    st.title("🌐 Market Intelligence & Supply Chain")
    
    tabs = st.tabs(["📢 Amazon VoC & Vision", "📦 Odoo Supply Chain"])

    # --- TAB A: AMAZON VOC ---
    with tabs[0]:
        st.markdown("Analyze Customer Sentiment from Screenshots (Review Dashboards, Return Reports) or Text.")
        
        col_v1, col_v2 = st.columns([1, 1.5])
        
        with col_v1:
            st.subheader("Input")
            upload_src = st.file_uploader("Upload Screenshot or Review Image", type=['png', 'jpg', 'jpeg'])
            
            if upload_src:
                img = Image.open(upload_src)
                st.image(img, caption="Review Source", use_column_width=True)
                
                if st.button("🔍 Analyze Screenshot"):
                    with st.spinner("Vision AI is reading data points..."):
                        prompt = """
                        Analyze this Amazon/Voice of Customer screenshot.
                        1. Extract any visible Star Ratings or NCX rates.
                        2. Summarize the main negative trends or keywords.
                        3. Classify top issues (e.g., "Broken Parts", "Wrong Size", "Shipping Damage").
                        4. Provide a sentiment score (1-10).
                        """
                        res = st.session_state.ai.analyze_image(img, prompt)
                        st.session_state.voc_result = res
                        st.session_state.voc_context = res # Save for chat

        with col_v2:
            st.subheader("Analysis & Chat")
            if 'voc_result' in st.session_state:
                with st.container(height=300, border=True):
                    st.markdown(st.session_state.voc_result)
                
                # Chat Interface
                st.markdown("#### 💬 Ask Questions about this Data")
                user_q = st.text_input("Ex: 'How many users mentioned broken wheels?'")
                if user_q and 'voc_context' in st.session_state:
                    with st.spinner("Thinking..."):
                        chat_prompt = f"Context: {st.session_state.voc_context}. User Question: {user_q}. Answer:"
                        ans = st.session_state.ai.generate(chat_prompt)
                        st.info(ans)
                
                st.divider()
                if st.button("🚨 Create CAPA from this Analysis"):
                    st.session_state.capa_prefill = st.session_state.voc_result
                    st.session_state.nav = "CAPA Manager"
                    st.rerun()
            else:
                st.info("Upload an image to begin analysis.")

    # --- TAB B: ODOO PARSER ---
    with tabs[1]:
        st.markdown("Upload Odoo Exports (Helpdesk Tickets, Inventory Forecast). **Auto-cleans garbage rows.**")
        
        o_file = st.file_uploader("Upload Odoo CSV/XLSX", type=['csv', 'xlsx'])
        
        if o_file:
            df = None
            err = None
            
            if o_file.name.endswith('.csv'):
                df, err = smart_odoo_parser(o_file)
            else:
                try:
                    df = pd.read_excel(o_file)
                except Exception as e:
                    err = str(e)
            
            if err:
                st.error(f"Could not parse file: {err}")
            elif df is not None:
                st.success(f"Successfully parsed {len(df)} rows.")
                
                # Auto-detect file type
                cols = [str(c).lower() for c in df.columns]
                is_helpdesk = any(x in cols for x in ['ticket', 'priority', 'stage', 'subject'])
                is_inventory = any(x in cols for x in ['forecast', 'on hand', 'qty', 'product title'])
                
                st.markdown(f"**Detected Type:** {'Helpdesk Support' if is_helpdesk else 'Inventory Forecast' if is_inventory else 'General Data'}")
                
                with st.expander("🔍 View Raw Data"):
                    st.dataframe(df.head(50))
                
                # Analytics
                c_o1, c_o2 = st.columns(2)
                
                with c_o1:
                    if is_helpdesk:
                        # Ticket Priority Distribution
                        if 'priority' in [c.lower() for c in df.columns]:
                            p_col = next(c for c in df.columns if c.lower() == 'priority')
                            fig = px.pie(df, names=p_col, title="Tickets by Priority", hole=0.4)
                            st.plotly_chart(fig, use_container_width=True)
                    elif is_inventory:
                        # Top Stock Levels
                        if 'on hand' in [c.lower() for c in df.columns] or 'total units' in [c.lower() for c in df.columns]:
                            q_col = next(c for c in df.columns if c.lower() in ['on hand', 'total units'])
                            p_col = next((c for c in df.columns if c.lower() in ['product', 'product title', 'sku']), None)
                            if p_col:
                                top_10 = df.nlargest(10, q_col)
                                fig = px.bar(top_10, x=p_col, y=q_col, title="Top 10 Inventory Levels")
                                st.plotly_chart(fig, use_container_width=True)

                with c_o2:
                    st.markdown("### 🤖 AI Insights")
                    if st.button("Generate Operational Insights"):
                        sample_data = df.head(20).to_string()
                        prompt = f"Analyze this Odoo data sample (Type: {'Helpdesk' if is_helpdesk else 'Inventory'}). Identify anomalies, critical items, or trends. Data: {sample_data}"
                        insight = st.session_state.ai.generate(prompt)
                        st.markdown(insight)

# --- MODULE 4: CAPA MANAGER ---
def render_capa_manager():
    st.title("🛡️ CAPA Management System")
    
    # Prefill from VoC if available
    desc_val = ""
    if 'capa_prefill' in st.session_state:
        st.info("✨ Data pre-filled from Amazon VoC Analysis")
        desc_val = st.session_state.capa_prefill
        # Clear it so it doesn't persist forever
        del st.session_state.capa_prefill

    tabs = st.tabs(["1. Intake & Risk", "2. Root Cause (RCA)", "3. Action Plan", "4. Review & Close"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.text_input("CAPA ID", value=f"CAPA-{int(time.time())}")
        src = c1.selectbox("Source Channel", ["Amazon Returns", "B2B Feedback", "Internal Audit", "Supplier QC"])
        c2.date_input("Date Opened")
        c2.selectbox("Owner", ["Quality Engineer", "Product Manager", "Ops Lead"])
        
        st.text_area("Problem Description / Non-Conformance", value=desc_val, height=150)
        
        st.markdown("### Initial Risk Assessment")
        r1, r2, r3 = st.columns(3)
        r1.selectbox("Severity", ["Minor", "Major", "Critical"])
        r2.selectbox("Occurrence", ["Rare", "Occasional", "Frequent"])
        r3.selectbox("Detection", ["High", "Medium", "Low"])

    with tabs[1]:
        st.subheader("Root Cause Analysis")
        rca_tool = st.radio("Tool", ["5 Whys", "Fishbone (Ishikawa)"], horizontal=True)
        
        if rca_tool == "5 Whys":
            c_why, c_ai = st.columns([2, 1])
            with c_why:
                w1 = st.text_input("1. Why did it happen?")
                w2 = st.text_input("2. Why did that happen?")
                w3 = st.text_input("3. Why is that the case?")
                w4 = st.text_input("4. Why?")
                w5 = st.text_input("5. Why? (Root Cause)")
            with c_ai:
                st.info("💡 Need help?")
                if st.button("Ask AI for Root Cause"):
                    if w1 and w2:
                        sugg = st.session_state.ai.generate(f"Based on these initial whys: '{w1}' -> '{w2}', suggest the likely root causes for a manufacturing/product issue.")
                        st.write(sugg)
                    else:
                        st.warning("Fill in the first 2 Whys.")

    with tabs[2]:
        st.subheader("Correction & Prevention")
        st.checkbox("Correction (Immediate Fix)")
        st.checkbox("Corrective Action (Prevent Recurrence)")
        st.checkbox("Preventive Action (Prevent Occurrence elsewhere)")
        
        st.text_area("Action Plan Details")
        
        st.markdown("#### FMEA Check")
        st.write("Does this action introduce new risks?")
        st.radio("New Risk?", ["No", "Yes - Require new FMEA"])

    with tabs[3]:
        st.subheader("Effectiveness Check")
        st.date_input("Verification Date")
        st.text_area("Evidence of Effectiveness")
        
        if st.button("Close CAPA", type="primary"):
            st.balloons()
            st.success("CAPA Record Closed & Archived.")

# --- MAIN NAVIGATION ---
def main():
    if 'nav' not in st.session_state:
        st.session_state.nav = "Dashboard"

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/polyclinic.png", width=60)
        st.title("PLI System")
        
        menu_opts = ["Dashboard", "Quality Planning", "Market Intelligence", "CAPA Manager"]
        
        # Sync session state with sidebar
        sel = st.radio("Navigation", menu_opts, index=menu_opts.index(st.session_state.nav))
        if sel != st.session_state.nav:
            st.session_state.nav = sel
            st.rerun()
        
        st.markdown("---")
        st.caption("System Status: 🟢 Online")
        if not api_key:
            st.warning("⚠️ AI Key Not Found")

    if st.session_state.nav == "Dashboard":
        render_dashboard()
    elif st.session_state.nav == "Quality Planning":
        render_quality_planning()
    elif st.session_state.nav == "Market Intelligence":
        render_market_intel()
    elif st.session_state.nav == "CAPA Manager":
        render_capa_manager()

if __name__ == "__main__":
    main()
