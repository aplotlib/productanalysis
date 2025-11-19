import streamlit as st
import pandas as pd
import logging
import time
from datetime import datetime

# --- CUSTOM MODULE IMPORTS ---
# We wrap imports in try/except to handle potential missing file issues gracefully
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, ProjectPlanGenerator
    from upload_handler import UploadHandler
    from dashboard import SimpleDashboard
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Critical System Error: Missing required modules. {str(e)}")
    MODULES_LOADED = False

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Vive Health Quality Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
def load_css():
    st.markdown("""
        <style>
        /* Global Fonts & Colors */
        .stApp {
            background-color: #F8FAFC;
        }
        h1, h2, h3 {
            color: #0F172A;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Header Gradient */
        .header-container {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .header-container h1 { color: white !important; margin-bottom: 0.5rem; }
        .header-container p { color: #DBEAFE; font-size: 1.1rem; }

        /* Card Styling */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        /* Buttons */
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .primary-btn button {
            background-color: #2563EB;
            color: white;
        }
        
        /* Recommendation Banners */
        .rec-banner-red {
            background-color: #FEF2F2;
            border-left: 5px solid #EF4444;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
        }
        .rec-banner-green {
            background-color: #F0FDF4;
            border-left: 5px solid #22C55E;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 4px;
            color: #64748B;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #EFF6FF;
            color: #2563EB;
        }
        </style>
    """, unsafe_allow_html=True)

# --- SESSION MANAGEMENT ---
def initialize_session_state():
    defaults = {
        'ai_analyzer': EnhancedAIAnalyzer() if MODULES_LOADED else None,
        'upload_handler': UploadHandler() if MODULES_LOADED else None,
        'dashboard': SimpleDashboard() if MODULES_LOADED else None,
        'qpp_data': None,           # Stores the generated Quality Project Plan
        'review_data': None,        # Stores processed review data
        'qpp_chat_history': [],     # Chat history for QPP mode
        'review_chat_history': [],  # Chat history for Review mode
        'current_mode': 'Quality Command Center',
        'user_api_key': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_status():
    """Robust check for API availability"""
    if st.session_state.ai_analyzer:
        status = st.session_state.ai_analyzer.get_api_status()
        return status.get('available', False)
    return False

# --- MODULE 1: QUALITY COMMAND CENTER ---
def render_quality_command_center():
    st.markdown("""
    <div class="header-container">
        <h1>🚀 Quality Command Center</h1>
        <p>ISO 13485 Aligned Project Execution • Regulatory Strategy • Risk Management</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Screening Gate (Part 0)
    with st.expander("📋 Part 0: Project Screening Gate (Mandatory)", expanded=not st.session_state.qpp_data):
        st.write("Answer the following to determine the regulatory pathway:")
        
        c1, c2 = st.columns(2)
        with c1:
            q1 = st.checkbox("1. Is the device sterile?", help="Requires sterilization validation (ISO 11135/11137)")
            q2 = st.checkbox("2. Class I w/ special controls, Class II, or higher?", help="Requires full Design Controls (21 CFR 820.30)")
            q3 = st.checkbox("3. Active instrument or software-driven?", help="Requires IEC 60601 / IEC 62304")
        with c2:
            q4 = st.checkbox("4. Mobility item (supports weight/movement)?", help="High liability risk")
            q5 = st.checkbox("5. Complex moving parts (e.g., Knee Brace)?", help="Mechanical failure risk")
            q6 = st.checkbox("6. High financial risk project?", help="Significant CAPEX or inventory investment")

        # Logic based on PDF
        is_critical = any([q1, q2, q3, q4, q5, q6])
        
        if is_critical:
            st.markdown("""
            <div class="rec-banner-red">
                <h3>🔴 Pathway: Critical Path (Comprehensive)</h3>
                <p>This project requires the <b>Full 5-Part QPP</b> to meet ISO/FDA regulatory standards.</p>
            </div>
            """, unsafe_allow_html=True)
            default_index = 0
        else:
            st.markdown("""
            <div class="rec-banner-green">
                <h3>🟢 Pathway: Fast Track (Streamlined)</h3>
                <p>This project qualifies for the <b>Streamlined QPP</b> (Charter + Risk Summary).</p>
            </div>
            """, unsafe_allow_html=True)
            default_index = 1

    st.divider()

    # 2. Input & Generation
    col_input, col_output = st.columns([1, 1.5])

    with col_input:
        st.markdown("### 📝 Project Definition")
        with st.form("qpp_gen_form"):
            mode_select = st.radio(
                "Confirm Planning Mode:", 
                ["Critical Path (Comprehensive)", "Fast Track (Streamlined)"],
                index=default_index
            )
            
            p_name = st.text_input("Product Name", placeholder="e.g., Post-Op Shoe V2")
            p_time = st.text_input("Target Timeline", placeholder="e.g., Launch Q3 2025")
            p_goal = st.text_area("Primary Goal / Problem Statement", 
                                 placeholder="e.g., Reduce return rate from 12% to <7% by fixing sizing chart.",
                                 height=120)
            
            submit = st.form_submit_button("✨ Generate Quality Plan", type="primary")

        if submit and p_name and p_goal:
            if not check_api_status():
                st.error("❌ OpenAI API Key missing. Please configure in Sidebar.")
            else:
                internal_mode = "critical" if "Critical" in mode_select else "fast_track"
                
                with st.spinner(f"🤖 Acting as Quality Director... Drafting {internal_mode} plan..."):
                    try:
                        generator = ProjectPlanGenerator(st.session_state.ai_analyzer.api_client)
                        result = generator.generate_plan(p_name, p_goal, p_time, internal_mode)
                        
                        if result['success']:
                            st.session_state.qpp_data = result
                            st.session_state.qpp_chat_history = [{
                                "role": "assistant",
                                "content": f"I have generated the **{mode_select}** for **{p_name}**. Please review the document on the right. I am ready to help you refine specific sections (e.g., 'Expand the FMEA' or 'Add more User Needs')."
                            }]
                            st.toast("Plan Generated Successfully!", icon="✅")
                        else:
                            st.error("Generation failed. Please check API limits.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # 3. Output & Interaction
    with col_output:
        if st.session_state.qpp_data:
            tab_doc, tab_chat = st.tabs(["📄 Plan Document", "💬 AI Assistant"])
            
            with tab_doc:
                st.markdown(f"### {st.session_state.qpp_data['mode'].replace('_', ' ').title()} Plan")
                st.markdown(st.session_state.qpp_data['content'])
                st.download_button(
                    "📥 Download Markdown",
                    st.session_state.qpp_data['content'],
                    file_name=f"QPP_{p_name.replace(' ','_')}.md"
                )
            
            with tab_chat:
                render_chat_interface(
                    history_key='qpp_chat_history',
                    context_prompt=f"You are a Quality Manager assistant. Context: {st.session_state.qpp_data['content']}"
                )
        else:
            st.info("👈 Fill out the Project Definition to generate your Quality Plan.")

# --- MODULE 2: REVIEW ANALYZER ---
def render_review_analyzer():
    st.markdown("""
    <div class="header-container">
        <h1>📊 Review Analyzer</h1>
        <p>Voice of Customer (VoC) Analysis • Sentiment Tracking • Issue Detection</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Upload
    uploaded_file = st.file_uploader("Upload Review Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        if st.session_state.review_data is None or uploaded_file.name != st.session_state.get('current_filename'):
            with st.spinner("Processing Data..."):
                file_bytes = uploaded_file.read()
                result = st.session_state.upload_handler.process_structured_file(file_bytes, uploaded_file.name)
                
                if result['success']:
                    st.session_state.review_data = result
                    st.session_state.current_filename = uploaded_file.name
                    # Initialize chat for this file
                    st.session_state.review_chat_history = [{
                        "role": "assistant",
                        "content": f"I've analyzed {uploaded_file.name}. I found {len(result.get('customer_feedback', {}).values())} products. Ask me about trends, top complaints, or specific ASINs."
                    }]
                else:
                    st.error(f"Upload failed: {result.get('errors')}")

    # 2. Dashboard & Analysis
    if st.session_state.review_data:
        data = st.session_state.review_data
        
        # Run AI Analysis on the data (if not done)
        if 'ai_analysis' not in data and check_api_status():
             with st.spinner("Running AI Classification & CAPA Generation..."):
                 # Extract reviews list from the complex dict structure
                 raw_reviews = []
                 for asin, reviews in data.get('customer_feedback', {}).items():
                     raw_reviews.extend(reviews)
                 
                 # Analyze
                 ai_results = st.session_state.ai_analyzer.analyze_reviews_comprehensive(
                     {'name': 'Uploaded Product', 'category': 'Medical Device'}, 
                     raw_reviews[:50] # Limit for speed/cost in demo
                 )
                 st.session_state.review_data['ai_analysis'] = ai_results

        # Layout
        tab_dash, tab_chat = st.tabs(["📈 Dashboard", "💬 Data Chat"])
        
        with tab_dash:
            st.session_state.dashboard.render_upload_status(data)
            if 'ai_analysis' in data:
                st.session_state.dashboard.render_analysis_overview({data.get('filename'): data['ai_analysis']})
                st.session_state.dashboard.render_capa_recommendations({data.get('filename'): data['ai_analysis']})
                st.session_state.dashboard.render_ai_insights({data.get('filename'): data['ai_analysis']})
        
        with tab_chat:
            context_str = "You are a Data Analyst. User has uploaded review data."
            if 'ai_analysis' in data:
                context_str += f" AI Findings: {str(data['ai_analysis'].get('ai_insights', ''))}"
            
            render_chat_interface(
                history_key='review_chat_history',
                context_prompt=context_str
            )

# --- SHARED CHAT COMPONENT ---
def render_chat_interface(history_key, context_prompt):
    """Reusable chat component with history management"""
    history = st.session_state[history_key]
    
    # Display history
    chat_container = st.container()
    with chat_container:
        for msg in history:
            avatar = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state[history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            if not check_api_status():
                st.error("AI Offline. Check API Key.")
                return

            try:
                # Construct messages for API
                api_messages = [{"role": "system", "content": context_prompt}]
                # Add last 5 turns for context window management
                api_messages.extend([
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state[history_key][-5:]
                ])
                
                # Stream-like UX (simulated as API doesn't support stream=True in wrapper)
                with st.spinner("Thinking..."):
                    response = st.session_state.ai_analyzer.api_client.call_api(api_messages)
                
                if response['success']:
                    full_response = response['result']
                    message_placeholder.markdown(full_response)
                    st.session_state[history_key].append({"role": "assistant", "content": full_response})
                else:
                    st.error("Error getting response from AI.")
            except Exception as e:
                st.error(f"Chat Error: {str(e)}")

# --- MAIN APP LOGIC ---
def main():
    load_css()
    initialize_session_state()

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.title("🏥 Vive Health")
        st.caption("Quality Assurance Platform v4.0")
        
        mode = st.radio("Select Module:", 
            ["Quality Command Center", "Review Analyzer"],
            captions=["Generate QPPs & Strategy", "Analyze Returns & VoC"]
        )
        
        st.divider()
        
        # System Status
        status = check_api_status()
        if status:
            st.success("✅ AI Engine Online")
        else:
            st.error("❌ AI Engine Offline")
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.rerun()
        
        st.info("💡 **Tip:** Use the Screening Gate in the Command Center to determine regulatory requirements.")

    # --- ROUTING ---
    if mode == "Quality Command Center":
        render_quality_command_center()
    elif mode == "Review Analyzer":
        render_review_analyzer()

if __name__ == "__main__":
    main()
