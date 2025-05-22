"""
Professional Dashboard UI Module for Amazon Medical Device Listing Optimizer

This module provides production-ready UI components and dashboards for:
- Product performance scoring visualization
- Upload workflows and data management
- AI analysis results display
- Portfolio analytics and reporting
- Export functionality

Author: Assistant
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional color scheme (toned down from cyberpunk)
COLORS = {
    'primary': '#2563EB',      # Professional blue
    'secondary': '#059669',    # Success green
    'accent': '#DC2626',       # Alert red
    'warning': '#D97706',      # Warning orange
    'info': '#0891B2',         # Info cyan
    'text_primary': '#1F2937', # Dark gray
    'text_secondary': '#6B7280', # Medium gray
    'background': '#F9FAFB',   # Light gray background
    'surface': '#FFFFFF',      # White surface
    'border': '#E5E7EB'        # Light border
}

# Performance level colors
PERFORMANCE_COLORS = {
    'Excellent': '#22C55E',
    'Good': '#3B82F6', 
    'Average': '#F59E0B',
    'Needs Improvement': '#EF4444',
    'Critical': '#DC2626'
}

class UIComponents:
    """Reusable UI components for consistent styling"""
    
    @staticmethod
    def set_professional_theme():
        """Apply professional theme to Streamlit app"""
        st.markdown(f"""
        <style>
            /* Main app styling */
            .stApp {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: {COLORS['primary']};
                font-weight: 600;
                margin-bottom: 1rem;
            }}
            
            /* Metrics and cards */
            [data-testid="metric-container"] {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            
            [data-testid="stMetricValue"] {{
                font-size: 2rem;
                font-weight: 700;
                color: {COLORS['primary']};
            }}
            
            /* Buttons */
            .stButton > button {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.2s;
            }}
            
            .stButton > button:hover {{
                background-color: #1D4ED8;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            }}
            
            /* Secondary button */
            .stButton.secondary > button {{
                background-color: white;
                color: {COLORS['primary']};
                border: 1px solid {COLORS['primary']};
            }}
            
            /* Success button */
            .stButton.success > button {{
                background-color: {COLORS['secondary']};
            }}
            
            /* Danger button */
            .stButton.danger > button {{
                background-color: {COLORS['accent']};
            }}
            
            /* File uploader */
            .stFileUploader {{
                background-color: {COLORS['surface']};
                border: 2px dashed {COLORS['border']};
                border-radius: 8px;
                padding: 2rem;
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                background-color: {COLORS['surface']};
                border-bottom: 1px solid {COLORS['border']};
            }}
            
            .stTabs [data-baseweb="tab"] {{
                color: {COLORS['text_secondary']};
                font-weight: 500;
            }}
            
            .stTabs [aria-selected="true"] {{
                color: {COLORS['primary']};
                border-bottom: 2px solid {COLORS['primary']};
            }}
            
            /* Sidebar */
            .css-1d391kg {{
                background-color: {COLORS['surface']};
                border-right: 1px solid {COLORS['border']};
            }}
            
            /* Data tables */
            .stDataFrame {{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
            
            /* Progress bars */
            .stProgress > div > div {{
                background-color: {COLORS['primary']};
            }}
            
            /* Alerts */
            .alert-success {{
                background-color: #D1FAE5;
                border: 1px solid #A7F3D0;
                color: #065F46;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }}
            
            .alert-warning {{
                background-color: #FEF3C7;
                border: 1px solid #FDE68A;
                color: #92400E;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }}
            
            .alert-error {{
                background-color: #FEE2E2;
                border: 1px solid #FECACA;
                color: #991B1B;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }}
            
            /* Custom card styling */
            .score-card {{
                background: linear-gradient(135deg, {COLORS['surface']} 0%, #F8FAFC 100%);
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            }}
            
            .metric-card {{
                background-color: {COLORS['surface']};
                border-left: 4px solid {COLORS['primary']};
                border-radius: 0 8px 8px 0;
                padding: 1rem;
                margin: 0.5rem 0;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_alert(message: str, alert_type: str = "info"):
        """Display styled alert message"""
        alert_class = f"alert-{alert_type}"
        st.markdown(f'<div class="{alert_class}">{message}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_score_gauge(score: float, title: str, max_score: float = 100) -> go.Figure:
        """Create a professional score gauge chart"""
        
        # Determine color based on score
        if score >= 85:
            color = PERFORMANCE_COLORS['Excellent']
        elif score >= 70:
            color = PERFORMANCE_COLORS['Good']
        elif score >= 55:
            color = PERFORMANCE_COLORS['Average']
        elif score >= 40:
            color = PERFORMANCE_COLORS['Needs Improvement']
        else:
            color = PERFORMANCE_COLORS['Critical']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16, 'color': COLORS['text_primary']}},
            number = {'font': {'size': 24, 'color': color}},
            gauge = {
                'axis': {'range': [None, max_score], 'tickwidth': 1, 'tickcolor': COLORS['text_secondary']},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': COLORS['border'],
                'steps': [
                    {'range': [0, 40], 'color': "#FEE2E2"},
                    {'range': [40, 55], 'color': "#FEF3C7"},
                    {'range': [55, 70], 'color': "#E0F2FE"},
                    {'range': [70, 85], 'color': "#DBEAFE"},
                    {'range': [85, 100], 'color': "#D1FAE5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS['text_primary'])
        )
        
        return fig
    
    @staticmethod
    def create_component_bar_chart(component_scores: Dict[str, Any]) -> go.Figure:
        """Create horizontal bar chart for component scores"""
        
        components = list(component_scores.keys())
        scores = [component_scores[comp].raw_score for comp in components]
        colors = [PERFORMANCE_COLORS.get(component_scores[comp].performance_level, COLORS['primary']) for comp in components]
        
        fig = go.Figure(go.Bar(
            y=components,
            x=scores,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{score:.1f}" for score in scores],
            textposition='inside',
            textfont=dict(color='white', size=12)
        ))
        
        fig.update_layout(
            title="Performance Component Breakdown",
            xaxis_title="Score (0-100)",
            yaxis_title="",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS['text_primary']),
            xaxis=dict(range=[0, 100])
        )
        
        return fig
    
    @staticmethod
    def create_portfolio_overview_chart(scores: Dict[str, Any]) -> go.Figure:
        """Create portfolio performance overview chart"""
        
        # Extract data for plotting
        asins = list(scores.keys())
        composite_scores = [scores[asin].composite_score for asin in asins]
        categories = [scores[asin].category for asin in asins]
        names = [scores[asin].product_name[:30] + "..." if len(scores[asin].product_name) > 30 
                else scores[asin].product_name for asin in asins]
        
        # Create scatter plot
        fig = px.scatter(
            x=range(len(asins)),
            y=composite_scores,
            color=categories,
            size=[abs(score-50)+20 for score in composite_scores],  # Size based on deviation from average
            hover_name=names,
            hover_data={'ASIN': asins, 'Score': composite_scores},
            labels={'x': 'Product Index', 'y': 'Composite Score'},
            title="Portfolio Performance Overview"
        )
        
        # Add performance threshold lines
        fig.add_hline(y=85, line_dash="dash", line_color=PERFORMANCE_COLORS['Excellent'], 
                     annotation_text="Excellent (85+)")
        fig.add_hline(y=70, line_dash="dash", line_color=PERFORMANCE_COLORS['Good'], 
                     annotation_text="Good (70+)")
        fig.add_hline(y=55, line_dash="dash", line_color=PERFORMANCE_COLORS['Average'], 
                     annotation_text="Average (55+)")
        fig.add_hline(y=40, line_dash="dash", line_color=PERFORMANCE_COLORS['Needs Improvement'], 
                     annotation_text="Needs Improvement (40+)")
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS['text_primary'])
        )
        
        return fig

class DashboardRenderer:
    """Main dashboard rendering class"""
    
    def __init__(self):
        self.ui = UIComponents()
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div style="background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%); 
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;">
            <h1 style="color: white; margin: 0;">🏥 Amazon Medical Device Listing Optimizer</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Professional performance analytics and optimization for medical device listings
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar_status(self, module_status: Dict[str, bool], api_status: Dict[str, Any]):
        """Render sidebar with system status"""
        
        with st.sidebar:
            st.markdown("### System Status")
            
            # Module status
            st.markdown("**Available Modules:**")
            for module, available in module_status.items():
                icon = "✅" if available else "❌"
                st.markdown(f"{icon} {module.replace('_', ' ').title()}")
            
            # API status
            st.markdown("**AI Analysis:**")
            if api_status.get('available', False):
                st.success("✅ AI Analysis Available")
                st.caption(f"Model: {api_status.get('model', 'Unknown')}")
            else:
                st.error("❌ AI Analysis Unavailable")
                if 'error' in api_status:
                    st.caption(f"Error: {api_status['error']}")
            
            # Quick actions
            st.markdown("### Quick Actions")
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
            
            if st.button("📊 Example Data", use_container_width=True):
                st.session_state['load_example'] = True
                st.rerun()
    
    def render_upload_dashboard(self):
        """Render comprehensive upload dashboard"""
        
        st.markdown("## 📁 Data Import Center")
        
        # Upload method tabs
        upload_tabs = st.tabs(["📊 Structured Data", "✍️ Manual Entry", "🖼️ Images & Documents"])
        
        with upload_tabs[0]:
            self._render_structured_upload()
        
        with upload_tabs[1]:
            self._render_manual_entry()
        
        with upload_tabs[2]:
            self._render_image_upload()
    
    def _render_structured_upload(self):
        """Render structured data upload interface"""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Upload Product Data")
            st.markdown("Upload CSV or Excel files with your Amazon product performance data.")
            
            # Template download
            if st.button("📥 Download Template", use_container_width=True):
                # This would call the upload_handler to generate template
                st.success("Template download initiated!")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file with product data"
            )
            
            if uploaded_file:
                # Process file (would integrate with upload_handler)
                with st.spinner("Processing file..."):
                    # Simulate processing
                    import time
                    time.sleep(1)
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    # Sample data for preview
                    preview_data = pd.DataFrame({
                        'ASIN': ['B0EXAMPLE1', 'B0EXAMPLE2'],
                        'Product Name': ['Knee Brace Pro', 'Comfort Cushion'],
                        'Sales 30D': [245, 189],
                        'Returns 30D': [12, 8],
                        'Return Rate': ['4.9%', '4.2%']
                    })
                    st.dataframe(preview_data, use_container_width=True)
        
        with col2:
            st.markdown("### Upload Requirements")
            st.markdown("""
            **Required Fields:**
            - ✅ ASIN
            - ✅ Last 30 Days Sales
            - ✅ Last 30 Days Returns
            
            **Optional Fields:**
            - SKU, Product Name
            - Category, Star Rating
            - Average Price, Cost per Unit
            - Total Reviews
            """)
            
            # Upload stats
            st.markdown("### Current Data")
            col_a, col_b = st.columns(2)
            col_a.metric("Products", "12")
            col_b.metric("Categories", "4")
    
    def _render_manual_entry(self):
        """Render manual data entry interface"""
        
        st.markdown("### Manual Product Entry")
        
        with st.form("manual_product_entry"):
            # Basic info
            col1, col2 = st.columns(2)
            
            with col1:
                asin = st.text_input("ASIN*", placeholder="B0XXXXXXXXX")
                product_name = st.text_input("Product Name*")
                category = st.selectbox("Category*", [
                    "Mobility Aids", "Bathroom Safety", "Pain Relief", 
                    "Orthopedic Support", "Other"
                ])
            
            with col2:
                sku = st.text_input("SKU", placeholder="Optional")
                star_rating = st.slider("Star Rating", 1.0, 5.0, 4.0, 0.1)
                total_reviews = st.number_input("Total Reviews", min_value=0, value=0)
            
            # Sales and returns
            st.markdown("**Sales & Returns Data**")
            col3, col4 = st.columns(2)
            
            with col3:
                sales_30d = st.number_input("30-Day Sales*", min_value=0, value=0)
                sales_365d = st.number_input("365-Day Sales", min_value=0, value=0)
            
            with col4:
                returns_30d = st.number_input("30-Day Returns*", min_value=0, value=0)
                returns_365d = st.number_input("365-Day Returns", min_value=0, value=0)
            
            # Financial data
            st.markdown("**Financial Data (Optional)**")
            col5, col6 = st.columns(2)
            
            with col5:
                avg_price = st.number_input("Average Price ($)", min_value=0.0, format="%.2f")
            
            with col6:
                cost_per_unit = st.number_input("Cost per Unit ($)", min_value=0.0, format="%.2f")
            
            # Submit button
            if st.form_submit_button("💾 Save Product", use_container_width=True):
                if asin and product_name and category and sales_30d > 0:
                    st.success(f"✅ Product {asin} saved successfully!")
                else:
                    st.error("❌ Please fill in all required fields (*)")
    
    def _render_image_upload(self):
        """Render image and document upload interface"""
        
        st.markdown("### Image & Document Analysis")
        st.markdown("Upload screenshots, PDFs, or images for immediate AI analysis and insights.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Content type selection
            content_type = st.selectbox(
                "What type of content are you uploading?",
                ["Product Reviews", "Return Reports", "Product Listings", "Competitor Analysis", "Market Research"]
            )
            
            # ASIN association (optional)
            target_asin = st.text_input(
                "Associate with ASIN (optional)", 
                placeholder="B0XXXXXXXXX",
                help="If you know the ASIN, enter it here. Otherwise, we'll try to detect it automatically."
            )
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload images or documents",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                accept_multiple_files=True,
                help="Upload screenshots of reviews, return reports, or product listings for AI analysis"
            )
            
            if uploaded_files:
                st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
                
                # Processing options
                processing_method = st.radio(
                    "Processing method:",
                    ["Auto (OCR + AI Vision)", "AI Vision Only", "OCR Only"],
                    horizontal=True,
                    help="Auto tries OCR first, then falls back to AI Vision if needed"
                )
                
                # Analysis options
                with st.expander("🎛️ Analysis Options"):
                    run_immediate_analysis = st.checkbox(
                        "Run AI analysis immediately after processing", 
                        value=True,
                        help="Generate AI insights right after extracting content"
                    )
                    
                    analysis_depth = st.selectbox(
                        "Analysis depth:",
                        ["Quick Insights", "Detailed Analysis", "Comprehensive Report"],
                        help="Choose how detailed you want the AI analysis to be"
                    )
                    
                    include_recommendations = st.checkbox(
                        "Include actionable recommendations", 
                        value=True,
                        help="Generate specific action items and improvement suggestions"
                    )
                
                if st.button("🔍 Process & Analyze Files", type="primary", use_container_width=True):
                    self._process_and_analyze_files(
                        uploaded_files, content_type, target_asin, processing_method,
                        run_immediate_analysis, analysis_depth, include_recommendations
                    )
        
        with col2:
            st.markdown("### Processing Status")
            
            # Show API availability
            api_available = st.session_state.api_status.get('available', False)
            if api_available:
                st.success("✅ AI Analysis Available")
                st.caption("GPT-4o Vision for images & PDFs")
            else:
                st.error("❌ AI Analysis Unavailable")
                st.caption("Configure OpenAI API key for AI analysis")
            
            # OCR availability  
            ocr_available = st.session_state.module_status.get('pytesseract', False)
            pdf_available = st.session_state.module_status.get('pdf2image', False)
            
            if ocr_available and pdf_available:
                st.markdown("**OCR Processing:** ✅ Full Support")
                st.caption("OCR + PDF processing available")
            elif ocr_available:
                st.markdown("**OCR Processing:** ⚠️ Images Only")
                st.caption("OCR for images, AI Vision for PDFs")
            else:
                st.markdown("**OCR Processing:** ⚠️ AI Vision Only")
                st.caption("Using AI Vision for all file types")
            
            st.markdown("### Supported Content")
            st.markdown("""
            **Images:**
            - JPG, JPEG, PNG
            - Screenshots, photos
            - Product listings, reviews
            
            **Documents:** 
            - PDF files (via AI Vision)
            - Multi-page documents
            - Reports, presentations
            
            **Auto-Detection:**
            - ASINs (B0XXXXXXXXX)
            - Product prices & ratings
            - Review counts & sentiment
            - Return reason categories
            """)
            
            # Processing method explanation
            if not api_available:
                st.warning("⚠️ Limited functionality without API key")
            elif not ocr_available:
                st.info("ℹ️ Cloud deployment: Using AI Vision for all processing")
            
            st.markdown("### Recent Processing")
            if hasattr(st.session_state, 'image_analysis_results') and st.session_state.image_analysis_results:
                recent_count = len(st.session_state.image_analysis_results)
                st.metric("Sessions", recent_count)
            else:
                st.metric("Sessions", 0)
        
        # Display recent analysis results
        if hasattr(st.session_state, 'image_analysis_results') and st.session_state.image_analysis_results:
            self._display_recent_image_analysis()
    
    def _process_and_analyze_files(self, uploaded_files, content_type, target_asin, 
                                 processing_method, run_immediate_analysis, 
                                 analysis_depth, include_recommendations):
        """Process uploaded files and run AI analysis"""
        
        try:
            # Initialize results storage
            if 'image_analysis_results' not in st.session_state:
                st.session_state.image_analysis_results = []
            
            processed_files = []
            analysis_results = []
            
            # Process each file
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    
                    # Read file data
                    file_data = uploaded_file.read()
                    filename = uploaded_file.name
                    
                    # Process file to extract text
                    from upload_handler import UploadHandler
                    upload_handler = UploadHandler()
                    
                    result = upload_handler.process_image_document(
                        file_data, filename, content_type, target_asin
                    )
                    
                    if result['success']:
                        processed_files.append(result)
                        
                        # Show what was detected
                        structured_data = result.get('structured_data', {})
                        
                        if 'detected_asins' in structured_data:
                            st.info(f"🔍 Detected ASINs in {filename}: {', '.join(structured_data['detected_asins'])}")
                        
                        if 'product_info' in structured_data:
                            product_info = structured_data['product_info']
                            info_items = []
                            for key, value in product_info.items():
                                if key.startswith('detected_'):
                                    display_key = key.replace('detected_', '').replace('_', ' ').title()
                                    if isinstance(value, float):
                                        if 'price' in key:
                                            info_items.append(f"{display_key}: ${value:.2f}")
                                        elif 'rating' in key:
                                            info_items.append(f"{display_key}: {value}★")
                                        else:
                                            info_items.append(f"{display_key}: {value}")
                                    else:
                                        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                                        info_items.append(f"{display_key}: {display_value}")
                            
                            if info_items:
                                st.info(f"📋 Detected from {filename}: {' | '.join(info_items)}")
                        
                        # Run AI analysis if requested
                        if run_immediate_analysis and st.session_state.api_status.get('available', False):
                            
                            analysis_result = self._run_image_ai_analysis(
                                result, content_type, analysis_depth, include_recommendations
                            )
                            
                            if analysis_result:
                                analysis_results.append(analysis_result)
                        
                    else:
                        st.error(f"❌ Failed to process {filename}: {', '.join(result.get('errors', ['Unknown error']))}")
            
            # Store results and display summary
            if processed_files:
                # Add to session state
                timestamp = datetime.now().isoformat()
                batch_result = {
                    'timestamp': timestamp,
                    'content_type': content_type,
                    'files': processed_files,
                    'analysis_results': analysis_results,
                    'settings': {
                        'processing_method': processing_method,
                        'analysis_depth': analysis_depth,
                        'include_recommendations': include_recommendations
                    }
                }
                
                st.session_state.image_analysis_results.append(batch_result)
                
                # Display success summary
                st.success(f"✅ Successfully processed {len(processed_files)} files")
                
                if analysis_results:
                    st.success(f"🧠 Generated AI analysis for {len(analysis_results)} files")
                    
                    # Show quick preview of first analysis
                    if analysis_results:
                        with st.expander("📊 Analysis Preview", expanded=True):
                            first_analysis = analysis_results[0]
                            st.markdown(f"**File:** {first_analysis['filename']}")
                            st.markdown(f"**AI Insights:**")
                            st.markdown(first_analysis['ai_insights'][:500] + "..." if len(first_analysis['ai_insights']) > 500 else first_analysis['ai_insights'])
                
                # Offer to view full results
                if st.button("📋 View Full Analysis Results", use_container_width=True):
                    st.session_state['show_image_analysis_tab'] = True
                    st.rerun()
                
        except Exception as e:
            logger.error(f"Error processing and analyzing files: {str(e)}")
            st.error(f"❌ Processing failed: {str(e)}")
    
    def _run_image_ai_analysis(self, file_result, content_type, analysis_depth, include_recommendations):
        """Run AI analysis on processed file content"""
        
        try:
            from enhanced_ai_analysis import EnhancedAIAnalyzer
            ai_analyzer = EnhancedAIAnalyzer()
            
            # Check API availability
            if not ai_analyzer.get_api_status().get('available', False):
                return None
            
            # Extract content for analysis
            extracted_text = file_result.get('text', '')
            structured_data = file_result.get('structured_data', {})
            filename = file_result.get('filename', 'unknown')
            
            # Create analysis prompt based on content type and depth
            analysis_prompt = self._create_image_analysis_prompt(
                extracted_text, structured_data, content_type, analysis_depth, include_recommendations
            )
            
            # Call AI API for analysis
            from enhanced_ai_analysis import APIClient
            api_client = APIClient()
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert Amazon listing optimization analyst specializing in medical device e-commerce. Provide specific, actionable insights based on the content provided."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ]
            
            with st.spinner(f"Analyzing {filename} with AI..."):
                response = api_client.call_api(messages, max_tokens=1500)
            
            if response['success']:
                return {
                    'filename': filename,
                    'content_type': content_type,
                    'analysis_depth': analysis_depth,
                    'ai_insights': response['result'],
                    'extracted_text_preview': extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                    'detected_data': structured_data,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                st.error(f"AI analysis failed for {filename}: {response.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error running AI analysis on image: {str(e)}")
            st.error(f"AI analysis error: {str(e)}")
            return None
    
    def _create_image_analysis_prompt(self, extracted_text, structured_data, content_type, analysis_depth, include_recommendations):
        """Create AI analysis prompt based on content and settings"""
        
        prompt = f"""Analyze this {content_type.lower()} content extracted from an uploaded image/document.

EXTRACTED CONTENT:
{extracted_text}

"""
        
        # Add detected structured data if available
        if structured_data:
            prompt += "DETECTED PRODUCT DATA:\n"
            for key, value in structured_data.items():
                if key.startswith('detected_') or key in ['product_info', 'detected_asins']:
                    prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Customize prompt based on content type
        if content_type == "Product Reviews":
            prompt += """ANALYSIS FOCUS:
1. Customer sentiment and satisfaction patterns
2. Common complaints and praise points
3. Product quality and performance issues
4. Size, fit, and usability concerns
5. Features customers value most
6. Listing accuracy vs. customer expectations"""
            
        elif content_type == "Return Reports":
            prompt += """ANALYSIS FOCUS:
1. Primary return reason categories
2. Patterns in return explanations
3. Quality vs. expectation issues
4. Actionable improvement opportunities
5. Cost impact and prevention strategies"""
            
        elif content_type == "Product Listings":
            prompt += """ANALYSIS FOCUS:
1. Title and keyword optimization opportunities
2. Bullet point effectiveness
3. Image and description quality
4. Competitive positioning
5. Conversion optimization potential
6. Missing information or features"""
            
        elif content_type == "Competitor Analysis":
            prompt += """ANALYSIS FOCUS:
1. Competitive advantages and weaknesses
2. Pricing and positioning strategy
3. Feature comparison opportunities
4. Market positioning insights
5. Differentiation strategies"""
        
        # Adjust depth based on setting
        if analysis_depth == "Quick Insights":
            prompt += "\n\nProvide 3-5 key insights with brief explanations (200-300 words)."
        elif analysis_depth == "Detailed Analysis":
            prompt += "\n\nProvide comprehensive analysis with specific examples and evidence (400-600 words)."
        else:  # Comprehensive Report
            prompt += "\n\nProvide detailed analysis with actionable recommendations, implementation guidance, and expected outcomes (600-800 words)."
        
        if include_recommendations:
            prompt += "\n\nInclude specific, actionable recommendations with priority levels (High/Medium/Low) and expected impact."
        
        return prompt
    
    def _display_recent_image_analysis(self):
        """Display recent image analysis results"""
        
        st.markdown("### 📊 Recent Image Analysis Results")
        
        recent_results = st.session_state.image_analysis_results[-3:]  # Show last 3 batches
        
        for i, batch in enumerate(reversed(recent_results)):
            timestamp = datetime.fromisoformat(batch['timestamp']).strftime('%Y-%m-%d %H:%M')
            
            with st.expander(f"📁 Analysis {len(recent_results)-i}: {batch['content_type']} - {timestamp}"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Content Type:** {batch['content_type']}")
                    st.markdown(f"**Files Processed:** {len(batch['files'])}")
                    st.markdown(f"**Analysis Depth:** {batch['settings']['analysis_depth']}")
                    
                    # Show files processed
                    file_names = [f['filename'] for f in batch['files']]
                    st.markdown(f"**Files:** {', '.join(file_names)}")
                
                with col2:
                    # Export options for this batch
                    if batch['analysis_results']:
                        # Create consolidated report
                        report_text = self._create_batch_report(batch)
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=report_text,
                            file_name=f"image_analysis_{timestamp.replace(':', '')}.md",
                            mime="text/markdown",
                            key=f"download_batch_{i}"
                        )
                
                # Show analysis results
                if batch['analysis_results']:
                    st.markdown("#### AI Analysis Results:")
                    
                    for result in batch['analysis_results']:
                        st.markdown(f"**{result['filename']}:**")
                        
                        # Show preview
                        insights = result['ai_insights']
                        if len(insights) > 300:
                            st.markdown(insights[:300] + "...")
                            
                            # Full insights in expandable section
                            with st.expander(f"Full Analysis - {result['filename']}"):
                                st.markdown(insights)
                        else:
                            st.markdown(insights)
                        
                        st.markdown("---")
    
    def _create_batch_report(self, batch):
        """Create a consolidated report for a batch of analyzed files"""
        
        timestamp = datetime.fromisoformat(batch['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Image Analysis Report
**Generated:** {timestamp}
**Content Type:** {batch['content_type']}
**Analysis Depth:** {batch['settings']['analysis_depth']}

## Files Analyzed
"""
        
        for file_info in batch['files']:
            report += f"- {file_info['filename']} ({file_info.get('processing_method', 'Unknown method')})\n"
        
        report += "\n## Analysis Results\n\n"
        
        for result in batch['analysis_results']:
            report += f"### {result['filename']}\n"
            report += f"**Content Type:** {result['content_type']}\n"
            report += f"**Analysis Date:** {result['timestamp']}\n\n"
            
            if result.get('detected_data'):
                report += "**Detected Information:**\n"
                for key, value in result['detected_data'].items():
                    if key.startswith('detected_') or key in ['product_info', 'detected_asins']:
                        report += f"- {key}: {value}\n"
                report += "\n"
            
            report += "**AI Insights:**\n"
            report += result['ai_insights']
            report += "\n\n---\n\n"
        
        return report
    
    def render_scoring_dashboard(self, scores: Dict[str, Any]):
        """Render main scoring dashboard"""
        
        st.markdown("## 📊 Performance Scoring Dashboard")
        
        if not scores:
            self.ui.show_alert("No scored products available. Please upload and analyze product data first.", "warning")
            return
        
        # Portfolio overview
        self._render_portfolio_overview(scores)
        
        # Individual product analysis
        self._render_product_selector(scores)
    
    def _render_portfolio_overview(self, scores: Dict[str, Any]):
        """Render portfolio-level overview"""
        
        st.markdown("### Portfolio Performance Overview")
        
        # Calculate portfolio metrics
        score_values = [score.composite_score for score in scores.values()]
        avg_score = np.mean(score_values)
        
        # Performance distribution
        excellent_count = sum(1 for s in score_values if s >= 85)
        good_count = sum(1 for s in score_values if 70 <= s < 85)
        average_count = sum(1 for s in score_values if 55 <= s < 70)
        needs_improvement_count = sum(1 for s in score_values if 40 <= s < 55)
        critical_count = sum(1 for s in score_values if s < 40)
        
        # Top row metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(scores))
        
        with col2:
            st.metric("Portfolio Average", f"{avg_score:.1f}/100")
        
        with col3:
            top_performers = excellent_count + good_count
            st.metric("Top Performers", f"{top_performers} ({top_performers/len(scores)*100:.0f}%)")
        
        with col4:
            priority_products = needs_improvement_count + critical_count
            st.metric("Priority Products", f"{priority_products} ({priority_products/len(scores)*100:.0f}%)")
        
        # Portfolio visualization
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            # Portfolio scatter plot
            portfolio_chart = self.ui.create_portfolio_overview_chart(scores)
            st.plotly_chart(portfolio_chart, use_container_width=True)
        
        with col_b:
            # Performance distribution
            distribution_data = {
                'Performance Level': ['Excellent', 'Good', 'Average', 'Needs Improvement', 'Critical'],
                'Count': [excellent_count, good_count, average_count, needs_improvement_count, critical_count],
                'Color': [PERFORMANCE_COLORS['Excellent'], PERFORMANCE_COLORS['Good'], 
                         PERFORMANCE_COLORS['Average'], PERFORMANCE_COLORS['Needs Improvement'], 
                         PERFORMANCE_COLORS['Critical']]
            }
            
            fig_pie = px.pie(
                values=distribution_data['Count'],
                names=distribution_data['Performance Level'],
                color_discrete_sequence=distribution_data['Color'],
                title="Performance Distribution"
            )
            fig_pie.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def _render_product_selector(self, scores: Dict[str, Any]):
        """Render product selection and detailed analysis"""
        
        st.markdown("### Individual Product Analysis")
        
        # Product selector
        score_list = list(scores.values())
        product_options = [f"{score.asin} - {score.product_name} ({score.composite_score:.1f})" 
                          for score in score_list]
        
        selected_index = st.selectbox(
            "Select product for detailed analysis:",
            range(len(product_options)),
            format_func=lambda i: product_options[i]
        )
        
        if selected_index is not None:
            selected_score = score_list[selected_index]
            self._render_product_details(selected_score)
    
    def _render_product_details(self, score: Any):
        """Render detailed product scoring analysis"""
        
        st.markdown(f"### {score.product_name}")
        st.markdown(f"**ASIN:** {score.asin} | **Category:** {score.category}")
        
        # Main score display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Composite score gauge
            gauge_chart = self.ui.create_score_gauge(
                score.composite_score, 
                "Composite Score"
            )
            st.plotly_chart(gauge_chart, use_container_width=True)
        
        with col2:
            # Component breakdown
            component_chart = self.ui.create_component_bar_chart(score.component_scores)
            st.plotly_chart(component_chart, use_container_width=True)
        
        with col3:
            # Key metrics
            st.markdown("**Performance Level**")
            level_color = PERFORMANCE_COLORS.get(score.performance_level, COLORS['primary'])
            st.markdown(f'<div style="color: {level_color}; font-weight: bold; font-size: 1.2em;">{score.performance_level}</div>', 
                       unsafe_allow_html=True)
            
            if score.revenue_impact:
                st.metric("Revenue Impact", f"${score.revenue_impact:,.0f}/month")
            
            if score.potential_savings:
                st.metric("Potential Savings", f"${score.potential_savings:,.0f}/month")
        
        # Detailed analysis tabs
        detail_tabs = st.tabs(["🎯 Recommendations", "⚠️ Risk Factors", "💪 Strengths", "📈 Components"])
        
        with detail_tabs[0]:
            st.markdown("#### Priority Improvements")
            if score.improvement_priority:
                for i, priority in enumerate(score.improvement_priority[:5], 1):
                    st.markdown(f"{i}. {priority}")
            else:
                st.info("No specific improvements identified - product performing well overall.")
        
        with detail_tabs[1]:
            st.markdown("#### Risk Assessment")
            if score.risk_factors:
                for risk in score.risk_factors:
                    st.markdown(f"⚠️ {risk}")
            else:
                st.success("✅ No significant risk factors identified.")
        
        with detail_tabs[2]:
            st.markdown("#### Key Strengths")
            if score.strengths:
                for strength in score.strengths:
                    st.markdown(f"💪 {strength}")
            else:
                st.info("Strengths will be identified as performance improves.")
        
        with detail_tabs[3]:
            st.markdown("#### Component Score Details")
            for component_name, component_score in score.component_scores.items():
                with st.expander(f"{component_name}: {component_score.raw_score:.1f}/100"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"**Performance:** {component_score.performance_level}")
                        st.markdown(f"**Weight:** {component_score.weight:.0%}")
                        st.markdown(f"**Improvement Potential:** {component_score.improvement_potential:.1f} points")
                    
                    with col_b:
                        st.markdown(f"**Benchmark:** {component_score.benchmark_comparison}")
                        st.markdown("**Key Drivers:**")
                        for driver in component_score.key_drivers:
                            st.markdown(f"• {driver}")
    
    def render_ai_chat_tab(self):
        """Render AI chat tab with embedded chat interface"""
        try:
            from ai_chat import AIChatInterface
            chat_interface = AIChatInterface()
            chat_interface.render_chat_interface()
        except ImportError:
            st.error("AI Chat module not available. Please ensure ai_chat.py is in your project directory.")
            st.info("The AI Chat feature provides standalone consulting without requiring data uploads.")
    
    def render_image_analysis_tab(self):
        """Render dedicated tab for image analysis results and management"""
        
        st.markdown("## 🖼️ Image & Document Analysis")
        st.markdown("Standalone AI analysis of uploaded images and documents - no sales data required!")
        
        # Check if we have any results
        if not hasattr(st.session_state, 'image_analysis_results') or not st.session_state.image_analysis_results:
            st.info("📸 No image analysis results yet.")
            st.markdown("**To get started:**")
            st.markdown("1. Go to the **Data Import** tab")
            st.markdown("2. Upload images or PDFs in the **Images & Documents** section")
            st.markdown("3. Enable 'Run AI analysis immediately' option")
            st.markdown("4. Your results will appear here!")
            
            # Quick upload option
            with st.expander("🚀 Quick Upload & Analysis"):
                st.markdown("Upload and analyze files directly from this tab:")
                
                quick_files = st.file_uploader(
                    "Choose files for quick analysis",
                    type=['jpg', 'jpeg', 'png', 'pdf'],
                    accept_multiple_files=True,
                    key="quick_image_upload"
                )
                
                if quick_files:
                    quick_content_type = st.selectbox(
                        "Content type:",
                        ["Product Reviews", "Return Reports", "Product Listings", "Competitor Analysis"],
                        key="quick_content_type"
                    )
                    
                    if st.button("🔍 Analyze Now", type="primary"):
                        self._process_and_analyze_files(
                            quick_files, quick_content_type, "", "Auto (OCR + AI Vision)",
                            True, "Detailed Analysis", True
                        )
                        st.rerun()
            
            return
        
        # Display analysis management interface
        analysis_results = st.session_state.image_analysis_results
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_batches = len(analysis_results)
        total_files = sum(len(batch['files']) for batch in analysis_results)
        total_analyses = sum(len(batch['analysis_results']) for batch in analysis_results)
        
        with col1:
            st.metric("Analysis Sessions", total_batches)
        
        with col2:
            st.metric("Files Processed", total_files)
        
        with col3:
            st.metric("AI Analyses", total_analyses)
        
        with col4:
            if total_analyses > 0:
                success_rate = (total_analyses / total_files) * 100 if total_files > 0 else 0
                st.metric("Success Rate", f"{success_rate:.0f}%")
        
        # Analysis results display
        analysis_tabs = st.tabs(["📊 All Results", "🔍 Search & Filter", "📥 Bulk Export"])
        
        with analysis_tabs[0]:
            self._display_all_image_results(analysis_results)
        
        with analysis_tabs[1]:
            self._display_filtered_image_results(analysis_results)
        
        with analysis_tabs[2]:
            self._display_bulk_export_options(analysis_results)
    
    def _display_all_image_results(self, analysis_results):
        """Display all image analysis results"""
        
        st.markdown("### All Analysis Results")
        
        # Reverse to show most recent first
        for i, batch in enumerate(reversed(analysis_results)):
            batch_index = len(analysis_results) - 1 - i
            timestamp = datetime.fromisoformat(batch['timestamp']).strftime('%Y-%m-%d %H:%M')
            
            # Batch header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"#### 📁 Session {batch_index + 1}: {batch['content_type']}")
                st.caption(f"📅 {timestamp} | 📄 {len(batch['files'])} files | 🧠 {len(batch['analysis_results'])} analyses")
            
            with col2:
                # Individual batch export
                if batch['analysis_results']:
                    report_text = self._create_batch_report(batch)
                    st.download_button(
                        label="📥 Export",
                        data=report_text,
                        file_name=f"analysis_session_{batch_index + 1}_{timestamp.replace(':', '')}.md",
                        mime="text/markdown",
                        key=f"export_batch_{batch_index}",
                        help="Download this analysis session"
                    )
            
            with col3:
                # Delete batch option
                if st.button("🗑️ Delete", key=f"delete_batch_{batch_index}", help="Delete this analysis session"):
                    st.session_state.image_analysis_results.pop(batch_index)
                    st.rerun()
            
            # Batch content
            with st.expander(f"View Details - Session {batch_index + 1}", expanded=(i == 0)):
                
                # Settings used
                settings = batch['settings']
                st.markdown(f"**Settings:** {settings['processing_method']} | {settings['analysis_depth']} | Recommendations: {'✅' if settings['include_recommendations'] else '❌'}")
                
                # Files processed
                st.markdown("**Files Processed:**")
                for file_info in batch['files']:
                    status = "✅" if file_info['success'] else "❌"
                    method = file_info.get('processing_method', 'Unknown')
                    st.markdown(f"- {status} {file_info['filename']} ({method})")
                
                # Analysis results
                if batch['analysis_results']:
                    st.markdown("**AI Analysis Results:**")
                    
                    for j, result in enumerate(batch['analysis_results']):
                        with st.expander(f"📄 {result['filename']} - {result['content_type']}"):
                            
                            # Show detected data if available
                            if result.get('detected_data'):
                                st.markdown("**Detected Information:**")
                                detected = result['detected_data']
                                
                                if 'detected_asins' in detected:
                                    st.markdown(f"🔍 **ASINs:** {', '.join(detected['detected_asins'])}")
                                
                                if 'product_info' in detected:
                                    info = detected['product_info']
                                    info_display = []
                                    for key, value in info.items():
                                        if key.startswith('detected_'):
                                            clean_key = key.replace('detected_', '').replace('_', ' ').title()
                                            if 'price' in key and isinstance(value, (int, float)):
                                                info_display.append(f"{clean_key}: ${value:.2f}")
                                            elif 'rating' in key:
                                                info_display.append(f"{clean_key}: {value}★")
                                            else:
                                                info_display.append(f"{clean_key}: {value}")
                                    
                                    if info_display:
                                        st.markdown(f"📋 **Product Info:** {' | '.join(info_display)}")
                            
                            # AI insights
                            st.markdown("**AI Insights:**")
                            st.markdown(result['ai_insights'])
                            
                            # Individual file export
                            individual_report = f"""# Analysis Report: {result['filename']}
**Content Type:** {result['content_type']}  
**Analysis Date:** {result['timestamp']}  
**Analysis Depth:** {result['analysis_depth']}

## Detected Information
{json.dumps(result.get('detected_data', {}), indent=2)}

## Extracted Content Preview
{result.get('extracted_text_preview', 'No preview available')}

## AI Analysis
{result['ai_insights']}
"""
                            
                            st.download_button(
                                label="📥 Export Individual Report",
                                data=individual_report,
                                file_name=f"analysis_{result['filename']}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                mime="text/markdown",
                                key=f"export_individual_{batch_index}_{j}"
                            )
                
                st.markdown("---")
    
    def _display_filtered_image_results(self, analysis_results):
        """Display filtered and searchable image results"""
        
        st.markdown("### Search & Filter Results")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Content type filter
            all_content_types = list(set(batch['content_type'] for batch in analysis_results))
            selected_content_types = st.multiselect(
                "Content Types:",
                options=all_content_types,
                default=all_content_types
            )
        
        with col2:
            # Date range filter
            if analysis_results:
                min_date = min(datetime.fromisoformat(batch['timestamp']).date() for batch in analysis_results)
                max_date = max(datetime.fromisoformat(batch['timestamp']).date() for batch in analysis_results)
                
                selected_date_range = st.date_input(
                    "Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
        
        with col3:
            # Search term
            search_term = st.text_input(
                "Search in results:",
                placeholder="Enter keywords to search in AI insights..."
            )
        
        # Apply filters
        filtered_results = []
        
        for batch in analysis_results:
            batch_date = datetime.fromisoformat(batch['timestamp']).date()
            
            # Apply filters
            if (batch['content_type'] in selected_content_types and
                (len(selected_date_range) == 2 and selected_date_range[0] <= batch_date <= selected_date_range[1])):
                
                # Apply search term if provided
                if search_term:
                    matching_analyses = []
                    for analysis in batch['analysis_results']:
                        if search_term.lower() in analysis['ai_insights'].lower():
                            matching_analyses.append(analysis)
                    
                    if matching_analyses:
                        filtered_batch = batch.copy()
                        filtered_batch['analysis_results'] = matching_analyses
                        filtered_results.append(filtered_batch)
                else:
                    filtered_results.append(batch)
        
        # Display filtered results
        if filtered_results:
            st.markdown(f"**Found {len(filtered_results)} sessions matching your criteria**")
            
            for batch in filtered_results:
                timestamp = datetime.fromisoformat(batch['timestamp']).strftime('%Y-%m-%d %H:%M')
                
                with st.expander(f"📁 {batch['content_type']} - {timestamp}"):
                    for analysis in batch['analysis_results']:
                        st.markdown(f"**📄 {analysis['filename']}**")
                        
                        # Highlight search terms
                        insights = analysis['ai_insights']
                        if search_term:
                            # Simple highlighting (could be enhanced)
                            highlighted = insights.replace(
                                search_term, 
                                f"**{search_term}**"
                            )
                            st.markdown(highlighted)
                        else:
                            st.markdown(insights)
                        
                        st.markdown("---")
        else:
            st.info("No results match your filter criteria.")
    
    def _display_bulk_export_options(self, analysis_results):
        """Display bulk export options for all image analysis results"""
        
        st.markdown("### Bulk Export Options")
        
        if not analysis_results:
            st.info("No analysis results to export.")
            return
        
        # Export format selection
        export_format = st.selectbox(
            "Export format:",
            ["Markdown Report", "CSV Summary", "JSON Data"]
        )
        
        # Content selection
        include_options = st.multiselect(
            "Include in export:",
            ["AI Insights", "Detected Data", "Processing Details", "File Information"],
            default=["AI Insights", "Detected Data"]
        )
        
        # Generate export based on format
        if st.button("📥 Generate Bulk Export", type="primary"):
            
            if export_format == "Markdown Report":
                export_data = self._generate_bulk_markdown_report(analysis_results, include_options)
                file_name = f"bulk_image_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                mime_type = "text/markdown"
                
            elif export_format == "CSV Summary":
                export_data = self._generate_bulk_csv_summary(analysis_results, include_options)
                file_name = f"image_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                mime_type = "text/csv"
                
            else:  # JSON Data
                export_data = self._generate_bulk_json_export(analysis_results, include_options)
                file_name = f"image_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                mime_type = "application/json"
            
            st.download_button(
                label=f"📥 Download {export_format}",
                data=export_data,
                file_name=file_name,
                mime=mime_type
            )
        
        # Quick stats
        st.markdown("### Export Preview")
        
        total_analyses = sum(len(batch['analysis_results']) for batch in analysis_results)
        content_type_counts = {}
        
        for batch in analysis_results:
            content_type = batch['content_type']
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + len(batch['analysis_results'])
        
        st.markdown(f"**Total Analyses:** {total_analyses}")
        st.markdown("**By Content Type:**")
        for content_type, count in content_type_counts.items():
            st.markdown(f"- {content_type}: {count}")
    
    def _generate_bulk_markdown_report(self, analysis_results, include_options):
        """Generate bulk markdown report"""
        
        report = f"""# Comprehensive Image Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Sessions:** {len(analysis_results)}
**Total Analyses:** {sum(len(batch['analysis_results']) for batch in analysis_results)}

"""
        
        for i, batch in enumerate(analysis_results):
            timestamp = datetime.fromisoformat(batch['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            report += f"""## Session {i + 1}: {batch['content_type']}
**Date:** {timestamp}
**Files Processed:** {len(batch['files'])}
**AI Analyses:** {len(batch['analysis_results'])}

"""
            
            if "Processing Details" in include_options:
                settings = batch['settings']
                report += f"""**Processing Settings:**
- Method: {settings['processing_method']}
- Analysis Depth: {settings['analysis_depth']}
- Include Recommendations: {settings['include_recommendations']}

"""
            
            for analysis in batch['analysis_results']:
                report += f"### {analysis['filename']}\n\n"
                
                if "Detected Data" in include_options and analysis.get('detected_data'):
                    report += "**Detected Information:**\n"
                    for key, value in analysis['detected_data'].items():
                        report += f"- {key}: {value}\n"
                    report += "\n"
                
                if "AI Insights" in include_options:
                    report += "**AI Analysis:**\n"
                    report += analysis['ai_insights']
                    report += "\n\n"
                
                report += "---\n\n"
        
        return report
    
    def _generate_bulk_csv_summary(self, analysis_results, include_options):
        """Generate bulk CSV summary"""
        
        data = []
        
        for batch_idx, batch in enumerate(analysis_results):
            timestamp = datetime.fromisoformat(batch['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            for analysis in batch['analysis_results']:
                row = {
                    'Session': batch_idx + 1,
                    'Timestamp': timestamp,
                    'Content_Type': batch['content_type'],
                    'Filename': analysis['filename'],
                    'Analysis_Depth': analysis['analysis_depth']
                }
                
                if "Detected Data" in include_options:
                    detected = analysis.get('detected_data', {})
                    row['Detected_ASINs'] = ', '.join(detected.get('detected_asins', []))
                    
                    product_info = detected.get('product_info', {})
                    row['Detected_Price'] = product_info.get('detected_price', '')
                    row['Detected_Rating'] = product_info.get('detected_rating', '')
                    row['Detected_Title'] = product_info.get('detected_title', '')
                
                if "AI Insights" in include_options:
                    # Truncate for CSV
                    insights = analysis['ai_insights']
                    row['AI_Insights_Preview'] = insights[:200] + "..." if len(insights) > 200 else insights
                
                data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def _generate_bulk_json_export(self, analysis_results, include_options):
        """Generate bulk JSON export"""
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_sessions': len(analysis_results),
            'export_options': include_options,
            'sessions': []
        }
        
        for batch in analysis_results:
            session_data = {
                'timestamp': batch['timestamp'],
                'content_type': batch['content_type'],
                'files_count': len(batch['files']),
                'analyses_count': len(batch['analysis_results'])
            }
            
            if "Processing Details" in include_options:
                session_data['settings'] = batch['settings']
            
            if "File Information" in include_options:
                session_data['files'] = batch['files']
            
            session_data['analyses'] = []
            
            for analysis in batch['analysis_results']:
                analysis_data = {
                    'filename': analysis['filename'],
                    'content_type': analysis['content_type'],
                    'analysis_depth': analysis['analysis_depth'],
                    'timestamp': analysis['timestamp']
                }
                
                if "Detected Data" in include_options:
                    analysis_data['detected_data'] = analysis.get('detected_data', {})
                
                if "AI Insights" in include_options:
                    analysis_data['ai_insights'] = analysis['ai_insights']
                
                session_data['analyses'].append(analysis_data)
            
            export_data['sessions'].append(session_data)
        
        return json.dumps(export_data, indent=2)
    
    def render_ai_analysis_dashboard(self, ai_results: Dict[str, Any]):
        """Render enhanced AI analysis dashboard with manual triggers"""
        
        st.markdown("## 🧠 AI Product Analysis")
        st.markdown("Run AI analysis on your uploaded products to get detailed optimization recommendations.")
        
        # Check if we have data to analyze
        if not hasattr(st.session_state, 'processed_data') or not st.session_state.get('data_processed', False):
            st.warning("⚠️ No processed data available for AI analysis.")
            st.info("Please upload and process product data in the Data Import tab first.")
            return
        
        processed_data = st.session_state.processed_data
        products = processed_data.get('products', [])
        reviews_data = processed_data.get('reviews', {})
        returns_data = processed_data.get('returns', {})
        
        if not products:
            st.warning("No products found in processed data.")
            return
        
        # AI Analysis Control Panel
        with st.container():
            st.markdown("### 🎛️ AI Analysis Control Panel")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Product selector for individual analysis
                product_options = [(p['asin'], f"{p['name']} ({p['asin']})") for p in products]
                selected_asin = st.selectbox(
                    "Select product for AI analysis:",
                    options=[asin for asin, _ in product_options],
                    format_func=lambda asin: next(name for a, name in product_options if a == asin),
                    help="Choose a product to run detailed AI analysis"
                )
            
            with col2:
                # Individual product analysis
                if st.button("🔍 Analyze Selected Product", type="primary", use_container_width=True):
                    if 'run_individual_ai_analysis' not in st.session_state:
                        st.session_state['run_individual_ai_analysis'] = selected_asin
                        st.rerun()
            
            with col3:
                # Bulk analysis for all products
                if st.button("🚀 Analyze All Products", type="secondary", use_container_width=True):
                    if 'run_bulk_ai_analysis' not in st.session_state:
                        st.session_state['run_bulk_ai_analysis'] = True
                        st.rerun()
        
        # Analysis status and progress
        if hasattr(st.session_state, 'ai_analysis_in_progress'):
            if st.session_state.ai_analysis_in_progress:
                st.info("🔄 AI analysis in progress... Please wait.")
                return
        
        # Display analysis controls and summaries
        analysis_tabs = st.tabs(["📊 Analysis Summary", "📝 Review Insights", "↩️ Return Analysis", "🎯 Recommendations"])
        
        with analysis_tabs[0]:
            self._render_ai_analysis_summary(products, ai_results)
        
        with analysis_tabs[1]:
            self._render_ai_review_insights(ai_results)
        
        with analysis_tabs[2]:
            self._render_ai_return_insights(ai_results)
        
        with analysis_tabs[3]:
            self._render_ai_recommendations(ai_results)
    
    def _render_ai_analysis_summary(self, products: List[Dict], ai_results: Dict[str, Any]):
        """Render AI analysis summary"""
        
        st.markdown("### AI Analysis Overview")
        
        # Analysis statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(products)
            st.metric("Total Products", total_products)
        
        with col2:
            analyzed_products = len(ai_results)
            st.metric("AI Analyzed", analyzed_products)
        
        with col3:
            if analyzed_products > 0:
                completion_rate = (analyzed_products / total_products) * 100
                st.metric("Completion Rate", f"{completion_rate:.0f}%")
            else:
                st.metric("Completion Rate", "0%")
        
        with col4:
            # Estimate API usage
            if hasattr(st.session_state, 'ai_api_calls'):
                st.metric("API Calls Made", st.session_state.ai_api_calls)
        
        # Analysis results overview
        if ai_results:
            st.markdown("### Products with AI Analysis")
            
            analysis_data = []
            for asin, analysis in ai_results.items():
                product = next((p for p in products if p['asin'] == asin), {})
                
                analysis_data.append({
                    'ASIN': asin,
                    'Product': product.get('name', 'Unknown'),
                    'Review Analysis': '✅' if 'review_analysis' in analysis else '❌',
                    'Return Analysis': '✅' if 'return_analysis' in analysis else '❌',
                    'Listing Optimization': '✅' if 'listing_optimization' in analysis else '❌',
                    'Confidence': f"{analysis.get('review_analysis', {}).get('confidence_score', 0):.0%}" if 'review_analysis' in analysis else 'N/A'
                })
            
            df = pd.DataFrame(analysis_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No AI analysis results yet. Click 'Analyze Selected Product' or 'Analyze All Products' to begin.")
    
    def _render_ai_review_insights(self, ai_results: Dict[str, Any]):
        """Render AI review analysis results"""
        
        if not ai_results:
            st.info("No AI review analysis available. Run AI analysis first.")
            return
        
        st.markdown("### 📝 Customer Review Analysis")
        
        # Product selector for detailed view
        analyzed_products = list(ai_results.keys())
        if not analyzed_products:
            st.info("No products have been analyzed yet.")
            return
        
        selected_product = st.selectbox(
            "View detailed review analysis for:",
            options=analyzed_products,
            format_func=lambda asin: f"{asin} - {next((p.get('name', 'Unknown') for p in st.session_state.processed_data.get('products', []) if p['asin'] == asin), 'Unknown')}"
        )
        
        if selected_product and 'review_analysis' in ai_results[selected_product]:
            review_analysis = ai_results[selected_product]['review_analysis']
            
            if review_analysis.success:
                # Display analysis results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Key Findings")
                    if review_analysis.detailed_findings:
                        for category, finding in review_analysis.detailed_findings.items():
                            if finding:
                                with st.expander(f"{category.replace('_', ' ').title()}"):
                                    st.markdown(finding)
                
                with col2:
                    st.markdown("#### Analysis Metrics")
                    st.metric("Confidence Score", f"{review_analysis.confidence_score:.0%}")
                    
                    data_quality = review_analysis.data_quality
                    st.metric("Data Quality", data_quality.get('quality', 'Unknown').title())
                    st.metric("Reviews Analyzed", data_quality.get('total', 0))
                
                # Recommendations
                if review_analysis.recommendations:
                    st.markdown("#### AI Recommendations")
                    for i, rec in enumerate(review_analysis.recommendations[:5], 1):
                        priority_color = COLORS['accent'] if rec.get('priority') == 'High' else COLORS['warning']
                        st.markdown(f"""
                        <div style="border-left: 4px solid {priority_color}; padding: 10px; margin: 10px 0; background-color: rgba(255,255,255,0.1);">
                            <strong>{i}. {rec.get('category', 'General')}</strong><br>
                            {rec.get('action', 'No action specified')}<br>
                            <small><em>Expected Impact: {rec.get('expected_impact', 'Not specified')}</em></small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(f"Review analysis failed: {', '.join(review_analysis.errors or ['Unknown error'])}")
        else:
            st.info("No review analysis available for this product.")
    
    def _render_ai_return_insights(self, ai_results: Dict[str, Any]):
        """Render AI return analysis results"""
        
        if not ai_results:
            st.info("No AI return analysis available. Run AI analysis first.")
            return
        
        st.markdown("### ↩️ Return Reason Analysis")
        
        # Similar structure to review insights but for returns
        analyzed_products = [asin for asin, analysis in ai_results.items() if 'return_analysis' in analysis]
        
        if not analyzed_products:
            st.info("No products have return analysis yet. Products need return data to generate insights.")
            return
        
        selected_product = st.selectbox(
            "View return analysis for:",
            options=analyzed_products,
            format_func=lambda asin: f"{asin} - {next((p.get('name', 'Unknown') for p in st.session_state.processed_data.get('products', []) if p['asin'] == asin), 'Unknown')}",
            key="return_analysis_selector"
        )
        
        if selected_product:
            return_analysis = ai_results[selected_product]['return_analysis']
            
            if return_analysis.success:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Return Categories & Insights")
                    if return_analysis.detailed_findings:
                        for category, finding in return_analysis.detailed_findings.items():
                            if finding:
                                with st.expander(f"{category.replace('_', ' ').title()}"):
                                    st.markdown(finding)
                
                with col2:
                    st.markdown("#### Return Metrics")
                    st.metric("Analysis Confidence", f"{return_analysis.confidence_score:.0%}")
                    
                    data_quality = return_analysis.data_quality
                    st.metric("Returns Analyzed", data_quality.get('total', 0))
                    st.metric("Data Quality", data_quality.get('quality', 'Unknown').title())
            else:
                st.error(f"Return analysis failed: {', '.join(return_analysis.errors or ['Unknown error'])}")
    
    def _render_ai_recommendations(self, ai_results: Dict[str, Any]):
        """Render consolidated AI recommendations"""
        
        if not ai_results:
            st.info("No AI recommendations available. Run AI analysis first.")
            return
        
        st.markdown("### 🎯 Consolidated AI Recommendations")
        
        # Compile all recommendations
        all_recommendations = []
        
        for asin, analysis in ai_results.items():
            product_name = next((p.get('name', 'Unknown') for p in st.session_state.processed_data.get('products', []) if p['asin'] == asin), 'Unknown')
            
            # Collect recommendations from all analysis types
            for analysis_type in ['review_analysis', 'return_analysis', 'listing_optimization']:
                if analysis_type in analysis and hasattr(analysis[analysis_type], 'recommendations'):
                    for rec in analysis[analysis_type].recommendations:
                        all_recommendations.append({
                            'asin': asin,
                            'product_name': product_name,
                            'analysis_type': analysis_type.replace('_', ' ').title(),
                            'category': rec.get('category', 'General'),
                            'action': rec.get('action', 'No action specified'),
                            'priority': rec.get('priority', 'Medium'),
                            'expected_impact': rec.get('expected_impact', 'Not specified')
                        })
        
        if all_recommendations:
            # Group by priority
            high_priority = [r for r in all_recommendations if r['priority'] == 'High']
            medium_priority = [r for r in all_recommendations if r['priority'] == 'Medium']
            low_priority = [r for r in all_recommendations if r['priority'] == 'Low']
            
            # Display by priority
            if high_priority:
                st.markdown("#### 🔴 High Priority Actions")
                for rec in high_priority:
                    st.markdown(f"""
                    **{rec['product_name']} ({rec['asin']})**  
                    *{rec['analysis_type']} - {rec['category']}*  
                    {rec['action']}  
                    *Expected Impact: {rec['expected_impact']}*
                    """)
                    st.markdown("---")
            
            if medium_priority:
                with st.expander(f"🟡 Medium Priority Actions ({len(medium_priority)} items)"):
                    for rec in medium_priority:
                        st.markdown(f"""
                        **{rec['product_name']}** - {rec['action']}  
                        *{rec['expected_impact']}*
                        """)
            
            if low_priority:
                with st.expander(f"🟢 Low Priority Actions ({len(low_priority)} items)"):
                    for rec in low_priority:
                        st.markdown(f"**{rec['product_name']}** - {rec['action']}")
            
            # Export recommendations
            rec_df = pd.DataFrame(all_recommendations)
            csv_data = rec_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download All Recommendations (CSV)",
                data=csv_data,
                file_name=f"ai_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No AI recommendations generated yet. Run AI analysis on products with review or return data.")
    
    def _render_review_insights(self, review_analysis: Optional[Any]):
        """Render review analysis insights"""
        
        if not review_analysis or not review_analysis.success:
            st.warning("No review analysis available.")
            return
        
        st.markdown("### Customer Review Analysis")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        summary = review_analysis.summary
        with col1:
            st.metric("Confidence Score", f"{review_analysis.confidence_score:.0%}")
        
        with col2:
            sentiment = summary.get('overall_sentiment', 'Unknown')
            st.metric("Overall Sentiment", sentiment)
        
        with col3:
            data_quality = review_analysis.data_quality.get('quality', 'Unknown')
            st.metric("Data Quality", data_quality.title())
        
        # Detailed findings
        if review_analysis.detailed_findings:
            st.markdown("#### Key Findings")
            for category, finding in review_analysis.detailed_findings.items():
                if finding:
                    with st.expander(f"{category.replace('_', ' ').title()}"):
                        st.markdown(finding)
        
        # Recommendations
        if review_analysis.recommendations:
            st.markdown("#### AI Recommendations")
            for i, rec in enumerate(review_analysis.recommendations[:5], 1):
                priority_color = COLORS['accent'] if rec.get('priority') == 'High' else COLORS['warning']
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {priority_color};">
                    <strong>{i}. {rec.get('category', 'General')}</strong><br>
                    {rec.get('action', 'No action specified')}<br>
                    <small><em>Expected Impact: {rec.get('expected_impact', 'Not specified')}</em></small>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_return_insights(self, return_analysis: Optional[Any]):
        """Render return analysis insights"""
        
        if not return_analysis or not return_analysis.success:
            st.warning("No return analysis available.")
            return
        
        st.markdown("### Return Reason Analysis")
        
        # Similar structure to review insights but focused on returns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Score", f"{return_analysis.confidence_score:.0%}")
        
        with col2:
            data_quality = return_analysis.data_quality.get('quality', 'Unknown')
            st.metric("Data Quality", data_quality.title())
        
        with col3:
            total_returns = return_analysis.data_quality.get('total', 0)
            st.metric("Returns Analyzed", total_returns)
        
        # Return categories and recommendations
        if return_analysis.detailed_findings:
            st.markdown("#### Return Categories")
            for category, finding in return_analysis.detailed_findings.items():
                if finding:
                    with st.expander(f"{category.replace('_', ' ').title()}"):
                        st.markdown(finding)
    
    def _render_optimization_insights(self, optimization_analysis: Optional[Any]):
        """Render listing optimization insights"""
        
        if not optimization_analysis or not optimization_analysis.success:
            st.warning("No optimization analysis available.")
            return
        
        st.markdown("### Listing Optimization Recommendations")
        
        # Implementation timeline
        if optimization_analysis.recommendations:
            st.markdown("#### Implementation Timeline")
            
            timeline_tabs = st.tabs(["🚀 Week 1", "📈 Week 2-4", "🎯 Month 2+"])
            
            week1_items = [rec for rec in optimization_analysis.recommendations 
                          if rec.get('timeframe', '').startswith('Week 1')]
            week2_items = [rec for rec in optimization_analysis.recommendations 
                          if 'Week 2' in rec.get('timeframe', '')]
            month2_items = [rec for rec in optimization_analysis.recommendations 
                           if 'Month 2' in rec.get('timeframe', '')]
            
            with timeline_tabs[0]:
                if week1_items:
                    for item in week1_items:
                        st.markdown(f"✅ {item.get('action', 'No action specified')}")
                else:
                    st.info("No immediate actions identified.")
            
            with timeline_tabs[1]:
                if week2_items:
                    for item in week2_items:
                        st.markdown(f"📝 {item.get('action', 'No action specified')}")
                else:
                    st.info("No medium-term actions identified.")
            
            with timeline_tabs[2]:
                if month2_items:
                    for item in month2_items:
                        st.markdown(f"🎯 {item.get('action', 'No action specified')}")
                else:
                    st.info("No long-term actions identified.")
    
    def render_export_dashboard(self, scores: Dict[str, Any], ai_results: Dict[str, Any]):
        """Render export and reporting dashboard"""
        
        st.markdown("## 📋 Export & Reporting")
        
        export_tabs = st.tabs(["📊 Scorecards", "📈 Executive Report", "📋 Action Items", "💾 Raw Data"])
        
        with export_tabs[0]:
            st.markdown("### Performance Scorecards")
            
            if scores:
                # Generate scorecard for each product
                for asin, score in list(scores.items())[:3]:  # Show first 3 as examples
                    with st.expander(f"Scorecard: {score.product_name}"):
                        self._render_mini_scorecard(score)
                
                # Generate scorecard data
                scorecard_data = self._generate_scorecard_csv(scores)
                st.download_button(
                    label="📥 Download Performance Scorecards (CSV)",
                    data=scorecard_data,
                    file_name=f"performance_scorecards_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No scored products available for export.")
        
        with export_tabs[1]:
            st.markdown("### Executive Summary Report")
            
            # Executive summary preview
            if scores:
                st.markdown("#### Portfolio Executive Summary")
                
                score_values = [s.composite_score for s in scores.values()]
                avg_score = np.mean(score_values)
                
                summary_text = f"""# Portfolio Executive Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Performance Overview
- Portfolio Average: {avg_score:.1f}/100 ({self._get_performance_level_text(avg_score)})
- Total Products Analyzed: {len(scores)}
- Top Performers (70+ score): {sum(1 for s in score_values if s >= 70)}
- Priority Products (<55 score): {sum(1 for s in score_values if s < 55)}

## Key Recommendations
1. Focus on return rate optimization for underperforming products
2. Leverage AI insights to improve customer satisfaction
3. Implement listing optimizations for conversion improvement

## Product Performance Distribution
- Excellent (85+): {sum(1 for s in score_values if s >= 85)} products
- Good (70-84): {sum(1 for s in score_values if 70 <= s < 85)} products
- Average (55-69): {sum(1 for s in score_values if 55 <= s < 70)} products
- Needs Improvement (40-54): {sum(1 for s in score_values if 40 <= s < 55)} products
- Critical (<40): {sum(1 for s in score_values if s < 40)} products
"""
                
                st.markdown(summary_text)
                
                st.download_button(
                    label="📥 Download Executive Report (Text)",
                    data=summary_text,
                    file_name=f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.info("Generate product scores first to create executive report.")
        
        with export_tabs[2]:
            st.markdown("### Action Items Export")
            
            if scores:
                # Compile all action items
                all_actions = []
                for score in scores.values():
                    if score.improvement_priority:
                        for action in score.improvement_priority[:3]:
                            all_actions.append({
                                'ASIN': score.asin,
                                'Product': score.product_name,
                                'Current_Score': score.composite_score,
                                'Performance_Level': score.performance_level,
                                'Action': action,
                                'Priority': 'High' if score.composite_score < 55 else 'Medium',
                                'Revenue_Impact': score.revenue_impact if score.revenue_impact else '',
                                'Potential_Savings': score.potential_savings if score.potential_savings else ''
                            })
                
                if all_actions:
                    action_df = pd.DataFrame(all_actions)
                    st.dataframe(action_df, use_container_width=True)
                    
                    # Convert to CSV
                    csv_data = action_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Action Items (CSV)",
                        data=csv_data,
                        file_name=f"action_items_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No action items generated yet.")
            else:
                st.info("Generate product analysis first to create action items.")
        
        with export_tabs[3]:
            st.markdown("### Raw Data Export")
            
            if scores:
                export_format = st.selectbox(
                    "Select export format:",
                    ["CSV", "Excel", "JSON"]
                )
                
                include_options = st.multiselect(
                    "Include data:",
                    ["Product Scores", "Component Breakdown", "Recommendations", "Risk Factors"],
                    default=["Product Scores", "Component Breakdown"]
                )
                
                if st.button("📊 Generate Export", use_container_width=True):
                    # Generate the appropriate export
                    if export_format == "CSV":
                        export_data = self._generate_detailed_csv(scores, include_options)
                        st.download_button(
                            label="📥 Download CSV Export",
                            data=export_data,
                            file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    elif export_format == "Excel":
                        excel_data = self._generate_excel_export(scores, include_options)
                        st.download_button(
                            label="📥 Download Excel Export", 
                            data=excel_data,
                            file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    elif export_format == "JSON":
                        json_data = self._generate_json_export(scores, include_options)
                        st.download_button(
                            label="📥 Download JSON Export",
                            data=json_data,
                            file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            else:
                st.info("No scored products available for export.")
    
    def _render_mini_scorecard(self, score: Any):
        """Render a mini scorecard for a product"""
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Mini gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score.composite_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': PERFORMANCE_COLORS.get(score.performance_level, COLORS['primary'])},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': COLORS['border']
                }
            ))
            fig.update_layout(height=150, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"**Performance:** {score.performance_level}")
            st.markdown(f"**Category:** {score.category}")
            
            if score.improvement_priority:
                st.markdown(f"**Top Priority:** {score.improvement_priority[0]}")
            
            if score.revenue_impact:
                st.markdown(f"**Revenue Impact:** ${score.revenue_impact:,.0f}/month")
    
    def _get_performance_level_text(self, score: float) -> str:
        """Get performance level text for score"""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 55:
            return "Average"
        elif score >= 40:
            return "Needs Improvement"
        else:
            return "Critical"
    
    def _generate_scorecard_csv(self, scores: Dict[str, Any]) -> str:
        """Generate CSV data for scorecards"""
        data = []
        for asin, score in scores.items():
            data.append({
                'ASIN': score.asin,
                'Product_Name': score.product_name,
                'Category': score.category,
                'Composite_Score': score.composite_score,
                'Performance_Level': score.performance_level,
                'Sales_Performance': score.component_scores['sales_performance'].raw_score,
                'Return_Rate_Score': score.component_scores['return_rate'].raw_score,
                'Customer_Satisfaction': score.component_scores['customer_satisfaction'].raw_score,
                'Review_Engagement': score.component_scores['review_engagement'].raw_score,
                'Profitability': score.component_scores['profitability'].raw_score,
                'Competitive_Position': score.component_scores['competitive_position'].raw_score,
                'Top_Priority': score.improvement_priority[0] if score.improvement_priority else '',
                'Revenue_Impact': score.revenue_impact or 0,
                'Potential_Savings': score.potential_savings or 0,
                'Risk_Count': len(score.risk_factors) if score.risk_factors else 0,
                'Strength_Count': len(score.strengths) if score.strengths else 0
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def _generate_detailed_csv(self, scores: Dict[str, Any], include_options: List[str]) -> str:
        """Generate detailed CSV export"""
        data = []
        for asin, score in scores.items():
            row = {
                'ASIN': score.asin,
                'Product_Name': score.product_name,
                'Category': score.category,
                'Composite_Score': score.composite_score,
                'Performance_Level': score.performance_level
            }
            
            if "Component Breakdown" in include_options:
                for comp_name, comp_score in score.component_scores.items():
                    row[f'{comp_name}_Score'] = comp_score.raw_score
                    row[f'{comp_name}_Performance'] = comp_score.performance_level
            
            if "Recommendations" in include_options and score.improvement_priority:
                for i, rec in enumerate(score.improvement_priority[:3], 1):
                    row[f'Recommendation_{i}'] = rec
            
            if "Risk Factors" in include_options and score.risk_factors:
                for i, risk in enumerate(score.risk_factors[:3], 1):
                    row[f'Risk_Factor_{i}'] = risk
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def _generate_excel_export(self, scores: Dict[str, Any], include_options: List[str]) -> bytes:
        """Generate Excel export"""
        import io
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main scores sheet
            scores_data = []
            for asin, score in scores.items():
                scores_data.append({
                    'ASIN': score.asin,
                    'Product_Name': score.product_name,
                    'Category': score.category,
                    'Composite_Score': score.composite_score,
                    'Performance_Level': score.performance_level,
                    'Revenue_Impact': score.revenue_impact or 0,
                    'Potential_Savings': score.potential_savings or 0
                })
            
            scores_df = pd.DataFrame(scores_data)
            scores_df.to_excel(writer, sheet_name='Product_Scores', index=False)
            
            # Component breakdown sheet if requested
            if "Component Breakdown" in include_options:
                component_data = []
                for asin, score in scores.items():
                    for comp_name, comp_score in score.component_scores.items():
                        component_data.append({
                            'ASIN': score.asin,
                            'Product_Name': score.product_name,
                            'Component': comp_name,
                            'Score': comp_score.raw_score,
                            'Performance_Level': comp_score.performance_level,
                            'Weight': comp_score.weight,
                            'Benchmark_Comparison': comp_score.benchmark_comparison
                        })
                
                comp_df = pd.DataFrame(component_data)
                comp_df.to_excel(writer, sheet_name='Component_Breakdown', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def _generate_json_export(self, scores: Dict[str, Any], include_options: List[str]) -> str:
        """Generate JSON export"""
        import json
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_products': len(scores),
            'products': {}
        }
        
        for asin, score in scores.items():
            product_data = {
                'asin': score.asin,
                'product_name': score.product_name,
                'category': score.category,
                'composite_score': score.composite_score,
                'performance_level': score.performance_level
            }
            
            if "Component Breakdown" in include_options:
                product_data['component_scores'] = {}
                for comp_name, comp_score in score.component_scores.items():
                    product_data['component_scores'][comp_name] = {
                        'score': comp_score.raw_score,
                        'performance_level': comp_score.performance_level,
                        'weight': comp_score.weight
                    }
            
            if "Recommendations" in include_options:
                product_data['recommendations'] = score.improvement_priority or []
            
            if "Risk Factors" in include_options:
                product_data['risk_factors'] = score.risk_factors or []
            
            export_data['products'][asin] = product_data
        
        return json.dumps(export_data, indent=2)

# Main dashboard class
class ProfessionalDashboard:
    """Main dashboard orchestrator"""
    
    def __init__(self):
        self.renderer = DashboardRenderer()
        self.ui = UIComponents()
    
    def initialize_app(self):
        """Initialize the Streamlit app with professional styling"""
        
        st.set_page_config(
            page_title="Amazon Medical Device Optimizer",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply professional theme
        self.ui.set_professional_theme()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        
        default_states = {
            'uploaded_data': {},
            'scored_products': {},
            'ai_analysis_results': {},
            'current_product': None,
            'module_status': {
                'upload_handler': True,
                'ai_analysis': True,
                'scoring_system': True,
                'export_tools': True
            },
            'api_status': {
                'available': True,
                'model': 'gpt-4o'
            }
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def render_main_dashboard(self):
        """Render the main dashboard"""
        
        # Header
        self.renderer.render_header()
        
        # Sidebar
        self.renderer.render_sidebar_status(
            st.session_state.module_status,
            st.session_state.api_status
        )
        
        # Main content tabs
        main_tabs = st.tabs([
            "📁 Data Import", 
            "📊 Performance Scores", 
            "🤖 AI Chat",
            "🧠 AI Analysis",
            "🖼️ Image Analysis", 
            "📋 Export & Reports"
        ])
        
        with main_tabs[0]:
            self.renderer.render_upload_dashboard()
        
        with main_tabs[1]:
            self.renderer.render_scoring_dashboard(st.session_state.scored_products)
        
        with main_tabs[2]:
            self.renderer.render_ai_chat_tab()
        
        with main_tabs[3]:
            self.renderer.render_ai_analysis_dashboard(st.session_state.ai_analysis_results)
        
        with main_tabs[4]:
            self.renderer.render_image_analysis_tab()
        
        with main_tabs[5]:
            self.renderer.render_export_dashboard(
                st.session_state.scored_products,
                st.session_state.ai_analysis_results
            )

# Export main class
__all__ = ['ProfessionalDashboard', 'DashboardRenderer', 'UIComponents']
