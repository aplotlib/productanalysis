"""
Professional Dashboard UI Module - FIXED VERSION
Production-ready UI components and dashboards with proper error handling
Author: Assistant
Version: 2.0 - All Issues Fixed
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports
DEPENDENCIES_AVAILABLE = {}

def safe_import_plotly():
    """Safely import plotly with fallback"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        DEPENDENCIES_AVAILABLE['plotly'] = True
        return go, px, make_subplots, True
    except ImportError:
        logger.warning("Plotly not available - charts will be disabled")
        DEPENDENCIES_AVAILABLE['plotly'] = False
        return None, None, None, False

def safe_import_requests():
    """Safely import requests"""
    try:
        import requests
        DEPENDENCIES_AVAILABLE['requests'] = True
        return requests, True
    except ImportError:
        logger.warning("Requests not available")
        DEPENDENCIES_AVAILABLE['requests'] = False
        return None, False

# Import dependencies
go, px, make_subplots, PLOTLY_AVAILABLE = safe_import_plotly()
requests, REQUESTS_AVAILABLE = safe_import_requests()

# Professional color scheme
COLORS = {
    'primary': '#2563EB',
    'secondary': '#059669',
    'accent': '#DC2626',
    'warning': '#D97706',
    'info': '#0891B2',
    'text_primary': '#1F2937',
    'text_secondary': '#6B7280',
    'background': '#F9FAFB',
    'surface': '#FFFFFF',
    'border': '#E5E7EB'
}

PERFORMANCE_COLORS = {
    'Excellent': '#22C55E',
    'Good': '#3B82F6', 
    'Average': '#F59E0B',
    'Needs Improvement': '#EF4444',
    'Critical': '#DC2626'
}

class UIComponents:
    """Reusable UI components with error handling"""
    
    @staticmethod
    def set_professional_theme():
        """Apply professional theme to Streamlit app"""
        try:
            st.markdown(f"""
            <style>
                .stApp {{
                    background-color: {COLORS['background']};
                    color: {COLORS['text_primary']};
                }}
                
                h1, h2, h3, h4, h5, h6 {{
                    color: {COLORS['primary']};
                    font-weight: 600;
                    margin-bottom: 1rem;
                }}
                
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
                
                .stButton > button {{
                    background-color: {COLORS['primary']};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 0.5rem 1rem;
                    font-weight: 500;
                    transition: all 0.2s;
                }}
                
                .stFileUploader {{
                    background-color: {COLORS['surface']};
                    border: 2px dashed {COLORS['border']};
                    border-radius: 8px;
                    padding: 2rem;
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
        except Exception as e:
            logger.error(f"Error applying theme: {str(e)}")
    
    @staticmethod
    def show_alert(message: str, alert_type: str = "info"):
        """Display styled alert message"""
        try:
            if alert_type == "success":
                st.success(message)
            elif alert_type == "warning":
                st.warning(message)
            elif alert_type == "error":
                st.error(message)
            else:
                st.info(message)
        except Exception as e:
            logger.error(f"Error showing alert: {str(e)}")
            st.write(message)
    
    @staticmethod
    def create_score_gauge(score: float, title: str, max_score: float = 100):
        """Create a professional score gauge chart"""
        if not PLOTLY_AVAILABLE:
            # Fallback to simple metric display
            st.metric(title, f"{score:.1f}/{max_score}")
            return None
        
        try:
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
                mode = "gauge+number",
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
                    ]
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
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {str(e)}")
            st.metric(title, f"{score:.1f}/{max_score}")
            return None
    
    @staticmethod
    def create_safe_bar_chart(data: Dict[str, float], title: str = "Performance Breakdown"):
        """Create bar chart with fallback"""
        if not PLOTLY_AVAILABLE or not data:
            # Fallback to simple display
            st.subheader(title)
            for key, value in data.items():
                st.write(f"**{key}:** {value:.1f}")
            return None
        
        try:
            components = list(data.keys())
            scores = list(data.values())
            
            fig = go.Figure(go.Bar(
                y=components,
                x=scores,
                orientation='h',
                marker=dict(color=COLORS['primary']),
                text=[f"{score:.1f}" for score in scores],
                textposition='inside',
                textfont=dict(color='white', size=12)
            ))
            
            fig.update_layout(
                title=title,
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
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            st.subheader(title)
            for key, value in data.items():
                st.write(f"**{key}:** {value:.1f}")
            return None

class DashboardRenderer:
    """Main dashboard rendering class with comprehensive error handling"""
    
    def __init__(self):
        self.ui = UIComponents()
    
    def render_header(self):
        """Render application header"""
        try:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%); 
                        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;">
                <h1 style="color: white; margin: 0;">🏥 Amazon Medical Device Listing Optimizer</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Professional performance analytics and optimization for medical device listings
                </p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering header: {str(e)}")
            st.title("🏥 Amazon Medical Device Listing Optimizer")
            st.caption("Professional performance analytics and optimization for medical device listings")
    
    def render_sidebar_status(self, module_status: Dict[str, bool], api_status: Dict[str, Any]):
        """Render sidebar with system status"""
        try:
            with st.sidebar:
                st.markdown("### System Status")
                
                # Module status
                st.markdown("**Available Modules:**")
                for module, available in module_status.items():
                    icon = "✅" if available else "❌"
                    module_name = module.replace('_', ' ').title()
                    st.markdown(f"{icon} {module_name}")
                
                # API status
                st.markdown("**AI Analysis:**")
                if api_status.get('available', False):
                    st.success("✅ AI Analysis Available")
                    if 'model' in api_status:
                        st.caption(f"Model: {api_status['model']}")
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
                    
        except Exception as e:
            logger.error(f"Error rendering sidebar: {str(e)}")
    
    def render_upload_dashboard(self):
        """Render comprehensive upload dashboard"""
        try:
            st.markdown("## 📁 Data Import Center")
            
            # Upload method tabs
            upload_tabs = st.tabs(["📊 Structured Data", "✍️ Manual Entry", "🖼️ Images & Documents"])
            
            with upload_tabs[0]:
                self._render_structured_upload()
            
            with upload_tabs[1]:
                self._render_manual_entry()
            
            with upload_tabs[2]:
                self._render_image_upload()
                
        except Exception as e:
            logger.error(f"Error rendering upload dashboard: {str(e)}")
            st.error("Upload dashboard error - using basic upload")
            self._render_basic_upload()
    
    def _render_basic_upload(self):
        """Basic upload fallback"""
        st.markdown("### Basic File Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Get the application controller to handle upload
            if hasattr(st.session_state, 'app_controller'):
                file_data = uploaded_file.read()
                success = st.session_state.app_controller.handle_data_upload(
                    'structured_file', (file_data, uploaded_file.name)
                )
                if success:
                    st.success("✅ File processed successfully")
    
    def _render_structured_upload(self):
        """Render structured data upload interface"""
        try:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Upload Product Data")
                st.markdown("Upload CSV or Excel files with your Amazon product performance data.")
                
                # Template download
                if st.button("📥 Download Template", use_container_width=True):
                    try:
                        # Import upload handler to generate template
                        from upload_handler import UploadHandler
                        upload_handler = UploadHandler()
                        template_data = upload_handler.create_template()
                        
                        st.download_button(
                            label="📥 Download Excel Template",
                            data=template_data,
                            file_name=f"product_data_template_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        logger.error(f"Template generation error: {str(e)}")
                        st.error("Template generation failed - upload handler not available")
                
                # File upload
                uploaded_file = st.file_uploader(
                    "Choose file",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload CSV or Excel file with product data"
                )
                
                if uploaded_file:
                    # Get app controller from session state
                    if 'app_controller' in st.session_state:
                        try:
                            file_data = uploaded_file.read()
                            success = st.session_state.app_controller.handle_data_upload(
                                'structured_file', (file_data, uploaded_file.name)
                            )
                            
                            if success:
                                with st.expander("📋 Data Preview"):
                                    if 'structured_data' in st.session_state.uploaded_data:
                                        df = st.session_state.uploaded_data['structured_data']
                                        st.dataframe(df.head(), use_container_width=True)
                                        
                        except Exception as e:
                            logger.error(f"File processing error: {str(e)}")
                            st.error(f"File processing failed: {str(e)}")
                    else:
                        st.error("Application controller not available")
            
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
                if 'processed_data' in st.session_state and st.session_state.processed_data:
                    products = st.session_state.processed_data.get('products', [])
                    categories = set(p.get('category', 'Other') for p in products)
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Products", len(products))
                    col_b.metric("Categories", len(categories))
                else:
                    col_a, col_b = st.columns(2)
                    col_a.metric("Products", "0")
                    col_b.metric("Categories", "0")
                    
        except Exception as e:
            logger.error(f"Error rendering structured upload: {str(e)}")
            self._render_basic_upload()
    
    def _render_manual_entry(self):
        """Render manual data entry interface"""
        try:
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
                    if asin and product_name and category and sales_30d >= 0:
                        # Prepare data for submission
                        manual_data = {
                            'asin': asin,
                            'product_name': product_name,
                            'category': category,
                            'sku': sku,
                            'sales_30d': sales_30d,
                            'returns_30d': returns_30d,
                            'sales_365d': sales_365d if sales_365d > 0 else None,
                            'returns_365d': returns_365d if returns_365d > 0 else None,
                            'star_rating': star_rating,
                            'total_reviews': total_reviews if total_reviews > 0 else None,
                            'average_price': avg_price if avg_price > 0 else None,
                            'cost_per_unit': cost_per_unit if cost_per_unit > 0 else None
                        }
                        
                        # Submit to app controller
                        if 'app_controller' in st.session_state:
                            try:
                                success = st.session_state.app_controller.handle_data_upload(
                                    'manual_entry', manual_data
                                )
                                if success:
                                    st.rerun()
                            except Exception as e:
                                logger.error(f"Manual entry error: {str(e)}")
                                st.error(f"Save failed: {str(e)}")
                        else:
                            st.error("Application controller not available")
                    else:
                        st.error("❌ Please fill in all required fields (*)")
                        
        except Exception as e:
            logger.error(f"Error rendering manual entry: {str(e)}")
            st.error("Manual entry form error")
    
    def _render_image_upload(self):
        """Render image and document upload interface"""
        try:
            st.markdown("### Image & Document Analysis")
            st.markdown("Upload screenshots, PDFs, or images for AI analysis and insights.")
            
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
                    
                    if st.button("🔍 Process & Analyze Files", type="primary", use_container_width=True):
                        if 'app_controller' in st.session_state:
                            try:
                                for uploaded_file in uploaded_files:
                                    file_data = uploaded_file.read()
                                    success = st.session_state.app_controller.handle_data_upload(
                                        'image_document', 
                                        (file_data, uploaded_file.name, content_type, target_asin)
                                    )
                                    if not success:
                                        st.error(f"Failed to process {uploaded_file.name}")
                                
                                st.rerun()
                                
                            except Exception as e:
                                logger.error(f"Image processing error: {str(e)}")
                                st.error(f"Processing failed: {str(e)}")
                        else:
                            st.error("Application controller not available")
            
            with col2:
                st.markdown("### Processing Status")
                
                # Show API availability
                api_available = st.session_state.get('api_status', {}).get('available', False)
                if api_available:
                    st.success("✅ AI Analysis Available")
                    st.caption("GPT-4o Vision for images & PDFs")
                else:
                    st.error("❌ AI Analysis Unavailable")
                    st.caption("Configure OpenAI API key for AI analysis")
                
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
                
        except Exception as e:
            logger.error(f"Error rendering image upload: {str(e)}")
            st.error("Image upload interface error")
    
    def render_scoring_dashboard(self, scores: Dict[str, Any]):
        """Render main scoring dashboard"""
        try:
            st.markdown("## 📊 Performance Scoring Dashboard")
            
            if not scores:
                self.ui.show_alert("No scored products available. Please upload and analyze product data first.", "warning")
                return
            
            # Portfolio overview
            self._render_portfolio_overview(scores)
            
            # Individual product analysis
            self._render_product_selector(scores)
            
        except Exception as e:
            logger.error(f"Error rendering scoring dashboard: {str(e)}")
            st.error("Scoring dashboard error")
    
    def _render_portfolio_overview(self, scores: Dict[str, Any]):
        """Render portfolio-level overview"""
        try:
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
            
            # Performance distribution chart
            if PLOTLY_AVAILABLE:
                try:
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        # Create scatter plot
                        fig = self._create_portfolio_scatter(scores)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_b:
                        # Performance distribution pie chart
                        fig_pie = self._create_distribution_pie(excellent_count, good_count, average_count, 
                                                              needs_improvement_count, critical_count)
                        if fig_pie:
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                except Exception as e:
                    logger.error(f"Error creating portfolio charts: {str(e)}")
                    self._render_portfolio_fallback(scores)
            else:
                self._render_portfolio_fallback(scores)
                
        except Exception as e:
            logger.error(f"Error rendering portfolio overview: {str(e)}")
            st.error("Portfolio overview error")
    
    def _create_portfolio_scatter(self, scores):
        """Create portfolio scatter plot"""
        try:
            asins = list(scores.keys())
            composite_scores = [scores[asin].composite_score for asin in asins]
            categories = [scores[asin].category for asin in asins]
            names = [scores[asin].product_name[:30] + "..." if len(scores[asin].product_name) > 30 
                    else scores[asin].product_name for asin in asins]
            
            fig = px.scatter(
                x=range(len(asins)),
                y=composite_scores,
                color=categories,
                size=[abs(score-50)+20 for score in composite_scores],
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
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return None
    
    def _create_distribution_pie(self, excellent, good, average, needs_improvement, critical):
        """Create performance distribution pie chart"""
        try:
            labels = ['Excellent', 'Good', 'Average', 'Needs Improvement', 'Critical']
            values = [excellent, good, average, needs_improvement, critical]
            colors = [PERFORMANCE_COLORS[label] for label in labels]
            
            # Filter out zero values
            filtered_data = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 0]
            if not filtered_data:
                return None
            
            labels, values, colors = zip(*filtered_data)
            
            fig = px.pie(
                values=values,
                names=labels,
                color_discrete_sequence=colors,
                title="Performance Distribution"
            )
            
            fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return None
    
    def _render_portfolio_fallback(self, scores):
        """Fallback portfolio view without charts"""
        try:
            st.markdown("#### Performance Summary")
            score_values = [score.composite_score for score in scores.values()]
            
            # Performance distribution
            excellent = sum(1 for s in score_values if s >= 85)
            good = sum(1 for s in score_values if 70 <= s < 85)
            average = sum(1 for s in score_values if 55 <= s < 70)
            needs_improvement = sum(1 for s in score_values if 40 <= s < 55)
            critical = sum(1 for s in score_values if s < 40)
            
            st.write(f"**Excellent (85+):** {excellent} products")
            st.write(f"**Good (70-84):** {good} products")
            st.write(f"**Average (55-69):** {average} products")
            st.write(f"**Needs Improvement (40-54):** {needs_improvement} products")
            st.write(f"**Critical (<40):** {critical} products")
            
        except Exception as e:
            logger.error(f"Error in portfolio fallback: {str(e)}")
    
    def _render_product_selector(self, scores: Dict[str, Any]):
        """Render product selection and detailed analysis"""
        try:
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
                
        except Exception as e:
            logger.error(f"Error rendering product selector: {str(e)}")
            st.error("Product selector error")
    
    def _render_product_details(self, score: Any):
        """Render detailed product scoring analysis"""
        try:
            st.markdown(f"### {score.product_name}")
            st.markdown(f"**ASIN:** {score.asin} | **Category:** {score.category}")
            
            # Main score display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Composite score gauge
                gauge_chart = self.ui.create_score_gauge(score.composite_score, "Composite Score")
                if gauge_chart:
                    st.plotly_chart(gauge_chart, use_container_width=True)
            
            with col2:
                # Component breakdown
                if hasattr(score, 'component_scores') and score.component_scores:
                    component_data = {name: comp.raw_score for name, comp in score.component_scores.items()}
                    component_chart = self.ui.create_safe_bar_chart(component_data, "Component Breakdown")
                    if component_chart:
                        st.plotly_chart(component_chart, use_container_width=True)
            
            with col3:
                # Key metrics
                st.markdown("**Performance Level**")
                level_color = PERFORMANCE_COLORS.get(score.performance_level, COLORS['primary'])
                st.markdown(f'<div style="color: {level_color}; font-weight: bold; font-size: 1.2em;">{score.performance_level}</div>', 
                           unsafe_allow_html=True)
                
                if hasattr(score, 'revenue_impact') and score.revenue_impact:
                    st.metric("Revenue Impact", f"${score.revenue_impact:,.0f}/month")
                
                if hasattr(score, 'potential_savings') and score.potential_savings:
                    st.metric("Potential Savings", f"${score.potential_savings:,.0f}/month")
            
            # Detailed analysis tabs
            detail_tabs = st.tabs(["🎯 Recommendations", "⚠️ Risk Factors", "💪 Strengths", "📈 Components"])
            
            with detail_tabs[0]:
                st.markdown("#### Priority Improvements")
                if hasattr(score, 'improvement_priority') and score.improvement_priority:
                    for i, priority in enumerate(score.improvement_priority[:5], 1):
                        st.markdown(f"{i}. {priority}")
                else:
                    st.info("No specific improvements identified - product performing well overall.")
            
            with detail_tabs[1]:
                st.markdown("#### Risk Assessment")
                if hasattr(score, 'risk_factors') and score.risk_factors:
                    for risk in score.risk_factors:
                        st.markdown(f"⚠️ {risk}")
                else:
                    st.success("✅ No significant risk factors identified.")
            
            with detail_tabs[2]:
                st.markdown("#### Key Strengths")
                if hasattr(score, 'strengths') and score.strengths:
                    for strength in score.strengths:
                        st.markdown(f"💪 {strength}")
                else:
                    st.info("Strengths will be identified as performance improves.")
            
            with detail_tabs[3]:
                st.markdown("#### Component Score Details")
                if hasattr(score, 'component_scores') and score.component_scores:
                    for component_name, component_score in score.component_scores.items():
                        with st.expander(f"{component_name}: {component_score.raw_score:.1f}/100"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown(f"**Performance:** {component_score.performance_level}")
                                st.markdown(f"**Weight:** {component_score.weight:.0%}")
                                if hasattr(component_score, 'improvement_potential'):
                                    st.markdown(f"**Improvement Potential:** {component_score.improvement_potential:.1f} points")
                            
                            with col_b:
                                if hasattr(component_score, 'benchmark_comparison'):
                                    st.markdown(f"**Benchmark:** {component_score.benchmark_comparison}")
                                if hasattr(component_score, 'key_drivers') and component_score.key_drivers:
                                    st.markdown("**Key Drivers:**")
                                    for driver in component_score.key_drivers:
                                        st.markdown(f"• {driver}")
                                        
        except Exception as e:
            logger.error(f"Error rendering product details: {str(e)}")
            st.error("Product details error")
    
    def render_ai_chat_tab(self):
        """Render AI chat tab with error handling"""
        try:
            from ai_chat import AIChatInterface
            chat_interface = AIChatInterface()
            chat_interface.render_chat_interface()
        except ImportError:
            st.error("AI Chat module not available. Please ensure ai_chat.py is in your project directory.")
            st.info("The AI Chat feature provides standalone consulting without requiring data uploads.")
        except Exception as e:
            logger.error(f"AI chat error: {str(e)}")
            st.error(f"AI Chat error: {str(e)}")
    
    def render_ai_analysis_dashboard(self, ai_results: Dict[str, Any]):
        """Render AI analysis dashboard"""
        try:
            st.markdown("## 🧠 AI Product Analysis")
            st.markdown("Run AI analysis on your uploaded products to get detailed optimization recommendations.")
            
            # Check if we have data to analyze
            if not hasattr(st.session_state, 'processed_data') or not st.session_state.get('data_processed', False):
                st.warning("⚠️ No processed data available for AI analysis.")
                st.info("Please upload and process product data in the Data Import tab first.")
                return
            
            processed_data = st.session_state.processed_data
            products = processed_data.get('products', [])
            
            if not products:
                st.warning("No products found in processed data.")
                return
            
            # AI Analysis Control Panel
            self._render_ai_control_panel(products, ai_results)
            
            # Analysis results display
            if ai_results:
                self._render_ai_results_tabs(ai_results)
            else:
                st.info("No AI analysis results yet. Run analysis on products with review or return data.")
                
        except Exception as e:
            logger.error(f"Error rendering AI analysis dashboard: {str(e)}")
            st.error("AI analysis dashboard error")
    
    def _render_ai_control_panel(self, products: List[Dict], ai_results: Dict[str, Any]):
        """Render AI analysis control panel"""
        try:
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
                    st.session_state['run_individual_ai_analysis'] = selected_asin
                    st.rerun()
            
            with col3:
                # Bulk analysis for all products
                if st.button("🚀 Analyze All Products", type="secondary", use_container_width=True):
                    st.session_state['run_bulk_ai_analysis'] = True
                    st.rerun()
            
            # Analysis status
            if st.session_state.get('ai_analysis_in_progress', False):
                st.info("🔄 AI analysis in progress... Please wait.")
                
        except Exception as e:
            logger.error(f"Error rendering AI control panel: {str(e)}")
    
    def _render_ai_results_tabs(self, ai_results: Dict[str, Any]):
        """Render AI analysis results in tabs"""
        try:
            analysis_tabs = st.tabs(["📊 Analysis Summary", "📝 Review Insights", "↩️ Return Analysis", "🎯 Recommendations"])
            
            with analysis_tabs[0]:
                self._render_ai_summary(ai_results)
            
            with analysis_tabs[1]:
                self._render_ai_review_insights(ai_results)
            
            with analysis_tabs[2]:
                self._render_ai_return_insights(ai_results)
            
            with analysis_tabs[3]:
                self._render_ai_recommendations(ai_results)
                
        except Exception as e:
            logger.error(f"Error rendering AI results tabs: {str(e)}")
    
    def _render_ai_summary(self, ai_results):
        """Render AI analysis summary"""
        try:
            st.markdown("### AI Analysis Overview")
            
            # Analysis statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_products = len(st.session_state.processed_data.get('products', []))
            analyzed_products = len(ai_results)
            
            with col1:
                st.metric("Total Products", total_products)
            
            with col2:
                st.metric("AI Analyzed", analyzed_products)
            
            with col3:
                if analyzed_products > 0:
                    completion_rate = (analyzed_products / total_products) * 100
                    st.metric("Completion Rate", f"{completion_rate:.0f}%")
                else:
                    st.metric("Completion Rate", "0%")
            
            with col4:
                st.metric("API Calls Made", st.session_state.get('ai_api_calls', 0))
            
            # Analysis results overview
            if ai_results:
                st.markdown("### Products with AI Analysis")
                
                analysis_data = []
                products = st.session_state.processed_data.get('products', [])
                
                for asin, analysis in ai_results.items():
                    product = next((p for p in products if p['asin'] == asin), {})
                    
                    analysis_data.append({
                        'ASIN': asin,
                        'Product': product.get('name', 'Unknown'),
                        'Review Analysis': '✅' if 'review_analysis' in analysis else '❌',
                        'Return Analysis': '✅' if 'return_analysis' in analysis else '❌',
                        'Listing Optimization': '✅' if 'listing_optimization' in analysis else '❌'
                    })
                
                df = pd.DataFrame(analysis_data)
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error rendering AI summary: {str(e)}")
    
    def _render_ai_review_insights(self, ai_results):
        """Render AI review analysis results"""
        try:
            st.markdown("### 📝 Customer Review Analysis")
            
            if not ai_results:
                st.info("No AI review analysis available. Run AI analysis first.")
                return
            
            analyzed_products = list(ai_results.keys())
            if not analyzed_products:
                st.info("No products have been analyzed yet.")
                return
            
            # Product selector
            products = st.session_state.processed_data.get('products', [])
            selected_product = st.selectbox(
                "View detailed review analysis for:",
                options=analyzed_products,
                format_func=lambda asin: f"{asin} - {next((p.get('name', 'Unknown') for p in products if p['asin'] == asin), 'Unknown')}"
            )
            
            if selected_product and 'review_analysis' in ai_results[selected_product]:
                review_analysis = ai_results[selected_product]['review_analysis']
                
                if hasattr(review_analysis, 'success') and review_analysis.success:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Key Findings")
                        if hasattr(review_analysis, 'detailed_findings') and review_analysis.detailed_findings:
                            for category, finding in review_analysis.detailed_findings.items():
                                if finding:
                                    with st.expander(f"{category.replace('_', ' ').title()}"):
                                        st.markdown(finding)
                    
                    with col2:
                        st.markdown("#### Analysis Metrics")
                        if hasattr(review_analysis, 'confidence_score'):
                            st.metric("Confidence Score", f"{review_analysis.confidence_score:.0%}")
                        
                        if hasattr(review_analysis, 'data_quality'):
                            data_quality = review_analysis.data_quality
                            st.metric("Data Quality", data_quality.get('quality', 'Unknown').title())
                            st.metric("Reviews Analyzed", data_quality.get('total', 0))
                    
                    # Recommendations
                    if hasattr(review_analysis, 'recommendations') and review_analysis.recommendations:
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
                    if hasattr(review_analysis, 'errors'):
                        st.error(f"Review analysis failed: {', '.join(review_analysis.errors or ['Unknown error'])}")
                    else:
                        st.error("Review analysis failed")
            else:
                st.info("No review analysis available for this product.")
                
        except Exception as e:
            logger.error(f"Error rendering AI review insights: {str(e)}")
            st.error("Review insights error")
    
    def _render_ai_return_insights(self, ai_results):
        """Render AI return analysis results"""
        try:
            st.markdown("### ↩️ Return Reason Analysis")
            
            analyzed_products = [asin for asin, analysis in ai_results.items() if 'return_analysis' in analysis]
            
            if not analyzed_products:
                st.info("No products have return analysis yet. Products need return data to generate insights.")
                return
            
            # Product selector
            products = st.session_state.processed_data.get('products', [])
            selected_product = st.selectbox(
                "View return analysis for:",
                options=analyzed_products,
                format_func=lambda asin: f"{asin} - {next((p.get('name', 'Unknown') for p in products if p['asin'] == asin), 'Unknown')}",
                key="return_analysis_selector"
            )
            
            if selected_product:
                return_analysis = ai_results[selected_product]['return_analysis']
                
                if hasattr(return_analysis, 'success') and return_analysis.success:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Return Categories & Insights")
                        if hasattr(return_analysis, 'detailed_findings') and return_analysis.detailed_findings:
                            for category, finding in return_analysis.detailed_findings.items():
                                if finding:
                                    with st.expander(f"{category.replace('_', ' ').title()}"):
                                        st.markdown(finding)
                    
                    with col2:
                        st.markdown("#### Return Metrics")
                        if hasattr(return_analysis, 'confidence_score'):
                            st.metric("Analysis Confidence", f"{return_analysis.confidence_score:.0%}")
                        
                        if hasattr(return_analysis, 'data_quality'):
                            data_quality = return_analysis.data_quality
                            st.metric("Returns Analyzed", data_quality.get('total', 0))
                            st.metric("Data Quality", data_quality.get('quality', 'Unknown').title())
                else:
                    if hasattr(return_analysis, 'errors'):
                        st.error(f"Return analysis failed: {', '.join(return_analysis.errors or ['Unknown error'])}")
                    else:
                        st.error("Return analysis failed")
                        
        except Exception as e:
            logger.error(f"Error rendering AI return insights: {str(e)}")
            st.error("Return insights error")
    
    def _render_ai_recommendations(self, ai_results):
        """Render consolidated AI recommendations"""
        try:
            st.markdown("### 🎯 Consolidated AI Recommendations")
            
            # Compile all recommendations
            all_recommendations = []
            products = st.session_state.processed_data.get('products', [])
            
            for asin, analysis in ai_results.items():
                product_name = next((p.get('name', 'Unknown') for p in products if p['asin'] == asin), 'Unknown')
                
                # Collect recommendations from all analysis types
                for analysis_type in ['review_analysis', 'return_analysis', 'listing_optimization']:
                    if (analysis_type in analysis and 
                        hasattr(analysis[analysis_type], 'recommendations') and
                        analysis[analysis_type].recommendations):
                        
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
                try:
                    rec_df = pd.DataFrame(all_recommendations)
                    csv_data = rec_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download All Recommendations (CSV)",
                        data=csv_data,
                        file_name=f"ai_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    logger.error(f"Error creating recommendations export: {str(e)}")
            else:
                st.info("No AI recommendations generated yet. Run AI analysis on products with review or return data.")
                
        except Exception as e:
            logger.error(f"Error rendering AI recommendations: {str(e)}")
            st.error("AI recommendations error")
    
    def render_export_dashboard(self, scores: Dict[str, Any], ai_results: Dict[str, Any]):
        """Render export and reporting dashboard"""
        try:
            st.markdown("## 📋 Export & Reporting")
            
            export_tabs = st.tabs(["📊 Scorecards", "📈 Executive Report", "📋 Action Items", "💾 Raw Data"])
            
            with export_tabs[0]:
                self._render_scorecards_export(scores)
            
            with export_tabs[1]:
                self._render_executive_report(scores, ai_results)
            
            with export_tabs[2]:
                self._render_action_items_export(scores)
            
            with export_tabs[3]:
                self._render_raw_data_export(scores, ai_results)
                
        except Exception as e:
            logger.error(f"Error rendering export dashboard: {str(e)}")
            st.error("Export dashboard error")
    
    def _render_scorecards_export(self, scores):
        """Render scorecards export section"""
        try:
            st.markdown("### Performance Scorecards")
            
            if scores:
                # Show preview of first few scorecards
                st.markdown("#### Preview")
                for asin, score in list(scores.items())[:2]:
                    with st.expander(f"Scorecard: {score.product_name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Composite Score", f"{score.composite_score:.1f}/100")
                            st.write(f"**Performance:** {score.performance_level}")
                            st.write(f"**Category:** {score.category}")
                        
                        with col2:
                            if hasattr(score, 'improvement_priority') and score.improvement_priority:
                                st.write(f"**Top Priority:** {score.improvement_priority[0]}")
                            if hasattr(score, 'revenue_impact') and score.revenue_impact:
                                st.metric("Revenue Impact", f"${score.revenue_impact:,.0f}/month")
                
                # Export button
                if 'app_controller' in st.session_state:
                    if st.button("📥 Generate Scorecards Export", use_container_width=True):
                        try:
                            export_data = st.session_state.app_controller.export_results('excel')
                            if export_data:
                                st.download_button(
                                    label="📥 Download Performance Scorecards (Excel)",
                                    data=export_data,
                                    file_name=f"performance_scorecards_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        except Exception as e:
                            logger.error(f"Export error: {str(e)}")
                            st.error(f"Export failed: {str(e)}")
            else:
                st.info("No scored products available for export.")
                
        except Exception as e:
            logger.error(f"Error rendering scorecards export: {str(e)}")
    
    def _render_executive_report(self, scores, ai_results):
        """Render executive report section"""
        try:
            st.markdown("### Executive Summary Report")
            
            if scores:
                score_values = [s.composite_score for s in scores.values()]
                avg_score = np.mean(score_values)
                
                # Generate executive summary
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
                
                st.markdown("#### Executive Summary Preview")
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
                
        except Exception as e:
            logger.error(f"Error rendering executive report: {str(e)}")
    
    def _render_action_items_export(self, scores):
        """Render action items export section"""
        try:
            st.markdown("### Action Items Export")
            
            if scores:
                # Compile all action items
                all_actions = []
                for score in scores.values():
                    if hasattr(score, 'improvement_priority') and score.improvement_priority:
                        for action in score.improvement_priority[:3]:
                            all_actions.append({
                                'ASIN': score.asin,
                                'Product': score.product_name,
                                'Current_Score': score.composite_score,
                                'Performance_Level': score.performance_level,
                                'Action': action,
                                'Priority': 'High' if score.composite_score < 55 else 'Medium',
                                'Revenue_Impact': getattr(score, 'revenue_impact', '') or '',
                                'Potential_Savings': getattr(score, 'potential_savings', '') or ''
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
                
        except Exception as e:
            logger.error(f"Error rendering action items export: {str(e)}")
    
    def _render_raw_data_export(self, scores, ai_results):
        """Render raw data export section"""
        try:
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
                    if 'app_controller' in st.session_state:
                        try:
                            if export_format.lower() == "excel":
                                export_data = st.session_state.app_controller.export_results('excel')
                                if export_data:
                                    st.download_button(
                                        label="📥 Download Excel Export",
                                        data=export_data,
                                        file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            elif export_format.lower() == "json":
                                export_data = st.session_state.app_controller.export_results('json')
                                if export_data:
                                    st.download_button(
                                        label="📥 Download JSON Export",
                                        data=export_data,
                                        file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                        mime="application/json"
                                    )
                            else:  # CSV
                                # Create basic CSV export
                                try:
                                    score_data = []
                                    for asin, score in scores.items():
                                        score_data.append({
                                            'ASIN': score.asin,
                                            'Product_Name': score.product_name,
                                            'Category': score.category,
                                            'Composite_Score': score.composite_score,
                                            'Performance_Level': score.performance_level
                                        })
                                    
                                    score_df = pd.DataFrame(score_data)
                                    csv_data = score_df.to_csv(index=False)
                                    
                                    st.download_button(
                                        label="📥 Download CSV Export",
                                        data=csv_data,
                                        file_name=f"product_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                        mime="text/csv"
                                    )
                                except Exception as e:
                                    logger.error(f"CSV export error: {str(e)}")
                                    st.error(f"CSV export failed: {str(e)}")
                        except Exception as e:
                            logger.error(f"Export generation error: {str(e)}")
                            st.error(f"Export failed: {str(e)}")
            else:
                st.info("No scored products available for export.")
                
        except Exception as e:
            logger.error(f"Error rendering raw data export: {str(e)}")
    
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

class ProfessionalDashboard:
    """Main dashboard orchestrator with comprehensive error handling"""
    
    def __init__(self):
        try:
            self.renderer = DashboardRenderer()
            self.ui = UIComponents()
            logger.info("Dashboard components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing dashboard components: {str(e)}")
            raise
    
    def initialize_app(self):
        """Initialize the Streamlit app with professional styling"""
        try:
            # Apply professional theme (page config handled in main app)
            self.ui.set_professional_theme()
            
        except Exception as e:
            logger.error(f"Error initializing app: {str(e)}")
            raise
    
    def render_main_dashboard(self):
        """Render the main dashboard with comprehensive error handling"""
        try:
            # Apply theme first (safely)
            try:
                self.ui.set_professional_theme()
            except Exception as e:
                logger.warning(f"Theme application failed: {str(e)}")
            
            # Store reference to app controller for upload handling
            if hasattr(st.session_state, 'app_controller'):
                st.session_state.app_controller = st.session_state.app_controller
            
            # Header
            self.renderer.render_header()
            
            # Sidebar
            module_status = st.session_state.get('module_status', {})
            api_status = st.session_state.get('api_status', {})
            self.renderer.render_sidebar_status(module_status, api_status)
            
            # Main content tabs
            main_tabs = st.tabs([
                "📁 Data Import", 
                "📊 Performance Scores", 
                "🤖 AI Chat",
                "🧠 AI Analysis",
                "📋 Export & Reports"
            ])
            
            with main_tabs[0]:
                self.renderer.render_upload_dashboard()
            
            with main_tabs[1]:
                scored_products = st.session_state.get('scored_products', {})
                self.renderer.render_scoring_dashboard(scored_products)
            
            with main_tabs[2]:
                self.renderer.render_ai_chat_tab()
            
            with main_tabs[3]:
                ai_results = st.session_state.get('ai_analysis_results', {})
                self.renderer.render_ai_analysis_dashboard(ai_results)
            
            with main_tabs[4]:
                scored_products = st.session_state.get('scored_products', {})
                ai_results = st.session_state.get('ai_analysis_results', {})
                self.renderer.render_export_dashboard(scored_products, ai_results)
                
        except Exception as e:
            logger.error(f"Error rendering main dashboard: {str(e)}")
            st.error(f"Dashboard rendering error: {str(e)}")
            
            # Fallback basic interface
            st.title("🏥 Amazon Medical Device Listing Optimizer")
            st.error("⚠️ Dashboard rendering failed. Using basic interface.")
            
            # Show system status
            st.markdown("### System Status")
            if 'module_status' in st.session_state:
                for module, available in st.session_state.module_status.items():
                    icon = "✅" if available else "❌"
                    st.markdown(f"{icon} {module.replace('_', ' ').title()}")

# Export main classes
__all__ = ['ProfessionalDashboard', 'DashboardRenderer', 'UIComponents']
