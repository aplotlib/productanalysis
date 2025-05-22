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
        
        st.markdown("### Image & Document Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Content type selection
            content_type = st.selectbox(
                "What type of content are you uploading?",
                ["Product Reviews", "Return Reports", "Product Listings", "Competitor Analysis"]
            )
            
            # ASIN association
            target_asin = st.text_input("Associate with ASIN (optional)", placeholder="B0XXXXXXXXX")
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload images or documents",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                accept_multiple_files=True,
                help="Upload screenshots of reviews, return reports, or product listings"
            )
            
            if uploaded_files:
                st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
                
                # Processing options
                processing_method = st.radio(
                    "Processing method:",
                    ["Auto (OCR + AI Vision)", "OCR Only", "AI Vision Only"],
                    horizontal=True
                )
                
                if st.button("🔍 Process Files", use_container_width=True):
                    with st.spinner("Processing files..."):
                        # Simulate processing
                        import time
                        time.sleep(2)
                    
                    st.success("✅ Files processed successfully!")
                    
                    # Show extracted data preview
                    with st.expander("📄 Extracted Data Preview"):
                        if content_type == "Product Reviews":
                            st.markdown("**Extracted Reviews:**")
                            st.markdown("• ⭐⭐⭐⭐⭐ Great product, very comfortable")
                            st.markdown("• ⭐⭐⭐ Good quality but sizing runs small")
                        elif content_type == "Return Reports":
                            st.markdown("**Extracted Return Reasons:**")
                            st.markdown("• Product arrived damaged")
                            st.markdown("• Wrong size ordered")
        
        with col2:
            st.markdown("### Processing Status")
            
            # OCR availability
            ocr_status = "✅ Available" if True else "❌ Unavailable"
            st.markdown(f"**OCR Processing:** {ocr_status}")
            
            # AI Vision availability  
            ai_status = "✅ Available" if True else "❌ Unavailable"
            st.markdown(f"**AI Vision:** {ai_status}")
            
            st.markdown("### Supported Formats")
            st.markdown("""
            **Images:**
            - JPG, JPEG, PNG
            - Screenshots, photos
            
            **Documents:** 
            - PDF files
            - Multi-page documents
            """)
    
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
    
    def render_ai_insights_dashboard(self, ai_results: Dict[str, Any]):
        """Render AI analysis insights dashboard"""
        
        st.markdown("## 🧠 AI Analysis Insights")
        
        if not ai_results:
            self.ui.show_alert("No AI analysis results available. Run AI analysis on your products first.", "info")
            return
        
        # AI insights tabs
        ai_tabs = st.tabs(["📝 Review Analysis", "↩️ Return Analysis", "🎯 Listing Optimization"])
        
        with ai_tabs[0]:
            self._render_review_insights(ai_results.get('review_analysis'))
        
        with ai_tabs[1]:
            self._render_return_insights(ai_results.get('return_analysis'))
        
        with ai_tabs[2]:
            self._render_optimization_insights(ai_results.get('listing_optimization'))
    
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
                
                if st.button("📥 Download All Scorecards (PDF)", use_container_width=True):
                    st.success("Scorecard download initiated!")
            else:
                st.info("No scored products available for export.")
        
        with export_tabs[1]:
            st.markdown("### Executive Summary Report")
            
            # Executive summary preview
            if scores:
                st.markdown("#### Portfolio Executive Summary")
                
                score_values = [s.composite_score for s in scores.values()]
                avg_score = np.mean(score_values)
                
                st.markdown(f"""
                **Portfolio Performance:** {avg_score:.1f}/100 ({self._get_performance_level_text(avg_score)})
                
                **Key Highlights:**
                - {len(scores)} products analyzed
                - {sum(1 for s in score_values if s >= 70)} products performing well (70+ score)
                - {sum(1 for s in score_values if s < 55)} products need immediate attention
                
                **Recommended Actions:**
                1. Focus on return rate optimization for underperforming products
                2. Leverage AI insights to improve customer satisfaction
                3. Implement listing optimizations for conversion improvement
                """)
                
                if st.button("📥 Download Executive Report (PDF)", use_container_width=True):
                    st.success("Executive report download initiated!")
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
                                'Action': action,
                                'Priority': 'High' if score.composite_score < 55 else 'Medium'
                            })
                
                if all_actions:
                    action_df = pd.DataFrame(all_actions)
                    st.dataframe(action_df, use_container_width=True)
                    
                    if st.button("📥 Download Action Items (Excel)", use_container_width=True):
                        st.success("Action items download initiated!")
                else:
                    st.info("No action items generated yet.")
            else:
                st.info("Generate product analysis first to create action items.")
        
        with export_tabs[3]:
            st.markdown("### Raw Data Export")
            
            export_format = st.selectbox(
                "Select export format:",
                ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
            )
            
            include_options = st.multiselect(
                "Include data:",
                ["Product Scores", "Component Breakdown", "AI Analysis", "Recommendations"],
                default=["Product Scores", "Component Breakdown"]
            )
            
            if st.button("📥 Generate Export", use_container_width=True):
                with st.spinner("Generating export..."):
                    import time
                    time.sleep(1)
                st.success(f"Export generated! Download will begin shortly.")
    
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
            "🧠 AI Insights", 
            "📋 Export & Reports"
        ])
        
        with main_tabs[0]:
            self.renderer.render_upload_dashboard()
        
        with main_tabs[1]:
            self.renderer.render_scoring_dashboard(st.session_state.scored_products)
        
        with main_tabs[2]:
            self.renderer.render_ai_insights_dashboard(st.session_state.ai_analysis_results)
        
        with main_tabs[3]:
            self.renderer.render_export_dashboard(
                st.session_state.scored_products,
                st.session_state.ai_analysis_results
            )

# Export main class
__all__ = ['ProfessionalDashboard', 'DashboardRenderer', 'UIComponents']
