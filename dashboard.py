"""
Text Analysis Dashboard Module for Medical Device Customer Feedback Analyzer

**PRIMARY FOCUS: Customer Feedback Text Analysis & Quality Management Dashboard**

This dashboard provides listing managers with:
✓ Customer feedback categorization and insights
✓ Date range filtering for temporal analysis
✓ Medical device quality management views
✓ CAPA (Corrective and Preventive Action) recommendations
✓ Risk assessment and safety monitoring
✓ ISO 13485 compliance-aware quality tracking

Author: Assistant
Version: 3.0 - Text Analysis Focused
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, Counter

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

# Import dependencies
go, px, make_subplots, PLOTLY_AVAILABLE = safe_import_plotly()

# Professional color scheme for medical device quality management
COLORS = {
    'primary': '#1E40AF',          # Professional blue
    'secondary': '#059669',        # Success green
    'accent': '#DC2626',           # Critical red
    'warning': '#D97706',          # Warning orange
    'info': '#0891B2',            # Info blue
    'text_primary': '#1F2937',    # Dark gray
    'text_secondary': '#6B7280',  # Medium gray
    'background': '#F9FAFB',      # Light background
    'surface': '#FFFFFF',         # White surface
    'border': '#E5E7EB'           # Light border
}

# Quality category colors (medical device specific)
QUALITY_COLORS = {
    'safety_concerns': '#DC2626',      # Critical red
    'efficacy_performance': '#B91C1C', # Dark red
    'durability_quality': '#EA580C',   # Orange
    'comfort_usability': '#D97706',    # Amber
    'assembly_instructions': '#0891B2', # Cyan
    'sizing_fit': '#7C3AED',          # Purple
    'shipping_packaging': '#059669',   # Green
    'positive_feedback': '#22C55E'     # Success green
}

# Risk level colors
RISK_COLORS = {
    'Critical': '#DC2626',
    'High': '#EA580C', 
    'Medium': '#D97706',
    'Low': '#059669',
    'Unknown': '#6B7280'
}

# CAPA priority colors
CAPA_COLORS = {
    'Critical': '#DC2626',
    'High': '#EA580C',
    'Medium': '#D97706',
    'Low': '#059669'
}

class UIComponents:
    """Reusable UI components for text analysis dashboard"""
    
    @staticmethod
    def set_text_analysis_theme():
        """Apply professional theme for text analysis dashboard"""
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
                
                .quality-category-card {{
                    background-color: {COLORS['surface']};
                    border-left: 4px solid {COLORS['primary']};
                    border-radius: 0 8px 8px 0;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }}
                
                .capa-card {{
                    background-color: {COLORS['surface']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                
                .risk-indicator {{
                    padding: 0.25rem 0.75rem;
                    border-radius: 4px;
                    font-weight: 600;
                    font-size: 0.875rem;
                    color: white;
                    display: inline-block;
                }}
                
                .feedback-item {{
                    background-color: {COLORS['surface']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 6px;
                    padding: 0.75rem;
                    margin: 0.25rem 0;
                }}
                
                .date-filter-container {{
                    background-color: {COLORS['surface']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 1rem 0;
                }}
            </style>
            """, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error applying theme: {str(e)}")
    
    @staticmethod
    def show_quality_alert(message: str, risk_level: str = "Medium"):
        """Display styled quality management alert"""
        try:
            color = RISK_COLORS.get(risk_level, COLORS['warning'])
            
            if risk_level == "Critical":
                st.error(f"🚨 {message}")
            elif risk_level == "High":
                st.error(f"⚠️ {message}")
            elif risk_level == "Medium":
                st.warning(f"⚡ {message}")
            else:
                st.info(f"ℹ️ {message}")
                
        except Exception as e:
            logger.error(f"Error showing quality alert: {str(e)}")
            st.write(message)
    
    @staticmethod
    def create_quality_category_chart(category_data: Dict[str, Any]):
        """Create quality category breakdown chart"""
        if not PLOTLY_AVAILABLE or not category_data:
            return None
        
        try:
            # Prepare data for chart
            categories = []
            counts = []
            colors = []
            
            for cat_id, cat_info in category_data.items():
                if cat_id == 'summary':
                    continue
                    
                count = cat_info.get('count', 0)
                if count > 0:
                    categories.append(cat_info.get('name', cat_id))
                    counts.append(count)
                    colors.append(QUALITY_COLORS.get(cat_id, COLORS['primary']))
            
            if not categories:
                return None
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                y=categories,
                x=counts,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{count} items" for count in counts],
                textposition='inside',
                textfont=dict(color='white', size=12)
            ))
            
            fig.update_layout(
                title="Customer Feedback by Quality Category",
                xaxis_title="Number of Feedback Items",
                yaxis_title="",
                height=max(300, len(categories) * 40),
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS['text_primary'])
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality category chart: {str(e)}")
            return None
    
    @staticmethod
    def create_temporal_trend_chart(trend_data: Dict[str, Any]):
        """Create temporal trend analysis chart"""
        if not PLOTLY_AVAILABLE or not trend_data:
            return None
        
        try:
            weekly_trends = trend_data.get('weekly_trends', {})
            if not weekly_trends:
                return None
            
            weeks = list(weekly_trends.keys())
            counts = list(weekly_trends.values())
            
            fig = go.Figure(go.Scatter(
                x=weeks,
                y=counts,
                mode='lines+markers',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8, color=COLORS['accent']),
                fill='tonexty',
                name='Feedback Volume'
            ))
            
            # Add trend line
            if len(counts) > 1:
                z = np.polyfit(range(len(counts)), counts, 1)
                p = np.poly1d(z)
                trend_line = p(range(len(counts)))
                
                fig.add_trace(go.Scatter(
                    x=weeks,
                    y=trend_line,
                    mode='lines',
                    line=dict(color=COLORS['warning'], width=2, dash='dash'),
                    name='Trend'
                ))
            
            fig.update_layout(
                title="Customer Feedback Temporal Trends",
                xaxis_title="Time Period",
                yaxis_title="Feedback Count",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS['text_primary']),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating temporal trend chart: {str(e)}")
            return None
    
    @staticmethod
    def create_risk_assessment_gauge(risk_level: str, risk_score: float):
        """Create risk assessment gauge"""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Color mapping for risk levels
            color_map = {
                'Low': COLORS['secondary'],
                'Medium': COLORS['warning'], 
                'High': COLORS['accent'],
                'Critical': '#B91C1C'
            }
            
            gauge_color = color_map.get(risk_level, COLORS['primary'])
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Assessment: {risk_level}", 'font': {'size': 16}},
                number={'font': {'size': 24, 'color': gauge_color}},
                gauge={
                    'axis': {'range': [None, 50], 'tickwidth': 1},
                    'bar': {'color': gauge_color, 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': COLORS['border'],
                    'steps': [
                        {'range': [0, 5], 'color': "#D1FAE5"},   # Low
                        {'range': [5, 15], 'color': "#FEF3C7"},  # Medium  
                        {'range': [15, 30], 'color': "#FEE2E2"}, # High
                        {'range': [30, 50], 'color': "#FECACA"}  # Critical
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
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
            
        except Exception as e:
            logger.error(f"Error creating risk gauge: {str(e)}")
            return None

class DateFilterRenderer:
    """Renders date filtering interface for temporal analysis"""
    
    @staticmethod
    def render_date_filter_controls():
        """Render date filtering controls"""
        try:
            st.markdown("### 📅 Date Range Analysis")
            
            with st.container():
                st.markdown("""
                <div class="date-filter-container">
                    <h4>Temporal Analysis Controls</h4>
                    <p>Filter customer feedback by date range to identify trends and patterns over time.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Date filter enable/disable
                    filter_enabled = st.checkbox(
                        "Enable Date Range Filtering",
                        value=st.session_state.get('date_filter_enabled', False),
                        help="Filter customer feedback analysis to specific time periods"
                    )
                    st.session_state.date_filter_enabled = filter_enabled
                    
                    if filter_enabled:
                        # Predefined date ranges
                        date_options = {
                            'last_7_days': 'Last 7 Days',
                            'last_30_days': 'Last 30 Days', 
                            'last_90_days': 'Last 90 Days',
                            'last_180_days': 'Last 6 Months',
                            'last_365_days': 'Last 12 Months',
                            'custom': 'Custom Date Range'
                        }
                        
                        selected_option = st.selectbox(
                            "Select Date Range",
                            options=list(date_options.keys()),
                            format_func=lambda x: date_options[x],
                            index=1,  # Default to last 30 days
                            key="date_filter_option"
                        )
                        
                        # Custom date range inputs
                        if selected_option == 'custom':
                            col_start, col_end = st.columns(2)
                            
                            with col_start:
                                start_date = st.date_input(
                                    "Start Date",
                                    value=date.today() - timedelta(days=30),
                                    max_value=date.today(),
                                    key="custom_start_date"
                                )
                            
                            with col_end:
                                end_date = st.date_input(
                                    "End Date", 
                                    value=date.today(),
                                    min_value=start_date if 'start_date' in locals() else date.today() - timedelta(days=365),
                                    max_value=date.today(),
                                    key="custom_end_date"
                                )
                
                with col2:
                    # Show current filter status
                    if filter_enabled:
                        current_filter = DateFilterRenderer._get_current_filter_info()
                        
                        if current_filter:
                            st.markdown("**Active Filter:**")
                            st.info(f"📅 {current_filter['label']}")
                            
                            if current_filter.get('days_span'):
                                st.metric("Days Analyzed", current_filter['days_span'])
                        else:
                            st.warning("⚠️ Please select valid date range")
                    else:
                        st.info("📊 Analyzing all available data")
                
                # Apply filter button
                if filter_enabled:
                    if st.button("🔄 Apply Date Filter", type="primary", use_container_width=True):
                        st.session_state.date_filter_changed = True
                        st.rerun()
                        
        except Exception as e:
            logger.error(f"Error rendering date filter controls: {str(e)}")
            st.error("Error rendering date filter controls")
    
    @staticmethod
    def _get_current_filter_info() -> Optional[Dict[str, Any]]:
        """Get current date filter information"""
        try:
            if not st.session_state.get('date_filter_enabled', False):
                return None
            
            option = st.session_state.get('date_filter_option', 'last_30_days')
            
            if option == 'custom':
                start_date = st.session_state.get('custom_start_date')
                end_date = st.session_state.get('custom_end_date')
                
                if start_date and end_date:
                    days_span = (end_date - start_date).days + 1
                    return {
                        'label': f"Custom: {start_date} to {end_date}",
                        'days_span': days_span,
                        'start_date': start_date,
                        'end_date': end_date
                    }
            else:
                # Predefined ranges
                date_map = {
                    'last_7_days': {'days': 7, 'label': 'Last 7 Days'},
                    'last_30_days': {'days': 30, 'label': 'Last 30 Days'},
                    'last_90_days': {'days': 90, 'label': 'Last 90 Days'},
                    'last_180_days': {'days': 180, 'label': 'Last 6 Months'},
                    'last_365_days': {'days': 365, 'label': 'Last 12 Months'}
                }
                
                if option in date_map:
                    config = date_map[option]
                    end_date = date.today()
                    start_date = end_date - timedelta(days=config['days'])
                    
                    return {
                        'label': config['label'],
                        'days_span': config['days'],
                        'start_date': start_date,
                        'end_date': end_date
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current filter info: {str(e)}")
            return None

class TextAnalysisDashboardRenderer:
    """Main dashboard renderer focused on text analysis and quality management"""
    
    def __init__(self):
        self.ui = UIComponents()
        self.date_filter = DateFilterRenderer()
    
    def render_header(self):
        """Render application header for text analysis focus"""
        try:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #1E40AF 0%, #1D4ED8 100%); 
                        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;">
                <h1 style="color: white; margin: 0;">🔍 Medical Device Customer Feedback Analyzer</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Text analysis and quality management for medical device customer feedback
                </p>
                <small style="opacity: 0.8;">ISO 13485 Quality Management Aware | CAPA Recommendations | Risk Assessment</small>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering header: {str(e)}")
            st.title("🔍 Medical Device Customer Feedback Analyzer")
    
    def render_sidebar_status(self, module_status: Dict[str, bool], api_status: Dict[str, Any]):
        """Render sidebar with system status"""
        try:
            with st.sidebar:
                st.markdown("### System Status")
                
                # Module status
                st.markdown("**Core Modules:**")
                core_modules = ['upload_handler', 'enhanced_ai_analysis', 'dashboard']
                for module in core_modules:
                    available = module_status.get(module, False)
                    icon = "✅" if available else "❌"
                    module_name = module.replace('_', ' ').title()
                    st.markdown(f"{icon} {module_name}")
                
                # Text analysis engine status
                st.markdown("**Text Analysis Engine:**")
                if module_status.get('text_analysis', True):  # Assume available
                    st.success("✅ Text Analysis Ready")
                    st.caption("Customer feedback categorization active")
                else:
                    st.error("❌ Text Analysis Unavailable")
                
                # AI enhancement status
                st.markdown("**AI Enhancement:**")
                if api_status.get('available', False):
                    st.success("✅ AI Enhancement Available")
                    if 'model' in api_status:
                        st.caption(f"Model: {api_status['model']}")
                else:
                    st.error("❌ AI Enhancement Unavailable")
                    if 'error' in api_status:
                        st.caption(f"Error: {api_status['error']}")
                
                # Quality management features
                st.markdown("### Quality Management")
                capa_enabled = st.session_state.get('capa_tracking_enabled', True)
                risk_enabled = st.session_state.get('risk_assessment_enabled', True)
                iso_mode = st.session_state.get('iso_compliance_mode', True)
                
                st.checkbox("CAPA Tracking", value=capa_enabled, key="capa_tracking_enabled")
                st.checkbox("Risk Assessment", value=risk_enabled, key="risk_assessment_enabled") 
                st.checkbox("ISO 13485 Mode", value=iso_mode, key="iso_compliance_mode")
                
                # Quick actions
                st.markdown("### Quick Actions")
                if st.button("🔄 Refresh Analysis", use_container_width=True):
                    st.rerun()
                
                if st.button("📊 Example Data", use_container_width=True):
                    st.session_state['load_example'] = True
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Error rendering sidebar: {str(e)}")
    
    def render_upload_dashboard(self):
        """Render upload dashboard focused on customer feedback"""
        try:
            st.markdown("## 📁 Customer Feedback Data Import")
            
            # Upload method tabs
            upload_tabs = st.tabs(["📊 Product Data", "💬 Customer Feedback", "🖼️ Documents & Screenshots"])
            
            with upload_tabs[0]:
                self._render_product_data_upload()
            
            with upload_tabs[1]:
                self._render_feedback_entry()
            
            with upload_tabs[2]:
                self._render_document_upload()
                
        except Exception as e:
            logger.error(f"Error rendering upload dashboard: {str(e)}")
            st.error("Upload dashboard error")
    
    def _render_product_data_upload(self):
        """Render product data upload (metadata only)"""
        try:
            st.markdown("### Product Information Upload")
            st.info("ℹ️ Upload basic product information (ASIN, name, category). Customer feedback will be entered separately.")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # File upload
                uploaded_file = st.file_uploader(
                    "Choose CSV or Excel file",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload file with product metadata (ASIN, Product Name, Category, etc.)"
                )
                
                if uploaded_file:
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
                st.markdown("### Required Fields")
                st.markdown("""
                **Essential:**
                - ✅ ASIN
                - ✅ Product Name  
                - ✅ Category
                
                **Optional:**
                - SKU
                - Current return rate
                - Star rating
                """)
                
                # Current data summary
                if 'processed_data' in st.session_state and st.session_state.processed_data:
                    products = st.session_state.processed_data.get('products', [])
                    st.metric("Products Loaded", len(products))
                else:
                    st.metric("Products Loaded", "0")
                    
        except Exception as e:
            logger.error(f"Error rendering product data upload: {str(e)}")
    
    def _render_feedback_entry(self):
        """Render customer feedback entry interface"""
        try:
            st.markdown("### Customer Feedback Entry")
            st.info("💬 Enter customer reviews, return reasons, and other feedback for text analysis.")
            
            # Product selection
            products = []
            if 'processed_data' in st.session_state and st.session_state.processed_data:
                products = st.session_state.processed_data.get('products', [])
            
            if not products:
                st.warning("⚠️ Please upload product data first to associate feedback with specific products.")
                return
            
            # Feedback entry form
            with st.form("customer_feedback_entry"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Product selection
                    product_options = [(p['asin'], f"{p['name']} ({p['asin']})") for p in products]
                    selected_asin = st.selectbox(
                        "Select Product",
                        options=[asin for asin, _ in product_options],
                        format_func=lambda asin: next(name for a, name in product_options if a == asin)
                    )
                    
                    # Feedback type
                    feedback_type = st.selectbox(
                        "Feedback Type",
                        options=['review', 'return_reason', 'customer_service', 'other'],
                        format_func=lambda x: {
                            'review': 'Customer Review',
                            'return_reason': 'Return Reason', 
                            'customer_service': 'Customer Service Inquiry',
                            'other': 'Other Feedback'
                        }[x]
                    )
                    
                    # Rating (for reviews)
                    rating = None
                    if feedback_type == 'review':
                        rating = st.slider("Star Rating", 1, 5, 3)
                
                with col2:
                    # Date
                    feedback_date = st.date_input(
                        "Feedback Date",
                        value=date.today(),
                        max_value=date.today()
                    )
                    
                    # Source
                    source = st.selectbox(
                        "Source",
                        options=['amazon_review', 'return_report', 'customer_email', 'phone_call', 'other'],
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                
                # Feedback text
                feedback_text = st.text_area(
                    "Customer Feedback Text",
                    placeholder="Enter the customer's exact feedback, review text, or return reason...",
                    height=150
                )
                
                # Submit button
                if st.form_submit_button("💾 Add Feedback", use_container_width=True):
                    if feedback_text.strip():
                        # Add feedback to session state
                        if 'uploaded_data' not in st.session_state:
                            st.session_state.uploaded_data = {}
                        
                        feedback_item = {
                            'text': feedback_text.strip(),
                            'rating': rating,
                            'date': feedback_date.strftime('%Y-%m-%d'),
                            'source': source,
                            'asin': selected_asin
                        }
                        
                        # Add to appropriate category
                        if feedback_type == 'review':
                            if 'manual_reviews' not in st.session_state.uploaded_data:
                                st.session_state.uploaded_data['manual_reviews'] = {}
                            if selected_asin not in st.session_state.uploaded_data['manual_reviews']:
                                st.session_state.uploaded_data['manual_reviews'][selected_asin] = []
                            
                            st.session_state.uploaded_data['manual_reviews'][selected_asin].append({
                                'review_text': feedback_text.strip(),
                                'rating': rating,
                                'date': feedback_date.strftime('%Y-%m-%d'),
                                'asin': selected_asin
                            })
                        else:
                            if 'manual_returns' not in st.session_state.uploaded_data:
                                st.session_state.uploaded_data['manual_returns'] = {}
                            if selected_asin not in st.session_state.uploaded_data['manual_returns']:
                                st.session_state.uploaded_data['manual_returns'][selected_asin] = []
                            
                            st.session_state.uploaded_data['manual_returns'][selected_asin].append({
                                'return_reason': feedback_text.strip(),
                                'date': feedback_date.strftime('%Y-%m-%d'),
                                'asin': selected_asin
                            })
                        
                        st.success(f"✅ Feedback added for {selected_asin}")
                        st.rerun()
                    else:
                        st.error("Please enter feedback text")
            
            # Show current feedback count
            if 'uploaded_data' in st.session_state:
                reviews_count = sum(len(reviews) for reviews in st.session_state.uploaded_data.get('manual_reviews', {}).values())
                returns_count = sum(len(returns) for returns in st.session_state.uploaded_data.get('manual_returns', {}).values())
                
                col_a, col_b = st.columns(2)
                col_a.metric("Reviews Entered", reviews_count)
                col_b.metric("Returns Entered", returns_count)
                    
        except Exception as e:
            logger.error(f"Error rendering feedback entry: {str(e)}")
            st.error("Feedback entry error")
    
    def _render_document_upload(self):
        """Render document upload for screenshots and PDFs"""
        try:
            st.markdown("### Document & Screenshot Analysis")
            st.info("🖼️ Upload Amazon screenshots, return reports, or PDFs for AI-powered text extraction.")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Content type selection
                content_type = st.selectbox(
                    "Document Type",
                    ["Product Reviews", "Return Reports", "Product Listings", "Customer Messages"]
                )
                
                # Optional ASIN
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
                    help="Upload screenshots of Amazon pages, return reports, or PDF documents"
                )
                
                if uploaded_files:
                    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
                    
                    if st.button("🔍 Process & Extract Text", type="primary", use_container_width=True):
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
                                logger.error(f"Document processing error: {str(e)}")
                                st.error(f"Processing failed: {str(e)}")
                        else:
                            st.error("Application controller not available")
            
            with col2:
                st.markdown("### AI Text Extraction")
                
                # Show API availability
                api_available = st.session_state.get('api_status', {}).get('available', False)
                if api_available:
                    st.success("✅ AI Text Extraction Available")
                    st.caption("GPT-4o Vision for images & PDFs")
                else:
                    st.error("❌ AI Extraction Unavailable")
                    st.caption("Configure OpenAI API key")
                
                st.markdown("### Supported Files")
                st.markdown("""
                **Images:**
                - Screenshots of Amazon pages
                - Return reason reports
                - Customer message screenshots
                
                **Documents:**
                - PDF return reports
                - Customer communication PDFs
                - Seller central exports
                """)
                
        except Exception as e:
            logger.error(f"Error rendering document upload: {str(e)}")
            st.error("Document upload error")
    
    def render_text_analysis_dashboard(self):
        """Render main text analysis dashboard"""
        try:
            st.markdown("## 🔍 Customer Feedback Text Analysis")
            
            # Check if we have analysis results
            if not st.session_state.get('text_analysis_complete', False):
                self._render_no_analysis_state()
                return
            
            text_results = st.session_state.get('text_analysis_results', {})
            if not text_results:
                self._render_no_analysis_state()
                return
            
            # Product selector
            selected_asin = self._render_product_selector(text_results)
            
            if selected_asin and selected_asin in text_results:
                analysis_result = text_results[selected_asin]
                
                # Main analysis sections
                analysis_tabs = st.tabs([
                    "📊 Quality Categories", 
                    "📈 Temporal Trends", 
                    "⚠️ Risk Assessment",
                    "🎯 CAPA Recommendations",
                    "💬 Customer Feedback"
                ])
                
                with analysis_tabs[0]:
                    self._render_quality_categories(analysis_result)
                
                with analysis_tabs[1]:
                    self._render_temporal_analysis(analysis_result)
                
                with analysis_tabs[2]:
                    self._render_risk_assessment(analysis_result)
                
                with analysis_tabs[3]:
                    self._render_capa_recommendations(analysis_result)
                
                with analysis_tabs[4]:
                    self._render_customer_feedback_details(selected_asin)
                    
        except Exception as e:
            logger.error(f"Error rendering text analysis dashboard: {str(e)}")
            st.error("Text analysis dashboard error")
    
    def _render_no_analysis_state(self):
        """Render state when no text analysis is available"""
        st.info("📋 No text analysis results available yet.")
        st.markdown("""
        **To get started:**
        1. Upload product data in the **Data Import** tab
        2. Add customer feedback (reviews, return reasons)
        3. Click **Process Data & Analyze Text** to run analysis
        4. View quality insights and CAPA recommendations here
        """)
        
        # Show example data option
        if st.button("📊 Load Example Customer Feedback Data", type="primary"):
            st.session_state['load_example'] = True
            st.rerun()
    
    def _render_product_selector(self, text_results: Dict[str, Any]) -> Optional[str]:
        """Render product selector for analysis results"""
        try:
            if not text_results:
                return None
            
            # Get product info
            products = st.session_state.get('processed_data', {}).get('products', [])
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                product_options = []
                for asin in text_results.keys():
                    product = next((p for p in products if p['asin'] == asin), None)
                    if product:
                        name = product['name']
                        feedback_count = text_results[asin].total_feedback_items
                        product_options.append((asin, f"{name} ({asin}) - {feedback_count} items"))
                    else:
                        feedback_count = text_results[asin].total_feedback_items
                        product_options.append((asin, f"Product {asin} - {feedback_count} items"))
                
                if product_options:
                    selected_index = st.selectbox(
                        "Select product for detailed analysis:",
                        range(len(product_options)),
                        format_func=lambda i: product_options[i][1],
                        key="text_analysis_product_selector"
                    )
                    
                    return product_options[selected_index][0]
            
            with col2:
                st.metric("Products Analyzed", len(text_results))
            
            with col3:
                total_feedback = sum(result.total_feedback_items for result in text_results.values())
                st.metric("Total Feedback Items", total_feedback)
            
            return None
            
        except Exception as e:
            logger.error(f"Error rendering product selector: {str(e)}")
            return None
    
    def _render_quality_categories(self, analysis_result):
        """Render quality category analysis"""
        try:
            st.markdown("### Medical Device Quality Categories")
            
            category_analysis = analysis_result.category_analysis
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            summary = category_analysis.get('summary', {})
            
            with col1:
                st.metric("Total Items", summary.get('total_feedback_items', 0))
            
            with col2:
                st.metric("Categorized", summary.get('total_categorized', 0))
            
            with col3:
                rate = summary.get('categorization_rate', 0)
                st.metric("Categorization Rate", f"{rate}%")
            
            with col4:
                # Count high-severity categories
                high_severity_count = sum(1 for cat_id, cat_data in category_analysis.items() 
                                        if cat_id != 'summary' and 
                                        cat_data.get('severity') == 'high' and 
                                        cat_data.get('count', 0) > 0)
                st.metric("High-Risk Categories", high_severity_count)
            
            # Category breakdown chart
            chart = self.ui.create_quality_category_chart(category_analysis)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Detailed category breakdown
            st.markdown("### Category Details")
            
            for cat_id, cat_data in category_analysis.items():
                if cat_id == 'summary':
                    continue
                
                count = cat_data.get('count', 0)
                if count == 0:
                    continue
                
                # Category card
                severity = cat_data.get('severity', 'medium')
                color = QUALITY_COLORS.get(cat_id, COLORS['primary'])
                
                with st.container():
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; 
                                background-color: white; border-radius: 0 8px 8px 0;">
                        <h4 style="margin: 0; color: {color};">{cat_data['name']}</h4>
                        <p><strong>{count} feedback items</strong> ({cat_data.get('percentage', 0)}% of total)</p>
                        <p><small><strong>Severity:</strong> {severity.title()} | 
                        <strong>ISO Reference:</strong> {cat_data.get('iso_reference', 'N/A')}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show common patterns
                    patterns = cat_data.get('common_patterns', [])
                    if patterns:
                        with st.expander(f"Common Patterns in {cat_data['name']}"):
                            for pattern in patterns:
                                st.markdown(f"• {pattern}")
                    
                    # CAPA indicator
                    if cat_data.get('requires_capa', False):
                        st.warning(f"⚠️ Category requires CAPA attention due to {severity} severity level")
                        
        except Exception as e:
            logger.error(f"Error rendering quality categories: {str(e)}")
            st.error("Quality categories error")
    
    def _render_temporal_analysis(self, analysis_result):
        """Render temporal trend analysis"""
        try:
            st.markdown("### Temporal Feedback Analysis")
            
            trend_analysis = analysis_result.trend_analysis
            
            # Trend overview
            col1, col2 = st.columns(2)
            
            with col1:
                span_days = trend_analysis.get('analysis_span_days', 0)
                st.metric("Analysis Span", f"{span_days} days")
                
                patterns = trend_analysis.get('trend_patterns', {})
                pattern = patterns.get('pattern', 'unknown')
                description = patterns.get('description', 'No pattern identified')
                
                st.markdown(f"**Trend Pattern:** {pattern.title()}")
                st.info(description)
            
            with col2:
                recent_avg = patterns.get('recent_average', 0)
                earlier_avg = patterns.get('earlier_average', 0)
                
                st.metric("Recent Period Avg", f"{recent_avg:.1f} items/week")
                st.metric("Earlier Period Avg", f"{earlier_avg:.1f} items/week")
            
            # Temporal trend chart
            chart = self.ui.create_temporal_trend_chart(trend_analysis)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("📊 Temporal chart requires more data points over time")
            
            # Weekly breakdown
            weekly_trends = trend_analysis.get('weekly_trends', {})
            if weekly_trends:
                st.markdown("### Weekly Feedback Breakdown")
                
                weekly_df = pd.DataFrame([
                    {'Week': week, 'Feedback Count': count} 
                    for week, count in weekly_trends.items()
                ])
                
                st.dataframe(weekly_df, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error rendering temporal analysis: {str(e)}")
            st.error("Temporal analysis error")
    
    def _render_risk_assessment(self, analysis_result):
        """Render risk assessment dashboard"""
        try:
            st.markdown("### Risk Assessment & Safety Monitoring")
            
            risk_level = analysis_result.risk_level
            risk_factors = analysis_result.risk_factors
            quality_assessment = analysis_result.quality_assessment
            
            # Risk overview
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Risk gauge
                risk_score = 0  # Would extract from analysis_result if available
                for factor in risk_factors:
                    if 'safety' in factor.lower():
                        risk_score += 15
                    elif 'quality' in factor.lower():
                        risk_score += 10
                    else:
                        risk_score += 5
                
                gauge = self.ui.create_risk_assessment_gauge(risk_level, risk_score)
                if gauge:
                    st.plotly_chart(gauge, use_container_width=True)
                else:
                    # Fallback display
                    risk_color = RISK_COLORS.get(risk_level, COLORS['warning'])
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background-color: white; 
                                border-radius: 8px; border: 2px solid {risk_color};">
                        <h2 style="color: {risk_color}; margin: 0;">{risk_level} Risk</h2>
                        <p style="margin: 0.5rem 0;">Score: {risk_score:.1f}/50</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Risk factors
                st.markdown("#### Risk Factors Identified")
                
                if risk_factors:
                    for i, factor in enumerate(risk_factors, 1):
                        if 'safety' in factor.lower():
                            st.error(f"{i}. 🚨 {factor}")
                        elif 'quality' in factor.lower():
                            st.warning(f"{i}. ⚠️ {factor}")
                        else:
                            st.info(f"{i}. ℹ️ {factor}")
                else:
                    st.success("✅ No significant risk factors identified")
                
                # Quality metrics
                if quality_assessment:
                    st.markdown("#### Quality Indicators")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        quality_score = quality_assessment.get('quality_score', 0)
                        st.metric("Quality Score", f"{quality_score}%")
                        
                        positive_ratio = quality_assessment.get('positive_ratio', 0)
                        st.metric("Positive Feedback", f"{positive_ratio}%")
                    
                    with col_b:
                        negative_ratio = quality_assessment.get('negative_ratio', 0)
                        st.metric("Negative Feedback", f"{negative_ratio}%")
                        
                        quality_level = quality_assessment.get('quality_level', 'Unknown')
                        st.metric("Quality Level", quality_level)
            
            # Immediate action requirements
            if quality_assessment.get('requires_immediate_action', False):
                st.error("🚨 **IMMEDIATE ACTION REQUIRED** - High-risk issues detected in customer feedback")
                
                high_risk_categories = quality_assessment.get('high_risk_categories', [])
                if high_risk_categories:
                    st.markdown("**High-Risk Categories Requiring Attention:**")
                    for category in high_risk_categories:
                        st.markdown(f"• {category}")
                        
        except Exception as e:
            logger.error(f"Error rendering risk assessment: {str(e)}")
            st.error("Risk assessment error")
    
    def _render_capa_recommendations(self, analysis_result):
        """Render CAPA (Corrective and Preventive Action) recommendations"""
        try:
            st.markdown("### CAPA Recommendations")
            st.caption("Corrective and Preventive Actions based on customer feedback analysis")
            
            capa_recommendations = analysis_result.capa_recommendations
            
            if not capa_recommendations:
                st.success("✅ No CAPA actions required at this time")
                st.info("Continue monitoring customer feedback for emerging quality issues")
                return
            
            # CAPA summary
            col1, col2, col3 = st.columns(3)
            
            critical_count = len([r for r in capa_recommendations if r.get('priority') == 'Critical'])
            high_count = len([r for r in capa_recommendations if r.get('priority') == 'High'])
            total_count = len(capa_recommendations)
            
            with col1:
                st.metric("Total CAPA Items", total_count)
            
            with col2:
                st.metric("Critical Priority", critical_count)
            
            with col3:
                st.metric("High Priority", high_count)
            
            # CAPA recommendations by priority
            priorities = ['Critical', 'High', 'Medium', 'Low']
            
            for priority in priorities:
                priority_capas = [r for r in capa_recommendations if r.get('priority') == priority]
                
                if not priority_capas:
                    continue
                
                # Priority section header
                priority_color = CAPA_COLORS.get(priority, COLORS['primary'])
                st.markdown(f"""
                <h4 style="color: {priority_color}; border-bottom: 2px solid {priority_color}; 
                           padding-bottom: 0.5rem;">
                    {priority} Priority CAPA Items ({len(priority_capas)})
                </h4>
                """, unsafe_allow_html=True)
                
                # Render each CAPA
                for i, capa in enumerate(priority_capas, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="capa-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h5 style="margin: 0; color: {priority_color};">
                                    CAPA-{priority[0]}{i:02d}: {capa.get('category', 'General')}
                                </h5>
                                <span class="risk-indicator" style="background-color: {priority_color};">
                                    {priority}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # CAPA details
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**Issue:** {capa.get('issue', 'No issue specified')}")
                            st.markdown(f"**Corrective Action:** {capa.get('corrective_action', 'No action specified')}")
                            st.markdown(f"**Preventive Action:** {capa.get('preventive_action', 'No action specified')}")
                        
                        with col_b:
                            st.markdown(f"**Timeline:** {capa.get('timeline', 'Not specified')}")
                            st.markdown(f"**Responsibility:** {capa.get('responsibility', 'Not assigned')}")
                            st.markdown(f"**Success Metric:** {capa.get('success_metric', 'Not defined')}")
                        
                        # ISO reference
                        iso_ref = capa.get('iso_reference', '')
                        if iso_ref:
                            st.caption(f"🔗 **ISO Reference:** {iso_ref}")
                        
                        st.markdown("---")
            
            # Export CAPA recommendations
            if st.button("📥 Export CAPA Report", type="secondary"):
                self._export_capa_report(capa_recommendations, analysis_result.product_name)
                
        except Exception as e:
            logger.error(f"Error rendering CAPA recommendations: {str(e)}")
            st.error("CAPA recommendations error")
    
    def _render_customer_feedback_details(self, selected_asin: str):
        """Render detailed customer feedback with categorization"""
        try:
            st.markdown("### Customer Feedback Details")
            
            # Get feedback data
            customer_feedback = st.session_state.get('processed_data', {}).get('customer_feedback', {})
            feedback_items = customer_feedback.get(selected_asin, [])
            
            if not feedback_items:
                st.info("No customer feedback items found for this product")
                return
            
            # Feedback filtering
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feedback_types = ['all'] + list(set(item.get('type', 'unknown') for item in feedback_items))
                selected_type = st.selectbox("Filter by Type", feedback_types, format_func=lambda x: x.replace('_', ' ').title())
            
            with col2:
                # Category filter (based on detected categories)
                categories = ['all']
                for item in feedback_items:
                    categories.extend(item.get('category_flags', []))
                unique_categories = ['all'] + list(set(categories))
                selected_category = st.selectbox("Filter by Category", unique_categories, format_func=lambda x: x.replace('_', ' ').title())
            
            with col3:
                # Sort options
                sort_options = ['date_desc', 'date_asc', 'rating_desc', 'rating_asc']
                selected_sort = st.selectbox("Sort by", sort_options, format_func=lambda x: {
                    'date_desc': 'Newest First',
                    'date_asc': 'Oldest First', 
                    'rating_desc': 'Highest Rating',
                    'rating_asc': 'Lowest Rating'
                }[x])
            
            # Filter and sort feedback
            filtered_items = feedback_items
            
            if selected_type != 'all':
                filtered_items = [item for item in filtered_items if item.get('type') == selected_type]
            
            if selected_category != 'all':
                filtered_items = [item for item in filtered_items if selected_category in item.get('category_flags', [])]
            
            # Sort items
            if selected_sort == 'date_desc':
                filtered_items.sort(key=lambda x: x.get('date', ''), reverse=True)
            elif selected_sort == 'date_asc':
                filtered_items.sort(key=lambda x: x.get('date', ''))
            elif selected_sort == 'rating_desc':
                filtered_items.sort(key=lambda x: x.get('rating', 0), reverse=True)
            elif selected_sort == 'rating_asc':
                filtered_items.sort(key=lambda x: x.get('rating', 0))
            
            # Display filtered items
            st.markdown(f"**Showing {len(filtered_items)} of {len(feedback_items)} feedback items**")
            
            for i, item in enumerate(filtered_items):
                with st.container():
                    # Item header
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    
                    with col_a:
                        item_type = item.get('type', 'unknown').replace('_', ' ').title()
                        st.markdown(f"**{item_type}** #{i+1}")
                    
                    with col_b:
                        item_date = item.get('date', 'Unknown date')
                        st.markdown(f"**Date:** {item_date}")
                    
                    with col_c:
                        rating = item.get('rating')
                        if rating:
                            stars = "⭐" * rating
                            st.markdown(f"**Rating:** {stars} ({rating}/5)")
                    
                    # Feedback text
                    text = item.get('text', '')
                    st.markdown(f"*\"{text}\"*")
                    
                    # Category flags
                    category_flags = item.get('category_flags', [])
                    if category_flags:
                        st.markdown("**Detected Categories:**")
                        for flag in category_flags:
                            if flag == 'positive_feedback':
                                st.success(f"✅ {flag.replace('_', ' ').title()}")
                            elif 'safety' in flag:
                                st.error(f"🚨 {flag.replace('_', ' ').title()}")
                            else:
                                color = QUALITY_COLORS.get(flag, COLORS['primary'])
                                st.markdown(f"""
                                <span style="background-color: {color}; color: white; padding: 0.25rem 0.5rem; 
                                            border-radius: 4px; font-size: 0.875rem;">
                                    {flag.replace('_', ' ').title()}
                                </span>
                                """, unsafe_allow_html=True)
                    
                    # Source
                    source = item.get('source', 'unknown')
                    st.caption(f"Source: {source.replace('_', ' ').title()}")
                    
                    st.markdown("---")
                    
        except Exception as e:
            logger.error(f"Error rendering customer feedback details: {str(e)}")
            st.error("Customer feedback details error")
    
    def _export_capa_report(self, capa_recommendations: List[Dict], product_name: str):
        """Export CAPA recommendations as downloadable report"""
        try:
            # Create CAPA report text
            report_lines = [
                "# CAPA REPORT - CUSTOMER FEEDBACK ANALYSIS",
                f"**Product:** {product_name}",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total CAPA Items:** {len(capa_recommendations)}",
                "",
                "## CORRECTIVE AND PREVENTIVE ACTION PLAN",
                ""
            ]
            
            # Group by priority
            priorities = ['Critical', 'High', 'Medium', 'Low']
            
            for priority in priorities:
                priority_capas = [r for r in capa_recommendations if r.get('priority') == priority]
                
                if not priority_capas:
                    continue
                
                report_lines.append(f"### {priority.upper()} PRIORITY ({len(priority_capas)} items)")
                report_lines.append("")
                
                for i, capa in enumerate(priority_capas, 1):
                    report_lines.extend([
                        f"#### CAPA-{priority[0]}{i:02d}: {capa.get('category', 'General')}",
                        f"**Issue:** {capa.get('issue', 'No issue specified')}",
                        f"**Corrective Action:** {capa.get('corrective_action', 'No action specified')}",
                        f"**Preventive Action:** {capa.get('preventive_action', 'No action specified')}",
                        f"**Timeline:** {capa.get('timeline', 'Not specified')}",
                        f"**Responsibility:** {capa.get('responsibility', 'Not assigned')}",
                        f"**Success Metric:** {capa.get('success_metric', 'Not defined')}",
                        f"**ISO Reference:** {capa.get('iso_reference', 'Not specified')}",
                        ""
                    ])
            
            report_text = "\n".join(report_lines)
            
            # Create download
            st.download_button(
                label="📥 Download CAPA Report",
                data=report_text,
                file_name=f"CAPA_Report_{product_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            logger.error(f"Error exporting CAPA report: {str(e)}")
            st.error("Failed to export CAPA report")
    
    def render_ai_chat_tab(self):
        """Render AI chat tab for quality management consultation"""
        try:
            from ai_chat import AIChatInterface
            chat_interface = AIChatInterface()
            chat_interface.render_chat_interface()
        except ImportError:
            st.error("AI Chat module not available. Please ensure ai_chat.py is in your project directory.")
            st.info("The AI Chat feature provides quality management consultation and listing optimization advice.")
        except Exception as e:
            logger.error(f"AI chat error: {str(e)}")
            st.error(f"AI Chat error: {str(e)}")

class ProfessionalDashboard:
    """Main dashboard orchestrator for text analysis and quality management"""
    
    def __init__(self):
        try:
            self.renderer = TextAnalysisDashboardRenderer()
            self.ui = UIComponents()
            logger.info("Text analysis dashboard components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing dashboard components: {str(e)}")
            raise
    
    def render_text_analysis_dashboard(self):
        """Render the main text analysis dashboard"""
        try:
            # Apply theme
            try:
                self.ui.set_text_analysis_theme()
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
            
            # Date filtering controls (prominent placement)
            self.renderer.date_filter.render_date_filter_controls()
            
            # Main content tabs
            main_tabs = st.tabs([
                "📁 Data Import", 
                "🔍 Text Analysis", 
                "🤖 AI Chat",
                "📋 Export & Reports"
            ])
            
            with main_tabs[0]:
                self.renderer.render_upload_dashboard()
            
            with main_tabs[1]:
                self.renderer.render_text_analysis_dashboard()
            
            with main_tabs[2]:
                self.renderer.render_ai_chat_tab()
            
            with main_tabs[3]:
                self._render_export_dashboard()
                
        except Exception as e:
            logger.error(f"Error rendering text analysis dashboard: {str(e)}")
            st.error(f"Dashboard rendering error: {str(e)}")
            
            # Fallback basic interface
            st.title("🔍 Medical Device Customer Feedback Analyzer")
            st.error("⚠️ Dashboard rendering failed. Using basic interface.")
    
    def _render_export_dashboard(self):
        """Render export and reporting dashboard"""
        try:
            st.markdown("## 📋 Export & Reporting")
            
            if not st.session_state.get('text_analysis_complete', False):
                st.info("Complete text analysis first to access export options")
                return
            
            export_tabs = st.tabs(["📊 Analysis Reports", "📋 CAPA Export", "💾 Raw Data"])
            
            with export_tabs[0]:
                self._render_analysis_reports_export()
            
            with export_tabs[1]:
                self._render_capa_export()
            
            with export_tabs[2]:
                self._render_raw_data_export()
                
        except Exception as e:
            logger.error(f"Error rendering export dashboard: {str(e)}")
            st.error("Export dashboard error")
    
    def _render_analysis_reports_export(self):
        """Render analysis reports export"""
        try:
            st.markdown("### Text Analysis Reports")
            
            text_results = st.session_state.get('text_analysis_results', {})
            if not text_results:
                st.warning("No text analysis results available for export")
                return
            
            # Report options
            report_options = st.multiselect(
                "Select report components:",
                ["Quality Category Analysis", "Risk Assessment", "CAPA Recommendations", 
                 "Temporal Trends", "Customer Feedback Summary"],
                default=["Quality Category Analysis", "Risk Assessment", "CAPA Recommendations"]
            )
            
            # Product selection for export
            products = list(text_results.keys())
            selected_products = st.multiselect(
                "Select products to include:",
                products,
                default=products[:3] if len(products) <= 3 else products[:1]
            )
            
            if st.button("📊 Generate Analysis Report", type="primary"):
                if selected_products and report_options:
                    report_content = self._generate_analysis_report(selected_products, report_options, text_results)
                    
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=report_content,
                        file_name=f"Text_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("Please select products and report components")
                    
        except Exception as e:
            logger.error(f"Error rendering analysis reports export: {str(e)}")
    
    def _render_capa_export(self):
        """Render CAPA export interface"""
        try:
            st.markdown("### CAPA Export & Tracking")
            
            text_results = st.session_state.get('text_analysis_results', {})
            if not text_results:
                st.warning("No CAPA recommendations available for export")
                return
            
            # Collect all CAPA recommendations
            all_capas = []
            for asin, result in text_results.items():
                for capa in result.capa_recommendations:
                    capa_with_product = capa.copy()
                    capa_with_product['product_asin'] = asin
                    capa_with_product['product_name'] = result.product_name
                    all_capas.append(capa_with_product)
            
            if not all_capas:
                st.success("✅ No CAPA items identified - all products performing well")
                return
            
            # CAPA summary
            col1, col2, col3 = st.columns(3)
            
            critical_capas = [c for c in all_capas if c.get('priority') == 'Critical']
            high_capas = [c for c in all_capas if c.get('priority') == 'High']
            
            with col1:
                st.metric("Total CAPA Items", len(all_capas))
            
            with col2:
                st.metric("Critical Priority", len(critical_capas))
            
            with col3:
                st.metric("High Priority", len(high_capas))
            
            # Export format
            export_format = st.selectbox("Export Format", ["Excel Spreadsheet", "CSV Data", "Markdown Report"])
            
            if st.button("📥 Export CAPA Tracking Sheet", type="primary"):
                if export_format == "Excel Spreadsheet":
                    excel_data = self._create_capa_excel(all_capas)
                    st.download_button(
                        label="📥 Download CAPA Tracking Sheet",
                        data=excel_data,
                        file_name=f"CAPA_Tracking_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "CSV Data":
                    csv_data = self._create_capa_csv(all_capas)
                    st.download_button(
                        label="📥 Download CAPA CSV",
                        data=csv_data,
                        file_name=f"CAPA_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:  # Markdown Report
                    md_data = self._create_capa_markdown(all_capas)
                    st.download_button(
                        label="📥 Download CAPA Report",
                        data=md_data,
                        file_name=f"CAPA_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
                    
        except Exception as e:
            logger.error(f"Error rendering CAPA export: {str(e)}")
    
    def _render_raw_data_export(self):
        """Render raw data export interface"""
        try:
            st.markdown("### Raw Data Export")
            
            # Export options
            data_options = st.multiselect(
                "Select data to export:",
                ["Customer Feedback Items", "Text Analysis Results", "Quality Categories", "Risk Assessments"],
                default=["Customer Feedback Items", "Text Analysis Results"]
            )
            
            export_format = st.selectbox("Format", ["CSV", "JSON", "Excel"])
            
            if st.button("💾 Export Raw Data", type="primary"):
                if data_options:
                    if export_format == "CSV":
                        data = self._create_raw_csv(data_options)
                        st.download_button(
                            label="📥 Download CSV",
                            data=data,
                            file_name=f"Raw_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        data = self._create_raw_json(data_options)
                        st.download_button(
                            label="📥 Download JSON",
                            data=data,
                            file_name=f"Raw_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
                else:
                    st.error("Please select data to export")
                    
        except Exception as e:
            logger.error(f"Error rendering raw data export: {str(e)}")
    
    def _generate_analysis_report(self, selected_products: List[str], report_options: List[str], 
                                text_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        try:
            report_lines = [
                "# MEDICAL DEVICE CUSTOMER FEEDBACK ANALYSIS REPORT",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Products Analyzed:** {len(selected_products)}",
                f"**Report Components:** {', '.join(report_options)}",
                ""
            ]
            
            for asin in selected_products:
                if asin in text_results:
                    result = text_results[asin]
                    
                    report_lines.extend([
                        f"## {result.product_name} ({asin})",
                        f"**Analysis Period:** {result.analysis_period}",
                        f"**Feedback Items Analyzed:** {result.total_feedback_items}",
                        f"**Risk Level:** {result.risk_level}",
                        ""
                    ])
                    
                    if "Quality Category Analysis" in report_options:
                        report_lines.extend(self._add_quality_analysis_section(result))
                    
                    if "Risk Assessment" in report_options:
                        report_lines.extend(self._add_risk_assessment_section(result))
                    
                    if "CAPA Recommendations" in report_options:
                        report_lines.extend(self._add_capa_section(result))
                    
                    report_lines.append("---")
                    report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            return "Error generating report"
    
    def _add_quality_analysis_section(self, result) -> List[str]:
        """Add quality analysis section to report"""
        lines = ["### Quality Category Analysis", ""]
        
        category_analysis = result.category_analysis
        for cat_id, cat_data in category_analysis.items():
            if cat_id == 'summary':
                continue
            
            count = cat_data.get('count', 0)
            if count > 0:
                lines.append(f"**{cat_data['name']}:** {count} items ({cat_data.get('percentage', 0)}%)")
                lines.append(f"- Severity: {cat_data.get('severity', 'Unknown')}")
                lines.append(f"- ISO Reference: {cat_data.get('iso_reference', 'N/A')}")
                lines.append("")
        
        return lines
    
    def _add_risk_assessment_section(self, result) -> List[str]:
        """Add risk assessment section to report"""
        lines = ["### Risk Assessment", ""]
        lines.append(f"**Risk Level:** {result.risk_level}")
        lines.append("")
        
        if result.risk_factors:
            lines.append("**Risk Factors:**")
            for factor in result.risk_factors:
                lines.append(f"- {factor}")
            lines.append("")
        
        return lines
    
    def _add_capa_section(self, result) -> List[str]:
        """Add CAPA section to report"""
        lines = ["### CAPA Recommendations", ""]
        
        if not result.capa_recommendations:
            lines.append("No CAPA actions required.")
            lines.append("")
            return lines
        
        for i, capa in enumerate(result.capa_recommendations, 1):
            lines.extend([
                f"**CAPA {i}: {capa.get('category', 'General')}**",
                f"- Priority: {capa.get('priority', 'Unknown')}",
                f"- Issue: {capa.get('issue', 'No issue specified')}",
                f"- Corrective Action: {capa.get('corrective_action', 'No action specified')}",
                f"- Timeline: {capa.get('timeline', 'Not specified')}",
                ""
            ])
        
        return lines
    
    def _create_capa_excel(self, all_capas: List[Dict]) -> bytes:
        """Create Excel CAPA tracking sheet"""
        try:
            # Create DataFrame
            capa_data = []
            for i, capa in enumerate(all_capas, 1):
                capa_data.append({
                    'CAPA_ID': f"CAPA-{i:03d}",
                    'Product_Name': capa.get('product_name', 'Unknown'),
                    'Product_ASIN': capa.get('product_asin', 'Unknown'),
                    'Priority': capa.get('priority', 'Medium'),
                    'Category': capa.get('category', 'General'),
                    'Issue': capa.get('issue', 'No issue specified'),
                    'Corrective_Action': capa.get('corrective_action', 'No action specified'),
                    'Preventive_Action': capa.get('preventive_action', 'No action specified'),
                    'Timeline': capa.get('timeline', 'Not specified'),
                    'Responsibility': capa.get('responsibility', 'Not assigned'),
                    'Success_Metric': capa.get('success_metric', 'Not defined'),
                    'ISO_Reference': capa.get('iso_reference', 'Not specified'),
                    'Status': 'Open',
                    'Date_Created': datetime.now().strftime('%Y-%m-%d'),
                    'Date_Due': '',
                    'Date_Completed': '',
                    'Notes': ''
                })
            
            df = pd.DataFrame(capa_data)
            
            # Create Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='CAPA Tracking', index=False)
                
                # Add formatting
                workbook = writer.book
                worksheet = writer.sheets['CAPA Tracking']
                
                # Header format
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#1E40AF',
                    'font_color': 'white',
                    'border': 1
                })
                
                # Apply header formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Set column widths
                worksheet.set_column('A:A', 12)  # CAPA_ID
                worksheet.set_column('B:C', 20)  # Product info
                worksheet.set_column('D:D', 10)  # Priority
                worksheet.set_column('E:E', 15)  # Category
                worksheet.set_column('F:L', 30)  # Actions and details
                worksheet.set_column('M:Q', 12)  # Status and dates
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating CAPA Excel: {str(e)}")
            return b""
    
    def _create_capa_csv(self, all_capas: List[Dict]) -> str:
        """Create CSV CAPA data"""
        try:
            capa_data = []
            for i, capa in enumerate(all_capas, 1):
                capa_data.append({
                    'CAPA_ID': f"CAPA-{i:03d}",
                    'Product_Name': capa.get('product_name', 'Unknown'),
                    'Product_ASIN': capa.get('product_asin', 'Unknown'),
                    'Priority': capa.get('priority', 'Medium'),
                    'Category': capa.get('category', 'General'),
                    'Issue': capa.get('issue', 'No issue specified'),
                    'Corrective_Action': capa.get('corrective_action', 'No action specified'),
                    'Timeline': capa.get('timeline', 'Not specified'),
                    'Responsibility': capa.get('responsibility', 'Not assigned')
                })
            
            df = pd.DataFrame(capa_data)
            return df.to_csv(index=False)
            
        except Exception as e:
            logger.error(f"Error creating CAPA CSV: {str(e)}")
            return ""
    
    def _create_capa_markdown(self, all_capas: List[Dict]) -> str:
        """Create markdown CAPA report"""
        try:
            lines = [
                "# CAPA TRACKING REPORT",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total CAPA Items:** {len(all_capas)}",
                ""
            ]
            
            for i, capa in enumerate(all_capas, 1):
                lines.extend([
                    f"## CAPA-{i:03d}: {capa.get('category', 'General')}",
                    f"**Product:** {capa.get('product_name', 'Unknown')} ({capa.get('product_asin', 'Unknown')})",
                    f"**Priority:** {capa.get('priority', 'Medium')}",
                    f"**Issue:** {capa.get('issue', 'No issue specified')}",
                    f"**Corrective Action:** {capa.get('corrective_action', 'No action specified')}",
                    f"**Timeline:** {capa.get('timeline', 'Not specified')}",
                    f"**Responsibility:** {capa.get('responsibility', 'Not assigned')}",
                    ""
                ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error creating CAPA markdown: {str(e)}")
            return "Error creating CAPA report"
    
    def _create_raw_csv(self, data_options: List[str]) -> str:
        """Create raw data CSV export"""
        try:
            if "Customer Feedback Items" in data_options:
                feedback_data = []
                customer_feedback = st.session_state.get('processed_data', {}).get('customer_feedback', {})
                
                for asin, items in customer_feedback.items():
                    for item in items:
                        feedback_data.append({
                            'ASIN': asin,
                            'Product_Name': item.get('product_name', 'Unknown'),
                            'Feedback_Type': item.get('type', 'unknown'),
                            'Text': item.get('text', ''),
                            'Rating': item.get('rating', ''),
                            'Date': item.get('date', ''),
                            'Source': item.get('source', 'unknown')
                        })
                
                df = pd.DataFrame(feedback_data)
                return df.to_csv(index=False)
            
            return "No data selected for export"
            
        except Exception as e:
            logger.error(f"Error creating raw CSV: {str(e)}")
            return "Error creating CSV export"
    
    def _create_raw_json(self, data_options: List[str]) -> str:
        """Create raw data JSON export"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'data_options': data_options
            }
            
            if "Customer Feedback Items" in data_options:
                export_data['customer_feedback'] = st.session_state.get('processed_data', {}).get('customer_feedback', {})
            
            if "Text Analysis Results" in data_options:
                # Convert text analysis results to serializable format
                text_results = st.session_state.get('text_analysis_results', {})
                serializable_results = {}
                
                for asin, result in text_results.items():
                    serializable_results[asin] = {
                        'product_name': result.product_name,
                        'analysis_period': result.analysis_period,
                        'total_feedback_items': result.total_feedback_items,
                        'risk_level': result.risk_level,
                        'risk_factors': result.risk_factors,
                        'category_analysis': result.category_analysis,
                        'quality_assessment': result.quality_assessment
                    }
                
                export_data['text_analysis_results'] = serializable_results
            
            return json.dumps(export_data, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error creating raw JSON: {str(e)}")
            return json.dumps({"error": "Failed to create JSON export"})

# Export main classes
__all__ = ['ProfessionalDashboard', 'TextAnalysisDashboardRenderer', 'UIComponents', 'DateFilterRenderer']
