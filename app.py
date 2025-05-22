"""
Amazon Medical Device Customer Feedback Analyzer - FIXED Main Application

**STABLE & ACCURATE VERSION**

Primary Focus: Upload Amazon review data → AI-powered analysis → Actionable insights
- Helium 10 review export optimization
- Medical device quality categorization  
- AI-enhanced sentiment and risk analysis
- CAPA recommendations for quality management

Author: Assistant
Version: 4.0 - Production Stable
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import traceback
import io
import re
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safe imports with graceful degradation
def safe_import(module_name, package=None):
    """Safely import modules with fallback"""
    try:
        if package:
            module = __import__(package, fromlist=[module_name])
            return getattr(module, module_name), True
        else:
            return __import__(module_name), True
    except ImportError as e:
        logger.warning(f"Module {module_name} not available: {str(e)}")
        return None, False

# Try importing custom modules
upload_handler, upload_available = safe_import('UploadHandler', 'upload_handler')
text_engine, engine_available = safe_import('TextAnalysisEngine', 'text_analysis_engine')
ai_analyzer, ai_available = safe_import('EnhancedAIAnalyzer', 'enhanced_ai_analysis')

# Application configuration
APP_CONFIG = {
    'title': 'Medical Device Customer Feedback Analyzer',
    'version': '4.0',
    'description': 'AI-powered analysis of Amazon review data for medical devices',
    'max_file_size_mb': 50,
    'supported_formats': ['.csv', '.xlsx'],
    'session_timeout_hours': 4
}

# Example data for demonstration
EXAMPLE_REVIEW_DATA = [
    {
        'Date': 'January 15, 2024',
        'Title': 'Great mobility aid but assembly was difficult',
        'Body': 'This rollator is very stable and helps me walk confidently. However, the assembly instructions were confusing and some screws were missing.',
        'Rating': 4,
        'Author': 'Sarah M.',
        'Verified': 'Verified Purchase'
    },
    {
        'Date': 'January 10, 2024', 
        'Title': 'Broke after one week',
        'Body': 'The wheel came off after just one week of light use. Very disappointed with the quality. Not safe to use.',
        'Rating': 1,
        'Author': 'John D.',
        'Verified': 'Verified Purchase'
    },
    {
        'Date': 'January 8, 2024',
        'Title': 'Perfect for my needs',
        'Body': 'Excellent quality rollator. Very comfortable seat and easy to maneuver. Highly recommend for anyone needing mobility assistance.',
        'Rating': 5,
        'Author': 'Mary K.',
        'Verified': 'Verified Purchase'
    }
]

def initialize_session_state():
    """Initialize session state with safe defaults"""
    defaults = {
        # Core data
        'uploaded_files': [],
        'processed_data': {},
        'analysis_results': {},
        'current_step': 'upload',
        
        # Processing flags
        'processing_in_progress': False,
        'analysis_complete': False,
        'error_message': None,
        'success_message': None,
        
        # Settings
        'show_debug': False,
        'ai_analysis_enabled': True,
        
        # Module status
        'upload_handler_available': upload_available,
        'text_engine_available': engine_available,
        'ai_analyzer_available': ai_available,
        
        # Session info
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'last_activity': datetime.now(),
        
        # Example data flag
        'using_example_data': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_module_status():
    """Check and display module availability"""
    with st.sidebar:
        st.markdown("### System Status")
        
        # Core modules
        st.markdown("**Core Components:**")
        modules = [
            ('Upload Handler', st.session_state.upload_handler_available),
            ('Text Analysis Engine', st.session_state.text_engine_available),
            ('AI Analyzer', st.session_state.ai_analyzer_available)
        ]
        
        for name, available in modules:
            icon = "✅" if available else "❌"
            st.markdown(f"{icon} {name}")
        
        # API status
        if st.session_state.ai_analyzer_available:
            try:
                import os
                api_key = os.environ.get('OPENAI_API_KEY') or st.secrets.get('openai_api_key', '')
                if api_key:
                    st.success("✅ AI API Key Configured")
                else:
                    st.error("❌ AI API Key Missing")
                    st.caption("Add OPENAI_API_KEY to environment or Streamlit secrets")
            except:
                st.warning("⚠️ Cannot verify API key")

def handle_file_upload():
    """Handle file upload with robust error handling"""
    st.markdown("## 📁 Upload Review Data")
    st.markdown("Upload Amazon review export files (CSV or Excel) for analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    # Example data option
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if uploaded_file is not None:
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            if st.button("🔍 Process File", type="primary", use_container_width=True):
                process_uploaded_file(uploaded_file)
    
    with col2:
        if st.button("📊 Try Example Data", use_container_width=True):
            load_example_data()

def process_uploaded_file(uploaded_file):
    """Process uploaded file with comprehensive error handling"""
    if not st.session_state.upload_handler_available:
        st.error("❌ Upload handler not available. Please check system status.")
        return
    
    try:
        st.session_state.processing_in_progress = True
        st.session_state.error_message = None
        
        with st.spinner("📄 Processing file..."):
            # Read file data
            file_data = uploaded_file.read()
            filename = uploaded_file.name
            
            # Initialize upload handler
            handler = upload_handler()
            
            # Process file
            result = handler.process_structured_file(file_data, filename)
            
            if result.get('success'):
                st.session_state.processed_data = result
                st.session_state.current_step = 'analysis'
                st.session_state.using_example_data = False
                st.success(f"✅ Successfully processed {filename}")
                st.rerun()
            else:
                errors = result.get('errors', ['Unknown error occurred'])
                st.error(f"❌ Processing failed: {'; '.join(map(str, errors))}")
                
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        st.error(f"❌ Error processing file: {str(e)}")
    finally:
        st.session_state.processing_in_progress = False

def load_example_data():
    """Load example data for demonstration"""
    try:
        # Create example dataframe
        df = pd.DataFrame(EXAMPLE_REVIEW_DATA)
        
        # Simulate processed data structure
        example_result = {
            'success': True,
            'export_format': 'helium10_reviews',
            'filename': 'example_reviews.csv',
            'products': [{
                'asin': 'B0EXAMPLE123',
                'name': 'Example Medical Rollator with Seat',
                'category': 'Mobility Aids',
                'total_reviews': len(EXAMPLE_REVIEW_DATA),
                'average_rating': sum(r['Rating'] for r in EXAMPLE_REVIEW_DATA) / len(EXAMPLE_REVIEW_DATA)
            }],
            'customer_feedback': {
                'B0EXAMPLE123': [
                    {
                        'type': 'review',
                        'text': f"{r['Title']} | {r['Body']}",
                        'review_title': r['Title'],
                        'review_body': r['Body'],
                        'rating': r['Rating'],
                        'date': datetime.strptime(r['Date'], '%B %d, %Y').strftime('%Y-%m-%d'),
                        'author': r['Author'],
                        'verified': r['Verified'],
                        'source': 'example_data',
                        'asin': 'B0EXAMPLE123'
                    } for r in EXAMPLE_REVIEW_DATA
                ]
            },
            'processing_summary': {
                'total_rows': len(EXAMPLE_REVIEW_DATA),
                'valid_reviews': len(EXAMPLE_REVIEW_DATA),
                'asin': 'B0EXAMPLE123',
                'product_name': 'Example Medical Rollator with Seat'
            }
        }
        
        st.session_state.processed_data = example_result
        st.session_state.current_step = 'analysis'
        st.session_state.using_example_data = True
        st.success("✅ Example data loaded successfully")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error loading example data: {str(e)}")
        st.error(f"❌ Error loading example data: {str(e)}")

def run_analysis():
    """Run comprehensive analysis on processed data"""
    if not st.session_state.text_engine_available:
        st.error("❌ Text analysis engine not available")
        return
    
    try:
        st.session_state.processing_in_progress = True
        processed_data = st.session_state.processed_data
        
        with st.spinner("🔍 Analyzing customer feedback..."):
            # Initialize text analysis engine
            engine = text_engine()
            
            # Get customer feedback data
            customer_feedback = processed_data.get('customer_feedback', {})
            products = processed_data.get('products', [])
            
            analysis_results = {}
            
            for product in products:
                asin = product['asin']
                feedback_items = customer_feedback.get(asin, [])
                
                if feedback_items:
                    # Run analysis
                    result = engine.analyze_helium10_reviews(
                        feedback_items, 
                        product
                    )
                    
                    if result.get('success'):
                        analysis_results[asin] = result
                    else:
                        st.warning(f"⚠️ Analysis failed for {product['name']}")
            
            if analysis_results:
                st.session_state.analysis_results = analysis_results
                st.session_state.analysis_complete = True
                st.session_state.current_step = 'results'
                st.success(f"✅ Analysis complete for {len(analysis_results)} product(s)")
                st.rerun()
            else:
                st.error("❌ No analysis results generated")
                
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        st.error(f"❌ Analysis failed: {str(e)}")
    finally:
        st.session_state.processing_in_progress = False

def display_analysis_results():
    """Display comprehensive analysis results"""
    st.markdown("## 📊 Analysis Results")
    
    if not st.session_state.analysis_results:
        st.warning("No analysis results available")
        return
    
    # Product selector if multiple products
    analysis_results = st.session_state.analysis_results
    
    if len(analysis_results) > 1:
        selected_asin = st.selectbox(
            "Select Product:",
            list(analysis_results.keys()),
            format_func=lambda x: analysis_results[x].get('product_name', x)
        )
    else:
        selected_asin = list(analysis_results.keys())[0]
    
    result = analysis_results[selected_asin]
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", result.get('total_reviews', 0))
    
    with col2:
        quality_score = result.get('quality_assessment', {}).get('quality_score', 0)
        st.metric("Quality Score", f"{quality_score:.1f}/100")
    
    with col3:
        risk_level = result.get('overall_risk_level', 'Unknown')
        st.metric("Risk Level", risk_level)
    
    with col4:
        capa_count = len(result.get('capa_recommendations', []))
        st.metric("CAPA Items", capa_count)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Quality Categories", 
        "⚠️ Risk Assessment", 
        "🎯 CAPA Recommendations",
        "🤖 AI Insights"
    ])
    
    with tab1:
        display_quality_categories(result)
    
    with tab2:
        display_risk_assessment(result)
    
    with tab3:
        display_capa_recommendations(result)
    
    with tab4:
        display_ai_insights(result)

def display_quality_categories(result):
    """Display quality category analysis"""
    category_analysis = result.get('category_analysis', {})
    
    if not category_analysis:
        st.info("No category analysis available")
        return
    
    st.markdown("### Medical Device Quality Categories")
    
    # Create simple visualization
    categories_with_issues = []
    for cat_id, cat_data in category_analysis.items():
        if cat_data.get('count', 0) > 0:
            categories_with_issues.append({
                'Category': cat_data.get('name', cat_id),
                'Count': cat_data.get('count', 0),
                'Percentage': cat_data.get('percentage', 0),
                'Severity': cat_data.get('severity', 'medium')
            })
    
    if categories_with_issues:
        df_categories = pd.DataFrame(categories_with_issues)
        st.dataframe(df_categories, use_container_width=True)
        
        # Show details for top categories
        st.markdown("### Category Details")
        for cat in categories_with_issues[:3]:  # Top 3
            with st.expander(f"{cat['Category']} ({cat['Count']} issues)"):
                st.markdown(f"**Severity:** {cat['Severity'].title()}")
                st.markdown(f"**Percentage of Reviews:** {cat['Percentage']:.1f}%")
    else:
        st.success("✅ No significant quality issues identified in categories")

def display_risk_assessment(result):
    """Display risk assessment"""
    risk_level = result.get('overall_risk_level', 'Unknown')
    risk_factors = result.get('risk_factors', [])
    
    st.markdown("### Risk Assessment Summary")
    
    # Risk level indicator
    risk_colors = {
        'Critical': '🔴',
        'High': '🟠', 
        'Medium': '🟡',
        'Low': '🟢',
        'Minimal': '🟢'
    }
    
    risk_icon = risk_colors.get(risk_level, '⚪')
    st.markdown(f"**Overall Risk Level:** {risk_icon} {risk_level}")
    
    # Risk factors
    if risk_factors:
        st.markdown("**Identified Risk Factors:**")
        for i, factor in enumerate(risk_factors, 1):
            st.markdown(f"{i}. {factor}")
    else:
        st.success("✅ No significant risk factors identified")
    
    # Quality assessment details
    quality_assessment = result.get('quality_assessment', {})
    if quality_assessment:
        st.markdown("### Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            positive_count = quality_assessment.get('positive_count', 0)
            st.metric("Positive Reviews", positive_count)
            
            avg_rating = quality_assessment.get('average_rating', 0)
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
        
        with col2:
            negative_count = quality_assessment.get('negative_count', 0)
            st.metric("Negative Reviews", negative_count)
            
            quality_level = quality_assessment.get('quality_level', 'Unknown')
            st.metric("Quality Level", quality_level)

def display_capa_recommendations(result):
    """Display CAPA recommendations"""
    capa_recommendations = result.get('capa_recommendations', [])
    
    if not capa_recommendations:
        st.success("✅ No CAPA actions required at this time")
        return
    
    st.markdown("### Corrective and Preventive Action (CAPA) Recommendations")
    
    # Group by priority
    priority_groups = {}
    for capa in capa_recommendations:
        priority = capa.get('priority', 'Medium')
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(capa)
    
    # Display by priority
    priority_order = ['Critical', 'High', 'Medium', 'Low']
    
    for priority in priority_order:
        if priority in priority_groups:
            capas = priority_groups[priority]
            
            st.markdown(f"#### {priority} Priority ({len(capas)} items)")
            
            for i, capa in enumerate(capas, 1):
                with st.expander(f"{priority}-{i}: {capa.get('category', 'General')}"):
                    st.markdown(f"**Issue:** {capa.get('issue_description', 'Not specified')}")
                    st.markdown(f"**Corrective Action:** {capa.get('corrective_action', 'Not specified')}")
                    st.markdown(f"**Timeline:** {capa.get('timeline', 'Not specified')}")
                    st.markdown(f"**Responsibility:** {capa.get('responsibility', 'Not specified')}")

def display_ai_insights(result):
    """Display AI-powered insights"""
    ai_insights = result.get('ai_insights', {})
    ai_available = result.get('ai_analysis_available', False)
    
    if not ai_available:
        st.info("🤖 AI analysis not available. Configure OpenAI API key for enhanced insights.")
        return
    
    if not ai_insights:
        st.warning("No AI insights available")
        return
    
    st.markdown("### AI-Powered Analysis")
    
    # Sentiment analysis
    if 'sentiment_analysis' in ai_insights:
        sentiment = ai_insights['sentiment_analysis']
        st.markdown("#### Sentiment Analysis")
        
        overall_sentiment = sentiment.get('overall_sentiment', 'Not analyzed')
        st.markdown(f"**Overall Sentiment:** {overall_sentiment}")
    
    # Medical device insights
    if 'medical_device_insights' in ai_insights:
        medical_insights = ai_insights['medical_device_insights']
        st.markdown("#### Medical Device Specific Insights")
        
        for insight_type, details in medical_insights.items():
            if details and isinstance(details, list) and len(details) > 0:
                st.markdown(f"**{insight_type.replace('_', ' ').title()}:**")
                for detail in details[:3]:  # Show top 3
                    st.markdown(f"• {detail}")
    
    # Actionable insights
    if 'actionable_insights' in ai_insights:
        actionable = ai_insights['actionable_insights']
        st.markdown("#### Actionable Recommendations")
        
        for insight_type, recommendations in actionable.items():
            if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                st.markdown(f"**{insight_type.replace('_', ' ').title()}:**")
                for rec in recommendations[:2]:  # Show top 2
                    st.markdown(f"• {rec}")

def main():
    """Main application entry point"""
    try:
        # Page config
        st.set_page_config(
            page_title=APP_CONFIG['title'],
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session
        initialize_session_state()
        
        # Header
        st.title("🏥 Medical Device Customer Feedback Analyzer")
        st.markdown(f"**Version {APP_CONFIG['version']}** - AI-powered analysis of Amazon review data")
        
        # Update last activity
        st.session_state.last_activity = datetime.now()
        
        # Sidebar status
        check_module_status()
        
        # Main workflow
        if st.session_state.current_step == 'upload':
            handle_file_upload()
            
        elif st.session_state.current_step == 'analysis':
            st.markdown("## 🔍 Data Processing")
            
            # Show processed data summary
            processed_data = st.session_state.processed_data
            if processed_data.get('success'):
                summary = processed_data.get('processing_summary', {})
                
                st.success("✅ Data processed successfully")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", summary.get('valid_reviews', 0))
                with col2:
                    st.metric("Product", summary.get('product_name', 'Unknown')[:30] + '...')
                with col3:
                    st.metric("ASIN", summary.get('asin', 'Unknown'))
                
                # Analysis button
                if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                    run_analysis()
                
                # Show raw data option
                with st.expander("📄 View Raw Data"):
                    if 'customer_feedback' in processed_data:
                        for asin, feedback in processed_data['customer_feedback'].items():
                            st.markdown(f"**{asin}** ({len(feedback)} items)")
                            df_feedback = pd.DataFrame(feedback)
                            st.dataframe(df_feedback[['date', 'rating', 'author', 'review_title']].head(), use_container_width=True)
            else:
                st.error("❌ Data processing failed")
                if st.button("🔄 Back to Upload"):
                    st.session_state.current_step = 'upload'
                    st.rerun()
        
        elif st.session_state.current_step == 'results':
            display_analysis_results()
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Analyze New Data"):
                    st.session_state.current_step = 'upload'
                    st.session_state.processed_data = {}
                    st.session_state.analysis_results = {}
                    st.session_state.analysis_complete = False
                    st.rerun()
            
            with col2:
                if st.button("📥 Export Results"):
                    st.info("Export functionality coming soon")
        
        # Processing indicator
        if st.session_state.processing_in_progress:
            st.info("⚙️ Processing in progress...")
        
        # Error display
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        
        # Success display
        if st.session_state.success_message:
            st.success(st.session_state.success_message)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("🚨 Application Error")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        
        if st.session_state.get('show_debug', False):
            st.exception(e)

if __name__ == "__main__":
    main()
