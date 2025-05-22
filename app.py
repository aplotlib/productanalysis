"""
Amazon Review Analyzer - Listing Optimization Focus
AI-powered analysis of Amazon review data for listing improvements

Author: Assistant
Version: 5.0 - Listing Optimization Focused
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import traceback
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports for your modules
def safe_import(module_name):
    """Safely import modules with fallback"""
    try:
        module = __import__(module_name)
        return module, True
    except ImportError as e:
        logger.warning(f"Module {module_name} not available: {str(e)}")
        return None, False

# Import your custom modules
enhanced_ai_analysis, ai_available = safe_import('enhanced_ai_analysis')
upload_handler, upload_available = safe_import('upload_handler')

# Application configuration
APP_CONFIG = {
    'title': 'Amazon Review Analyzer - Listing Optimization',
    'version': '5.0',
    'description': 'AI-powered Amazon review analysis for listing improvements',
    'max_file_size_mb': 50,
    'supported_formats': ['.csv', '.xlsx']
}

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'uploaded_data': None,
        'analysis_results': None,
        'current_step': 'upload',
        'processing': False,
        'ai_analyzer': None,
        'ai_status': None,
        'error_message': None,
        'success_message': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_ai_status():
    """Check AI availability and initialize analyzer"""
    if not ai_available:
        return {
            'available': False,
            'error': 'Enhanced AI Analysis module not found',
            'message': 'AI analysis features are not available'
        }
    
    try:
        if st.session_state.ai_analyzer is None:
            st.session_state.ai_analyzer = enhanced_ai_analysis.EnhancedAIAnalyzer()
        
        status = st.session_state.ai_analyzer.get_api_status()
        return status
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
            'message': 'AI initialization failed'
        }

def display_ai_status():
    """Display AI status in sidebar"""
    with st.sidebar:
        st.markdown("### 🤖 AI Status")
        
        status = check_ai_status()
        
        if status.get('available'):
            st.success("✅ AI Analysis Ready")
            st.caption("GPT-4o powered insights available")
        else:
            st.error("❌ AI Analysis Unavailable")
            st.caption(status.get('error', 'Unknown error'))
            
            # Show configuration help
            with st.expander("🔧 Configuration Help"):
                st.markdown("""
                **To enable AI analysis:**
                1. Add OpenAI API key to Streamlit secrets:
                   ```
                   [secrets]
                   openai_api_key = "your-api-key-here"
                   ```
                2. Or set environment variable:
                   ```
                   OPENAI_API_KEY=your-api-key-here
                   ```
                """)

def load_example_data():
    """Load example data for demonstration"""
    example_data = {
        'Date': [
            'Reviewed in the United States on May 19, 2025',
            'Reviewed in the United States on May 15, 2025',
            'Reviewed in the United States on May 10, 2025',
            'Reviewed in the United States on May 8, 2025',
            'Reviewed in the United States on May 5, 2025'
        ],
        'Author': ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5'],
        'Verified': ['yes', 'yes', 'yes', 'yes', 'yes'],
        'Title': [
            'Great product but instructions unclear',
            'Broke after one week', 
            'Perfect for my needs',
            'Good quality but expensive',
            'Amazing results!'
        ],
        'Body': [
            'This product works well but the assembly instructions were confusing. Would be 5 stars with better documentation.',
            'The product broke after just one week of normal use. Very disappointed with the quality.',
            'Exactly what I needed. Great quality and fast shipping. Highly recommend.',
            'Good product quality but feels overpriced compared to similar items.',
            'Exceeded my expectations! This has made such a difference in my daily routine.'
        ],
        'Rating': [4, 1, 5, 3, 5],
        'Helpful': ['-', '-', '-', '-', '-'],
        'Images': ['-', '-', '-', '-', '-'],
        'Videos': ['-', '-', '-', '-', '-'],
        'URL': [''] * 5,
        'Variation': ['B00EXAMPLE123'] * 5,
        'Style': ['Size: 1 Count (Pack of 1)'] * 5
    }
    
    return pd.DataFrame(example_data)

def handle_file_upload():
    """Handle file upload interface"""
    st.markdown("## 📁 Upload Amazon Review Data")
    st.markdown("Upload your Amazon review export (CSV or Excel) for AI-powered listing optimization insights")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose your review export file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file exported from Amazon or review scraping tools"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if uploaded_file is not None:
            # Display file info
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            if st.button("🚀 Process Reviews", type="primary", use_container_width=True):
                process_uploaded_file(uploaded_file)
    
    with col2:
        if st.button("📊 Try Example Data", use_container_width=True):
            load_example_reviews()

def process_uploaded_file(uploaded_file):
    """Process uploaded review file"""
    try:
        st.session_state.processing = True
        st.session_state.error_message = None
        
        with st.spinner("📊 Processing review data..."):
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_columns = ['Title', 'Body', 'Rating']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Clean and prepare data
            clean_data = prepare_review_data(df)
            
            if clean_data['success']:
                st.session_state.uploaded_data = clean_data
                st.session_state.current_step = 'analysis'
                st.success(f"✅ Processed {clean_data['total_reviews']} reviews successfully")
                st.rerun()
            else:
                st.error(f"❌ Error processing data: {clean_data.get('error', 'Unknown error')}")
                
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        st.error(f"❌ Error processing file: {str(e)}")
    finally:
        st.session_state.processing = False

def load_example_reviews():
    """Load example review data"""
    try:
        df = load_example_data()
        clean_data = prepare_review_data(df)
        
        if clean_data['success']:
            st.session_state.uploaded_data = clean_data
            st.session_state.current_step = 'analysis'
            st.success("✅ Example data loaded successfully")
            st.rerun()
        else:
            st.error("❌ Error loading example data")
    except Exception as e:
        st.error(f"❌ Error loading example data: {str(e)}")

def prepare_review_data(df):
    """Prepare and clean review data"""
    try:
        # Extract product info
        product_info = {
            'asin': df['Variation'].iloc[0] if 'Variation' in df.columns else 'Unknown',
            'name': 'Product Name', # Could extract from filename or other source
            'total_reviews': len(df)
        }
        
        # Prepare reviews for analysis
        reviews = []
        for _, row in df.iterrows():
            if pd.notna(row['Body']) and len(str(row['Body']).strip()) > 10:
                review = {
                    'text': f"{row.get('Title', '')} | {row.get('Body', '')}",
                    'title': row.get('Title', ''),
                    'body': row.get('Body', ''),
                    'rating': row.get('Rating', 3),
                    'date': row.get('Date', ''),
                    'author': row.get('Author', ''),
                    'verified': row.get('Verified', '') == 'yes'
                }
                reviews.append(review)
        
        return {
            'success': True,
            'product_info': product_info,
            'reviews': reviews,
            'total_reviews': len(reviews),
            'raw_data': df
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def display_data_summary():
    """Display summary of uploaded data"""
    if not st.session_state.uploaded_data:
        return
    
    data = st.session_state.uploaded_data
    product_info = data['product_info']
    reviews = data['reviews']
    
    st.markdown("## 📊 Review Data Summary")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", data['total_reviews'])
    
    with col2:
        ratings = [r['rating'] for r in reviews if r['rating']]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        st.metric("Average Rating", f"{avg_rating:.1f}/5")
    
    with col3:
        verified_count = sum(1 for r in reviews if r.get('verified'))
        st.metric("Verified Reviews", verified_count)
    
    with col4:
        st.metric("Product ASIN", product_info['asin'])
    
    # Rating distribution
    if ratings:
        st.markdown("### Rating Distribution")
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        rating_df = pd.DataFrame([
            {'Rating': k, 'Count': v, 'Percentage': (v/len(ratings))*100}
            for k, v in sorted(rating_counts.items())
        ])
        
        st.dataframe(rating_df, use_container_width=True)
    
    # Analysis button
    if st.button("🤖 Run AI Analysis", type="primary", use_container_width=True):
        run_ai_analysis()

def run_ai_analysis():
    """Run AI analysis on uploaded data"""
    if not st.session_state.uploaded_data:
        st.error("No data to analyze")
        return
    
    # Check AI status
    ai_status = check_ai_status()
    
    if not ai_status.get('available'):
        st.error("❌ AI analysis not available")
        st.info("Running basic analysis instead...")
        run_basic_analysis()
        return
    
    try:
        st.session_state.processing = True
        
        with st.spinner("🤖 Running AI analysis... This may take a moment."):
            data = st.session_state.uploaded_data
            
            # Create listing optimization focused prompt
            result = analyze_for_listing_optimization(
                data['product_info'], 
                data['reviews']
            )
            
            if result.get('success'):
                st.session_state.analysis_results = result
                st.session_state.current_step = 'results'
                st.success("✅ AI analysis completed successfully")
                st.rerun()
            else:
                st.error(f"❌ AI analysis failed: {result.get('error', 'Unknown error')}")
                st.info("Running basic analysis instead...")
                run_basic_analysis()
                
    except Exception as e:
        logger.error(f"AI analysis error: {str(e)}")
        st.error(f"❌ AI analysis error: {str(e)}")
        st.info("Running basic analysis instead...")
        run_basic_analysis()
    finally:
        st.session_state.processing = False

def analyze_for_listing_optimization(product_info, reviews):
    """Run AI analysis focused on listing optimization"""
    try:
        # Prepare reviews for AI analysis
        review_texts = []
        for review in reviews[:50]:  # Limit to 50 reviews for token management
            text = f"Rating: {review['rating']}/5\n"
            text += f"Title: {review['title']}\n"
            text += f"Review: {review['body']}\n"
            review_texts.append(text)
        
        combined_reviews = "\n---\n".join(review_texts)
        
        # Create listing optimization prompt
        prompt = f"""
        Analyze these Amazon reviews for listing optimization insights. Focus on actionable improvements for the product listing.
        
        Product ASIN: {product_info.get('asin', 'Unknown')}
        Total Reviews Analyzed: {len(review_texts)}
        
        Reviews:
        {combined_reviews}
        
        Provide analysis in this JSON format:
        {{
            "listing_optimization": {{
                "title_improvements": ["specific keyword suggestions", "title structure improvements"],
                "bullet_point_gaps": ["missing features customers mention", "benefits to highlight"],
                "image_recommendations": ["what images customers want to see", "angles/scenarios to show"],
                "description_enhancements": ["details to add", "pain points to address"]
            }},
            "customer_insights": {{
                "love_most": [["feature", "example quote"]],
                "hate_most": [["issue", "example quote"]],
                "common_complaints": [["complaint", "frequency", "suggested fix"]],
                "unexpected_uses": ["use cases not highlighted in listing"],
                "size_fit_issues": ["sizing guidance needed"]
            }},
            "content_optimization": {{
                "faq_needed": ["questions customers have"],
                "confusion_points": ["what customers don't understand"],
                "missing_specs": ["specifications customers want"],
                "competitor_mentions": ["what customers compare to"]
            }},
            "conversion_opportunities": {{
                "trust_signals": ["what builds confidence"],
                "hesitation_points": ["what makes customers hesitate"],
                "price_feedback": ["price-related comments"],
                "urgency_triggers": ["what makes customers buy"]
            }},
            "summary": {{
                "overall_sentiment": "positive/negative/mixed",
                "key_strengths": ["top 3 strengths"],
                "critical_issues": ["top 3 issues to fix"],
                "listing_score": "0-100 based on review feedback"
            }}
        }}
        """
        
        # Use your AI analyzer
        api_result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": "You are an Amazon listing optimization expert specializing in converting customer feedback into actionable listing improvements."},
            {"role": "user", "content": prompt}
        ])
        
        if api_result['success']:
            try:
                analysis = json.loads(api_result['result'])
                return {
                    'success': True,
                    'analysis': analysis,
                    'ai_powered': True,
                    'reviews_analyzed': len(review_texts),
                    'timestamp': datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                return {
                    'success': True,
                    'raw_analysis': api_result['result'],
                    'ai_powered': True,
                    'reviews_analyzed': len(review_texts),
                    'timestamp': datetime.now().isoformat()
                }
        else:
            return {
                'success': False,
                'error': api_result.get('error', 'AI analysis failed')
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_basic_analysis():
    """Fallback basic analysis when AI is unavailable"""
    if not st.session_state.uploaded_data:
        return
    
    try:
        data = st.session_state.uploaded_data
        reviews = data['reviews']
        
        # Basic metrics
        ratings = [r['rating'] for r in reviews if r['rating']]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        positive_reviews = [r for r in reviews if r['rating'] >= 4]
        negative_reviews = [r for r in reviews if r['rating'] <= 2]
        
        # Simple keyword analysis
        all_text = ' '.join([r['body'].lower() for r in reviews if r['body']]).lower()
        
        # Basic insights
        basic_analysis = {
            'success': True,
            'ai_powered': False,
            'basic_metrics': {
                'total_reviews': len(reviews),
                'average_rating': round(avg_rating, 2),
                'positive_reviews': len(positive_reviews),
                'negative_reviews': len(negative_reviews),
                'satisfaction_rate': len(positive_reviews) / len(reviews) * 100 if reviews else 0
            },
            'simple_insights': {
                'needs_improvement': len(negative_reviews) > len(reviews) * 0.3,
                'overall_sentiment': 'positive' if avg_rating >= 4 else 'negative' if avg_rating <= 2 else 'mixed'
            }
        }
        
        st.session_state.analysis_results = basic_analysis
        st.session_state.current_step = 'results'
        st.warning("🤖 AI Analysis Unavailable - Showing Basic Analysis")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Basic analysis failed: {str(e)}")

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.analysis_results:
        st.error("No analysis results available")
        return
    
    results = st.session_state.analysis_results
    
    if results.get('ai_powered'):
        display_ai_results(results)
    else:
        display_basic_results(results)

def display_ai_results(results):
    """Display AI-powered analysis results"""
    st.markdown("## 🤖 AI Analysis Results")
    
    # Check if we have structured analysis or raw analysis
    if 'analysis' in results:
        analysis = results['analysis']
        
        # Create tabs for different aspects
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Listing Optimization",
            "💡 Customer Insights", 
            "📝 Content Gaps",
            "💰 Conversion Opportunities"
        ])
        
        with tab1:
            display_listing_optimization(analysis.get('listing_optimization', {}))
        
        with tab2:
            display_customer_insights(analysis.get('customer_insights', {}))
        
        with tab3:
            display_content_optimization(analysis.get('content_optimization', {}))
        
        with tab4:
            display_conversion_opportunities(analysis.get('conversion_opportunities', {}))
        
        # Summary at the bottom
        if 'summary' in analysis:
            display_summary(analysis['summary'])
    
    elif 'raw_analysis' in results:
        # Display raw AI response if JSON parsing failed
        st.markdown("### AI Analysis Results")
        st.markdown(results['raw_analysis'])

def display_listing_optimization(optimization_data):
    """Display listing optimization recommendations"""
    st.markdown("### 🎯 Listing Optimization Recommendations")
    
    if optimization_data.get('title_improvements'):
        st.markdown("**📝 Title Improvements:**")
        for improvement in optimization_data['title_improvements']:
            st.markdown(f"• {improvement}")
    
    if optimization_data.get('bullet_point_gaps'):
        st.markdown("**🔸 Bullet Point Enhancements:**")
        for gap in optimization_data['bullet_point_gaps']:
            st.markdown(f"• {gap}")
    
    if optimization_data.get('image_recommendations'):
        st.markdown("**📸 Image Recommendations:**")
        for rec in optimization_data['image_recommendations']:
            st.markdown(f"• {rec}")
    
    if optimization_data.get('description_enhancements'):
        st.markdown("**📄 Description Enhancements:**")
        for enhancement in optimization_data['description_enhancements']:
            st.markdown(f"• {enhancement}")

def display_customer_insights(insights_data):
    """Display customer insights"""
    st.markdown("### 💡 Customer Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if insights_data.get('love_most'):
            st.markdown("**😍 What Customers Love:**")
            for item in insights_data['love_most']:
                if isinstance(item, list) and len(item) >= 2:
                    st.markdown(f"• **{item[0]}**: {item[1]}")
    
    with col2:
        if insights_data.get('hate_most'):
            st.markdown("**😤 What Customers Dislike:**")
            for item in insights_data['hate_most']:
                if isinstance(item, list) and len(item) >= 2:
                    st.markdown(f"• **{item[0]}**: {item[1]}")
    
    if insights_data.get('common_complaints'):
        st.markdown("**⚠️ Common Complaints & Fixes:**")
        for complaint in insights_data['common_complaints']:
            if isinstance(complaint, list) and len(complaint) >= 3:
                st.markdown(f"• **Issue**: {complaint[0]} | **Fix**: {complaint[2]}")

def display_content_optimization(content_data):
    """Display content optimization suggestions"""
    st.markdown("### 📝 Content Optimization")
    
    if content_data.get('faq_needed'):
        st.markdown("**❓ FAQ Items to Add:**")
        for faq in content_data['faq_needed']:
            st.markdown(f"• {faq}")
    
    if content_data.get('confusion_points'):
        st.markdown("**🤔 Customer Confusion Points:**")
        for point in content_data['confusion_points']:
            st.markdown(f"• {point}")
    
    if content_data.get('missing_specs'):
        st.markdown("**📋 Missing Specifications:**")
        for spec in content_data['missing_specs']:
            st.markdown(f"• {spec}")

def display_conversion_opportunities(conversion_data):
    """Display conversion optimization opportunities"""
    st.markdown("### 💰 Conversion Opportunities")
    
    if conversion_data.get('trust_signals'):
        st.markdown("**✅ Trust Building Opportunities:**")
        for signal in conversion_data['trust_signals']:
            st.markdown(f"• {signal}")
    
    if conversion_data.get('hesitation_points'):
        st.markdown("**🤚 Customer Hesitation Points:**")
        for point in conversion_data['hesitation_points']:
            st.markdown(f"• {point}")

def display_summary(summary_data):
    """Display analysis summary"""
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment = summary_data.get('overall_sentiment', 'Unknown')
        st.metric("Overall Sentiment", sentiment.title())
    
    with col2:
        score = summary_data.get('listing_score', 'N/A')
        st.metric("Listing Score", f"{score}/100" if isinstance(score, (int, float)) else score)
    
    with col3:
        strengths = summary_data.get('key_strengths', [])
        st.metric("Key Strengths", len(strengths))
    
    if summary_data.get('critical_issues'):
        st.markdown("**🔴 Critical Issues to Address:**")
        for issue in summary_data['critical_issues']:
            st.markdown(f"• {issue}")

def display_basic_results(results):
    """Display basic analysis results"""
    st.markdown("## 📊 Basic Analysis Results")
    st.warning("🤖 AI analysis unavailable - showing basic metrics only")
    
    metrics = results.get('basic_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", metrics.get('total_reviews', 0))
    
    with col2:
        st.metric("Average Rating", f"{metrics.get('average_rating', 0)}/5")
    
    with col3:
        st.metric("Positive Reviews", metrics.get('positive_reviews', 0))
    
    with col4:
        satisfaction = metrics.get('satisfaction_rate', 0)
        st.metric("Satisfaction Rate", f"{satisfaction:.1f}%")
    
    st.info("💡 **Upgrade to AI Analysis**: Configure your OpenAI API key to get detailed listing optimization recommendations, customer insights, and actionable improvements.")

def main():
    """Main application"""
    try:
        # Page config
        st.set_page_config(
            page_title=APP_CONFIG['title'],
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session
        initialize_session_state()
        
        # Header
        st.title("📊 Amazon Review Analyzer")
        st.markdown(f"**{APP_CONFIG['description']}** - Version {APP_CONFIG['version']}")
        
        # Sidebar with AI status
        display_ai_status()
        
        # Main workflow
        if st.session_state.current_step == 'upload':
            handle_file_upload()
            
        elif st.session_state.current_step == 'analysis':
            display_data_summary()
            
        elif st.session_state.current_step == 'results':
            display_analysis_results()
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Analyze New Data", use_container_width=True):
                    st.session_state.current_step = 'upload'
                    st.session_state.uploaded_data = None
                    st.session_state.analysis_results = None
                    st.rerun()
            
            with col2:
                if st.button("📥 Export Results", use_container_width=True):
                    st.info("📥 Export functionality coming soon!")
        
        # Processing indicator
        if st.session_state.processing:
            st.info("⚙️ Processing...")
        
        # Error/success messages
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        
        if st.session_state.success_message:
            st.success(st.session_state.success_message)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("🚨 Application Error")
        st.error("An unexpected error occurred. Please refresh and try again.")
        
        with st.expander("🔍 Error Details"):
            st.exception(e)

if __name__ == "__main__":
    main()
