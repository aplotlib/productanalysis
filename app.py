"""
Amazon Review Analyzer - Listing Optimization Focus
AI-powered analysis of Amazon review data for listing improvements

Author: Assistant
Version: 5.1 - Enhanced with AI Chat & Better Formatting
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
ai_chat, chat_available = safe_import('ai_chat')

# Application configuration
APP_CONFIG = {
    'title': 'Amazon Review Analyzer - Listing Optimization',
    'version': '5.1',
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
        'success_message': None,
        'show_chat': False,
        'current_listing_title': '',
        'current_listing_description': '',
        'chat_session': None
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
            
            # AI Chat toggle
            if st.button("💬 Open AI Chat", use_container_width=True):
                st.session_state.show_chat = True
                st.rerun()
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

def display_ai_chat():
    """Display AI chat interface"""
    if not st.session_state.show_chat or not chat_available:
        return
    
    with st.expander("💬 AI Listing Optimization Chat", expanded=True):
        col1, col2 = st.columns([5, 1])
        
        with col2:
            if st.button("❌ Close"):
                st.session_state.show_chat = False
                st.rerun()
        
        with col1:
            st.markdown("**Ask questions about listing optimization, get advice, or discuss your analysis results**")
        
        # Initialize chat session if needed
        if st.session_state.chat_session is None and chat_available:
            st.session_state.chat_session = ai_chat.ChatSession('listing_optimizer')
        
        # Chat interface
        if st.session_state.chat_session:
            # Display chat history
            for message in st.session_state.chat_session.messages[-10:]:  # Show last 10 messages
                role = message['role']
                content = message['content']
                
                if role == 'user':
                    with st.chat_message("user"):
                        st.markdown(content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(content)
            
            # Chat input
            user_input = st.chat_input("Ask about listing optimization...")
            
            if user_input:
                # Add context about current analysis if available
                context_message = user_input
                if st.session_state.analysis_results:
                    context_message = f"Context: I'm analyzing Amazon reviews for listing optimization. Current question: {user_input}"
                
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_session.send_message(context_message)
                    st.markdown(response)
                
                st.rerun()

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
    
    # Optional listing information section
    with st.expander("📝 Current Listing Information (Optional)", expanded=False):
        st.markdown("Paste your current listing title and description to get comparison insights")
        
        current_title = st.text_area(
            "Current Listing Title:",
            value=st.session_state.current_listing_title,
            height=100,
            placeholder="Paste your current Amazon listing title here..."
        )
        
        current_description = st.text_area(
            "Current Listing Description:",
            value=st.session_state.current_listing_description,
            height=200,
            placeholder="Paste your current Amazon listing description/bullet points here..."
        )
        
        if st.button("💾 Save Listing Info"):
            st.session_state.current_listing_title = current_title
            st.session_state.current_listing_description = current_description
            st.success("✅ Listing information saved")
    
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
    
    # Show AI chat option
    display_ai_chat()
    
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
            
            # Create comprehensive listing optimization analysis
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
    """Run comprehensive AI analysis focused on listing optimization"""
    try:
        # Prepare comprehensive review analysis
        review_summaries = []
        for review in reviews[:30]:  # Use more reviews for better analysis
            summary = f"Rating: {review['rating']}/5 stars\n"
            summary += f"Title: {review.get('title', '')}\n"
            summary += f"Review: {review.get('body', '')}\n"
            summary += f"Verified: {'Yes' if review.get('verified') else 'No'}\n"
            review_summaries.append(summary)
        
        # Include current listing context if available
        listing_context = ""
        if st.session_state.current_listing_title or st.session_state.current_listing_description:
            listing_context = f"\n\nCURRENT LISTING INFO:\n"
            if st.session_state.current_listing_title:
                listing_context += f"Current Title: {st.session_state.current_listing_title}\n"
            if st.session_state.current_listing_description:
                listing_context += f"Current Description: {st.session_state.current_listing_description}\n"
        
        # Use your existing AI analyzer with enhanced prompts
        prompt = f"""
        Analyze these Amazon customer reviews for comprehensive listing optimization insights.
        
        Product ASIN: {product_info.get('asin', 'Unknown')}
        Total Reviews: {len(review_summaries)}
        {listing_context}
        
        CUSTOMER REVIEWS:
        {chr(10).join(review_summaries)}
        
        Provide detailed analysis in structured format focusing on:
        
        1. LISTING OPTIMIZATION RECOMMENDATIONS
        2. CUSTOMER SENTIMENT ANALYSIS  
        3. CONTENT GAPS AND OPPORTUNITIES
        4. COMPETITIVE POSITIONING INSIGHTS
        5. IMMEDIATE ACTION ITEMS
        
        Format your response clearly with headers and bullet points for easy reading.
        """
        
        # Call your AI analyzer
        result = st.session_state.ai_analyzer.analyze_reviews_comprehensive(
            product_info, 
            reviews
        )
        
        if result.get('success'):
            return {
                'success': True,
                'ai_analysis': result,
                'reviews_analyzed': len(review_summaries),
                'timestamp': datetime.now().isoformat(),
                'has_listing_context': bool(listing_context)
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'AI analysis failed')
            }
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
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
    
    st.markdown("## 🤖 AI Analysis Results")
    
    # Show AI chat for discussing results
    display_ai_chat()
    
    if results.get('ai_analysis'):
        display_comprehensive_ai_results(results)
    else:
        display_basic_results(results)

def display_comprehensive_ai_results(results):
    """Display comprehensive AI analysis results with better formatting"""
    ai_analysis = results['ai_analysis']
    
    # Summary metrics at top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Reviews Analyzed", results.get('reviews_analyzed', 0))
    
    with col2:
        st.metric("AI Analysis", "✅ Complete")
    
    with col3:
        sentiment = ai_analysis.get('overall_sentiment', 'Unknown')
        st.metric("Overall Sentiment", sentiment.title())
    
    with col4:
        if results.get('has_listing_context'):
            st.metric("Listing Context", "✅ Included")
        else:
            st.metric("Listing Context", "❌ Not Provided")
    
    # Main analysis sections
    if ai_analysis.get('listing_improvements'):
        st.markdown("### 🎯 Listing Optimization Recommendations")
        st.info(ai_analysis['listing_improvements'])
    
    if ai_analysis.get('safety_concerns'):
        st.markdown("### ⚠️ Customer Concerns")
        for concern in ai_analysis['safety_concerns']:
            st.warning(f"• {concern}")
    
    if ai_analysis.get('top_quality_issues'):
        st.markdown("### 🔍 Top Issues to Address")
        for i, issue in enumerate(ai_analysis['top_quality_issues'], 1):
            st.markdown(f"**{i}.** {issue}")
    
    if ai_analysis.get('immediate_actions'):
        st.markdown("### 🚀 Immediate Action Items")
        for action in ai_analysis['immediate_actions']:
            st.markdown(f"• {action}")
    
    if ai_analysis.get('customer_education'):
        st.markdown("### 📚 Customer Education Opportunities")
        st.info(ai_analysis['customer_education'])
    
    # Raw AI response for debugging
    with st.expander("🔍 View Raw AI Analysis"):
        if ai_analysis.get('raw_response'):
            st.text(ai_analysis['raw_response'])
        else:
            st.json(ai_analysis)

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
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Analyze New Data", use_container_width=True):
                    st.session_state.current_step = 'upload'
                    st.session_state.uploaded_data = None
                    st.session_state.analysis_results = None
                    st.rerun()
            
            with col2:
                if st.button("💬 Discuss Results", use_container_width=True):
                    st.session_state.show_chat = True
                    st.rerun()
            
            with col3:
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
