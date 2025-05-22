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
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import re

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
        'chat_session': None,
        'selected_date_range': None,
        'filtered_data': None,
        'rating_trends': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def parse_amazon_date(date_string):
    """Parse Amazon review date format: 'Reviewed in the United States on May 19, 2025'"""
    try:
        if pd.isna(date_string) or not date_string:
            return None
            
        # Extract date part after "on "
        if "on " in str(date_string):
            date_part = str(date_string).split("on ")[-1]
        else:
            date_part = str(date_string)
        
        # Try multiple date formats
        date_formats = [
            '%B %d, %Y',      # May 19, 2025
            '%b %d, %Y',      # May 19, 2025
            '%m/%d/%Y',       # 5/19/2025
            '%Y-%m-%d',       # 2025-05-19
            '%d/%m/%Y'        # 19/5/2025
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_part.strip(), fmt).date()
            except ValueError:
                continue
        
        # If all formats fail, try pandas date parser
        return pd.to_datetime(date_part, errors='coerce').date()
        
    except Exception as e:
        logger.warning(f"Could not parse date '{date_string}': {str(e)}")
        return None

def calculate_rating_trends(reviews_df):
    """Calculate rating trends over time"""
    try:
        # Parse dates
        reviews_df['parsed_date'] = reviews_df['Date'].apply(parse_amazon_date)
        
        # Filter out rows with unparseable dates
        valid_reviews = reviews_df[reviews_df['parsed_date'].notna()].copy()
        
        if len(valid_reviews) == 0:
            return {
                'success': False,
                'error': 'No valid dates found in reviews'
            }
        
        # Sort by date
        valid_reviews = valid_reviews.sort_values('parsed_date')
        
        # Calculate monthly averages
        valid_reviews['year_month'] = valid_reviews['parsed_date'].apply(lambda x: x.strftime('%Y-%m'))
        monthly_ratings = valid_reviews.groupby('year_month').agg({
            'Rating': ['mean', 'count'],
            'parsed_date': 'first'
        }).reset_index()
        
        monthly_ratings.columns = ['year_month', 'avg_rating', 'review_count', 'first_date']
        monthly_ratings['avg_rating'] = monthly_ratings['avg_rating'].round(2)
        
        # Convert first_date to datetime for proper handling
        monthly_ratings['date'] = pd.to_datetime(monthly_ratings['first_date'])
        
        # Calculate overall trend
        if len(monthly_ratings) >= 2:
            first_month = monthly_ratings.iloc[0]['avg_rating']
            last_month = monthly_ratings.iloc[-1]['avg_rating']
            trend_direction = 'improving' if last_month > first_month else 'declining' if last_month < first_month else 'stable'
            trend_magnitude = abs(last_month - first_month)
        else:
            trend_direction = 'insufficient_data'
            trend_magnitude = 0
        
        return {
            'success': True,
            'monthly_data': monthly_ratings,
            'date_range': {
                'earliest': valid_reviews['parsed_date'].min(),
                'latest': valid_reviews['parsed_date'].max()
            },
            'trend_analysis': {
                'direction': trend_direction,
                'magnitude': round(trend_magnitude, 2),
                'current_avg': monthly_ratings.iloc[-1]['avg_rating'] if len(monthly_ratings) > 0 else 0,
                'total_months': len(monthly_ratings)
            },
            'total_valid_reviews': len(valid_reviews)
        }
        
    except Exception as e:
        logger.error(f"Error calculating rating trends: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

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
    if not st.session_state.show_chat:
        return
    
    if not chat_available:
        st.error("❌ AI Chat module not available")
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
            try:
                st.session_state.chat_session = ai_chat.ChatSession('listing_optimizer')
            except Exception as e:
                st.error(f"Failed to initialize chat: {str(e)}")
                return
        
        # Add context about current analysis to chat
        if st.session_state.analysis_results and st.session_state.chat_session:
            # Prepare context message for the AI
            context_info = []
            results = st.session_state.analysis_results
            
            if results.get('review_categories'):
                categories = results['review_categories']
                top_issues = sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
                context_info.append("Current Analysis Context:")
                context_info.append(f"- Total reviews analyzed: {results.get('reviews_analyzed', 0)}")
                for category_key, category_data in top_issues:
                    category_name = category_key.replace('_', ' ').title()
                    context_info.append(f"- {category_name}: {category_data['count']} reviews")
            
            if context_info and len(st.session_state.chat_session.messages) == 0:
                # Add initial context message
                context_message = "\n".join(context_info)
                st.session_state.chat_session.add_assistant_message(
                    f"I can see you've just completed an analysis. Here's what I found:\n\n{context_message}\n\nHow can I help you interpret these results or improve your listing?"
                )
        
        # Chat interface
        if st.session_state.chat_session:
            # Display chat history
            chat_container = st.container()
            
            with chat_container:
                if not st.session_state.chat_session.messages:
                    st.info("👋 Hi! I'm your Amazon listing optimization expert. Ask me anything about improving your listings!")
                else:
                    # Display conversation
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
                enhanced_message = user_input
                if st.session_state.analysis_results:
                    enhanced_message = f"Based on my current review analysis, {user_input}"
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.chat_session.send_message(enhanced_message)
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"Chat error: {str(e)}")
                            st.markdown("I'm having trouble right now. Please check your API configuration.")
                
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
    """Prepare and clean review data with enhanced date parsing"""
    try:
        # Parse dates first
        df['parsed_date'] = df['Date'].apply(parse_amazon_date)
        
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
                    'parsed_date': row.get('parsed_date'),
                    'author': row.get('Author', ''),
                    'verified': row.get('Verified', '') == 'yes'
                }
                reviews.append(review)
        
        # Calculate rating trends
        trends = calculate_rating_trends(df)
        
        return {
            'success': True,
            'product_info': product_info,
            'reviews': reviews,
            'total_reviews': len(reviews),
            'raw_data': df,
            'rating_trends': trends,
            'date_range': trends.get('date_range') if trends.get('success') else None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def display_data_summary():
    """Display summary of uploaded data with date filtering"""
    if not st.session_state.uploaded_data:
        return
    
    data = st.session_state.uploaded_data
    product_info = data['product_info']
    reviews = data['reviews']
    
    st.markdown("## 📊 Review Data Summary")
    
    # Show AI chat option
    display_ai_chat()
    
    # Date filtering section
    if data.get('date_range'):
        display_date_filtering(data)
    
    # Use filtered data if available, otherwise use all data
    current_reviews = st.session_state.filtered_data if st.session_state.filtered_data else reviews
    current_data_label = "Filtered" if st.session_state.filtered_data else "All"
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(f"{current_data_label} Reviews", len(current_reviews))
    
    with col2:
        ratings = [r['rating'] for r in current_reviews if r['rating']]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        st.metric("Average Rating", f"{avg_rating:.1f}/5")
    
    with col3:
        verified_count = sum(1 for r in current_reviews if r.get('verified'))
        st.metric("Verified Reviews", verified_count)
    
    with col4:
        st.metric("Product ASIN", product_info['asin'])
    
    # Rating trends
    if data.get('rating_trends') and data['rating_trends'].get('success'):
        display_rating_trends(data['rating_trends'])
    
    # Rating distribution
    if ratings:
        display_rating_distribution(current_reviews, current_data_label)
    
    # Analysis button
    if st.button("🤖 Run AI Analysis", type="primary", use_container_width=True):
        run_ai_analysis()

def display_date_filtering(data):
    """Display date filtering controls"""
    date_range = data['date_range']
    
    with st.expander("📅 Filter by Date Range", expanded=False):
        st.markdown("**Select date range to analyze specific time periods**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date_range['earliest'],
                min_value=date_range['earliest'],
                max_value=date_range['latest']
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=date_range['latest'],
                min_value=date_range['earliest'],
                max_value=date_range['latest']
            )
        
        if st.button("🔍 Apply Date Filter"):
            # Filter reviews by date range
            filtered_reviews = []
            for review in data['reviews']:
                if review.get('parsed_date'):
                    if start_date <= review['parsed_date'] <= end_date:
                        filtered_reviews.append(review)
            
            st.session_state.filtered_data = filtered_reviews
            st.session_state.selected_date_range = {
                'start': start_date,
                'end': end_date
            }
            st.success(f"✅ Filtered to {len(filtered_reviews)} reviews from {start_date} to {end_date}")
            st.rerun()
        
        if st.session_state.filtered_data:
            if st.button("🔄 Clear Filter"):
                st.session_state.filtered_data = None
                st.session_state.selected_date_range = None
                st.rerun()

def display_rating_trends(trends_data):
    """Display rating trends analysis"""
    with st.expander("📈 Rating Trends Over Time", expanded=True):
        if not trends_data.get('success'):
            st.error(f"❌ Could not calculate trends: {trends_data.get('error')}")
            return
        
        monthly_data = trends_data['monthly_data']
        trend_analysis = trends_data['trend_analysis']
        
        # Trend summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direction = trend_analysis['direction']
            direction_emoji = {
                'improving': '📈',
                'declining': '📉', 
                'stable': '➡️',
                'insufficient_data': '❓'
            }
            st.metric("Trend Direction", f"{direction_emoji.get(direction, '❓')} {direction.title()}")
        
        with col2:
            st.metric("Change Magnitude", f"{trend_analysis['magnitude']:.2f} stars")
        
        with col3:
            st.metric("Current Average", f"{trend_analysis['current_avg']:.1f}/5")
        
        with col4:
            st.metric("Months of Data", trend_analysis['total_months'])
        
        # Monthly data table
        if len(monthly_data) > 0:
            st.markdown("**Monthly Rating Breakdown:**")
            
            # Format for display - fix the datetime issue
            display_data = monthly_data.copy()
            
            # Safely format dates
            try:
                if 'date' in display_data.columns:
                    # Make sure date column is datetime
                    display_data['date'] = pd.to_datetime(display_data['date'])
                    display_data['Month'] = display_data['date'].dt.strftime('%B %Y')
                else:
                    # Fallback to year_month if date column is missing
                    display_data['Month'] = display_data['year_month']
            except Exception as e:
                # If datetime formatting fails, use year_month as fallback
                logger.warning(f"Date formatting error: {str(e)}")
                display_data['Month'] = display_data['year_month']
            
            display_data['Average Rating'] = display_data['avg_rating']
            display_data['Review Count'] = display_data['review_count']
            
            st.dataframe(
                display_data[['Month', 'Average Rating', 'Review Count']], 
                use_container_width=True
            )
            
            # Time period comparison
            if len(monthly_data) >= 2:
                display_time_period_comparison(monthly_data)

def display_time_period_comparison(monthly_data):
    """Display comparison between time periods"""
    st.markdown("**📊 Time Period Comparison**")
    
    try:
        # Split data into first half and second half
        mid_point = len(monthly_data) // 2
        first_half = monthly_data.iloc[:mid_point]
        second_half = monthly_data.iloc[mid_point:]
        
        if len(first_half) > 0 and len(second_half) > 0:
            first_avg = first_half['avg_rating'].mean()
            second_avg = second_half['avg_rating'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Earlier Period Avg", f"{first_avg:.2f}/5")
            
            with col2:
                st.metric("Recent Period Avg", f"{second_avg:.2f}/5")
            
            with col3:
                change = second_avg - first_avg
                change_direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                st.metric("Change", f"{change_direction} {abs(change):.2f}")
            
            # Interpretation
            if abs(change) >= 0.3:
                if change > 0:
                    st.success(f"🎉 **Significant Improvement**: Ratings increased by {change:.2f} stars over time")
                else:
                    st.error(f"⚠️ **Declining Trend**: Ratings decreased by {abs(change):.2f} stars over time")
            elif abs(change) >= 0.1:
                if change > 0:
                    st.info(f"📈 **Slight Improvement**: Ratings increased by {change:.2f} stars")
                else:
                    st.warning(f"📉 **Slight Decline**: Ratings decreased by {abs(change):.2f} stars")
            else:
                st.info("➡️ **Stable Performance**: No significant change in ratings over time")
    
    except Exception as e:
        logger.warning(f"Error in time period comparison: {str(e)}")
        st.info("Unable to calculate time period comparison")

def display_rating_distribution(reviews, data_label):
    """Display rating distribution"""
    st.markdown(f"### {data_label} Rating Distribution")
    ratings = [r['rating'] for r in reviews if r['rating']]
    
    if ratings:
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        rating_df = pd.DataFrame([
            {'Rating': k, 'Count': v, 'Percentage': (v/len(ratings))*100}
            for k, v in sorted(rating_counts.items())
        ])
        
        st.dataframe(rating_df, use_container_width=True)

def run_ai_analysis():
    """Run AI analysis on uploaded data with date filtering support"""
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
        
        # Use filtered data if available
        reviews_to_analyze = st.session_state.filtered_data if st.session_state.filtered_data else st.session_state.uploaded_data['reviews']
        data_context = "filtered" if st.session_state.filtered_data else "all"
        
        with st.spinner(f"🤖 Running AI analysis on {len(reviews_to_analyze)} {data_context} reviews..."):
            data = st.session_state.uploaded_data
            
            # Create comprehensive listing optimization analysis
            result = analyze_for_listing_optimization(
                data['product_info'], 
                reviews_to_analyze,
                data_context
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

def analyze_for_listing_optimization(product_info, reviews, data_context="all"):
    """Run comprehensive AI analysis focused on listing optimization - ALL REVIEWS"""
    try:
        # Process ALL reviews, not just a subset
        review_count_text = f"ALL {len(reviews)}" if data_context == "all" else f"{len(reviews)} filtered"
        st.info(f"🔍 Analyzing {review_count_text} reviews for comprehensive insights...")
        
        # Include date context if filtered
        date_context = ""
        if data_context == "filtered" and st.session_state.selected_date_range:
            date_range = st.session_state.selected_date_range
            date_context = f"\n\nDATE FILTER APPLIED: Analyzing reviews from {date_range['start']} to {date_range['end']}"
        
        # Prepare ALL reviews for analysis
        all_review_data = []
        for i, review in enumerate(reviews):
            review_data = {
                'id': i + 1,
                'rating': review['rating'],
                'title': review.get('title', ''),
                'body': review.get('body', ''),
                'verified': review.get('verified', False),
                'date': review.get('date', ''),
                'author': review.get('author', ''),
                'parsed_date': review.get('parsed_date')
            }
            all_review_data.append(review_data)
        
        # Include current listing context if available
        listing_context = ""
        if st.session_state.current_listing_title or st.session_state.current_listing_description:
            listing_context = f"\n\nCURRENT LISTING INFO:\n"
            if st.session_state.current_listing_title:
                listing_context += f"Current Title: {st.session_state.current_listing_title}\n"
            if st.session_state.current_listing_description:
                listing_context += f"Current Description: {st.session_state.current_listing_description}\n"
        
        # Run categorization analysis
        categorization_result = categorize_all_reviews(all_review_data, product_info, listing_context + date_context)
        
        # Run comprehensive analysis using your existing analyzer
        comprehensive_result = st.session_state.ai_analyzer.analyze_reviews_comprehensive(
            product_info, 
            reviews  # Pass all reviews
        )
        
        if categorization_result.get('success') and comprehensive_result.get('success'):
            return {
                'success': True,
                'ai_analysis': comprehensive_result,
                'review_categories': categorization_result['categories'],
                'reviews_analyzed': len(reviews),
                'timestamp': datetime.now().isoformat(),
                'has_listing_context': bool(listing_context),
                'data_context': data_context,
                'date_filter': st.session_state.selected_date_range if data_context == "filtered" else None
            }
        else:
            return {
                'success': False,
                'error': 'Analysis failed'
            }
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def categorize_all_reviews(all_reviews, product_info, listing_context=""):
    """Categorize ALL reviews into specific complaint/praise categories"""
    try:
        # Create batches for large review sets to avoid token limits
        batch_size = 50  # Reduce batch size for better AI processing
        all_categories = {}
        
        for i in range(0, len(all_reviews), batch_size):
            batch = all_reviews[i:i + batch_size]
            
            # Prepare batch for analysis - simplified format
            batch_reviews = []
            for review in batch:
                review_text = f"Review {review['id']}: [{review['rating']}/5] {review['title']} - {review['body'][:200]}..."
                batch_reviews.append(review_text)
            
            # Simplified categorization prompt for better parsing
            categorization_prompt = f"""
            Categorize these Amazon reviews into common issue categories. Return ONLY valid JSON.
            
            Product: {product_info.get('asin', 'Unknown')}
            Reviews {i+1} to {min(i+batch_size, len(all_reviews))}:
            
            {chr(10).join(batch_reviews)}
            
            Return JSON in this exact format (no extra text):
            {{
                "size_issues": {{"count": 0, "trend": "size complaint summary", "examples": ["quote1"]}},
                "quality_issues": {{"count": 0, "trend": "quality complaint summary", "examples": ["quote1"]}},
                "noise_issues": {{"count": 0, "trend": "noise complaint summary", "examples": ["quote1"]}},
                "fit_issues": {{"count": 0, "trend": "fit complaint summary", "examples": ["quote1"]}},
                "durability_issues": {{"count": 0, "trend": "durability complaint summary", "examples": ["quote1"]}},
                "ease_of_use": {{"count": 0, "trend": "usability feedback", "examples": ["quote1"]}},
                "value_issues": {{"count": 0, "trend": "price/value concerns", "examples": ["quote1"]}},
                "positive_highlights": {{"count": 0, "trend": "most praised features", "examples": ["quote1"]}}
            }}
            
            Only include categories with count > 0. Set count to actual number of matching reviews.
            """
            
            # Call AI for this batch
            batch_result = st.session_state.ai_analyzer.api_client.call_api([
                {"role": "system", "content": "You are a review categorization expert. Return only valid JSON with no additional text or explanations."},
                {"role": "user", "content": categorization_prompt}
            ], max_tokens=800)
            
            if batch_result['success']:
                try:
                    # Clean the response to ensure it's valid JSON
                    response_text = batch_result['result'].strip()
                    
                    # Remove any text before the first { or after the last }
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_text = response_text[start_idx:end_idx + 1]
                        batch_categories = json.loads(json_text)
                        
                        # Merge batch results with overall results
                        for category, data in batch_categories.items():
                            if category not in all_categories:
                                all_categories[category] = {
                                    'count': 0,
                                    'reviews': [],
                                    'trend': ''
                                }
                            
                            all_categories[category]['count'] += data.get('count', 0)
                            all_categories[category]['reviews'].extend(data.get('examples', []))
                            if data.get('trend') and data['count'] > 0:
                                all_categories[category]['trend'] = data['trend']
                    else:
                        st.warning(f"Invalid JSON structure in batch {i//batch_size + 1}")
                        continue
                
                except json.JSONDecodeError as e:
                    st.warning(f"Could not parse categorization for batch {i//batch_size + 1}: {str(e)}")
                    continue
            else:
                st.warning(f"AI call failed for batch {i//batch_size + 1}")
            
            # Update progress
            progress = min(i + batch_size, len(all_reviews))
            st.info(f"📊 Processed {progress}/{len(all_reviews)} reviews...")
        
        # Filter out empty categories
        filtered_categories = {k: v for k, v in all_categories.items() if v['count'] > 0}
        
        return {
            'success': True,
            'categories': filtered_categories,
            'total_categorized': sum(cat['count'] for cat in filtered_categories.values())
        }
        
    except Exception as e:
        logger.error(f"Categorization error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'categories': {}
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
        data_context = results.get('data_context', 'all')
        context_label = "📅 Filtered Data" if data_context == "filtered" else "📊 All Data"
        st.metric("Analysis Scope", context_label)
    
    with col3:
        sentiment = ai_analysis.get('overall_sentiment', 'Unknown')
        st.metric("Overall Sentiment", sentiment.title())
    
    with col4:
        if results.get('has_listing_context'):
            st.metric("Listing Context", "✅ Included")
        else:
            st.metric("Listing Context", "❌ Not Provided")
    
    # Show date filter info if applied
    if results.get('date_filter'):
        date_filter = results['date_filter']
        st.info(f"📅 **Date Filter Applied**: Analyzing reviews from {date_filter['start']} to {date_filter['end']}")
    
    # Review Categories Section - NEW!
    if results.get('review_categories'):
        display_review_categories(results['review_categories'])
    
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

def display_modern_date_filtering(data):
    """Modern date filtering interface"""
    date_range = data['date_range']
    
    with st.expander("📅 Time Period Analysis", expanded=False):
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4>📅 Focus Your Analysis</h4>
            <p>Analyze specific time periods to understand performance changes and trends</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date_range['earliest'],
                min_value=date_range['earliest'],
                max_value=date_range['latest']
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=date_range['latest'],
                min_value=date_range['earliest'],
                max_value=date_range['latest']
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("🔍 Apply Filter", use_container_width=True):
                # Filter reviews by date range
                filtered_reviews = []
                for review in data['reviews']:
                    if review.get('parsed_date'):
                        if start_date <= review['parsed_date'] <= end_date:
                            filtered_reviews.append(review)
                
                st.session_state.filtered_data = filtered_reviews
                st.session_state.selected_date_range = {
                    'start': start_date,
                    'end': end_date
                }
                
                st.markdown(f"""
                <div class="success-banner">
                    <h4>✅ Filter Applied</h4>
                    <p>Now analyzing {len(filtered_reviews)} reviews from {start_date} to {end_date}</p>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
        
        if st.session_state.filtered_data:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                filter_info = st.session_state.selected_date_range
                st.info(f"📅 **Active Filter**: {filter_info['start']} to {filter_info['end']} ({len(st.session_state.filtered_data)} reviews)")
            
            with col2:
                if st.button("🔄 Clear Filter", use_container_width=True):
                    st.session_state.filtered_data = None
                    st.session_state.selected_date_range = None
                    st.rerun()

def display_modern_rating_distribution(reviews, data_label):
    """Modern rating distribution display"""
    st.markdown(f"### ⭐ {data_label} Rating Breakdown")
    
    ratings = [r['rating'] for r in reviews if r['rating']]
    
    if ratings:
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        # Create modern visual rating distribution
        cols = st.columns(5)
        for i, (rating, count) in enumerate(sorted(rating_counts.items(), reverse=True)):
            with cols[i % 5]:
                percentage = (count / len(ratings)) * 100
                color = "#4CAF50" if rating >= 4 else "#FF9800" if rating >= 3 else "#F44336"
                
                st.markdown(f"""
                <div style="background: {color}; padding: 1rem; border-radius: 8px; 
                           color: white; text-align: center; margin-bottom: 1rem;">
                    <h3>{rating} ⭐</h3>
                    <h4>{count} reviews</h4>
                    <p>{percentage:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Data table for detailed view
        with st.expander("📊 Detailed Breakdown"):
            rating_df = pd.DataFrame([
                {
                    'Rating': f"{k} ⭐", 
                    'Count': v, 
                    'Percentage': f"{(v/len(ratings))*100:.1f}%",
                    'Bar': '█' * int((v/max(rating_counts.values()))*20)
                }
                for k, v in sorted(rating_counts.items(), reverse=True)
            ])
            
            st.dataframe(rating_df, use_container_width=True, hide_index=True)

def display_review_categories(categories):
    """Enhanced modern review categories display"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; margin: 2rem 0;">
        <h3>📊 AI Review Categorization</h3>
        <p>Customer feedback organized by themes and actionable insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create modern tabs
    tab1, tab2 = st.tabs(["🔴 Issues & Opportunities", "🟢 Strengths & Positives"])
    
    with tab1:
        # Issue categories with modern styling
        issue_categories = [
            'size_issues', 'quality_issues', 'noise_issues', 'fit_issues', 
            'durability_issues', 'value_issues'
        ]
        
        issues_found = []
        for category_key in issue_categories:
            if category_key in categories and categories[category_key]['count'] > 0:
                issues_found.append((category_key, categories[category_key]))
        
        if issues_found:
            # Sort by count (most common first)
            issues_found.sort(key=lambda x: x[1]['count'], reverse=True)
            
            for category_key, category_data in issues_found:
                category_name = category_key.replace('_', ' ').title()
                urgency_color = "#F44336" if category_data['count'] > 10 else "#FF9800" if category_data['count'] > 5 else "#FFC107"
                
                with st.expander(f"🔴 {category_name} ({category_data['count']} reviews)", expanded=category_data['count'] > 10):
                    st.markdown(f"""
                    <div style="background: {urgency_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {urgency_color};">
                        <h4>🎯 Most Common Pattern</h4>
                        <p><strong>{category_data['trend']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**📝 Example Customer Feedback:**")
                    for i, example in enumerate(category_data.get('reviews', category_data.get('examples', []))[:3]):
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 3px solid #dee2e6;">
                            <em>"{example}"</em>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    remaining = len(category_data.get('reviews', category_data.get('examples', []))) - 3
                    if remaining > 0:
                        st.caption(f"...and {remaining} more similar reviews")
        else:
            st.markdown("""
            <div style="background: #4CAF5020; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #4CAF50;">
                <h3>🎉 Excellent News!</h3>
                <p>No significant complaint categories identified in your reviews</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Positive categories
        positive_categories = ['positive_highlights', 'ease_of_use']
        
        positives_found = []
        for category_key in positive_categories:
            if category_key in categories and categories[category_key]['count'] > 0:
                positives_found.append((category_key, categories[category_key]))
        
        if positives_found:
            for category_key, category_data in positives_found:
                category_name = category_key.replace('_', ' ').title()
                
                with st.expander(f"🟢 {category_name} ({category_data['count']} reviews)", expanded=True):
                    st.markdown(f"""
                    <div style="background: #4CAF5020; padding: 1rem; border-radius: 8px; border-left: 4px solid #4CAF50;">
                        <h4>⭐ Customer Love</h4>
                        <p><strong>{category_data['trend']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**💬 Happy Customer Quotes:**")
                    for example in category_data.get('reviews', category_data.get('examples', []))[:3]:
                        st.markdown(f"""
                        <div style="background: #e8f5e8; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 3px solid #4CAF50;">
                            <em>"{example}"</em>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No specific positive highlight categories identified")
    
    # Modern summary with actionable insights
    total_categorized = sum(cat['count'] for cat in categories.values())
    sorted_categories = sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True)
    
    st.markdown("### 🎯 Action Priority Matrix")
    
    if len(sorted_categories) >= 3:
        top_3 = sorted_categories[:3]
        
        col1, col2, col3 = st.columns(3)
        for i, (category_key, category_data) in enumerate(top_3):
            category_name = category_key.replace('_', ' ').title()
            priority_colors = ["#F44336", "#FF9800", "#FFC107"]
            priority_labels = ["🔥 Critical", "⚠️ Important", "📝 Notable"]
            
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div style="background: {priority_colors[i]}20; padding: 1rem; border-radius: 8px; 
                           border: 2px solid {priority_colors[i]}; text-align: center;">
                    <h4>{priority_labels[i]}</h4>
                    <h5>{category_name}</h5>
                    <p><strong>{category_data['count']} reviews</strong></p>
                    <p><em>{category_data['trend'][:50]}...</em></p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown(f"**📈 Analysis Summary:** {total_categorized} reviews successfully categorized into actionable themes")

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
    """Main application with modern UI/UX"""
    try:
        # Page config with modern styling
        st.set_page_config(
            page_title="Amazon Review Analyzer",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Modern CSS styling
        st.markdown("""
        <style>
        /* Main app styling */
        .main > div {
            padding-top: 2rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* Card styling */
        .custom-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        
        /* Success styling */
        .success-banner {
            background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        /* Warning styling */
        .warning-banner {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        /* Metrics styling */
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            border: none;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: #f8f9fa;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session
        initialize_session_state()
        
        # Modern header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Amazon Review Analyzer</h1>
            <p>AI-powered listing optimization through customer feedback analysis</p>
            <p><em>Transform reviews into actionable insights • Version 5.1</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar with modern AI status
        display_modern_sidebar()
        
        # Main workflow with improved UX
        if st.session_state.current_step == 'upload':
            handle_modern_file_upload()
            
        elif st.session_state.current_step == 'analysis':
            display_modern_data_summary()
            
        elif st.session_state.current_step == 'results':
            display_modern_analysis_results()
            
            # Modern action buttons
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.current_step = 'upload'
                    st.session_state.uploaded_data = None
                    st.session_state.analysis_results = None
                    st.rerun()
            
            with col2:
                if st.button("💬 Discuss Results", use_container_width=True):
                    st.session_state.show_chat = True
                    st.rerun()
            
            with col3:
                if st.button("📊 View Data", use_container_width=True):
                    st.session_state.current_step = 'analysis'
                    st.rerun()
            
            with col4:
                if st.button("📥 Export", use_container_width=True):
                    st.info("📥 Export functionality coming soon!")
        
        # Modern processing indicator
        if st.session_state.processing:
            st.markdown("""
            <div class="success-banner">
                <h4>🔄 Processing Your Analysis...</h4>
                <p>Our AI is working hard to analyze your reviews. This may take a moment.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Modern error/success messages
        if st.session_state.error_message:
            st.markdown(f"""
            <div class="warning-banner">
                <h4>⚠️ Attention Needed</h4>
                <p>{st.session_state.error_message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.success_message:
            st.markdown(f"""
            <div class="success-banner">
                <h4>✅ Success!</h4>
                <p>{st.session_state.success_message}</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.markdown("""
        <div class="warning-banner">
            <h4>🚨 Application Error</h4>
            <p>An unexpected error occurred. Please refresh and try again.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 Technical Details"):
            st.exception(e)

def display_modern_sidebar():
    """Modern sidebar with enhanced AI status"""
    with st.sidebar:
        st.markdown("### 🤖 AI Intelligence Hub")
        
        status = check_ai_status()
        
        if status.get('available'):
            st.markdown("""
            <div style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); 
                        padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                <h4>✅ AI Ready</h4>
                <p>GPT-4o Analysis Active</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Modern AI Chat toggle
            if st.button("💬 Launch AI Chat", use_container_width=True):
                st.session_state.show_chat = True
                st.rerun()
        else:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                <h4>❌ AI Offline</h4>
                <p>Configuration Required</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Configuration help
            with st.expander("🔧 Setup Guide"):
                st.markdown("""
                **Quick Setup:**
                
                1. **Streamlit Secrets:**
                ```toml
                [secrets]
                openai_api_key = "your-key-here"
                ```
                
                2. **Environment Variable:**
                ```bash
                export OPENAI_API_KEY="your-key-here"
                ```
                
                3. **Restart the application**
                """)
        
        # Modern app info
        st.markdown("---")
        st.markdown("### 📊 App Features")
        features = [
            "🤖 AI-Powered Analysis",
            "📈 Rating Trends",
            "📅 Date Filtering", 
            "💬 Smart Chat",
            "📋 Review Categories",
            "🎯 Listing Optimization"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")

def handle_modern_file_upload():
    """Modern file upload interface"""
    
    # Optional listing information section with modern styling
    with st.expander("📝 Current Listing Information (Optional)", expanded=False):
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4>📋 Listing Context</h4>
            <p>Add your current listing details for comparative insights and optimization recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        current_title = st.text_area(
            "Current Listing Title:",
            value=st.session_state.current_listing_title,
            height=100,
            placeholder="Example: Premium Wireless Bluetooth Headphones - Noise Cancelling, 30Hr Battery..."
        )
        
        current_description = st.text_area(
            "Current Listing Description/Bullets:",
            value=st.session_state.current_listing_description,
            height=200,
            placeholder="• Feature 1: Premium sound quality with advanced drivers\n• Feature 2: Long-lasting 30-hour battery life\n• Feature 3: Active noise cancellation technology..."
        )
        
        if st.button("💾 Save Listing Context", use_container_width=True):
            st.session_state.current_listing_title = current_title
            st.session_state.current_listing_description = current_description
            st.success("✅ Listing information saved for analysis context")
    
    # Modern upload section
    st.markdown("""
    <div class="custom-card">
        <h3>📁 Upload Your Review Data</h3>
        <p>Upload Amazon review exports (CSV/Excel) for comprehensive AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with modern styling
    uploaded_file = st.file_uploader(
        "Choose your review export file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported: CSV, Excel (.xlsx, .xls) • Max size: 50MB"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if uploaded_file is not None:
            # Modern file info display
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4>📄 {uploaded_file.name}</h4>
                <p><strong>Size:</strong> {file_size_mb:.1f} MB • <strong>Type:</strong> {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Process & Analyze", type="primary", use_container_width=True):
                process_uploaded_file(uploaded_file)
    
    with col2:
        if st.button("📊 Try Demo Data", use_container_width=True):
            load_example_reviews()
    
    # Modern help section
    with st.expander("❓ Need Help?"):
        st.markdown("""
        **Supported File Formats:**
        - ✅ CSV files from Amazon exports
        - ✅ Excel files (.xlsx, .xls)
        - ✅ Helium 10 review exports
        
        **Required Columns:**
        - `Title` - Review titles
        - `Body` - Review content
        - `Rating` - Star ratings (1-5)
        - `Date` - Review dates
        
        **Tips for Best Results:**
        - Include 50+ reviews for comprehensive analysis
        - Ensure review text is in English (multilingual support coming)
        - Add your current listing info for better optimization insights
        """)

def display_modern_data_summary():
    """Modern data summary with enhanced UX"""
    if not st.session_state.uploaded_data:
        return
    
    data = st.session_state.uploaded_data
    product_info = data['product_info']
    reviews = data['reviews']
    
    st.markdown("## 📊 Review Data Dashboard")
    
    # Show AI chat option
    display_ai_chat()
    
    # Date filtering section with modern styling
    if data.get('date_range'):
        display_modern_date_filtering(data)
    
    # Use filtered data if available
    current_reviews = st.session_state.filtered_data if st.session_state.filtered_data else reviews
    current_data_label = "Filtered Analysis" if st.session_state.filtered_data else "Complete Dataset"
    
    # Modern metrics dashboard
    st.markdown(f"### 📈 {current_data_label} Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(current_reviews)}</h2>
            <p>Total Reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ratings = [r['rating'] for r in current_reviews if r['rating']]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        rating_color = "#4CAF50" if avg_rating >= 4 else "#FF9800" if avg_rating >= 3 else "#F44336"
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {rating_color}">{avg_rating:.1f}/5</h2>
            <p>Average Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        verified_count = sum(1 for r in current_reviews if r.get('verified'))
        verification_rate = (verified_count / len(current_reviews)) * 100 if current_reviews else 0
        st.markdown(f"""
        <div class="metric-card">
            <h2>{verified_count}</h2>
            <p>Verified ({verification_rate:.0f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{product_info['asin']}</h2>
            <p>Product ASIN</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Rating trends with modern display
    if data.get('rating_trends') and data['rating_trends'].get('success'):
        display_rating_trends(data['rating_trends'])
    
    # Modern rating distribution
    if ratings:
        display_modern_rating_distribution(current_reviews, current_data_label)
    
    # Modern analysis CTA
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
        <h3>🤖 Ready for AI Analysis?</h3>
        <p>Transform your customer feedback into actionable listing optimization insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch AI Analysis", type="primary", use_container_width=True):
        run_ai_analysis()

def display_modern_analysis_results():
    """Modern analysis results display"""
    if not st.session_state.analysis_results:
        st.error("No analysis results available")
        return
    
    results = st.session_state.analysis_results
    
    # Modern results header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); 
                padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2>🤖 AI Analysis Complete</h2>
        <p>Your comprehensive review analysis is ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show AI chat for discussing results
    display_ai_chat()
    
    if results.get('ai_analysis'):
        display_comprehensive_ai_results(results)
    else:
        display_basic_results(results)

if __name__ == "__main__":
    main()
