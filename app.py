import ocr_processor
import data_analysis
import import_template
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import re
import logging
import requests
import base64
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Optional libraries for PDF/image processing
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SUPPORT_EMAIL = "alexander.popoff@vivehealth.com"
DEFAULT_MODEL = "gpt-4o"

# Initialize session state variables
if 'openai_api_connected' not in st.session_state:
    st.session_state.openai_api_connected = False
    
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'ai_model': DEFAULT_MODEL,
        'max_tokens': 1500,
        'temperature': 0.7
    }
    
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
    
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'current_product' not in st.session_state:
    st.session_state.current_product = None

#=========================================================================
# API Integration
#=========================================================================
def test_openai_connection() -> bool:
    """Test connection to OpenAI API."""
    try:
        # Check if API key is in streamlit secrets
        if 'openai_api_key' in st.secrets:
            api_key = st.secrets['openai_api_key']
            
            # Simple test request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to OpenAI API")
                return True
            else:
                logger.error(f"Failed to connect to OpenAI API: {response.status_code}")
                return False
        else:
            logger.warning(f"OpenAI API key not found in streamlit secrets. Please contact {SUPPORT_EMAIL}")
            return False
    except Exception as e:
        logger.error(f"Error testing OpenAI connection: {str(e)}")
        return False

def call_openai_api(messages, model=None, max_tokens=800):
    """Call OpenAI API with messages."""
    try:
        if model is None:
            model = st.session_state.settings.get('ai_model', DEFAULT_MODEL)
        
        api_key = st.secrets.get('openai_api_key', '')
        if not api_key:
            logger.warning(f"OpenAI API key not found. Please contact {SUPPORT_EMAIL}")
            return f"AI tools not currently available. Please contact {SUPPORT_EMAIL} to resolve this issue."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": st.session_state.settings.get('temperature', 0.7)
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return f"Error: AI assistant encountered a problem (HTTP {response.status_code}). Please contact {SUPPORT_EMAIL} to resolve this issue."
        
    except Exception as e:
        logger.exception(f"Error calling OpenAI API: {str(e)}")
        return f"Error: The AI assistant encountered an unexpected problem. Please contact {SUPPORT_EMAIL} to resolve this issue."

#=========================================================================
# Utility Functions
#=========================================================================
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default if divisor is zero."""
    try:
        if b == 0:
            return default
        return a / b
    except Exception as e:
        logger.error(f"Error in safe_divide: {str(e)}")
        return default

def format_currency(value: float) -> str:
    """Format a value as currency."""
    if pd.isna(value) or value is None:
        return "-"
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """Format a value as percentage."""
    if pd.isna(value) or value is None:
        return "-"
    return f"{value:.2f}%"

def format_number(value: float, decimals: int = 2) -> str:
    """Format a value as number with commas."""
    if pd.isna(value) or value is None:
        return "-"
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"

def show_toast(message: str, type: str = "success"):
    """Show a toast notification to the user."""
    toast_types = {
        "success": "#6cc24a",
        "error": "#d9534f",
        "warning": "#f7941d",
        "info": "#5bc0de"
    }
    toast_color = toast_types.get(type, toast_types["info"])
    
    st.markdown(f"""
    <div style="position: fixed; bottom: 20px; right: 20px; padding: 12px 24px; 
    background-color: {toast_color}; color: white; border-radius: 8px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 9999;">
    {message}
    </div>
    <script>
    setTimeout(function() {{
        document.querySelector('div[style*="position: fixed"]').style.display = 'none';
    }}, 3000);
    </script>
    """, unsafe_allow_html=True)

def to_excel(df: pd.DataFrame, sheet_name='Product Analysis'):
    """Convert dataframe to Excel file for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Format the Excel sheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Set column widths
        for i, col in enumerate(df.columns):
            column_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_width)
        
        output.seek(0)
    return output

def create_sample_import_template():
    """Create a sample import template for users to download."""
    return import_template.create_import_template()

#=========================================================================
# File Processing Functions
#=========================================================================
def process_pdf_with_ocr(pdf_file):
    pdf_bytes = pdf_file.read()
    return ocr_processor.process_pdf_with_ocr(pdf_bytes)
    
def process_image_with_ocr(image_file):
    image_bytes = image_file.read()
    return ocr_processor.process_image_with_ocr(image_bytes)
    
def extract_reviews_from_ocr(ocr_text):
    return ocr_processor.extract_amazon_reviews_data(ocr_text)

#=========================================================================
# Analysis Functions
#=========================================================================
def categorize_reviews(reviews_data):
    """Categorize reviews using AI analysis."""
    try:
        # Extract necessary data
        reviews_sample = reviews_data.head(100).to_dict('records') if isinstance(reviews_data, pd.DataFrame) else reviews_data[:100]
        
        # Create system prompt
        system_prompt = """You are an expert medical device review analyst for Vive Health. 
        Categorize each review by topic, sentiment, and specific issue category.
        Your categories should be consistent and focused on medical device-specific concerns."""
        
        # Create user prompt
        user_prompt = f"""
        Please categorize the following {len(reviews_sample)} product reviews.
        
        For each review, provide:
        1. Primary Topic: Product Quality, Usability, Instructions, Packaging, Customer Service, Pricing, Shipping
        2. Sentiment: Positive, Neutral, Negative
        3. Issue Category: Functionality, Durability, Comfort, Safety, Appearance, Documentation, Size/Fit, None
        
        Return the results in a structured JSON format that can be easily parsed.
        """
        
        # Create messages for API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n\nReviews: " + json.dumps(reviews_sample)}
        ]
        
        # Get AI response
        response = call_openai_api(messages, model=DEFAULT_MODEL, max_tokens=2000)
        
        # Parse and process response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                categorized_data = json.loads(json_match.group(1))
            else:
                # Try to parse the whole response as JSON
                categorized_data = json.loads(response)
                
            return categorized_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return {"error": "Failed to parse JSON response", "raw_response": response}
        
    except Exception as e:
        logger.error(f"Error categorizing reviews: {str(e)}")
        return {"error": str(e)}

def analyze_product_reviews(review_data, product_info, historical_data=None):
    """Generate AI-powered analysis of product reviews and returns for Vive Health medical devices."""
    try:
        # Create system prompt with context
        system_prompt = """You are an expert medical device product analyst specializing in customer feedback analysis for Vive Health.
        Provide detailed, actionable insights on customer reviews, ratings, and return reasons for medical devices.
        Focus on identifying specific product issues, listing problems, and documentation gaps that impact customer satisfaction.
        
        Vive Health's product categories include mobility aids, pain relief devices, bathroom safety equipment, sleep & comfort products,
        fitness & recovery items, daily living aids, and respiratory care devices. Each category has unique customer needs, use cases,
        regulatory considerations, and common issues.
        
        Your analysis should categorize all feedback, identify patterns, and provide specific, actionable recommendations that
        consider both business impact and medical device regulatory requirements. Focus on helping Vive Health improve both
        their products and their product listings.
        """
        
        # Create user prompt with review data
        user_prompt = f"""
        Please analyze the following Vive Health medical device product data:
        
        Product: {product_info['name']} (SKU: {product_info.get('sku', 'N/A')}, ASIN: {product_info.get('asin', 'N/A')})
        Category: {product_info.get('category', 'Medical Device')}
        
        The data contains {len(review_data)} customer reviews, ratings, and return reasons.
        
        Please provide a comprehensive analysis including:
        
        1. Overview & Summary:
        - Overall sentiment distribution (positive, neutral, negative)
        - Average rating and rating distribution
        - Most common return reasons
        - Key themes identified across all feedback
        
        2. Detailed Analysis:
        - Top issues mentioned in negative reviews (categorized by product, listing, instructions, packaging)
        - Top positive aspects mentioned in favorable reviews
        - Specific technical or functional problems identified
        - Usability and accessibility concerns for this medical device
        - Competing products mentioned and comparative feedback
        - Documentation or instruction issues
        - Safety concerns or potential regulatory issues
        
        3. Actionable Recommendations:
        - Product design or functionality improvements
        - Product listing and description enhancements
        - Instruction manual or documentation updates
        - Packaging improvements
        - Customer education opportunities
        
        4. Implementation Classification:
        For each recommendation, categorize as:
        - High risk, high effort, high reward 
        - Low effort, high reward
        - Medium risk, medium effort, medium reward
        - Low risk, low effort, low reward
        
        Include specific quotes from reviews to support key findings where relevant. Focus on actionable insights
        that can drive measurable improvements in customer satisfaction, reduced returns, and higher ratings.
        """
        
        # Add historical data analysis if provided
        if historical_data is not None and len(historical_data) > 0:
            user_prompt += f"""
            
            Additionally, analyze the historical data provided containing {len(historical_data)}
            previous reviews/returns to identify:
            
            5. Trend Analysis:
            - Changes in sentiment over time
            - Recurring issues that haven't been resolved
            - New emerging issues
            - Impact of previous product improvements
            - Seasonal patterns in feedback or returns
            - Progress on previously identified issues
            """
        
        # Create messages for API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Get AI response
        return call_openai_api(messages, model=DEFAULT_MODEL, max_tokens=1500)
    except Exception as e:
        logger.error(f"Error analyzing product reviews: {str(e)}")
        return f"Error generating product analysis. Please contact {SUPPORT_EMAIL} for assistance."

def generate_improvement_recommendations(analysis_results, product_info):
    """Generate specific improvement recommendations based on analysis results."""
    try:
        # Create system prompt
        system_prompt = """You are a medical device product improvement specialist for Vive Health.
        Your task is to generate specific, actionable recommendations for product and listing improvements
        based on customer feedback analysis. Focus on regulatory-compliant, practical solutions that
        address the most impactful issues."""
        
        # Create user prompt
        user_prompt = f"""
        Based on the following analysis of customer feedback for {product_info['name']} (a {product_info.get('category', 'medical device')}),
        provide detailed, specific recommendations for improvements in the following areas:
        
        1. Product Design/Functionality
        2. Product Listing/Description
        3. Instructions/Documentation
        4. Packaging
        
        For each recommendation:
        - Describe the specific change in detail
        - Explain how it addresses customer concerns
        - Categorize as: High risk/high effort/high reward, Low effort/high reward, Medium risk/medium effort/medium reward, or Low risk/low effort/low reward
        - Provide an implementation priority (1-5, where 1 is highest priority)
        - Note any regulatory or compliance considerations
        
        Analysis Summary:
        {analysis_results}
        """
        
        # Create messages for API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Get AI response
        response = call_openai_api(messages, model=DEFAULT_MODEL, max_tokens=1500)
        
        return response
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return None

#=========================================================================
# UI Components
#=========================================================================
def render_header():
    """Render the application header."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Product Review Analysis Tool")
        st.markdown("Analyze product reviews, ratings, and returns for Vive Health medical devices.")
    with col2:
        # Display API connection status
        if st.session_state.openai_api_connected:
            st.success("API Connected")
        else:
            st.error("API Disconnected")
            
    # Test API connection
    if not st.session_state.openai_api_connected:
        if test_openai_connection():
            st.session_state.openai_api_connected = True
            show_toast("Successfully connected to OpenAI API", "success")
            st.rerun()

def render_sidebar():
    """Render the sidebar with settings and information."""
    with st.sidebar:
        st.header("Settings")
        
        # API Model selection
        st.session_state.settings['ai_model'] = st.selectbox(
            "AI Model",
            options=["gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        
        # Max tokens
        st.session_state.settings['max_tokens'] = st.slider(
            "Max Response Tokens",
            min_value=500,
            max_value=4000,
            value=st.session_state.settings.get('max_tokens', 1500),
            step=100
        )
        
        # Temperature
        st.session_state.settings['temperature'] = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings.get('temperature', 0.7),
            step=0.1
        )
        
        st.markdown("---")
        
        # Export sample template
        sample_template = create_sample_import_template()
        st.download_button(
            label="Download Sample Template",
            data=sample_template,
            file_name="product_analysis_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("---")
        
        # Support info
        st.subheader("Support")
        st.markdown(f"For assistance, contact: [{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL})")

def render_file_upload():
    """Render the file upload section."""
    st.header("Import Data")
    
    tabs = st.tabs(["Structured Data Import", "Document Import", "Historical Data"])
    
    # Tab 1: Structured Data Import
    with tabs[0]:
        st.markdown("""
        Upload a CSV or Excel file with product data. Required columns:
        - **ASIN*** (Mandatory)
        - **Last 30 Days Sales*** (Mandatory)
        - **Last 30 Days Returns*** (Mandatory)
        - SKU (Optional)
        - Last 365 Days Sales (Optional)
        - Last 365 Days Returns (Optional)
        - Star Rating (Optional)
        - Total Reviews (Optional)
        """)
        
        uploaded_file = st.file_uploader(
            "Upload product data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="structured_data"
        )
        
        if uploaded_file:
            with st.spinner("Processing data file..."):
                df, error = process_excel_or_csv(uploaded_file)
                
                if error:
                    st.error(error)
                else:
                    st.session_state.uploaded_files['structured_data'] = df
                    st.success(f"Successfully processed file with {len(df)} products.")
                    st.dataframe(df.head())
    
    # Tab 2: Document Import
    with tabs[1]:
        st.markdown("""
        Upload PDFs, images of reviews, return reports, or screenshots. 
        The system will use OCR to extract data when possible.
        """)
        
        doc_files = st.file_uploader(
            "Upload documents (PDF, Images)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="documents"
        )
        
        if doc_files:
            # Process each document
            processed_docs = []
            
            for doc in doc_files:
                with st.spinner(f"Processing {doc.name}..."):
                    # Process based on file type
                    file_ext = doc.name.split('.')[-1].lower()
                    
                    if file_ext == 'pdf':
                        text = process_pdf_with_ocr(doc)
                    elif file_ext in ['png', 'jpg', 'jpeg']:
                        text = process_image_with_ocr(doc)
                    else:
                        text = f"Unsupported document type: {file_ext}"
                        
                    processed_docs.append({
                        "filename": doc.name,
                        "text": text,
                        "type": file_ext
                    })
            
            if processed_docs:
                st.session_state.uploaded_files['documents'] = processed_docs
                
                # Show preview of extracted text from first document
                if len(processed_docs) > 0:
                    with st.expander("Preview of extracted text"):
                        st.text(processed_docs[0]["text"][:1000] + "...")
    
    # Tab 3: Historical Data
    with tabs[2]:
        st.markdown("""
        Upload historical data for trend analysis (optional).
        Use the same format as the structured data import.
        """)
        
        hist_file = st.file_uploader(
            "Upload historical data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="historical_data"
        )
        
        if hist_file:
            with st.spinner("Processing historical data..."):
                hist_df, error = process_excel_or_csv(hist_file)
                
                if error:
                    st.error(error)
                else:
                    st.session_state.uploaded_files['historical_data'] = hist_df
                    st.success(f"Successfully processed historical data with {len(hist_df)} entries.")
                    st.dataframe(hist_df.head())

def render_product_selection():
    """Render the product selection section."""
    st.header("Select Product to Analyze")
    
    # Check if we have structured data
    if 'structured_data' not in st.session_state.uploaded_files:
        st.warning("Please upload structured product data first.")
        return
    
    df = st.session_state.uploaded_files['structured_data']
    
    # Create display columns based on available data
    display_cols = ['ASIN', 'SKU'] if 'SKU' in df.columns else ['ASIN']
    if 'Product Name' in df.columns:
        display_cols.append('Product Name')
    
    # Add star rating if available
    if 'Star Rating' in df.columns:
        display_cols.append('Star Rating')
    
    # Display product table for selection
    selected_index = st.selectbox(
        "Select a product to analyze:",
        options=range(len(df)),
        format_func=lambda i: " | ".join([str(df.iloc[i][col]) for col in display_cols])
    )
    
    # Create product info dictionary
    product_row = df.iloc[selected_index]
    product_info = {
        'asin': product_row['ASIN'],
        'sku': product_row['SKU'] if 'SKU' in product_row else "N/A",
        'name': product_row['Product Name'] if 'Product Name' in product_row else f"Product {product_row['ASIN']}",
        'category': product_row['Category'] if 'Category' in product_row else "Medical Device",
        'star_rating': product_row['Star Rating'] if 'Star Rating' in product_row else None,
        'total_reviews': product_row['Total Reviews'] if 'Total Reviews' in product_row else None,
        'sales_30d': product_row['Last 30 Days Sales'],
        'returns_30d': product_row['Last 30 Days Returns'],
        'sales_365d': product_row['Last 365 Days Sales'] if 'Last 365 Days Sales' in product_row else None,
        'returns_365d': product_row['Last 365 Days Returns'] if 'Last 365 Days Returns' in product_row else None,
    }
    
    # Calculate metrics
    product_info['return_rate_30d'] = safe_divide(product_info['returns_30d'], product_info['sales_30d']) * 100
    if product_info['sales_365d'] is not None and product_info['returns_365d'] is not None:
        product_info['return_rate_365d'] = safe_divide(product_info['returns_365d'], product_info['sales_365d']) * 100
    
    # Store selected product in session state
    st.session_state.current_product = product_info
    
    # Display selected product info in metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("30-Day Sales", format_number(product_info['sales_30d'], 0))
    with col2:
        st.metric("30-Day Returns", format_number(product_info['returns_30d'], 0))
    with col3:
        st.metric("30-Day Return Rate", format_percent(product_info['return_rate_30d']))
    with col4:
        if product_info['star_rating'] is not None:
            st.metric("Star Rating", f"{product_info['star_rating']:.1f} ★")
    
    # Get review data for the selected product from documents if available
    if 'documents' in st.session_state.uploaded_files:
        docs = st.session_state.uploaded_files['documents']
        reviews_data = []
        
        for doc in docs:
            # Extract reviews from OCR text
            text = doc["text"]
            extracted_reviews = extract_reviews_from_ocr(text)
            reviews_data.extend(extracted_reviews)
        
        st.write(f"Found {len(reviews_data)} reviews in uploaded documents.")
    
    # Get historical data for the selected product if available
    historical_data = None
    if 'historical_data' in st.session_state.uploaded_files:
        hist_df = st.session_state.uploaded_files['historical_data']
        # Filter for the selected product if ASIN column exists
        if 'ASIN' in hist_df.columns:
            hist_filtered = hist_df[hist_df['ASIN'] == product_info['asin']]
            if len(hist_filtered) > 0:
                historical_data = hist_filtered
                st.write(f"Found {len(historical_data)} historical data points for this product.")
    
    # Run analysis button
    if st.button("Analyze Product", type="primary"):
        # Check if we have both product info and review data
        if st.session_state.current_product:
            with st.spinner("Analyzing product reviews and returns..."):
                # Run analysis
                analysis_result = analyze_product_reviews(
                    reviews_data if 'reviews_data' in locals() else [],
                    product_info,
                    historical_data
                )
                
                # Store result in session state
                st.session_state.analysis_results[product_info['asin']] = {
                    'product_info': product_info,
                    'analysis': analysis_result,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Generate recommendations
                recommendations = generate_improvement_recommendations(analysis_result, product_info)
                st.session_state.analysis_results[product_info['asin']]['recommendations'] = recommendations
                
                show_toast("Analysis complete", "success")
                st.rerun()
        else:
            st.error("Please select a product to analyze.")

def render_analysis_results():
    """Render the analysis results section."""
    st.header("Analysis Results")
    
    # Check if we have results to display
    if not st.session_state.analysis_results:
        st.info("No analysis results available. Please select and analyze a product.")
        return
    
    # If we have a current product selected, display its results
    if st.session_state.current_product:
        product_asin = st.session_state.current_product['asin']
        
        if product_asin in st.session_state.analysis_results:
            result = st.session_state.analysis_results[product_asin]
            product_info = result['product_info']
            analysis = result['analysis']
            recommendations = result.get('recommendations', None)
            
            # Display info
            st.subheader(f"Analysis for {product_info['name']} ({product_info['asin']})")
            st.caption(f"Analyzed on: {result['timestamp']}")
            
            # Create tabs for different sections of the analysis
            tabs = st.tabs(["Summary", "Detailed Analysis", "Recommendations", "Visualizations", "Export"])
            
            # Tab 1: Summary
            with tabs[0]:
                st.markdown(analysis)
            
            # Tab 2: Detailed Analysis
            with tabs[1]:
                if recommendations:
                    st.markdown(recommendations)
                else:
                    st.markdown(analysis)
            
            # Tab 3: Recommendations
            with tabs[2]:
                # Create a dataframe to display recommendations in a structured way
                st.subheader("Prioritized Recommendations")
                
                # This is a placeholder - in a real implementation, you would parse the 
                # recommendations from the AI response and structure them
                recommendations_data = [
                    {
                        "Category": "Product Design", 
                        "Recommendation": "Improve stability of the walker by widening the base", 
                        "Effort/Risk/Reward": "Medium risk, medium effort, high reward",
                        "Priority": "1"
                    },
                    {
                        "Category": "Listing", 
                        "Recommendation": "Update product description to clearly state weight capacity", 
                        "Effort/Risk/Reward": "Low risk, low effort, high reward",
                        "Priority": "1"
                    },
                    {
                        "Category": "Documentation", 
                        "Recommendation": "Provide clearer assembly instructions with diagrams", 
                        "Effort/Risk/Reward": "Low risk, medium effort, high reward",
                        "Priority": "2"
                    },
                ]
                
                rec_df = pd.DataFrame(recommendations_data)
                st.dataframe(rec_df)
                
                # Action plan
                st.subheader("Implementation Action Plan")
                st.markdown("""
                Based on the recommendations, here's a suggested action plan:
                
                1. **Immediate Actions** (Next 30 days):
                   - Update product listing with clear weight capacity information
                   - Create improved assembly instructions
                
                2. **Medium-term Actions** (60-90 days):
                   - Redesign walker base for increased stability
                   - Update packaging to better protect the product during shipping
                
                3. **Long-term Actions** (90+ days):
                   - Develop new size variations to accommodate different user needs
                   - Consider additional features based on competitive analysis
                """)
            
            # Tab 4: Visualizations
            with tabs[3]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Return reasons chart (placeholder data)
                    return_reasons = {
                        "Item defective or doesn't work": 35,
                        "Bought by mistake": 25,
                        "Item arrived damaged": 15,
                        "Performance not adequate": 10,
                        "Better price available": 5,
                        "Other": 10
                    }
                    
                    fig1 = px.pie(
                        values=list(return_reasons.values()),
                        names=list(return_reasons.keys()),
                        title="Return Reasons"
                    )
                    st.plotly_chart(fig1)
                
                with col2:
                    # Sentiment analysis chart (placeholder data)
                    sentiment = {
                        "Positive": 60,
                        "Neutral": 15,
                        "Negative": 25
                    }
                    
                    fig2 = px.bar(
                        x=list(sentiment.keys()),
                        y=list(sentiment.values()),
                        title="Review Sentiment",
                        color=list(sentiment.keys()),
                        color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'}
                    )
                    st.plotly_chart(fig2)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Issue categories (placeholder data)
                    issues = {
                        "Stability": 40,
                        "Size/Fit": 25,
                        "Assembly": 15,
                        "Durability": 10,
                        "Packaging": 5,
                        "Other": 5
                    }
                    
                    fig3 = px.bar(
                        x=list(issues.keys()),
                        y=list(issues.values()),
                        title="Issue Categories",
                        color=list(issues.keys())
                    )
                    st.plotly_chart(fig3)
                
                with col4:
                    # Star rating distribution (placeholder data)
                    stars = {
                        "5 ★": 10,
                        "4 ★": 5,
                        "3 ★": 3,
                        "2 ★": 1,
                        "1 ★": 1
                    }
                    
                    fig4 = px.bar(
                        x=list(stars.keys()),
                        y=list(stars.values()),
                        title="Star Rating Distribution",
                        color=list(stars.keys()),
                        color_discrete_map={
                            "5 ★": "darkgreen",
                            "4 ★": "lightgreen",
                            "3 ★": "gold",
                            "2 ★": "orange",
                            "1 ★": "red"
                        }
                    )
                    st.plotly_chart(fig4)
            
            # Tab 5: Export
            with tabs[4]:
                st.subheader("Export Analysis")
                
                # Export to PDF is typically more complex in Streamlit
                # This is a placeholder for where you would implement PDF export functionality
                st.download_button(
                    label="Export as PDF Report",
                    data="PDF export not implemented in this demo",
                    file_name=f"{product_info['asin']}_analysis.pdf",
                    mime="application/pdf",
                    disabled=True
                )
                
                # Export to Excel (simplified)
                export_data = {
                    "Product Info": {
                        "ASIN": product_info['asin'],
                        "SKU": product_info['sku'],
                        "Name": product_info['name'],
                        "Category": product_info['category'],
                        "Star Rating": product_info.get('star_rating', "N/A"),
                        "30 Day Sales": product_info['sales_30d'],
                        "30 Day Returns": product_info['returns_30d'],
                        "30 Day Return Rate": product_info['return_rate_30d'],
                    },
                    "Analysis Summary": analysis,
                    "Recommendations": recommendations if recommendations else ""
                }
                
                # Convert to DataFrame for Excel export
                export_df = pd.DataFrame([export_data])
                
                excel_data = to_excel(export_df, f"{product_info['asin']}_Analysis")
                
                st.download_button(
                    label="Export as Excel Report",
                    data=excel_data,
                    file_name=f"{product_info['asin']}_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Export raw data
                st.download_button(
                    label="Export Raw JSON Data",
                    data=json.dumps(result, indent=2),
                    file_name=f"{product_info['asin']}_raw_data.json",
                    mime="application/json"
                )
        else:
            st.warning(f"No analysis results available for {st.session_state.current_product['name']}. Please run analysis first.")
    else:
        # If no current product, let the user select from available analyses
        available_analyses = list(st.session_state.analysis_results.keys())
        if available_analyses:
            selected_asin = st.selectbox("Select a previously analyzed product:", available_analyses)
            if st.button("Show Analysis"):
                # Set the selected product as current
                st.session_state.current_product = st.session_state.analysis_results[selected_asin]['product_info']
                st.rerun()
        else:
            st.info("No analysis results available. Please select and analyze a product.")

def render_help_section():
    """Render the help and documentation section."""
    st.header("Help & Documentation")
    
    help_tabs = st.tabs(["User Guide", "FAQ", "Examples", "Support"])
    
    with help_tabs[0]:
        st.subheader("Getting Started")
        st.markdown("""
        1. **Import Data**: Start by uploading your product data file in CSV or Excel format.
        2. **Upload Documents**: If you have PDF reports or screenshots, upload them for OCR processing.
        3. **Select Product**: Choose the product you want to analyze.
        4. **Run Analysis**: Click the "Analyze Product" button to process the data.
        5. **Review Results**: Explore the analysis, recommendations, and visualizations.
        6. **Export Reports**: Download the analysis in your preferred format.
        """)
        
        st.subheader("Required Data Format")
        st.markdown("""
        The application requires certain fields to perform analysis:
        
        * **Mandatory fields** (marked with *):
          * ASIN*
          * Last 30 Days Sales*
          * Last 30 Days Returns*
        
        * **Recommended fields** (will improve analysis):
          * SKU
          * Product Name
          * Category
          * Star Rating
          * Total Reviews
          * Last 365 Days Sales
          * Last 365 Days Returns
        
        You can download a sample template from the sidebar.
        """)
    
    with help_tabs[1]:
        st.subheader("Frequently Asked Questions")
        
        with st.expander("What types of files can I upload?"):
            st.markdown("""
            The application supports:
            
            * **Structured Data**: CSV, Excel (.xlsx, .xls)
            * **Documents**: PDF, PNG, JPG, JPEG
            
            For best results, ensure your CSV/Excel files follow the template format.
            """)
        
        with st.expander("How does the OCR functionality work?"):
            st.markdown("""
            The OCR (Optical Character Recognition) function extracts text from:
            
            * PDF reports
            * Screenshots of return data
            * Images of product listings
            
            The extracted text is then processed to identify reviews, return reasons, 
            and other relevant information for analysis.
            
            For the best OCR results:
            
            * Use high-resolution images
            * Ensure text is clearly visible
            * Avoid images with complex backgrounds
            """)
        
        with st.expander("What if I don't have all the required data?"):
            st.markdown("""
            The application requires at minimum:
            
            * ASIN
            * Last 30 Days Sales
            * Last 30 Days Returns
            
            Without these minimum fields, analysis cannot be completed. If you're missing 
            other fields, the application will still function but with potentially less 
            detailed analysis.
            """)
        
        with st.expander("How accurate is the AI analysis?"):
            st.markdown("""
            The AI analysis uses advanced GPT-4o technology to evaluate review content and 
            return reasons. While generally highly accurate, it works best with:
            
            * Larger datasets (more reviews provide better insights)
            * Clear review text (vague reviews are harder to categorize)
            * English language content (non-English may have reduced accuracy)
            
            Always review AI recommendations in the context of your product knowledge.
            """)
    
    with help_tabs[2]:
        st.subheader("Example Data")
        
        st.markdown("### Example 1: CSV Product Data")
        
        example_data = pd.DataFrame({
            "SKU": ["MOB1116BLU"],
            "ASIN": ["B0DT7NW5VY"],
            "Product Name": ["Tri-Rollator With Seat"],
            "Category": ["Mobility Aids"],
            "Last 30 Days Sales": [491],
            "Last 30 Days Returns": [10],
            "Last 365 Days Sales": [5840],
            "Last 365 Days Returns": [67],
            "Star Rating": [3.9],
            "Total Reviews": [20]
        })
        
        st.dataframe(example_data)
        
        st.markdown("### Example 2: Return Reasons")
        
        example_returns = pd.DataFrame({
            "Order ID": ["114-1106156-2607429", "113-1770004-6998632", "114-6826075-5417831"],
            "Return Reason": ["Item defective or doesn't work", "Seat too small", "Bars seem defective"],
            "Return Date": ["05/21/2025", "05/09/2025", "05/13/2025"]
        })
        
        st.dataframe(example_returns)
        
        st.markdown("### Example 3: Customer Review")
        
        st.markdown("""
        > "Vive 3-wheel Walker with seat was put together my mom went to go use it and it folded in she almost fell. She's a 125 lb the seating is not comfortable for her. I want a full refund if she's barely seating in the seat I know a 300 lb person cannot fit."
        >
        > Star Rating: ★★☆☆☆
        """)
    
    with help_tabs[3]:
        st.subheader("Support Resources")
        
        st.markdown(f"""
        If you encounter issues or need assistance:
        
        * **Contact Support**: Email [{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL})
        * **Report Bugs**: Please include screenshots and step-by-step reproduction steps
        * **Request Features**: We welcome suggestions for new features or improvements
        
        For API connection issues, ensure your Streamlit secrets.toml file contains a valid OpenAI API key.
        """)

#=========================================================================
# Main Application
#=========================================================================
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Product Review Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main sections
    tabs = st.tabs(["Import", "Analyze", "Results", "Help"])
    
    with tabs[0]:
        render_file_upload()
    
    with tabs[1]:
        render_product_selection()
    
    with tabs[2]:
        render_analysis_results()
    
    with tabs[3]:
        render_help_section()

if __name__ == "__main__":
    main()
