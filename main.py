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
    df = pd.DataFrame({
        "SKU*": ["MOB1116BLU", "BAT2234RED"],
        "ASIN*": ["B0DT7NW5VY", "B0DT8XYZ123"],
        "Product Name": ["Tri-Rollator With Seat", "Vive Shower Chair"],
        "Category": ["Mobility Aids", "Bathroom Safety"],
        "Last 30 Days Sales*": [491, 325],
        "Last 30 Days Returns*": [10, 8],
        "Last 365 Days Sales": [5840, 3900],
        "Last 365 Days Returns": [67, 45],
        "Star Rating": [3.9, 4.2],
        "Total Reviews": [20, 35],
        "Average Price": [89.99, 59.99]
    })
    return to_excel(df, "Import Template")

#=========================================================================
# File Processing Functions
#=========================================================================
def process_excel_or_csv(uploaded_file):
    """Process uploaded Excel or CSV file."""
    try:
        # Determine file type by extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file format: {file_ext}. Please upload CSV or Excel files."
        
        # Validate required columns
        required_columns = ['ASIN', 'Last 30 Days Sales', 'Last 30 Days Returns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        return df, None
    except Exception as e:
        logger.error(f"Error processing Excel/CSV file: {str(e)}")
        return None, f"Error processing file: {str(e)}"

def process_pdf_with_ocr(pdf_file):
    """Extract text from PDF using OCR."""
    if not OCR_AVAILABLE:
        return "OCR libraries not available. Please install pytesseract and pdf2image."
    
    try:
        # Read file bytes
        pdf_bytes = pdf_file.read()
        
        # Convert PDF to images
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_bytes(pdf_bytes, dpi=300, output_folder=path)
            
            # Extract text from each image
            text_content = []
            for img in images:
                text = pytesseract.image_to_string(img)
                text_content.append(text)
        
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error processing PDF with OCR: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def process_image_with_ocr(image_file):
    """Extract text from image using OCR."""
    if not OCR_AVAILABLE:
        return "OCR libraries not available. Please install pytesseract."
    
    try:
        # Open image file
        img = Image.open(image_file)
        
        # Convert to text using OCR
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.error(f"Error processing image with OCR: {str(e)}")
        return f"Error processing image: {str(e)}"

def extract_reviews_from_ocr(ocr_text):
    """Extract structured review data from OCR text."""
    # This is a placeholder function that would need to be customized 
    # based on the exact format of the OCR output
    reviews = []
    
    # Example pattern for extracting reviews (would need to be adjusted)
    order_pattern = r"Order ID:\s+([A-Za-z0-9\-]+)"
    comment_pattern = r"Buyer Comment:\s+(.*?)(?:Request Date|$)"
    
    order_ids = re.findall(order_pattern, ocr_text)
    comments = re.findall(comment_pattern, ocr_text, re.DOTALL)
    
    # Match order IDs with comments when possible
    for i in range(min(len(order_ids), len(comments))):
        reviews.append({
            "order_id": order_ids[i],
            "comment": comments[i].strip(),
        })
    
    return reviews

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
            value=st.session_state.settings.get('temperatur
