# Basic test to verify API key works
def test_openai_api():
    """Test if the OpenAI API key works with a minimal request"""
    global api_key
    
    if not api_key or not AVAILABLE_MODULES['requests']:
        return False
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("OpenAI API test successful")
            return True
        else:
            logger.error(f"OpenAI API test failed: {response.status_code} - {response.text}")
            return False
            
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
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Remove the API_KEY_NAME constant since we're directly accessing the key
# Constants
SUPPORT_EMAIL = "alexander.popoff@vivehealth.com"

# Track available modules
AVAILABLE_MODULES = {
    'pandas': False,
    'numpy': False, 
    'plotly': False,
    'pillow': False,
    'requests': False,
    'ocr': False,
    'xlsx_writer': False,
    'ai_api': False      # New module for AI API functionality
}

# Try importing pandas
try:
    import pandas as pd
    AVAILABLE_MODULES['pandas'] = True
except ImportError:
    logger.warning("pandas module not available")

# Try importing numpy
try:
    import numpy as np
    AVAILABLE_MODULES['numpy'] = True
except ImportError:
    logger.warning("numpy module not available")

# Try importing plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    AVAILABLE_MODULES['plotly'] = True
except ImportError:
    logger.warning("plotly module not available")

# Try importing PIL
try:
    from PIL import Image
    AVAILABLE_MODULES['pillow'] = True
except ImportError:
    logger.warning("PIL module not available")

# Try importing requests - will be needed for direct API calls
try:
    import requests
    AVAILABLE_MODULES['requests'] = True
except ImportError:
    logger.warning("requests module not available")

# Try importing xlsxwriter
try:
    import xlsxwriter
    AVAILABLE_MODULES['xlsx_writer'] = True
except ImportError:
    logger.warning("xlsxwriter module not available")

# Try importing OCR modules - these will fail if not installed
try:
    import pytesseract
    import pdf2image
    AVAILABLE_MODULES['ocr'] = True
except ImportError:
    logger.warning("OCR modules not available")

# Try OpenAI API integration with direct HTTP requests
api_key = None
try:
    # Try to get from streamlit secrets directly - case sensitive!
    api_key = st.secrets.get("openai_api_key", None)
    if api_key:
        logger.info("Found openai_api_key in Streamlit secrets")
        AVAILABLE_MODULES['ai_api'] = True
    else:
        # Try alternate capitalizations
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if api_key:
            logger.info("Found OPENAI_API_KEY in Streamlit secrets")
            AVAILABLE_MODULES['ai_api'] = True
        else:
            # Last resort - try environment variable
            api_key = os.environ.get("OPENAI_API_KEY", None)
            if api_key:
                logger.info("Found OPENAI_API_KEY in environment variables")
                AVAILABLE_MODULES['ai_api'] = True
            else:
                logger.warning("OpenAI API key not found in any location")
except Exception as e:
    logger.error(f"Error accessing OpenAI API key: {str(e)}")
    
# Debug output to console
if api_key:
    logger.info(f"API key found: {api_key[:5]}...{api_key[-4:]}")
else:
    logger.warning("No API key found")

# Try importing local modules - use safe imports
try:
    import ocr_processor
    import data_analysis
    import import_template
    import ai_analysis  # New module for AI-specific analysis functions
    HAS_LOCAL_MODULES = True
    logger.info("Successfully imported local modules")
except ImportError as e:
    HAS_LOCAL_MODULES = False
    logger.warning(f"Could not import local modules: {str(e)}")

# Initialize session state variables
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
    
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'current_product' not in st.session_state:
    st.session_state.current_product = None

if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = {}

# Example preloaded data
EXAMPLE_DATA = {
    "structured_data": pd.DataFrame({
        "ASIN": ["B08HMCVJ8L", "B09X5DL3WK", "B07PDMJR4Q"],
        "SKU": ["VH-KNE01", "VH-BPM23", "VH-WCH14"],
        "Product Name": ["Premium Knee Brace", "Blood Pressure Monitor", "Folding Wheelchair"],
        "Category": ["Orthopedic Support", "Blood Pressure Monitors", "Mobility Aids"],
        "Last 30 Days Sales": [230, 185, 42],
        "Last 30 Days Returns": [12, 24, 5],
        "Last 365 Days Sales": [2850, 1950, 520],
        "Last 365 Days Returns": [142, 198, 61],
        "Star Rating": [4.6, 3.9, 4.3],
        "Total Reviews": [312, 246, 87],
        "Product Description": [
            "Supportive knee brace for sports injuries and arthritis relief.",
            "Digital blood pressure monitor with large display and memory function.",
            "Lightweight folding wheelchair with padded armrests and swing-away footrests."
        ]
    }),
    "manual_reviews": {
        "B08HMCVJ8L": [
            {"rating": 5, "review_text": "This knee brace is amazing! Significant pain relief and very comfortable to wear all day.", "asin": "B08HMCVJ8L"},
            {"rating": 4, "review_text": "Good quality and support, but the sizing runs a bit small. I had to return the first one.", "asin": "B08HMCVJ8L"},
            {"rating": 5, "review_text": "Using this for my recovery after ACL surgery and it provides perfect support.", "asin": "B08HMCVJ8L"},
            {"rating": 2, "review_text": "The velcro started coming off after just 2 weeks of use. Disappointed with durability.", "asin": "B08HMCVJ8L"},
            {"rating": 5, "review_text": "Best knee brace I've tried. The adjustable straps make it perfect for my needs.", "asin": "B08HMCVJ8L"}
        ]
    },
    "manual_returns": {
        "B08HMCVJ8L": [
            {"return_reason": "Too small - needs larger size", "asin": "B08HMCVJ8L"},
            {"return_reason": "Velcro stopped sticking after a few uses", "asin": "B08HMCVJ8L"},
            {"return_reason": "Not comfortable enough for all-day wear", "asin": "B08HMCVJ8L"},
            {"return_reason": "Didn't provide enough support for my knee", "asin": "B08HMCVJ8L"}
        ]
    }
}

# Amazon product categories for medical devices
MED_DEVICE_CATEGORIES = [
    "Mobility Aids", 
    "Bathroom Safety", 
    "Pain Relief", 
    "Sleep & Comfort", 
    "Fitness & Recovery", 
    "Daily Living Aids", 
    "Respiratory Care",
    "Blood Pressure Monitors",
    "Diabetes Care",
    "Orthopedic Support",
    "First Aid",
    "Wound Care",
    "Other"
]

# Amazon listing optimization metrics
LISTING_METRICS = {
    "Title Effectiveness": "How well the product title attracts clicks and conveys key features",
    "Bullet Points": "Quality and conversion focus of the key feature bullet points",
    "Description Quality": "Effectiveness of the product description",
    "Image Quality": "Clarity, quantity, and sales effectiveness of product images",
    "Keywords Coverage": "How well the listing covers relevant search terms",
    "Q&A Completeness": "Addresses common customer questions proactively",
    "Review Response": "Seller engagement with customer reviews"
}

# Setup the Streamlit page
st.set_page_config(
    page_title="Amazon Medical Device Listing Optimizer",
    page_icon="⚕️",
    layout="wide"
)

# Add AI analysis functions if the OpenAI module is available
if AVAILABLE_MODULES['ai_api']:
    def analyze_with_ai(text, analysis_type='sentiment'):
        """
        Analyze text using OpenAI API
        
        Parameters:
        - text: The text to analyze
        - analysis_type: Type of analysis to perform
        
        Returns:
        - Dictionary with analysis results
        """
        global api_key
        
        try:
            if not api_key:
                logger.warning("OpenAI API key not available for analysis")
                return {"success": False, "error": "API key not available"}
                
            prompt = ""
            if analysis_type == 'sentiment':
                prompt = f"""Analyze the sentiment of this Amazon review for a medical device. 
                Focus on identifying key factors that influence the customer's purchase decision 
                and satisfaction. Extract specific feedback about product features, quality, 
                packaging, and usability. Classify the sentiment as positive, negative, or neutral.
                
                Review: {text}
                """
            elif analysis_type == 'listing_optimization':
                prompt = f"""Analyze this Amazon medical device listing text and identify 
                specific improvements to increase conversion rate and sales. Focus on:
                1. SEO optimization for Amazon's A9 algorithm
                2. Clarity of features and benefits
                3. Competitive differentiation
                4. Customer pain points addressed
                5. Social proof elements
                
                Listing: {text}
                """
            elif analysis_type == 'return_analysis':
                prompt = f"""Analyze this return reason for an Amazon medical device 
                product to identify actionable improvements. Determine if the return 
                relates to:
                1. Product design or quality issues
                2. Customer expectation mismatch
                3. Listing accuracy problems (images, description)
                4. Sizing or fit issues
                5. Packaging or delivery problems
                
                Return reason: {text}
                """
            elif analysis_type == 'image_feedback':
                prompt = f"""Review this description of a product image for an Amazon 
                medical device listing. Identify improvements to make the image more 
                effective for conversion, including:
                1. Clarity and professional appearance
                2. Feature demonstration
                3. Size/scale reference
                4. Use case visualization
                5. Competitive differentiation
                
                Image description: {text}
                """
            
            messages = [
                {"role": "system", "content": "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."},
                {"role": "user", "content": prompt}
            ]
            
            if not AVAILABLE_MODULES.get('requests', False):
                logger.error("requests module is required for API calls")
                return {"success": False, "error": "requests module not available"}
            
            # API call with direct request to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 750,
                "temperature": 0.2
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return {"success": True, "result": result}
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"success": False, "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_listing_optimization(product_info):
        """
        Use AI to analyze and provide recommendations for Amazon listing optimization
        
        Parameters:
        - product_info: Dictionary with product details
        
        Returns:
        - Dictionary with optimization recommendations
        """
        try:
            prompt = f"""Analyze this Amazon medical device product and provide actionable 
            recommendations to optimize the listing for higher conversion rates and reduced returns.
            
            Product name: {product_info.get('name', 'Unknown')}
            Category: {product_info.get('category', 'Medical Device')}
            Description: {product_info.get('description', '')}
            30-Day Return Rate: {product_info.get('return_rate_30d', 'N/A')}%
            Star Rating: {product_info.get('star_rating', 'N/A')}
            
            Provide the following:
            1. Title optimization recommendations (for better CTR and keyword relevance)
            2. Bullet points strategy (highlight key benefits and features)
            3. Description improvements (storytelling and problem-solution format)
            4. Image optimization suggestions (specific shots and demonstrations needed)
            5. Keywords to target for this specific medical device
            6. A+ Content recommendations (if applicable)
            7. Common customer questions to address proactively
            """
            
            messages = [
                {"role": "system", "content": "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."},
                {"role": "user", "content": prompt}
            ]
            
            # API call with direct request to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.2
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return {"success": True, "result": result}
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"success": False, "error": f"API error: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Listing optimization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_improvement_recommendations(product_info, reviews_data, returns_data):
        """
        Generate AI-powered recommendations for product improvements based on reviews and returns
        
        Parameters:
        - product_info: Dictionary with product details
        - reviews_data: List of product reviews
        - returns_data: List of return reasons
        
        Returns:
        - Dictionary with recommendations
        """
        try:
            # Prepare the data for analysis
            reviews_text = "\n".join([f"Rating: {r.get('rating', 'N/A')} - {r.get('review_text', '')}" 
                                      for r in reviews_data[:20]])  # Limit to 20 reviews to avoid token limits
            
            returns_text = "\n".join([f"Return reason: {r.get('return_reason', '')}" 
                                     for r in returns_data[:20]])  # Limit to 20 return reasons
            
            prompt = f"""As an Amazon listing optimization specialist for medical devices, analyze the following 
            product data and provide actionable recommendations:
            
            Product: {product_info.get('name', 'Unknown')}
            Category: {product_info.get('category', 'Medical Device')}
            30-Day Return Rate: {product_info.get('return_rate_30d', 'N/A')}%
            Star Rating: {product_info.get('star_rating', 'N/A')}
            
            Customer Reviews:
            {reviews_text}
            
            Return Reasons:
            {returns_text}
            
            Please provide:
            1. Top 3-5 product improvement recommendations based on customer feedback
            2. Specific listing improvements to reduce return rate
            3. New or improved image recommendations based on customer confusion
            4. Keywords and features to emphasize based on positive reviews
            5. Features that need better explanation in the listing
            6. Competitive differentiators to highlight more prominently
            """
            
            messages = [
                {"role": "system", "content": "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."},
                {"role": "user", "content": prompt}
            ]
            
            # API call with direct request to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 1200,
                "temperature": 0.2
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return {"success": True, "result": result}
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"success": False, "error": f"API error: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    """Main application entry point."""
    # Render header
    st.title("Amazon Medical Device Listing Optimizer")
    st.subheader("Optimize product listings, reduce returns, and improve ratings for medical devices on Amazon")
    
    # Display available modules in sidebar
    with st.sidebar:
        st.header("Settings")
        st.write("Contact support: " + SUPPORT_EMAIL)
        
        # 1-click example loader
        if st.button("Load Example Data", type="primary"):
            st.session_state.uploaded_files = EXAMPLE_DATA.copy()
            st.success("Example data loaded successfully!")
            st.rerun()
        
        st.subheader("Available Modules")
        for module, available in AVAILABLE_MODULES.items():
            if available:
                st.success(f"✅ {module}")
            else:
                st.error(f"❌ {module}")
        
        # AI API Status Section
        st.subheader("AI API Status")
        if AVAILABLE_MODULES['ai_api']:
            st.success("✅ Connected to OpenAI API (GPT-4o)")
            
            # Show a mini usage tracker
            if 'ai_api_calls' not in st.session_state:
                st.session_state.ai_api_calls = 0
            
            st.metric("AI API Calls", st.session_state.ai_api_calls)
            
            # Add a test button
            if st.button("Test API Connection"):
                with st.spinner("Testing API connection..."):
                    if test_openai_api():
                        st.success("✅ API connection successful!")
                    else:
                        st.error("❌ API connection failed. Check logs for details.")
        else:
            st.error("❌ OpenAI API not configured")
            st.info("API key should be named 'openai_api_key' in your Streamlit app settings")
            
            # Add manual entry option for debugging
            with st.expander("Debug API Connection"):
                temp_key = st.text_input("Enter API Key for testing", type="password")
                if temp_key and st.button("Test Key"):
                    global api_key
                    api_key = temp_key
                    with st.spinner("Testing API connection..."):
                        if test_openai_api():
                            st.success("✅ Key works! Use this key in your Streamlit settings.")
                        else:
                            st.error("❌ Key doesn't work. Check if it's valid.")
    
    
    # Main content tabs
    tabs = st.tabs(["Import", "Analyze", "Listing Optimization", "AI Insights", "Help"])
    
    with tabs[0]:
        render_file_upload()
    
    with tabs[1]:
        render_product_selection()
    
    with tabs[2]:
        render_listing_optimization()
    
    with tabs[3]:
        render_ai_insights()
    
    with tabs[4]:
        render_help_section()

def render_file_upload():
    """Render the file upload section."""
    st.header("Import Data")
    
    upload_tabs = st.tabs(["Structured Data Import", "Manual Entry", "Image/Review Import", "Historical Data"])
    
    # Tab 1: Structured Data Import
    with upload_tabs[0]:
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
        
        # Download sample template
        if HAS_LOCAL_MODULES and AVAILABLE_MODULES['xlsx_writer']:
            sample_template = import_template.create_import_template()
            st.download_button(
                label="Download Sample Template",
                data=sample_template,
                file_name="amazon_listing_optimization_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        uploaded_file = st.file_uploader(
            "Upload product data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="structured_data"
        )
        
        if uploaded_file:
            with st.spinner("Processing data file..."):
                if AVAILABLE_MODULES['pandas']:
                    try:
                        # Process file based on extension
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_ext == 'csv':
                            df = pd.read_csv(uploaded_file)
                        elif file_ext in ['xlsx', 'xls']:
                            df = pd.read_excel(uploaded_file)
                        else:
                            st.error(f"Unsupported file format: {file_ext}")
                            return
                        
                        # Validate required columns
                        required_columns = ['ASIN', 'Last 30 Days Sales', 'Last 30 Days Returns']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if missing_columns:
                            st.error(f"Missing required columns: {', '.join(missing_columns)}")
                            return
                        
                        st.session_state.uploaded_files['structured_data'] = df
                        st.success(f"Successfully processed file with {len(df)} products.")
                        st.dataframe(df.head())
                        
                        # Automatically run AI analysis if API is available
                        if AVAILABLE_MODULES['ai_api'] and st.checkbox("Automatically run AI analysis on imported data", value=True):
                            st.info("AI analysis will be run when you select a product in the Analyze tab")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                else:
                    st.error("pandas module is not available for processing files.")
    
    # Tab 2: Manual Entry
    with upload_tabs[1]:
        st.markdown("""
        Manually enter product details for analysis. Great for analyzing a single Amazon listing
        without needing to upload a spreadsheet.
        
        **Required fields are marked with an asterisk (*)**
        """)
        
        # Create a form for manual data entry
        with st.form("manual_entry_form"):
            # Product Details Section
            st.subheader("Product Details")
            col1, col2 = st.columns(2)
            
            with col1:
                asin = st.text_input("ASIN* (Amazon Standard Identification Number)", help="Required field")
                sku = st.text_input("SKU (Stock Keeping Unit)", help="Optional")
            
            with col2:
                product_name = st.text_input("Product Name*", help="Required field")
                category = st.selectbox(
                    "Category*", 
                    options=MED_DEVICE_CATEGORIES,
                    help="Required field"
                )
            
            # Amazon Listing Information
            st.subheader("Amazon Listing Information")
            
            product_description = st.text_area(
                "Product Description/Bullet Points",
                help="Copy and paste your Amazon listing content for AI analysis",
                height=150
            )
            
            listing_url = st.text_input("Amazon Listing URL", help="Optional - for reference only")
            
            # Sales & Returns Section
            st.subheader("Sales & Returns Data")
            col1, col2 = st.columns(2)
            
            with col1:
                sales_30d = st.number_input("Last 30 Days Sales*", min_value=0, help="Required field")
                sales_365d = st.number_input("Last 365 Days Sales", min_value=0, help="Optional")
            
            with col2:
                returns_30d = st.number_input("Last 30 Days Returns*", min_value=0, help="Required field")
                returns_365d = st.number_input("Last 365 Days Returns", min_value=0, help="Optional")
            
            # Ratings & Reviews Section
            st.subheader("Ratings & Reviews")
            col1, col2 = st.columns(2)
            
            with col1:
                star_rating = st.slider("Star Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1, help="Optional")
            
            with col2:
                total_reviews = st.number_input("Total Reviews", min_value=0, help="Optional")
            
            # Manual Reviews & Return Reasons (optional)
            st.subheader("Customer Reviews & Return Reasons (Optional)")
            
            # Reviews input
            st.text_area(
                "Reviews (One per line, format: Rating - Review Text, e.g., '4 - Good product but could be better')",
                height=150,
                key="manual_reviews",
                help="Optional: Enter reviews one per line with rating and text separated by a dash"
            )
            
            # Return reasons input
            st.text_area(
                "Return Reasons (One per line)",
                height=150,
                key="manual_returns",
                help="Optional: Enter return reasons one per line"
            )
            
            # Competitive positioning
            st.subheader("Competitive Positioning (Optional)")
            st.text_area(
                "Top Competing Products (One per line, format: ASIN - Product Name)",
                height=100,
                key="competing_products",
                help="Optional: Enter competing products one per line"
            )
            
            # Submit button
            submitted = st.form_submit_button("Save Product Data")
        
        # Process manual entry when form is submitted
        if submitted:
            # Validate required fields
            if not asin or not product_name or not category or sales_30d == 0:
                st.error("Please fill in all required fields marked with an asterisk (*)")
            else:
                try:
                    # Create a DataFrame with the manually entered data
                    manual_data = {
                        "ASIN": [asin],
                        "SKU": [sku],
                        "Product Name": [product_name],
                        "Category": [category],
                        "Product Description": [product_description],
                        "Listing URL": [listing_url],
                        "Last 30 Days Sales": [sales_30d],
                        "Last 30 Days Returns": [returns_30d],
                        "Last 365 Days Sales": [sales_365d],
                        "Last 365 Days Returns": [returns_365d],
                        "Star Rating": [star_rating],
                        "Total Reviews": [total_reviews]
                    }
                    
                    if AVAILABLE_MODULES['pandas']:
                        df = pd.DataFrame(manual_data)
                        
                        # Store in session state
                        if 'structured_data' in st.session_state.uploaded_files:
                            # Append to existing data if it exists
                            existing_df = st.session_state.uploaded_files['structured_data']
                            # Check if this ASIN already exists
                            if asin in existing_df['ASIN'].values:
                                # Update the existing row
                                existing_df = existing_df[existing_df['ASIN'] != asin]
                                updated_df = pd.concat([existing_df, df], ignore_index=True)
                                st.session_state.uploaded_files['structured_data'] = updated_df
                                st.success(f"Updated product {asin} in the dataset")
                            else:
                                # Append the new row
                                updated_df = pd.concat([existing_df, df], ignore_index=True)
                                st.session_state.uploaded_files['structured_data'] = updated_df
                                st.success(f"Added product {asin} to the dataset")
                        else:
                            # Create new dataset
                            st.session_state.uploaded_files['structured_data'] = df
                            st.success(f"Created new dataset with product {asin}")
                        
                        # Process manual reviews if provided
                        manual_reviews = st.session_state.get("manual_reviews", "")
                        if manual_reviews:
                            reviews_data = []
                            for line in manual_reviews.strip().split('\n'):
                                if ' - ' in line:
                                    try:
                                        rating_str, text = line.split(' - ', 1)
                                        rating = int(rating_str.strip())
                                        if 1 <= rating <= 5:
                                            reviews_data.append({
                                                "rating": rating,
                                                "review_text": text.strip(),
                                                "asin": asin
                                            })
                                    except:
                                        pass  # Skip invalid lines
                            
                            if reviews_data:
                                # Store reviews data
                                if 'manual_reviews' not in st.session_state.uploaded_files:
                                    st.session_state.uploaded_files['manual_reviews'] = {}
                                
                                st.session_state.uploaded_files['manual_reviews'][asin] = reviews_data
                                st.success(f"Saved {len(reviews_data)} reviews for {asin}")
                        
                        # Process manual return reasons if provided
                        manual_returns = st.session_state.get("manual_returns", "")
                        if manual_returns:
                            return_reasons = [reason.strip() for reason in manual_returns.strip().split('\n') if reason.strip()]
                            
                            if return_reasons:
                                # Store return reasons
                                if 'manual_returns' not in st.session_state.uploaded_files:
                                    st.session_state.uploaded_files['manual_returns'] = {}
                                
                                st.session_state.uploaded_files['manual_returns'][asin] = [
                                    {"return_reason": reason, "asin": asin} for reason in return_reasons
                                ]
                                st.success(f"Saved {len(return_reasons)} return reasons for {asin}")
                        
                        # Process competing products if provided
                        competing_products = st.session_state.get("competing_products", "")
                        if competing_products:
                            competitors = []
                            for line in competing_products.strip().split('\n'):
                                if ' - ' in line:
                                    try:
                                        comp_asin, comp_name = line.split(' - ', 1)
                                        competitors.append({
                                            "asin": comp_asin.strip(),
                                            "name": comp_name.strip(),
                                            "product_asin": asin
                                        })
                                    except:
                                        pass  # Skip invalid lines
                            
                            if competitors:
                                # Store competitors
                                if 'competitors' not in st.session_state.uploaded_files:
                                    st.session_state.uploaded_files['competitors'] = {}
                                
                                st.session_state.uploaded_files['competitors'][asin] = competitors
                                st.success(f"Saved {len(competitors)} competing products for {asin}")
                        
                        # Display the entered data
                        st.subheader("Entered Product Data")
                        st.dataframe(df)
                    else:
                        st.error("pandas module is not available. Cannot process manual entry.")
                except Exception as e:
                    st.error(f"Error processing manual entry: {str(e)}")
    
    # Tab 3: Image/Review Import
    with upload_tabs[2]:
        st.markdown("""
        Upload screenshots of Amazon reviews, listings, return reports, or product images. 
        The system will extract text and analyze the content.
        """)
        
        if not AVAILABLE_MODULES['ocr']:
            st.warning("OCR processing is not available. To enable this feature, install pytesseract and pdf2image.")
        
        doc_files = st.file_uploader(
            "Upload screenshots or images (PDF, PNG, JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="documents",
            disabled=not AVAILABLE_MODULES['ocr']
        )
        
        # Add option to specify what type of content is being uploaded
        image_content_type = st.selectbox(
            "What content are you uploading?",
            options=["Product Reviews", "Product Listing", "Return Reports", "Product Images", "Competitor Listings", "Mixed Content"],
            index=0
        )
        
        if doc_files and HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']:
            # Process each document
            processed_docs = []
            
            for doc in doc_files:
                with st.spinner(f"Processing {doc.name}..."):
                    # Process based on file type
                    file_ext = doc.name.split('.')[-1].lower()
                    
                    if file_ext == 'pdf':
                        text = ocr_processor.process_pdf_with_ocr(doc.read())
                    elif file_ext in ['png', 'jpg', 'jpeg']:
                        text = ocr_processor.process_image_with_ocr(doc.read())
                    else:
                        text = f"Unsupported document type: {file_ext}"
                        
                    processed_docs.append({
                        "filename": doc.name,
                        "text": text,
                        "type": file_ext,
                        "content_type": image_content_type
                    })
            
            if processed_docs:
                st.session_state.uploaded_files['documents'] = processed_docs
                
                # Show preview of extracted text from first document
                if len(processed_docs) > 0:
                    with st.expander("Preview of extracted text"):
                        st.text(processed_docs[0]["text"][:1000] + "...")
                        
                    # Run AI analysis on the extracted text if enabled
                    if AVAILABLE_MODULES['ai_api'] and st.button("Analyze Extracted Content with AI"):
                        with st.spinner("Analyzing content with AI..."):
                            analysis_type = ""
                            if image_content_type == "Product Reviews":
                                analysis_type = "sentiment"
                            elif image_content_type in ["Product Listing", "Competitor Listings"]:
                                analysis_type = "listing_optimization"
                            elif image_content_type == "Return Reports":
                                analysis_type = "return_analysis"
                            elif image_content_type == "Product Images":
                                analysis_type = "image_feedback"
                            
                            if analysis_type:
                                # Increment the API call counter
                                if 'ai_api_calls' in st.session_state:
                                    st.session_state.ai_api_calls += 1
                                
                                result = analyze_with_ai(processed_docs[0]["text"], analysis_type)
                                if result["success"]:
                                    st.subheader("AI Analysis Results")
                                    st.markdown(result["result"])
                                else:
                                    st.error(f"AI analysis failed: {result.get('error', 'Unknown error')}")
        elif doc_files and not (HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']):
            st.error("OCR processing is not available. Document processing is skipped.")
    
    # Tab 4: Historical Data
    with upload_tabs[3]:
        st.markdown("""
        Upload historical sales and return data (optional).
        This helps identify trends and seasonality in your Amazon product performance.
        """)
        
        hist_file = st.file_uploader(
            "Upload historical data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="historical_data"
        )
        
        if hist_file and AVAILABLE_MODULES['pandas']:
            with st.spinner("Processing historical data..."):
                try:
                    # Process file based on extension
                    file_ext = hist_file.name.split('.')[-1].lower()
                    
                    if file_ext == 'csv':
                        hist_df = pd.read_csv(hist_file)
                    elif file_ext in ['xlsx', 'xls']:
                        hist_df = pd.read_excel(hist_file)
                    else:
                        st.error(f"Unsupported file format: {file_ext}")
                        return
                    
                    st.session_state.uploaded_files['historical_data'] = hist_df
                    st.success(f"Successfully processed historical data with {len(hist_df)} entries.")
                    st.dataframe(hist_df.head())
                    
                    # Automatically generate trend visualization if dates are present
                    if 'Date' in hist_df.columns and AVAILABLE_MODULES['plotly']:
                        try:
                            # Convert to datetime if not already
                            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
                            
                            # Check if we have sales or return data
                            plot_cols = []
                            if 'Sales' in hist_df.columns:
                                plot_cols.append('Sales')
                            if 'Returns' in hist_df.columns:
                                plot_cols.append('Returns')
                            if 'Return Rate' in hist_df.columns:
                                plot_cols.append('Return Rate')
                                
                            if plot_cols:
                                st.subheader("Historical Trends")
                                fig = px.line(hist_df, x='Date', y=plot_cols, title="Historical Performance")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating trend visualization: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing historical data: {str(e)}")
        elif hist_file and not AVAILABLE_MODULES['pandas']:
            st.error("pandas module is not available for processing files.")

def render_product_selection():
    """Render the product selection section."""
    st.header("Select Product to Analyze")
    
    # Check if we have structured data
    if 'structured_data' not in st.session_state.uploaded_files:
        st.warning("Please upload structured product data first or use Manual Entry.")
        return
    
    if not AVAILABLE_MODULES['pandas']:
        st.error("pandas module is required for product selection and analysis.")
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
        'description': product_row['Product Description'] if 'Product Description' in product_row else "",
        'listing_url': product_row['Listing URL'] if 'Listing URL' in product_row else "",
        'star_rating': product_row['Star Rating'] if 'Star Rating' in product_row else None,
        'total_reviews': product_row['Total Reviews'] if 'Total Reviews' in product_row else None,
        'sales_30d': product_row['Last 30 Days Sales'],
        'returns_30d': product_row['Last 30 Days Returns'],
        'sales_365d': product_row['Last 365 Days Sales'] if 'Last 365 Days Sales' in product_row else None,
        'returns_365d': product_row['Last 365 Days Returns'] if 'Last 365 Days Returns' in product_row else None,
    }
    
    # Calculate metrics
    safe_divide = lambda a, b: (a / b) * 100 if b > 0 else 0
    product_info['return_rate_30d'] = safe_divide(product_info['returns_30d'], product_info['sales_30d'])
    if product_info['sales_365d'] is not None and product_info['returns_365d'] is not None:
        product_info['return_rate_365d'] = safe_divide(product_info['returns_365d'], product_info['sales_365d'])
    
    # Store selected product in session state
    st.session_state.current_product = product_info
    
    # Calculate additional e-commerce metrics
    monthly_revenue = product_info['sales_30d'] * 50  # Assuming average selling price of $50
    transaction_fee = monthly_revenue * 0.15  # Assuming 15% Amazon fee
    cost_of_returns = product_info['returns_30d'] * 15  # Assuming $15 cost per return processing
    
    # Display selected product info in metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("30-Day Sales", f"{product_info['sales_30d']:,}")
        st.metric("Est. Monthly Revenue", f"${monthly_revenue:,.2f}")
    with col2:
        st.metric("30-Day Returns", f"{product_info['returns_30d']:,}")
        st.metric("Return Processing Cost", f"${cost_of_returns:,.2f}")
    with col3:
        st.metric("30-Day Return Rate", f"{product_info['return_rate_30d']:.2f}%")
        st.metric("Amazon Fees (Est.)", f"${transaction_fee:,.2f}")
    with col4:
        if product_info['star_rating'] is not None:
            st.metric("Star Rating", f"{product_info['star_rating']:.1f} ★")
            # Calculate rating improvement impact
            potential_impact = 0
            if product_info['star_rating'] < 4.5:
                potential_impact = ((4.5 - product_info['star_rating']) / product_info['star_rating']) * monthly_revenue * 0.10
                st.metric("Rating Improvement Value", f"${potential_impact:,.2f}")
    
    # Display listing URL if available
    if product_info['listing_url']:
        st.markdown(f"[View Amazon Listing]({product_info['listing_url']})")
    
    # Get review data for the selected product from documents if available
    reviews_data = []
    
    # 1. Check for OCR-extracted reviews from documents
    if 'documents' in st.session_state.uploaded_files and HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']:
        docs = st.session_state.uploaded_files['documents']
        
        for doc in docs:
            # Extract reviews from OCR text if content type is reviews
            if doc.get("content_type") == "Product Reviews":
                text = doc["text"]
                extracted_reviews = ocr_processor.extract_amazon_reviews_data(text)
                if "reviews" in extracted_reviews:
                    reviews_data.extend(extracted_reviews["reviews"])
    
    # 2. Check for manually entered reviews
    if 'manual_reviews' in st.session_state.uploaded_files:
        manual_reviews = st.session_state.uploaded_files['manual_reviews']
        if product_info['asin'] in manual_reviews:
            reviews_data.extend(manual_reviews[product_info['asin']])
    
    if reviews_data:
        st.write(f"Found {len(reviews_data)} reviews for this product.")
    
    # Get return reasons data
    return_reasons_data = []
    
    # Check for manually entered return reasons
    if 'manual_returns' in st.session_state.uploaded_files:
        manual_returns = st.session_state.uploaded_files['manual_returns']
        if product_info['asin'] in manual_returns:
            return_reasons_data.extend(manual_returns[product_info['asin']])
    
    if return_reasons_data:
        st.write(f"Found {len(return_reasons_data)} return reasons for this product.")
    
    # Get competitors data if available
    competitors_data = []
    if 'competitors' in st.session_state.uploaded_files:
        competitors = st.session_state.uploaded_files['competitors']
        if product_info['asin'] in competitors:
            competitors_data.extend(competitors[product_info['asin']])
            
    if competitors_data:
        st.write(f"Found {len(competitors_data)} competing products for comparison.")
    
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
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        if st.button("Standard Analysis", type="primary"):
            # Check if we have both product info and review data
            if st.session_state.current_product:
                with st.spinner("Analyzing product reviews and returns..."):
                    try:
                        # Run analysis if we have the proper modules and local dependencies
                        if HAS_LOCAL_MODULES:
                            # Run analysis
                            analysis_result = f"Analysis for {product_info['name']} ({product_info['asin']})\n\n"
                            analysis_result += f"30-Day Sales: {product_info['sales_30d']}\n"
                            analysis_result += f"30-Day Returns: {product_info['returns_30d']}\n"
                            analysis_result += f"30-Day Return Rate: {product_info['return_rate_30d']:.2f}%\n"
                            
                            if reviews_data:
                                review_analysis = data_analysis.analyze_reviews(reviews_data)
                                analysis_result += f"\nReview Analysis:\n"
                                analysis_result += f"Total Reviews: {review_analysis['total_reviews']}\n"
                                if review_analysis['average_rating']:
                                    analysis_result += f"Average Rating: {review_analysis['average_rating']:.1f}\n"
                            
                            # Store result in session state
                            st.session_state.analysis_results[product_info['asin']] = {
                                'product_info': product_info,
                                'analysis': analysis_result,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'reviews_data': reviews_data if reviews_data else [],
                                'return_reasons_data': return_reasons_data if return_reasons_data else [],
                                'historical_data': historical_data,
                                'competitors_data': competitors_data if competitors_data else []
                            }
                        else:
                            # Basic analysis without local modules
                            analysis_result = f"Basic Analysis for {product_info['name']} ({product_info['asin']})\n\n"
                            analysis_result += f"30-Day Sales: {product_info['sales_30d']}\n"
                            analysis_result += f"30-Day Returns: {product_info['returns_30d']}\n"
                            analysis_result += f"30-Day Return Rate: {product_info['return_rate_30d']:.2f}%\n"
                            
                            st.session_state.analysis_results[product_info['asin']] = {
                                'product_info': product_info,
                                'analysis': analysis_result,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'reviews_data': reviews_data if reviews_data else [],
                                'return_reasons_data': return_reasons_data if return_reasons_data else [],
                                'historical_data': historical_data,
                                'competitors_data': competitors_data if competitors_data else []
                            }
                        
                        st.success("Analysis complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.error("Please select a product to analyze.")
    
    with analysis_col2:
        if AVAILABLE_MODULES['ai_api']:
            if st.button("AI-Enhanced Analysis", type="secondary"):
                if st.session_state.current_product:
                    with st.spinner("Performing AI-enhanced analysis..."):
                        try:
                            # Increment the API call counter
                            if 'ai_api_calls' in st.session_state:
                                st.session_state.ai_api_calls += 1
                                
                            # First run standard analysis if not already done
                            if product_info['asin'] not in st.session_state.analysis_results:
                                analysis_result = f"Analysis for {product_info['name']} ({product_info['asin']})\n\n"
                                analysis_result += f"30-Day Sales: {product_info['sales_30d']}\n"
                                analysis_result += f"30-Day Returns: {product_info['returns_30d']}\n"
                                analysis_result += f"30-Day Return Rate: {product_info['return_rate_30d']:.2f}%\n"
                                
                                if reviews_data:
                                    review_analysis = data_analysis.analyze_reviews(reviews_data) if HAS_LOCAL_MODULES else {'total_reviews': len(reviews_data), 'average_rating': None}
                                    analysis_result += f"\nReview Analysis:\n"
                                    analysis_result += f"Total Reviews: {review_analysis['total_reviews']}\n"
                                    if review_analysis.get('average_rating'):
                                        analysis_result += f"Average Rating: {review_analysis['average_rating']:.1f}\n"
                                
                                # Store standard analysis
                                st.session_state.analysis_results[product_info['asin']] = {
                                    'product_info': product_info,
                                    'analysis': analysis_result,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'reviews_data': reviews_data if reviews_data else [],
                                    'return_reasons_data': return_reasons_data if return_reasons_data else [],
                                    'historical_data': historical_data,
                                    'competitors_data': competitors_data if competitors_data else []
                                }
                            
                            # Now run AI analysis
                            ai_insights = {}
                            
                            # 1. Analyze individual reviews with AI (up to 10 to avoid excessive API calls)
                            if reviews_data:
                                review_insights = []
                                for review in reviews_data[:10]:  # Limit to 10 reviews
                                    review_text = review.get('review_text', '')
                                    if review_text:
                                        # Analyze sentiment
                                        sentiment = analyze_with_ai(review_text, 'sentiment')
                                        if sentiment.get('success', False):
                                            review_insights.append({
                                                'review': review_text,
                                                'rating': review.get('rating', 'N/A'),
                                                'analysis': sentiment.get('result', 'Analysis failed')
                                            })
                                
                                ai_insights['review_insights'] = review_insights
                            
                            # 2. Analyze return reasons with AI
                            if return_reasons_data:
                                return_insights = []
                                for return_data in return_reasons_data[:10]:  # Limit to 10 return reasons
                                    return_reason = return_data.get('return_reason', '')
                                    if return_reason:
                                        analysis = analyze_with_ai(return_reason, 'return_analysis')
                                        if analysis.get('success', False):
                                            return_insights.append({
                                                'return_reason': return_reason,
                                                'analysis': analysis.get('result', 'Analysis failed')
                                            })
                                
                                ai_insights['return_insights'] = return_insights
                            
                            # 3. Generate listing optimization recommendations
                            listing_optimization = analyze_listing_optimization(product_info)
                            if listing_optimization.get('success', False):
                                ai_insights['listing_optimization'] = listing_optimization.get('result', 'Analysis failed')
                            
                            # 4. Generate improvement recommendations
                            if reviews_data or return_reasons_data:
                                recommendations = generate_improvement_recommendations(
                                    product_info,
                                    reviews_data,
                                    return_reasons_data
                                )
                                
                                if recommendations.get('success', False):
                                    ai_insights['recommendations'] = recommendations.get('result', 'Recommendation generation failed')
                            
                            # Store AI insights in session state
                            st.session_state.ai_insights[product_info['asin']] = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'insights': ai_insights
                            }
                            
                            st.success("AI analysis complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during AI analysis: {str(e)}")
                else:
                    st.error("Please select a product to analyze.")
        else:
            st.warning("AI-Enhanced Analysis requires OpenAI API key in Streamlit app settings")

def render_listing_optimization():
    """Render the listing optimization section."""
    st.header("Amazon Listing Optimization")
    
    # Check if we have a current product selected
    if not st.session_state.current_product:
        st.info("Please select a product in the Analyze tab first.")
        return
    
    product_info = st.session_state.current_product
    
    st.subheader(f"Optimize Listing for {product_info['name']} ({product_info['asin']})")
    
    # Check if AI is available
    if not AVAILABLE_MODULES['ai_api']:
        st.warning("OpenAI API is required for listing optimization. Please add your API key to Streamlit secrets.")
        return
    
    # Tabs for different optimization areas
    tabs = st.tabs(["Current Listing", "Title Optimization", "Bullet Points", "Description", "Keywords", "Images", "Competitive Analysis"])
    
    # Tab 1: Current Listing
    with tabs[0]:
        st.markdown("### Current Amazon Listing")
        
        # Display current listing info if available
        if product_info['description']:
            st.markdown("#### Product Description")
            st.text_area("Current Description", value=product_info['description'], height=300, disabled=True)
        else:
            st.warning("No product description available. Add one in the Manual Entry tab.")
        
        # Option to paste Amazon listing content if not already entered
        if not product_info['description']:
            with st.form("update_listing_form"):
                description = st.text_area("Paste your Amazon listing content (title, bullets, description)", height=300)
                submit = st.form_submit_button("Update Listing")
                
                if submit and description:
                    # Update the product info in session state
                    product_info['description'] = description
                    st.session_state.current_product = product_info
                    
                    # Update in the structured data DataFrame as well
                    if 'structured_data' in st.session_state.uploaded_files:
                        df = st.session_state.uploaded_files['structured_data']
                        df.loc[df['ASIN'] == product_info['asin'], 'Product Description'] = description
                        st.session_state.uploaded_files['structured_data'] = df
                    
                    st.success("Listing content updated!")
                    st.rerun()
        
        # Amazon listing score based on key metrics
        st.markdown("### Listing Quality Score")
        
        # Calculate estimated listing quality score (placeholder logic)
        scores = {}
        if product_info['description']:
            # Simple metrics based on description length and content
            desc = product_info['description']
            scores["Title Effectiveness"] = min(len(desc.split('\n')[0].split()) * 5, 100) if '\n' in desc else 60
            scores["Bullet Points"] = min(desc.count('\n') * 10, 100) if desc.count('\n') > 0 else 40
            scores["Description Quality"] = min(len(desc) / 20, 100)
            scores["Image Quality"] = 75  # Placeholder
            scores["Keywords Coverage"] = 65  # Placeholder
            scores["Q&A Completeness"] = 50  # Placeholder
            scores["Review Response"] = 40  # Placeholder
            
            # Get star rating impact
            if product_info['star_rating']:
                scores["Customer Satisfaction"] = min(product_info['star_rating'] * 20, 100)
            
            # Get return rate impact
            if product_info['return_rate_30d'] is not None:
                scores["Return Rate"] = max(100 - (product_info['return_rate_30d'] * 5), 0)
        else:
            # Default scores if no description
            for metric in LISTING_METRICS:
                scores[metric] = 50
        
        # Display listing quality score with gauge chart
        if AVAILABLE_MODULES['plotly']:
            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores)
            
            # Create gauge chart for overall score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = overall_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Listing Quality"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "khaki"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display individual metric scores
            st.markdown("### Listing Metrics")
            
            # Create a dataframe for the metrics
            metric_df = pd.DataFrame({
                'Metric': list(scores.keys()),
                'Score': list(scores.values()),
                'Description': [LISTING_METRICS.get(m, "") for m in scores.keys()]
            })
            
            # Create a bar chart
            fig = px.bar(
                metric_df, 
                x='Metric', 
                y='Score', 
                color='Score',
                color_continuous_scale=["red", "yellow", "green"],
                range_color=[0, 100],
                hover_data=['Description'],
                title="Listing Quality Metrics"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # AI analysis button
        if st.button("Generate AI Optimization Recommendations", type="primary"):
            with st.spinner("Analyzing listing with AI..."):
                # Increment the API call counter
                if 'ai_api_calls' in st.session_state:
                    st.session_state.ai_api_calls += 1
                
                listing_optimization = analyze_listing_optimization(product_info)
                if listing_optimization.get('success', False):
                    # Store in AI insights
                    if product_info['asin'] not in st.session_state.ai_insights:
                        st.session_state.ai_insights[product_info['asin']] = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'insights': {}
                        }
                    
                    st.session_state.ai_insights[product_info['asin']]['insights']['listing_optimization'] = listing_optimization.get('result')
                    
                    st.success("Listing optimization recommendations generated!")
                    st.markdown("### AI Recommendations")
                    st.markdown(listing_optimization.get('result'))
                else:
                    st.error(f"AI analysis failed: {listing_optimization.get('error', 'Unknown error')}")
    
    # Tab 2: Title Optimization
    with tabs[1]:
        st.markdown("### Amazon Title Optimization")
        st.markdown("""
        A great Amazon title should:
        - Include main keywords
        - Be within 200 characters
        - List key benefits
        - Include brand name
        - Mention the specific model/type
        """)
        
        # Extract current title
        current_title = ""
        if product_info['description']:
            lines = product_info['description'].split('\n')
            if lines:
                current_title = lines[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Title")
            st.text_area("Current", value=current_title, height=100, disabled=True)
            
            # Title length analysis
            if current_title:
                title_length = len(current_title)
                if title_length < 80:
                    st.warning(f"Title length: {title_length}/200 characters - Consider adding more relevant keywords")
                elif title_length > 180:
                    st.warning(f"Title length: {title_length}/200 characters - Getting close to the limit")
                else:
                    st.success(f"Title length: {title_length}/200 characters - Good length")
        
        with col2:
            st.markdown("#### Optimized Title")
            
            # Get AI recommendations for title if available
            if product_info['asin'] in st.session_state.ai_insights and 'listing_optimization' in st.session_state.ai_insights[product_info['asin']]['insights']:
                listing_insights = st.session_state.ai_insights[product_info['asin']]['insights']['listing_optimization']
                
                # Try to extract title recommendations
                title_section = ""
                if "Title" in listing_insights and "recommendations" in listing_insights.lower():
                    lines = listing_insights.split('\n')
                    in_title_section = False
                    for line in lines:
                        if "Title" in line and ":" in line:
                            in_title_section = True
                            title_section += line + "\n"
                        elif in_title_section and line.strip() and not any(section in line for section in ["Bullet", "Description", "Image", "Keyword"]):
                            title_section += line + "\n"
                        elif in_title_section and any(section in line for section in ["Bullet", "Description", "Image", "Keyword"]):
                            in_title_section = False
                
                if title_section:
                    st.markdown(title_section)
                else:
                    st.info("Generate full listing recommendations to see title optimization suggestions")
            else:
                st.info("Run AI Optimization to get title recommendations")
        
        # Generate title button
        if st.button("Generate Optimized Title", key="title_button"):
            with st.spinner("Generating optimized title..."):
                # Increment the API call counter
                if 'ai_api_calls' in st.session_state:
                    st.session_state.ai_api_calls += 1
                
                prompt = f"""Create an optimized Amazon title for this medical device product:
                
                Product: {product_info['name']}
                Category: {product_info['category']}
                Current Title: {current_title}
                
                Follow Amazon's best practices:
                - Maximum 200 characters
                - Include key search terms
                - Format: Brand + Model + Type + Key Features/Benefits
                - No promotional language like "best" or "top-rated"
                - No special characters beyond basic punctuation
                
                Provide just the optimized title text.
                """
                
                messages = [
                    {"role": "system", "content": "You are an expert Amazon listing optimization specialist for medical devices."},
                    {"role": "user", "content": prompt}
                ]
                
                try:
                    # Check which version of the OpenAI SDK is being used
                    if 'client' in globals():  # New SDK (v1.0.0+)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=200,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    else:  # Legacy SDK (<v1.0.0)
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=200,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    
                    st.success("Optimized title generated!")
                    st.markdown("### AI-Generated Title")
                    st.markdown(result)
                    
                    # Display length analysis
                    title_length = len(result)
                    if title_length < 80:
                        st.warning(f"Title length: {title_length}/200 characters - Consider adding more relevant keywords")
                    elif title_length > 180:
                        st.warning(f"Title length: {title_length}/200 characters - Getting close to the limit")
                    else:
                        st.success(f"Title length: {title_length}/200 characters - Good length")
                except Exception as e:
                    st.error(f"Error generating title: {str(e)}")
    
    # Tab 3: Bullet Points
    with tabs[2]:
        st.markdown("### Bullet Points Optimization")
        st.markdown("""
        Effective bullet points should:
        - Focus on key benefits (not just features)
        - Address customer pain points
        - Be concise but descriptive
        - Include relevant keywords
        - Address common questions/concerns
        """)
        
        # Extract current bullet points
        bullet_points = []
        if product_info['description']:
            lines = product_info['description'].split('\n')
            in_bullets = False
            for line in lines:
                if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*'):
                    bullet_points.append(line.strip())
                    in_bullets = True
                elif in_bullets and line.strip() and not line.strip().startswith('•') and not line.strip().startswith('-') and not line.strip().startswith('*'):
                    in_bullets = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Bullet Points")
            if bullet_points:
                for bp in bullet_points:
                    st.markdown(bp)
            else:
                st.info("No bullet points detected in the product description")
        
        with col2:
            st.markdown("#### Optimized Bullet Points")
            
            # Get AI recommendations for bullet points if available
            if product_info['asin'] in st.session_state.ai_insights and 'listing_optimization' in st.session_state.ai_insights[product_info['asin']]['insights']:
                listing_insights = st.session_state.ai_insights[product_info['asin']]['insights']['listing_optimization']
                
                # Try to extract bullet point recommendations
                bullet_section = ""
                if "Bullet" in listing_insights:
                    lines = listing_insights.split('\n')
                    in_bullet_section = False
                    for line in lines:
                        if "Bullet" in line:
                            in_bullet_section = True
                            bullet_section += line + "\n"
                        elif in_bullet_section and line.strip() and not any(section in line for section in ["Title", "Description", "Image", "Keyword"]):
                            bullet_section += line + "\n"
                        elif in_bullet_section and any(section in line for section in ["Title", "Description", "Image", "Keyword"]):
                            in_bullet_section = False
                
                if bullet_section:
                    st.markdown(bullet_section)
                else:
                    st.info("Generate full listing recommendations to see bullet point optimization suggestions")
            else:
                st.info("Run AI Optimization to get bullet point recommendations")
        
        # Generate bullet points button
        if st.button("Generate Optimized Bullet Points", key="bullet_button"):
            with st.spinner("Generating optimized bullet points..."):
                # Increment the API call counter
                if 'ai_api_calls' in st.session_state:
                    st.session_state.ai_api_calls += 1
                
                prompt = f"""Create 5 optimized bullet points for this Amazon medical device listing:
                
                Product: {product_info['name']}
                Category: {product_info['category']}
                Description: {product_info['description']}
                
                Current Bullet Points:
                {chr(10).join(bullet_points) if bullet_points else "None provided"}
                
                Follow Amazon's best practices:
                - Focus on benefits, not just features
                - Address customer pain points and questions
                - Include relevant keywords
                - Start with capital letters
                - Keep each point under 200 characters
                - Format as complete sentences
                - No promotional language like "best" or "top-rated"
                
                Include one bullet point specifically addressing quality/durability and one addressing comfort/ease of use.
                """
                
                messages = [
                    {"role": "system", "content": "You are an expert Amazon listing optimization specialist for medical devices."},
                    {"role": "user", "content": prompt}
                ]
                
                try:
                    # Check which version of the OpenAI SDK is being used
                    if 'client' in globals():  # New SDK (v1.0.0+)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=500,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    else:  # Legacy SDK (<v1.0.0)
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=500,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    
                    st.success("Optimized bullet points generated!")
                    st.markdown("### AI-Generated Bullet Points")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error generating bullet points: {str(e)}")
    
    # Tab 4: Description
    with tabs[3]:
        st.markdown("### Product Description Optimization")
        st.markdown("""
        An effective Amazon product description should:
        - Expand on features and benefits
        - Use HTML formatting for readability
        - Include keywords naturally
        - Address customer objections
        - Provide use cases and scenarios
        """)
        
        # Extract current description (excluding title and bullets)
        current_description = ""
        if product_info['description']:
            lines = product_info['description'].split('\n')
            if len(lines) > 1:
                # Skip the first line (title) and any bullet points
                in_description = False
                for line in lines[1:]:
                    if not line.strip().startswith('•') and not line.strip().startswith('-') and not line.strip().startswith('*') and line.strip():
                        in_description = True
                        current_description += line + "\n"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Description")
            st.text_area("Current Description", value=current_description, height=300, disabled=True)
        
        with col2:
            st.markdown("#### Optimized Description")
            
            # Get AI recommendations for description if available
            if product_info['asin'] in st.session_state.ai_insights and 'listing_optimization' in st.session_state.ai_insights[product_info['asin']]['insights']:
                listing_insights = st.session_state.ai_insights[product_info['asin']]['insights']['listing_optimization']
                
                # Try to extract description recommendations
                description_section = ""
                if "Description" in listing_insights:
                    lines = listing_insights.split('\n')
                    in_description_section = False
                    for line in lines:
                        if "Description" in line:
                            in_description_section = True
                            description_section += line + "\n"
                        elif in_description_section and line.strip() and not any(section in line for section in ["Title", "Bullet", "Image", "Keyword"]):
                            description_section += line + "\n"
                        elif in_description_section and any(section in line for section in ["Title", "Bullet", "Image", "Keyword"]):
                            in_description_section = False
                
                if description_section:
                    st.markdown(description_section)
                else:
                    st.info("Generate full listing recommendations to see description optimization suggestions")
            else:
                st.info("Run AI Optimization to get description recommendations")
        
        # Generate description button
        if st.button("Generate Optimized Description", key="description_button"):
            with st.spinner("Generating optimized description..."):
                # Increment the API call counter
                if 'ai_api_calls' in st.session_state:
                    st.session_state.ai_api_calls += 1
                
                prompt = f"""Create an optimized Amazon product description for this medical device:
                
                Product: {product_info['name']}
                Category: {product_info['category']}
                Current Description: {current_description}
                
                Follow Amazon's best practices:
                - Write in HTML format with paragraph tags (<p>) and line breaks
                - Expand on features and benefits beyond the bullet points
                - Include keywords naturally throughout the text
                - Address potential customer questions and objections
                - Describe specific use cases and scenarios
                - Highlight quality, comfort, ease of use, and durability
                - Avoid excessive capitalization and promotional language
                
                The description should be 3-5 paragraphs and include HTML formatting.
                """
                
                messages = [
                    {"role": "system", "content": "You are an expert Amazon listing optimization specialist for medical devices."},
                    {"role": "user", "content": prompt}
                ]
                
                try:
                    # Check which version of the OpenAI SDK is being used
                    if 'client' in globals():  # New SDK (v1.0.0+)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    else:  # Legacy SDK (<v1.0.0)
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    
                    st.success("Optimized description generated!")
                    st.markdown("### AI-Generated Description")
                    st.code(result, language="html")
                    
                    # Display preview
                    st.markdown("### Preview")
                    st.markdown(result, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating description: {str(e)}")
    
    # Tab 5: Keywords
    with tabs[4]:
        st.markdown("### Keyword Optimization")
        st.markdown("""
        Effective Amazon keywords strategy:
        - Include relevant search terms customers use
        - Focus on long-tail keywords for medical devices
        - Include synonyms and related terms
        - Add appropriate medical terminology
        - Consider customer language (not just technical terms)
        """)
        
        # Get AI recommendations for keywords if available
        if product_info['asin'] in st.session_state.ai_insights and 'listing_optimization' in st.session_state.ai_insights[product_info['asin']]['insights']:
            listing_insights = st.session_state.ai_insights[product_info['asin']]['insights']['listing_optimization']
            
            # Try to extract keyword recommendations
            keyword_section = ""
            if "Keyword" in listing_insights:
                lines = listing_insights.split('\n')
                in_keyword_section = False
                for line in lines:
                    if "Keyword" in line:
                        in_keyword_section = True
                        keyword_section += line + "\n"
                    elif in_keyword_section and line.strip() and not any(section in line for section in ["Title", "Bullet", "Description", "Image"]):
                        keyword_section += line + "\n"
                    elif in_keyword_section and any(section in line for section in ["Title", "Bullet", "Description", "Image"]):
                        in_keyword_section = False
            
            if keyword_section:
                st.markdown("#### Keyword Recommendations")
                st.markdown(keyword_section)
            else:
                st.info("Generate full listing recommendations to see keyword suggestions")
        else:
            st.info("Run AI Optimization to get keyword recommendations")
        
        # Generate keywords button
        if st.button("Generate Keyword Recommendations", key="keyword_button"):
            with st.spinner("Generating keyword recommendations..."):
                # Increment the API call counter
                if 'ai_api_calls' in st.session_state:
                    st.session_state.ai_api_calls += 1
                
                prompt = f"""Generate keyword recommendations for this Amazon medical device listing:
                
                Product: {product_info['name']}
                Category: {product_info['category']}
                Description: {product_info['description']}
                
                Please provide:
                1. Primary keywords (5-7 most important search terms)
                2. Secondary keywords (8-10 additional relevant terms)
                3. Long-tail keyword phrases (5-7 specific search phrases)
                4. Backend keywords (for Amazon backend search terms field)
                5. Competitor keywords (terms used by top competitors)
                
                Focus on medical terminology, symptoms, conditions, and use cases relevant to this product.
                """
                
                messages = [
                    {"role": "system", "content": "You are an expert Amazon SEO specialist for medical devices with deep knowledge of search patterns and keyword optimization."},
                    {"role": "user", "content": prompt}
                ]
                
                try:
                    # Check which version of the OpenAI SDK is being used
                    if 'client' in globals():  # New SDK (v1.0.0+)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    else:  # Legacy SDK (<v1.0.0)
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    
                    st.success("Keyword recommendations generated!")
                    st.markdown("### AI-Generated Keywords")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error generating keywords: {str(e)}")
    
    # Tab 6: Images
    with tabs[5]:
        st.markdown("### Image Optimization")
        st.markdown("""
        Amazon product image best practices:
        - Main image with white background
        - Multiple images showing different angles
        - In-use/lifestyle images
        - Size reference images
        - Feature highlight images
        - Comparison images
        - Packaging images
        """)
        
        # Get AI recommendations for images if available
        if product_info['asin'] in st.session_state.ai_insights and 'listing_optimization' in st.session_state.ai_insights[product_info['asin']]['insights']:
            listing_insights = st.session_state.ai_insights[product_info['asin']]['insights']['listing_optimization']
            
            # Try to extract image recommendations
            image_section = ""
            if "Image" in listing_insights:
                lines = listing_insights.split('\n')
                in_image_section = False
                for line in lines:
                    if "Image" in line:
                        in_image_section = True
                        image_section += line + "\n"
                    elif in_image_section and line.strip() and not any(section in line for section in ["Title", "Bullet", "Description", "Keyword"]):
                        image_section += line + "\n"
                    elif in_image_section and any(section in line for section in ["Title", "Bullet", "Description", "Keyword"]):
                        in_image_section = False
            
            if image_section:
                st.markdown("#### Image Recommendations")
                st.markdown(image_section)
            else:
                st.info("Generate full listing recommendations to see image optimization suggestions")
        else:
            st.info("Run AI Optimization to get image recommendations")
        
        # Image checklist
        st.markdown("### Amazon Image Checklist")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Essential Images")
            main_image = st.checkbox("Main image with white background", value=False)
            alternate_angles = st.checkbox("3+ alternate angle images", value=False)
            product_in_use = st.checkbox("Product in use / lifestyle image", value=False)
            size_reference = st.checkbox("Size reference image", value=False)
            features_highlighted = st.checkbox("Features/benefits highlighted", value=False)
        
        with col2:
            st.markdown("#### Enhanced Images")
            packaging = st.checkbox("Packaging image", value=False)
            accessories = st.checkbox("Accessories/included items", value=False)
            instructions = st.checkbox("How-to-use visual guide", value=False)
            comparison = st.checkbox("Comparison with similar products", value=False)
            infographic = st.checkbox("Benefits infographic", value=False)
        
        # Calculate image score
        essential_count = sum([main_image, alternate_angles, product_in_use, size_reference, features_highlighted])
        enhanced_count = sum([packaging, accessories, instructions, comparison, infographic])
        
        image_score = (essential_count / 5) * 70 + (enhanced_count / 5) * 30
        
        # Display image score
        st.markdown(f"### Image Score: {image_score:.1f}/100")
        
        # Image score gauge
        if AVAILABLE_MODULES['plotly']:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = image_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Image Optimization Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "khaki"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate image recommendations button
        if st.button("Generate Image Recommendations", key="image_button"):
            with st.spinner("Generating image recommendations..."):
                # Increment the API call counter
                if 'ai_api_calls' in st.session_state:
                    st.session_state.ai_api_calls += 1
                
                prompt = f"""Create detailed image recommendations for this Amazon medical device listing:
                
                Product: {product_info['name']}
                Category: {product_info['category']}
                Description: {product_info['description']}
                
                Current image score: {image_score}/100
                
                Provide the following:
                1. Specific types of images needed for this product (7-9 total images)
                2. Key features that should be highlighted in close-ups
                3. How to show the product in use (lifestyle images)
                4. Size reference recommendations
                5. Any infographics that would help explain benefits
                6. How to visually address common customer questions/concerns
                
                Be specific to this type of medical device and focus on images that would increase conversion rate.
                """
                
                messages = [
                    {"role": "system", "content": "You are an expert Amazon listing optimization specialist for medical devices with expertise in product photography and image optimization."},
                    {"role": "user", "content": prompt}
                ]
                
                try:
                    # Check which version of the OpenAI SDK is being used
                    if 'client' in globals():  # New SDK (v1.0.0+)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    else:  # Legacy SDK (<v1.0.0)
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        result = response.choices[0].message.content
                    
                    st.success("Image recommendations generated!")
                    st.markdown("### AI-Generated Image Recommendations")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error generating image recommendations: {str(e)}")
    
    # Tab 7: Competitive Analysis
    with tabs[6]:
        st.markdown("### Competitive Analysis")
        
        # Get competitors data
        competitors = []
        if 'competitors' in st.session_state.uploaded_files and product_info['asin'] in st.session_state.uploaded_files['competitors']:
            competitors = st.session_state.uploaded_files['competitors'][product_info['asin']]
        
        if competitors:
            st.markdown(f"#### {len(competitors)} Competing Products")
            
            for comp in competitors:
                st.markdown(f"- {comp['name']} ([{comp['asin']}](https://www.amazon.com/dp/{comp['asin']}))")
        else:
            st.info("No competitors data available. Add competing products in the Manual Entry tab.")
        
        # Add competitor form
        with st.form("add_competitor"):
            st.markdown("### Add Competitor")
            comp_asin = st.text_input("Competitor ASIN")
            comp_name = st.text_input("Competitor Product Name")
            
            submitted = st.form_submit_button("Add Competitor")
            
            if submitted and comp_asin and comp_name:
                # Add to competitors list
                if 'competitors' not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files['competitors'] = {}
                
                if product_info['asin'] not in st.session_state.uploaded_files['competitors']:
                    st.session_state.uploaded_files['competitors'][product_info['asin']] = []
                
                st.session_state.uploaded_files['competitors'][product_info['asin']].append({
                    "asin": comp_asin,
                    "name": comp_name,
                    "product_asin": product_info['asin']
                })
                
                st.success(f"Added competitor: {comp_name}")
                st.rerun()
        
        # Competitive analysis button
        if st.button("Generate Competitive Analysis", key="competitive_button"):
            if not competitors:
                st.warning("Please add competitors first to perform competitive analysis")
            else:
                with st.spinner("Generating competitive analysis..."):
                    # Increment the API call counter
                    if 'ai_api_calls' in st.session_state:
                        st.session_state.ai_api_calls += 1
                    
                    competitor_list = "\n".join([f"- {comp['name']} (ASIN: {comp['asin']})" for comp in competitors])
                    
                    prompt = f"""Perform a competitive analysis for this Amazon medical device and its competitors:
                    
                    Your Product: {product_info['name']} (ASIN: {product_info['asin']})
                    Category: {product_info['category']}
                    
                    Competitors:
                    {competitor_list}
                    
                    Provide the following analysis:
                    1. Competitive positioning strategy
                    2. Key differentiators to highlight in your listing
                    3. Price positioning recommendations
                    4. Feature comparison strategy
                    5. Unique selling propositions to emphasize
                    6. Weaknesses of competitors to address
                    7. Customer pain points competitors aren't solving
                    
                    Focus on how to optimize the Amazon listing to stand out from these specific competitors.
                    """
                    
                    messages = [
                        {"role": "system", "content": "You are an expert Amazon marketplace strategist specializing in competitive analysis and differentiation for medical device products."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    try:
                        # Check which version of the OpenAI SDK is being used
                        if 'client' in globals():  # New SDK (v1.0.0+)
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages,
                                max_tokens=1000,
                                temperature=0.2
                            )
                            result = response.choices[0].message.content
                        else:  # Legacy SDK (<v1.0.0)
                            response = openai.ChatCompletion.create(
                                model="gpt-4o",
                                messages=messages,
                                max_tokens=1000,
                                temperature=0.2
                            )
                            result = response.choices[0].message.content
                        
                        st.success("Competitive analysis generated!")
                        st.markdown("### AI-Generated Competitive Analysis")
                        st.markdown(result)
                        
                        # Store in AI insights
                        if product_info['asin'] not in st.session_state.ai_insights:
                            st.session_state.ai_insights[product_info['asin']] = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'insights': {}
                            }
                        
                        st.session_state.ai_insights[product_info['asin']]['insights']['competitive_analysis'] = result
                    except Exception as e:
                        st.error(f"Error generating competitive analysis: {str(e)}")

def render_ai_insights():
    """Render the AI insights section."""
    st.header("AI Insights")
    
    if not AVAILABLE_MODULES['ai_api']:
        st.warning("AI insights require OpenAI API integration. Please add an API key named 'openai_api_key' in your Streamlit app settings.")
        return
    
    # Check if we have AI insights to display
    if not st.session_state.ai_insights:
        st.info("No AI insights available. Please run an AI-Enhanced Analysis on a product.")
        return
    
    # If we have a current product selected, display its AI insights
    if st.session_state.current_product:
        product_asin = st.session_state.current_product['asin']
        
        if product_asin in st.session_state.ai_insights:
            insights = st.session_state.ai_insights[product_asin]
            product_info = st.session_state.current_product
            
            # Display info
            st.subheader(f"AI Insights for {product_info['name']} ({product_info['asin']})")
            st.caption(f"Analyzed on: {insights['timestamp']}")
            
            # Create tabs for different sections of the AI insights
            tabs = st.tabs(["Overview", "Listing Optimization", "Review Analysis", "Return Analysis", "Export"])
            
            # Tab 1: Overview
            with tabs[0]:
                st.markdown("### Key Insights Summary")
                
                # Overview dashboard with metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Return Rate", f"{product_info.get('return_rate_30d', 0):.2f}%")
                    
                    # Benchmarking
                    if product_info.get('return_rate_30d', 0) < 3:
                        st.success("Below average return rate for medical devices")
                    elif product_info.get('return_rate_30d', 0) < 7:
                        st.info("Average return rate for medical devices")
                    else:
                        st.error("Above average return rate for medical devices")
                
                with col2:
                    if product_info.get('star_rating'):
                        st.metric("Star Rating", f"{product_info.get('star_rating', 0):.1f} ★")
                        
                        # Rating benchmarking
                        if product_info.get('star_rating', 0) >= 4.5:
                            st.success("Excellent rating")
                        elif product_info.get('star_rating', 0) >= 4.0:
                            st.info("Good rating")
                        elif product_info.get('star_rating', 0) >= 3.5:
                            st.warning("Average rating")
                        else:
                            st.error("Below average rating")
                
                with col3:
                    # Calculate conversion impact
                    if product_info.get('return_rate_30d') is not None:
                        potential_savings = product_info.get('sales_30d', 0) * (product_info.get('return_rate_30d', 0) / 100) * 20  # $20 per return saved
                        st.metric("Potential Monthly Savings", f"${potential_savings:.2f}")
                        st.caption("If return rate improved by 1%")
                
                # Display recommendations if available
                if 'recommendations' in insights['insights']:
                    st.markdown("### AI Product Recommendations")
                    st.markdown(insights['insights']['recommendations'])
                else:
                    st.info("Run an AI-Enhanced Analysis to generate comprehensive recommendations.")
            
            # Tab 2: Listing Optimization
            with tabs[1]:
                if 'listing_optimization' in insights['insights']:
                    st.markdown("### Listing Optimization Recommendations")
                    st.markdown(insights['insights']['listing_optimization'])
                    
                    # Add competitive analysis if available
                    if 'competitive_analysis' in insights['insights']:
                        st.markdown("### Competitive Analysis")
                        st.markdown(insights['insights']['competitive_analysis'])
                else:
                    st.info("No listing optimization insights available. Go to the Listing Optimization tab to generate recommendations.")
            
            # Tab 3: Review Analysis
            with tabs[2]:
                if 'review_insights' in insights['insights'] and insights['insights']['review_insights']:
                    review_insights = insights['insights']['review_insights']
                    
                    st.markdown(f"### Analysis of {len(review_insights)} Reviews")
                    
                    # Extract common themes
                    positive_themes = set()
                    negative_themes = set()
                    improvement_suggestions = set()
                    
                    # Process all reviews to extract themes
                    for insight in review_insights:
                        analysis = insight['analysis']
                        
                        # Look for positive feedback
                        if "positive" in analysis.lower():
                            # Extract positive themes with simple pattern matching
                            for line in analysis.split('\n'):
                                if "positive" in line.lower() and ":" in line:
                                    theme = line.split(":", 1)[1].strip()
                                    positive_themes.add(theme)
                        
                        # Look for negative feedback
                        if "negative" in analysis.lower() or "concern" in analysis.lower():
                            # Extract negative themes with simple pattern matching
                            for line in analysis.split('\n'):
                                if any(term in line.lower() for term in ["negative", "concern", "issue"]) and ":" in line:
                                    theme = line.split(":", 1)[1].strip()
                                    negative_themes.add(theme)
                        
                        # Look for improvement suggestions
                        if "improve" in analysis.lower() or "suggestion" in analysis.lower():
                            for line in analysis.split('\n'):
                                if any(term in line.lower() for term in ["improve", "suggestion", "recommend"]) and ":" in line:
                                    suggestion = line.split(":", 1)[1].strip()
                                    improvement_suggestions.add(suggestion)
                    
                    # Display themes
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Common Positive Themes")
                        if positive_themes:
                            for theme in positive_themes:
                                st.success(theme)
                        else:
                            st.info("No consistent positive themes identified")
                    
                    with col2:
                        st.markdown("#### Common Negative Themes")
                        if negative_themes:
                            for theme in negative_themes:
                                st.error(theme)
                        else:
                            st.info("No consistent negative themes identified")
                    
                    st.markdown("#### Improvement Suggestions")
                    if improvement_suggestions:
                        for suggestion in improvement_suggestions:
                            st.info(suggestion)
                    else:
                        st.info("No specific improvement suggestions identified")
                    
                    # Individual review insights
                    st.markdown("#### Individual Review Analysis")
                    for i, insight in enumerate(review_insights):
                        with st.expander(f"Review {i+1}: {insight['review'][:50]}... (Rating: {insight['rating']})"):
                            st.markdown("**Original Review**")
                            st.write(insight['review'])
                            
                            st.markdown("**AI Analysis**")
                            st.markdown(insight['analysis'])
                else:
                    st.info("No AI review insights available. Run an AI-Enhanced Analysis with reviews data to generate insights.")
            
            # Tab 4: Return Analysis
            with tabs[3]:
                if 'return_insights' in insights['insights'] and insights['insights']['return_insights']:
                    return_insights = insights['insights']['return_insights']
                    
                    st.markdown(f"### Analysis of {len(return_insights)} Return Reasons")
                    
                    # Extract common return categories
                    quality_issues = []
                    sizing_issues = []
                    expectation_issues = []
                    instruction_issues = []
                    other_issues = []
                    
                    # Process all returns to categorize
                    for insight in return_insights:
                        return_reason = insight['return_reason'].lower()
                        analysis = insight['analysis'].lower()
                        
                        if any(term in return_reason or term in analysis for term in ["quality", "defect", "broke", "damaged", "not working"]):
                            quality_issues.append(insight)
                        elif any(term in return_reason or term in analysis for term in ["size", "fit", "too small", "too large"]):
                            sizing_issues.append(insight)
                        elif any(term in return_reason or term in analysis for term in ["expect", "thought", "not as described", "not like picture"]):
                            expectation_issues.append(insight)
                        elif any(term in return_reason or term in analysis for term in ["confus", "instruction", "manual", "how to", "couldn't figure"]):
                            instruction_issues.append(insight)
                        else:
                            other_issues.append(insight)
                    
                    # Calculate percentages
                    total_returns = len(return_insights)
                    return_categories = {
                        "Quality Issues": len(quality_issues),
                        "Sizing Issues": len(sizing_issues),
                        "Expectation Mismatch": len(expectation_issues),
                        "Instruction Problems": len(instruction_issues),
                        "Other Issues": len(other_issues)
                    }
                    
                    # Create visualization
                    if AVAILABLE_MODULES['plotly']:
                        fig = px.pie(
                            names=list(return_categories.keys()),
                            values=list(return_categories.values()),
                            title="Return Reasons Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display return categories
                    for category, items in [
                        ("Quality Issues", quality_issues),
                        ("Sizing Issues", sizing_issues),
                        ("Expectation Mismatch", expectation_issues),
                        ("Instruction Problems", instruction_issues),
                        ("Other Issues", other_issues)
                    ]:
                        if items:
                            with st.expander(f"{category} ({len(items)})", expanded=(category == "Quality Issues")):
                                for item in items:
                                    st.markdown(f"**Return Reason:** {item['return_reason']}")
                                    st.markdown(f"**Analysis:** {item['analysis']}")
                                    st.divider()
                    
                    # Generate recommendations
                    if st.button("Generate Return Reduction Plan", key="return_plan"):
                        with st.spinner("Generating return reduction plan..."):
                            # Increment the API call counter
                            if 'ai_api_calls' in st.session_state:
                                st.session_state.ai_api_calls += 1
                            
                            # Prepare categories breakdown
                            categories_text = "\n".join([f"{category}: {count} returns ({count/total_returns*100:.1f}%)" 
                                                        for category, count in return_categories.items()])
                            
                            # Prepare return reasons text
                            return_reasons_text = "\n".join([f"- {insight['return_reason']}" for insight in return_insights])
                            
                            prompt = f"""Create an actionable return reduction plan for this Amazon medical device:
                            
                            Product: {product_info['name']} (ASIN: {product_info['asin']})
                            Category: {product_info['category']}
                            Current Return Rate: {product_info.get('return_rate_30d', 0):.2f}%
                            
                            Return Reasons Categories:
                            {categories_text}
                            
                            Specific Return Reasons:
                            {return_reasons_text}
                            
                            Please provide:
                            1. Immediate listing changes to reduce returns (top 3 priorities)
                            2. Product improvement recommendations
                            3. Packaging/instructions improvements
                            4. Customer expectation management strategies
                            5. Expected impact on return rate for each recommendation
                            
                            The plan should be specific, actionable, and prioritized by impact.
                            """
                            
                            messages = [
                                {"role": "system", "content": "You are an expert Amazon listing optimization specialist with deep expertise in reducing return rates for medical devices."},
                                {"role": "user", "content": prompt}
                            ]
                            
                            try:
                                # Check which version of the OpenAI SDK is being used
                                if 'client' in globals():  # New SDK (v1.0.0+)
                                    response = client.chat.completions.create(
                                        model="gpt-4o",
                                        messages=messages,
                                        max_tokens=1000,
                                        temperature=0.2
                                    )
                                    result = response.choices[0].message.content
                                else:  # Legacy SDK (<v1.0.0)
                                    response = openai.ChatCompletion.create(
                                        model="gpt-4o",
                                        messages=messages,
                                        max_tokens=1000,
                                        temperature=0.2
                                    )
                                    result = response.choices[0].message.content
                                
                                st.success("Return reduction plan generated!")
                                st.markdown("### Return Reduction Plan")
                                st.markdown(result)
                                
                                # Store in AI insights
                                st.session_state.ai_insights[product_info['asin']]['insights']['return_reduction_plan'] = result
                            except Exception as e:
                                st.error(f"Error generating return reduction plan: {str(e)}")
                else:
                    st.info("No AI return insights available. Run an AI-Enhanced Analysis with return reasons data to generate insights.")
            
            # Tab 5: Export
            with tabs[4]:
                st.markdown("### Export AI Insights")
                
                # Prepare content for export
                export_content = f"# AI Insights for {product_info['name']} ({product_info['asin']})\n"
                export_content += f"Generated on: {insights['timestamp']}\n\n"
                
                if 'recommendations' in insights['insights']:
                    export_content += "## Recommendations\n"
                    export_content += insights['insights']['recommendations'] + "\n\n"
                
                if 'listing_optimization' in insights['insights']:
                    export_content += "## Listing Optimization\n"
                    export_content += insights['insights']['listing_optimization'] + "\n\n"
                
                if 'return_reduction_plan' in insights['insights']:
                    export_content += "## Return Reduction Plan\n"
                    export_content += insights['insights']['return_reduction_plan'] + "\n\n"
                
                if 'competitive_analysis' in insights['insights']:
                    export_content += "## Competitive Analysis\n"
                    export_content += insights['insights']['competitive_analysis'] + "\n\n"
                
                # Download buttons
                st.download_button(
                    label="Export as Markdown",
                    data=export_content,
                    file_name=f"{product_info['asin']}_ai_insights.md",
                    mime="text/markdown"
                )
                
                # Export to Excel (if xlsxwriter available)
                if HAS_LOCAL_MODULES and AVAILABLE_MODULES['xlsx_writer'] and AVAILABLE_MODULES['pandas'] and 'review_insights' in insights['insights']:
                    try:
                        # Create Excel output
                        output = io.BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        
                        # Create review insights sheet
                        if 'review_insights' in insights['insights'] and insights['insights']['review_insights']:
                            reviews_data = []
                            for insight in insights['insights']['review_insights']:
                                reviews_data.append({
                                    'Review': insight['review'],
                                    'Rating': insight['rating'],
                                    'Analysis': insight['analysis']
                                })
                            
                            reviews_df = pd.DataFrame(reviews_data)
                            reviews_df.to_excel(writer, sheet_name='Review Analysis', index=False)
                        
                        # Create return insights sheet
                        if 'return_insights' in insights['insights'] and insights['insights']['return_insights']:
                            returns_data = []
                            for insight in insights['insights']['return_insights']:
                                returns_data.append({
                                    'Return Reason': insight['return_reason'],
                                    'Analysis': insight['analysis']
                                })
                            
                            returns_df = pd.DataFrame(returns_data)
                            returns_df.to_excel(writer, sheet_name='Return Analysis', index=False)
                        
                        # Create recommendations sheet
                        if 'recommendations' in insights['insights']:
                            recommendations_df = pd.DataFrame({
                                'Section': ['Product Recommendations'],
                                'Content': [insights['insights']['recommendations']]
                            })
                            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
                        
                        # Create listing optimization sheet
                        if 'listing_optimization' in insights['insights']:
                            listing_df = pd.DataFrame({
                                'Section': ['Listing Optimization'],
                                'Content': [insights['insights']['listing_optimization']]
                            })
                            listing_df.to_excel(writer, sheet_name='Listing Optimization', index=False)
                        
                        # Save the workbook
                        writer.close()
                        
                        # Provide download button for Excel
                        st.download_button(
                            label="Export as Excel",
                            data=output.getvalue(),
                            file_name=f"{product_info['asin']}_ai_insights.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error creating Excel export: {str(e)}")
                else:
                    st.warning("Excel export requires xlsxwriter and pandas. These modules may not be available.")
        else:
            st.warning(f"No AI insights available for {st.session_state.current_product['name']}. Please run an AI-Enhanced Analysis first.")
    else:
        # If no current product, let the user select from available analyses
        available_insights = list(st.session_state.ai_insights.keys())
        if available_insights:
            selected_asin = st.selectbox("Select a previously analyzed product:", available_insights)
            if st.button("Show AI Insights"):
                # Find the product info from analysis results
                if selected_asin in st.session_state.analysis_results:
                    st.session_state.current_product = st.session_state.analysis_results[selected_asin]['product_info']
                    st.rerun()
                else:
                    st.error("Product information not found. Please run a standard analysis first.")
        else:
            st.info("No AI insights available. Please select and analyze a product with AI-Enhanced Analysis.")

def render_help_section():
    """Render the help and documentation section."""
    st.header("Help & Documentation")
    
    help_tabs = st.tabs(["Quick Start", "Import Options", "AI Features", "Support"])
    
    with help_tabs[0]:
        st.subheader("Quick Start Guide")
        st.markdown("""
        ### 1. Import Your Amazon Data
        - Click the **Load Example Data** button in the sidebar to try with sample data
        - Or upload your own data in the **Import** tab
        - You can import structured data (CSV/Excel), enter data manually, or upload images/screenshots
        
        ### 2. Select and Analyze a Product
        - In the **Analyze** tab, select a product from your imported data
        - Click **AI-Enhanced Analysis** to get deep insights
        
        ### 3. Optimize Your Amazon Listing
        - The **Listing Optimization** tab provides recommendations for:
          - Title optimization
          - Bullet points
          - Product description
          - Keywords
          - Images
          - Competitive positioning
        
        ### 4. Review Detailed AI Insights
        - The **AI Insights** tab breaks down:
          - Review analysis and patterns
          - Return reasons analysis
          - Specific improvement recommendations
        
        ### 5. Export Recommendations
        - Export insights as Markdown or Excel for sharing with your team
        """)
        
        st.info("💡 **Pro Tip:** Start with the example data to explore all features, then import your actual product data when you're ready.")
    
    with help_tabs[1]:
        st.subheader("Data Import Options")
        
        st.markdown("""
        ### Structured Data Import
        Upload a CSV or Excel file with your Amazon product data. Required columns:
        - **ASIN** (Amazon Standard Identification Number)
        - **Last 30 Days Sales** 
        - **Last 30 Days Returns**
        
        Additional columns that improve analysis:
        - Product Name
        - SKU
        - Category
        - Product Description
        - Star Rating
        - Total Reviews
        - Last 365 Days Sales/Returns
        
        ### Manual Entry
        Enter data for a single product when you don't have a spreadsheet:
        - Basic product info
        - Sales and returns
        - Reviews and return reasons
        - Competing products
        
        ### Image/Review Import
        Upload screenshots of:
        - Amazon product reviews
        - Return reports
        - Your product listing
        - Competitor listings
        
        The system uses OCR to extract text for analysis.
        
        ### Historical Data
        Upload historical sales/returns data to identify trends and seasonality.
        """)
    
    with help_tabs[2]:
        st.subheader("AI Features")
        
        st.markdown("""
        This tool uses OpenAI's GPT-4o model to provide advanced e-commerce optimization:
        
        ### Listing Optimization
        - **Title Optimization**: Keyword-rich, conversion-focused titles
        - **Bullet Points**: Benefit-focused feature highlights
        - **Description Enhancement**: Persuasive, well-formatted content
        - **Keyword Research**: Relevant search terms for your specific medical device
        - **Image Recommendations**: Specific photo types to improve conversions
        
        ### Customer Feedback Analysis
        - **Review Pattern Detection**: Identifies common themes in customer reviews
        - **Return Reason Analysis**: Categorizes and prioritizes return issues
        - **Sentiment Analysis**: Extracts positive and negative feedback
        
        ### Strategic Recommendations
        - **Return Reduction Plans**: Targeted actions to reduce return rates
        - **Competitive Analysis**: Positioning against similar products
        - **Product Improvement Suggestions**: Based on customer feedback
        
        ### Data Visualization
        - Return rate trends
        - Review sentiment analysis
        - Listing quality scoring
        - Return reason categorization
        """)
        
        st.warning("Note: The AI requires your OpenAI API key to be configured in Streamlit app settings. The API key should be named 'openai_api_key'.")
    
    with help_tabs[3]:
        st.subheader("Support Resources")
        
        st.markdown(f"""
        If you encounter issues or need assistance:
        
        * **Contact Support**: Email [{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL})
        * **Report Bugs**: Please include screenshots and step-by-step reproduction steps
        * **Request Features**: We welcome suggestions for new features or improvements
        
        Available modules are shown in the sidebar. If a module is marked as unavailable,
        that functionality will be limited in the application.
        """)
        
        st.info("💡 **Tip:** Use the 'Load Example Data' button in the sidebar to see how the app works with sample data.")

# Run the application
if __name__ == "__main__":
    main()
