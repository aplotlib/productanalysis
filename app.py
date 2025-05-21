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
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SUPPORT_EMAIL = "alexander.popoff@vivehealth.com"
API_KEY_NAME = "OPENAI_API_KEY"  # For storing in streamlit secrets

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

# Try importing requests
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

# Try importing OpenAI - for AI analysis
try:
    import openai
    # Check if we can actually use the API
    api_key = os.environ.get(API_KEY_NAME) or st.secrets.get(API_KEY_NAME, None)
    if api_key:
        openai.api_key = api_key
        AVAILABLE_MODULES['ai_api'] = True
    else:
        logger.warning("OpenAI API key not found in environment or secrets")
except ImportError:
    logger.warning("OpenAI module not available")

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

# Medical device classification constants
FDA_DEVICE_CLASSES = {
    'Class I': 'Low risk, subject to general controls',
    'Class II': 'Moderate risk, subject to special controls', 
    'Class III': 'High risk, subject to premarket approval'
}

# Common medical device categories for Amazon
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

# Setup the Streamlit page
st.set_page_config(
    page_title="Medical Device Review Analysis Tool",
    page_icon="🩺",
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
        try:
            prompt = ""
            if analysis_type == 'sentiment':
                prompt = f"""Analyze the sentiment of this product review for a medical device. 
                Consider medical-specific language and concerns. Extract concerns about safety, 
                efficacy, comfort, ease of use, and durability. Classify the overall sentiment 
                as positive, negative, or neutral.
                
                Review: {text}
                """
            elif analysis_type == 'medical_concerns':
                prompt = f"""Identify any medical concerns or adverse events mentioned in this review 
                for a medical device. Flag any mentions of: pain, discomfort, injury, malfunction, 
                inaccuracy, allergic reactions, or other health-related issues. Also extract any 
                mentions of specific medical conditions or treatments.
                
                Review: {text}
                """
            elif analysis_type == 'compliance_issues':
                prompt = f"""Analyze this medical device product review for potential regulatory 
                compliance issues. Flag if the review mentions: product claims beyond labeled use, 
                product defects, safety issues, manufacturing quality concerns, or potential 
                reportable events. Categorize the compliance risk as: none, low, medium, or high.
                
                Review: {text}
                """
            elif analysis_type == 'return_analysis':
                prompt = f"""Analyze this medical device return reason to identify root causes. 
                Determine if the return is due to: defect, user error, unmet expectations, 
                competitive comparison, medical efficacy, comfort, or sizing issues.
                
                Return reason: {text}
                """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in medical device quality analysis, FDA regulations, and customer feedback interpretation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def classify_medical_device(product_name, category, description=""):
        """
        Use AI to classify a medical device according to FDA risk classes
        
        Parameters:
        - product_name: Name of the product
        - category: Product category
        - description: Product description if available
        
        Returns:
        - Dictionary with classification and reasoning
        """
        try:
            prompt = f"""Classify the following medical device according to FDA risk classes (I, II, or III).
            Consider the device type, intended use, and risk profile.
            
            Product name: {product_name}
            Category: {category}
            Description: {description}
            
            Provide the following:
            1. Likely FDA class (I, II, or III)
            2. Brief reasoning for the classification
            3. Potential regulatory considerations
            4. Whether the device likely requires prescription or is OTC
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in medical device regulatory classification, FDA regulations, and compliance requirements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Device classification error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_improvement_recommendations(product_info, reviews_data, returns_data):
        """
        Generate AI-powered recommendations for product improvements
        
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
            
            prompt = f"""As a medical device quality expert, analyze the following product data and provide
            improvement recommendations:
            
            Product: {product_info.get('name', 'Unknown')}
            Category: {product_info.get('category', 'Medical Device')}
            30-Day Return Rate: {product_info.get('return_rate_30d', 'N/A')}%
            Star Rating: {product_info.get('star_rating', 'N/A')}
            
            Customer Reviews:
            {reviews_text}
            
            Return Reasons:
            {returns_text}
            
            Based on this data, provide:
            1. Top 3-5 actionable improvement recommendations
            2. Potential quality issues requiring investigation
            3. Risk assessment from a regulatory perspective
            4. Competitive differentiation opportunities
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in medical device quality improvement, FDA regulations, and customer experience optimization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            result = response.choices[0].message.content
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    """Main application entry point."""
    # Render header
    st.title("Medical Device Review Analysis Tool")
    st.subheader("Analyze product reviews, ratings, and returns for medical devices on Amazon")
    
    # Display available modules in sidebar
    with st.sidebar:
        st.header("Settings")
        st.write("Contact support: " + SUPPORT_EMAIL)
        
        st.subheader("Available Modules")
        for module, available in AVAILABLE_MODULES.items():
            if available:
                st.success(f"✅ {module}")
            else:
                st.error(f"❌ {module}")
        
        # AI API Configuration (if not available)
        if not AVAILABLE_MODULES['ai_api']:
            st.warning("AI analysis requires an OpenAI API key")
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            if api_key and st.button("Save API Key"):
                # In a production app, this should be stored securely
                os.environ[API_KEY_NAME] = api_key
                try:
                    import openai
                    openai.api_key = api_key
                    # Test the key
                    openai.Completion.create(
                        model="davinci",
                        prompt="Test",
                        max_tokens=5
                    )
                    AVAILABLE_MODULES['ai_api'] = True
                    st.success("API key verified and saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to verify API key: {str(e)}")
    
    # Main content tabs
    tabs = st.tabs(["Import", "Analyze", "Results", "AI Insights", "Help"])
    
    with tabs[0]:
        render_file_upload()
    
    with tabs[1]:
        render_product_selection()
    
    with tabs[2]:
        render_analysis_results()
    
    with tabs[3]:
        render_ai_insights()
    
    with tabs[4]:
        render_help_section()

def render_file_upload():
    """Render the file upload section."""
    st.header("Import Data")
    
    upload_tabs = st.tabs(["Structured Data Import", "Manual Entry", "Document Import", "Historical Data"])
    
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
                file_name="medical_device_analysis_template.xlsx",
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
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                else:
                    st.error("pandas module is not available for processing files.")
    
    # Tab 2: Manual Entry
    with upload_tabs[1]:
        st.markdown("""
        Manually enter product details for analysis. This is useful for analyzing a single product
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
            
            # Additional Medical Device Info
            st.subheader("Medical Device Information")
            col1, col2 = st.columns(2)
            
            with col1:
                device_class = st.selectbox(
                    "FDA Device Class (if known)",
                    options=["Unknown", "Class I", "Class II", "Class III"],
                    help="FDA classification for regulatory requirements"
                )
                prescription = st.selectbox(
                    "Prescription Status",
                    options=["OTC (Over-the-counter)", "Rx (Prescription)", "Unknown"],
                    help="Whether the device requires a prescription"
                )
            
            with col2:
                product_description = st.text_area(
                    "Product Description",
                    help="Brief description of the product for better AI analysis",
                    height=100
                )
            
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
            st.subheader("Manual Reviews & Return Reasons (Optional)")
            
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
                        "FDA Device Class": [device_class],
                        "Prescription Status": [prescription],
                        "Product Description": [product_description],
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
                        
                        # Display the entered data
                        st.subheader("Entered Product Data")
                        st.dataframe(df)
                    else:
                        st.error("pandas module is not available. Cannot process manual entry.")
                except Exception as e:
                    st.error(f"Error processing manual entry: {str(e)}")
    
    # Tab 3: Document Import
    with upload_tabs[2]:
        st.markdown("""
        Upload PDFs, images of reviews, return reports, or screenshots. 
        The system will use OCR to extract data when possible.
        """)
        
        if not AVAILABLE_MODULES['ocr']:
            st.warning("OCR processing is not available. To enable this feature, install pytesseract and pdf2image.")
        
        doc_files = st.file_uploader(
            "Upload documents (PDF, Images)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="documents",
            disabled=not AVAILABLE_MODULES['ocr']
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
                        "type": file_ext
                    })
            
            if processed_docs:
                st.session_state.uploaded_files['documents'] = processed_docs
                
                # Show preview of extracted text from first document
                if len(processed_docs) > 0:
                    with st.expander("Preview of extracted text"):
                        st.text(processed_docs[0]["text"][:1000] + "...")
        elif doc_files and not (HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']):
            st.error("OCR processing is not available. Document processing is skipped.")
    
    # Tab 4: Historical Data
    with upload_tabs[3]:
        st.markdown("""
        Upload historical data for trend analysis (optional).
        Use the same format as the structured data import.
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
        'device_class': product_row['FDA Device Class'] if 'FDA Device Class' in product_row else "Unknown",
        'prescription': product_row['Prescription Status'] if 'Prescription Status' in product_row else "Unknown",
        'description': product_row['Product Description'] if 'Product Description' in product_row else "",
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
    
    # Display selected product info in metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("30-Day Sales", f"{product_info['sales_30d']:,}")
    with col2:
        st.metric("30-Day Returns", f"{product_info['returns_30d']:,}")
    with col3:
        st.metric("30-Day Return Rate", f"{product_info['return_rate_30d']:.2f}%")
    with col4:
        if product_info['star_rating'] is not None:
            st.metric("Star Rating", f"{product_info['star_rating']:.1f} ★")
    
    # Display FDA device class if available
    if product_info['device_class'] != "Unknown":
        st.info(f"FDA Classification: {product_info['device_class']} - {FDA_DEVICE_CLASSES.get(product_info['device_class'], '')}")
    
    # Get review data for the selected product from documents if available
    reviews_data = []
    
    # 1. Check for OCR-extracted reviews from documents
    if 'documents' in st.session_state.uploaded_files and HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']:
        docs = st.session_state.uploaded_files['documents']
        
        for doc in docs:
            # Extract reviews from OCR text
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
    
    # AI Device Classification
    if AVAILABLE_MODULES['ai_api'] and product_info['device_class'] == "Unknown":
        if st.button("Use AI to Classify Device"):
            with st.spinner("Classifying medical device..."):
                classification = classify_medical_device(
                    product_info['name'],
                    product_info['category'],
                    product_info['description']
                )
                
                if classification["success"]:
                    st.info("AI Device Classification Result:")
                    st.markdown(classification["result"])
                else:
                    st.error(f"Classification failed: {classification['error']}")
    
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
                                'historical_data': historical_data
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
                                'historical_data': historical_data
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
                                    'historical_data': historical_data
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
                                        # Analyze medical concerns
                                        medical_concerns = analyze_with_ai(review_text, 'medical_concerns')
                                        # Analyze compliance issues
                                        compliance = analyze_with_ai(review_text, 'compliance_issues')
                                        
                                        review_insights.append({
                                            'review': review_text,
                                            'rating': review.get('rating', 'N/A'),
                                            'sentiment_analysis': sentiment.get('result', 'Analysis failed') if sentiment.get('success', False) else 'Analysis failed',
                                            'medical_concerns': medical_concerns.get('result', 'Analysis failed') if medical_concerns.get('success', False) else 'Analysis failed',
                                            'compliance_issues': compliance.get('result', 'Analysis failed') if compliance.get('success', False) else 'Analysis failed'
                                        })
                                
                                ai_insights['review_insights'] = review_insights
                            
                            # 2. Analyze return reasons with AI
                            if return_reasons_data:
                                return_insights = []
                                for return_data in return_reasons_data[:10]:  # Limit to 10 return reasons
                                    return_reason = return_data.get('return_reason', '')
                                    if return_reason:
                                        analysis = analyze_with_ai(return_reason, 'return_analysis')
                                        return_insights.append({
                                            'return_reason': return_reason,
                                            'analysis': analysis.get('result', 'Analysis failed') if analysis.get('success', False) else 'Analysis failed'
                                        })
                                
                                ai_insights['return_insights'] = return_insights
                            
                            # 3. Generate improvement recommendations
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
            st.warning("AI-Enhanced Analysis requires OpenAI API key")

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
            
            # Display info
            st.subheader(f"Analysis for {product_info['name']} ({product_info['asin']})")
            st.caption(f"Analyzed on: {result['timestamp']}")
            
            # Create tabs for different sections of the analysis
            tabs = st.tabs(["Summary", "Detailed Analysis", "Visualizations", "Export"])
            
            # Tab 1: Summary
            with tabs[0]:
                # Display regulatory information
                if product_info.get('device_class', 'Unknown') != 'Unknown':
                    st.info(f"FDA Classification: {product_info['device_class']} - {FDA_DEVICE_CLASSES.get(product_info['device_class'], '')}")
                
                st.markdown(analysis)
                
                # Show return rate benchmarks
                if product_info.get('return_rate_30d') is not None:
                    st.subheader("Return Rate Assessment")
                    
                    # Define benchmark ranges for medical devices
                    if product_info['return_rate_30d'] < 2.0:
                        st.success(f"Return rate ({product_info['return_rate_30d']:.2f}%) is excellent - well below industry average for medical devices")
                    elif product_info['return_rate_30d'] < 5.0:
                        st.info(f"Return rate ({product_info['return_rate_30d']:.2f}%) is good - near industry average for medical devices")
                    elif product_info['return_rate_30d'] < 10.0:
                        st.warning(f"Return rate ({product_info['return_rate_30d']:.2f}%) is elevated - above industry average for medical devices")
                    else:
                        st.error(f"Return rate ({product_info['return_rate_30d']:.2f}%) is high - significantly above industry average for medical devices")
            
            # Tab 2: Detailed Analysis
            with tabs[1]:
                if 'reviews_data' in result and result['reviews_data'] and HAS_LOCAL_MODULES:
                    review_analysis = data_analysis.analyze_reviews(result['reviews_data'])
                    
                    st.subheader("Review Analysis")
                    st.write(f"Total Reviews: {review_analysis['total_reviews']}")
                    
                    if review_analysis['average_rating']:
                        st.write(f"Average Rating: {review_analysis['average_rating']:.1f}")
                    
                    # Display sentiment breakdown
                    if review_analysis.get('sentiment'):
                        st.subheader("Sentiment Breakdown")
                        sentiment = review_analysis['sentiment']
                        
                        # Create columns for sentiment display
                        sent_col1, sent_col2, sent_col3 = st.columns(3)
                        with sent_col1:
                            st.metric("Positive", f"{sentiment.get('positive', 0)}")
                        with sent_col2:
                            st.metric("Neutral", f"{sentiment.get('neutral', 0)}")
                        with sent_col3:
                            st.metric("Negative", f"{sentiment.get('negative', 0)}")
                    
                    # Display common topics
                    if review_analysis.get('common_topics'):
                        st.subheader("Common Topics")
                        topics = sorted(review_analysis['common_topics'].items(), key=lambda x: x[1], reverse=True)
                        
                        # Medical device specific topics to highlight
                        med_device_concerns = ['pain', 'comfort', 'effective', 'quality', 'safety', 
                                              'adjustable', 'stability', 'support', 'instructions',
                                              'durable', 'easy', 'fit', 'size', 'material']
                        
                        for topic, count in topics[:10]:  # Top 10 topics
                            if any(concern in topic.lower() for concern in med_device_concerns):
                                st.warning(f"{topic}: {count} mentions - Medical Device Key Factor")
                            else:
                                st.write(f"{topic}: {count} mentions")
                    
                    # Display return reasons if available
                    if 'return_reasons_data' in result and result['return_reasons_data']:
                        st.subheader("Return Reasons")
                        return_reasons = [item.get('return_reason', '') for item in result['return_reasons_data']]
                        unique_reasons = {}
                        for reason in return_reasons:
                            if reason in unique_reasons:
                                unique_reasons[reason] += 1
                            else:
                                unique_reasons[reason] = 1
                        
                        # Medical device specific return reasons to highlight
                        med_device_returns = ['defective', 'broke', 'quality', 'safety', 'malfunction', 
                                             'uncomfortable', 'difficult', 'pain', 'damaged', 'wrong size']
                        
                        for reason, count in sorted(unique_reasons.items(), key=lambda x: x[1], reverse=True):
                            if any(r_word in reason.lower() for r_word in med_device_returns):
                                st.error(f"{reason}: {count} - Potential Quality Issue")
                            else:
                                st.write(f"{reason}: {count}")
                else:
                    st.write("No detailed analysis available for this product.")
            
            # Tab 3: Visualizations
            with tabs[2]:
                if 'reviews_data' in result and result['reviews_data'] and HAS_LOCAL_MODULES and AVAILABLE_MODULES['plotly']:
                    review_analysis = data_analysis.analyze_reviews(result['reviews_data'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment chart
                        sentiment_chart = data_analysis.create_sentiment_chart(review_analysis)
                        if sentiment_chart:
                            st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    with col2:
                        # Rating distribution chart
                        rating_chart = data_analysis.create_rating_distribution_chart(review_analysis)
                        if rating_chart:
                            st.plotly_chart(rating_chart, use_container_width=True)
                    
                    # Topics chart
                    topics_chart = data_analysis.create_topics_chart(review_analysis)
                    if topics_chart:
                        st.plotly_chart(topics_chart, use_container_width=True)
                    
                    # Return reasons chart if available
                    if 'return_reasons_data' in result and result['return_reasons_data'] and len(result['return_reasons_data']) > 0:
                        return_reasons = [item.get('return_reason', '') for item in result['return_reasons_data']]
                        reason_counts = {}
                        for reason in return_reasons:
                            if reason in reason_counts:
                                reason_counts[reason] += 1
                            else:
                                reason_counts[reason] = 1
                        
                        fig = px.pie(
                            values=list(reason_counts.values()),
                            names=list(reason_counts.keys()),
                            title="Return Reasons",
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display historical trend if available
                    if 'historical_data' in result and result['historical_data'] is not None:
                        hist_data = result['historical_data']
                        if 'Date' in hist_data.columns and 'Return Rate' in hist_data.columns:
                            st.subheader("Historical Return Rate Trend")
                            fig = px.line(
                                hist_data, 
                                x='Date', 
                                y='Return Rate',
                                title="Return Rate Over Time",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    if not AVAILABLE_MODULES['plotly']:
                        st.warning("Plotly is not available. Visualizations cannot be displayed.")
                    elif not ('reviews_data' in result and result['reviews_data']):
                        st.warning("No review data available for visualization.")
                    else:
                        st.warning("Dependencies for visualization are not available.")
            
            # Tab 4: Export
            with tabs[3]:
                st.subheader("Export Analysis")
                
                # Export as text
                st.download_button(
                    label="Export as Text",
                    data=analysis,
                    file_name=f"{product_info['asin']}_analysis.txt",
                    mime="text/plain"
                )
                
                # Export to Excel (if xlsxwriter available)
                if HAS_LOCAL_MODULES and AVAILABLE_MODULES['xlsx_writer'] and AVAILABLE_MODULES['pandas']:
                    # Create structured data for export
                    if 'reviews_data' in result and result['reviews_data']:
                        review_analysis = data_analysis.analyze_reviews(result['reviews_data'])
                        
                        export_data = data_analysis.generate_report_data(
                            product_info, 
                            {"total_returns": product_info['returns_30d']}, 
                            review_analysis
                        )
                        
                        excel_data = data_analysis.create_excel_report(export_data)
                        
                        st.download_button(
                            label="Export as Excel Report",
                            data=excel_data,
                            file_name=f"{product_info['asin']}_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.warning("Excel export requires xlsxwriter and pandas. These modules are not available.")
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

def render_ai_insights():
    """Render the AI insights section."""
    st.header("AI Insights")
    
    if not AVAILABLE_MODULES['ai_api']:
        st.warning("AI insights require OpenAI API integration. Please add your API key in the Settings panel.")
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
            tabs = st.tabs(["Recommendations", "Review Analysis", "Return Analysis"])
            
            # Tab 1: Recommendations
            with tabs[0]:
                if 'recommendations' in insights['insights']:
                    st.markdown(insights['insights']['recommendations'])
                else:
                    st.info("No AI recommendations available. Run an AI-Enhanced Analysis to generate recommendations.")
            
            # Tab 2: Review Analysis
            with tabs[1]:
                if 'review_insights' in insights['insights'] and insights['insights']['review_insights']:
                    review_insights = insights['insights']['review_insights']
                    
                    for i, insight in enumerate(review_insights):
                        with st.expander(f"Review {i+1}: {insight['review'][:50]}... (Rating: {insight['rating']})"):
                            st.subheader("Original Review")
                            st.write(insight['review'])
                            
                            st.subheader("Sentiment Analysis")
                            st.markdown(insight['sentiment_analysis'])
                            
                            st.subheader("Medical Concerns")
                            st.markdown(insight['medical_concerns'])
                            
                            st.subheader("Compliance Issues")
                            st.markdown(insight['compliance_issues'])
                else:
                    st.info("No AI review insights available. Run an AI-Enhanced Analysis with reviews data to generate insights.")
            
            # Tab 3: Return Analysis
            with tabs[2]:
                if 'return_insights' in insights['insights'] and insights['insights']['return_insights']:
                    return_insights = insights['insights']['return_insights']
                    
                    for i, insight in enumerate(return_insights):
                        with st.expander(f"Return Reason {i+1}: {insight['return_reason'][:50]}..."):
                            st.subheader("Original Return Reason")
                            st.write(insight['return_reason'])
                            
                            st.subheader("AI Analysis")
                            st.markdown(insight['analysis'])
                else:
                    st.info("No AI return insights available. Run an AI-Enhanced Analysis with return reasons data to generate insights.")
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
    
    help_tabs = st.tabs(["User Guide", "FAQ", "Medical Device Context", "Support"])
    
    with help_tabs[0]:
        st.subheader("Getting Started")
        st.markdown("""
        1. **Import Data**: Choose one of these methods:
           - Upload a CSV or Excel file with product data
           - Manually enter product details in the "Manual Entry" tab
           - Upload PDF reports or screenshots for OCR processing
        
        2. **Select Product**: Choose the product you want to analyze.
        
        3. **Run Analysis**: You have two options:
           - Standard Analysis: Basic metrics and visualizations
           - AI-Enhanced Analysis: Deep insights using AI (requires API key)
        
        4. **Review Results**: Explore the analysis, recommendations, and visualizations:
           - Summary: Quick overview of key metrics
           - Detailed Analysis: In-depth breakdown of reviews and returns
           - Visualizations: Charts and graphs of the data
           - Export: Download your analysis results
        
        5. **AI Insights**: Review AI-powered analysis:
           - Recommendations: Product improvement suggestions
           - Review Analysis: Detailed breakdown of individual reviews
           - Return Analysis: Root cause analysis of return reasons
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
          * FDA Device Class
          * Prescription Status
          * Product Description
          * Star Rating
          * Total Reviews
          * Last 365 Days Sales
          * Last 365 Days Returns
        
        You can download a sample template from the Import tab.
        """)
    
    with help_tabs[1]:
        st.subheader("Frequently Asked Questions")
        
        with st.expander("How does the AI analysis work?"):
            st.markdown("""
            The AI-Enhanced Analysis leverages advanced natural language processing to:
            
            1. **Analyze Reviews**: Identifies sentiment, medical concerns, and compliance issues
            2. **Analyze Return Reasons**: Determines root causes of returns
            3. **Generate Recommendations**: Provides actionable insights for product improvement
            
            The AI is specifically trained to understand medical device terminology and regulatory context,
            making it ideal for analyzing medical device products sold on Amazon.
            
            To use this feature, you need to provide an OpenAI API key in the Settings panel.
            """)
        
        with st.expander("What is FDA Device Classification?"):
            st.markdown("""
            The FDA classifies medical devices into three categories based on risk:
            
            * **Class I**: Low risk devices (e.g., bandages, handheld surgical instruments)
               - Subject to general controls
               - Usually exempt from premarket notification (510(k))
            
            * **Class II**: Moderate risk devices (e.g., powered wheelchairs, infusion pumps)
               - Subject to special controls
               - Usually require premarket notification (510(k))
            
            * **Class III**: High risk devices (e.g., implantable pacemakers)
               - Subject to premarket approval (PMA)
               - Require clinical trials
            
            The AI can help classify your product if you're unsure of its FDA designation.
            """)
        
        with st.expander("How can I improve the quality of my analysis?"):
            st.markdown("""
            To get the most comprehensive analysis:
            
            1. **Provide complete data**: Include as many optional fields as possible
            2. **Add detailed reviews**: The more review text, the better the AI analysis
            3. **Include return reasons**: Specific reasons help identify quality issues
            4. **Add product descriptions**: Helps with device classification and context
            5. **Use historical data**: Enables trend analysis and pattern detection
            
            The AI analysis improves with more context about your product.
            """)
        
        with st.expander("How should I use the recommendations?"):
            st.markdown("""
            The AI recommendations are designed to:
            
            1. **Identify quality issues**: Potential defects or design flaws
            2. **Highlight regulatory concerns**: Compliance risks that need attention
            3. **Suggest improvements**: Product enhancements based on customer feedback
            4. **Prioritize actions**: Most impactful changes to reduce returns
            
            Use these insights for:
            - Product development planning
            - Quality improvement initiatives
            - Regulatory compliance checks
            - Customer communication strategies
            
            Always validate AI recommendations with your quality team before implementation.
            """)
    
    with help_tabs[2]:
        st.subheader("Medical Device Context")
        
        st.markdown("""
        ### Medical Device Regulations
        
        Products sold as medical devices, even on Amazon, are subject to various regulations:
        
        * **FDA Regulations**: Class I, II, and III classifications with different requirements
        * **Quality System Regulation (QSR)**: Requirements for design, manufacturing, packaging, etc.
        * **Medical Device Reporting (MDR)**: Mandatory reporting of adverse events
        * **Labeling Requirements**: Specific content and format for labels and instructions
        
        ### Common Medical Device Issues
        
        Medical devices sold on Amazon often face these specific challenges:
        
        * **User Expectations**: Consumers may expect medical-grade performance from wellness products
        * **Regulatory Grey Areas**: Some products exist between medical device and consumer product categories
        * **Return Rate Considerations**: Medical devices typically have higher acceptable return rates (3-8%) compared to general consumer products (1-3%)
        * **Special Quality Concerns**: Safety, effectiveness, and durability are especially important
        
        ### Using This Tool for Medical Devices
        
        This tool is specifically designed to help with:
        
        * Identifying potential regulatory compliance issues
        * Detecting safety concerns from customer feedback
        * Understanding quality issues specific to medical devices
        * Benchmarking return rates against medical device standards
        * Generating improvement recommendations with regulatory context
        """)
    
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

# Run the application
if __name__ == "__main__":
    main()
