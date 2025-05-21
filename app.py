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

# Track available modules
AVAILABLE_MODULES = {
    'pandas': False,
    'numpy': False, 
    'plotly': False,
    'pillow': False,
    'requests': False,
    'ocr': False,
    'xlsx_writer': False
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

# Try importing local modules - use safe imports
try:
    import ocr_processor
    import data_analysis
    import import_template
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

# Setup the Streamlit page
st.set_page_config(
    page_title="Product Review Analysis Tool",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main application entry point."""
    # Render header
    st.title("Product Review Analysis Tool")
    st.subheader("Analyze product reviews, ratings, and returns for Vive Health medical devices")
    
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
    
    # Main content tabs
    tabs = st.tabs(["Import", "Analyze", "Results", "Help"])
    
    with tabs[0]:
        render_file_upload()
    
    with tabs[1]:
        render_product_selection()
    
    with tabs[2]:
        render_analysis_results()
    
    with tabs[3]:
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
                file_name="product_analysis_template.xlsx",
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
                    options=["Mobility Aids", "Bathroom Safety", "Pain Relief", "Sleep & Comfort", 
                            "Fitness & Recovery", "Daily Living Aids", "Respiratory Care", "Other"],
                    help="Required field"
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
    
    # Run analysis button
    if st.button("Analyze Product", type="primary"):
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
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    
                    st.success("Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
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
            
            # Display info
            st.subheader(f"Analysis for {product_info['name']} ({product_info['asin']})")
            st.caption(f"Analyzed on: {result['timestamp']}")
            
            # Create tabs for different sections of the analysis
            tabs = st.tabs(["Summary", "Detailed Analysis", "Visualizations", "Export"])
            
            # Tab 1: Summary
            with tabs[0]:
                st.markdown(analysis)
            
            # Tab 2: Detailed Analysis
            with tabs[1]:
                if 'reviews_data' in result and result['reviews_data'] and HAS_LOCAL_MODULES:
                    review_analysis = data_analysis.analyze_reviews(result['reviews_data'])
                    
                    st.subheader("Review Analysis")
                    st.write(f"Total Reviews: {review_analysis['total_reviews']}")
                    
                    if review_analysis['average_rating']:
                        st.write(f"Average Rating: {review_analysis['average_rating']:.1f}")
                    
                    # Display sentiment breakdown
                    if review_analysis['sentiment']:
                        st.subheader("Sentiment Breakdown")
                        sentiment = review_analysis['sentiment']
                        st.write(f"Positive: {sentiment.get('positive', 0)}")
                        st.write(f"Neutral: {sentiment.get('neutral', 0)}")
                        st.write(f"Negative: {sentiment.get('negative', 0)}")
                    
                    # Display common topics
                    if review_analysis['common_topics']:
                        st.subheader("Common Topics")
                        topics = sorted(review_analysis['common_topics'].items(), key=lambda x: x[1], reverse=True)
                        for topic, count in topics[:10]:  # Top 10 topics
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
                        
                        for reason, count in sorted(unique_reasons.items(), key=lambda x: x[1], reverse=True):
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

def render_help_section():
    """Render the help and documentation section."""
    st.header("Help & Documentation")
    
    help_tabs = st.tabs(["User Guide", "FAQ", "Examples", "Support"])
    
    with help_tabs[0]:
        st.subheader("Getting Started")
        st.markdown("""
        1. **Import Data**: Choose one of these methods:
           - Upload a CSV or Excel file with product data
           - Manually enter product details in the "Manual Entry" tab
           - Upload PDF reports or screenshots for OCR processing
        
        2. **Select Product**: Choose the product you want to analyze.
        
        3. **Run Analysis**: Click the "Analyze Product" button to process the data.
        
        4. **Review Results**: Explore the analysis, recommendations, and visualizations.
        
        5. **Export Reports**: Download the analysis in your preferred format.
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
        
        You can download a sample template from the Import tab.
        """)
    
    with help_tabs[1]:
        st.subheader("Frequently Asked Questions")
        
        with st.expander("How do I enter data manually?"):
            st.markdown("""
            1. Go to the "Import" tab and select the "Manual Entry" sub-tab
            2. Fill in the required fields marked with an asterisk (*)
            3. Optionally, add reviews and return reasons in the text areas
            4. Click "Save Product Data" to add the product to your dataset
            5. You can then analyze this product just like imported data
            """)
        
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
            
            Note: OCR functionality requires pytesseract and pdf2image to be installed.
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
    
    with help_tabs[2]:
        st.subheader("Example Data")
        
        st.markdown("### Example 1: CSV Product Data")
        
        example_data = [
            {"SKU": "MOB1116BLU", "ASIN": "B0DT7NW5VY", "Product Name": "Tri-Rollator With Seat", 
             "Category": "Mobility Aids", "Last 30 Days Sales": 491, "Last 30 Days Returns": 10},
            {"SKU": "BAT2234RED", "ASIN": "B0DT8XYZ123", "Product Name": "Vive Shower Chair", 
             "Category": "Bathroom Safety", "Last 30 Days Sales": 325, "Last 30 Days Returns": 8},
            {"SKU": "KOMF352WHT", "ASIN": "B08CK7MN45", "Product Name": "Comfort Cushion", 
             "Category": "Comfort Products", "Last 30 Days Sales": 278, "Last 30 Days Returns": 5}
        ]
        
        if AVAILABLE_MODULES['pandas']:
            st.dataframe(pd.DataFrame(example_data))
        else:
            st.markdown("""
            | SKU | ASIN | Product Name | Category | Last 30 Days Sales | Last 30 Days Returns |
            | --- | ---- | ------------ | -------- | ----------------- | ------------------- |
            | MOB1116BLU | B0DT7NW5VY | Tri-Rollator With Seat | Mobility Aids | 491 | 10 |
            | BAT2234RED | B0DT8XYZ123 | Vive Shower Chair | Bathroom Safety | 325 | 8 |
            | KOMF352WHT | B08CK7MN45 | Comfort Cushion | Comfort Products | 278 | 5 |
            """)
        
        st.markdown("### Example 2: Return Reasons")
        
        example_returns = [
            {"Order ID": "114-1106156-2607429", "Return Reason": "Item defective or doesn't work", "Return Date": "05/21/2025"},
            {"Order ID": "113-1770004-6998632", "Return Reason": "Seat too small", "Return Date": "05/09/2025"},
            {"Order ID": "114-6826075-5417831", "Return Reason": "Bars seem defective", "Return Date": "05/13/2025"}
        ]
        
        if AVAILABLE_MODULES['pandas']:
            st.dataframe(pd.DataFrame(example_returns))
        else:
            st.markdown("""
            | Order ID | Return Reason | Return Date |
            | -------- | ------------- | ----------- |
            | 114-1106156-2607429 | Item defective or doesn't work | 05/21/2025 |
            | 113-1770004-6998632 | Seat too small | 05/09/2025 |
            | 114-6826075-5417831 | Bars seem defective | 05/13/2025 |
            """)
        
        st.markdown("### Example 3: Manual Review Entry")
        
        st.markdown("""
        When entering reviews manually, use the format `Rating - Review Text` with one review per line:
        
        ```
        5 - Love this walker! Very stable and easy to fold.
        3 - Decent product but assembly instructions could be clearer.
        2 - Seat is too small for me, not comfortable for longer sitting periods.
        4 - Good quality but heavier than expected.
        ```
        """)
        
        st.markdown("### Example 4: Manual Return Reasons Entry")
        
        st.markdown("""
        When entering return reasons manually, put one reason per line:
        
        ```
        Item too small for customer
        Defective wheel locking mechanism
        Customer ordered wrong color
        Assembly difficulty
        ```
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
