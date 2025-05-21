import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
import xlsxwriter
import pytesseract
import pdf2image
import io
import os
import json
import re
import logging
import base64
import tempfile
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union
import textwrap

# Import local modules
try:
    import ocr_processor
    import data_analysis
    import import_template
    import ai_analysis
    HAS_LOCAL_MODULES = True
    logging.info("Successfully imported local modules")
except ImportError as e:
    HAS_LOCAL_MODULES = False
    logging.warning(f"Could not import local modules: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SUPPORT_EMAIL = "alexander.popoff@vivehealth.com"
API_TIMEOUT = 30  # Seconds for API timeout
MAX_BATCH_SIZE = 50  # Max items to process in a batch for API
MAX_SINGLE_ANALYSIS = 20  # Max items for single analysis

# Custom color scheme - Cyberpunk theme
COLOR_PRIMARY = "#9945FF"  # Bright purple
COLOR_SECONDARY = "#14F195"  # Neon green
COLOR_DANGER = "#FF5678"  # Neon pink/red
COLOR_WARNING = "#FFD166"  # Amber
COLOR_SUCCESS = "#01FFC3"  # Bright teal
COLOR_INFO = "#14C5F0"  # Bright blue
COLOR_BACKGROUND = "#0D1020"  # Dark blue-black
COLOR_PANEL = "#1A1A2E"  # Slightly lighter blue-black
COLOR_TEXT = "#E0E0FF"  # Light lavender

# --- API KEY HANDLING ---
try:
    api_key = st.secrets["openai_api_key"]
    logger.info("API key loaded from Streamlit secrets")
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        logger.info("API key loaded from environment variables")
    else:
        logger.warning("No OpenAI API key found")

# Track available modules
AVAILABLE_MODULES = {
    'pandas': False,
    'numpy': False, 
    'plotly': False,
    'pillow': False,
    'requests': False,
    'ocr': False,
    'xlsx_writer': False,
    'ai_api': False
}

# Check if modules are available
try:
    import pandas as pd
    AVAILABLE_MODULES['pandas'] = True
except ImportError:
    logger.warning("pandas module not available")

try:
    import numpy as np
    AVAILABLE_MODULES['numpy'] = True
except ImportError:
    logger.warning("numpy module not available")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    AVAILABLE_MODULES['plotly'] = True
except ImportError:
    logger.warning("plotly module not available")

try:
    from PIL import Image
    AVAILABLE_MODULES['pillow'] = True
except ImportError:
    logger.warning("PIL module not available")

try:
    import requests
    AVAILABLE_MODULES['requests'] = True
except ImportError:
    logger.warning("requests module not available")

try:
    import xlsxwriter
    AVAILABLE_MODULES['xlsx_writer'] = True
except ImportError:
    logger.warning("xlsxwriter module not available")

try:
    import pytesseract
    import pdf2image
    AVAILABLE_MODULES['ocr'] = True
except ImportError:
    logger.warning("OCR modules not available")

# Check if API is available (based on API key and requests module)
if api_key and AVAILABLE_MODULES['requests']:
    AVAILABLE_MODULES['ai_api'] = True
    logger.info("AI API module is available")
else:
    logger.warning("AI API is not available - missing API key or requests module")

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

# Return reason categories
RETURN_CATEGORIES = {
    "Size/Fit Issues": ["too small", "too big", "wrong size", "not fit", "sizing", "doesn't fit", "doesn't measure", "fit issue", "size issue"],
    "Quality/Durability": ["broke", "broken", "poor quality", "cheaply made", "fell apart", "stop working", "not durable", "damaged", "defective", "stopped working", "quality issue", "malfunctioning", "not working"],
    "Comfort Problems": ["uncomfortable", "not comfortable", "painful", "hurts", "irritates", "comfort issue", "discomfort", "pain", "hard to wear", "difficult to wear"],
    "Expectation Mismatch": ["not as described", "not as expected", "different", "misleading", "expected more", "not as advertised", "not what I expected", "thought it would", "expected it to"],
    "Listing Accuracy": ["not as shown", "picture is wrong", "looks different", "image", "pictured", "photo", "picture showed", "showed different", "advertised different"],
    "Packaging/Delivery": ["arrived damaged", "bad packaging", "shipping issue", "packaging problem", "damaged in transit", "delivery issue", "opened package"],
    "Usage Difficulty": ["hard to use", "difficult to", "complicated", "confusing", "user friendly", "instruction", "challenging to use", "couldn't figure out", "not easy to use"],
    "Missing Parts": ["missing", "incomplete", "parts missing", "not included", "didn't include", "without", "lacking"],
    "Medical Effectiveness": ["didn't help", "no relief", "not effective", "doesn't work", "didn't work", "not working", "ineffective", "no improvement", "didn't provide"],
    "Better Alternative": ["found better", "alternative", "different product", "another option", "another brand", "better one", "upgrading to"],
    "Other": []  # Catch-all for anything not matching above
}

# Review theme categories
REVIEW_CATEGORIES = {
    "Ease of Use": ["easy to use", "simple to", "user friendly", "intuitive", "straightforward", "convenient", "easy setup", "operates easily"],
    "Comfort": ["comfortable", "soft", "cushioned", "padding", "gentle", "snug", "cozy", "no pain", "doesn't hurt", "comfortable to wear"],
    "Quality/Durability": ["well made", "durable", "quality", "sturdy", "solid", "built to last", "robust", "strong", "premium", "high-quality"],
    "Pain Relief": ["pain relief", "helped pain", "reduced pain", "alleviated pain", "pain free", "pain reduction", "relief", "less pain"],
    "Size/Fit": ["fit", "size", "adjustment", "adjustable", "fits well", "sizing", "too small", "too large", "perfect fit", "measurements"],
    "Value": ["worth the money", "good value", "price", "affordable", "expensive", "cost", "worth every penny", "bargain", "overpriced"],
    "Medical Effectiveness": ["works well", "effective", "helps", "improvement", "better mobility", "recovery", "healing", "therapeutic", "treatment"],
    "Support/Stability": ["support", "stability", "secure", "stays in place", "doesn't slip", "holds", "stabilizing", "maintains position"],
    "Materials": ["material", "fabric", "breathable", "latex", "neoprene", "cotton", "synthetic", "plastic", "metal", "silicone"],
    "Instructions": ["instructions", "manual", "directions", "guide", "tutorial", "confusing instructions", "clear instructions"],
    "Recommendation": ["recommend", "suggested", "doctor recommended", "physician", "therapist", "healthcare provider", "professional"],
    "Other": []
}

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

# Initialize session state variables
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
    
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'current_product' not in st.session_state:
    st.session_state.current_product = None

if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = {}

if 'ai_api_calls' not in st.session_state:
    st.session_state.ai_api_calls = 0

if 'api_call_in_progress' not in st.session_state:
    st.session_state.api_call_in_progress = False

# Set up custom Streamlit theme with cyberpunk aesthetics
def set_cyberpunk_theme():
    # CSS for cyberpunk/dark theme
    st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: """ + COLOR_BACKGROUND + """;
            color: """ + COLOR_TEXT + """;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: """ + COLOR_PRIMARY + """;
            text-shadow: 0 0 10px rgba(153, 69, 255, 0.5);
            letter-spacing: 1px;
        }
        
        /* Text and Links */
        p, ol, ul, dl {
            color: """ + COLOR_TEXT + """;
        }
        
        a {
            color: """ + COLOR_SECONDARY + """;
            text-decoration: none;
        }
        
        a:hover {
            color: """ + COLOR_PRIMARY + """;
            text-decoration: underline;
        }
        
        /* Sidebar */
        .css-1d391kg, .css-12oz5g7 {
            background-color: """ + COLOR_PANEL + """;
        }
        
        /* Inputs, buttons, sliders */
        .stButton>button {
            border: 2px solid """ + COLOR_PRIMARY + """;
            border-radius: 4px;
            color: """ + COLOR_TEXT + """;
            background-color: rgba(153, 69, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: """ + COLOR_PRIMARY + """;
            color: white;
            box-shadow: 0 0 15px """ + COLOR_PRIMARY + """;
        }
        
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: rgba(26, 26, 46, 0.8);
            color: """ + COLOR_TEXT + """;
            border: 1px solid rgba(153, 69, 255, 0.5);
            border-radius: 4px;
        }
        
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
            border-color: """ + COLOR_PRIMARY + """;
            box-shadow: 0 0 10px rgba(153, 69, 255, 0.5);
        }
        
        /* Tables */
        .stDataFrame {
            background-color: """ + COLOR_PANEL + """;
        }
        
        .stDataFrame [data-testid="stTable"] {
            background-color: rgba(26, 26, 46, 0.8);
            border: 1px solid """ + COLOR_PRIMARY + """;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            color: """ + COLOR_SECONDARY + """;
            text-shadow: 0 0 10px rgba(20, 241, 149, 0.5);
        }
        
        [data-testid="stMetricDelta"] {
            background-color: transparent;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(26, 26, 46, 0.8);
            border: 1px solid rgba(153, 69, 255, 0.3);
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            margin-right: 2px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(153, 69, 255, 0.2);
            border-color: """ + COLOR_PRIMARY + """;
            border-bottom: 2px solid """ + COLOR_PRIMARY + """;
        }
        
        /* Cards and containers */
        [data-testid="stExpander"] {
            background-color: """ + COLOR_PANEL + """;
            border: 1px solid rgba(153, 69, 255, 0.3);
            border-radius: 4px;
        }
        
        /* Success/Warning/Error messages */
        .element-container div[data-stale="false"] div[data-baseweb="notification"] {
            margin: 0;
            border-radius: 5px;
        }
        
        /* Neon glow effects for important elements */
        .neon-glow {
            text-shadow: 0 0 10px rgba(153, 69, 255, 0.8);
        }
        
        .neon-box {
            box-shadow: 0 0 10px rgba(153, 69, 255, 0.5);
        }
        
        /* Custom spinner */
        .custom-spinner {
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 3px solid rgba(20, 241, 149, 0.3);
            border-radius: 50%;
            border-top-color: """ + COLOR_SECONDARY + """;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to {transform: rotate(360deg);}
        }
        
        /* Custom horizontal rule */
        .custom-hr {
            border: 0;
            height: 1px;
            background-image: linear-gradient(to right, rgba(153, 69, 255, 0), rgba(153, 69, 255, 0.75), rgba(153, 69, 255, 0));
            margin: 20px 0;
        }
        
        /* Improve contrast for text on dark background */
        .streamlit-expanderHeader, .stMarkdown, p {
            color: """ + COLOR_TEXT + """ !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def safe_divide(a, b, default=0):
    """Safely divide a by b, returning default if b is zero."""
    return (a / b) if b != 0 else default

def wrap_text(text, width=50):
    """Wrap text to specified width."""
    return '<br>'.join(textwrap.wrap(text, width=width))

def format_currency(amount):
    """Format amount as currency."""
    return f"${amount:,.2f}"

def format_percent(value, decimals=2):
    """Format value as percentage."""
    return f"{value:.{decimals}f}%"

def display_colored_metric(label, value, delta=None, delta_color="normal", help_text=None, style="default"):
    """Display a metric with cyberpunk styling."""
    if style == "money":
        value_text = format_currency(value)
    elif style == "percent":
        value_text = format_percent(value)
    else:
        value_text = value
    
    st.metric(label=label, value=value_text, delta=delta, delta_color=delta_color, help=help_text)

def generate_excel_download_link(df, filename):
    """Generate a download link for Excel file."""
    if not AVAILABLE_MODULES['xlsx_writer']:
        return None

    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Data']
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#9945FF',
                'font_color': 'white',
                'border': 1
            })
            
            # Write the column headers with the defined format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Auto-adjust columns' width
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
                worksheet.set_column(i, i, max_len)
        
        # Reset pointer to beginning and read the data
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-link">Download {filename}</a>'
        return href
    except Exception as e:
        logger.error(f"Error generating Excel download link: {str(e)}")
        return None

def categorize_text(text, categories_dict):
    """
    Categorize a text string based on keyword matches in the categories dictionary.
    
    Args:
        text: The text to categorize
        categories_dict: Dictionary of category names to lists of keywords
        
    Returns:
        The best matching category name
    """
    if not text:
        return "Other"
        
    text = text.lower()
    matches = {}
    
    # Count matches for each category
    for category, keywords in categories_dict.items():
        count = 0
        for keyword in keywords:
            if keyword.lower() in text:
                count += 1
        matches[category] = count
    
    # Find best match or return "Other"
    best_match = max(matches.items(), key=lambda x: x[1]) if matches else ("Other", 0)
    return best_match[0] if best_match[1] > 0 else "Other"

def categorize_returns(returns_data):
    """
    Categorize return reasons into predefined categories.
    
    Args:
        returns_data: List of dictionaries with return reasons
        
    Returns:
        Dictionary with categories and their counts, and categorized returns
    """
    categories = defaultdict(list)
    
    for return_item in returns_data:
        reason = return_item.get('return_reason', '')
        if reason:
            category = categorize_text(reason, RETURN_CATEGORIES)
            categories[category].append(return_item)
    
    # If no items were categorized, add to Other
    if not categories and returns_data:
        categories["Other"] = returns_data
        
    # Count items in each category
    category_counts = {cat: len(items) for cat, items in categories.items()}
    
    return {
        "categories": categories,
        "counts": category_counts
    }

def categorize_reviews(reviews_data):
    """
    Categorize reviews into predefined theme categories.
    
    Args:
        reviews_data: List of dictionaries with review text
        
    Returns:
        Dictionary with categories, their counts, and sentiment scores
    """
    categories = defaultdict(list)
    
    for review in reviews_data:
        text = review.get('review_text', '')
        rating = review.get('rating', 3)
        
        if text:
            category = categorize_text(text, REVIEW_CATEGORIES)
            categories[category].append(review)
    
    # If no items were categorized, add to Other
    if not categories and reviews_data:
        categories["Other"] = reviews_data
    
    # Calculate sentiment scores (1-5) for each category
    category_sentiment = {}
    for cat, reviews in categories.items():
        ratings = [r.get('rating', 3) for r in reviews if r.get('rating') is not None]
        if ratings:
            category_sentiment[cat] = sum(ratings) / len(ratings)
        else:
            category_sentiment[cat] = 3  # Neutral default
    
    # Count items in each category
    category_counts = {cat: len(items) for cat, items in categories.items()}
    
    return {
        "categories": categories,
        "counts": category_counts,
        "sentiment": category_sentiment
    }

def summarize_reviews_for_api(reviews_data, max_items=MAX_SINGLE_ANALYSIS):
    """
    Summarize reviews data for API analysis to reduce token count.
    
    Args:
        reviews_data: List of review dictionaries
        max_items: Maximum number of full reviews to include
    
    Returns:
        Summarized reviews text for API
    """
    if not reviews_data:
        return ""
    
    # Group reviews by rating
    reviews_by_rating = defaultdict(list)
    for review in reviews_data:
        rating = review.get('rating', 3)
        reviews_by_rating[rating].append(review)
    
    # Generate summary text
    summary = f"SUMMARY OF {len(reviews_data)} REVIEWS:\n\n"
    
    # Add distribution by rating
    summary += "Rating distribution:\n"
    for rating in sorted(reviews_by_rating.keys(), reverse=True):
        count = len(reviews_by_rating[rating])
        summary += f"{rating}-star: {count} reviews ({count/len(reviews_data)*100:.1f}%)\n"
    
    summary += "\n"
    
    # Add sample reviews (most useful ones - prioritize detailed high and low ratings)
    # Sort reviews by length (longer reviews often have more detail)
    all_sorted_reviews = []
    for rating in [1, 2, 5, 4, 3]:  # Prioritize extreme ratings
        if rating in reviews_by_rating:
            # Sort by length of review text
            sorted_reviews = sorted(reviews_by_rating[rating], 
                                    key=lambda x: len(x.get('review_text', '')), 
                                    reverse=True)
            all_sorted_reviews.extend(sorted_reviews)
    
    # Take the top N most detailed reviews
    sample_reviews = all_sorted_reviews[:max_items]
    
    if sample_reviews:
        summary += f"SAMPLE REVIEWS (showing {len(sample_reviews)} most detailed reviews):\n\n"
        for i, review in enumerate(sample_reviews, 1):
            rating = review.get('rating', 'Unknown')
            text = review.get('review_text', 'No text')
            summary += f"{i}. ({rating}-star) {text}\n\n"
    
    # Add categorized themes
    categorized = categorize_reviews(reviews_data)
    categories = categorized["categories"]
    
    summary += "COMMON THEMES IN REVIEWS:\n"
    for category, reviews in categories.items():
        if len(reviews) > 1:  # Only include categories with multiple reviews
            avg_rating = sum(r.get('rating', 3) for r in reviews) / len(reviews)
            sample_text = reviews[0].get('review_text', 'No example')[:100] + "..."
            summary += f"- {category}: {len(reviews)} mentions, avg {avg_rating:.1f} stars.\n"
            summary += f"  Example: \"{sample_text}\"\n\n"
    
    return summary

def summarize_returns_for_api(returns_data, max_items=MAX_SINGLE_ANALYSIS):
    """
    Summarize return reasons for API analysis to reduce token count.
    
    Args:
        returns_data: List of return reason dictionaries
        max_items: Maximum number of full return reasons to include
    
    Returns:
        Summarized return reasons text for API
    """
    if not returns_data:
        return ""
    
    # Generate summary text
    summary = f"SUMMARY OF {len(returns_data)} RETURN REASONS:\n\n"
    
    # Add categorized return reasons
    categorized = categorize_returns(returns_data)
    categories = categorized["categories"]
    counts = categorized["counts"]
    
    summary += "CATEGORIES OF RETURN REASONS:\n"
    for category, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percent = (count / len(returns_data)) * 100
            summary += f"- {category}: {count} returns ({percent:.1f}%)\n"
            
            # Add examples from this category (max 2 examples per category)
            returns_in_category = categories[category][:2]
            for i, return_item in enumerate(returns_in_category, 1):
                reason = return_item.get('return_reason', 'No reason provided')
                summary += f"  {i}. \"{reason}\"\n"
            summary += "\n"
    
    # Add all return reasons (limited to max_items)
    if len(returns_data) > 0:
        summary += f"ALL RETURN REASONS (showing up to {max_items}):\n\n"
        for i, return_item in enumerate(returns_data[:max_items], 1):
            reason = return_item.get('return_reason', 'No reason provided')
            summary += f"{i}. {reason}\n"
    
    return summary

def batch_process_with_ai(items, process_func, batch_size=MAX_BATCH_SIZE, **kwargs):
    """
    Process a large number of items with AI in batches.
    
    Args:
        items: List of items to process
        process_func: Function to process each batch
        batch_size: Size of each batch
        **kwargs: Additional arguments for process_func
        
    Returns:
        Combined results from all batches
    """
    results = []
    total_items = len(items)
    
    # Process in batches
    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_result = process_func(batch, **kwargs)
        if batch_result.get("success", False):
            results.append(batch_result.get("result", ""))
    
    # Combine results
    return " ".join(results)

def create_download_link(content, file_name):
    """Create a download link for file content."""
    b64 = base64.b64encode(content.encode()).decode() 
    href = f'<a href="data:text/plain;base64,{b64}" download="{file_name}">{file_name}</a>'
    return href

# --- OPENAI API FUNCTIONS ---
def call_openai_api(messages, temperature=0.7, max_tokens=1024, model="gpt-4o", image_data=None):
    """
    Call OpenAI API with the provided messages.
    
    Args:
        messages: List of message dictionaries
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens for generation
        model: Model to use
        image_data: Optional image data to analyze with vision capabilities
        
    Returns:
        Dictionary with results or error
    """
    if not api_key:
        logger.error("API Key not found. Please set it up in your environment.")
        return {"success": False, "error": "API Key not configured."}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # If we have image data, add it to the message content
        if image_data and isinstance(messages, list) and len(messages) > 0:
            # Use GPT-4 Vision model
            vision_model = "gpt-4-vision-preview"
            
            # Encode the image data
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode('utf-8')
            else:
                # If it's already a base64 string
                base64_image = image_data
            
            # Create the message with image content
            content = [
                {"type": "text", "text": messages[-1].get("content", "Analyze this image")}
            ]
            
            # Add the image data
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"  # Request high detail analysis
                }
            })
            
            # Replace the content of the last message
            messages[-1]["content"] = content
            
            # Use the vision model
            model = vision_model
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        st.session_state.api_call_in_progress = True
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=API_TIMEOUT
        )
        
        # Increment API call counter
        if 'ai_api_calls' in st.session_state:
            st.session_state.ai_api_calls += 1
        
        st.session_state.api_call_in_progress = False
        
        # If successful
        if response.status_code == 200:
            return {
                "success": True,
                "result": response.json()["choices"][0]["message"]["content"]
            }
        else:
            return {
                "success": False,
                "error": f"Error: {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.Timeout:
        st.session_state.api_call_in_progress = False
        logger.error("API call timed out")
        return {"success": False, "error": f"Request timed out after {API_TIMEOUT} seconds. The server might be experiencing high traffic."}
    except Exception as e:
        st.session_state.api_call_in_progress = False
        logger.error(f"API call error: {str(e)}")
        return {"success": False, "error": f"Error: {str(e)}"}

def test_openai_api():
    """Test if the OpenAI API key works with a minimal request."""
    if not api_key or not AVAILABLE_MODULES['requests']:
        return False
    
    messages = [{"role": "user", "content": "Hello, this is a test."}]
    
    try:
        result = call_openai_api(
            messages=messages,
            temperature=0.5,
            max_tokens=10
        )
        
        return result.get("success", False)
    except Exception as e:
        logger.error(f"Error testing OpenAI API: {str(e)}")
        return False

def analyze_with_ai(text, analysis_type='sentiment'):
    """
    Analyze text using OpenAI API
    
    Args:
        text: The text to analyze
        analysis_type: Type of analysis to perform
    
    Returns:
        Dictionary with analysis results
    """
    try:
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
        elif analysis_type == 'summarize_reviews':
            prompt = f"""You're an expert at analyzing Amazon customer reviews for medical devices.
            Analyze these customer reviews and summarize the key patterns and insights. 
            Focus on identifying:
            
            1. Common praises and complaints
            2. Product features most frequently mentioned (positively or negatively)
            3. Suggestions for product improvements based on customer feedback
            4. Any concerning quality, safety, or durability issues mentioned
            5. Keywords and terms customers use to describe the product (for listing optimization)
            
            Format your analysis in clear sections with bullet points for easy scanning.
            
            Reviews: {text}
            """
        elif analysis_type == 'summarize_returns':
            prompt = f"""You're an expert at analyzing Amazon product return reasons for medical devices.
            Analyze these product return reasons and summarize the key patterns and insights.
            Focus on identifying:
            
            1. Most common reasons for returns
            2. Product issues that need immediate attention
            3. Confusion points in the product listing that led to returns
            4. Sizing, fit, or comfort issues
            5. Actionable recommendations to reduce the return rate
            
            Format your analysis in clear sections with bullet points for easy scanning.
            
            Return reasons: {text}
            """
        
        messages = [
            {"role": "system", "content": "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."},
            {"role": "user", "content": prompt}
        ]
        
        return call_openai_api(messages, temperature=0.2, max_tokens=750)
    except Exception as e:
        logger.error(f"AI analysis error: {str(e)}")
        return {"success": False, "error": str(e)}

def analyze_image_with_vision(image_data, prompt=None):
    """
    Use OpenAI's Vision capability to analyze image content directly.
    
    Args:
        image_data: Binary image data
        prompt: Optional specific instruction for analysis
        
    Returns:
        Dictionary with analysis results
    """
    if not prompt:
        prompt = """Analyze this image from an Amazon medical device listing.
        If it contains reviews, extract the ratings and review text.
        If it contains product information, summarize the key features and specifications.
        If it contains return reasons, identify the main issues customers are reporting.
        Format the information in a structured way that's easy to read.
        """
    
    messages = [
        {"role": "system", "content": "You are an expert at analyzing Amazon product listings, reviews, and return data for medical devices."},
        {"role": "user", "content": prompt}
    ]
    
    return call_openai_api(messages, temperature=0.2, max_tokens=1500, image_data=image_data)

def process_image_file(file_data, file_name, content_type, asin=None):
    """
    Process an image file using multiple methods.
    Try OCR first, fall back to Vision API if OCR fails.
    
    Args:
        file_data: Binary file data
        file_name: Name of the file
        content_type: Type of content (reviews, listing, etc.)
        asin: Optional ASIN to associate with the file
        
    Returns:
        Dictionary with extracted text and processing info
    """
    result = {
        "filename": file_name,
        "type": file_name.split('.')[-1].lower(),
        "content_type": content_type,
        "asin": asin,
        "text": "",
        "processing_method": None,
        "success": False,
        "error": None
    }
    
    # First try OCR if available
    ocr_success = False
    if AVAILABLE_MODULES['ocr'] and HAS_LOCAL_MODULES:
        try:
            if file_name.lower().endswith('.pdf'):
                result["text"] = ocr_processor.process_pdf_with_ocr(file_data)
            else:  # image file
                result["text"] = ocr_processor.process_image_with_ocr(file_data)
            
            if result["text"] and "Error processing" not in result["text"]:
                result["processing_method"] = "OCR"
                result["success"] = True
                ocr_success = True
            else:
                result["error"] = result["text"] if "Error" in result["text"] else "OCR processing failed"
                ocr_success = False
        except Exception as e:
            result["error"] = f"OCR processing error: {str(e)}"
            ocr_success = False
    
    # Fall back to Vision API if OCR failed or unavailable
    if not ocr_success and AVAILABLE_MODULES['ai_api']:
        try:
            # Prepare prompt based on content type
            prompt = f"This is an image from an Amazon medical device {content_type.lower()}. "
            
            if content_type == "Product Reviews":
                prompt += "Please extract all customer reviews including star ratings and review text. Format as 'Rating: X stars - Review text'."
            elif content_type == "Return Reports":
                prompt += "Please extract all return reasons. Format as a list of return reasons."
            elif content_type == "Product Listing":
                prompt += "Please extract product name, features, specifications, and description."
            elif content_type == "Product Images":
                prompt += "Please describe what this product image shows and any text visible in the image."
            else:
                prompt += "Please extract all relevant text and information from this image."
            
            vision_result = analyze_image_with_vision(file_data, prompt)
            
            if vision_result["success"]:
                result["text"] = vision_result["result"]
                result["processing_method"] = "Vision API"
                result["success"] = True
            else:
                if not result["error"]:  # Only update if we don't already have an OCR error
                    result["error"] = vision_result.get("error", "Vision API processing failed")
        except Exception as e:
            if not result["error"]:  # Only update if we don't already have an OCR error
                result["error"] = f"Vision API processing error: {str(e)}"
    
    # If all processing failed
    if not result["success"]:
        if not result["error"]:
            result["error"] = "All processing methods failed"
        result["text"] = f"Could not extract text: {result['error']}"
    
    return result

def analyze_listing_optimization(product_info):
    """
    Use AI to analyze and provide recommendations for Amazon listing optimization
    
    Args:
        product_info: Dictionary with product details
    
    Returns:
        Dictionary with optimization recommendations
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
        
        return call_openai_api(messages, temperature=0.2, max_tokens=1000)
    except Exception as e:
        logger.error(f"Listing optimization error: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_improvement_recommendations(product_info, reviews_data, returns_data):
    """
    Generate AI-powered recommendations for product improvements based on reviews and returns
    
    Args:
        product_info: Dictionary with product details
        reviews_data: List of product reviews
        returns_data: List of return reasons
    
    Returns:
        Dictionary with recommendations
    """
    try:
        # Prepare the data for analysis - use summaries for larger datasets
        if len(reviews_data) > MAX_SINGLE_ANALYSIS:
            reviews_text = summarize_reviews_for_api(reviews_data)
        else:
            reviews_text = "\n".join([f"Rating: {r.get('rating', 'N/A')} - {r.get('review_text', '')}" 
                                      for r in reviews_data[:MAX_SINGLE_ANALYSIS]])
        
        if len(returns_data) > MAX_SINGLE_ANALYSIS:
            returns_text = summarize_returns_for_api(returns_data)
        else:
            returns_text = "\n".join([f"Return reason: {r.get('return_reason', '')}" 
                                      for r in returns_data[:MAX_SINGLE_ANALYSIS]])
        
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
        
        return call_openai_api(messages, temperature=0.2, max_tokens=1200)
    except Exception as e:
        logger.error(f"Recommendation generation error: {str(e)}")
        return {"success": False, "error": str(e)}

def process_reviews_with_ai(reviews_data):
    """
    Process reviews with AI analysis, handling large volumes through summarization.
    
    Args:
        reviews_data: List of review dictionaries
    
    Returns:
        AI analysis results
    """
    try:
        # Handle empty reviews
        if not reviews_data:
            return {"success": True, "result": "No reviews available for analysis."}
        
        # For larger datasets, use summary approach
        if len(reviews_data) > MAX_SINGLE_ANALYSIS:
            reviews_summary = summarize_reviews_for_api(reviews_data)
            result = analyze_with_ai(reviews_summary, 'summarize_reviews')
            return result
        else:
            # For smaller datasets, analyze each review individually
            review_insights = []
            for review in reviews_data:
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
            
            # Compile results
            result_text = "## Review Analysis\n\n"
            for i, insight in enumerate(review_insights, 1):
                result_text += f"### Review {i} (Rating: {insight['rating']})\n"
                result_text += f"**Text:** {insight['review']}\n\n"
                result_text += f"**Analysis:** {insight['analysis']}\n\n"
                result_text += "---\n\n"
            
            return {"success": True, "result": result_text}
    
    except Exception as e:
        logger.error(f"Error processing reviews with AI: {str(e)}")
        return {"success": False, "error": str(e)}

def process_returns_with_ai(returns_data):
    """
    Process return reasons with AI analysis, handling large volumes through summarization.
    
    Args:
        returns_data: List of return reason dictionaries
    
    Returns:
        AI analysis results
    """
    try:
        # Handle empty returns
        if not returns_data:
            return {"success": True, "result": "No return reasons available for analysis."}
        
        # For larger datasets, use summary approach
        if len(returns_data) > MAX_SINGLE_ANALYSIS:
            returns_summary = summarize_returns_for_api(returns_data)
            result = analyze_with_ai(returns_summary, 'summarize_returns')
            return result
        else:
            # For smaller datasets, analyze each return reason individually
            return_insights = []
            for return_data in returns_data:
                return_reason = return_data.get('return_reason', '')
                if return_reason:
                    analysis = analyze_with_ai(return_reason, 'return_analysis')
                    if analysis.get('success', False):
                        return_insights.append({
                            'return_reason': return_reason,
                            'analysis': analysis.get('result', 'Analysis failed')
                        })
            
            # Compile results
            result_text = "## Return Reason Analysis\n\n"
            for i, insight in enumerate(return_insights, 1):
                result_text += f"### Return Reason {i}\n"
                result_text += f"**Text:** {insight['return_reason']}\n\n"
                result_text += f"**Analysis:** {insight['analysis']}\n\n"
                result_text += "---\n\n"
            
            return {"success": True, "result": result_text}
    
    except Exception as e:
        logger.error(f"Error processing returns with AI: {str(e)}")
        return {"success": False, "error": str(e)}

# --- DATA EXPORT FUNCTIONS ---

def export_analysis_to_excel(product_info, analysis_results):
    """
    Export analysis results to Excel.
    
    Args:
        product_info: Dictionary with product details
        analysis_results: Dictionary with analysis results
        
    Returns:
        Excel file as BytesIO object
    """
    if not AVAILABLE_MODULES['xlsx_writer'] or not AVAILABLE_MODULES['pandas']:
        return None
    
    try:
        # Create a BytesIO object to store the Excel file
        output = io.BytesIO()
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create Product Overview sheet
            product_df = pd.DataFrame({
                'Metric': [
                    'ASIN', 'SKU', 'Product Name', 'Category',
                    '30-Day Sales', '30-Day Returns', '30-Day Return Rate', 
                    '365-Day Sales', '365-Day Returns', '365-Day Return Rate',
                    'Star Rating', 'Total Reviews'
                ],
                'Value': [
                    product_info.get('asin', 'N/A'),
                    product_info.get('sku', 'N/A'),
                    product_info.get('name', 'N/A'),
                    product_info.get('category', 'N/A'),
                    product_info.get('sales_30d', 'N/A'),
                    product_info.get('returns_30d', 'N/A'),
                    f"{product_info.get('return_rate_30d', 'N/A')}%",
                    product_info.get('sales_365d', 'N/A'),
                    product_info.get('returns_365d', 'N/A'),
                    f"{product_info.get('return_rate_365d', 'N/A')}%" if 'return_rate_365d' in product_info else 'N/A',
                    product_info.get('star_rating', 'N/A'),
                    product_info.get('total_reviews', 'N/A')
                ]
            })
            
            product_df.to_excel(writer, sheet_name='Product Overview', index=False)
            
            # Get the workbook and worksheet objects
            workbook = writer.book
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#9945FF',
                'font_color': 'white',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            
            # Format Product Overview sheet
            worksheet = writer.sheets['Product Overview']
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 40)
            worksheet.write('A1', 'Metric', header_format)
            worksheet.write('B1', 'Value', header_format)
            
            # Export reviews if available
            reviews_data = analysis_results.get('reviews_data', [])
            if reviews_data:
                # Create dataframe for reviews
                reviews_df = pd.DataFrame([
                    {
                        'Rating': r.get('rating', 'N/A'),
                        'Review Text': r.get('review_text', 'N/A'),
                        'Category': categorize_text(r.get('review_text', ''), REVIEW_CATEGORIES)
                    } for r in reviews_data
                ])
                
                reviews_df.to_excel(writer, sheet_name='Reviews', index=False)
                
                # Format Reviews sheet
                worksheet = writer.sheets['Reviews']
                worksheet.set_column('A:A', 10)
                worksheet.set_column('B:B', 80, cell_format)
                worksheet.set_column('C:C', 20)
                worksheet.write('A1', 'Rating', header_format)
                worksheet.write('B1', 'Review Text', header_format)
                worksheet.write('C1', 'Category', header_format)
                
                # Create Review Categories sheet with counts
                categorized = categorize_reviews(reviews_data)
                categories_counts = categorized["counts"]
                sentiment_scores = categorized["sentiment"]
                
                categories_df = pd.DataFrame({
                    'Category': list(categories_counts.keys()),
                    'Count': list(categories_counts.values()),
                    'Avg Rating': [sentiment_scores.get(cat, 'N/A') for cat in categories_counts.keys()]
                })
                
                categories_df.to_excel(writer, sheet_name='Review Categories', index=False)
                
                # Format Review Categories sheet
                worksheet = writer.sheets['Review Categories']
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 15)
                worksheet.set_column('C:C', 15)
                worksheet.write('A1', 'Category', header_format)
                worksheet.write('B1', 'Count', header_format)
                worksheet.write('C1', 'Avg Rating', header_format)
            
            # Export return reasons if available
            returns_data = analysis_results.get('return_reasons_data', [])
            if returns_data:
                # Create dataframe for return reasons
                returns_df = pd.DataFrame([
                    {
                        'Return Reason': r.get('return_reason', 'N/A'),
                        'Category': categorize_text(r.get('return_reason', ''), RETURN_CATEGORIES)
                    } for r in returns_data
                ])
                
                returns_df.to_excel(writer, sheet_name='Return Reasons', index=False)
                
                # Format Return Reasons sheet
                worksheet = writer.sheets['Return Reasons']
                worksheet.set_column('A:A', 80, cell_format)
                worksheet.set_column('B:B', 20)
                worksheet.write('A1', 'Return Reason', header_format)
                worksheet.write('B1', 'Category', header_format)
                
                # Create Return Categories sheet with counts
                categorized = categorize_returns(returns_data)
                categories_counts = categorized["counts"]
                
                categories_df = pd.DataFrame({
                    'Category': list(categories_counts.keys()),
                    'Count': list(categories_counts.values()),
                    'Percentage': [f"{(count / len(returns_data) * 100):.1f}%" for count in categories_counts.values()]
                })
                
                categories_df.to_excel(writer, sheet_name='Return Categories', index=False)
                
                # Format Return Categories sheet
                worksheet = writer.sheets['Return Categories']
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 15)
                worksheet.set_column('C:C', 15)
                worksheet.write('A1', 'Category', header_format)
                worksheet.write('B1', 'Count', header_format)
                worksheet.write('C1', 'Percentage', header_format)
            
            # Create AI Insights sheet if available
            if product_info.get('asin') in st.session_state.ai_insights:
                insights = st.session_state.ai_insights[product_info.get('asin')]
                insights_sheet = writer.book.add_worksheet('AI Insights')
                
                # Set column widths
                insights_sheet.set_column('A:A', 30)
                insights_sheet.set_column('B:B', 100, cell_format)
                
                # Write headers
                insights_sheet.write('A1', 'Analysis Type', header_format)
                insights_sheet.write('B1', 'Results', header_format)
                
                row = 1
                # Add each type of insight
                insight_types = {
                    'listing_optimization': 'Listing Optimization',
                    'recommendations': 'Product Recommendations',
                    'review_insights': 'Review Analysis',
                    'return_insights': 'Return Analysis'
                }
                
                for key, label in insight_types.items():
                    if key in insights['insights']:
                        data = insights['insights'][key]
                        
                        if isinstance(data, str):
                            insights_sheet.write(row, 0, label)
                            insights_sheet.write(row, 1, data)
                            row += 1
                        elif isinstance(data, list) and data:
                            insights_sheet.write(row, 0, label)
                            insights_sheet.write(row, 1, f"Found {len(data)} items")
                            row += 1
                            
                            if key == 'review_insights':
                                # Add summary for review insights
                                for i, item in enumerate(data[:5]):  # Limit to first 5
                                    insights_sheet.write(row, 0, f"Review {i+1}")
                                    insights_sheet.write(row, 1, f"Rating: {item.get('rating', 'N/A')}, Text: {item.get('review', 'N/A')[:100]}...")
                                    row += 1
                            
                            elif key == 'return_insights':
                                # Add summary for return insights
                                for i, item in enumerate(data[:5]):  # Limit to first 5
                                    insights_sheet.write(row, 0, f"Return Reason {i+1}")
                                    insights_sheet.write(row, 1, item.get('return_reason', 'N/A')[:100])
                                    row += 1
        
        # Reset buffer position and return the Excel file
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return None

# --- UI RENDERING FUNCTIONS ---

def main():
    """Main application entry point."""
    # Apply cyberpunk theme
    set_cyberpunk_theme()
    
    # Render header
    st.title("⚕️ Amazon Medical Device Listing Optimizer")
    st.markdown("""
    <p class="neon-glow">Optimize product listings, reduce returns, and improve ratings for medical devices on Amazon</p>
    <div class="custom-hr"></div>
    """, unsafe_allow_html=True)
    
    # Display available modules in sidebar
    with st.sidebar:
        st.header("Settings")
        st.markdown(f"<p>Contact support: <a href='mailto:{SUPPORT_EMAIL}'>{SUPPORT_EMAIL}</a></p>", unsafe_allow_html=True)
        
        # 1-click example loader
        if st.button("Load Example Data", type="primary"):
            st.session_state.uploaded_files = EXAMPLE_DATA.copy()
            st.success("Example data loaded successfully!")
            st.experimental_rerun()
        
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
                    # Store temporarily
                    global api_key
                    temp_api_key = api_key  # Save current key
                    api_key = temp_key      # Test new key
                    
                    with st.spinner("Testing API connection..."):
                        if test_openai_api():
                            st.success("✅ Key works! Use this key in your Streamlit settings.")
                        else:
                            st.error("❌ Key doesn't work. Check if it's valid.")
                        
                    api_key = temp_api_key  # Restore original key
    
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
    st.header("🔄 Import Data")
    
    upload_tabs = st.tabs(["Structured Data Import", "Manual Entry", "Image/Review Import", "Historical Data"])
    
    # Tab 1: Structured Data Import
    with upload_tabs[0]:
        st.markdown("""
        Upload a CSV or Excel file with product data. Required columns:
        - **ASIN*** (Mandatory)
        - **Last 30 Days Sales*** (Mandatory)
        - **Last 30 Days Returns*** (Mandatory)
        
        Additional helpful columns:
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
                label="📥 Download Sample Template",
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
                        
                        # Enhanced preview with styling
                        st.markdown("### Data Preview")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("### Dataset Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**Total Products:** {len(df)}")
                        
                        with col2:
                            if 'Last 30 Days Sales' in df.columns:
                                total_sales = df['Last 30 Days Sales'].sum()
                                st.markdown(f"**Total 30-Day Sales:** {total_sales:,}")
                        
                        with col3:
                            if 'Last 30 Days Returns' in df.columns and 'Last 30 Days Sales' in df.columns:
                                total_returns = df['Last 30 Days Returns'].sum()
                                overall_return_rate = (total_returns / df['Last 30 Days Sales'].sum()) * 100 if df['Last 30 Days Sales'].sum() > 0 else 0
                                st.markdown(f"**Overall Return Rate:** {overall_return_rate:.2f}%")
                        
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
            st.subheader("📋 Product Details")
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
            st.subheader("📝 Amazon Listing Information")
            
            product_description = st.text_area(
                "Product Description/Bullet Points",
                help="Copy and paste your Amazon listing content for AI analysis",
                height=150
            )
            
            listing_url = st.text_input("Amazon Listing URL", help="Optional - for reference only")
            
            # Sales & Returns Section
            st.subheader("📊 Sales & Returns Data")
            col1, col2 = st.columns(2)
            
            with col1:
                sales_30d = st.number_input("Last 30 Days Sales*", min_value=0, help="Required field")
                sales_365d = st.number_input("Last 365 Days Sales", min_value=0, help="Optional")
            
            with col2:
                returns_30d = st.number_input("Last 30 Days Returns*", min_value=0, help="Required field")
                returns_365d = st.number_input("Last 365 Days Returns", min_value=0, help="Optional")
            
            # Ratings & Reviews Section
            st.subheader("⭐ Ratings & Reviews")
            col1, col2 = st.columns(2)
            
            with col1:
                star_rating = st.slider("Star Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1, help="Optional")
            
            with col2:
                total_reviews = st.number_input("Total Reviews", min_value=0, help="Optional")
            
            # Manual Reviews & Return Reasons (optional)
            st.subheader("👥 Customer Reviews & Return Reasons (Optional)")
            
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
            st.subheader("🏆 Competitive Positioning (Optional)")
            st.text_area(
                "Top Competing Products (One per line, format: ASIN - Product Name)",
                height=100,
                key="competing_products",
                help="Optional: Enter competing products one per line"
            )
            
            # Submit button
            submitted = st.form_submit_button("💾 Save Product Data")
        
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
                        st.dataframe(df, use_container_width=True)
                        
                        # Show product card
                        st.markdown("### Product Preview")
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Placeholder for product image or icon
                            st.markdown("""
                            <div style="background-color: rgba(153, 69, 255, 0.1); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid rgba(153, 69, 255, 0.3);">
                                <h3 style="color: #9945FF;">📦</h3>
                                <p>Product Image Placeholder</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**{product_name}**")
                            st.markdown(f"**ASIN:** {asin} | **SKU:** {sku}")
                            st.markdown(f"**Category:** {category}")
                            st.markdown(f"**Rating:** {'⭐' * int(star_rating)} ({star_rating})")
                            
                            # Calculate return rate
                            return_rate = (returns_30d / sales_30d * 100) if sales_30d > 0 else 0
                            st.markdown(f"**Return Rate:** {return_rate:.2f}%")
                            
                            if listing_url:
                                st.markdown(f"[View on Amazon]({listing_url})")
                    else:
                        st.error("pandas module is not available. Cannot process manual entry.")
                except Exception as e:
                    st.error(f"Error processing manual entry: {str(e)}")
    
    # Tab 3: Image/Review Import
    with upload_tabs[2]:
        st.markdown("""
        Upload screenshots of Amazon reviews, listings, return reports, or product images. 
        The system will analyze the content using OCR or AI Vision, depending on what's available.
        """)
        
        doc_files = st.file_uploader(
            "Upload screenshots or images (PDF, PNG, JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="documents"
        )
        
        # Add option to specify what type of content is being uploaded
        image_content_type = st.selectbox(
            "What content are you uploading?",
            options=["Product Reviews", "Product Listing", "Return Reports", "Product Images", "Competitor Listings", "Mixed Content"],
            index=0
        )
        
        # Add processing method selection
        processing_method = st.radio(
            "Processing method:",
            options=["Auto (try OCR, fall back to AI Vision)", "AI Vision only", "OCR only"],
            index=0,
            horizontal=True
        )
        
        # Add ASIN selector if structured data is available
        selected_asin = None
        if 'structured_data' in st.session_state.uploaded_files and AVAILABLE_MODULES['pandas']:
            df = st.session_state.uploaded_files['structured_data']
            asins = df['ASIN'].tolist()
            
            if asins:
                selected_asin = st.selectbox(
                    "Select product ASIN to associate with this content",
                    options=asins,
                    format_func=lambda asin: f"{asin} - {df[df['ASIN'] == asin]['Product Name'].values[0] if 'Product Name' in df.columns else 'Unknown'}"
                )
        
        if doc_files:
            # Check if processing is possible
            if processing_method == "OCR only" and not (HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']):
                st.error("OCR processing is not available. Please install pytesseract and pdf2image, or select a different processing method.")
            elif processing_method == "AI Vision only" and not AVAILABLE_MODULES['ai_api']:
                st.error("AI Vision processing is not available. Please add your OpenAI API key in settings, or select a different processing method.")
            elif processing_method == "Auto (try OCR, fall back to AI Vision)" and not (AVAILABLE_MODULES['ocr'] or AVAILABLE_MODULES['ai_api']):
                st.error("Neither OCR nor AI Vision processing is available. Please install required modules or add your OpenAI API key.")
            else:
                # Process each document
                processed_docs = []
                
                for doc in doc_files:
                    with st.spinner(f"Processing {doc.name}..."):
                        # Read file data
                        file_data = doc.read()
                        
                        if processing_method == "OCR only" and HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']:
                            # Process with OCR only
                            file_ext = doc.name.split('.')[-1].lower()
                            
                            try:
                                if file_ext == 'pdf':
                                    text = ocr_processor.process_pdf_with_ocr(file_data)
                                elif file_ext in ['png', 'jpg', 'jpeg']:
                                    text = ocr_processor.process_image_with_ocr(file_data)
                                else:
                                    text = f"Unsupported document type: {file_ext}"
                                
                                processed_docs.append({
                                    "filename": doc.name,
                                    "text": text,
                                    "type": file_ext,
                                    "content_type": image_content_type,
                                    "asin": selected_asin,
                                    "processing_method": "OCR"
                                })
                            except Exception as e:
                                st.error(f"Error processing {doc.name} with OCR: {str(e)}")
                                
                        elif processing_method == "AI Vision only" and AVAILABLE_MODULES['ai_api']:
                            # Process with AI Vision only
                            try:
                                # Prepare prompt based on content type
                                prompt = f"This is an image from an Amazon medical device {image_content_type.lower()}. "
                                
                                if image_content_type == "Product Reviews":
                                    prompt += "Please extract all customer reviews including star ratings and review text. Format as 'Rating: X stars - Review text'."
                                elif image_content_type == "Return Reports":
                                    prompt += "Please extract all return reasons. Format as a list of return reasons."
                                elif image_content_type == "Product Listing":
                                    prompt += "Please extract product name, features, specifications, and description."
                                elif image_content_type == "Product Images":
                                    prompt += "Please describe what this product image shows and any text visible in the image."
                                
                                vision_result = analyze_image_with_vision(file_data, prompt)
                                
                                if vision_result["success"]:
                                    processed_docs.append({
                                        "filename": doc.name,
                                        "text": vision_result["result"],
                                        "type": doc.name.split('.')[-1].lower(),
                                        "content_type": image_content_type,
                                        "asin": selected_asin,
                                        "processing_method": "AI Vision"
                                    })
                                else:
                                    st.error(f"Error processing {doc.name} with AI Vision: {vision_result.get('error', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error processing {doc.name} with AI Vision: {str(e)}")
                        
                        else:  # Auto method (try OCR, fall back to AI Vision)
                            # Process with the most comprehensive approach
                            result = process_image_file(file_data, doc.name, image_content_type, selected_asin)
                            
                            if result["success"]:
                                processed_docs.append(result)
                            else:
                                st.error(f"Error processing {doc.name}: {result.get('error', 'Unknown error')}")
                
                if processed_docs:
                    st.session_state.uploaded_files['documents'] = processed_docs
                    
                    # Show preview of extracted text from first document
                    if len(processed_docs) > 0:
                        with st.expander("Preview of extracted text"):
                            st.write(f"**Processing method:** {processed_docs[0].get('processing_method', 'Unknown')}")
                            st.text(processed_docs[0]["text"][:1000] + ("..." if len(processed_docs[0]["text"]) > 1000 else ""))
                        
                        # Enhanced preview with context
                        st.markdown("### Document Analysis")
                        
                        # Create tabs for each document
                        doc_tabs = st.tabs([f"Doc {i+1}: {doc['filename']}" for i, doc in enumerate(processed_docs)])
                        
                        for i, (tab, doc) in enumerate(zip(doc_tabs, processed_docs)):
                            with tab:
                                method = doc.get('processing_method', 'Unknown')
                                st.markdown(f"**File:** {doc['filename']}")
                                st.markdown(f"**Type:** {doc['content_type']}")
                                st.markdown(f"**ASIN:** {doc['asin'] if doc['asin'] else 'Not specified'}")
                                st.markdown(f"**Processed with:** {method}")
                                
                                # Special handling based on content type
                                if doc['content_type'] == "Product Reviews":
                                    # Try to extract structured review data
                                    if method == "OCR" and 'extract_amazon_reviews_data' in dir(ocr_processor):
                                        extracted_data = ocr_processor.extract_amazon_reviews_data(doc['text'])
                                        reviews = extracted_data.get("reviews", [])
                                    else:
                                        # For AI Vision, use the text to try to extract reviews
                                        reviews = []
                                        lines = doc['text'].split('\n')
                                        for line in lines:
                                            if 'Rating:' in line and ' - ' in line:
                                                try:
                                                    rating_part, review_text = line.split(' - ', 1)
                                                    rating = int(rating_part.replace('Rating:', '').strip().split()[0])
                                                    reviews.append({
                                                        'rating': rating,
                                                        'review_text': review_text.strip(),
                                                        'asin': doc['asin']
                                                    })
                                                except:
                                                    # Skip lines that don't parse cleanly
                                                    pass
                                    
                                    st.markdown(f"**Extracted {len(reviews)} reviews**")
                                    
                                    # Show preview of extracted reviews
                                    if reviews:
                                        for j, review in enumerate(reviews[:3]):  # Show first 3
                                            st.markdown(f"**Review {j+1}:** {review.get('rating', 'N/A')} stars")
                                            st.markdown(f">{review.get('review_text', 'No text')[:100]}...")
                                        
                                        # Add a button to save these reviews
                                        if selected_asin and st.button(f"Save {len(reviews)} reviews to product {selected_asin}", key=f"save_reviews_{i}"):
                                            # Add ASIN to each review
                                            for review in reviews:
                                                review['asin'] = selected_asin
                                            
                                            # Store in session state
                                            if 'manual_reviews' not in st.session_state.uploaded_files:
                                                st.session_state.uploaded_files['manual_reviews'] = {}
                                            
                                            if selected_asin in st.session_state.uploaded_files['manual_reviews']:
                                                # Append to existing reviews
                                                st.session_state.uploaded_files['manual_reviews'][selected_asin].extend(reviews)
                                            else:
                                                # Create new entry
                                                st.session_state.uploaded_files['manual_reviews'][selected_asin] = reviews
                                            
                                            st.success(f"Added {len(reviews)} reviews to product {selected_asin}")
                                
                                elif doc['content_type'] == "Return Reports":
                                    # Try to extract return reasons
                                    if method == "OCR" and 'extract_amazon_return_data' in dir(ocr_processor):
                                        extracted_data = ocr_processor.extract_amazon_return_data(doc['text'])
                                        returns = extracted_data.get("returns", [])
                                    else:
                                        # For AI Vision, use the text to try to extract return reasons
                                        returns = []
                                        lines = doc['text'].split('\n')
                                        for line in lines:
                                            if line.strip() and not line.startswith('Please ') and not line.startswith('This is'):
                                                returns.append({
                                                    'return_reason': line.strip(),
                                                    'asin': doc['asin']
                                                })
                                    
                                    st.markdown(f"**Extracted {len(returns)} return reasons**")
                                    
                                    # Show preview of extracted return reasons
                                    if returns:
                                        for j, return_item in enumerate(returns[:3]):  # Show first 3
                                            st.markdown(f"**Return {j+1}:** {return_item.get('return_reason', 'No reason')[:100]}...")
                                        
                                        # Add a button to save these return reasons
                                        if selected_asin and st.button(f"Save {len(returns)} return reasons to product {selected_asin}", key=f"save_returns_{i}"):
                                            # Add ASIN to each return
                                            for return_item in returns:
                                                return_item['asin'] = selected_asin
                                            
                                            # Store in session state
                                            if 'manual_returns' not in st.session_state.uploaded_files:
                                                st.session_state.uploaded_files['manual_returns'] = {}
                                            
                                            if selected_asin in st.session_state.uploaded_files['manual_returns']:
                                                # Append to existing returns
                                                st.session_state.uploaded_files['manual_returns'][selected_asin].extend(returns)
                                            else:
                                                # Create new entry
                                                st.session_state.uploaded_files['manual_returns'][selected_asin] = returns
                                            
                                            st.success(f"Added {len(returns)} return reasons to product {selected_asin}")
                                
                                elif doc['content_type'] == "Product Listing":
                                    # Show the extracted listing text
                                    st.markdown("**Extracted Listing Content:**")
                                    st.text_area("Listing Text", value=doc['text'], height=250, disabled=True)
                                
                                elif doc['content_type'] == "Product Images":
                                    # Show the image description
                                    st.markdown("**Image Description:**")
                                    st.text_area("Description", value=doc['text'], height=250, disabled=True)
                        
                        # Run AI analysis on the extracted text if enabled
                        if AVAILABLE_MODULES['ai_api'] and st.button("Analyze All Extracted Content with AI"):
                            with st.spinner("Analyzing content with AI..."):
                                for i, doc in enumerate(processed_docs):
                                    st.markdown(f"### Analyzing: {doc['filename']}")
                                    
                                    analysis_type = ""
                                    if doc['content_type'] == "Product Reviews":
                                        analysis_type = "sentiment"
                                    elif doc['content_type'] in ["Product Listing", "Competitor Listings"]:
                                        analysis_type = "listing_optimization"
                                    elif doc['content_type'] == "Return Reports":
                                        analysis_type = "return_analysis"
                                    elif doc['content_type'] == "Product Images":
                                        analysis_type = "image_feedback"
                                    
                                    if analysis_type:
                                        result = analyze_with_ai(doc["text"], analysis_type)
                                        if result["success"]:
                                            st.markdown(f"#### 🧠 AI Analysis Results for {doc['filename']}")
                                            st.markdown(result["result"])
                                        else:
                                            st.error(f"AI analysis failed for {doc['filename']}: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("No documents were successfully processed. Please try again or select a different processing method.")
    
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
                    
                    # Enhanced data preview
                    st.markdown("### Historical Data Preview")
                    st.dataframe(hist_df.head(), use_container_width=True)
                    
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
                                st.markdown("### 📈 Historical Trends")
                                
                                # Create a more cyberpunk-styled plot
                                fig = px.line(hist_df, x='Date', y=plot_cols, title="Historical Performance")
                                
                                # Update layout for cyberpunk style
                                fig.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor=COLOR_BACKGROUND,
                                    paper_bgcolor=COLOR_BACKGROUND,
                                    font_color=COLOR_TEXT,
                                    title_font_color=COLOR_PRIMARY,
                                    legend_title_font_color=COLOR_PRIMARY,
                                    legend_title_font_size=12,
                                    xaxis=dict(
                                        title_font=dict(color=COLOR_SECONDARY),
                                        tickfont=dict(color=COLOR_TEXT),
                                        gridcolor='rgba(153, 69, 255, 0.1)',
                                        zerolinecolor='rgba(153, 69, 255, 0.2)',
                                    ),
                                    yaxis=dict(
                                        title_font=dict(color=COLOR_SECONDARY),
                                        tickfont=dict(color=COLOR_TEXT),
                                        gridcolor='rgba(153, 69, 255, 0.1)',
                                        zerolinecolor='rgba(153, 69, 255, 0.2)',
                                    )
                                )
                                
                                # Update line colors to match cyberpunk theme
                                colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_INFO]
                                for i, col in enumerate(plot_cols):
                                    fig.data[i].line.color = colors[i % len(colors)]
                                    fig.data[i].line.width = 3
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add summary metrics
                                if len(hist_df) > 1:
                                    st.markdown("### Summary Metrics")
                                    metric_cols = st.columns(len(plot_cols))
                                    
                                    for i, col in enumerate(plot_cols):
                                        latest = hist_df[col].iloc[-1]
                                        previous = hist_df[col].iloc[-2]
                                        delta = latest - previous
                                        delta_percent = (delta / previous) * 100 if previous != 0 else 0
                                        
                                        with metric_cols[i]:
                                            st.metric(
                                                label=f"Latest {col}",
                                                value=f"{latest:.2f}" if "Rate" in col else f"{latest:,.0f}",
                                                delta=f"{delta_percent:+.2f}%" if delta_percent != 0 else None,
                                                delta_color="inverse" if "Returns" in col or "Rate" in col else "normal"
                                            )
                        except Exception as e:
                            st.error(f"Error creating trend visualization: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing historical data: {str(e)}")
        elif hist_file and not AVAILABLE_MODULES['pandas']:
            st.error("pandas module is not available for processing files.")

def render_product_selection():
    """Render the product selection section."""
    st.header("🔎 Select Product to Analyze")
    
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
    
    # Create a more visually appealing product selector
    st.markdown("### Select a Product")
    
    # Create a dataframe for display with key metrics
    display_df = df[display_cols].copy()
    
    # Add return rate if available
    if 'Last 30 Days Returns' in df.columns and 'Last 30 Days Sales' in df.columns:
        display_df['Return Rate'] = (df['Last 30 Days Returns'] / df['Last 30 Days Sales'] * 100).round(2).astype(str) + '%'
    
    # Better styled dataframe for selection
    st.dataframe(display_df, use_container_width=True)
    
    # Display product selector
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
    product_info['return_rate_30d'] = safe_divide(product_info['returns_30d'], product_info['sales_30d']) * 100
    if product_info['sales_365d'] is not None and product_info['returns_365d'] is not None:
        product_info['return_rate_365d'] = safe_divide(product_info['returns_365d'], product_info['sales_365d']) * 100
    
    # Store selected product in session state
    st.session_state.current_product = product_info
    
    # Calculate additional e-commerce metrics
    monthly_revenue = product_info['sales_30d'] * 50  # Assuming average selling price of $50
    transaction_fee = monthly_revenue * 0.15  # Assuming 15% Amazon fee
    cost_of_returns = product_info['returns_30d'] * 15  # Assuming $15 cost per return processing
    
    # Create a visually appealing product card
    st.markdown("### Product Overview")
    
    # Header card with product name and basic info
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {COLOR_PANEL} 0%, rgba(153, 69, 255, 0.1) 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px; 
                border: 1px solid rgba(153, 69, 255, 0.3); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h2 style="margin:0; color: {COLOR_PRIMARY}; text-shadow: 0 0 10px rgba(153, 69, 255, 0.5);">
            {product_info['name']}
        </h2>
        <p style="margin:5px 0;">
            <span style="background-color: rgba(153, 69, 255, 0.2); padding: 3px 8px; border-radius: 20px; 
                  font-size: 12px; margin-right: 5px;">
                ASIN: {product_info['asin']}
            </span>
            <span style="background-color: rgba(153, 69, 255, 0.2); padding: 3px 8px; border-radius: 20px; 
                  font-size: 12px; margin-right: 5px;">
                SKU: {product_info['sku']}
            </span>
            <span style="background-color: rgba(20, 241, 149, 0.2); padding: 3px 8px; border-radius: 20px; 
                  font-size: 12px;">
                {product_info['category']}
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Key Metrics in a grid
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        display_colored_metric("30-Day Sales", product_info['sales_30d'])
        display_colored_metric("Est. Monthly Revenue", monthly_revenue, style="money")
    
    with metric_col2:
        display_colored_metric("30-Day Returns", product_info['returns_30d'])
        display_colored_metric("Return Processing Cost", cost_of_returns, style="money")
    
    with metric_col3:
        display_colored_metric("30-Day Return Rate", product_info['return_rate_30d'], style="percent")
        display_colored_metric("Amazon Fees (Est.)", transaction_fee, style="money")
    
    with metric_col4:
        if product_info['star_rating'] is not None:
            display_colored_metric("Star Rating", f"{product_info['star_rating']:.1f} ★")
            # Calculate rating improvement impact
            potential_impact = 0
            if product_info['star_rating'] < 4.5:
                potential_impact = ((4.5 - product_info['star_rating']) / product_info['star_rating']) * monthly_revenue * 0.10
                display_colored_metric("Rating Impact Value", potential_impact, style="money")
    
    # Display listing URL if available
    if product_info['listing_url']:
        st.markdown(f"[View Amazon Listing]({product_info['listing_url']})")
    
    # Get review data for the selected product from documents if available
    reviews_data = []
    
    # 1. Check for OCR-extracted reviews from documents
    if 'documents' in st.session_state.uploaded_files and HAS_LOCAL_MODULES and AVAILABLE_MODULES['ocr']:
        docs = st.session_state.uploaded_files['documents']
        
        for doc in docs:
            # Extract reviews from OCR text if content type is reviews and matches current ASIN
            if doc.get("content_type") == "Product Reviews" and (doc.get("asin") is None or doc.get("asin") == product_info['asin']):
                text = doc["text"]
                extracted_reviews = ocr_processor.extract_amazon_reviews_data(text) if 'extract_amazon_reviews_data' in dir(ocr_processor) else {"reviews": []}
                if "reviews" in extracted_reviews:
                    extracted_reviews_data = extracted_reviews["reviews"]
                    # Add ASIN if not present
                    for review in extracted_reviews_data:
                        if 'asin' not in review:
                            review['asin'] = product_info['asin']
                    reviews_data.extend(extracted_reviews_data)
    
    # 2. Check for manually entered reviews
    if 'manual_reviews' in st.session_state.uploaded_files:
        manual_reviews = st.session_state.uploaded_files['manual_reviews']
        if product_info['asin'] in manual_reviews:
            reviews_data.extend(manual_reviews[product_info['asin']])
    
    if reviews_data:
        st.markdown(f"Found {len(reviews_data)} reviews for this product.")
        
        # Show categorized review summary
        if len(reviews_data) > 0:
            st.markdown("### Review Themes")
            categorized_reviews = categorize_reviews(reviews_data)
            categories = categorized_reviews["categories"]
            counts = categorized_reviews["counts"]
            sentiment = categorized_reviews["sentiment"]
            
            # Create a summary visualization
            if AVAILABLE_MODULES['plotly']:
                # Prepare data for visualization
                categories_list = list(counts.keys())
                counts_list = list(counts.values())
                sentiment_list = [sentiment.get(cat, 3) for cat in categories_list]
                
                # Create a horizontal bar chart with sentiment coloring
                fig = go.Figure()
                
                # Add bars
                fig.add_trace(go.Bar(
                    y=categories_list,
                    x=counts_list,
                    orientation='h',
                    marker=dict(
                        color=[
                            # Color gradient based on sentiment (1-5 stars)
                            f'rgba({255-int((s-1)*50)}, {int((s-1)*50)}, {255}, 0.8)' for s in sentiment_list
                        ],
                        line=dict(color='rgba(153, 69, 255, 0.6)', width=1)
                    ),
                    text=[f"{count} reviews<br>Avg: {sentiment.get(cat, 0):.1f}★" for count, cat in zip(counts_list, categories_list)],
                    textposition='auto',
                    hoverinfo='text',
                    hovertext=[f"Category: {cat}<br>Count: {count}<br>Avg Rating: {sentiment.get(cat, 0):.1f}★" 
                              for cat, count in zip(categories_list, counts_list)]
                ))
                
                # Update layout
                fig.update_layout(
                    title="Review Themes by Frequency and Sentiment",
                    xaxis_title="Number of Reviews",
                    yaxis_title="Theme Category",
                    template="plotly_dark",
                    plot_bgcolor=COLOR_BACKGROUND,
                    paper_bgcolor=COLOR_BACKGROUND,
                    font_color=COLOR_TEXT,
                    height=max(300, len(categories_list) * 40),  # Adjust height based on number of categories
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(
                        categoryorder='total ascending'  # Sort by count
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display example reviews for each category
            st.markdown("#### Example Review for Each Theme")
            theme_cols = st.columns(2)
            
            for i, (category, reviews) in enumerate(categories.items()):
                if reviews:  # Skip empty categories
                    # Sort by length to get more detailed examples
                    sorted_reviews = sorted(reviews, key=lambda x: len(x.get('review_text', '')), reverse=True)
                    example = sorted_reviews[0] if sorted_reviews else reviews[0]
                    
                    with theme_cols[i % 2]:
                        rating = example.get('rating', 'N/A')
                        text = example.get('review_text', 'No text')
                        avg_rating = sentiment.get(category, 0)
                        
                        st.markdown(f"""
                        <div style="background-color: rgba(153, 69, 255, 0.1); padding: 15px; border-radius: 8px; 
                                margin-bottom: 15px; border: 1px solid rgba(153, 69, 255, 0.2);">
                            <h4 style="margin-top: 0; color: {COLOR_PRIMARY};">
                                {category} ({counts[category]} reviews, {avg_rating:.1f}★)
                            </h4>
                            <p style="color: {'#14F195' if rating >= 4 else '#FFD166' if rating >= 3 else '#FF5678'}; 
                                    font-weight: bold; margin-bottom: 5px;">
                                {rating} stars
                            </p>
                            <p style="font-style: italic; margin-top: 0;">"{text}"</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Get return reasons data
    return_reasons_data = []
    
    # Check for manually entered return reasons
    if 'manual_returns' in st.session_state.uploaded_files:
        manual_returns = st.session_state.uploaded_files['manual_returns']
        if product_info['asin'] in manual_returns:
            return_reasons_data.extend(manual_returns[product_info['asin']])
    
    if return_reasons_data:
        st.markdown(f"Found {len(return_reasons_data)} return reasons for this product.")
        
        # Show categorized return reasons
        if len(return_reasons_data) > 0:
            st.markdown("### Return Reason Categories")
            categorized_returns = categorize_returns(return_reasons_data)
            categories = categorized_returns["categories"]
            counts = categorized_returns["counts"]
            
            # Create a summary visualization
            if AVAILABLE_MODULES['plotly']:
                # Prepare data for visualization
                categories_list = list(counts.keys())
                counts_list = list(counts.values())
                
                # Create a horizontal bar chart
                fig = go.Figure()
                
                # Determine color based on category criticality
                colors = []
                for cat in categories_list:
                    if cat in ["Quality/Durability", "Medical Effectiveness"]:
                        colors.append(COLOR_DANGER)  # Critical issues
                    elif cat in ["Expectation Mismatch", "Listing Accuracy", "Size/Fit Issues"]:
                        colors.append(COLOR_WARNING)  # Important issues
                    else:
                        colors.append(COLOR_INFO)  # Standard issues
                
                # Add bars
                fig.add_trace(go.Bar(
                    y=categories_list,
                    x=counts_list,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(153, 69, 255, 0.6)', width=1)
                    ),
                    text=[f"{count} returns ({count/len(return_reasons_data)*100:.1f}%)" for count in counts_list],
                    textposition='auto',
                    hoverinfo='text',
                    hovertext=[f"Category: {cat}<br>Count: {count}<br>Percentage: {count/len(return_reasons_data)*100:.1f}%" 
                              for cat, count in zip(categories_list, counts_list)]
                ))
                
                # Update layout
                fig.update_layout(
                    title="Return Reasons by Category",
                    xaxis_title="Number of Returns",
                    yaxis_title="Return Category",
                    template="plotly_dark",
                    plot_bgcolor=COLOR_BACKGROUND,
                    paper_bgcolor=COLOR_BACKGROUND,
                    font_color=COLOR_TEXT,
                    height=max(300, len(categories_list) * 40),  # Adjust height based on number of categories
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(
                        categoryorder='total ascending'  # Sort by count
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display example return reasons for each category
            st.markdown("#### Example Return Reason for Each Category")
            reason_cols = st.columns(2)
            
            for i, (category, returns) in enumerate(categories.items()):
                if returns:  # Skip empty categories
                    # Sort by length to get more detailed examples
                    sorted_returns = sorted(returns, key=lambda x: len(x.get('return_reason', '')), reverse=True)
                    example = sorted_returns[0] if sorted_returns else returns[0]
                    
                    with reason_cols[i % 2]:
                        text = example.get('return_reason', 'No reason provided')
                        
                        # Determine category color
                        if category in ["Quality/Durability", "Medical Effectiveness"]:
                            cat_color = COLOR_DANGER  # Critical issues
                        elif category in ["Expectation Mismatch", "Listing Accuracy", "Size/Fit Issues"]:
                            cat_color = COLOR_WARNING  # Important issues
                        else:
                            cat_color = COLOR_INFO  # Standard issues
                        
                        st.markdown(f"""
                        <div style="background-color: rgba(153, 69, 255, 0.1); padding: 15px; border-radius: 8px; 
                                margin-bottom: 15px; border: 1px solid rgba(153, 69, 255, 0.2);">
                            <h4 style="margin-top: 0; color: {cat_color};">
                                {category} ({counts[category]} returns)
                            </h4>
                            <p style="font-style: italic; margin-top: 0;">"{text}"</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Get competitors data if available
    competitors_data = []
    if 'competitors' in st.session_state.uploaded_files:
        competitors = st.session_state.uploaded_files['competitors']
        if product_info['asin'] in competitors:
            competitors_data.extend(competitors[product_info['asin']])
            
    if competitors_data:
        st.markdown(f"Found {len(competitors_data)} competing products for comparison.")
    
    # Get historical data for the selected product if available
    historical_data = None
    if 'historical_data' in st.session_state.uploaded_files:
        hist_df = st.session_state.uploaded_files['historical_data']
        # Filter for the selected product if ASIN column exists
        if 'ASIN' in hist_df.columns:
            hist_filtered = hist_df[hist_df['ASIN'] == product_info['asin']]
            if len(hist_filtered) > 0:
                historical_data = hist_filtered
                st.markdown(f"Found {len(historical_data)} historical data points for this product.")
    
    # Analysis buttons
    st.markdown("### 📊 Run Analysis")
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        if st.button("Standard Analysis", type="primary", key="standard_analysis"):
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
                                # Enhanced review analysis with categorization
                                review_analysis = data_analysis.analyze_reviews(reviews_data)
                                categorized_reviews = categorize_reviews(reviews_data)
                                
                                analysis_result += f"\nReview Analysis:\n"
                                analysis_result += f"Total Reviews: {review_analysis['total_reviews']}\n"
                                if review_analysis['average_rating']:
                                    analysis_result += f"Average Rating: {review_analysis['average_rating']:.1f}\n"
                                
                                # Add categorized review summary
                                analysis_result += "\nReview Categories:\n"
                                for category, count in categorized_reviews['counts'].items():
                                    sentiment = categorized_reviews['sentiment'].get(category, 0)
                                    analysis_result += f"- {category}: {count} reviews, Avg Rating: {sentiment:.1f}\n"
                            
                            # Enhanced return reason analysis with categorization
                            if return_reasons_data:
                                categorized_returns = categorize_returns(return_reasons_data)
                                
                                analysis_result += f"\nReturn Reason Analysis:\n"
                                analysis_result += f"Total Return Reasons: {len(return_reasons_data)}\n"
                                
                                # Add categorized return reason summary
                                analysis_result += "\nReturn Categories:\n"
                                for category, count in categorized_returns['counts'].items():
                                    percent = (count / len(return_reasons_data)) * 100
                                    analysis_result += f"- {category}: {count} returns ({percent:.1f}%)\n"
                            
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
                            
                            # Add categorized reviews if available
                            if reviews_data:
                                categorized_reviews = categorize_reviews(reviews_data)
                                analysis_result += f"\nReview Analysis:\n"
                                analysis_result += f"Total Reviews: {len(reviews_data)}\n"
                                
                                # Add categorized review summary
                                analysis_result += "\nReview Categories:\n"
                                for category, count in categorized_reviews['counts'].items():
                                    sentiment = categorized_reviews['sentiment'].get(category, 0)
                                    analysis_result += f"- {category}: {count} reviews, Avg Rating: {sentiment:.1f}\n"
                            
                            # Add categorized return reasons if available
                            if return_reasons_data:
                                categorized_returns = categorize_returns(return_reasons_data)
                                analysis_result += f"\nReturn Reason Analysis:\n"
                                analysis_result += f"Total Return Reasons: {len(return_reasons_data)}\n"
                                
                                # Add categorized return reason summary
                                analysis_result += "\nReturn Categories:\n"
                                for category, count in categorized_returns['counts'].items():
                                    percent = (count / len(return_reasons_data)) * 100
analysis_result += f"- {category}: {count} returns ({percent:.1f}%)\n"
                            
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
                        
                        # Show results immediately instead of rerun
                        st.markdown("### Analysis Results")
                        st.text(analysis_result)
                        
                        # Generate Excel download link
                        if AVAILABLE_MODULES['xlsx_writer']:
                            st.markdown("### Export Results")
                            excel_data = export_analysis_to_excel(
                                product_info, 
                                st.session_state.analysis_results[product_info['asin']]
                            )
                            
                            if excel_data:
                                st.download_button(
                                    label="📥 Download Analysis as Excel",
                                    data=excel_data,
                                    file_name=f"{product_info['asin']}_analysis.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.error("Please select a product to analyze.")
    
    with analysis_col2:
        if AVAILABLE_MODULES['ai_api']:
            if st.button("AI-Enhanced Analysis", type="secondary", key="ai_analysis"):
                if st.session_state.current_product:
                    # First check if API call is in progress
                    if st.session_state.api_call_in_progress:
                        st.warning("An AI analysis is already in progress. Please wait for it to complete.")
                        return
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Run standard analysis if not already done (20%)
                        status_text.text("Step 1/5: Running standard analysis...")
                        
                        if product_info['asin'] not in st.session_state.analysis_results:
                            analysis_result = f"Analysis for {product_info['name']} ({product_info['asin']})\n\n"
                            analysis_result += f"30-Day Sales: {product_info['sales_30d']}\n"
                            analysis_result += f"30-Day Returns: {product_info['returns_30d']}\n"
                            analysis_result += f"30-Day Return Rate: {product_info['return_rate_30d']:.2f}%\n"
                            
                            if reviews_data:
                                # Use built-in review analysis
                                categorized_reviews = categorize_reviews(reviews_data)
                                
                                analysis_result += f"\nReview Analysis:\n"
                                analysis_result += f"Total Reviews: {len(reviews_data)}\n"
                                
                                # Add categorized review summary
                                analysis_result += "\nReview Categories:\n"
                                for category, count in categorized_reviews['counts'].items():
                                    sentiment = categorized_reviews['sentiment'].get(category, 0)
                                    analysis_result += f"- {category}: {count} reviews, Avg Rating: {sentiment:.1f}\n"
                            
                            # Enhanced return reason analysis with categorization
                            if return_reasons_data:
                                categorized_returns = categorize_returns(return_reasons_data)
                                
                                analysis_result += f"\nReturn Reason Analysis:\n"
                                analysis_result += f"Total Return Reasons: {len(return_reasons_data)}\n"
                                
                                # Add categorized return reason summary
                                analysis_result += "\nReturn Categories:\n"
                                for category, count in categorized_returns['counts'].items():
                                    percent = (count / len(return_reasons_data)) * 100
                                    analysis_result += f"- {category}: {count} returns ({percent:.1f}%)\n"
                            
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
                        
                        progress_bar.progress(20)
                        
                        # Initialize AI insights structure
                        ai_insights = {}
                        
                        # Step 2: Analyze reviews with AI (40%)
                        if reviews_data:
                            status_text.text("Step 2/5: Analyzing customer reviews...")
                            review_analysis = process_reviews_with_ai(reviews_data)
                            if review_analysis.get('success', False):
                                ai_insights['review_insights'] = review_analysis.get('result', 'Analysis failed')
                        else:
                            ai_insights['review_insights'] = "No reviews available for analysis."
                        
                        progress_bar.progress(40)
                        
                        # Step 3: Analyze return reasons with AI (60%)
                        if return_reasons_data:
                            status_text.text("Step 3/5: Analyzing return reasons...")
                            return_analysis = process_returns_with_ai(return_reasons_data)
                            if return_analysis.get('success', False):
                                ai_insights['return_insights'] = return_analysis.get('result', 'Analysis failed')
                        else:
                            ai_insights['return_insights'] = "No return reasons available for analysis."
                        
                        progress_bar.progress(60)
                        
                        # Step 4: Generate listing optimization recommendations (80%)
                        status_text.text("Step 4/5: Generating listing optimization recommendations...")
                        listing_optimization = analyze_listing_optimization(product_info)
                        if listing_optimization.get('success', False):
                            ai_insights['listing_optimization'] = listing_optimization.get('result', 'Analysis failed')
                        
                        progress_bar.progress(80)
                        
                        # Step 5: Generate improvement recommendations (100%)
                        status_text.text("Step 5/5: Generating improvement recommendations...")
                        if reviews_data or return_reasons_data:
                            recommendations = generate_improvement_recommendations(
                                product_info,
                                reviews_data,
                                return_reasons_data
                            )
                            
                            if recommendations.get('success', False):
                                ai_insights['recommendations'] = recommendations.get('result', 'Recommendation generation failed')
                        
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        
                        # Store AI insights in session state
                        st.session_state.ai_insights[product_info['asin']] = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'insights': ai_insights
                        }
                        
                        st.success("AI analysis complete! View detailed results in the AI Insights tab.")
                        
                        # Show a preview of the AI insights
                        with st.expander("View AI Analysis Preview"):
                            if 'recommendations' in ai_insights:
                                st.markdown("### 🧠 AI Recommendations")
                                st.markdown(ai_insights['recommendations'])
                            
                            if 'listing_optimization' in ai_insights:
                                st.markdown("### 📋 Listing Optimization")
                                st.markdown(ai_insights['listing_optimization'][:500] + "... (see full analysis in AI Insights tab)")
                        
                        # Generate Excel download link
                        if AVAILABLE_MODULES['xlsx_writer']:
                            st.markdown("### Export Results")
                            excel_data = export_analysis_to_excel(
                                product_info, 
                                st.session_state.analysis_results[product_info['asin']]
                            )
                            
                            if excel_data:
                                st.download_button(
                                    label="📥 Download Analysis as Excel",
                                    data=excel_data,
                                    file_name=f"{product_info['asin']}_analysis.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    except Exception as e:
                        st.error(f"Error during AI analysis: {str(e)}")
                else:
                    st.error("Please select a product to analyze.")
        else:
            st.warning("AI-Enhanced Analysis requires OpenAI API key in Streamlit app settings")

def render_listing_optimization():
    """Render the listing optimization section."""
    st.header("📝 Amazon Listing Optimization")
    
    # Check if we have a current product selected
    if not st.session_state.current_product:
        st.info("Please select a product in the Analyze tab first.")
        return
    
    product_info = st.session_state.current_product
    
    st.subheader(f"Optimize Listing for {product_info['name']} ({product_info['asin']})")
    
    # Check if AI is available
    if not AVAILABLE_MODULES['ai_api']:
        st.warning("OpenAI API is required for listing optimization. Please add your API key to Streamlit app settings.")
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
                    st.experimental_rerun()
        
        # AI analysis button
        if st.button("Generate AI Optimization Recommendations", type="primary"):
            with st.spinner("Analyzing listing with AI..."):
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

    # For tabs 3-7 (other optimization areas), similar pattern as tab 2
    # These tabs would extract the relevant sections from the AI recommendations
    # and display them along with product info

def render_ai_insights():
    """Render the AI insights section."""
    st.header("🧠 AI Insights")
    
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
                else:
                    st.info("No listing optimization insights available. Go to the Listing Optimization tab to generate recommendations.")
            
            # Tab 3: Review Analysis
            with tabs[2]:
                if 'review_insights' in insights['insights']:
                    st.markdown("### Review Analysis")
                    st.markdown(insights['insights']['review_insights'])
                else:
                    st.info("No AI review insights available. Run an AI-Enhanced Analysis with reviews data to generate insights.")
            
            # Tab 4: Return Analysis
            with tabs[3]:
                if 'return_insights' in insights['insights']:
                    st.markdown("### Return Reason Analysis")
                    st.markdown(insights['insights']['return_insights'])
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
                
                if 'review_insights' in insights['insights']:
                    export_content += "## Review Analysis\n"
                    export_content += insights['insights']['review_insights'] + "\n\n"
                
                if 'return_insights' in insights['insights']:
                    export_content += "## Return Analysis\n"
                    export_content += insights['insights']['return_insights'] + "\n\n"
                
                # Download buttons
                st.download_button(
                    label="Export as Markdown",
                    data=export_content,
                    file_name=f"{product_info['asin']}_ai_insights.md",
                    mime="text/markdown"
                )
                
                # Generate Excel export if available
                if AVAILABLE_MODULES['xlsx_writer'] and product_info['asin'] in st.session_state.analysis_results:
                    excel_data = export_analysis_to_excel(
                        product_info, 
                        st.session_state.analysis_results[product_info['asin']]
                    )
                    
                    if excel_data:
                        st.download_button(
                            label="Export as Excel",
                            data=excel_data,
                            file_name=f"{product_info['asin']}_ai_insights.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
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
                    st.experimental_rerun()
                else:
                    st.error("Product information not found. Please run a standard analysis first.")
        else:
            st.info("No AI insights available. Please select and analyze a product with AI-Enhanced Analysis.")

def render_help_section():
    """Render the help and documentation section."""
    st.header("ℹ️ Help & Documentation")
    
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
        - Export insights as Markdown for sharing with your team
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
        
        The system analyzes images with OCR or AI Vision to extract text for analysis.
        
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
        """)
        
        st.warning("Note: The AI requires your OpenAI API key to be configured in Streamlit app settings as 'openai_api_key'.")
    
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
        
        st.info("💡 **Tip:** Use the 'Test API Connection' button in the sidebar to verify your OpenAI API key is working correctly.")

# Run the application
if __name__ == "__main__":
    main()
