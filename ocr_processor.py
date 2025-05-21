import io
import re
import logging
import pandas as pd
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

# Check if OCR dependencies are available
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
    logger.info("OCR dependencies are available")
except ImportError:
    logger.warning("OCR dependencies (pytesseract, pdf2image) are not available")

def process_pdf_with_ocr(pdf_bytes):
    """Extract text from PDF using OCR."""
    if not OCR_AVAILABLE:
        return "OCR processing not available. Install pytesseract and pdf2image."
    
    try:
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

def process_image_with_ocr(image_bytes):
    """Extract text from image using OCR."""
    if not OCR_AVAILABLE:
        return "OCR processing not available. Install pytesseract."
    
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to text using OCR
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.error(f"Error processing image with OCR: {str(e)}")
        return f"Error processing image: {str(e)}"

def extract_amazon_returns_data(ocr_text):
    """Extract Amazon returns data from OCR text."""
    # Just perform basic text analysis if OCR is not available
    if not OCR_AVAILABLE:
        return "Text analysis available, but OCR extraction is not available."
    
    # Patterns for Amazon returns page
    order_id_pattern = r"Order ID:\s+([\w\-]+)"
    return_reason_pattern = r"Return Reason:\s+(.*?)(?=Buyer Comment:|Request Date:|$)"
    buyer_comment_pattern = r"Buyer Comment:\s+(.*?)(?=Request Date:|$)"
    request_date_pattern = r"Request Date:\s+(\d{2}/\d{2}/\d{4})"
    
    # Extract data using regex
    order_ids = re.findall(order_id_pattern, ocr_text)
    return_reasons = re.findall(return_reason_pattern, ocr_text, re.DOTALL)
    buyer_comments = re.findall(buyer_comment_pattern, ocr_text, re.DOTALL)
    request_dates = re.findall(request_date_pattern, ocr_text)
    
    # Clean up extracted text
    return_reasons = [reason.strip() for reason in return_reasons]
    buyer_comments = [comment.strip() for comment in buyer_comments]
    
    # Create a list of returns data
    returns_data = []
    for i in range(min(len(order_ids), len(return_reasons), len(request_dates))):
        returns_data.append({
            "order_id": order_ids[i] if i < len(order_ids) else "",
            "return_reason": return_reasons[i] if i < len(return_reasons) else "",
            "buyer_comment": buyer_comments[i] if i < len(buyer_comments) else "",
            "request_date": request_dates[i] if i < len(request_dates) else ""
        })
    
    return returns_data

def extract_amazon_reviews_data(ocr_text):
    """Extract Amazon reviews data from OCR text."""
    # Just perform basic text analysis if OCR is not available
    if not OCR_AVAILABLE:
        return {"overall_rating": None, "reviews": []}
    
    # Patterns for Amazon reviews
    review_pattern = r"((?:\d+ star|★+)[^\n]*?)(?=\d+ star|★+|$)"
    star_rating_pattern = r"(\d+(?:\.\d+)?) out of 5"
    star_count_pattern = r"(\d+) (?:star|★)"
    
    # Extract reviews
    reviews = re.findall(review_pattern, ocr_text, re.DOTALL)
    
    # Extract overall rating if present
    overall_rating = None
    overall_match = re.search(star_rating_pattern, ocr_text)
    if overall_match:
        overall_rating = float(overall_match.group(1))
    
    # Process reviews
    reviews_data = []
    for review in reviews:
        # Try to extract star rating
        star_match = re.search(star_count_pattern, review)
        stars = int(star_match.group(1)) if star_match else None
        
        # Clean up review text
        review_text = re.sub(r"\d+ star|★+", "", review).strip()
        
        if review_text:
            reviews_data.append({
                "rating": stars,
                "review_text": review_text
            })
    
    return {
        "overall_rating": overall_rating,
        "reviews": reviews_data
    }

def convert_ocr_to_dataframe(extracted_data):
    """Convert extracted OCR data to pandas DataFrame."""
    if not OCR_AVAILABLE:
        return pd.DataFrame()
    
    if not extracted_data:
        return pd.DataFrame()
    
    # If the data is already in list format
    if isinstance(extracted_data, list):
        return pd.DataFrame(extracted_data)
    
    # If the data is in nested dictionary format
    if isinstance(extracted_data, dict) and "reviews" in extracted_data:
        return pd.DataFrame(extracted_data["reviews"])
    
    # If it's a different format, create a simple one-row dataframe
    return pd.DataFrame([extracted_data])
