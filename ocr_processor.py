import io
import re
import logging
import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import tempfile

logger = logging.getLogger(__name__)

def process_pdf_with_ocr(pdf_bytes):
    """Extract text from PDF using OCR."""
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

def extract_voc_data(ocr_text):
    """Extract Voice of Customer data from OCR text."""
    # This function extracts structured data from the Voice of Customer section
    # The exact patterns will depend on the format of your VOC data
    
    # Sample patterns for the provided examples
    product_pattern = r"(Vive \d+ Wheel Walker[^-]*)-(.*?)(?=ASIN|$)"
    asin_pattern = r"ASIN\s*([A-Z0-9]+)"
    sku_pattern = r"SKU\s*([A-Z0-9]+)"
    star_rating_pattern = r"Star rating\s*([★☆]+)"
    return_rate_pattern = r"Return rate\s*(.*?)(?=\n|$)"
    
    # Extract data
    product_match = re.search(product_pattern, ocr_text)
    product_name = product_match.group(0) if product_match else None
    
    asin_match = re.search(asin_pattern, ocr_text)
    asin = asin_match.group(1) if asin_match else None
    
    sku_match = re.search(sku_pattern, ocr_text)
    sku = sku_match.group(1) if sku_match else None
    
    # Extract comments sections
    returns_section = re.search(r"Returns(.*?)Reviews", ocr_text, re.DOTALL)
    returns_text = returns_section.group(1) if returns_section else ""
    
    reviews_section = re.search(r"Reviews(.*?)(?:Other sources|$)", ocr_text, re.DOTALL)
    reviews_text = reviews_section.group(1) if reviews_section else ""
    
    # Process returns
    return_items = re.findall(r"Order ID:[^\n]*\n(.*?)(?=Order ID:|$)", returns_text, re.DOTALL)
    returns = [item.strip() for item in return_items if item.strip()]
    
    # Process reviews
    review_items = re.findall(r"(?:Frame|Vive)[^\n]*\n(.*?)(?=Frame|Vive|$)", reviews_text, re.DOTALL)
    reviews = [item.strip() for item in review_items if item.strip()]
    
    # Create structured data
    voc_data = {
        "product_name": product_name,
        "asin": asin,
        "sku": sku,
        "returns": returns,
        "reviews": reviews
    }
    
    return voc_data

def convert_ocr_to_dataframe(extracted_data):
    """Convert extracted OCR data to pandas DataFrame."""
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

def process_document(file_bytes, file_name):
    """Process a document file and extract structured data."""
    file_ext = file_name.split('.')[-1].lower()
    
    # Extract text using appropriate method
    if file_ext == 'pdf':
        text = process_pdf_with_ocr(file_bytes)
    elif file_ext in ['png', 'jpg', 'jpeg']:
        text = process_image_with_ocr(file_bytes)
    else:
        return None, f"Unsupported file type: {file_ext}"
    
    # Determine document type based on content patterns
    if "Order ID:" in text and "Return Reason:" in text:
        data = extract_amazon_returns_data(text)
        doc_type = "amazon_returns"
    elif "Voice of the Customer" in text:
        data = extract_voc_data(text)
        doc_type = "voice_of_customer"
    elif "Customer reviews" in text or "star rating" in text:
        data = extract_amazon_reviews_data(text)
        doc_type = "amazon_reviews"
    else:
        # Generic text extraction if we can't identify a specific format
        data = {"raw_text": text}
        doc_type = "generic_text"
    
    return {
        "type": doc_type,
        "data": data,
        "raw_text": text
    }
