"""
Enhanced Upload Handler Module for Amazon Medical Device Listing Optimizer

This module provides robust, production-ready upload functionality for:
- Excel/CSV structured data
- Manual data entry and validation
- Image/PDF processing with OCR and AI Vision
- Data validation and error handling
- Template generation

Author: Assistant
Version: 2.0
"""

import io
import os
import re
import base64
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check available modules
MODULES_AVAILABLE = {
    'xlsxwriter': False,
    'openpyxl': False,
    'pillow': False,
    'pytesseract': False,
    'pdf2image': False,
    'requests': False
}

# Try importing optional modules
try:
    import xlsxwriter
    MODULES_AVAILABLE['xlsxwriter'] = True
    logger.info("XlsxWriter available for Excel formatting")
except ImportError:
    logger.warning("XlsxWriter not available")

try:
    import openpyxl
    MODULES_AVAILABLE['openpyxl'] = True
    logger.info("Openpyxl available for Excel reading")
except ImportError:
    logger.warning("Openpyxl not available")

try:
    from PIL import Image
    MODULES_AVAILABLE['pillow'] = True
except ImportError:
    logger.warning("Pillow not available")

try:
    import pytesseract
    MODULES_AVAILABLE['pytesseract'] = True
except ImportError:
    logger.warning("Pytesseract not available")

try:
    from pdf2image import convert_from_bytes
    MODULES_AVAILABLE['pdf2image'] = True
except ImportError:
    logger.warning("pdf2image not available")

try:
    import requests
    MODULES_AVAILABLE['requests'] = True
except ImportError:
    logger.warning("Requests not available")

# Constants
REQUIRED_COLUMNS = ['ASIN', 'Last 30 Days Sales', 'Last 30 Days Returns']
OPTIONAL_COLUMNS = [
    'SKU', 'Product Name', 'Category', 'Product Description', 'Listing URL',
    'Last 365 Days Sales', 'Last 365 Days Returns', 'Star Rating', 
    'Total Reviews', 'Average Price', 'Cost per Unit', 'Profit Margin'
]

MEDICAL_DEVICE_CATEGORIES = [
    "Mobility Aids", "Bathroom Safety", "Pain Relief", "Sleep & Comfort", 
    "Fitness & Recovery", "Daily Living Aids", "Respiratory Care",
    "Blood Pressure Monitors", "Diabetes Care", "Orthopedic Support",
    "First Aid", "Wound Care", "Compression Wear", "Exercise Equipment",
    "Home Diagnostics", "Therapy & Rehabilitation", "Other"
]

MAX_FILE_SIZE_MB = 50
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
SUPPORTED_DOC_FORMATS = ['.pdf']
SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.xls']

class UploadError(Exception):
    """Custom exception for upload-related errors"""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class FileProcessor:
    """Handles file processing and validation"""
    
    @staticmethod
    def validate_file_size(file_data: bytes, max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
        """Validate file size"""
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise UploadError(f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)")
        return True
    
    @staticmethod
    def validate_file_format(filename: str, allowed_formats: List[str]) -> bool:
        """Validate file format"""
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_formats:
            raise UploadError(f"Unsupported file format: {file_ext}. Allowed formats: {', '.join(allowed_formats)}")
        return True
    
    @staticmethod
    def detect_encoding(file_data: bytes) -> str:
        """Detect file encoding for CSV files"""
        try:
            # Try common encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file_data.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'  # Default fallback
        except Exception:
            return 'utf-8'

class DataValidator:
    """Handles data validation and cleaning"""
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that all required columns are present"""
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        return len(missing_cols) == 0, missing_cols
    
    @staticmethod
    def validate_asin_format(asin: str) -> bool:
        """Validate ASIN format (10 characters, alphanumeric)"""
        if not isinstance(asin, str):
            return False
        asin = str(asin).strip()
        return len(asin) == 10 and asin.isalnum()
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame) -> Dict[str, List[int]]:
        """Validate numeric columns and return problematic rows"""
        numeric_cols = ['Last 30 Days Sales', 'Last 30 Days Returns', 
                       'Last 365 Days Sales', 'Last 365 Days Returns',
                       'Star Rating', 'Total Reviews', 'Average Price']
        
        errors = {}
        for col in numeric_cols:
            if col in df.columns:
                # Find non-numeric values
                problematic_rows = []
                for idx, value in enumerate(df[col]):
                    if pd.notna(value):
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            problematic_rows.append(idx + 2)  # +2 for Excel row number (header + 0-index)
                
                if problematic_rows:
                    errors[col] = problematic_rows
        
        return errors
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataframe"""
        df_clean = df.copy()
        
        # Strip whitespace from string columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace('nan', np.nan)
        
        # Convert numeric columns
        numeric_cols = ['Last 30 Days Sales', 'Last 30 Days Returns', 
                       'Last 365 Days Sales', 'Last 365 Days Returns',
                       'Star Rating', 'Total Reviews', 'Average Price']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Validate ASINs
        if 'ASIN' in df_clean.columns:
            df_clean['ASIN'] = df_clean['ASIN'].astype(str).str.strip()
        
        return df_clean
    
    @staticmethod
    def validate_business_logic(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate business logic rules"""
        warnings = []
        
        for idx, row in df.iterrows():
            excel_row = idx + 2  # Excel row number
            
            # Sales should be positive
            if 'Last 30 Days Sales' in df.columns and row['Last 30 Days Sales'] <= 0:
                warnings.append({
                    'row': excel_row,
                    'type': 'warning',
                    'message': 'Sales should be greater than 0'
                })
            
            # Returns should not exceed sales
            if ('Last 30 Days Sales' in df.columns and 'Last 30 Days Returns' in df.columns):
                if pd.notna(row['Last 30 Days Sales']) and pd.notna(row['Last 30 Days Returns']):
                    if row['Last 30 Days Returns'] > row['Last 30 Days Sales']:
                        warnings.append({
                            'row': excel_row,
                            'type': 'error',
                            'message': 'Returns cannot exceed sales'
                        })
            
            # Star rating should be between 1 and 5
            if 'Star Rating' in df.columns and pd.notna(row['Star Rating']):
                if not (1 <= row['Star Rating'] <= 5):
                    warnings.append({
                        'row': excel_row,
                        'type': 'warning',
                        'message': 'Star rating should be between 1 and 5'
                    })
            
            # ASIN format validation
            if 'ASIN' in df.columns:
                if not DataValidator.validate_asin_format(row['ASIN']):
                    warnings.append({
                        'row': excel_row,
                        'type': 'error',
                        'message': f'Invalid ASIN format: {row["ASIN"]}'
                    })
        
        return warnings

class StructuredDataUploader:
    """Handles structured data uploads (CSV, Excel)"""
    
    @staticmethod
    def process_csv_file(file_data: bytes, filename: str) -> Dict[str, Any]:
        """Process CSV file upload"""
        try:
            # Validate file
            FileProcessor.validate_file_size(file_data)
            FileProcessor.validate_file_format(filename, ['.csv'])
            
            # Detect encoding
            encoding = FileProcessor.detect_encoding(file_data)
            
            # Read CSV
            try:
                df = pd.read_csv(io.BytesIO(file_data), encoding=encoding)
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 fails
                df = pd.read_csv(io.BytesIO(file_data), encoding='latin-1')
            
            return StructuredDataUploader._process_dataframe(df, filename)
            
        except Exception as e:
            logger.error(f"Error processing CSV file {filename}: {str(e)}")
            raise UploadError(f"Failed to process CSV file: {str(e)}")
    
    @staticmethod
    def process_excel_file(file_data: bytes, filename: str) -> Dict[str, Any]:
        """Process Excel file upload"""
        try:
            # Validate file
            FileProcessor.validate_file_size(file_data)
            FileProcessor.validate_file_format(filename, ['.xlsx', '.xls'])
            
            # Read Excel file
            try:
                # Try with openpyxl first for .xlsx
                if filename.lower().endswith('.xlsx') and MODULES_AVAILABLE['openpyxl']:
                    df = pd.read_excel(io.BytesIO(file_data), engine='openpyxl')
                else:
                    # Fallback to default engine
                    df = pd.read_excel(io.BytesIO(file_data))
            except Exception as e:
                logger.error(f"Failed to read Excel file with primary method: {str(e)}")
                # Try alternative approach
                df = pd.read_excel(io.BytesIO(file_data), engine=None)
            
            return StructuredDataUploader._process_dataframe(df, filename)
            
        except Exception as e:
            logger.error(f"Error processing Excel file {filename}: {str(e)}")
            raise UploadError(f"Failed to process Excel file: {str(e)}")
    
    @staticmethod
    def _process_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Process and validate dataframe"""
        result = {
            'success': False,
            'filename': filename,
            'data': None,
            'warnings': [],
            'errors': [],
            'summary': {}
        }
        
        try:
            # Basic validation
            if df.empty:
                raise DataValidationError("File is empty")
            
            # Check required columns
            has_required, missing_cols = DataValidator.validate_required_columns(df)
            if not has_required:
                raise DataValidationError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Clean the dataframe
            df_clean = DataValidator.clean_dataframe(df)
            
            # Validate numeric columns
            numeric_errors = DataValidator.validate_numeric_columns(df_clean)
            for col, rows in numeric_errors.items():
                result['errors'].append({
                    'type': 'validation',
                    'message': f"Non-numeric values in column '{col}' at rows: {', '.join(map(str, rows))}"
                })
            
            # Validate business logic
            business_warnings = DataValidator.validate_business_logic(df_clean)
            result['warnings'].extend(business_warnings)
            
            # Separate errors and warnings
            errors_only = [w for w in business_warnings if w['type'] == 'error']
            warnings_only = [w for w in business_warnings if w['type'] == 'warning']
            
            # If there are critical errors, don't process
            if numeric_errors or errors_only:
                result['errors'].extend([{'type': 'business_logic', 'message': err['message'], 'row': err['row']} 
                                       for err in errors_only])
                return result
            
            # Calculate metrics for summary
            total_rows = len(df_clean)
            valid_asins = df_clean['ASIN'].notna().sum()
            avg_return_rate = 0
            
            if 'Last 30 Days Sales' in df_clean.columns and 'Last 30 Days Returns' in df_clean.columns:
                sales_col = df_clean['Last 30 Days Sales']
                returns_col = df_clean['Last 30 Days Returns']
                valid_rows = (sales_col > 0) & (returns_col >= 0)
                if valid_rows.any():
                    return_rates = (returns_col[valid_rows] / sales_col[valid_rows]) * 100
                    avg_return_rate = return_rates.mean()
            
            result.update({
                'success': True,
                'data': df_clean,
                'summary': {
                    'total_products': total_rows,
                    'valid_asins': valid_asins,
                    'average_return_rate': round(avg_return_rate, 2),
                    'columns_imported': list(df_clean.columns),
                    'optional_columns_present': [col for col in OPTIONAL_COLUMNS if col in df_clean.columns]
                }
            })
            
            return result
            
        except Exception as e:
            result['errors'].append({
                'type': 'processing',
                'message': str(e)
            })
            return result

class ManualDataEntry:
    """Handles manual data entry and validation"""
    
    @staticmethod
    def validate_manual_entry(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate manually entered data"""
        result = {
            'success': False,
            'data': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check required fields
            required_fields = ['asin', 'product_name', 'category', 'sales_30d', 'returns_30d']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                result['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
                return result
            
            # Validate ASIN
            if not DataValidator.validate_asin_format(data['asin']):
                result['errors'].append(f"Invalid ASIN format: {data['asin']}")
            
            # Validate numeric fields
            numeric_fields = {
                'sales_30d': 'Last 30 Days Sales',
                'returns_30d': 'Last 30 Days Returns',
                'sales_365d': 'Last 365 Days Sales',
                'returns_365d': 'Last 365 Days Returns',
                'star_rating': 'Star Rating',
                'total_reviews': 'Total Reviews',
                'average_price': 'Average Price'
            }
            
            for field, display_name in numeric_fields.items():
                if field in data and data[field] is not None:
                    try:
                        value = float(data[field])
                        if field in ['sales_30d', 'returns_30d', 'sales_365d', 'returns_365d'] and value < 0:
                            result['errors'].append(f"{display_name} cannot be negative")
                        elif field == 'star_rating' and not (1 <= value <= 5):
                            result['warnings'].append(f"{display_name} should be between 1 and 5")
                    except (ValueError, TypeError):
                        result['errors'].append(f"{display_name} must be a valid number")
            
            # Business logic validation
            if data.get('returns_30d', 0) > data.get('sales_30d', 0):
                result['errors'].append("30-day returns cannot exceed 30-day sales")
            
            if data.get('returns_365d', 0) > data.get('sales_365d', 0):
                result['warnings'].append("365-day returns exceed 365-day sales")
            
            # If no errors, prepare the data
            if not result['errors']:
                clean_data = ManualDataEntry._clean_manual_data(data)
                result.update({
                    'success': True,
                    'data': clean_data
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating manual entry: {str(e)}")
            result['errors'].append(f"Validation error: {str(e)}")
            return result
    
    @staticmethod
    def _clean_manual_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize manual data"""
        clean_data = {}
        
        # String fields
        string_fields = ['asin', 'sku', 'product_name', 'category', 'description', 'listing_url']
        for field in string_fields:
            if field in data and data[field]:
                clean_data[field] = str(data[field]).strip()
        
        # Numeric fields
        numeric_fields = ['sales_30d', 'returns_30d', 'sales_365d', 'returns_365d', 
                         'star_rating', 'total_reviews', 'average_price']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    clean_data[field] = float(data[field])
                except (ValueError, TypeError):
                    pass  # Skip invalid numeric values
        
        return clean_data
    
    @staticmethod
    def parse_reviews_text(reviews_text: str, asin: str) -> List[Dict[str, Any]]:
        """Parse manually entered reviews text"""
        reviews = []
        if not reviews_text.strip():
            return reviews
        
        lines = reviews_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for pattern: "Rating - Review text" or "Rating: Review text"
            for separator in [' - ', ': ', ' – ', ' — ']:
                if separator in line:
                    try:
                        rating_part, review_text = line.split(separator, 1)
                        rating = int(rating_part.strip())
                        if 1 <= rating <= 5:
                            reviews.append({
                                'rating': rating,
                                'review_text': review_text.strip(),
                                'asin': asin,
                                'source': 'manual_entry'
                            })
                            break
                    except (ValueError, IndexError):
                        continue
        
        return reviews
    
    @staticmethod
    def parse_returns_text(returns_text: str, asin: str) -> List[Dict[str, Any]]:
        """Parse manually entered return reasons text"""
        returns = []
        if not returns_text.strip():
            return returns
        
        lines = returns_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                returns.append({
                    'return_reason': line,
                    'asin': asin,
                    'source': 'manual_entry'
                })
        
        return returns

class ImageDocumentProcessor:
    """Handles image and document processing with OCR and AI Vision"""
    
    @staticmethod
    def process_image_file(file_data: bytes, filename: str, content_type: str, 
                          asin: Optional[str] = None) -> Dict[str, Any]:
        """Process image file with OCR or AI Vision"""
        result = {
            'success': False,
            'filename': filename,
            'content_type': content_type,
            'asin': asin,
            'text': '',
            'processing_method': None,
            'structured_data': None,
            'errors': []
        }
        
        try:
            # Validate file
            FileProcessor.validate_file_size(file_data)
            FileProcessor.validate_file_format(filename, SUPPORTED_IMAGE_FORMATS)
            
            # Try OCR first if available
            if MODULES_AVAILABLE['pytesseract'] and MODULES_AVAILABLE['pillow']:
                try:
                    text = ImageDocumentProcessor._process_with_ocr(file_data)
                    if text and len(text.strip()) > 10:  # Minimum text threshold
                        result.update({
                            'success': True,
                            'text': text,
                            'processing_method': 'OCR'
                        })
                        
                        # Extract structured data based on content type
                        structured_data = ImageDocumentProcessor._extract_structured_data(text, content_type)
                        if structured_data:
                            result['structured_data'] = structured_data
                        
                        return result
                except Exception as e:
                    logger.warning(f"OCR processing failed: {str(e)}")
                    result['errors'].append(f"OCR failed: {str(e)}")
            
            # Fallback to placeholder for AI Vision (would need API integration)
            result.update({
                'success': False,
                'text': 'AI Vision processing not available in this module',
                'processing_method': 'None',
                'errors': ['Neither OCR nor AI Vision processing succeeded']
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}")
            result['errors'].append(str(e))
            return result
    
    @staticmethod
    def process_pdf_file(file_data: bytes, filename: str, content_type: str,
                        asin: Optional[str] = None) -> Dict[str, Any]:
        """Process PDF file with OCR or AI Vision fallback"""
        result = {
            'success': False,
            'filename': filename,
            'content_type': content_type,
            'asin': asin,
            'text': '',
            'processing_method': None,
            'structured_data': None,
            'errors': []
        }
        
        try:
            # Validate file
            FileProcessor.validate_file_size(file_data)
            FileProcessor.validate_file_format(filename, SUPPORTED_DOC_FORMATS)
            
            # Try PDF OCR only if all required modules are available
            ocr_success = False
            if (MODULES_AVAILABLE['pdf2image'] and 
                MODULES_AVAILABLE['pytesseract'] and 
                MODULES_AVAILABLE['pillow']):
                try:
                    text = ImageDocumentProcessor._process_pdf_with_ocr(file_data)
                    if text and len(text.strip()) > 10:
                        result.update({
                            'success': True,
                            'text': text,
                            'processing_method': 'PDF OCR'
                        })
                        ocr_success = True
                except Exception as e:
                    logger.warning(f"PDF OCR processing failed: {str(e)}")
                    result['errors'].append(f"PDF OCR failed: {str(e)}")
            else:
                # OCR modules not available, skip OCR attempt
                result['errors'].append("PDF OCR modules not available in cloud environment")
            
            # Use AI Vision for PDF if OCR failed or unavailable
            if not ocr_success:
                logger.info(f"Falling back to AI Vision for PDF: {filename}")
                
                # For PDFs, we'll treat them as images and let AI Vision handle them
                ai_vision_result = ImageDocumentProcessor._process_pdf_with_ai_vision(
                    file_data, filename, content_type
                )
                
                if ai_vision_result['success']:
                    result.update({
                        'success': True,
                        'text': ai_vision_result['text'],
                        'processing_method': 'AI Vision (PDF)',
                        'structured_data': ai_vision_result.get('structured_data')
                    })
                else:
                    result['errors'].extend(ai_vision_result.get('errors', []))
            
            # Extract structured data if we have text
            if result['success'] and result['text']:
                if not result.get('structured_data'):
                    structured_data = ImageDocumentProcessor._extract_structured_data(
                        result['text'], content_type
                    )
                    if structured_data:
                        result['structured_data'] = structured_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            result['errors'].append(str(e))
            return result
    
    @staticmethod
    def _process_pdf_with_ai_vision(file_data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """Process PDF using AI Vision API directly"""
        
        try:
            # Get API key
            api_key = None
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                    api_key = st.secrets['openai_api_key']
            except:
                pass
            
            if not api_key:
                import os
                api_key = os.environ.get('OPENAI_API_KEY')
            
            if not api_key:
                return {
                    'success': False,
                    'errors': ['OpenAI API key not configured for PDF processing'],
                    'text': 'AI Vision PDF analysis requires API key configuration'
                }
            
            # Encode PDF to base64 for AI Vision
            import base64
            base64_pdf = base64.b64encode(file_data).decode('utf-8')
            
            # Create PDF-specific prompt
            if content_type == "Product Reviews":
                prompt = """Analyze this PDF document containing Amazon product review data and extract:

1. ASIN (Amazon product identifier - format B0XXXXXXXXX)
2. Product name and details
3. All customer reviews with star ratings and review text
4. Overall product statistics (average rating, total reviews)
5. Product price and other details if visible

Please provide a comprehensive analysis in this format:
ASIN: [detected ASIN or "Not found"]
PRODUCT_NAME: [full product title]
OVERALL_RATING: [X.X out of 5 stars]
TOTAL_REVIEWS: [number of reviews]

REVIEWS:
[For each review found:]
RATING: [X] stars
REVIEW: [exact review text]

QUANTITATIVE_ANALYSIS:
- Rating Distribution: [breakdown by star rating]
- Common Themes: [categorize reviews into themes]
- Sentiment Analysis: [overall positive/negative patterns]

Provide specific insights and recommendations based on the review patterns found."""
                
            elif content_type == "Return Reports":
                prompt = """Analyze this PDF document containing Amazon return report data and extract:

1. ASIN (Amazon product identifier)
2. Product information
3. All return reasons and quantities
4. Return patterns and categories
5. Timeframes and return trends if visible

Format your response as:
ASIN: [detected ASIN or "Not found"]
PRODUCT_NAME: [product name]
TOTAL_RETURNS: [number if visible]

RETURN_REASONS:
[List each return reason found]

CATEGORIZED_ANALYSIS:
- Size/Fit Issues: [count and percentage]
- Quality/Defect Issues: [count and percentage]
- Functionality Issues: [count and percentage]
- Shipping/Packaging Issues: [count and percentage]
- Customer Expectation Issues: [count and percentage]

RECOMMENDATIONS:
[Specific actionable recommendations to reduce returns]

Provide quantitative analysis with percentages and patterns."""
                
            else:  # General PDF analysis
                prompt = f"""Analyze this PDF document related to Amazon {content_type.lower()} and extract all relevant product information, data, and insights. Focus on:

1. Product identifiers (ASINs)
2. Performance metrics and data
3. Customer feedback and patterns
4. Actionable business insights
5. Quantitative analysis where possible

Provide structured analysis with specific recommendations."""
            
            # Make API call to GPT-4o
            import requests
            import json
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Note: GPT-4o can process PDFs directly
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{base64_pdf}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2500,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60  # Longer timeout for PDF processing
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                # Parse the structured response
                structured_data = ImageDocumentProcessor._parse_ai_vision_response(
                    ai_response, content_type
                )
                
                return {
                    'success': True,
                    'text': ai_response,
                    'structured_data': structured_data,
                    'processing_method': 'AI Vision (PDF)'
                }
            else:
                error_msg = f"AI Vision PDF processing error: {response.status_code}"
                logger.error(f"{error_msg}: {response.text}")
                return {
                    'success': False,
                    'errors': [error_msg],
                    'text': f'PDF analysis failed: {error_msg}'
                }
                
        except Exception as e:
            logger.error(f"AI Vision PDF processing error: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'text': f'PDF processing failed: {str(e)}'
            }
    
    @staticmethod
    def _process_with_ocr(image_data: bytes) -> str:
        """Process image with OCR"""
        if not MODULES_AVAILABLE['pytesseract'] or not MODULES_AVAILABLE['pillow']:
            raise Exception("OCR modules not available")
        
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text with pytesseract
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            raise Exception(f"OCR failed: {str(e)}")
    
    @staticmethod
    def _process_pdf_with_ocr(pdf_data: bytes) -> str:
        """Process PDF with OCR"""
        if not MODULES_AVAILABLE['pdf2image'] or not MODULES_AVAILABLE['pytesseract']:
            raise Exception("PDF processing modules not available")
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_data)
            
            all_text = []
            for i, image in enumerate(images):
                try:
                    # Extract text from each page
                    text = pytesseract.image_to_string(image, config='--psm 6')
                    if text.strip():
                        all_text.append(f"Page {i+1}:\n{text.strip()}")
                except Exception as e:
                    logger.warning(f"Failed to process PDF page {i+1}: {str(e)}")
            
            return '\n\n'.join(all_text)
            
        except Exception as e:
            logger.error(f"PDF OCR processing error: {str(e)}")
            raise Exception(f"PDF OCR failed: {str(e)}")
    
    @staticmethod
    def _extract_structured_data(text: str, content_type: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from OCR text based on content type"""
        try:
            result = {}
            
            # Always try to extract ASINs and product info first
            asins = ImageDocumentProcessor._extract_asins(text)
            product_info = ImageDocumentProcessor._extract_product_info(text)
            
            if asins:
                result['detected_asins'] = asins
                result['primary_asin'] = asins[0]  # Use first detected ASIN as primary
            
            if product_info:
                result['product_info'] = product_info
            
            # Then extract content-specific data
            if content_type == "Product Reviews":
                reviews = ImageDocumentProcessor._extract_reviews_from_text(text)
                result.update(reviews)
            elif content_type == "Return Reports":
                returns = ImageDocumentProcessor._extract_returns_from_text(text)
                result.update(returns)
            elif content_type == "Product Listing":
                listing = ImageDocumentProcessor._extract_listing_from_text(text)
                result.update(listing)
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return None
    
    @staticmethod
    def _extract_asins(text: str) -> List[str]:
        """Extract Amazon ASINs from text"""
        # ASIN pattern: B + 9 alphanumeric characters
        asin_pattern = r'B[0-9A-Z]{9}'
        asins = re.findall(asin_pattern, text.upper())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_asins = []
        for asin in asins:
            if asin not in seen:
                seen.add(asin)
                unique_asins.append(asin)
        
        return unique_asins
    
    @staticmethod
    def _extract_product_info(text: str) -> Dict[str, Any]:
        """Extract product information from text"""
        product_info = {}
        
        # Look for price patterns
        price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # $29.99
            r'Price[:\s]*\$(\d+(?:\.\d{2})?)',  # Price: $29.99
            r'(\d+(?:\.\d{2})?)\s*dollars?'  # 29.99 dollars
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    product_info['detected_price'] = float(matches[0])
                    break
                except ValueError:
                    continue
        
        # Look for star ratings
        rating_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*5\s*star',
            r'(\d+(?:\.\d+)?)\s*star',
            r'Rating[:\s]*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in rating_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    rating = float(matches[0])
                    if 1 <= rating <= 5:
                        product_info['detected_rating'] = rating
                        break
                except ValueError:
                    continue
        
        # Look for review counts
        review_patterns = [
            r'(\d+(?:,\d+)*)\s*(?:customer\s*)?reviews?',
            r'(\d+(?:,\d+)*)\s*ratings?',
            r'Based on (\d+(?:,\d+)*)\s*reviews?'
        ]
        
        for pattern in review_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    review_count = int(matches[0].replace(',', ''))
                    product_info['detected_review_count'] = review_count
                    break
                except ValueError:
                    continue
        
        # Look for product titles (usually longer lines near the top)
        lines = text.split('\n')
        potential_titles = []
        
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            # Skip lines that look like navigation, prices, ratings, etc.
            if (len(line) > 20 and len(line) < 150 and 
                not re.search(r'[\$\d+\.\d+]', line) and
                not re.search(r'\d+\s*star', line, re.IGNORECASE) and
                not line.lower().startswith(('add to', 'buy now', 'price', 'rating', 'customer'))):
                potential_titles.append(line)
        
        if potential_titles:
            product_info['detected_title'] = potential_titles[0]
        
        # Look for brand names (common medical device brands)
        medical_brands = [
            'drive medical', 'vive', 'carex', 'medline', 'invacare', 'lumex', 
            'graham field', 'nova', 'cardinal health', 'mckesson', 'dmi',
            'essential medical', 'compass health', 'mobility', 'healthcare'
        ]
        
        for brand in medical_brands:
            if brand.lower() in text.lower():
                product_info['detected_brand'] = brand.title()
                break
        
        return product_info
    
    @staticmethod
    def _extract_reviews_from_text(text: str) -> Dict[str, Any]:
        """Extract review data from OCR text"""
        reviews = []
        lines = text.split('\n')
        
        # Look for star ratings and review text patterns
        star_pattern = r'(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*5\s*star'
        rating_pattern = r'(\d+)\s*star'
        
        current_rating = None
        current_review = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for star ratings
            star_match = re.search(star_pattern, line, re.IGNORECASE)
            rating_match = re.search(rating_pattern, line, re.IGNORECASE)
            
            if star_match:
                current_rating = float(star_match.group(1))
                current_review = []
            elif rating_match:
                current_rating = int(rating_match.group(1))
                current_review = []
            elif current_rating is not None:
                # This might be review text
                if len(line) > 10:  # Minimum length for review text
                    current_review.append(line)
                    
                    # If we have accumulated enough text, save the review
                    if len(' '.join(current_review)) > 20:
                        reviews.append({
                            'rating': current_rating,
                            'review_text': ' '.join(current_review),
                            'source': 'ocr_extraction'
                        })
                        current_rating = None
                        current_review = []
        
        return {'reviews': reviews, 'total_extracted': len(reviews)}
    
    @staticmethod
    def _extract_returns_from_text(text: str) -> Dict[str, Any]:
        """Extract return reason data from OCR text"""
        returns = []
        lines = text.split('\n')
        
        # Look for common return reason patterns
        return_keywords = ['return', 'refund', 'exchange', 'reason', 'defective', 'broken', 'wrong']
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Minimum length
                # Check if line contains return-related keywords or seems like a reason
                if any(keyword in line.lower() for keyword in return_keywords) or len(line) > 30:
                    # Clean up the line
                    clean_line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
                    if clean_line and len(clean_line) > 5:
                        returns.append({
                            'return_reason': clean_line,
                            'source': 'ocr_extraction'
                        })
        
        return {'returns': returns, 'total_extracted': len(returns)}
    
    @staticmethod
    def _extract_listing_from_text(text: str) -> Dict[str, Any]:
        """Extract product listing information from OCR text"""
        result = {'title': '', 'bullets': [], 'description': '', 'price': '', 'rating': ''}
        
        lines = text.split('\n')
        
        # Simple extraction logic (would need refinement based on actual Amazon layout)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for price patterns
            price_match = re.search(r'\$(\d+(?:\.\d{2})?)', line)
            if price_match:
                result['price'] = price_match.group(0)
            
            # Look for star ratings
            rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*5', line)
            if rating_match:
                result['rating'] = rating_match.group(1)
            
            # Title is usually one of the first longer lines
            if not result['title'] and len(line) > 20 and not any(char in line for char in ['$', '★', '•']):
                result['title'] = line
            
            # Bullet points often start with bullets or are formatted lists
            if re.match(r'^[-•*]\s*', line) and len(line) > 10:
                result['bullets'].append(re.sub(r'^[-•*]\s*', '', line))
        
        return result

class TemplateGenerator:
    """Generates import templates and examples"""
    
    @staticmethod
    def create_enhanced_template() -> bytes:
        """Create an enhanced import template with examples and validation"""
        
        # Sample data with more realistic examples
        data = {
            "ASIN*": ["B0DT7NW5VY", "B0DT8XYZ123", "B08CK7MN45", "B0EXAMPLE1", "B0EXAMPLE2"],
            "SKU": ["VH-TRI-001", "VH-SHW-234", "VH-CUS-352", "VH-MOB-456", "VH-SUP-789"],
            "Product Name": [
                "Vive Tri-Rollator with Seat and Storage",
                "Premium Shower Chair with Back Support", 
                "Memory Foam Seat Cushion",
                "4-Wheel Mobility Scooter",
                "Compression Knee Support Brace"
            ],
            "Category": [
                "Mobility Aids", "Bathroom Safety", "Comfort Products", 
                "Mobility Aids", "Orthopedic Support"
            ],
            "Last 30 Days Sales*": [491, 325, 278, 156, 623],
            "Last 30 Days Returns*": [12, 8, 5, 18, 31],
            "Last 365 Days Sales": [5840, 3900, 2950, 1890, 7250],
            "Last 365 Days Returns": [145, 95, 78, 234, 298],
            "Star Rating": [4.2, 4.5, 4.1, 3.8, 4.6],
            "Total Reviews": [287, 156, 423, 89, 501],
            "Average Price": [129.99, 79.99, 39.99, 899.99, 24.99],
            "Cost per Unit": [65.00, 35.00, 18.00, 450.00, 12.50],
            "Profit Margin": [50.0, 56.3, 55.0, 50.0, 50.0]
        }
        
        df = pd.DataFrame(data)
        
        output = io.BytesIO()
        
        if MODULES_AVAILABLE['xlsxwriter']:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Product Data Template', index=False)
                
                workbook = writer.book
                worksheet = writer.sheets['Product Data Template']
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True, 'text_wrap': True, 'valign': 'top',
                    'fg_color': '#2E5A87', 'font_color': 'white', 'border': 1
                })
                
                required_format = workbook.add_format({
                    'fg_color': '#FFE4E1', 'border': 1
                })
                
                optional_format = workbook.add_format({
                    'fg_color': '#F0F8F0', 'border': 1
                })
                
                money_format = workbook.add_format({
                    'num_format': '$#,##0.00', 'border': 1
                })
                
                percent_format = workbook.add_format({
                    'num_format': '0.0%', 'border': 1
                })
                
                # Apply header formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Apply data formatting
                for row_num in range(1, len(df) + 1):
                    for col_num, col_name in enumerate(df.columns):
                        cell_value = df.iloc[row_num-1, col_num]
                        
                        if col_name in REQUIRED_COLUMNS:
                            cell_format = required_format
                        else:
                            cell_format = optional_format
                        
                        if col_name in ['Average Price', 'Cost per Unit']:
                            cell_format = money_format
                        elif col_name == 'Profit Margin':
                            cell_format = percent_format
                            cell_value = cell_value / 100  # Convert to decimal for percentage format
                        
                        worksheet.write(row_num, col_num, cell_value, cell_format)
                
                # Add instructions worksheet
                instructions_data = {
                    'Field Name': list(df.columns),
                    'Required?': ['Yes' if col in REQUIRED_COLUMNS else 'No' for col in df.columns],
                    'Description': [
                        'Amazon Standard Identification Number (10 characters)',
                        'Your internal Stock Keeping Unit code',
                        'Full product name as shown on Amazon',
                        'Product category for classification',
                        'Total units sold in the last 30 days',
                        'Total units returned in the last 30 days',
                        'Total units sold in the last 365 days',
                        'Total units returned in the last 365 days',
                        'Average customer rating (1-5 stars)',
                        'Total number of customer reviews',
                        'Average selling price on Amazon',
                        'Your cost to produce/acquire the product',
                        'Profit margin percentage'
                    ]
                }
                
                instructions_df = pd.DataFrame(instructions_data)
                instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
                
                inst_worksheet = writer.sheets['Instructions']
                
                # Format instructions sheet
                for col_num, value in enumerate(instructions_df.columns.values):
                    inst_worksheet.write(0, col_num, value, header_format)
                
                # Set column widths
                inst_worksheet.set_column('A:A', 25)
                inst_worksheet.set_column('B:B', 12)
                inst_worksheet.set_column('C:C', 60)
                
                # Set main sheet column widths
                worksheet.set_column('A:B', 12)
                worksheet.set_column('C:C', 30)
                worksheet.set_column('D:D', 20)
                worksheet.set_column('E:L', 15)
                
                # Add legend
                legend_row = len(df) + 3
                worksheet.write(legend_row, 0, "Legend:", header_format)
                worksheet.write(legend_row + 1, 0, "Required fields", required_format)
                worksheet.write(legend_row + 2, 0, "Optional fields", optional_format)
                
        else:
            # Basic Excel without formatting
            df.to_excel(output, sheet_name='Product Data Template', index=False)
        
        output.seek(0)
        return output.getvalue()

# Main upload handler class that coordinates all upload types
class UploadHandler:
    """Main upload handler that coordinates all upload functionality"""
    
    def __init__(self):
        self.structured_uploader = StructuredDataUploader()
        self.manual_entry = ManualDataEntry()
        self.image_processor = ImageDocumentProcessor()
        self.template_generator = TemplateGenerator()
    
    def get_available_modules(self) -> Dict[str, bool]:
        """Get status of available modules"""
        return MODULES_AVAILABLE.copy()
    
    def create_template(self) -> bytes:
        """Create download template"""
        return self.template_generator.create_enhanced_template()
    
    def process_structured_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Process structured data file (CSV or Excel)"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.csv':
                return self.structured_uploader.process_csv_file(file_data, filename)
            elif file_ext in ['.xlsx', '.xls']:
                return self.structured_uploader.process_excel_file(file_data, filename)
            else:
                raise UploadError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing structured file {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    def process_manual_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process manual data entry"""
        return self.manual_entry.validate_manual_entry(data)
    
    def parse_manual_reviews(self, reviews_text: str, asin: str) -> List[Dict[str, Any]]:
        """Parse manual review entries"""
        return self.manual_entry.parse_reviews_text(reviews_text, asin)
    
    def parse_manual_returns(self, returns_text: str, asin: str) -> List[Dict[str, Any]]:
        """Parse manual return entries"""
        return self.manual_entry.parse_returns_text(returns_text, asin)
    
    def process_image_document(self, file_data: bytes, filename: str, 
                             content_type: str, asin: Optional[str] = None) -> Dict[str, Any]:
        """Process image or document file"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in SUPPORTED_IMAGE_FORMATS:
                return self.image_processor.process_image_file(file_data, filename, content_type, asin)
            elif file_ext in SUPPORTED_DOC_FORMATS:
                return self.image_processor.process_pdf_file(file_data, filename, content_type, asin)
            else:
                raise UploadError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing image/document {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }

# Export the main class and exceptions
__all__ = ['UploadHandler', 'UploadError', 'DataValidationError', 'MEDICAL_DEVICE_CATEGORIES']
