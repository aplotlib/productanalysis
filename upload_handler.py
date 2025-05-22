"""
Enhanced Upload Handler Module for Medical Device Customer Feedback Analyzer

**ENHANCED FOR TEXT ANALYSIS WORKFLOW**

This module provides comprehensive upload functionality optimized for customer feedback
text analysis and quality management, with enhanced support for:
✓ Date range filtering and temporal data processing
✓ Medical device-specific feedback categorization
✓ Quality management compliance (ISO 13485)
✓ Customer feedback export processing (Sellerboard/Amazon)
✓ AI-powered document extraction and text analysis
✓ Enhanced data validation for quality management

Author: Assistant
Version: 3.0 - Text Analysis Enhanced
Compliance: ISO 13485 Quality Management Aware
"""

import io
import os
import re
import base64
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import json

# Configure logging for quality management traceability
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check available modules for enhanced functionality
MODULES_AVAILABLE = {
    'xlsxwriter': False,
    'openpyxl': False,
    'pillow': False,
    'pytesseract': False,
    'pdf2image': False,
    'requests': False
}

# Safe module imports
try:
    import xlsxwriter
    MODULES_AVAILABLE['xlsxwriter'] = True
    logger.info("XlsxWriter available for enhanced Excel formatting")
except ImportError:
    logger.warning("XlsxWriter not available - using basic Excel functionality")

try:
    import openpyxl
    MODULES_AVAILABLE['openpyxl'] = True
    logger.info("Openpyxl available for Excel reading")
except ImportError:
    logger.warning("Openpyxl not available - limited Excel support")

try:
    from PIL import Image
    MODULES_AVAILABLE['pillow'] = True
except ImportError:
    logger.warning("Pillow not available - image processing disabled")

try:
    import pytesseract
    MODULES_AVAILABLE['pytesseract'] = True
except ImportError:
    logger.warning("Pytesseract not available - OCR processing disabled")

try:
    from pdf2image import convert_from_bytes
    MODULES_AVAILABLE['pdf2image'] = True
except ImportError:
    logger.warning("pdf2image not available - PDF OCR disabled")

try:
    import requests
    MODULES_AVAILABLE['requests'] = True
except ImportError:
    logger.warning("Requests not available - AI processing may be limited")

# Enhanced constants for text analysis workflow
REQUIRED_COLUMNS = ['ASIN', 'Product Name']  # Simplified for text analysis focus
OPTIONAL_PRODUCT_COLUMNS = [
    'SKU', 'Category', 'Product Description', 'Listing URL',
    'Current Return Rate', 'Star Rating', 'Total Reviews', 
    'Average Price', 'Cost per Unit', 'Last Updated'
]

# Medical device categories (enhanced for quality management)
MEDICAL_DEVICE_CATEGORIES = [
    "Mobility Aids", "Bathroom Safety", "Pain Relief", "Sleep & Comfort", 
    "Fitness & Recovery", "Daily Living Aids", "Respiratory Care",
    "Blood Pressure Monitors", "Diabetes Care", "Orthopedic Support",
    "First Aid", "Wound Care", "Compression Wear", "Exercise Equipment",
    "Home Diagnostics", "Therapy & Rehabilitation", "Incontinence Care",
    "Vision & Hearing Aids", "Surgical Supplies", "Diagnostic Equipment",
    "Other Medical Device"
]

# Export file formats commonly used by Amazon sellers
SUPPORTED_EXPORT_FORMATS = {
    'sellerboard': {
        'description': 'Sellerboard export files',
        'date_columns': ['Date', 'Order Date', 'Return Date', 'Review Date'],
        'text_columns': ['Return Reason', 'Review Text', 'Customer Comment', 'Feedback'],
        'rating_columns': ['Rating', 'Star Rating', 'Review Rating']
    },
    'amazon_seller_central': {
        'description': 'Amazon Seller Central exports',
        'date_columns': ['Order Date', 'Return Request Date', 'Refund Date'],
        'text_columns': ['Return Reason', 'Buyer Comment', 'Customer Message'],
        'rating_columns': ['Rating']
    },
    'helium10': {
        'description': 'Helium 10 export files',
        'date_columns': ['Date', 'Review Date'],
        'text_columns': ['Review Text', 'Review Title'],
        'rating_columns': ['Rating', 'Stars']
    },
    'jungle_scout': {
        'description': 'Jungle Scout exports',
        'date_columns': ['Date', 'Review Date'],
        'text_columns': ['Review', 'Review Text'],
        'rating_columns': ['Rating']
    }
}

# Enhanced file size and format limits
MAX_FILE_SIZE_MB = 100  # Increased for large export files
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
SUPPORTED_DOC_FORMATS = ['.pdf', '.docx', '.txt']
SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv']

# Date format patterns for parsing various export formats
DATE_PATTERNS = [
    '%Y-%m-%d',           # 2024-01-15
    '%m/%d/%Y',           # 01/15/2024
    '%d/%m/%Y',           # 15/01/2024
    '%Y/%m/%d',           # 2024/01/15
    '%m-%d-%Y',           # 01-15-2024
    '%d-%m-%Y',           # 15-01-2024
    '%B %d, %Y',          # January 15, 2024
    '%b %d, %Y',          # Jan 15, 2024
    '%d %B %Y',           # 15 January 2024
    '%d %b %Y',           # 15 Jan 2024
    '%Y%m%d',             # 20240115
    '%m/%d/%y',           # 01/15/24
    '%d/%m/%y'            # 15/01/24
]

class UploadError(Exception):
    """Custom exception for upload-related errors"""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DateProcessingError(Exception):
    """Custom exception for date processing errors"""
    pass

class EnhancedFileProcessor:
    """Enhanced file processing with focus on customer feedback data"""
    
    @staticmethod
    def validate_file_size(file_data: bytes, max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
        """Validate file size with enhanced limits for export files"""
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise UploadError(f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)")
        return True
    
    @staticmethod
    def validate_file_format(filename: str, allowed_formats: List[str]) -> bool:
        """Enhanced file format validation"""
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_formats:
            raise UploadError(f"Unsupported file format: {file_ext}. Allowed formats: {', '.join(allowed_formats)}")
        return True
    
    @staticmethod
    def detect_encoding(file_data: bytes) -> str:
        """Enhanced encoding detection for international export files"""
        try:
            # Try UTF-8 first (most common)
            file_data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        
        # Try common encodings used by Amazon/export tools
        encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'ascii']
        for encoding in encodings:
            try:
                file_data.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Fallback to utf-8 with error handling
        return 'utf-8'
    
    @staticmethod
    def detect_export_format(df: pd.DataFrame, filename: str) -> Optional[str]:
        """Detect the export format based on column names and structure"""
        column_names = [col.lower() for col in df.columns]
        
        # Check for Sellerboard format
        sellerboard_indicators = ['asin', 'sku', 'product name', 'return reason', 'order date']
        if sum(1 for indicator in sellerboard_indicators if any(indicator in col for col in column_names)) >= 3:
            return 'sellerboard'
        
        # Check for Amazon Seller Central format
        amazon_indicators = ['order-id', 'sku', 'return-reason', 'buyer-comment']
        if sum(1 for indicator in amazon_indicators if any(indicator in col for col in column_names)) >= 2:
            return 'amazon_seller_central'
        
        # Check for Helium 10 format
        h10_indicators = ['review date', 'review text', 'rating', 'reviewer']
        if sum(1 for indicator in h10_indicators if any(indicator in col for col in column_names)) >= 3:
            return 'helium10'
        
        # Check for Jungle Scout format
        js_indicators = ['product name', 'review', 'stars', 'date']
        if sum(1 for indicator in js_indicators if any(indicator in col for col in column_names)) >= 3:
            return 'jungle_scout'
        
        return None

class EnhancedDataValidator:
    """Enhanced data validation for customer feedback and quality management"""
    
    @staticmethod
    def validate_product_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that essential product columns are present"""
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        return len(missing_cols) == 0, missing_cols
    
    @staticmethod
    def validate_asin_format(asin: str) -> bool:
        """Enhanced ASIN validation with better pattern matching"""
        if not isinstance(asin, str):
            return False
        
        asin = str(asin).strip().upper()
        
        # Standard ASIN format: B followed by 9 alphanumeric characters
        if re.match(r'^B[A-Z0-9]{9}$', asin):
            return True
        
        # Some variations (older ASINs)
        if re.match(r'^[A-Z0-9]{10}$', asin) and len(asin) == 10:
            return True
        
        return False
    
    @staticmethod
    def validate_date_format(date_str: str) -> Tuple[bool, Optional[datetime]]:
        """Enhanced date validation with multiple format support"""
        if not date_str or pd.isna(date_str):
            return False, None
        
        date_str = str(date_str).strip()
        
        for pattern in DATE_PATTERNS:
            try:
                parsed_date = datetime.strptime(date_str, pattern)
                # Validate date is reasonable (not in future, not too old)
                if datetime(2020, 1, 1) <= parsed_date <= datetime.now() + timedelta(days=1):
                    return True, parsed_date
            except ValueError:
                continue
        
        return False, None
    
    @staticmethod
    def validate_feedback_text(text: str, min_length: int = 5) -> bool:
        """Validate customer feedback text quality"""
        if not text or pd.isna(text):
            return False
        
        text = str(text).strip()
        
        # Check minimum length
        if len(text) < min_length:
            return False
        
        # Check for meaningful content (not just punctuation/numbers)
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        return True
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced dataframe cleaning for customer feedback analysis"""
        df_clean = df.copy()
        
        # Strip whitespace from all string columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace('nan', np.nan)
                df_clean[col] = df_clean[col].replace('', np.nan)
        
        # Standardize ASIN format
        asin_columns = [col for col in df_clean.columns if 'asin' in col.lower()]
        for col in asin_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
        
        # Parse and standardize dates
        date_columns = [col for col in df_clean.columns 
                       if any(date_term in col.lower() for date_term in ['date', 'time', 'created', 'updated'])]
        
        for col in date_columns:
            df_clean[col] = df_clean[col].apply(EnhancedDataValidator._parse_date_safe)
        
        # Clean numeric columns
        numeric_columns = [col for col in df_clean.columns 
                          if any(num_term in col.lower() for num_term in ['rating', 'star', 'score', 'price', 'cost'])]
        
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    @staticmethod
    def _parse_date_safe(date_str: str) -> Optional[str]:
        """Safely parse date string to standard format"""
        is_valid, parsed_date = EnhancedDataValidator.validate_date_format(date_str)
        if is_valid and parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        return None
    
    @staticmethod
    def validate_business_logic(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Enhanced business logic validation for quality management"""
        warnings = []
        
        for idx, row in df.iterrows():
            excel_row = idx + 2  # Excel row number
            
            # ASIN validation
            asin_columns = [col for col in df.columns if 'asin' in col.lower()]
            for col in asin_columns:
                if col in row and row[col]:
                    if not EnhancedDataValidator.validate_asin_format(row[col]):
                        warnings.append({
                            'row': excel_row,
                            'type': 'error',
                            'column': col,
                            'message': f'Invalid ASIN format: {row[col]}'
                        })
            
            # Rating validation
            rating_columns = [col for col in df.columns if any(term in col.lower() for term in ['rating', 'star'])]
            for col in rating_columns:
                if col in row and pd.notna(row[col]):
                    rating = row[col]
                    if not (1 <= rating <= 5):
                        warnings.append({
                            'row': excel_row,
                            'type': 'warning',
                            'column': col,
                            'message': f'Rating {rating} outside expected range (1-5)'
                        })
            
            # Date validation
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                if col in row and row[col]:
                    is_valid, _ = EnhancedDataValidator.validate_date_format(str(row[col]))
                    if not is_valid:
                        warnings.append({
                            'row': excel_row,
                            'type': 'warning',
                            'column': col,
                            'message': f'Date format not recognized: {row[col]}'
                        })
            
            # Feedback text validation
            text_columns = [col for col in df.columns 
                           if any(term in col.lower() for term in ['text', 'comment', 'reason', 'feedback', 'review'])]
            for col in text_columns:
                if col in row and row[col]:
                    if not EnhancedDataValidator.validate_feedback_text(str(row[col])):
                        warnings.append({
                            'row': excel_row,
                            'type': 'warning',
                            'column': col,
                            'message': f'Feedback text may be too short or invalid: {str(row[col])[:50]}...'
                        })
        
        return warnings

class ExportFileProcessor:
    """Specialized processor for Amazon/marketplace export files"""
    
    @staticmethod
    def process_sellerboard_export(df: pd.DataFrame) -> Dict[str, Any]:
        """Process Sellerboard export files for customer feedback analysis"""
        try:
            result = {
                'success': False,
                'products': [],
                'customer_feedback': {},
                'processing_summary': {},
                'export_format': 'sellerboard'
            }
            
            # Map Sellerboard columns to standard format
            column_mapping = ExportFileProcessor._get_sellerboard_column_mapping(df.columns)
            
            # Rename columns to standard format
            df_mapped = df.rename(columns=column_mapping)
            
            # Extract product information
            if 'asin' in df_mapped.columns:
                products = ExportFileProcessor._extract_products_from_export(df_mapped)
                result['products'] = products
            
            # Extract customer feedback (reviews and returns)
            feedback = ExportFileProcessor._extract_feedback_from_export(df_mapped, 'sellerboard')
            result['customer_feedback'] = feedback
            
            # Processing summary
            result['processing_summary'] = {
                'total_rows': len(df),
                'products_found': len(result['products']),
                'feedback_items': sum(len(items) for items in result['customer_feedback'].values()),
                'date_range': ExportFileProcessor._get_date_range(df_mapped)
            }
            
            result['success'] = True
            return result
            
        except Exception as e:
            logger.error(f"Error processing Sellerboard export: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to process Sellerboard export: {str(e)}",
                'export_format': 'sellerboard'
            }
    
    @staticmethod
    def process_amazon_export(df: pd.DataFrame) -> Dict[str, Any]:
        """Process Amazon Seller Central export files"""
        try:
            result = {
                'success': False,
                'products': [],
                'customer_feedback': {},
                'processing_summary': {},
                'export_format': 'amazon_seller_central'
            }
            
            # Map Amazon columns to standard format
            column_mapping = ExportFileProcessor._get_amazon_column_mapping(df.columns)
            df_mapped = df.rename(columns=column_mapping)
            
            # Extract products and feedback
            products = ExportFileProcessor._extract_products_from_export(df_mapped)
            feedback = ExportFileProcessor._extract_feedback_from_export(df_mapped, 'amazon_seller_central')
            
            result['products'] = products
            result['customer_feedback'] = feedback
            
            result['processing_summary'] = {
                'total_rows': len(df),
                'products_found': len(products),
                'feedback_items': sum(len(items) for items in feedback.values()),
                'date_range': ExportFileProcessor._get_date_range(df_mapped)
            }
            
            result['success'] = True
            return result
            
        except Exception as e:
            logger.error(f"Error processing Amazon export: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to process Amazon export: {str(e)}",
                'export_format': 'amazon_seller_central'
            }
    
    @staticmethod
    def _get_sellerboard_column_mapping(columns: List[str]) -> Dict[str, str]:
        """Get column mapping for Sellerboard exports"""
        mapping = {}
        
        for col in columns:
            col_lower = col.lower()
            
            # ASIN mapping
            if 'asin' in col_lower:
                mapping[col] = 'asin'
            # Product name mapping
            elif any(term in col_lower for term in ['product name', 'title', 'product title']):
                mapping[col] = 'product_name'
            # SKU mapping
            elif 'sku' in col_lower:
                mapping[col] = 'sku'
            # Date mappings
            elif any(term in col_lower for term in ['order date', 'date ordered']):
                mapping[col] = 'order_date'
            elif any(term in col_lower for term in ['return date', 'refund date']):
                mapping[col] = 'return_date'
            elif any(term in col_lower for term in ['review date']):
                mapping[col] = 'review_date'
            # Feedback text mappings
            elif any(term in col_lower for term in ['return reason', 'return comment']):
                mapping[col] = 'return_reason'
            elif any(term in col_lower for term in ['review text', 'review']):
                mapping[col] = 'review_text'
            elif any(term in col_lower for term in ['buyer comment', 'customer comment']):
                mapping[col] = 'customer_comment'
            # Rating mappings
            elif any(term in col_lower for term in ['rating', 'star rating', 'stars']):
                mapping[col] = 'rating'
        
        return mapping
    
    @staticmethod
    def _get_amazon_column_mapping(columns: List[str]) -> Dict[str, str]:
        """Get column mapping for Amazon Seller Central exports"""
        mapping = {}
        
        for col in columns:
            col_lower = col.lower().replace('-', '_').replace(' ', '_')
            
            if 'asin' in col_lower:
                mapping[col] = 'asin'
            elif 'sku' in col_lower:
                mapping[col] = 'sku'
            elif any(term in col_lower for term in ['product_name', 'item_name']):
                mapping[col] = 'product_name'
            elif any(term in col_lower for term in ['order_date', 'purchase_date']):
                mapping[col] = 'order_date'
            elif any(term in col_lower for term in ['return_request_date', 'return_date']):
                mapping[col] = 'return_date'
            elif any(term in col_lower for term in ['return_reason']):
                mapping[col] = 'return_reason'
            elif any(term in col_lower for term in ['buyer_comment', 'customer_message']):
                mapping[col] = 'customer_comment'
            elif 'rating' in col_lower:
                mapping[col] = 'rating'
        
        return mapping
    
    @staticmethod
    def _extract_products_from_export(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract unique products from export data"""
        products = []
        
        if 'asin' not in df.columns:
            return products
        
        # Group by ASIN to get unique products
        unique_asins = df['asin'].dropna().unique()
        
        for asin in unique_asins:
            if not asin or asin == 'nan':
                continue
            
            product_rows = df[df['asin'] == asin]
            
            # Get product information from first row
            first_row = product_rows.iloc[0]
            
            product = {
                'asin': asin,
                'name': first_row.get('product_name', f'Product {asin}'),
                'sku': first_row.get('sku', ''),
                'category': ExportFileProcessor._infer_category(first_row.get('product_name', '')),
                'total_feedback_items': len(product_rows)
            }
            
            products.append(product)
        
        return products
    
    @staticmethod
    def _extract_feedback_from_export(df: pd.DataFrame, export_format: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract customer feedback from export data"""
        feedback = defaultdict(list)
        
        if 'asin' not in df.columns:
            return dict(feedback)
        
        for _, row in df.iterrows():
            asin = row.get('asin')
            if not asin or asin == 'nan':
                continue
            
            # Extract return feedback
            return_reason = row.get('return_reason', '')
            if return_reason and str(return_reason).strip() and str(return_reason) != 'nan':
                feedback_item = {
                    'type': 'return_reason',
                    'text': str(return_reason).strip(),
                    'date': ExportFileProcessor._extract_date(row, ['return_date', 'order_date']),
                    'source': f'{export_format}_export',
                    'asin': asin
                }
                feedback[asin].append(feedback_item)
            
            # Extract review feedback
            review_text = row.get('review_text', '')
            if review_text and str(review_text).strip() and str(review_text) != 'nan':
                feedback_item = {
                    'type': 'review',
                    'text': str(review_text).strip(),
                    'rating': row.get('rating'),
                    'date': ExportFileProcessor._extract_date(row, ['review_date', 'order_date']),
                    'source': f'{export_format}_export',
                    'asin': asin
                }
                feedback[asin].append(feedback_item)
            
            # Extract customer comments
            customer_comment = row.get('customer_comment', '')
            if customer_comment and str(customer_comment).strip() and str(customer_comment) != 'nan':
                feedback_item = {
                    'type': 'customer_comment',
                    'text': str(customer_comment).strip(),
                    'date': ExportFileProcessor._extract_date(row, ['order_date', 'return_date']),
                    'source': f'{export_format}_export',
                    'asin': asin
                }
                feedback[asin].append(feedback_item)
        
        return dict(feedback)
    
    @staticmethod
    def _extract_date(row: pd.Series, date_columns: List[str]) -> str:
        """Extract date from row using priority column list"""
        for col in date_columns:
            if col in row and row[col] and str(row[col]) != 'nan':
                date_str = str(row[col])
                is_valid, parsed_date = EnhancedDataValidator.validate_date_format(date_str)
                if is_valid and parsed_date:
                    return parsed_date.strftime('%Y-%m-%d')
        
        # Fallback to current date
        return datetime.now().strftime('%Y-%m-%d')
    
    @staticmethod
    def _infer_category(product_name: str) -> str:
        """Infer medical device category from product name"""
        if not product_name:
            return 'Other Medical Device'
        
        name_lower = product_name.lower()
        
        # Category inference based on keywords
        category_keywords = {
            'Mobility Aids': ['rollator', 'walker', 'wheelchair', 'cane', 'mobility', 'scooter'],
            'Bathroom Safety': ['shower', 'bath', 'toilet', 'bathroom', 'grab bar', 'shower chair'],
            'Pain Relief': ['pain', 'relief', 'heat', 'cold', 'therapy', 'massage', 'tens'],
            'Orthopedic Support': ['brace', 'support', 'knee', 'back', 'ankle', 'wrist', 'orthopedic'],
            'Compression Wear': ['compression', 'stocking', 'sock', 'sleeve', 'support'],
            'Blood Pressure Monitors': ['blood pressure', 'bp monitor', 'sphygmomanometer'],
            'Diabetes Care': ['glucose', 'diabetes', 'blood sugar', 'insulin', 'lancet'],
            'Respiratory Care': ['nebulizer', 'cpap', 'oxygen', 'breathing', 'respiratory'],
            'Sleep & Comfort': ['pillow', 'mattress', 'cushion', 'sleep', 'comfort']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        return 'Other Medical Device'
    
    @staticmethod
    def _get_date_range(df: pd.DataFrame) -> Dict[str, str]:
        """Get date range from dataframe"""
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        all_dates = []
        for col in date_columns:
            if col in df.columns:
                valid_dates = df[col].dropna()
                for date_str in valid_dates:
                    is_valid, parsed_date = EnhancedDataValidator.validate_date_format(str(date_str))
                    if is_valid and parsed_date:
                        all_dates.append(parsed_date)
        
        if all_dates:
            return {
                'start_date': min(all_dates).strftime('%Y-%m-%d'),
                'end_date': max(all_dates).strftime('%Y-%m-%d'),
                'total_days': (max(all_dates) - min(all_dates)).days + 1
            }
        
        return {'start_date': 'Unknown', 'end_date': 'Unknown', 'total_days': 0}

class AIDocumentProcessor:
    """Enhanced AI-powered document processing for customer feedback extraction"""
    
    @staticmethod
    def process_image_with_ai_vision(file_data: bytes, filename: str, content_type: str,
                                   asin: Optional[str] = None) -> Dict[str, Any]:
        """Process image using AI Vision with enhanced prompts for customer feedback"""
        try:
            # Get API key
            api_key = AIDocumentProcessor._get_api_key()
            if not api_key:
                return {
                    'success': False,
                    'errors': ['OpenAI API key not configured for AI document processing'],
                    'processing_method': 'none'
                }
            
            # Encode image to base64
            base64_image = base64.b64encode(file_data).decode('utf-8')
            
            # Create enhanced prompt for customer feedback extraction
            prompt = AIDocumentProcessor._create_feedback_extraction_prompt(content_type, asin)
            
            # Make API call
            response = AIDocumentProcessor._call_vision_api(api_key, prompt, base64_image)
            
            if response['success']:
                # Parse the AI response for structured data
                structured_data = AIDocumentProcessor._parse_ai_response(
                    response['content'], content_type
                )
                
                return {
                    'success': True,
                    'text': response['content'],
                    'structured_data': structured_data,
                    'processing_method': 'AI Vision',
                    'confidence_score': response.get('confidence', 0.8)
                }
            else:
                return {
                    'success': False,
                    'errors': [response['error']],
                    'processing_method': 'AI Vision'
                }
                
        except Exception as e:
            logger.error(f"AI Vision processing error: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'processing_method': 'AI Vision'
            }
    
    @staticmethod
    def _get_api_key() -> Optional[str]:
        """Get API key for AI processing"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                return st.secrets['openai_api_key']
        except:
            pass
        
        import os
        return os.environ.get('OPENAI_API_KEY')
    
    @staticmethod
    def _create_feedback_extraction_prompt(content_type: str, asin: Optional[str] = None) -> str:
        """Create enhanced prompt for customer feedback extraction"""
        base_prompt = """You are an expert in analyzing Amazon seller data and customer feedback for medical device products. """
        
        if content_type == "Product Reviews":
            prompt = base_prompt + f"""
Analyze this image containing Amazon product review data and extract all customer feedback information.

Focus on extracting:
1. ASIN (Amazon product identifier - format B0XXXXXXXXX)
2. Product name and category
3. All customer reviews with exact text and star ratings
4. Review dates if visible
5. Reviewer information if available
6. Overall product statistics (average rating, total reviews)

Format your response as structured data:

ASIN: [detected ASIN or "Not found"]
PRODUCT_NAME: [full product name]
CATEGORY: [inferred medical device category]
AVERAGE_RATING: [X.X out of 5 if visible]
TOTAL_REVIEWS: [number if visible]

CUSTOMER_REVIEWS:
[For each review found:]
REVIEW_ID: [sequential number]
RATING: [X stars]
DATE: [review date if visible, format YYYY-MM-DD]
REVIEWER: [reviewer name if visible]
REVIEW_TEXT: "[exact review text]"

QUALITY_INSIGHTS:
- Most common positive themes
- Most common negative themes  
- Quality concerns mentioned
- Safety issues if any
- Recommendations for improvement

Provide specific, actionable insights for medical device quality management.
"""
        
        elif content_type == "Return Reports":
            prompt = base_prompt + f"""
Analyze this image containing Amazon return report data and extract all return information.

Focus on extracting:
1. ASIN and product information
2. Return reasons and customer comments
3. Return dates and order information
4. Return quantities and patterns
5. Customer messages or feedback

Format your response as:

ASIN: [detected ASIN]
PRODUCT_NAME: [product name]
RETURN_SUMMARY: [total returns if visible]

RETURN_DETAILS:
[For each return found:]
ORDER_ID: [order ID if visible]
RETURN_DATE: [date if visible, format YYYY-MM-DD]
RETURN_REASON: "[exact return reason]"
CUSTOMER_COMMENT: "[any customer feedback]"
QUANTITY: [quantity returned if visible]

QUALITY_ANALYSIS:
- Return reason categories (defective, wrong size, not as described, etc.)
- Patterns in return reasons
- Quality issues identified
- Safety concerns if any
- Recommended corrective actions

Focus on actionable quality management insights for medical device compliance.
"""
        
        else:
            prompt = base_prompt + f"""
Analyze this image for Amazon product and customer feedback information related to medical devices.

Extract any relevant information including:
- Product identifiers (ASINs)
- Customer feedback, reviews, or comments
- Product performance data
- Quality issues or concerns
- Dates and timestamps
- Any actionable insights for quality improvement

Provide a structured analysis focused on quality management and customer satisfaction insights.
"""
        
        if asin:
            prompt += f"\n\nNote: This document should relate to ASIN {asin}. Verify this matches the detected ASIN."
        
        return prompt
    
    @staticmethod
    def _call_vision_api(api_key: str, prompt: str, base64_image: str) -> Dict[str, Any]:
        """Call OpenAI Vision API"""
        if not MODULES_AVAILABLE['requests']:
            return {
                'success': False,
                'error': 'Requests module not available for API calls'
            }
        
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
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
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'usage': result.get('usage', {})
                }
            else:
                return {
                    'success': False,
                    'error': f"API error {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"API call failed: {str(e)}"
            }
    
    @staticmethod
    def _parse_ai_response(ai_response: str, content_type: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        structured_data = {}
        
        try:
            # Extract ASIN
            asin_match = re.search(r'ASIN:\s*([B][A-Z0-9]{9}|[A-Z0-9]{10})', ai_response)
            if asin_match:
                structured_data['detected_asin'] = asin_match.group(1)
            
            # Extract product name
            name_match = re.search(r'PRODUCT_NAME:\s*(.+)', ai_response)
            if name_match:
                structured_data['product_name'] = name_match.group(1).strip()
            
            # Extract category
            category_match = re.search(r'CATEGORY:\s*(.+)', ai_response)
            if category_match:
                structured_data['category'] = category_match.group(1).strip()
            
            if content_type == "Product Reviews":
                # Extract reviews
                reviews = []
                review_pattern = r'REVIEW_ID:\s*(\d+)\s*\nRATING:\s*(\d+)\s*stars?\s*\nDATE:\s*([^\n]*)\s*\nREVIEWER:\s*([^\n]*)\s*\nREVIEW_TEXT:\s*"([^"]*)"'
                
                for match in re.finditer(review_pattern, ai_response, re.MULTILINE | re.DOTALL):
                    review = {
                        'review_id': match.group(1),
                        'rating': int(match.group(2)),
                        'date': match.group(3).strip(),
                        'reviewer': match.group(4).strip(),
                        'review_text': match.group(5).strip()
                    }
                    reviews.append(review)
                
                structured_data['reviews'] = reviews
            
            elif content_type == "Return Reports":
                # Extract returns
                returns = []
                return_pattern = r'ORDER_ID:\s*([^\n]*)\s*\nRETURN_DATE:\s*([^\n]*)\s*\nRETURN_REASON:\s*"([^"]*)"\s*\nCUSTOMER_COMMENT:\s*"([^"]*)"'
                
                for match in re.finditer(return_pattern, ai_response, re.MULTILINE | re.DOTALL):
                    return_item = {
                        'order_id': match.group(1).strip(),
                        'return_date': match.group(2).strip(),
                        'return_reason': match.group(3).strip(),
                        'customer_comment': match.group(4).strip()
                    }
                    returns.append(return_item)
                
                structured_data['returns'] = returns
            
            # Extract quality insights
            quality_match = re.search(r'QUALITY_(?:INSIGHTS|ANALYSIS):\s*(.+?)(?=\n[A-Z_]+:|$)', ai_response, re.DOTALL)
            if quality_match:
                structured_data['quality_insights'] = quality_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
        
        return structured_data

class EnhancedTemplateGenerator:
    """Enhanced template generation for customer feedback data collection"""
    
    @staticmethod
    def create_customer_feedback_template() -> bytes:
        """Create comprehensive template for customer feedback data collection"""
        
        # Enhanced template data with customer feedback focus
        template_data = {
            "ASIN*": ["B0DT7NW5VY", "B0DT8XYZ123", "B08CK7MN45", "B0EXAMPLE1", "B0EXAMPLE2"],
            "Product Name*": [
                "Vive Tri-Rollator with Seat and Storage",
                "Premium Shower Chair with Back Support", 
                "Memory Foam Seat Cushion",
                "4-Wheel Mobility Scooter",
                "Compression Knee Support Brace"
            ],
            "Category": [
                "Mobility Aids", "Bathroom Safety", "Pain Relief", 
                "Mobility Aids", "Orthopedic Support"
            ],
            "SKU": ["VH-TRI-001", "VH-SHW-234", "VH-CUS-352", "VH-MOB-456", "VH-SUP-789"],
            "Current Return Rate %": [4.9, 4.0, 5.2, 11.5, 4.9],
            "Star Rating": [4.2, 4.5, 4.1, 3.8, 4.6],
            "Total Reviews": [287, 156, 423, 89, 501],
            "Last Updated": ["2024-01-15", "2024-01-15", "2024-01-15", "2024-01-15", "2024-01-15"]
        }
        
        # Customer feedback examples
        feedback_data = {
            "ASIN": ["B0DT7NW5VY", "B0DT7NW5VY", "B0DT8XYZ123", "B08CK7MN45", "B0EXAMPLE1"],
            "Feedback Type": ["Review", "Return Reason", "Review", "Return Reason", "Customer Comment"],
            "Date": ["2024-01-10", "2024-01-12", "2024-01-14", "2024-01-13", "2024-01-15"],
            "Rating": [2, None, 4, None, 5],
            "Customer Feedback Text": [
                "The wheels started squeaking loudly after just 2 weeks of use. Very annoying and seems cheaply made.",
                "Too heavy for elderly user to maneuver easily. Difficult to lift over thresholds.",
                "Good chair but the legs could be more stable on wet surfaces. Overall satisfied with purchase.",
                "Product arrived damaged in shipping. Poor packaging protection.",
                "Excellent quality and very comfortable. Highly recommend this product!"
            ],
            "Source": ["Amazon Review", "Return Report", "Amazon Review", "Return Report", "Customer Email"]
        }
        
        try:
            output = io.BytesIO()
            
            if MODULES_AVAILABLE['xlsxwriter']:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Product information sheet
                    products_df = pd.DataFrame(template_data)
                    products_df.to_excel(writer, sheet_name='Product Information', index=False)
                    
                    # Customer feedback sheet
                    feedback_df = pd.DataFrame(feedback_data)
                    feedback_df.to_excel(writer, sheet_name='Customer Feedback', index=False)
                    
                    # Get workbook and add formatting
                    workbook = writer.book
                    
                    # Define formats
                    header_format = workbook.add_format({
                        'bold': True, 'text_wrap': True, 'valign': 'top',
                        'fg_color': '#1E40AF', 'font_color': 'white', 'border': 1
                    })
                    
                    required_format = workbook.add_format({
                        'fg_color': '#FEE2E2', 'border': 1
                    })
                    
                    optional_format = workbook.add_format({
                        'fg_color': '#F0F8F0', 'border': 1
                    })
                    
                    feedback_format = workbook.add_format({
                        'fg_color': '#FEF3C7', 'border': 1, 'text_wrap': True
                    })
                    
                    # Format product information sheet
                    product_sheet = writer.sheets['Product Information']
                    
                    # Apply header formatting
                    for col_num, value in enumerate(products_df.columns.values):
                        product_sheet.write(0, col_num, value, header_format)
                    
                    # Apply data formatting
                    for row_num in range(1, len(products_df) + 1):
                        for col_num, col_name in enumerate(products_df.columns):
                            cell_value = products_df.iloc[row_num-1, col_num]
                            
                            if col_name.endswith('*'):
                                cell_format = required_format
                            else:
                                cell_format = optional_format
                            
                            product_sheet.write(row_num, col_num, cell_value, cell_format)
                    
                    # Format customer feedback sheet
                    feedback_sheet = writer.sheets['Customer Feedback']
                    
                    # Apply header formatting
                    for col_num, value in enumerate(feedback_df.columns.values):
                        feedback_sheet.write(0, col_num, value, header_format)
                    
                    # Apply data formatting
                    for row_num in range(1, len(feedback_df) + 1):
                        for col_num, col_name in enumerate(feedback_df.columns):
                            cell_value = feedback_df.iloc[row_num-1, col_num]
                            feedback_sheet.write(row_num, col_num, cell_value, feedback_format)
                    
                    # Add instructions sheet
                    instructions_data = {
                        'Section': ['Product Information', 'Product Information', 'Customer Feedback', 'Customer Feedback', 'Customer Feedback'],
                        'Field Name': ['ASIN*', 'Product Name*', 'Feedback Type', 'Customer Feedback Text', 'Date'],
                        'Required?': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
                        'Description': [
                            'Amazon Standard Identification Number (10 characters, e.g., B0DT7NW5VY)',
                            'Full product name as shown on Amazon listing',
                            'Type of feedback: Review, Return Reason, Customer Comment, etc.',
                            'Exact customer feedback text - this is the main data for analysis',
                            'Date of feedback in YYYY-MM-DD format'
                        ],
                        'Examples': [
                            'B0DT7NW5VY, B08CK7MN45',
                            'Vive Tri-Rollator with Seat, Premium Shower Chair',
                            'Review, Return Reason, Customer Comment',
                            'Product broke after 2 weeks, Too heavy to lift, Great quality!',
                            '2024-01-15, 2024-01-10'
                        ]
                    }
                    
                    instructions_df = pd.DataFrame(instructions_data)
                    instructions_df.to_excel(writer, sheet_name='Instructions & Examples', index=False)
                    
                    inst_sheet = writer.sheets['Instructions & Examples']
                    
                    # Format instructions sheet
                    for col_num, value in enumerate(instructions_df.columns.values):
                        inst_sheet.write(0, col_num, value, header_format)
                    
                    # Set column widths
                    product_sheet.set_column('A:A', 12)  # ASIN
                    product_sheet.set_column('B:B', 35)  # Product Name
                    product_sheet.set_column('C:C', 20)  # Category
                    product_sheet.set_column('D:H', 15)  # Other columns
                    
                    feedback_sheet.set_column('A:A', 12)  # ASIN
                    feedback_sheet.set_column('B:B', 15)  # Feedback Type
                    feedback_sheet.set_column('C:C', 12)  # Date
                    feedback_sheet.set_column('D:D', 8)   # Rating
                    feedback_sheet.set_column('E:E', 50)  # Feedback Text
                    feedback_sheet.set_column('F:F', 15)  # Source
                    
                    inst_sheet.set_column('A:A', 20)  # Section
                    inst_sheet.set_column('B:B', 25)  # Field Name
                    inst_sheet.set_column('C:C', 12)  # Required
                    inst_sheet.set_column('D:D', 60)  # Description
                    inst_sheet.set_column('E:E', 40)  # Examples
                    
                    # Add legend
                    legend_row = len(products_df) + 3
                    product_sheet.write(legend_row, 0, "Legend:", header_format)
                    product_sheet.write(legend_row + 1, 0, "Required fields (*)", required_format)
                    product_sheet.write(legend_row + 2, 0, "Optional fields", optional_format)
                    
            else:
                # Basic Excel without formatting
                with pd.ExcelWriter(output) as writer:
                    pd.DataFrame(template_data).to_excel(writer, sheet_name='Product Information', index=False)
                    pd.DataFrame(feedback_data).to_excel(writer, sheet_name='Customer Feedback', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating template: {str(e)}")
            # Fallback to basic CSV
            df = pd.DataFrame(template_data)
            output = io.BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return output.getvalue()

class EnhancedUploadHandler:
    """Main enhanced upload handler for customer feedback analysis workflow"""
    
    def __init__(self):
        self.file_processor = EnhancedFileProcessor()
        self.data_validator = EnhancedDataValidator()
        self.export_processor = ExportFileProcessor()
        self.ai_processor = AIDocumentProcessor()
        self.template_generator = EnhancedTemplateGenerator()
        
        logger.info("Enhanced Upload Handler initialized for text analysis workflow")
    
    def get_available_modules(self) -> Dict[str, bool]:
        """Get status of available enhanced modules"""
        return MODULES_AVAILABLE.copy()
    
    def get_supported_export_formats(self) -> Dict[str, str]:
        """Get supported export formats with descriptions"""
        return {key: value['description'] for key, value in SUPPORTED_EXPORT_FORMATS.items()}
    
    def create_template(self) -> bytes:
        """Create enhanced template for customer feedback data collection"""
        return self.template_generator.create_customer_feedback_template()
    
    def process_structured_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Enhanced structured file processing with export format detection"""
        try:
            # Validate file
            self.file_processor.validate_file_size(file_data)
            self.file_processor.validate_file_format(filename, SUPPORTED_DATA_FORMATS)
            
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Process based on file type
            if file_ext == '.csv':
                df = self._read_csv_enhanced(file_data)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._read_excel_enhanced(file_data, filename)
            elif file_ext == '.tsv':
                df = pd.read_csv(io.BytesIO(file_data), sep='\t')
            else:
                raise UploadError(f"Unsupported file format: {file_ext}")
            
            # Detect export format
            export_format = self.file_processor.detect_export_format(df, filename)
            
            # Process based on detected format
            if export_format == 'sellerboard':
                return self.export_processor.process_sellerboard_export(df)
            elif export_format == 'amazon_seller_central':
                return self.export_processor.process_amazon_export(df)
            else:
                # Process as generic structured data
                return self._process_generic_structured_data(df, filename)
                
        except Exception as e:
            logger.error(f"Error processing structured file {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    def _read_csv_enhanced(self, file_data: bytes) -> pd.DataFrame:
        """Enhanced CSV reading with better encoding detection"""
        encoding = self.file_processor.detect_encoding(file_data)
        
        try:
            # Try with detected encoding
            df = pd.read_csv(io.BytesIO(file_data), encoding=encoding)
        except UnicodeDecodeError:
            # Fallback encodings
            for fallback_encoding in ['latin-1', 'cp1252', 'utf-8']:
                try:
                    df = pd.read_csv(io.BytesIO(file_data), encoding=fallback_encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UploadError("Could not decode CSV file with any supported encoding")
        
        return df
    
    def _read_excel_enhanced(self, file_data: bytes, filename: str) -> pd.DataFrame:
        """Enhanced Excel reading with multiple sheet handling"""
        try:
            if filename.lower().endswith('.xlsx') and MODULES_AVAILABLE['openpyxl']:
                # Read with openpyxl for better .xlsx support
                excel_file = pd.ExcelFile(io.BytesIO(file_data), engine='openpyxl')
            else:
                # Use default engine
                excel_file = pd.ExcelFile(io.BytesIO(file_data))
            
            # If multiple sheets, try to find the main data sheet
            if len(excel_file.sheet_names) > 1:
                # Look for sheets with customer feedback data
                for sheet_name in excel_file.sheet_names:
                    if any(keyword in sheet_name.lower() 
                          for keyword in ['data', 'export', 'feedback', 'review', 'return']):
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        break
                else:
                    # Use the first sheet with substantial data
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        if len(df) > 1 and len(df.columns) > 1:  # Has actual data
                            break
                    else:
                        df = pd.read_excel(excel_file, sheet_name=0)  # Fallback to first sheet
            else:
                df = pd.read_excel(excel_file, sheet_name=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise UploadError(f"Failed to read Excel file: {str(e)}")
    
    def _process_generic_structured_data(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Process generic structured data with enhanced validation"""
        result = {
            'success': False,
            'filename': filename,
            'products': [],
            'customer_feedback': {},
            'processing_summary': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Basic validation
            if df.empty:
                raise DataValidationError("File is empty")
            
            # Clean the dataframe
            df_clean = self.data_validator.clean_dataframe(df)
            
            # Enhanced validation
            business_warnings = self.data_validator.validate_business_logic(df_clean)
            result['warnings'].extend(business_warnings)
            
            # Separate errors and warnings
            errors_only = [w for w in business_warnings if w['type'] == 'error']
            warnings_only = [w for w in business_warnings if w['type'] == 'warning']
            
            if errors_only:
                result['errors'].extend(errors_only)
                return result
            
            # Try to extract products and feedback
            products, feedback = self._extract_products_and_feedback(df_clean)
            
            result.update({
                'success': True,
                'products': products,
                'customer_feedback': feedback,
                'processing_summary': {
                    'total_rows': len(df_clean),
                    'products_found': len(products),
                    'feedback_items': sum(len(items) for items in feedback.values()),
                    'columns_processed': list(df_clean.columns)
                }
            })
            
            return result
            
        except Exception as e:
            result['errors'].append({
                'type': 'processing',
                'message': str(e)
            })
            return result
    
    def _extract_products_and_feedback(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Extract products and customer feedback from generic dataframe"""
        products = []
        feedback = defaultdict(list)
        
        # Look for ASIN column
        asin_column = None
        for col in df.columns:
            if 'asin' in col.lower():
                asin_column = col
                break
        
        if not asin_column:
            logger.warning("No ASIN column found, treating as feedback-only data")
            return products, dict(feedback)
        
        # Extract unique products
        unique_asins = df[asin_column].dropna().unique()
        
        for asin in unique_asins:
            if not asin or str(asin) == 'nan':
                continue
            
            # Get product info from first occurrence
            product_rows = df[df[asin_column] == asin]
            first_row = product_rows.iloc[0]
            
            # Look for product name column
            product_name = None
            for col in df.columns:
                if any(term in col.lower() for term in ['product', 'name', 'title']):
                    product_name = first_row.get(col, f'Product {asin}')
                    break
            
            if not product_name:
                product_name = f'Product {asin}'
            
            product = {
                'asin': str(asin),
                'name': str(product_name),
                'category': self.export_processor._infer_category(str(product_name)),
                'total_feedback_items': len(product_rows)
            }
            products.append(product)
            
            # Extract feedback from all rows for this ASIN
            for _, row in product_rows.iterrows():
                feedback_items = self._extract_feedback_from_row(row, asin)
                feedback[str(asin)].extend(feedback_items)
        
        return products, dict(feedback)
    
    def _extract_feedback_from_row(self, row: pd.Series, asin: str) -> List[Dict[str, Any]]:
        """Extract feedback items from a single row"""
        feedback_items = []
        
        # Look for feedback text columns
        text_columns = [col for col in row.index 
                       if any(term in col.lower() 
                             for term in ['text', 'comment', 'reason', 'feedback', 'review', 'message'])]
        
        for col in text_columns:
            text_value = row[col]
            if text_value and str(text_value).strip() and str(text_value) != 'nan':
                # Determine feedback type from column name
                col_lower = col.lower()
                if 'return' in col_lower or 'reason' in col_lower:
                    feedback_type = 'return_reason'
                elif 'review' in col_lower:
                    feedback_type = 'review'
                else:
                    feedback_type = 'customer_comment'
                
                # Look for rating
                rating = None
                rating_columns = [c for c in row.index if 'rating' in c.lower() or 'star' in c.lower()]
                if rating_columns:
                    rating = row[rating_columns[0]]
                    if pd.isna(rating):
                        rating = None
                
                # Look for date
                date_str = datetime.now().strftime('%Y-%m-%d')  # Default to today
                date_columns = [c for c in row.index if 'date' in c.lower()]
                if date_columns:
                    date_value = row[date_columns[0]]
                    if date_value and str(date_value) != 'nan':
                        is_valid, parsed_date = self.data_validator.validate_date_format(str(date_value))
                        if is_valid and parsed_date:
                            date_str = parsed_date.strftime('%Y-%m-%d')
                
                feedback_item = {
                    'type': feedback_type,
                    'text': str(text_value).strip(),
                    'rating': rating,
                    'date': date_str,
                    'source': 'uploaded_file',
                    'asin': str(asin)
                }
                
                feedback_items.append(feedback_item)
        
        return feedback_items
    
    def process_manual_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced manual entry processing for customer feedback"""
        result = {
            'success': False,
            'data': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Enhanced validation for feedback-focused workflow
            required_fields = ['asin', 'product_name']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                result['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
                return result
            
            # Validate ASIN
            if not self.data_validator.validate_asin_format(data['asin']):
                result['errors'].append(f"Invalid ASIN format: {data['asin']}")
            
            # Validate feedback text if provided
            feedback_text = data.get('feedback_text', '')
            if feedback_text and not self.data_validator.validate_feedback_text(feedback_text):
                result['warnings'].append("Feedback text may be too short for meaningful analysis")
            
            # If no errors, prepare the data
            if not result['errors']:
                clean_data = self._clean_manual_data(data)
                result.update({
                    'success': True,
                    'data': clean_data
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating manual entry: {str(e)}")
            result['errors'].append(f"Validation error: {str(e)}")
            return result
    
    def _clean_manual_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced manual data cleaning for feedback workflow"""
        clean_data = {}
        
        # Product information
        clean_data['asin'] = str(data['asin']).strip().upper()
        clean_data['product_name'] = str(data['product_name']).strip()
        clean_data['category'] = data.get('category', 'Other Medical Device')
        clean_data['sku'] = data.get('sku', '').strip()
        
        # Optional feedback data
        if data.get('feedback_text'):
            clean_data['feedback_text'] = str(data['feedback_text']).strip()
            clean_data['feedback_type'] = data.get('feedback_type', 'customer_comment')
            clean_data['feedback_date'] = data.get('feedback_date', datetime.now().strftime('%Y-%m-%d'))
            
            # Rating if provided
            if data.get('rating'):
                try:
                    rating = float(data['rating'])
                    if 1 <= rating <= 5:
                        clean_data['rating'] = rating
                except (ValueError, TypeError):
                    pass
        
        # Numeric fields
        numeric_fields = ['current_return_rate', 'star_rating', 'total_reviews', 'average_price']
        for field in numeric_fields:
            if data.get(field) is not None:
                try:
                    clean_data[field] = float(data[field])
                except (ValueError, TypeError):
                    pass
        
        return clean_data
    
    def process_image_document(self, file_data: bytes, filename: str, 
                             content_type: str, asin: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced image/document processing with AI-powered extraction"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Validate file
            if file_ext in SUPPORTED_IMAGE_FORMATS:
                self.file_processor.validate_file_format(filename, SUPPORTED_IMAGE_FORMATS)
                # Process image with AI Vision
                return self.ai_processor.process_image_with_ai_vision(file_data, filename, content_type, asin)
            
            elif file_ext in SUPPORTED_DOC_FORMATS:
                self.file_processor.validate_file_format(filename, SUPPORTED_DOC_FORMATS)
                
                if file_ext == '.pdf':
                    # Try AI Vision for PDF (GPT-4o can handle PDFs directly)
                    return self.ai_processor.process_image_with_ai_vision(file_data, filename, content_type, asin)
                else:
                    return {
                        'success': False,
                        'errors': [f'Document format {file_ext} not yet supported for AI processing'],
                        'processing_method': 'none'
                    }
            else:
                raise UploadError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing image/document {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }

# Export the main enhanced class and exceptions
__all__ = [
    'EnhancedUploadHandler', 
    'UploadError', 
    'DataValidationError', 
    'DateProcessingError',
    'MEDICAL_DEVICE_CATEGORIES',
    'SUPPORTED_EXPORT_FORMATS'
]
