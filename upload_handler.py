"""
Enhanced Upload Handler Module for Medical Device Customer Feedback Analyzer

**OPTIMIZED FOR HELIUM 10 REVIEW EXPORTS**

This module provides comprehensive upload functionality specifically optimized for Helium 10 
review export files with enhanced accuracy and usability for medical device quality management.

Key Enhancements:
✓ Precise Helium 10 format detection and parsing
✓ ASIN extraction from filename (B00TZ73MUY format)
✓ Product name extraction from filename  
✓ Accurate column mapping for review analysis
✓ Enhanced date parsing and validation
✓ Medical device quality categorization ready
✓ Quality management compliance (ISO 13485)

Author: Assistant
Version: 3.1 - Helium 10 Optimized
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

# Enhanced constants for Helium 10 processing
REQUIRED_COLUMNS = ['ASIN', 'Product Name']  # For general uploads
HELIUM10_REQUIRED_COLUMNS = ['Date', 'Body', 'Rating']  # For Helium 10 detection
HELIUM10_FULL_COLUMNS = [
    'Date', 'Author', 'Verified', 'Helpful', 'Title', 'Body', 
    'Rating', 'Images', 'Videos', 'URL', 'Variation', 'Style'
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

# Export file formats - ENHANCED HELIUM 10 SUPPORT
SUPPORTED_EXPORT_FORMATS = {
    'helium10_reviews': {
        'description': 'Helium 10 Review Export Files',
        'filename_pattern': r'^([B][A-Z0-9]{9})\s\s(.+?)\s\s(\d{8})\.csv$',
        'required_columns': HELIUM10_REQUIRED_COLUMNS,
        'full_columns': HELIUM10_FULL_COLUMNS,
        'date_column': 'Date',
        'text_column': 'Body',
        'rating_column': 'Rating',
        'title_column': 'Title',
        'author_column': 'Author',
        'verification_column': 'Verified',
        'date_formats': ['%B %d, %Y', '%m/%d/%Y', '%Y-%m-%d']  # Common H10 date formats
    },
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
    }
}

# Enhanced file size and format limits
MAX_FILE_SIZE_MB = 100
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
SUPPORTED_DOC_FORMATS = ['.pdf', '.docx', '.txt']
SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv']

# Enhanced date format patterns specifically for Helium 10
HELIUM10_DATE_PATTERNS = [
    '%B %d, %Y',          # January 15, 2024 (most common H10 format)
    '%b %d, %Y',          # Jan 15, 2024
    '%m/%d/%Y',           # 01/15/2024
    '%d/%m/%Y',           # 15/01/2024  
    '%Y-%m-%d',           # 2024-01-15
    '%m-%d-%Y',           # 01-15-2024
    '%d-%m-%Y',           # 15-01-2024
    '%Y/%m/%d',           # 2024/01/15
    '%d %B %Y',           # 15 January 2024
    '%d %b %Y',           # 15 Jan 2024
]

class UploadError(Exception):
    """Custom exception for upload-related errors"""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class Helium10ProcessingError(Exception):
    """Custom exception for Helium 10 specific processing errors"""
    pass

class Helium10FileProcessor:
    """Specialized processor for Helium 10 review export files"""
    
    @staticmethod
    def detect_helium10_format(df: pd.DataFrame, filename: str) -> bool:
        """
        Accurately detect Helium 10 review export format
        
        Args:
            df: DataFrame to check
            filename: Original filename
            
        Returns:
            bool: True if this is a Helium 10 review export
        """
        try:
            # Check filename pattern first
            filename_match = Helium10FileProcessor.parse_helium10_filename(filename)
            if not filename_match:
                return False
            
            # Check column structure - must have exact H10 columns
            df_columns = list(df.columns)
            
            # Check for required columns
            required_present = all(col in df_columns for col in HELIUM10_REQUIRED_COLUMNS)
            if not required_present:
                return False
            
            # Check for typical H10 column count (should be 12)
            if len(df_columns) != 12:
                logger.warning(f"Unexpected column count for H10: {len(df_columns)} (expected 12)")
                # Don't fail completely, but log the discrepancy
            
            # Verify rating column contains integers 1-5
            if 'Rating' in df.columns:
                rating_values = df['Rating'].dropna().unique()
                valid_ratings = all(isinstance(r, (int, float)) and 1 <= r <= 5 for r in rating_values if pd.notna(r))
                if not valid_ratings:
                    return False
            
            # Check for typical H10 review text in Body column
            if 'Body' in df.columns:
                body_samples = df['Body'].dropna().head(5)
                if len(body_samples) == 0:
                    return False  # No review text found
            
            logger.info(f"Helium 10 format detected: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error detecting Helium 10 format: {str(e)}")
            return False
    
    @staticmethod
    def parse_helium10_filename(filename: str) -> Optional[Dict[str, str]]:
        """
        Parse Helium 10 filename to extract ASIN, product name, and date
        
        Format: {ASIN}  {Product_Name}  {Export_Date}.csv
        Example: B00TZ73MUY  Vive Alternating Air Pressure Mattress Pad  The O 20250522.csv
        
        Args:
            filename: Original filename
            
        Returns:
            Dict with parsed components or None if invalid
        """
        try:
            # Remove .csv extension
            name_without_ext = filename.replace('.csv', '')
            
            # Pattern to match: ASIN (2 spaces) Product Name (2 spaces) Date
            pattern = r'^([B][A-Z0-9]{9})\s\s(.+?)\s\s(\d{8})$'
            match = re.match(pattern, name_without_ext)
            
            if match:
                asin = match.group(1)
                product_name = match.group(2).strip()
                date_str = match.group(3)
                
                # Parse date (YYYYMMDD)
                try:
                    export_date = datetime.strptime(date_str, '%Y%m%d').date()
                except ValueError:
                    logger.warning(f"Could not parse date from filename: {date_str}")
                    export_date = None
                
                return {
                    'asin': asin,
                    'product_name': product_name,
                    'export_date': export_date.strftime('%Y-%m-%d') if export_date else date_str,
                    'original_filename': filename
                }
            
            # Fallback: try to extract just ASIN if pattern doesn't match exactly
            asin_match = re.search(r'^([B][A-Z0-9]{9})', name_without_ext)
            if asin_match:
                asin = asin_match.group(1)
                remaining = name_without_ext[len(asin):].strip()
                
                # Try to find date at the end
                date_match = re.search(r'(\d{8})$', remaining)
                if date_match:
                    date_str = date_match.group(1)
                    product_name = remaining[:-8].strip()
                else:
                    product_name = remaining
                    date_str = datetime.now().strftime('%Y%m%d')
                
                logger.warning(f"Fallback parsing used for filename: {filename}")
                return {
                    'asin': asin,
                    'product_name': product_name,
                    'export_date': date_str,
                    'original_filename': filename
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Helium 10 filename {filename}: {str(e)}")
            return None
    
    @staticmethod
    def process_helium10_export(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Process Helium 10 export with maximum accuracy
        
        Args:
            df: Raw dataframe from CSV
            filename: Original filename
            
        Returns:
            Structured result dictionary
        """
        try:
            result = {
                'success': False,
                'products': [],
                'customer_feedback': {},
                'processing_summary': {},
                'export_format': 'helium10_reviews',
                'filename': filename,
                'warnings': [],
                'errors': []
            }
            
            # Parse filename for product information
            filename_data = Helium10FileProcessor.parse_helium10_filename(filename)
            if not filename_data:
                raise Helium10ProcessingError(f"Could not parse Helium 10 filename: {filename}")
            
            asin = filename_data['asin']
            product_name = filename_data['product_name']
            export_date = filename_data['export_date']
            
            # Validate and clean the dataframe
            df_clean = Helium10FileProcessor._clean_helium10_dataframe(df)
            
            if df_clean.empty:
                raise Helium10ProcessingError("No valid review data found in file")
            
            # Process review data
            reviews = Helium10FileProcessor._extract_reviews_from_helium10(df_clean, asin)
            
            # Create product entry
            product = {
                'asin': asin,
                'name': product_name,
                'category': Helium10FileProcessor._infer_medical_device_category(product_name),
                'total_reviews': len(reviews),
                'export_date': export_date,
                'filename': filename
            }
            
            # Calculate basic metrics
            if reviews:
                ratings = [r['rating'] for r in reviews if r.get('rating') is not None]
                if ratings:
                    product['average_rating'] = round(sum(ratings) / len(ratings), 2)
                    product['rating_distribution'] = {
                        str(i): len([r for r in ratings if r == i]) for i in range(1, 6)
                    }
                
                # Date range
                dates = [r['date'] for r in reviews if r.get('date')]
                if dates:
                    product['date_range'] = {
                        'earliest': min(dates),
                        'latest': max(dates)
                    }
            
            # Prepare results
            result.update({
                'success': True,
                'products': [product],
                'customer_feedback': {asin: reviews},
                'processing_summary': {
                    'total_rows': len(df),
                    'valid_reviews': len(reviews),
                    'asin': asin,
                    'product_name': product_name,
                    'export_date': export_date,
                    'average_rating': product.get('average_rating'),
                    'date_range': product.get('date_range')
                }
            })
            
            logger.info(f"Successfully processed Helium 10 export: {len(reviews)} reviews for {asin}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing Helium 10 export {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'export_format': 'helium10_reviews',
                'errors': [str(e)]
            }
    
    @staticmethod
    def _clean_helium10_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate Helium 10 dataframe"""
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Clean text columns
        text_columns = ['Title', 'Body', 'Author']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', ''], np.nan)
        
        # Clean and validate ratings
        if 'Rating' in df_clean.columns:
            df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
            # Keep only valid ratings (1-5)
            valid_rating_mask = df_clean['Rating'].between(1, 5, inclusive='both')
            invalid_ratings = (~valid_rating_mask) & df_clean['Rating'].notna()
            if invalid_ratings.any():
                logger.warning(f"Found {invalid_ratings.sum()} invalid ratings - these will be excluded")
                df_clean.loc[invalid_ratings, 'Rating'] = np.nan
        
        # Parse and standardize dates
        if 'Date' in df_clean.columns:
            df_clean['Date_Parsed'] = df_clean['Date'].apply(
                Helium10FileProcessor._parse_helium10_date
            )
            # Keep only rows with valid dates
            valid_date_mask = df_clean['Date_Parsed'].notna()
            if not valid_date_mask.all():
                invalid_count = (~valid_date_mask).sum()
                logger.warning(f"Found {invalid_count} rows with invalid dates - these will be excluded")
                df_clean = df_clean[valid_date_mask]
        
        # Require non-empty Body text for meaningful analysis
        if 'Body' in df_clean.columns:
            valid_body_mask = df_clean['Body'].notna() & (df_clean['Body'].str.len() >= 10)
            if not valid_body_mask.all():
                invalid_count = (~valid_body_mask).sum()
                logger.warning(f"Found {invalid_count} rows with insufficient review text - these will be excluded")
                df_clean = df_clean[valid_body_mask]
        
        return df_clean
    
    @staticmethod
    def _parse_helium10_date(date_str: str) -> Optional[str]:
        """Parse Helium 10 date with multiple format support"""
        if not date_str or pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        for pattern in HELIUM10_DATE_PATTERNS:
            try:
                parsed_date = datetime.strptime(date_str, pattern)
                # Validate date is reasonable (not in future, not too old for reviews)
                if datetime(2015, 1, 1) <= parsed_date <= datetime.now() + timedelta(days=1):
                    return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    @staticmethod
    def _extract_reviews_from_helium10(df: pd.DataFrame, asin: str) -> List[Dict[str, Any]]:
        """Extract reviews from cleaned Helium 10 dataframe"""
        reviews = []
        
        for _, row in df.iterrows():
            try:
                # Extract review text from Body (primary) and Title (secondary)
                body_text = row.get('Body', '').strip() if pd.notna(row.get('Body')) else ''
                title_text = row.get('Title', '').strip() if pd.notna(row.get('Title')) else ''
                
                # Combine title and body for complete review text
                review_parts = []
                if title_text and title_text.lower() != 'nan':
                    review_parts.append(f"Title: {title_text}")
                if body_text and body_text.lower() != 'nan':
                    review_parts.append(body_text)
                
                if not review_parts:
                    continue  # Skip if no meaningful text
                
                combined_text = " | ".join(review_parts)
                
                # Create review item
                review = {
                    'type': 'review',
                    'text': combined_text,
                    'review_body': body_text,
                    'review_title': title_text,
                    'rating': int(row['Rating']) if pd.notna(row.get('Rating')) else None,
                    'date': row.get('Date_Parsed', datetime.now().strftime('%Y-%m-%d')),
                    'author': row.get('Author', '').strip() if pd.notna(row.get('Author')) else 'Anonymous',
                    'verified': row.get('Verified', '').strip() if pd.notna(row.get('Verified')) else 'Unknown',
                    'helpful_votes': row.get('Helpful', '').strip() if pd.notna(row.get('Helpful')) else '0',
                    'has_images': bool(row.get('Images', '').strip()) if pd.notna(row.get('Images')) else False,
                    'has_videos': bool(row.get('Videos', '').strip()) if pd.notna(row.get('Videos')) else False,
                    'variation': row.get('Variation', '').strip() if pd.notna(row.get('Variation')) else '',
                    'style': row.get('Style', '').strip() if pd.notna(row.get('Style')) else '',
                    'source': 'helium10_export',
                    'asin': asin
                }
                
                reviews.append(review)
                
            except Exception as e:
                logger.warning(f"Error processing review row: {str(e)}")
                continue
        
        # Sort by date (newest first)
        reviews.sort(key=lambda x: x['date'], reverse=True)
        
        return reviews
    
    @staticmethod
    def _infer_medical_device_category(product_name: str) -> str:
        """Infer medical device category from Helium 10 product name"""
        if not product_name:
            return 'Other Medical Device'
        
        name_lower = product_name.lower()
        
        # Enhanced category inference for medical devices
        category_keywords = {
            'Mobility Aids': [
                'rollator', 'walker', 'wheelchair', 'cane', 'mobility', 'scooter',
                'tri rollator', '3 wheel', '4 wheel', 'rolling walker', 'seat walker'
            ],
            'Bathroom Safety': [
                'shower', 'bath', 'toilet', 'bathroom', 'grab bar', 'shower chair',
                'bath seat', 'shower bench', 'tub', 'safety rail'
            ],
            'Pain Relief': [
                'pain', 'relief', 'heat', 'cold', 'therapy', 'massage', 'tens',
                'heating pad', 'ice pack', 'pain relief', 'therapeutic'
            ],
            'Sleep & Comfort': [
                'mattress', 'pillow', 'cushion', 'sleep', 'comfort', 'bed',
                'mattress pad', 'air mattress', 'pressure mattress', 'foam'
            ],
            'Orthopedic Support': [
                'brace', 'support', 'knee', 'back', 'ankle', 'wrist', 'orthopedic',
                'lumbar', 'cervical', 'posture', 'spine'
            ],
            'Compression Wear': [
                'compression', 'stocking', 'sock', 'sleeve', 'support hose',
                'medical socks', 'circulation'
            ],
            'Blood Pressure Monitors': [
                'blood pressure', 'bp monitor', 'sphygmomanometer', 'pressure cuff'
            ],
            'Diabetes Care': [
                'glucose', 'diabetes', 'blood sugar', 'insulin', 'lancet',
                'diabetic', 'blood glucose'
            ],
            'Respiratory Care': [
                'nebulizer', 'cpap', 'oxygen', 'breathing', 'respiratory',
                'inhaler', 'spirometer'
            ]
        }
        
        # Score each category based on keyword matches
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in name_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            return best_category
        
        return 'Other Medical Device'

class EnhancedFileProcessor:
    """Enhanced file processing with specialized Helium 10 support"""
    
    @staticmethod
    def validate_file_size(file_data: bytes, max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
        """Validate file size"""
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
        """Enhanced encoding detection"""
        try:
            # Try UTF-8 first (most common for H10 exports)
            file_data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        
        # Try common encodings
        encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'ascii']
        for encoding in encodings:
            try:
                file_data.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'
    
    @staticmethod
    def detect_export_format(df: pd.DataFrame, filename: str) -> Optional[str]:
        """Enhanced export format detection with Helium 10 priority"""
        
        # PRIORITY: Check for Helium 10 format first
        if Helium10FileProcessor.detect_helium10_format(df, filename):
            return 'helium10_reviews'
        
        # Check other formats
        column_names = [col.lower() for col in df.columns]
        
        # Check for Sellerboard format
        sellerboard_indicators = ['asin', 'sku', 'product name', 'return reason', 'order date']
        if sum(1 for indicator in sellerboard_indicators if any(indicator in col for col in column_names)) >= 3:
            return 'sellerboard'
        
        # Check for Amazon Seller Central format
        amazon_indicators = ['order-id', 'sku', 'return-reason', 'buyer-comment']
        if sum(1 for indicator in amazon_indicators if any(indicator in col for col in column_names)) >= 2:
            return 'amazon_seller_central'
        
        return None

class UploadHandler:
    """Main enhanced upload handler optimized for Helium 10 and medical device analysis"""
    
    def __init__(self):
        self.file_processor = EnhancedFileProcessor()
        self.helium10_processor = Helium10FileProcessor()
        
        logger.info("Enhanced Upload Handler initialized - Helium 10 optimized for medical device analysis")
    
    def get_available_modules(self) -> Dict[str, bool]:
        """Get status of available modules"""
        return MODULES_AVAILABLE.copy()
    
    def get_supported_export_formats(self) -> Dict[str, str]:
        """Get supported export formats with descriptions"""
        return {key: value['description'] for key, value in SUPPORTED_EXPORT_FORMATS.items()}
    
    def process_structured_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Enhanced structured file processing with Helium 10 optimization
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            
        Returns:
            Structured processing result
        """
        try:
            # Validate file
            self.file_processor.validate_file_size(file_data)
            self.file_processor.validate_file_format(filename, SUPPORTED_DATA_FORMATS)
            
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Read file based on extension
            if file_ext == '.csv':
                df = self._read_csv_enhanced(file_data)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._read_excel_enhanced(file_data, filename)
            elif file_ext == '.tsv':
                df = pd.read_csv(io.BytesIO(file_data), sep='\t')
            else:
                raise UploadError(f"Unsupported file format: {file_ext}")
            
            # Detect export format (Helium 10 gets priority)
            export_format = self.file_processor.detect_export_format(df, filename)
            
            # Process based on detected format
            if export_format == 'helium10_reviews':
                logger.info(f"Processing as Helium 10 review export: {filename}")
                return self.helium10_processor.process_helium10_export(df, filename)
            elif export_format == 'sellerboard':
                return self._process_sellerboard_export(df)
            elif export_format == 'amazon_seller_central':
                return self._process_amazon_export(df)
            else:
                # Fallback to generic processing
                logger.info(f"Processing as generic structured data: {filename}")
                return self._process_generic_structured_data(df, filename)
                
        except Exception as e:
            logger.error(f"Error processing structured file {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    def _read_csv_enhanced(self, file_data: bytes) -> pd.DataFrame:
        """Enhanced CSV reading optimized for Helium 10 exports"""
        encoding = self.file_processor.detect_encoding(file_data)
        
        try:
            # Try with detected encoding
            df = pd.read_csv(io.BytesIO(file_data), encoding=encoding)
        except UnicodeDecodeError:
            # Fallback encodings
            for fallback_encoding in ['latin-1', 'cp1252', 'utf-8']:
                try:
                    df = pd.read_csv(io.BytesIO(file_data), encoding=fallback_encoding)
                    logger.info(f"Successfully read CSV with {fallback_encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UploadError("Could not decode CSV file with any supported encoding")
        
        return df
    
    def _read_excel_enhanced(self, file_data: bytes, filename: str) -> pd.DataFrame:
        """Enhanced Excel reading"""
        try:
            if filename.lower().endswith('.xlsx') and MODULES_AVAILABLE['openpyxl']:
                excel_file = pd.ExcelFile(io.BytesIO(file_data), engine='openpyxl')
            else:
                excel_file = pd.ExcelFile(io.BytesIO(file_data))
            
            # Use first sheet with data
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if len(df) > 1 and len(df.columns) > 1:
                    return df
            
            # Fallback to first sheet
            return pd.read_excel(excel_file, sheet_name=0)
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise UploadError(f"Failed to read Excel file: {str(e)}")
    
    def _process_generic_structured_data(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Process generic structured data (fallback)"""
        result = {
            'success': True,
            'filename': filename,
            'data': df,
            'export_format': 'generic',
            'processing_summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns)
            },
            'warnings': ['File format not recognized - processed as generic data']
        }
        
        return result
    
    # Placeholder methods for other export formats (keeping existing functionality)
    def _process_sellerboard_export(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process Sellerboard export (placeholder)"""
        return {
            'success': True,
            'export_format': 'sellerboard',
            'data': df,
            'processing_summary': {
                'total_rows': len(df),
                'message': 'Sellerboard processing not yet implemented'
            }
        }
    
    def _process_amazon_export(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process Amazon export (placeholder)"""
        return {
            'success': True,
            'export_format': 'amazon_seller_central',
            'data': df,
            'processing_summary': {
                'total_rows': len(df),
                'message': 'Amazon export processing not yet implemented'
            }
        }

# Export the main enhanced class and exceptions
__all__ = [
    'UploadHandler', 
    'UploadError', 
    'DataValidationError', 
    'Helium10ProcessingError',
    'MEDICAL_DEVICE_CATEGORIES',
    'SUPPORTED_EXPORT_FORMATS'
]
