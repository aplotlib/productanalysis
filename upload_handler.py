"""
Fixed Upload Handler Module for Medical Device Customer Feedback Analyzer

**STABLE VERSION - HELIUM 10 OPTIMIZED**

Enhanced upload functionality with robust error handling and accurate Helium 10 processing.

Author: Assistant  
Version: 4.0 - Production Stable
"""

import io
import os
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports
def safe_import(module_name):
    try:
        return __import__(module_name), True
    except ImportError:
        logger.warning(f"Module {module_name} not available")
        return None, False

# Check for optional dependencies
openpyxl, has_openpyxl = safe_import('openpyxl')
xlsxwriter, has_xlsxwriter = safe_import('xlsxwriter')

# Constants
MAX_FILE_SIZE_MB = 50
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']

# Medical device categories
MEDICAL_DEVICE_CATEGORIES = [
    "Mobility Aids", "Bathroom Safety", "Pain Relief", "Sleep & Comfort", 
    "Fitness & Recovery", "Daily Living Aids", "Respiratory Care",
    "Blood Pressure Monitors", "Diabetes Care", "Orthopedic Support",
    "First Aid", "Wound Care", "Compression Wear", "Exercise Equipment",
    "Other Medical Device"
]

# Helium 10 date patterns
DATE_PATTERNS = [
    '%B %d, %Y',      # January 15, 2024 (most common)
    '%b %d, %Y',      # Jan 15, 2024
    '%m/%d/%Y',       # 01/15/2024
    '%Y-%m-%d',       # 2024-01-15
    '%d/%m/%Y',       # 15/01/2024
    '%m-%d-%Y',       # 01-15-2024
]

# Standard Helium 10 columns
HELIUM10_COLUMNS = [
    'Date', 'Author', 'Verified', 'Helpful', 'Title', 'Body', 
    'Rating', 'Images', 'Videos', 'URL', 'Variation', 'Style'
]

class UploadError(Exception):
    """Custom exception for upload errors"""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class FileValidator:
    """File validation utilities"""
    
    @staticmethod
    def validate_file_size(file_data: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> bool:
        """Validate file size"""
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > max_mb:
            raise UploadError(f"File too large ({size_mb:.1f}MB). Maximum size: {max_mb}MB")
        return True
    
    @staticmethod
    def validate_file_format(filename: str) -> str:
        """Validate and return file extension"""
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise UploadError(f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}")
        return file_ext
    
    @staticmethod
    def detect_encoding(file_data: bytes) -> str:
        """Detect file encoding"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                file_data.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'  # Fallback

class Helium10Processor:
    """Specialized processor for Helium 10 review exports"""
    
    @staticmethod
    def detect_helium10_format(df: pd.DataFrame, filename: str) -> bool:
        """Detect if this is a Helium 10 export"""
        try:
            # Check filename pattern
            if not Helium10Processor._check_filename_pattern(filename):
                return False
            
            # Check required columns
            required_cols = ['Date', 'Body', 'Rating']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check data structure
            if len(df) == 0:
                return False
            
            # Validate rating column
            rating_col = df['Rating'].dropna()
            if len(rating_col) > 0:
                valid_ratings = rating_col.between(1, 5, inclusive='both').all()
                if not valid_ratings:
                    return False
            
            logger.info(f"Helium 10 format detected: {filename}")
            return True
            
        except Exception as e:
            logger.warning(f"Error detecting Helium 10 format: {str(e)}")
            return False
    
    @staticmethod
    def _check_filename_pattern(filename: str) -> bool:
        """Check if filename matches Helium 10 pattern"""
        # Pattern: ASIN  Product_Name  Date.csv
        name_without_ext = filename.replace('.csv', '')
        
        # Look for ASIN pattern at start
        asin_pattern = r'^[B][A-Z0-9]{9}'
        if re.match(asin_pattern, name_without_ext):
            return True
        
        # Alternative patterns
        if 'review' in filename.lower() or 'helium' in filename.lower():
            return True
        
        return False
    
    @staticmethod
    def parse_filename_info(filename: str) -> Dict[str, str]:
        """Extract product info from filename"""
        name_without_ext = filename.replace('.csv', '').replace('.xlsx', '').replace('.xls', '')
        
        # Try exact Helium 10 pattern first
        pattern = r'^([B][A-Z0-9]{9})\s+(.+?)\s+(\d{8})$'
        match = re.match(pattern, name_without_ext)
        
        if match:
            return {
                'asin': match.group(1),
                'product_name': match.group(2).strip(),
                'export_date': match.group(3),
                'source': 'helium10_filename'
            }
        
        # Fallback: try to extract ASIN
        asin_match = re.search(r'([B][A-Z0-9]{9})', name_without_ext)
        if asin_match:
            asin = asin_match.group(1)
            remaining = name_without_ext.replace(asin, '').strip()
            
            return {
                'asin': asin,
                'product_name': remaining or 'Unknown Product',
                'export_date': datetime.now().strftime('%Y%m%d'),
                'source': 'filename_fallback'
            }
        
        # Last resort
        return {
            'asin': 'UNKNOWN',
            'product_name': name_without_ext or 'Unknown Product',
            'export_date': datetime.now().strftime('%Y%m%d'),
            'source': 'filename_generic'
        }
    
    @staticmethod
    def process_helium10_data(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Process Helium 10 data with robust error handling"""
        try:
            # Parse filename
            file_info = Helium10Processor.parse_filename_info(filename)
            asin = file_info['asin']
            product_name = file_info['product_name']
            
            # Clean dataframe
            df_clean = Helium10Processor._clean_dataframe(df)
            
            if df_clean.empty:
                raise DataValidationError("No valid review data found")
            
            # Extract reviews
            reviews = Helium10Processor._extract_reviews(df_clean, asin)
            
            # Create product entry
            product = {
                'asin': asin,
                'name': product_name,
                'category': Helium10Processor._infer_category(product_name),
                'total_reviews': len(reviews),
                'filename': filename
            }
            
            # Calculate metrics
            if reviews:
                ratings = [r['rating'] for r in reviews if r.get('rating') is not None]
                if ratings:
                    product['average_rating'] = round(sum(ratings) / len(ratings), 2)
                
                dates = [r['date'] for r in reviews if r.get('date')]
                if dates:
                    product['date_range'] = {
                        'earliest': min(dates),
                        'latest': max(dates)
                    }
            
            return {
                'success': True,
                'export_format': 'helium10_reviews',
                'filename': filename,
                'products': [product],
                'customer_feedback': {asin: reviews},
                'processing_summary': {
                    'total_rows': len(df),
                    'valid_reviews': len(reviews),
                    'asin': asin,
                    'product_name': product_name,
                    'average_rating': product.get('average_rating'),
                    'date_range': product.get('date_range')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing Helium 10 data: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe"""
        df_clean = df.copy()
        
        # Remove empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Clean text columns
        text_cols = ['Title', 'Body', 'Author'] 
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', ''], np.nan)
        
        # Clean ratings
        if 'Rating' in df_clean.columns:
            df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
            # Keep only valid ratings
            valid_mask = df_clean['Rating'].between(1, 5, inclusive='both') | df_clean['Rating'].isna()
            df_clean = df_clean[valid_mask]
        
        # Parse dates
        if 'Date' in df_clean.columns:
            df_clean['Date_Parsed'] = df_clean['Date'].apply(Helium10Processor._parse_date)
            # Keep rows with valid dates
            df_clean = df_clean[df_clean['Date_Parsed'].notna()]
        
        # Require meaningful review text
        if 'Body' in df_clean.columns:
            valid_body = df_clean['Body'].notna() & (df_clean['Body'].str.len() >= 5)
            df_clean = df_clean[valid_body]
        
        return df_clean
    
    @staticmethod
    def _parse_date(date_str: str) -> Optional[str]:
        """Parse date with multiple format support"""
        if not date_str or pd.isna(date_str) or str(date_str).lower() in ['nan', 'none']:
            return None
        
        date_str = str(date_str).strip()
        
        for pattern in DATE_PATTERNS:
            try:
                parsed = datetime.strptime(date_str, pattern)
                # Validate reasonable date range
                if datetime(2010, 1, 1) <= parsed <= datetime.now() + timedelta(days=1):
                    return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def _extract_reviews(df: pd.DataFrame, asin: str) -> List[Dict[str, Any]]:
        """Extract reviews from cleaned dataframe"""
        reviews = []
        
        for _, row in df.iterrows():
            try:
                # Get text components
                title = str(row.get('Title', '')).strip() if pd.notna(row.get('Title')) else ''
                body = str(row.get('Body', '')).strip() if pd.notna(row.get('Body')) else ''
                
                # Skip if no meaningful text
                if not body and not title:
                    continue
                
                # Combine text
                text_parts = []
                if title and title.lower() != 'nan':
                    text_parts.append(f"TITLE: {title}")
                if body and body.lower() != 'nan':
                    text_parts.append(body)
                
                combined_text = " | ".join(text_parts)
                
                # Create review
                review = {
                    'type': 'review',
                    'text': combined_text,
                    'review_title': title,
                    'review_body': body,
                    'rating': int(row['Rating']) if pd.notna(row.get('Rating')) else None,
                    'date': row.get('Date_Parsed', datetime.now().strftime('%Y-%m-%d')),
                    'author': str(row.get('Author', 'Anonymous')).strip(),
                    'verified': str(row.get('Verified', 'Unknown')).strip(),
                    'helpful_votes': str(row.get('Helpful', '0')).strip(),
                    'has_images': bool(str(row.get('Images', '')).strip()),
                    'has_videos': bool(str(row.get('Videos', '')).strip()),
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
    def _infer_category(product_name: str) -> str:
        """Infer medical device category"""
        if not product_name:
            return 'Other Medical Device'
        
        name_lower = product_name.lower()
        
        # Category keywords
        keywords = {
            'Mobility Aids': ['rollator', 'walker', 'wheelchair', 'cane', 'mobility'],
            'Bathroom Safety': ['shower', 'bath', 'toilet', 'bathroom', 'grab bar'],
            'Pain Relief': ['pain', 'relief', 'heat', 'cold', 'therapy', 'tens'],
            'Sleep & Comfort': ['mattress', 'pillow', 'cushion', 'sleep', 'bed'],
            'Orthopedic Support': ['brace', 'support', 'knee', 'back', 'ankle'],
            'Blood Pressure Monitors': ['blood pressure', 'bp monitor'],
            'Diabetes Care': ['glucose', 'diabetes', 'blood sugar'],
        }
        
        # Find best match
        best_score = 0
        best_category = 'Other Medical Device'
        
        for category, kw_list in keywords.items():
            score = sum(1 for kw in kw_list if kw in name_lower)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category

class FileReader:
    """File reading utilities"""
    
    @staticmethod
    def read_csv(file_data: bytes) -> pd.DataFrame:
        """Read CSV with encoding detection"""
        encoding = FileValidator.detect_encoding(file_data)
        
        try:
            return pd.read_csv(io.BytesIO(file_data), encoding=encoding)
        except Exception as e:
            # Try fallback encodings
            for fallback in ['latin-1', 'cp1252']:
                try:
                    return pd.read_csv(io.BytesIO(file_data), encoding=fallback)
                except:
                    continue
            raise UploadError(f"Could not read CSV file: {str(e)}")
    
    @staticmethod
    def read_excel(file_data: bytes, filename: str) -> pd.DataFrame:
        """Read Excel with multiple engine support"""
        try:
            # Try openpyxl for .xlsx
            if filename.lower().endswith('.xlsx') and has_openpyxl:
                excel_file = pd.ExcelFile(io.BytesIO(file_data), engine='openpyxl')
            else:
                # Fallback to default engine
                excel_file = pd.ExcelFile(io.BytesIO(file_data))
            
            # Find sheet with data
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if len(df) > 0 and len(df.columns) > 1:
                    return df
            
            # Use first sheet as fallback
            return pd.read_excel(excel_file, sheet_name=0)
            
        except Exception as e:
            raise UploadError(f"Could not read Excel file: {str(e)}")

class UploadHandler:
    """Main upload handler class"""
    
    def __init__(self):
        self.validator = FileValidator()
        self.helium10_processor = Helium10Processor()
        self.file_reader = FileReader()
        
        logger.info("Upload Handler initialized - Helium 10 optimized")
    
    def process_structured_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Process uploaded file with comprehensive error handling"""
        try:
            # Validate file
            self.validator.validate_file_size(file_data)
            file_ext = self.validator.validate_file_format(filename)
            
            # Read file based on extension
            if file_ext == '.csv':
                df = self.file_reader.read_csv(file_data)
            elif file_ext in ['.xlsx', '.xls']:
                df = self.file_reader.read_excel(file_data, filename)
            else:
                raise UploadError(f"Unsupported format: {file_ext}")
            
            logger.info(f"File read successfully: {len(df)} rows, {len(df.columns)} columns")
            
            # Check if Helium 10 format
            if self.helium10_processor.detect_helium10_format(df, filename):
                return self.helium10_processor.process_helium10_data(df, filename)
            else:
                # Generic processing
                return self._process_generic_data(df, filename)
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    def _process_generic_data(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Process generic data format"""
        try:
            # Try to identify key columns
            columns = [col.lower() for col in df.columns]
            
            # Look for common patterns
            has_reviews = any('review' in col or 'body' in col or 'comment' in col for col in columns)
            has_ratings = any('rating' in col or 'star' in col for col in columns)
            has_dates = any('date' in col for col in columns)
            
            if has_reviews:
                logger.info("Generic review data detected")
                # Convert to standard format
                return self._convert_generic_to_standard(df, filename)
            else:
                # Just return as structured data
                return {
                    'success': True,
                    'export_format': 'generic',
                    'filename': filename,
                    'data': df,
                    'processing_summary': {
                        'total_rows': len(df),
                        'total_columns': len(df.columns),
                        'columns': list(df.columns)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing generic data: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    def _convert_generic_to_standard(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Convert generic review data to standard format"""
        try:
            # Map columns
            column_mapping = self._map_columns(df.columns)
            
            reviews = []
            asin = 'GENERIC001'
            product_name = f"Product from {filename}"
            
            for _, row in df.iterrows():
                try:
                    # Extract text
                    text_parts = []
                    if column_mapping.get('title'):
                        title = str(row.get(column_mapping['title'], '')).strip()
                        if title and title.lower() != 'nan':
                            text_parts.append(f"TITLE: {title}")
                    
                    if column_mapping.get('body'):
                        body = str(row.get(column_mapping['body'], '')).strip()
                        if body and body.lower() != 'nan':
                            text_parts.append(body)
                    
                    if not text_parts:
                        continue
                    
                    combined_text = " | ".join(text_parts)
                    
                    # Extract other fields
                    rating = None
                    if column_mapping.get('rating'):
                        try:
                            rating = int(float(row.get(column_mapping['rating'], 0)))
                            if not 1 <= rating <= 5:
                                rating = None
                        except:
                            rating = None
                    
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    if column_mapping.get('date'):
                        date_val = row.get(column_mapping['date'])
                        if pd.notna(date_val):
                            # Try to parse date
                            parsed_date = self.helium10_processor._parse_date(str(date_val))
                            if parsed_date:
                                date_str = parsed_date
                    
                    review = {
                        'type': 'review',
                        'text': combined_text,
                        'review_title': text_parts[0] if text_parts else '',
                        'review_body': text_parts[-1] if text_parts else '',
                        'rating': rating,
                        'date': date_str,
                        'author': 'Anonymous',
                        'verified': 'Unknown',
                        'source': 'generic_import',
                        'asin': asin
                    }
                    
                    reviews.append(review)
                    
                except Exception as e:
                    logger.warning(f"Error processing row: {str(e)}")
                    continue
            
            # Create result
            product = {
                'asin': asin,
                'name': product_name,
                'category': 'Other Medical Device',
                'total_reviews': len(reviews),
                'filename': filename
            }
            
            if reviews:
                ratings = [r['rating'] for r in reviews if r.get('rating') is not None]
                if ratings:
                    product['average_rating'] = round(sum(ratings) / len(ratings), 2)
            
            return {
                'success': True,
                'export_format': 'generic_reviews',
                'filename': filename,
                'products': [product],
                'customer_feedback': {asin: reviews},
                'processing_summary': {
                    'total_rows': len(df),
                    'valid_reviews': len(reviews),
                    'asin': asin,
                    'product_name': product_name,
                    'column_mapping': column_mapping
                }
            }
            
        except Exception as e:
            logger.error(f"Error converting generic data: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'errors': [str(e)]
            }
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map generic columns to standard fields"""
        mapping = {}
        
        for col in columns:
            col_lower = col.lower()
            
            # Title mapping
            if 'title' in col_lower:
                mapping['title'] = col
            
            # Body/text mapping
            elif any(term in col_lower for term in ['body', 'review', 'text', 'comment', 'feedback']):
                mapping['body'] = col
            
            # Rating mapping
            elif any(term in col_lower for term in ['rating', 'star', 'score']):
                mapping['rating'] = col
            
            # Date mapping
            elif 'date' in col_lower:
                mapping['date'] = col
        
        return mapping

# Export main class
__all__ = ['UploadHandler', 'UploadError', 'DataValidationError', 'MEDICAL_DEVICE_CATEGORIES']
