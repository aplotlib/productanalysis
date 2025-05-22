"""
Amazon Medical Device Listing Optimizer - Main Application (FIXED)

Compatibility: Python 3.8+, Streamlit 1.28+
Production Ready: Yes
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import traceback
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Safe imports with error handling
MODULES_LOADED = {}

def safe_import(module_name, from_module=None):
    """Safely import modules with compatibility checks"""
    try:
        if from_module:
            module = __import__(from_module, fromlist=[module_name])
            imported = getattr(module, module_name)
        else:
            imported = __import__(module_name)
        
        MODULES_LOADED[module_name] = True
        return imported, True
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {str(e)}")
        MODULES_LOADED[module_name] = False
        return None, False

# Import custom modules with error handling
upload_handler_module, upload_available = safe_import('upload_handler')
scoring_module, scoring_available = safe_import('product_scoring')
ai_analysis_module, ai_available = safe_import('enhanced_ai_analysis')
dashboard_module, dashboard_available = safe_import('dashboard')

if upload_available:
    from upload_handler import UploadHandler, UploadError, DataValidationError, MEDICAL_DEVICE_CATEGORIES
else:
    logger.error("Upload handler module not available")
    
if scoring_available:
    from product_scoring import CompositeScoring, CompositeScore, SCORING_WEIGHTS, CATEGORY_BENCHMARKS
else:
    logger.error("Scoring module not available")

if ai_available:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AnalysisResult
else:
    logger.error("AI analysis module not available")

if dashboard_available:
    from dashboard import ProfessionalDashboard
else:
    logger.error("Dashboard module not available")

# Application configuration
APP_CONFIG = {
    'title': 'Amazon Medical Device Listing Optimizer',
    'version': '2.0',
    'description': 'Professional performance analytics for medical device listings',
    'support_email': 'support@listingoptimizer.com',
    'max_products_free': 50,
    'session_timeout_hours': 4,
    'python_version': '3.8+',
    'streamlit_version': '1.28+'
}

# Example data - realistic medical device data
EXAMPLE_DATA = {
    'products': [
        {
            'asin': 'B0DT7NW5VY',
            'name': 'Vive Tri-Rollator with Seat and Storage',
            'category': 'Mobility Aids',
            'sku': 'VH-TRI-001',
            'sales_30d': 491,
            'returns_30d': 24,
            'sales_365d': 5840,
            'returns_365d': 285,
            'star_rating': 4.2,
            'total_reviews': 287,
            'average_price': 129.99,
            'cost_per_unit': 65.00,
            'description': 'Premium tri-wheel rollator with padded seat, storage pouch, and easy-fold design for seniors and mobility assistance.'
        },
        {
            'asin': 'B0DT8XYZ123',
            'name': 'Premium Shower Chair with Back Support',
            'category': 'Bathroom Safety',
            'sku': 'VH-SHW-234',
            'sales_30d': 325,
            'returns_30d': 13,
            'sales_365d': 3900,
            'returns_365d': 156,
            'star_rating': 4.5,
            'total_reviews': 156,
            'average_price': 79.99,
            'cost_per_unit': 35.00,
            'description': 'Adjustable shower chair with antimicrobial seat, non-slip feet, and ergonomic back support for bathroom safety.'
        },
        {
            'asin': 'B08CK7MN45',
            'name': 'Memory Foam Seat Cushion',
            'category': 'Pain Relief',
            'sku': 'VH-CUS-352',
            'sales_30d': 623,
            'returns_30d': 31,
            'sales_365d': 7250,
            'returns_365d': 362,
            'star_rating': 4.6,
            'total_reviews': 501,
            'average_price': 39.99,
            'cost_per_unit': 18.00,
            'description': 'Orthopedic memory foam seat cushion with cooling gel insert for pressure relief and improved posture.'
        }
    ],
    'reviews': {
        'B0DT7NW5VY': [
            {'rating': 5, 'review_text': 'This rollator is amazing! Very stable and the seat is so comfortable. Great for my daily walks.', 'asin': 'B0DT7NW5VY'},
            {'rating': 4, 'review_text': 'Good quality rollator but the assembly instructions could be clearer. Works great once set up.', 'asin': 'B0DT7NW5VY'},
            {'rating': 2, 'review_text': 'The wheels started squeaking after just 2 weeks. Also heavier than expected.', 'asin': 'B0DT7NW5VY'},
        ],
        'B0DT8XYZ123': [
            {'rating': 5, 'review_text': 'Excellent shower chair! Very sturdy and the back support is perfect. Easy to adjust height.', 'asin': 'B0DT8XYZ123'},
            {'rating': 4, 'review_text': 'Good chair but the legs could be a bit more stable. Overall satisfied with purchase.', 'asin': 'B0DT8XYZ123'},
        ]
    },
    'returns': {
        'B0DT7NW5VY': [
            {'return_reason': 'Wheels started squeaking loudly after 2 weeks of use', 'asin': 'B0DT7NW5VY'},
            {'return_reason': 'Too heavy for elderly user to maneuver easily', 'asin': 'B0DT7NW5VY'},
        ],
        'B0DT8XYZ123': [
            {'return_reason': 'Chair legs not stable enough on wet surfaces', 'asin': 'B0DT8XYZ123'},
        ]
    }
}

class SafeDataProcessor:
    """Thread-safe data processing with proper error handling"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.upload_handler = None
        self.scoring_system = None
        self.ai_analyzer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with error handling"""
        try:
            if upload_available:
                self.upload_handler = UploadHandler()
                logger.info("Upload handler initialized")
            
            if scoring_available:
                self.scoring_system = CompositeScoring()
                logger.info("Scoring system initialized")
            
            if ai_available:
                self.ai_analyzer = EnhancedAIAnalyzer()
                logger.info("AI analyzer initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
    
    def safe_convert_numeric(self, value, default=0):
        """Safely convert values to numeric"""
        if pd.isna(value) or value == '' or value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_convert_int(self, value, default=0):
        """Safely convert values to integer"""
        numeric_val = self.safe_convert_numeric(value, default)
        try:
            return int(numeric_val)
        except (ValueError, TypeError):
            return default
    
    def process_uploaded_data(self, uploaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Thread-safe data processing"""
        with self.lock:
            try:
                processed_data = {
                    'products': [],
                    'reviews': {},
                    'returns': {},
                    'processing_summary': {}
                }
                
                # Process structured data
                if 'structured_data' in uploaded_data:
                    df = uploaded_data['structured_data']
                    logger.info(f"Processing {len(df)} products from structured data")
                    
                    for _, row in df.iterrows():
                        try:
                            product = {
                                'asin': str(row.get('ASIN', '')).strip(),
                                'name': str(row.get('Product Name', f"Product {row.get('ASIN', 'Unknown')}")),
                                'category': str(row.get('Category', 'Other')),
                                'sku': str(row.get('SKU', '')),
                                'sales_30d': self.safe_convert_int(row.get('Last 30 Days Sales', 0)),
                                'returns_30d': self.safe_convert_int(row.get('Last 30 Days Returns', 0)),
                                'sales_365d': self.safe_convert_int(row.get('Last 365 Days Sales')) if pd.notna(row.get('Last 365 Days Sales')) else None,
                                'returns_365d': self.safe_convert_int(row.get('Last 365 Days Returns')) if pd.notna(row.get('Last 365 Days Returns')) else None,
                                'star_rating': self.safe_convert_numeric(row.get('Star Rating')) if pd.notna(row.get('Star Rating')) else None,
                                'total_reviews': self.safe_convert_int(row.get('Total Reviews')) if pd.notna(row.get('Total Reviews')) else None,
                                'average_price': self.safe_convert_numeric(row.get('Average Price')) if pd.notna(row.get('Average Price')) else None,
                                'cost_per_unit': self.safe_convert_numeric(row.get('Cost per Unit')) if pd.notna(row.get('Cost per Unit')) else None,
                                'description': str(row.get('Product Description', ''))
                            }
                            
                            # Validate essential data
                            if product['asin'] and product['sales_30d'] >= 0:
                                processed_data['products'].append(product)
                            else:
                                logger.warning(f"Skipping invalid product: {product['asin']}")
                                
                        except Exception as e:
                            logger.error(f"Error processing product row: {str(e)}")
                            continue
                    
                    processed_data['processing_summary']['structured_products'] = len(processed_data['products'])
                
                # Process manual reviews and returns
                for key in ['manual_reviews', 'manual_returns']:
                    if key in uploaded_data:
                        processed_data[key.replace('manual_', '')] = uploaded_data[key]
                        processed_data['processing_summary'][key] = sum(len(data) for data in uploaded_data[key].values())
                
                # Process extracted documents
                if 'documents' in uploaded_data:
                    doc_reviews, doc_returns = self._process_document_extractions(uploaded_data['documents'])
                    
                    # Merge with existing data
                    for asin, reviews in doc_reviews.items():
                        processed_data['reviews'].setdefault(asin, []).extend(reviews)
                    
                    for asin, returns in doc_returns.items():
                        processed_data['returns'].setdefault(asin, []).extend(returns)
                
                logger.info(f"Data processing complete: {processed_data['processing_summary']}")
                return processed_data
                
            except Exception as e:
                logger.error(f"Error processing uploaded data: {str(e)}")
                raise Exception(f"Failed to process uploaded data: {str(e)}")
    
    def _process_document_extractions(self, documents: List[Dict[str, Any]]) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Process extracted data from documents"""
        reviews = {}
        returns = {}
        
        for doc in documents:
            if not doc.get('success') or not doc.get('asin'):
                continue
            
            asin = doc['asin']
            content_type = doc.get('content_type', '')
            structured_data = doc.get('structured_data', {})
            
            if content_type == 'Product Reviews' and 'reviews' in structured_data:
                reviews.setdefault(asin, []).extend(structured_data['reviews'])
            
            elif content_type == 'Return Reports' and 'returns' in structured_data:
                returns.setdefault(asin, []).extend(structured_data['returns'])
        
        return reviews, returns
    
    def calculate_scores(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite scores with error handling"""
        if not self.scoring_system:
            return {}
        
        try:
            products = processed_data.get('products', [])
            if not products:
                logger.warning("No products available for scoring")
                return {}
            
            scores = {}
            logger.info(f"Calculating scores for {len(products)} products")
            
            for product in products:
                try:
                    asin = product['asin']
                    ai_analysis = st.session_state.get('ai_analysis_results', {}).get(asin)
                    
                    score = self.scoring_system.score_single_product(
                        product, 
                        products,
                        ai_analysis
                    )
                    
                    scores[asin] = score
                    logger.debug(f"Calculated score for {asin}: {score.composite_score:.1f}")
                    
                except Exception as e:
                    logger.error(f"Failed to calculate score for {product.get('asin', 'unknown')}: {str(e)}")
                    continue
            
            logger.info(f"Successfully calculated scores for {len(scores)} products")
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating scores: {str(e)}")
            return {}
    
    def run_ai_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI analysis with proper error handling"""
        if not self.ai_analyzer:
            return {}
        
        try:
            products = processed_data.get('products', [])
            reviews_data = processed_data.get('reviews', {})
            returns_data = processed_data.get('returns', {})
            
            if not products:
                logger.warning("No products available for AI analysis")
                return {}
            
            # Check API status
            api_status = self.ai_analyzer.get_api_status()
            if not api_status.get('available', False):
                logger.warning(f"AI API not available: {api_status.get('error', 'Unknown error')}")
                return {}
            
            ai_results = {}
            logger.info(f"Running AI analysis for {len(products)} products")
            
            for product in products:
                asin = product['asin']
                
                try:
                    product_reviews = reviews_data.get(asin, [])
                    product_returns = returns_data.get(asin, [])
                    
                    if not product_reviews and not product_returns:
                        logger.debug(f"No review or return data for {asin}, skipping AI analysis")
                        continue
                    
                    analysis_results = self.ai_analyzer.analyze_product_comprehensive(
                        product, product_reviews, product_returns
                    )
                    
                    ai_results[asin] = analysis_results
                    logger.debug(f"Completed AI analysis for {asin}")
                    
                except Exception as e:
                    logger.error(f"AI analysis failed for {asin}: {str(e)}")
                    continue
            
            logger.info(f"Completed AI analysis for {len(ai_results)} products")
            return ai_results
            
        except Exception as e:
            logger.error(f"Error running AI analysis: {str(e)}")
            return {}

class SessionManager:
    """Improved session state management with proper locking"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state with thread safety"""
        default_state = {
            # Data storage
            'uploaded_data': {},
            'processed_data': {},
            'scored_products': {},
            'ai_analysis_results': {},
            
            # UI state
            'current_tab': 0,
            'selected_product': None,
            'show_example_data': False,
            
            # Processing state
            'data_processed': False,
            'scores_calculated': False,
            'ai_analysis_complete': False,
            'processing_locked': False,
            
            # Module status
            'module_status': MODULES_LOADED.copy(),
            'api_status': {'available': False, 'error': 'Not tested'},
            
            # Timestamps
            'session_start': datetime.now(),
            'last_activity': datetime.now(),
            
            # Settings
            'auto_calculate_scores': True,
            'auto_run_ai_analysis': False,
            'show_debug_info': False,
            
            # Error tracking
            'error_count': 0,
            'last_error': None
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        logger.info("Session state initialized")
    
    @staticmethod
    def update_activity():
        """Update last activity timestamp"""
        st.session_state.last_activity = datetime.now()
    
    @staticmethod
    def check_session_timeout():
        """Check if session has timed out"""
        if 'last_activity' in st.session_state:
            time_diff = datetime.now() - st.session_state.last_activity
            hours_inactive = time_diff.total_seconds() / 3600
            
            if hours_inactive > APP_CONFIG['session_timeout_hours']:
                logger.warning(f"Session timeout after {hours_inactive:.1f} hours of inactivity")
                return True
        
        return False
    
    @staticmethod
    def safe_state_update(key: str, value: Any, force: bool = False):
        """Safely update session state with locking"""
        if force or not st.session_state.get('processing_locked', False):
            st.session_state[key] = value
            return True
        return False
    
    @staticmethod
    def load_example_data():
        """Load example data for demonstration"""
        try:
            logger.info("Loading example data")
            
            # Convert example data to proper format
            example_df = pd.DataFrame(EXAMPLE_DATA['products'])
            
            # Rename columns to match expected format
            column_mapping = {
                'asin': 'ASIN',
                'name': 'Product Name',
                'category': 'Category',
                'sku': 'SKU',
                'sales_30d': 'Last 30 Days Sales',
                'returns_30d': 'Last 30 Days Returns',
                'sales_365d': 'Last 365 Days Sales',
                'returns_365d': 'Last 365 Days Returns',
                'star_rating': 'Star Rating',
                'total_reviews': 'Total Reviews',
                'average_price': 'Average Price',
                'cost_per_unit': 'Cost per Unit',
                'description': 'Product Description'
            }
            
            example_df = example_df.rename(columns=column_mapping)
            
            # Store in session state
            st.session_state.uploaded_data = {
                'structured_data': example_df,
                'manual_reviews': EXAMPLE_DATA['reviews'],
                'manual_returns': EXAMPLE_DATA['returns']
            }
            
            st.session_state.show_example_data = True
            logger.info("Example data loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load example data: {str(e)}")
            st.error(f"Failed to load example data: {str(e)}")

class ApplicationController:
    """Main application controller with improved error handling"""
    
    def __init__(self):
        self.data_processor = SafeDataProcessor()
        self.dashboard = None
        
        # Initialize dashboard if available
        if dashboard_available:
            try:
                self.dashboard = ProfessionalDashboard()
                logger.info("Dashboard initialized")
            except Exception as e:
                logger.error(f"Failed to initialize dashboard: {str(e)}")
        
        # Initialize session
        SessionManager.initialize_session()
        
        # Update module status
        self._update_module_status()
    
    def _update_module_status(self):
        """Update module availability status"""
        try:
            st.session_state.module_status.update(MODULES_LOADED)
            
            # Check AI analyzer status
            if self.data_processor.ai_analyzer:
                api_status = self.data_processor.ai_analyzer.get_api_status()
                st.session_state.api_status = api_status
                st.session_state.module_status['ai_analysis'] = api_status.get('available', False)
            
            logger.debug(f"Module status updated: {st.session_state.module_status}")
            
        except Exception as e:
            logger.error(f"Error updating module status: {str(e)}")
    
    def handle_data_upload(self, upload_type: str, data: Any) -> bool:
        """Handle different types of data uploads with proper error handling"""
        try:
            SessionManager.update_activity()
            
            if not self.data_processor.upload_handler:
                st.error("❌ Upload functionality not available")
                return False
            
            success = False
            
            if upload_type == 'structured_file':
                file_data, filename = data
                result = self.data_processor.upload_handler.process_structured_file(file_data, filename)
                
                if result['success']:
                    # Merge with existing data
                    if 'structured_data' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['structured_data'] = result['data']
                    else:
                        existing_df = st.session_state.uploaded_data['structured_data']
                        combined_df = self._safe_concat_dataframes(existing_df, result['data'])
                        st.session_state.uploaded_data['structured_data'] = combined_df
                    
                    st.success(f"✅ Successfully uploaded {filename}")
                    success = True
                else:
                    errors = result.get('errors', ['Unknown error'])
                    error_msg = '; '.join([str(e) if isinstance(e, str) else e.get('message', str(e)) for e in errors])
                    st.error(f"❌ Upload failed: {error_msg}")
            
            elif upload_type == 'manual_entry':
                result = self.data_processor.upload_handler.process_manual_entry(data)
                
                if result['success']:
                    # Convert to DataFrame format and add to structured data
                    manual_data = result['data']
                    df_row = self._convert_manual_to_df_row(manual_data)
                    
                    # Add to structured data
                    if 'structured_data' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['structured_data'] = pd.DataFrame([df_row])
                    else:
                        existing_df = st.session_state.uploaded_data['structured_data']
                        # Remove existing entry with same ASIN if present
                        existing_df = existing_df[existing_df['ASIN'] != manual_data['asin']]
                        new_df = self._safe_concat_dataframes(existing_df, pd.DataFrame([df_row]))
                        st.session_state.uploaded_data['structured_data'] = new_df
                    
                    st.success(f"✅ Product {manual_data['asin']} saved successfully")
                    success = True
                else:
                    errors = result.get('errors', ['Unknown error'])
                    st.error(f"❌ Validation failed: {'; '.join(errors)}")
            
            elif upload_type == 'image_document':
                file_data, filename, content_type, asin = data
                result = self.data_processor.upload_handler.process_image_document(
                    file_data, filename, content_type, asin
                )
                
                if result['success']:
                    if 'documents' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['documents'] = []
                    
                    st.session_state.uploaded_data['documents'].append(result)
                    st.success(f"✅ Successfully processed {filename}")
                    
                    # Show extracted information
                    structured_data = result.get('structured_data', {})
                    if 'detected_asins' in structured_data:
                        st.info(f"🔍 Detected ASINs: {', '.join(structured_data['detected_asins'])}")
                    
                    success = True
                else:
                    errors = result.get('errors', ['Unknown error'])
                    st.error(f"❌ Processing failed: {'; '.join(errors)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling data upload: {str(e)}")
            st.error(f"❌ Upload error: {str(e)}")
            return False
    
    def _safe_concat_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Safely concatenate DataFrames with schema compatibility"""
        if df1.empty:
            return df2
        if df2.empty:
            return df1
        
        # Ensure all columns exist in both DataFrames
        all_columns = set(df1.columns) | set(df2.columns)
        for col in all_columns:
            if col not in df1.columns:
                df1[col] = None
            if col not in df2.columns:
                df2[col] = None
        
        # Reorder columns to match
        df1 = df1.reindex(columns=sorted(all_columns))
        df2 = df2.reindex(columns=sorted(all_columns))
        
        return pd.concat([df1, df2], ignore_index=True)
    
    def _convert_manual_to_df_row(self, manual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert manual entry data to DataFrame row format"""
        return {
            'ASIN': manual_data['asin'],
            'Product Name': manual_data.get('product_name', ''),
            'Category': manual_data.get('category', ''),
            'SKU': manual_data.get('sku', ''),
            'Last 30 Days Sales': manual_data.get('sales_30d', 0),
            'Last 30 Days Returns': manual_data.get('returns_30d', 0),
            'Last 365 Days Sales': manual_data.get('sales_365d'),
            'Last 365 Days Returns': manual_data.get('returns_365d'),
            'Star Rating': manual_data.get('star_rating'),
            'Total Reviews': manual_data.get('total_reviews'),
            'Average Price': manual_data.get('average_price'),
            'Cost per Unit': manual_data.get('cost_per_unit'),
            'Product Description': manual_data.get('description', '')
        }
    
    def process_data(self) -> bool:
        """Process all uploaded data with improved error handling"""
        try:
            SessionManager.update_activity()
            
            if not st.session_state.uploaded_data:
                st.warning("No data to process. Please upload data first.")
                return False
            
            # Set processing lock
            st.session_state.processing_locked = True
            
            try:
                with st.spinner("Processing uploaded data..."):
                    processed_data = self.data_processor.process_uploaded_data(st.session_state.uploaded_data)
                    st.session_state.processed_data = processed_data
                    st.session_state.data_processed = True
                
                # Auto-calculate scores if enabled
                if st.session_state.auto_calculate_scores and processed_data.get('products'):
                    self.calculate_scores()
                
                summary = processed_data.get('processing_summary', {})
                st.success(f"✅ Data processing complete! Processed {summary.get('structured_products', 0)} products")
                return True
                
            finally:
                # Always release lock
                st.session_state.processing_locked = False
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            st.error(f"❌ Data processing failed: {str(e)}")
            st.session_state.processing_locked = False
            return False
    
    def calculate_scores(self) -> bool:
        """Calculate composite scores with proper error handling"""
        try:
            SessionManager.update_activity()
            
            if not st.session_state.data_processed or not st.session_state.processed_data:
                st.warning("Please process data first before calculating scores.")
                return False
            
            with st.spinner("Calculating performance scores..."):
                scores = self.data_processor.calculate_scores(st.session_state.processed_data)
                st.session_state.scored_products = scores
                st.session_state.scores_calculated = True
            
            if scores:
                try:
                    avg_score = np.mean([score.composite_score for score in scores.values()])
                    st.success(f"✅ Calculated scores for {len(scores)} products (Avg: {avg_score:.1f}/100)")
                    return True
                except Exception as e:
                    logger.warning(f"Error calculating average score: {str(e)}")
                    st.success(f"✅ Calculated scores for {len(scores)} products")
                    return True
            else:
                st.warning("No scores calculated. Check that you have valid product data.")
                return False
            
        except Exception as e:
            logger.error(f"Error calculating scores: {str(e)}")
            st.error(f"❌ Score calculation failed: {str(e)}")
            return False
    
    def run_ai_analysis(self) -> bool:
        """Run AI analysis with proper error handling"""
        try:
            SessionManager.update_activity()
            
            if not st.session_state.api_status.get('available', False):
                st.error("❌ AI analysis not available. Please check your API configuration.")
                return False
            
            if not st.session_state.data_processed or not st.session_state.processed_data:
                st.warning("Please process data first before running AI analysis.")
                return False
            
            with st.spinner("Running AI analysis... This may take a few minutes."):
                ai_results = self.data_processor.run_ai_analysis(st.session_state.processed_data)
                st.session_state.ai_analysis_results = ai_results
                st.session_state.ai_analysis_complete = True
            
            if ai_results:
                st.success(f"✅ AI analysis complete for {len(ai_results)} products")
                return True
            else:
                st.warning("No AI analysis results generated. Check that you have review or return data.")
                return False
            
        except Exception as e:
            logger.error(f"Error running AI analysis: {str(e)}")
            st.error(f"❌ AI analysis failed: {str(e)}")
            return False
    
    def export_results(self, export_format: str = 'excel') -> Optional[bytes]:
        """Export analysis results with proper error handling"""
        try:
            SessionManager.update_activity()
            
            if not st.session_state.scored_products:
                st.warning("No scored products available for export.")
                return None
            
            if not scoring_available:
                st.error("Scoring module not available for export.")
                return None
            
            with st.spinner(f"Generating {export_format} export..."):
                if export_format == 'excel':
                    scores_df = self.data_processor.scoring_system.export_scores(
                        st.session_state.scored_products, 'dataframe'
                    )
                    
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        scores_df.to_excel(writer, sheet_name='Product Scores', index=False)
                        
                        # Add summary sheet
                        try:
                            summary_data = {
                                'Metric': ['Total Products', 'Average Score', 'Top Performers (70+)', 'Needs Improvement (<55)'],
                                'Value': [
                                    len(st.session_state.scored_products),
                                    f"{np.mean([s.composite_score for s in st.session_state.scored_products.values()]):.1f}",
                                    len([s for s in st.session_state.scored_products.values() if s.composite_score >= 70]),
                                    len([s for s in st.session_state.scored_products.values() if s.composite_score < 55])
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        except Exception as e:
                            logger.warning(f"Could not create summary sheet: {str(e)}")
                    
                    output.seek(0)
                    return output.getvalue()
                
                elif export_format == 'json':
                    export_data = self.data_processor.scoring_system.export_scores(
                        st.session_state.scored_products, 'detailed_report'
                    )
                    return json.dumps(export_data, indent=2, default=str).encode('utf-8')
                
                else:
                    st.error(f"Unsupported export format: {export_format}")
                    return None
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            st.error(f"❌ Export failed: {str(e)}")
            return None
    
    def run_application(self):
        """Main application entry point with comprehensive error handling"""
        try:
            # Check for session timeout
            if SessionManager.check_session_timeout():
                st.warning("⏰ Session has timed out due to inactivity. Please refresh the page.")
                if st.button("🔄 Refresh Session"):
                    st.rerun()
                return
            
            # Initialize dashboard if available
            if self.dashboard:
                try:
                    self.dashboard.initialize_app()
                except Exception as e:
                    logger.error(f"Dashboard initialization error: {str(e)}")
                    st.error("Dashboard initialization failed. Using minimal interface.")
                    self._render_minimal_interface()
                    return
            else:
                self._render_minimal_interface()
                return
            
            # Handle example data loading
            if st.session_state.get('load_example', False) or st.session_state.show_example_data:
                SessionManager.load_example_data()
                if st.session_state.uploaded_data and not st.session_state.data_processed:
                    self.process_data()
                st.session_state['load_example'] = False
            
            # Render main dashboard
            try:
                self.dashboard.render_main_dashboard()
            except Exception as e:
                logger.error(f"Dashboard rendering error: {str(e)}")
                st.error(f"Dashboard error: {str(e)}")
                self._render_minimal_interface()
            
            # Handle background processing triggers
            self._handle_processing_triggers()
            
        except Exception as e:
            logger.critical(f"Critical application error: {str(e)}")
            logger.critical(traceback.format_exc())
            self._render_error_interface(str(e))
    
    def _render_minimal_interface(self):
        """Render minimal interface when dashboard is not available"""
        st.title("🏥 Amazon Medical Device Listing Optimizer")
        st.error("⚠️ Dashboard module not available. Using minimal interface.")
        
        st.markdown("### System Status")
        for module, available in st.session_state.module_status.items():
            icon = "✅" if available else "❌"
            st.markdown(f"{icon} {module.replace('_', ' ').title()}")
        
        # Basic upload interface
        st.markdown("### Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            file_data = uploaded_file.read()
            success = self.handle_data_upload('structured_file', (file_data, uploaded_file.name))
            
            if success:
                if st.button("Process Data"):
                    self.process_data()
                
                if st.session_state.data_processed and st.button("Calculate Scores"):
                    self.calculate_scores()
    
    def _render_error_interface(self, error_message: str):
        """Render error interface when critical errors occur"""
        st.error("🚨 Critical Application Error")
        st.error("The application encountered a critical error and cannot continue.")
        st.error(f"Error: {error_message}")
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **System Information:**
            - App Version: {APP_CONFIG['version']}
            - Python Version: {APP_CONFIG['python_version']}
            - Streamlit Version: {APP_CONFIG['streamlit_version']}
            
            **Module Status:**
            """)
            
            for module, available in st.session_state.module_status.items():
                status = "Available" if available else "Not Available"
                st.markdown(f"- {module}: {status}")
            
            st.markdown(f"""
            **Common Solutions:**
            1. Refresh the page and try again
            2. Clear your browser cache
            3. Check that all required modules are installed
            4. Verify API key configuration
            5. Contact support if the issue persists
            
            **Support:** {APP_CONFIG['support_email']}
            """)
    
    def _handle_processing_triggers(self):
        """Handle background processing triggers with proper error handling"""
        try:
            # Auto-process data if new uploads detected
            if (st.session_state.uploaded_data and 
                not st.session_state.data_processed and 
                not st.session_state.get('processing_locked', False)):
                
                logger.info("Auto-processing new data")
                if self.process_data():
                    st.rerun()
            
            # Handle individual AI analysis trigger
            if 'run_individual_ai_analysis' in st.session_state:
                target_asin = st.session_state['run_individual_ai_analysis']
                del st.session_state['run_individual_ai_analysis']
                
                if self._run_individual_ai_analysis(target_asin):
                    st.success(f"✅ AI analysis complete for {target_asin}")
                    st.rerun()
            
            # Handle bulk AI analysis trigger
            if 'run_bulk_ai_analysis' in st.session_state:
                del st.session_state['run_bulk_ai_analysis']
                
                if self.run_ai_analysis():
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Error in processing triggers: {str(e)}")
    
    def _run_individual_ai_analysis(self, target_asin: str) -> bool:
        """Run AI analysis for a specific product with error handling"""
        try:
            if not st.session_state.data_processed or not st.session_state.processed_data:
                st.error("No processed data available for AI analysis.")
                return False
            
            if not st.session_state.api_status.get('available', False):
                st.error("❌ AI analysis not available. Please check your API configuration.")
                return False
            
            processed_data = st.session_state.processed_data
            products = processed_data.get('products', [])
            reviews_data = processed_data.get('reviews', {})
            returns_data = processed_data.get('returns', {})
            
            # Find target product
            target_product = next((p for p in products if p['asin'] == target_asin), None)
            if not target_product:
                st.error(f"Product {target_asin} not found in processed data.")
                return False
            
            # Get data for this product
            product_reviews = reviews_data.get(target_asin, [])
            product_returns = returns_data.get(target_asin, [])
            
            if not product_reviews and not product_returns:
                st.warning(f"No review or return data found for {target_asin}. AI analysis requires customer feedback data.")
                return False
            
            with st.spinner(f"Running AI analysis for {target_product['name']}..."):
                analysis_results = self.data_processor.ai_analyzer.analyze_product_comprehensive(
                    target_product, product_reviews, product_returns
                )
                
                # Store results
                if target_asin not in st.session_state.ai_analysis_results:
                    st.session_state.ai_analysis_results[target_asin] = {}
                
                st.session_state.ai_analysis_results[target_asin] = analysis_results
                st.session_state.ai_analysis_complete = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error running individual AI analysis for {target_asin}: {str(e)}")
            st.error(f"❌ AI analysis failed for {target_asin}: {str(e)}")
            return False

def main():
    """Application entry point with comprehensive error handling"""
    try:
        # Set Streamlit page config
        st.set_page_config(
            page_title=APP_CONFIG['title'],
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Create and run application
        app = ApplicationController()
        app.run_application()
        
    except Exception as e:
        logger.critical(f"Fatal application error: {str(e)}")
        logger.critical(traceback.format_exc())
        
        # Fallback error display
        st.error("🚨 Fatal Application Error")
        st.error("The application failed to start properly.")
        st.error(f"Error: {str(e)}")
        
        st.markdown(f"""
        **Emergency Support:**
        - Email: {APP_CONFIG['support_email']}
        - Version: {APP_CONFIG['version']}
        
        **Quick Actions:**
        1. Refresh the page
        2. Clear browser cache
        3. Check Python/Streamlit versions
        """)

if __name__ == "__main__":
    main()
