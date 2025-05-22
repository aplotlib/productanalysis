"""
Amazon Medical Device Listing Optimizer - Main Application

This is the main application file that integrates all modules:
- Upload Handler: Data import and validation
- Product Scoring: 0-100 composite scoring system  
- Enhanced AI Analysis: AI-powered insights and recommendations
- Professional Dashboard: Business-ready UI and visualizations

Author: Assistant
Version: 2.0
Production Ready: Yes
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

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

# Import our custom modules
try:
    from upload_handler import UploadHandler, UploadError, DataValidationError, MEDICAL_DEVICE_CATEGORIES
    from product_scoring import CompositeScoring, CompositeScore, SCORING_WEIGHTS, CATEGORY_BENCHMARKS
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AnalysisResult
    from dashboard import ProfessionalDashboard, DashboardRenderer, UIComponents
    
    MODULES_LOADED = {
        'upload_handler': True,
        'product_scoring': True, 
        'enhanced_ai_analysis': True,
        'dashboard': True
    }
    logger.info("All custom modules loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import custom modules: {str(e)}")
    st.error(f"Critical Error: Failed to load application modules. {str(e)}")
    st.stop()

# Application configuration
APP_CONFIG = {
    'title': 'Amazon Medical Device Listing Optimizer',
    'version': '2.0',
    'description': 'Professional performance analytics and optimization for medical device listings',
    'support_email': 'support@listingoptimizer.com',
    'max_products_free': 50,
    'session_timeout_hours': 4
}

# Example data for demonstration
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
        },
        {
            'asin': 'B0EXAMPLE1',
            'name': '4-Wheel Mobility Scooter',
            'category': 'Mobility Aids',
            'sku': 'VH-MOB-456',
            'sales_30d': 89,
            'returns_30d': 12,
            'sales_365d': 1068,
            'returns_365d': 144,
            'star_rating': 3.8,
            'total_reviews': 89,
            'average_price': 899.99,
            'cost_per_unit': 450.00,
            'description': 'Compact 4-wheel electric mobility scooter with 15-mile range and comfortable captain seat.'
        }
    ],
    'reviews': {
        'B0DT7NW5VY': [
            {'rating': 5, 'review_text': 'This rollator is amazing! Very stable and the seat is so comfortable. Great for my daily walks.', 'asin': 'B0DT7NW5VY'},
            {'rating': 4, 'review_text': 'Good quality rollator but the assembly instructions could be clearer. Works great once set up.', 'asin': 'B0DT7NW5VY'},
            {'rating': 2, 'review_text': 'The wheels started squeaking after just 2 weeks. Also heavier than expected.', 'asin': 'B0DT7NW5VY'},
            {'rating': 5, 'review_text': 'Perfect for my mom who has mobility issues. She loves the storage pouch and comfortable seat.', 'asin': 'B0DT7NW5VY'},
            {'rating': 3, 'review_text': 'Decent rollator but the brakes are a bit stiff. Good value for the price though.', 'asin': 'B0DT7NW5VY'}
        ],
        'B0DT8XYZ123': [
            {'rating': 5, 'review_text': 'Excellent shower chair! Very sturdy and the back support is perfect. Easy to adjust height.', 'asin': 'B0DT8XYZ123'},
            {'rating': 4, 'review_text': 'Good chair but the legs could be a bit more stable. Overall satisfied with purchase.', 'asin': 'B0DT8XYZ123'},
            {'rating': 5, 'review_text': 'This chair made showering so much safer for my elderly father. Highly recommend!', 'asin': 'B0DT8XYZ123'}
        ]
    },
    'returns': {
        'B0DT7NW5VY': [
            {'return_reason': 'Wheels started squeaking loudly after 2 weeks of use', 'asin': 'B0DT7NW5VY'},
            {'return_reason': 'Too heavy for elderly user to maneuver easily', 'asin': 'B0DT7NW5VY'},
            {'return_reason': 'Seat height not adjustable enough for user', 'asin': 'B0DT7NW5VY'},
            {'return_reason': 'Arrived with damaged storage pouch', 'asin': 'B0DT7NW5VY'}
        ],
        'B0DT8XYZ123': [
            {'return_reason': 'Chair legs not stable enough on wet surfaces', 'asin': 'B0DT8XYZ123'},
            {'return_reason': 'Back support too low for user comfort', 'asin': 'B0DT8XYZ123'}
        ]
    }
}

class DataProcessor:
    """Handles data processing and transformation between modules"""
    
    def __init__(self, upload_handler: UploadHandler, scoring_system: CompositeScoring, 
                 ai_analyzer: EnhancedAIAnalyzer):
        self.upload_handler = upload_handler
        self.scoring_system = scoring_system
        self.ai_analyzer = ai_analyzer
    
    def process_uploaded_data(self, uploaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process uploaded data and prepare for scoring and analysis"""
        
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
                    product = {
                        'asin': row['ASIN'],
                        'name': row.get('Product Name', f"Product {row['ASIN']}"),
                        'category': row.get('Category', 'Other'),
                        'sku': row.get('SKU', ''),
                        'sales_30d': int(row['Last 30 Days Sales']),
                        'returns_30d': int(row['Last 30 Days Returns']),
                        'sales_365d': int(row.get('Last 365 Days Sales', 0)) if pd.notna(row.get('Last 365 Days Sales', 0)) else None,
                        'returns_365d': int(row.get('Last 365 Days Returns', 0)) if pd.notna(row.get('Last 365 Days Returns', 0)) else None,
                        'star_rating': float(row.get('Star Rating', 0)) if pd.notna(row.get('Star Rating', 0)) else None,
                        'total_reviews': int(row.get('Total Reviews', 0)) if pd.notna(row.get('Total Reviews', 0)) else None,
                        'average_price': float(row.get('Average Price', 0)) if pd.notna(row.get('Average Price', 0)) else None,
                        'cost_per_unit': float(row.get('Cost per Unit', 0)) if pd.notna(row.get('Cost per Unit', 0)) else None,
                        'description': row.get('Product Description', '')
                    }
                    processed_data['products'].append(product)
                
                processed_data['processing_summary']['structured_products'] = len(processed_data['products'])
            
            # Process manual reviews
            if 'manual_reviews' in uploaded_data:
                for asin, reviews in uploaded_data['manual_reviews'].items():
                    processed_data['reviews'][asin] = reviews
                
                processed_data['processing_summary']['manual_reviews'] = sum(len(reviews) for reviews in processed_data['reviews'].values())
            
            # Process manual returns
            if 'manual_returns' in uploaded_data:
                for asin, returns in uploaded_data['manual_returns'].items():
                    processed_data['returns'][asin] = returns
                
                processed_data['processing_summary']['manual_returns'] = sum(len(returns) for returns in processed_data['returns'].values())
            
            # Process extracted documents
            if 'documents' in uploaded_data:
                doc_reviews, doc_returns = self._process_document_extractions(uploaded_data['documents'])
                
                # Merge with existing data
                for asin, reviews in doc_reviews.items():
                    if asin in processed_data['reviews']:
                        processed_data['reviews'][asin].extend(reviews)
                    else:
                        processed_data['reviews'][asin] = reviews
                
                for asin, returns in doc_returns.items():
                    if asin in processed_data['returns']:
                        processed_data['returns'][asin].extend(returns)
                    else:
                        processed_data['returns'][asin] = returns
            
            logger.info(f"Data processing complete: {processed_data['processing_summary']}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing uploaded data: {str(e)}")
            raise DataValidationError(f"Failed to process uploaded data: {str(e)}")
    
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
                if asin not in reviews:
                    reviews[asin] = []
                reviews[asin].extend(structured_data['reviews'])
            
            elif content_type == 'Return Reports' and 'returns' in structured_data:
                if asin not in returns:
                    returns[asin] = []
                returns[asin].extend(structured_data['returns'])
        
        return reviews, returns
    
    def calculate_scores(self, processed_data: Dict[str, Any]) -> Dict[str, CompositeScore]:
        """Calculate composite scores for all products"""
        
        try:
            products = processed_data['products']
            scores = {}
            
            if not products:
                logger.warning("No products available for scoring")
                return scores
            
            logger.info(f"Calculating scores for {len(products)} products")
            
            for product in products:
                try:
                    # Get AI analysis if available
                    asin = product['asin']
                    ai_analysis = st.session_state.ai_analysis_results.get(asin)
                    
                    # Calculate score
                    score = self.scoring_system.score_single_product(
                        product, 
                        products,  # Pass all products for competitive analysis
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
            raise
    
    def run_ai_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Dict[str, AnalysisResult]]:
        """Run AI analysis for all products with available data"""
        
        try:
            products = processed_data['products']
            reviews_data = processed_data['reviews']
            returns_data = processed_data['returns']
            
            ai_results = {}
            
            if not products:
                logger.warning("No products available for AI analysis")
                return ai_results
            
            # Check API status
            api_status = self.ai_analyzer.get_api_status()
            if not api_status.get('available', False):
                logger.warning(f"AI API not available: {api_status.get('error', 'Unknown error')}")
                return ai_results
            
            logger.info(f"Running AI analysis for {len(products)} products")
            
            for product in products:
                asin = product['asin']
                
                try:
                    # Get reviews and returns for this product
                    product_reviews = reviews_data.get(asin, [])
                    product_returns = returns_data.get(asin, [])
                    
                    if not product_reviews and not product_returns:
                        logger.debug(f"No review or return data for {asin}, skipping AI analysis")
                        continue
                    
                    # Run comprehensive analysis
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
            raise

class SessionManager:
    """Manages application session state and data persistence"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state with default values"""
        
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
            
            # Module status
            'module_status': MODULES_LOADED.copy(),
            'api_status': {'available': False, 'error': 'Not tested'},
            
            # Timestamps
            'session_start': datetime.now(),
            'last_activity': datetime.now(),
            
            # Settings
            'auto_calculate_scores': True,
            'auto_run_ai_analysis': False,  # Requires manual trigger due to API costs
            'show_debug_info': False
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
            raise

class ApplicationController:
    """Main application controller that orchestrates all components"""
    
    def __init__(self):
        # Initialize components
        self.upload_handler = UploadHandler()
        self.scoring_system = CompositeScoring()
        self.ai_analyzer = EnhancedAIAnalyzer()
        self.dashboard = ProfessionalDashboard()
        self.data_processor = DataProcessor(
            self.upload_handler, self.scoring_system, self.ai_analyzer
        )
        
        # Initialize session
        SessionManager.initialize_session()
        
        # Update module status
        self._update_module_status()
        
        logger.info("Application controller initialized")
    
    def _update_module_status(self):
        """Update module availability status"""
        
        try:
            # Check upload handler modules
            st.session_state.module_status.update(self.upload_handler.get_available_modules())
            
            # Check AI analyzer status
            api_status = self.ai_analyzer.get_api_status()
            st.session_state.api_status = api_status
            st.session_state.module_status['ai_analysis'] = api_status.get('available', False)
            
            logger.debug(f"Module status updated: {st.session_state.module_status}")
            
        except Exception as e:
            logger.error(f"Error updating module status: {str(e)}")
    
    def handle_data_upload(self, upload_type: str, data: Any) -> bool:
        """Handle different types of data uploads"""
        
        try:
            SessionManager.update_activity()
            
            if upload_type == 'structured_file':
                # Process structured file
                file_data, filename = data
                result = self.upload_handler.process_structured_file(file_data, filename)
                
                if result['success']:
                    if 'structured_data' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['structured_data'] = result['data']
                    else:
                        # Merge with existing data
                        existing_df = st.session_state.uploaded_data['structured_data']
                        combined_df = pd.concat([existing_df, result['data']], ignore_index=True)
                        st.session_state.uploaded_data['structured_data'] = combined_df
                    
                    st.success(f"✅ Successfully uploaded {filename}")
                    return True
                else:
                    st.error(f"❌ Upload failed: {', '.join(result.get('errors', ['Unknown error']))}")
                    return False
            
            elif upload_type == 'manual_entry':
                # Process manual entry
                result = self.upload_handler.process_manual_entry(data)
                
                if result['success']:
                    # Convert to DataFrame format and add to structured data
                    manual_data = result['data']
                    
                    # Create DataFrame row
                    df_row = {
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
                    
                    # Add to structured data
                    if 'structured_data' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['structured_data'] = pd.DataFrame([df_row])
                    else:
                        existing_df = st.session_state.uploaded_data['structured_data']
                        # Remove existing entry with same ASIN if present
                        existing_df = existing_df[existing_df['ASIN'] != manual_data['asin']]
                        new_df = pd.concat([existing_df, pd.DataFrame([df_row])], ignore_index=True)
                        st.session_state.uploaded_data['structured_data'] = new_df
                    
                    st.success(f"✅ Product {manual_data['asin']} saved successfully")
                    return True
                else:
                    st.error(f"❌ Validation failed: {', '.join(result.get('errors', ['Unknown error']))}")
                    return False
            
            elif upload_type == 'image_document':
                # Process image/document
                file_data, filename, content_type, asin = data
                result = self.upload_handler.process_image_document(file_data, filename, content_type, asin)
                
                if result['success']:
                    if 'documents' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['documents'] = []
                    
                    st.session_state.uploaded_data['documents'].append(result)
                    
                    # Show detected ASINs and product info
                    structured_data = result.get('structured_data', {})
                    
                    if 'detected_asins' in structured_data:
                        detected_asins = structured_data['detected_asins']
                        st.success(f"✅ Successfully processed {filename}")
                        st.info(f"🔍 Detected ASINs: {', '.join(detected_asins)}")
                        
                        # Auto-associate with primary ASIN if not manually specified
                        if not asin and 'primary_asin' in structured_data:
                            result['asin'] = structured_data['primary_asin']
                            st.info(f"📎 Auto-associated with ASIN: {structured_data['primary_asin']}")
                    
                    if 'product_info' in structured_data:
                        product_info = structured_data['product_info']
                        info_items = []
                        if 'detected_price' in product_info:
                            info_items.append(f"Price: ${product_info['detected_price']:.2f}")
                        if 'detected_rating' in product_info:
                            info_items.append(f"Rating: {product_info['detected_rating']}★")
                        if 'detected_title' in product_info:
                            info_items.append(f"Title: {product_info['detected_title'][:50]}...")
                        
                        if info_items:
                            st.info(f"📋 Detected info: {' | '.join(info_items)}")
                    
                    return True
                else:
                    st.error(f"❌ Processing failed: {', '.join(result.get('errors', ['Unknown error']))}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling data upload: {str(e)}")
            st.error(f"❌ Upload error: {str(e)}")
            return False
    
    def process_data(self) -> bool:
        """Process all uploaded data"""
        
        try:
            SessionManager.update_activity()
            
            if not st.session_state.uploaded_data:
                st.warning("No data to process. Please upload data first.")
                return False
            
            with st.spinner("Processing uploaded data..."):
                processed_data = self.data_processor.process_uploaded_data(st.session_state.uploaded_data)
                st.session_state.processed_data = processed_data
                st.session_state.data_processed = True
            
            # Auto-calculate scores if enabled
            if st.session_state.auto_calculate_scores and processed_data['products']:
                self.calculate_scores()
            
            st.success(f"✅ Data processing complete! {processed_data['processing_summary']}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            st.error(f"❌ Data processing failed: {str(e)}")
            return False
    
    def calculate_scores(self) -> bool:
        """Calculate composite scores for all products"""
        
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
                avg_score = np.mean([score.composite_score for score in scores.values()])
                st.success(f"✅ Calculated scores for {len(scores)} products (Avg: {avg_score:.1f}/100)")
                return True
            else:
                st.warning("No scores calculated. Check that you have valid product data.")
                return False
            
        except Exception as e:
            logger.error(f"Error calculating scores: {str(e)}")
            st.error(f"❌ Score calculation failed: {str(e)}")
            return False
    
    def run_ai_analysis(self) -> bool:
        """Run AI analysis for all products"""
        
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
        """Export analysis results"""
        
        try:
            SessionManager.update_activity()
            
            if not st.session_state.scored_products:
                st.warning("No scored products available for export.")
                return None
            
            with st.spinner(f"Generating {export_format} export..."):
                if export_format == 'excel':
                    # Generate Excel export
                    scores_df = self.scoring_system.export_scores(st.session_state.scored_products, 'dataframe')
                    
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        scores_df.to_excel(writer, sheet_name='Product Scores', index=False)
                        
                        # Add summary sheet
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
                    
                    output.seek(0)
                    return output.getvalue()
                
                elif export_format == 'json':
                    # Generate JSON export
                    export_data = self.scoring_system.export_scores(st.session_state.scored_products, 'detailed_report')
                    return json.dumps(export_data, indent=2, default=str).encode('utf-8')
                
                else:
                    st.error(f"Unsupported export format: {export_format}")
                    return None
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            st.error(f"❌ Export failed: {str(e)}")
            return None
    
    def run_application(self):
        """Main application entry point"""
        
        try:
            # Initialize dashboard
            self.dashboard.initialize_app()
            
            # Check for session timeout
            if SessionManager.check_session_timeout():
                st.warning("⏰ Session has timed out due to inactivity. Please refresh the page.")
                if st.button("🔄 Refresh Session"):
                    st.rerun()
                return
            
            # Handle example data loading
            if st.session_state.get('load_example', False) or st.session_state.show_example_data:
                SessionManager.load_example_data()
                if st.session_state.uploaded_data and not st.session_state.data_processed:
                    self.process_data()
                st.session_state['load_example'] = False
            
            # Render main dashboard
            self.dashboard.render_main_dashboard()
            
            # Handle background processing triggers
            self._handle_processing_triggers()
            
        except Exception as e:
            logger.error(f"Critical application error: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"❌ Critical Error: {str(e)}")
            st.error("Please refresh the page and try again. If the problem persists, contact support.")
    
    def _handle_processing_triggers(self):
        """Handle background processing triggers"""
        
        # Auto-process data if new uploads detected
        if (st.session_state.uploaded_data and 
            not st.session_state.data_processed and 
            not st.session_state.get('processing_in_progress', False)):
            
            st.session_state['processing_in_progress'] = True
            if self.process_data():
                st.rerun()
            st.session_state['processing_in_progress'] = False
        
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
    
    def _run_individual_ai_analysis(self, target_asin: str) -> bool:
        """Run AI analysis for a specific product"""
        
        try:
            if not st.session_state.data_processed or not st.session_state.processed_data:
                st.error("No processed data available for AI analysis.")
                return False
            
            # Check API status
            if not st.session_state.api_status.get('available', False):
                st.error("❌ AI analysis not available. Please check your API configuration.")
                return False
            
            processed_data = st.session_state.processed_data
            products = processed_data['products']
            reviews_data = processed_data['reviews']
            returns_data = processed_data['returns']
            
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
            
            # Set progress flag
            st.session_state['ai_analysis_in_progress'] = True
            
            with st.spinner(f"Running AI analysis for {target_product['name']}..."):
                # Run comprehensive analysis for this product
                analysis_results = self.ai_analyzer.analyze_product_comprehensive(
                    target_product, product_reviews, product_returns
                )
                
                # Store results
                if target_asin not in st.session_state.ai_analysis_results:
                    st.session_state.ai_analysis_results[target_asin] = {}
                
                st.session_state.ai_analysis_results[target_asin] = analysis_results
                st.session_state.ai_analysis_complete = True
            
            # Clear progress flag
            st.session_state['ai_analysis_in_progress'] = False
            return True
            
        except Exception as e:
            logger.error(f"Error running individual AI analysis for {target_asin}: {str(e)}")
            st.error(f"❌ AI analysis failed for {target_asin}: {str(e)}")
            st.session_state['ai_analysis_in_progress'] = False
            return False

def main():
    """Application entry point"""
    
    try:
        # Create and run application
        app = ApplicationController()
        app.run_application()
        
    except Exception as e:
        logger.critical(f"Fatal application error: {str(e)}")
        logger.critical(traceback.format_exc())
        
        st.error("🚨 Fatal Application Error")
        st.error("The application encountered a critical error and cannot continue.")
        st.error(f"Error: {str(e)}")
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common solutions:**
            1. Refresh the page and try again
            2. Clear your browser cache
            3. Check that all required modules are installed
            4. Verify API key configuration
            5. Contact support if the issue persists
            """)
            
            st.markdown(f"**Support Email:** {APP_CONFIG['support_email']}")
            st.markdown(f"**App Version:** {APP_CONFIG['version']}")

if __name__ == "__main__":
    main()
