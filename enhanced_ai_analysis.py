"""
Enhanced AI Analysis Module - FIXED VERSION

**STABLE AI INTEGRATION**

Provides robust AI-powered analysis using OpenAI GPT-4o with comprehensive
error handling and fallback mechanisms.

Author: Assistant
Version: 4.0 - Production Stable
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

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

# Check for dependencies
requests, has_requests = safe_import('requests')

# API Configuration
API_TIMEOUT = 30
MAX_RETRIES = 2
MAX_TOKENS = 1500

class APIClient:
    """Robust OpenAI API client with error handling"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                for key_name in ["openai_api_key", "OPENAI_API_KEY"]:
                    if key_name in st.secrets:
                        logger.info("Found API key in Streamlit secrets")
                        return st.secrets[key_name]
        except:
            pass
        
        # Try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Found API key in environment")
            return api_key
        
        logger.warning("No OpenAI API key found")
        return None
    
    def is_available(self) -> bool:
        """Check if API is available"""
        return bool(self.api_key and has_requests)
    
    def call_api(self, messages: List[Dict[str, str]], 
                model: str = "gpt-4o",
                temperature: float = 0.1,
                max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """Make API call with retry logic"""
        
        if not self.is_available():
            return {
                "success": False,
                "error": "API not available - missing key or requests module",
                "result": None
            }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Making API call (attempt {attempt + 1}/{MAX_RETRIES})")
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "result": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "model": model
                    }
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    import time
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API error {response.status_code}: {response.text[:200]}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "result": None
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": "API timeout after multiple attempts",
                        "result": None
                    }
            except Exception as e:
                logger.error(f"API call error: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": f"API call failed: {str(e)}",
                        "result": None
                    }
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "result": None
        }

class PromptTemplates:
    """AI prompt templates for medical device analysis"""
    
    @staticmethod
    def create_review_analysis_prompt(product_info: Dict[str, Any], 
                                    review_summaries: List[str]) -> str:
        """Create prompt for review analysis"""
        
        product_name = product_info.get('name', 'Unknown Product')
        product_category = product_info.get('category', 'Medical Device')
        
        return f"""You are an expert medical device quality analyst. Analyze these customer reviews for actionable quality insights.

PRODUCT: {product_name}
CATEGORY: {product_category}
TOTAL REVIEWS: {len(review_summaries)}

CUSTOMER REVIEWS:
{chr(10).join(review_summaries)}

Provide analysis in this EXACT format:

## OVERALL SENTIMENT
[Positive/Negative/Mixed] with [confidence %]

## SAFETY CONCERNS  
[List any safety issues - mark CRITICAL if severe]

## TOP QUALITY ISSUES
1. [Most frequent issue with count]
2. [Second most frequent issue with count]
3. [Third most frequent issue with count]

## EFFECTIVENESS CONCERNS
[Any issues about product not working or being ineffective]

## COMFORT & USABILITY ISSUES
[Problems with ease of use, comfort, ergonomics]

## IMMEDIATE ACTIONS NEEDED
[Top 3 actions to take within 24-48 hours]

## LISTING IMPROVEMENTS
[Specific Amazon listing changes to prevent issues]

## CUSTOMER EDUCATION NEEDS
[What customers need to know to use product properly]

Focus on actionable insights that can improve customer satisfaction and safety."""
    
    @staticmethod
    def create_risk_assessment_prompt(analysis_summary: str) -> str:
        """Create prompt for risk assessment"""
        
        return f"""As a medical device risk management expert, assess the risk level based on this customer feedback analysis:

ANALYSIS SUMMARY:
{analysis_summary}

Provide assessment in this format:

## OVERALL RISK LEVEL
[Critical/High/Medium/Low] - [justification]

## SAFETY RISK ASSESSMENT
[Specific safety risks identified with severity]

## REGULATORY RISK
[Any FDA or regulatory compliance concerns]

## BUSINESS RISK
[Impact on sales, reputation, customer satisfaction]

## RISK MITIGATION PRIORITIES
1. [Highest priority action]
2. [Second priority action]  
3. [Third priority action]

## MONITORING RECOMMENDATIONS
[What to track going forward]

Be specific about risk levels and recommended actions."""

class ResponseParser:
    """Parse and structure AI responses"""
    
    @staticmethod
    def parse_review_analysis(content: str) -> Dict[str, Any]:
        """Parse review analysis response"""
        
        parsed = {
            'overall_sentiment': '',
            'safety_concerns': [],
            'top_quality_issues': [],
            'effectiveness_concerns': '',
            'comfort_usability_issues': '',
            'immediate_actions': [],
            'listing_improvements': '',
            'customer_education': ''
        }
        
        try:
            # Extract sections using regex
            sections = {
                'overall_sentiment': r'## OVERALL SENTIMENT\s*\n(.*?)(?=## |$)',
                'safety_concerns': r'## SAFETY CONCERNS\s*\n(.*?)(?=## |$)',
                'top_quality_issues': r'## TOP QUALITY ISSUES\s*\n(.*?)(?=## |$)',
                'effectiveness_concerns': r'## EFFECTIVENESS CONCERNS\s*\n(.*?)(?=## |$)',
                'comfort_usability_issues': r'## COMFORT & USABILITY ISSUES\s*\n(.*?)(?=## |$)',
                'immediate_actions': r'## IMMEDIATE ACTIONS NEEDED\s*\n(.*?)(?=## |$)',
                'listing_improvements': r'## LISTING IMPROVEMENTS\s*\n(.*?)(?=## |$)',
                'customer_education': r'## CUSTOMER EDUCATION NEEDS\s*\n(.*?)(?=## |$)'
            }
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    
                    # Parse lists for certain sections
                    if section_name in ['safety_concerns', 'top_quality_issues', 'immediate_actions']:
                        # Extract numbered or bulleted lists
                        items = []
                        lines = section_content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                                # Clean up the line
                                cleaned = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                                if cleaned:
                                    items.append(cleaned)
                        parsed[section_name] = items if items else [section_content]
                    else:
                        parsed[section_name] = section_content
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing review analysis: {str(e)}")
            return parsed
    
    @staticmethod
    def parse_risk_assessment(content: str) -> Dict[str, Any]:
        """Parse risk assessment response"""
        
        parsed = {
            'overall_risk_level': '',
            'safety_risk_assessment': '',
            'regulatory_risk': '',
            'business_risk': '',
            'risk_mitigation_priorities': [],
            'monitoring_recommendations': ''
        }
        
        try:
            sections = {
                'overall_risk_level': r'## OVERALL RISK LEVEL\s*\n(.*?)(?=## |$)',
                'safety_risk_assessment': r'## SAFETY RISK ASSESSMENT\s*\n(.*?)(?=## |$)',
                'regulatory_risk': r'## REGULATORY RISK\s*\n(.*?)(?=## |$)',
                'business_risk': r'## BUSINESS RISK\s*\n(.*?)(?=## |$)',
                'risk_mitigation_priorities': r'## RISK MITIGATION PRIORITIES\s*\n(.*?)(?=## |$)',
                'monitoring_recommendations': r'## MONITORING RECOMMENDATIONS\s*\n(.*?)(?=## |$)'
            }
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    
                    if section_name == 'risk_mitigation_priorities':
                        # Extract numbered list
                        items = []
                        lines = section_content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and line[0].isdigit():
                                cleaned = re.sub(r'^\d+\.\s*', '', line).strip()
                                if cleaned:
                                    items.append(cleaned)
                        parsed[section_name] = items if items else [section_content]
                    else:
                        parsed[section_name] = section_content
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing risk assessment: {str(e)}")
            return parsed

class EnhancedAIAnalyzer:
    """Main AI analyzer class with robust error handling"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        
        if self.api_client.is_available():
            logger.info("Enhanced AI Analyzer initialized - API available")
        else:
            logger.warning("Enhanced AI Analyzer initialized - API not available")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API availability status"""
        if not self.api_client.api_key:
            return {
                'available': False,
                'error': 'API key not configured',
                'suggestions': [
                    'Add OPENAI_API_KEY to environment variables',
                    'Add openai_api_key to Streamlit secrets'
                ]
            }
        
        if not has_requests:
            return {
                'available': False,
                'error': 'Requests module not available',
                'suggestions': ['Install requests module: pip install requests']
            }
        
        # Test API with minimal call
        test_response = self.api_client.call_api([
            {"role": "user", "content": "Test connection"}
        ], max_tokens=10)
        
        return {
            'available': test_response['success'],
            'error': test_response.get('error'),
            'model': test_response.get('model', 'gpt-4o')
        }
    
    def analyze_reviews_comprehensive(self, product_info: Dict[str, Any],
                                    reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive AI analysis of reviews"""
        
        if not self.api_client.is_available():
            return {
                'success': False,
                'error': 'AI analysis not available',
                'ai_analysis_available': False
            }
        
        try:
            # Prepare review summaries
            review_summaries = self._prepare_review_summaries(reviews)
            
            if not review_summaries:
                return {
                    'success': False,
                    'error': 'No review data to analyze',
                    'ai_analysis_available': False
                }
            
            # Create analysis prompt
            prompt = self.prompt_templates.create_review_analysis_prompt(
                product_info, review_summaries
            )
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": "You are an expert medical device quality analyst with deep knowledge of customer feedback analysis and risk assessment."},
                {"role": "user", "content": prompt}
            ])
            
            if not response['success']:
                return {
                    'success': False,
                    'error': response['error'],
                    'ai_analysis_available': False
                }
            
            # Parse response
            analysis_results = self.response_parser.parse_review_analysis(response['result'])
            
            # Add metadata
            analysis_results.update({
                'success': True,
                'ai_analysis_available': True,
                'analysis_timestamp': datetime.now().isoformat(),
                'reviews_analyzed': len(reviews),
                'model_used': response.get('model', 'gpt-4o'),
                'raw_response': response['result']  # For debugging
            })
            
            logger.info(f"AI analysis completed for {len(reviews)} reviews")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ai_analysis_available': False
            }
    
    def assess_risk_level(self, analysis_summary: str) -> Dict[str, Any]:
        """AI-powered risk assessment"""
        
        if not self.api_client.is_available():
            return {
                'success': False,
                'error': 'AI risk assessment not available'
            }
        
        try:
            # Create risk assessment prompt
            prompt = self.prompt_templates.create_risk_assessment_prompt(analysis_summary)
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": "You are a medical device risk management expert specializing in customer feedback risk analysis."},
                {"role": "user", "content": prompt}
            ])
            
            if not response['success']:
                return {
                    'success': False,
                    'error': response['error']
                }
            
            # Parse response
            risk_results = self.response_parser.parse_risk_assessment(response['result'])
            
            # Add metadata
            risk_results.update({
                'success': True,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': response.get('model', 'gpt-4o')
            })
            
            return risk_results
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_review_summaries(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """Prepare review summaries for AI analysis"""
        summaries = []
        
        # Sort by rating (lowest first for issue identification)
        sorted_reviews = sorted(reviews, key=lambda x: x.get('rating', 3))
        
        # Take up to 15 reviews (balanced sample)
        selected_reviews = sorted_reviews[:15]
        
        for i, review in enumerate(selected_reviews, 1):
            try:
                text = review.get('text', '').strip()
                if not text:
                    continue
                
                # Limit text length
                text = text[:400]
                
                # Add rating if available
                rating_text = ""
                if review.get('rating') is not None:
                    rating_text = f" (Rating: {review['rating']}/5)"
                
                # Add verification if available
                verified_text = ""
                if review.get('verified') == 'Verified Purchase':
                    verified_text = " [Verified]"
                
                summary = f"Review {i}{rating_text}{verified_text}: {text}"
                summaries.append(summary)
                
            except Exception as e:
                logger.warning(f"Error preparing review summary: {str(e)}")
                continue
        
        return summaries
    
    def analyze_product_comprehensive(self, product_info: Dict[str, Any],
                                    reviews: List[Dict[str, Any]] = None,
                                    returns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive product analysis (main interface)"""
        
        # Combine reviews and returns
        all_feedback = []
        
        if reviews:
            all_feedback.extend(reviews)
        
        if returns:
            # Convert returns to review format
            for return_item in returns:
                feedback_item = {
                    'text': return_item.get('return_reason', ''),
                    'type': 'return',
                    'rating': 1,  # Assume low satisfaction for returns
                    'date': return_item.get('date', ''),
                    'source': 'return_data'
                }
                all_feedback.append(feedback_item)
        
        if not all_feedback:
            return {
                'success': False,
                'error': 'No feedback data provided',
                'ai_analysis_available': False
            }
        
        # Run comprehensive analysis
        return self.analyze_reviews_comprehensive(product_info, all_feedback)

# Export main class
__all__ = ['EnhancedAIAnalyzer', 'APIClient']
