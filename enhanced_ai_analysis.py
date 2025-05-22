"""
Enhanced AI Analysis Module for Amazon Medical Device Listing Optimizer

This module provides precise, product-specific AI analysis with:
- Structured, actionable recommendations
- Product-specific insights based on actual data
- Professional formatting for business users
- Efficient token management for large datasets
- Robust error handling and fallbacks

Author: Assistant
Version: 2.0
"""

import logging
import os
import json
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing OpenAI for fallback SDK support
try:
    import openai
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False
    logger.warning("OpenAI SDK not available, using direct API calls")

# Constants
API_TIMEOUT = 45
MAX_RETRIES = 3
TOKEN_LIMIT_GPT4 = 8000  # Conservative limit for GPT-4
MAX_ITEMS_DETAILED_ANALYSIS = 15
MAX_ITEMS_BULK_ANALYSIS = 50

# Medical device specific analysis keywords
MEDICAL_ANALYSIS_KEYWORDS = {
    'safety_concerns': [
        'unsafe', 'dangerous', 'injury', 'hurt', 'hazard', 'broken', 'defective',
        'sharp', 'cuts', 'pinch', 'trap', 'unstable', 'tip', 'fall'
    ],
    'efficacy_issues': [
        'doesnt work', 'ineffective', 'no relief', 'no help', 'useless',
        'waste of money', 'doesnt help', 'no improvement', 'no difference'
    ],
    'comfort_problems': [
        'uncomfortable', 'painful', 'hurts', 'sore', 'irritating', 'rough',
        'hard', 'stiff', 'tight', 'pinches', 'digs in', 'pressure'
    ],
    'durability_concerns': [
        'broke', 'broken', 'fell apart', 'cheaply made', 'flimsy', 'weak',
        'tore', 'ripped', 'cracked', 'split', 'bent', 'snapped'
    ],
    'sizing_fit_issues': [
        'too small', 'too big', 'wrong size', 'doesnt fit', 'tight', 'loose',
        'runs small', 'runs large', 'sizing chart wrong', 'measurements off'
    ],
    'positive_indicators': [
        'love it', 'excellent', 'perfect', 'amazing', 'great quality',
        'highly recommend', 'best purchase', 'life changer', 'works great'
    ]
}

# Amazon listing optimization focus areas
LISTING_OPTIMIZATION_AREAS = {
    'title_optimization': [
        'keywords', 'search terms', 'findability', 'character limit',
        'brand name', 'model number', 'key benefits'
    ],
    'bullet_points': [
        'benefits over features', 'pain points', 'use cases',
        'dimensions', 'materials', 'certifications'
    ],
    'images': [
        'lifestyle shots', 'size reference', 'feature callouts',
        'before/after', 'usage demonstration', 'packaging'
    ],
    'description': [
        'detailed benefits', 'comparison chart', 'use instructions',
        'care instructions', 'warranty info', 'compliance'
    ]
}

@dataclass
class AnalysisResult:
    """Structured container for analysis results"""
    success: bool
    analysis_type: str
    product_asin: str
    product_name: str
    timestamp: str
    summary: Dict[str, Any]
    detailed_findings: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    confidence_score: float
    data_quality: Dict[str, Any]
    errors: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class RecommendationItem:
    """Structured recommendation with priority and expected impact"""
    category: str
    issue: str
    recommendation: str
    priority: str  # High, Medium, Low
    expected_impact: str
    implementation_effort: str  # Easy, Medium, Hard
    specific_action: str
    success_metric: str

class APIClient:
    """Handles OpenAI API communication with robust error handling"""
    
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
                        logger.info(f"Found API key in Streamlit secrets")
                        return st.secrets[key_name]
        except (ImportError, AttributeError, KeyError):
            pass
        
        # Try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Found API key in environment variables")
            return api_key
        
        logger.warning("No API key found")
        return None
    
    def call_api(self, messages: List[Dict[str, str]], model: str = "gpt-4o", 
                temperature: float = 0.1, max_tokens: int = 1500) -> Dict[str, Any]:
        """Make API call with retry logic and error handling"""
        
        if not self.api_key:
            return {
                "success": False,
                "error": "API key not configured",
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
                    data=json.dumps(payload),
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
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    return {
                        "success": False,
                        "error": f"API error {response.status_code}: {response.text}",
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

class DataPreprocessor:
    """Prepares and optimizes data for AI analysis"""
    
    @staticmethod
    def analyze_review_quality(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality and characteristics of review data"""
        if not reviews:
            return {"total": 0, "quality": "no_data"}
        
        total_reviews = len(reviews)
        ratings = [r.get('rating') for r in reviews if r.get('rating')]
        texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        
        # Calculate text length statistics
        text_lengths = [len(text) for text in texts if text]
        avg_text_length = np.mean(text_lengths) if text_lengths else 0
        
        # Rating distribution
        rating_dist = Counter(ratings) if ratings else {}
        
        # Quality assessment
        quality = "high"
        if avg_text_length < 50:
            quality = "low"
        elif avg_text_length < 150:
            quality = "medium"
        
        return {
            "total": total_reviews,
            "quality": quality,
            "avg_text_length": round(avg_text_length, 1),
            "rating_distribution": dict(rating_dist),
            "has_detailed_text": len([t for t in texts if len(t) > 100]),
            "avg_rating": round(np.mean(ratings), 2) if ratings else None
        }
    
    @staticmethod
    def analyze_return_quality(returns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality and characteristics of return data"""
        if not returns:
            return {"total": 0, "quality": "no_data"}
        
        total_returns = len(returns)
        reasons = [r.get('return_reason', '') for r in returns if r.get('return_reason')]
        
        # Calculate reason length statistics
        reason_lengths = [len(reason) for reason in reasons if reason]
        avg_reason_length = np.mean(reason_lengths) if reason_lengths else 0
        
        # Quality assessment
        quality = "high"
        if avg_reason_length < 20:
            quality = "low"
        elif avg_reason_length < 50:
            quality = "medium"
        
        return {
            "total": total_returns,
            "quality": quality,
            "avg_reason_length": round(avg_reason_length, 1),
            "detailed_reasons": len([r for r in reasons if len(r) > 50])
        }
    
    @staticmethod
    def create_smart_summary(reviews: List[Dict[str, Any]], 
                           returns: List[Dict[str, Any]], 
                           max_tokens: int = 4000) -> Dict[str, str]:
        """Create intelligent summary prioritizing most important data"""
        
        # Analyze data quality
        review_quality = DataPreprocessor.analyze_review_quality(reviews)
        return_quality = DataPreprocessor.analyze_return_quality(returns)
        
        summary = {}
        
        # Summarize reviews intelligently
        if reviews:
            summary['reviews'] = DataPreprocessor._summarize_reviews_smart(
                reviews, review_quality, max_tokens // 2
            )
        
        # Summarize returns intelligently
        if returns:
            summary['returns'] = DataPreprocessor._summarize_returns_smart(
                returns, return_quality, max_tokens // 2
            )
        
        return summary
    
    @staticmethod
    def _summarize_reviews_smart(reviews: List[Dict[str, Any]], 
                               quality_info: Dict[str, Any], 
                               max_tokens: int) -> str:
        """Create smart review summary prioritizing valuable insights"""
        
        # Sort reviews by value for analysis
        def review_value_score(review):
            rating = review.get('rating', 3)
            text_length = len(review.get('review_text', ''))
            
            # Prioritize extreme ratings and detailed reviews
            rating_value = abs(rating - 3) * 2  # Extreme ratings more valuable
            length_value = min(text_length / 100, 3)  # Cap length value
            
            return rating_value + length_value
        
        sorted_reviews = sorted(reviews, key=review_value_score, reverse=True)
        
        # Take top reviews up to token limit
        selected_reviews = sorted_reviews[:MAX_ITEMS_DETAILED_ANALYSIS]
        
        # Create summary
        summary_parts = [
            f"REVIEW ANALYSIS ({len(reviews)} total reviews)",
            f"Average Rating: {quality_info.get('avg_rating', 'N/A')}",
            f"Rating Distribution: {quality_info.get('rating_distribution', {})}",
            ""
        ]
        
        # Add detailed reviews
        summary_parts.append("DETAILED REVIEWS (most valuable for analysis):")
        for i, review in enumerate(selected_reviews, 1):
            rating = review.get('rating', 'N/A')
            text = review.get('review_text', '')[:300]  # Truncate long reviews
            summary_parts.append(f"{i}. [{rating}★] {text}")
        
        # Add pattern analysis
        all_text = ' '.join([r.get('review_text', '') for r in reviews]).lower()
        patterns = []
        
        for category, keywords in MEDICAL_ANALYSIS_KEYWORDS.items():
            mentions = sum(1 for keyword in keywords if keyword in all_text)
            if mentions > 0:
                patterns.append(f"{category}: {mentions} mentions")
        
        if patterns:
            summary_parts.extend(["", "PATTERN ANALYSIS:"] + patterns)
        
        return '\n'.join(summary_parts)
    
    @staticmethod
    def _summarize_returns_smart(returns: List[Dict[str, Any]], 
                               quality_info: Dict[str, Any], 
                               max_tokens: int) -> str:
        """Create smart return summary with categorization"""
        
        # Categorize returns
        categories = defaultdict(list)
        for return_item in returns:
            reason = return_item.get('return_reason', '').lower()
            
            # Simple categorization based on keywords
            categorized = False
            for category, keywords in MEDICAL_ANALYSIS_KEYWORDS.items():
                if any(keyword in reason for keyword in keywords):
                    categories[category].append(return_item)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(return_item)
        
        # Create summary
        summary_parts = [
            f"RETURN ANALYSIS ({len(returns)} total returns)",
            ""
        ]
        
        # Add category breakdown
        summary_parts.append("RETURN CATEGORIES:")
        for category, items in categories.items():
            if items:
                percentage = (len(items) / len(returns)) * 100
                summary_parts.append(f"- {category}: {len(items)} returns ({percentage:.1f}%)")
        
        # Add detailed return reasons
        summary_parts.extend(["", "DETAILED RETURN REASONS:"])
        for i, return_item in enumerate(returns[:MAX_ITEMS_DETAILED_ANALYSIS], 1):
            reason = return_item.get('return_reason', '')
            summary_parts.append(f"{i}. {reason}")
        
        return '\n'.join(summary_parts)

class ProductSpecificAnalyzer:
    """Provides product-specific, targeted AI analysis"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.preprocessor = DataPreprocessor()
    
    def analyze_reviews_comprehensive(self, product_info: Dict[str, Any], 
                                    reviews: List[Dict[str, Any]]) -> AnalysisResult:
        """Comprehensive, product-specific review analysis"""
        
        try:
            # Analyze data quality
            review_quality = self.preprocessor.analyze_review_quality(reviews)
            
            if review_quality['total'] == 0:
                return AnalysisResult(
                    success=False,
                    analysis_type="review_analysis",
                    product_asin=product_info.get('asin', 'unknown'),
                    product_name=product_info.get('name', 'Unknown Product'),
                    timestamp=datetime.now().isoformat(),
                    summary={},
                    detailed_findings={},
                    recommendations=[],
                    confidence_score=0.0,
                    data_quality=review_quality,
                    errors=["No review data available"]
                )
            
            # Create targeted summary
            review_summary = self.preprocessor._summarize_reviews_smart(
                reviews, review_quality, 3000
            )
            
            # Create product-specific prompt
            system_prompt = self._create_review_system_prompt(product_info)
            user_prompt = self._create_review_analysis_prompt(
                product_info, review_summary, review_quality
            )
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], max_tokens=1800)
            
            if not response['success']:
                return AnalysisResult(
                    success=False,
                    analysis_type="review_analysis",
                    product_asin=product_info.get('asin', 'unknown'),
                    product_name=product_info.get('name', 'Unknown Product'),
                    timestamp=datetime.now().isoformat(),
                    summary={},
                    detailed_findings={},
                    recommendations=[],
                    confidence_score=0.0,
                    data_quality=review_quality,
                    errors=[response['error']]
                )
            
            # Parse and structure the response
            analysis_content = response['result']
            structured_analysis = self._parse_review_analysis(analysis_content)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(review_quality, len(reviews))
            
            return AnalysisResult(
                success=True,
                analysis_type="review_analysis",
                product_asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                timestamp=datetime.now().isoformat(),
                summary=structured_analysis.get('summary', {}),
                detailed_findings=structured_analysis.get('findings', {}),
                recommendations=structured_analysis.get('recommendations', []),
                confidence_score=confidence,
                data_quality=review_quality
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive review analysis: {str(e)}")
            return AnalysisResult(
                success=False,
                analysis_type="review_analysis",
                product_asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                timestamp=datetime.now().isoformat(),
                summary={},
                detailed_findings={},
                recommendations=[],
                confidence_score=0.0,
                data_quality={},
                errors=[str(e)]
            )
    
    def analyze_returns_comprehensive(self, product_info: Dict[str, Any], 
                                    returns: List[Dict[str, Any]]) -> AnalysisResult:
        """Comprehensive, product-specific return analysis"""
        
        try:
            # Analyze data quality
            return_quality = self.preprocessor.analyze_return_quality(returns)
            
            if return_quality['total'] == 0:
                return AnalysisResult(
                    success=False,
                    analysis_type="return_analysis",
                    product_asin=product_info.get('asin', 'unknown'),
                    product_name=product_info.get('name', 'Unknown Product'),
                    timestamp=datetime.now().isoformat(),
                    summary={},
                    detailed_findings={},
                    recommendations=[],
                    confidence_score=0.0,
                    data_quality=return_quality,
                    errors=["No return data available"]
                )
            
            # Create targeted summary
            return_summary = self.preprocessor._summarize_returns_smart(
                returns, return_quality, 3000
            )
            
            # Create product-specific prompt
            system_prompt = self._create_return_system_prompt(product_info)
            user_prompt = self._create_return_analysis_prompt(
                product_info, return_summary, return_quality
            )
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], max_tokens=1800)
            
            if not response['success']:
                return AnalysisResult(
                    success=False,
                    analysis_type="return_analysis",
                    product_asin=product_info.get('asin', 'unknown'),
                    product_name=product_info.get('name', 'Unknown Product'),
                    timestamp=datetime.now().isoformat(),
                    summary={},
                    detailed_findings={},
                    recommendations=[],
                    confidence_score=0.0,
                    data_quality=return_quality,
                    errors=[response['error']]
                )
            
            # Parse and structure the response
            analysis_content = response['result']
            structured_analysis = self._parse_return_analysis(analysis_content)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(return_quality, len(returns))
            
            return AnalysisResult(
                success=True,
                analysis_type="return_analysis",
                product_asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                timestamp=datetime.now().isoformat(),
                summary=structured_analysis.get('summary', {}),
                detailed_findings=structured_analysis.get('findings', {}),
                recommendations=structured_analysis.get('recommendations', []),
                confidence_score=confidence,
                data_quality=return_quality
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive return analysis: {str(e)}")
            return AnalysisResult(
                success=False,
                analysis_type="return_analysis",
                product_asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                timestamp=datetime.now().isoformat(),
                summary={},
                detailed_findings={},
                recommendations=[],
                confidence_score=0.0,
                data_quality={},
                errors=[str(e)]
            )
    
    def generate_listing_optimization(self, product_info: Dict[str, Any], 
                                    review_analysis: Optional[AnalysisResult] = None,
                                    return_analysis: Optional[AnalysisResult] = None) -> AnalysisResult:
        """Generate comprehensive listing optimization recommendations"""
        
        try:
            # Create optimization prompt with all available data
            system_prompt = self._create_optimization_system_prompt(product_info)
            user_prompt = self._create_optimization_prompt(
                product_info, review_analysis, return_analysis
            )
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], max_tokens=2000)
            
            if not response['success']:
                return AnalysisResult(
                    success=False,
                    analysis_type="listing_optimization",
                    product_asin=product_info.get('asin', 'unknown'),
                    product_name=product_info.get('name', 'Unknown Product'),
                    timestamp=datetime.now().isoformat(),
                    summary={},
                    detailed_findings={},
                    recommendations=[],
                    confidence_score=0.0,
                    data_quality={},
                    errors=[response['error']]
                )
            
            # Parse optimization recommendations
            optimization_content = response['result']
            structured_optimization = self._parse_optimization_analysis(optimization_content)
            
            # Calculate confidence based on available data
            confidence = 0.7  # Base confidence
            if review_analysis and review_analysis.success:
                confidence += 0.15
            if return_analysis and return_analysis.success:
                confidence += 0.15
            
            return AnalysisResult(
                success=True,
                analysis_type="listing_optimization",
                product_asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                timestamp=datetime.now().isoformat(),
                summary=structured_optimization.get('summary', {}),
                detailed_findings=structured_optimization.get('findings', {}),
                recommendations=structured_optimization.get('recommendations', []),
                confidence_score=min(confidence, 1.0),
                data_quality={"has_review_data": review_analysis is not None,
                            "has_return_data": return_analysis is not None}
            )
            
        except Exception as e:
            logger.error(f"Error in listing optimization: {str(e)}")
            return AnalysisResult(
                success=False,
                analysis_type="listing_optimization",
                product_asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                timestamp=datetime.now().isoformat(),
                summary={},
                detailed_findings={},
                recommendations=[],
                confidence_score=0.0,
                data_quality={},
                errors=[str(e)]
            )
    
    def _create_review_system_prompt(self, product_info: Dict[str, Any]) -> str:
        """Create product-specific system prompt for review analysis"""
        category = product_info.get('category', 'Medical Device')
        return f"""You are an expert Amazon listing optimization specialist with deep expertise in {category} products.

Your role is to analyze customer reviews for medical devices and provide specific, actionable insights that help listing managers:
1. Identify specific product issues that drive negative reviews
2. Understand what customers value most about the product
3. Find opportunities to improve the Amazon listing
4. Reduce return rates through better customer expectations

Focus on being specific, data-driven, and actionable. Avoid generic advice."""
    
    def _create_review_analysis_prompt(self, product_info: Dict[str, Any], 
                                     review_summary: str, 
                                     quality_info: Dict[str, Any]) -> str:
        """Create detailed prompt for review analysis"""
        
        return_rate = product_info.get('return_rate_30d', 'Unknown')
        star_rating = product_info.get('star_rating', 'Unknown')
        
        return f"""Analyze these customer reviews for: {product_info.get('name', 'Unknown Product')}
ASIN: {product_info.get('asin', 'Unknown')}
Category: {product_info.get('category', 'Medical Device')}
Current Return Rate: {return_rate}%
Current Star Rating: {star_rating}

{review_summary}

Provide analysis in this EXACT format:

## SUMMARY
- Overall sentiment: [Positive/Mixed/Negative]
- Primary satisfaction drivers: [List top 3]
- Primary complaint categories: [List top 3]
- Listing accuracy assessment: [Good/Fair/Poor with explanation]

## KEY FINDINGS
### Positive Feedback Themes
[List 3-5 specific things customers love with examples]

### Negative Feedback Themes  
[List 3-5 specific issues with examples and frequency]

### Quality & Durability Issues
[Specific problems mentioned with frequency]

### Sizing & Fit Issues
[Specific sizing problems with examples]

### Usage & Comfort Issues
[Specific usability problems with examples]

## IMMEDIATE ACTION ITEMS
1. **[Priority Level] [Category]**: [Specific issue] 
   - Action: [Specific recommendation]
   - Expected Impact: [Specific outcome]

2. **[Priority Level] [Category]**: [Specific issue]
   - Action: [Specific recommendation] 
   - Expected Impact: [Specific outcome]

[Continue for 3-5 action items]

Base your analysis on the actual review data provided. Be specific and cite examples from the reviews."""
    
    def _create_return_system_prompt(self, product_info: Dict[str, Any]) -> str:
        """Create product-specific system prompt for return analysis"""
        category = product_info.get('category', 'Medical Device')
        return f"""You are an expert Amazon return reduction specialist focusing on {category} products.

Your goal is to analyze return reasons and provide specific, implementable solutions that will:
1. Reduce return rates through listing improvements
2. Identify product quality issues that need addressing
3. Improve customer expectation management
4. Suggest specific listing changes to prevent returns

Focus on root cause analysis and specific, measurable solutions."""
    
    def _create_return_analysis_prompt(self, product_info: Dict[str, Any], 
                                     return_summary: str, 
                                     quality_info: Dict[str, Any]) -> str:
        """Create detailed prompt for return analysis"""
        
        return_rate = product_info.get('return_rate_30d', 'Unknown')
        sales_30d = product_info.get('sales_30d', 'Unknown')
        
        return f"""Analyze return reasons for: {product_info.get('name', 'Unknown Product')}
ASIN: {product_info.get('asin', 'Unknown')}
Category: {product_info.get('category', 'Medical Device')}
Current Return Rate: {return_rate}%
30-Day Sales Volume: {sales_30d}

{return_summary}

Provide analysis in this EXACT format:

## SUMMARY
- Total return rate impact: [High/Medium/Low risk level]
- Primary return driver: [Single biggest cause]
- Preventable return percentage: [Estimate % preventable through listing changes]
- Product quality concerns: [Yes/No with explanation]

## RETURN CATEGORIZATION
### Size/Fit Issues ({MEDICAL_ANALYSIS_KEYWORDS['sizing_fit_issues']})
- Frequency: [Count and percentage]
- Specific problems: [List with examples]
- Root cause: [Why this happens]

### Quality/Durability Issues  
- Frequency: [Count and percentage]
- Specific problems: [List with examples]
- Severity assessment: [Minor/Major/Critical]

### Expectation Mismatch
- Frequency: [Count and percentage]
- Specific mismatches: [What customers expected vs. reality]
- Listing accuracy issues: [Specific problems with current listing]

### Medical Efficacy Issues
- Frequency: [Count and percentage]
- Specific concerns: [What's not working for customers]
- Usage instruction issues: [Any instruction problems]

## RETURN REDUCTION PLAN
### Immediate Listing Changes (0-2 weeks)
1. **Title**: [Specific change needed]
2. **Bullet Points**: [Specific additions/changes]
3. **Images**: [Specific image improvements needed]
4. **Description**: [Specific information to add]

### Product Improvements (1-3 months)
1. **Design**: [Specific product changes needed]
2. **Quality**: [Specific quality improvements]
3. **Packaging**: [Packaging/instruction improvements]

### Expected Impact
- Estimated return rate reduction: [Specific percentage]
- Timeline for improvement: [Realistic timeframe]
- Success metrics: [How to measure improvement]

Focus on specific, actionable recommendations based on the actual return data provided."""
    
    def _create_optimization_system_prompt(self, product_info: Dict[str, Any]) -> str:
        """Create system prompt for listing optimization"""
        category = product_info.get('category', 'Medical Device')
        return f"""You are an expert Amazon listing optimization specialist for {category} products with a track record of increasing conversion rates and reducing return rates.

Your expertise includes:
- Amazon SEO and keyword optimization
- Medical device marketing and compliance
- Customer psychology and purchase decision factors
- Image optimization and visual storytelling
- A+ Content strategy

Provide specific, implementable recommendations that will improve sales conversion and reduce returns."""
    
    def _create_optimization_prompt(self, product_info: Dict[str, Any], 
                                  review_analysis: Optional[AnalysisResult] = None,
                                  return_analysis: Optional[AnalysisResult] = None) -> str:
        """Create comprehensive optimization prompt"""
        
        # Build context from analysis
        context_sections = []
        
        if review_analysis and review_analysis.success:
            context_sections.append(f"REVIEW INSIGHTS:\n{review_analysis.summary}")
        
        if return_analysis and return_analysis.success:
            context_sections.append(f"RETURN INSIGHTS:\n{return_analysis.summary}")
        
        context = "\n\n".join(context_sections) if context_sections else "No review or return analysis available."
        
        return f"""Optimize the Amazon listing for: {product_info.get('name', 'Unknown Product')}
ASIN: {product_info.get('asin', 'Unknown')}
Category: {product_info.get('category', 'Medical Device')}
Current Return Rate: {product_info.get('return_rate_30d', 'Unknown')}%
Current Star Rating: {product_info.get('star_rating', 'Unknown')}
Description: {product_info.get('description', 'No description available')}

{context}

Provide optimization recommendations in this EXACT format:

## LISTING OPTIMIZATION STRATEGY

### Title Optimization
- Current issues: [Problems with current title if known]
- Recommended title structure: [Specific format]
- Priority keywords: [5-7 most important terms]
- Character optimization: [How to maximize 200-character limit]

### Bullet Point Strategy
- Key benefit #1: [Specific benefit with customer pain point it solves]
- Key benefit #2: [Specific benefit with customer pain point it solves]  
- Key benefit #3: [Specific benefit with customer pain point it solves]
- Key benefit #4: [Specific benefit with customer pain point it solves]
- Key benefit #5: [Specific benefit with customer pain point it solves]

### Image Optimization Plan
1. **Main image**: [Specific requirements]
2. **Feature highlight**: [What to showcase]
3. **Size reference**: [How to show scale]
4. **Lifestyle shot**: [Specific usage scenario]
5. **Comparison chart**: [What to compare]
6. **Infographic**: [Key information to visualize]
7. **Packaging shot**: [What customer receives]

### Description Enhancement
- Opening hook: [Compelling first sentence]
- Problem-solution narrative: [Specific customer problem this solves]
- Key differentiators: [What makes this better than competitors]
- Trust signals: [Certifications, guarantees, etc.]
- Usage instructions: [Clear how-to information]

### Keyword Strategy
- Primary keywords: [Most important search terms]
- Secondary keywords: [Supporting search terms]
- Long-tail keywords: [Specific phrases customers use]
- Backend keywords: [For Amazon search term field]

### Conversion Optimization
- Price positioning: [How to present value]
- Social proof elements: [Reviews, testimonials to highlight]
- Urgency/scarcity tactics: [If appropriate]
- Trust building elements: [Guarantees, certifications]

## IMPLEMENTATION PRIORITY
### Week 1 (Quick Wins)
1. [Specific action item]
2. [Specific action item]
3. [Specific action item]

### Week 2-4 (Major Changes)  
1. [Specific action item]
2. [Specific action item]
3. [Specific action item]

### Month 2+ (Advanced Optimization)
1. [Specific action item]
2. [Specific action item]

## SUCCESS METRICS
- Target conversion rate improvement: [Specific percentage]
- Target return rate reduction: [Specific percentage]  
- Timeline for results: [Realistic expectations]
- Key performance indicators: [What to track]

Base all recommendations on the product category, customer feedback data provided, and Amazon best practices for medical devices."""
    
    def _calculate_confidence_score(self, quality_info: Dict[str, Any], 
                                  data_count: int) -> float:
        """Calculate confidence score based on data quality and quantity"""
        
        base_score = 0.5
        
        # Data quantity factor
        if data_count >= 50:
            quantity_bonus = 0.3
        elif data_count >= 20:
            quantity_bonus = 0.2
        elif data_count >= 10:
            quantity_bonus = 0.1
        else:
            quantity_bonus = 0.0
        
        # Data quality factor
        quality = quality_info.get('quality', 'low')
        if quality == 'high':
            quality_bonus = 0.2
        elif quality == 'medium':
            quality_bonus = 0.1
        else:
            quality_bonus = 0.0
        
        return min(base_score + quantity_bonus + quality_bonus, 1.0)
    
    def _parse_review_analysis(self, content: str) -> Dict[str, Any]:
        """Parse structured review analysis response"""
        
        sections = {
            'summary': {},
            'findings': {},
            'recommendations': []
        }
        
        try:
            # Extract summary section
            summary_match = re.search(r'## SUMMARY\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1)
                sections['summary'] = self._parse_summary_section(summary_text)
            
            # Extract findings
            findings_match = re.search(r'## KEY FINDINGS\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1)
                sections['findings'] = self._parse_findings_section(findings_text)
            
            # Extract action items
            actions_match = re.search(r'## IMMEDIATE ACTION ITEMS\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if actions_match:
                actions_text = actions_match.group(1)
                sections['recommendations'] = self._parse_action_items(actions_text)
        
        except Exception as e:
            logger.error(f"Error parsing review analysis: {str(e)}")
        
        return sections
    
    def _parse_return_analysis(self, content: str) -> Dict[str, Any]:
        """Parse structured return analysis response"""
        
        sections = {
            'summary': {},
            'findings': {},
            'recommendations': []
        }
        
        try:
            # Extract summary
            summary_match = re.search(r'## SUMMARY\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1)
                sections['summary'] = self._parse_summary_section(summary_text)
            
            # Extract categorization
            cat_match = re.search(r'## RETURN CATEGORIZATION\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if cat_match:
                cat_text = cat_match.group(1)
                sections['findings']['categorization'] = cat_text
            
            # Extract reduction plan
            plan_match = re.search(r'## RETURN REDUCTION PLAN\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if plan_match:
                plan_text = plan_match.group(1)
                sections['recommendations'] = self._parse_reduction_plan(plan_text)
        
        except Exception as e:
            logger.error(f"Error parsing return analysis: {str(e)}")
        
        return sections
    
    def _parse_optimization_analysis(self, content: str) -> Dict[str, Any]:
        """Parse structured optimization response"""
        
        sections = {
            'summary': {},
            'findings': {},
            'recommendations': []
        }
        
        try:
            # Extract optimization strategy
            strategy_match = re.search(r'## LISTING OPTIMIZATION STRATEGY\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if strategy_match:
                strategy_text = strategy_match.group(1)
                sections['findings']['strategy'] = strategy_text
            
            # Extract implementation plan
            impl_match = re.search(r'## IMPLEMENTATION PRIORITY\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if impl_match:
                impl_text = impl_match.group(1)
                sections['recommendations'] = self._parse_implementation_plan(impl_text)
            
            # Extract success metrics
            metrics_match = re.search(r'## SUCCESS METRICS\s*\n(.*?)(?=## |$)', content, re.DOTALL)
            if metrics_match:
                metrics_text = metrics_match.group(1)
                sections['summary']['success_metrics'] = metrics_text
        
        except Exception as e:
            logger.error(f"Error parsing optimization analysis: {str(e)}")
        
        return sections
    
    def _parse_summary_section(self, text: str) -> Dict[str, str]:
        """Parse summary bullet points"""
        summary = {}
        lines = text.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip(' -•').lower().replace(' ', '_')
                value = value.strip()
                summary[key] = value
        
        return summary
    
    def _parse_findings_section(self, text: str) -> Dict[str, str]:
        """Parse findings sections"""
        findings = {}
        current_section = None
        
        lines = text.strip().split('\n')
        for line in lines:
            if line.startswith('###'):
                current_section = line.replace('#', '').strip().lower().replace(' ', '_')
                findings[current_section] = ""
            elif current_section and line.strip():
                findings[current_section] += line.strip() + "\n"
        
        return findings
    
    def _parse_action_items(self, text: str) -> List[Dict[str, str]]:
        """Parse action items into structured format"""
        recommendations = []
        
        # Look for numbered items
        items = re.findall(r'\d+\.\s*\*\*(.*?)\*\*:\s*(.*?)(?=\n\s*-|\n\d+\.|\n\n|$)', text, re.DOTALL)
        
        for priority_category, description in items:
            # Parse priority and category
            if '][' in priority_category:
                priority, category = priority_category.split('][', 1)
                priority = priority.strip('[]')
                category = category.strip('[]')
            else:
                priority = "Medium"
                category = priority_category
            
            # Extract action and impact
            parts = description.split('\n')
            action = ""
            impact = ""
            
            for part in parts:
                part = part.strip()
                if part.startswith('- Action:'):
                    action = part.replace('- Action:', '').strip()
                elif part.startswith('- Expected Impact:'):
                    impact = part.replace('- Expected Impact:', '').strip()
            
            recommendations.append({
                'priority': priority,
                'category': category,
                'action': action,
                'expected_impact': impact
            })
        
        return recommendations
    
    def _parse_reduction_plan(self, text: str) -> List[Dict[str, str]]:
        """Parse return reduction plan"""
        recommendations = []
        
        # Extract sections
        sections = re.findall(r'### (.*?)\n(.*?)(?=### |$)', text, re.DOTALL)
        
        for section_name, section_content in sections:
            items = re.findall(r'\d+\.\s*\*\*(.*?)\*\*:\s*(.*?)(?=\n\d+\.|\n\n|$)', section_content, re.DOTALL)
            
            for item_name, item_description in items:
                recommendations.append({
                    'category': section_name.strip(),
                    'priority': 'High' if 'Immediate' in section_name else 'Medium',
                    'action': f"{item_name}: {item_description.strip()}",
                    'expected_impact': 'Return rate reduction'
                })
        
        return recommendations
    
    def _parse_implementation_plan(self, text: str) -> List[Dict[str, str]]:
        """Parse implementation priority plan"""
        recommendations = []
        
        # Extract time-based sections
        sections = re.findall(r'### (.*?)\n(.*?)(?=### |$)', text, re.DOTALL)
        
        for timeframe, section_content in sections:
            items = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)', section_content, re.DOTALL)
            
            for item in items:
                recommendations.append({
                    'category': 'Listing Optimization',
                    'priority': 'High' if 'Week 1' in timeframe else 'Medium',
                    'action': item.strip(),
                    'timeframe': timeframe.strip(),
                    'expected_impact': 'Conversion improvement'
                })
        
        return recommendations

# Main Enhanced AI Analysis Class
class EnhancedAIAnalyzer:
    """Main class coordinating all AI analysis functionality"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.analyzer = ProductSpecificAnalyzer(self.api_client)
        self.preprocessor = DataPreprocessor()
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check API availability and status"""
        if not self.api_client.api_key:
            return {
                'available': False,
                'error': 'API key not configured',
                'suggestions': [
                    'Add OPENAI_API_KEY to environment variables',
                    'Add openai_api_key to Streamlit secrets'
                ]
            }
        
        # Test API with minimal call
        test_response = self.api_client.call_api([
            {"role": "user", "content": "Test"}
        ], max_tokens=10)
        
        return {
            'available': test_response['success'],
            'error': test_response.get('error'),
            'model': test_response.get('model', 'gpt-4o')
        }
    
    def analyze_product_comprehensive(self, product_info: Dict[str, Any],
                                    reviews: List[Dict[str, Any]] = None,
                                    returns: List[Dict[str, Any]] = None) -> Dict[str, AnalysisResult]:
        """Run comprehensive analysis for a product"""
        
        results = {}
        
        # Review analysis
        if reviews:
            logger.info(f"Starting review analysis for {product_info.get('name', 'Unknown')}")
            results['review_analysis'] = self.analyzer.analyze_reviews_comprehensive(
                product_info, reviews
            )
        
        # Return analysis
        if returns:
            logger.info(f"Starting return analysis for {product_info.get('name', 'Unknown')}")
            results['return_analysis'] = self.analyzer.analyze_returns_comprehensive(
                product_info, returns
            )
        
        # Listing optimization
        logger.info(f"Starting listing optimization for {product_info.get('name', 'Unknown')}")
        results['listing_optimization'] = self.analyzer.generate_listing_optimization(
            product_info, 
            results.get('review_analysis'),
            results.get('return_analysis')
        )
        
        return results
    
    def export_analysis_results(self, results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Export analysis results in structured format"""
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'analyses': {}
        }
        
        for analysis_type, result in results.items():
            if result.success:
                export_data['analyses'][analysis_type] = result.to_dict()
        
        return export_data

# Export main classes
__all__ = [
    'EnhancedAIAnalyzer', 
    'AnalysisResult', 
    'RecommendationItem',
    'APIClient',
    'ProductSpecificAnalyzer'
]
