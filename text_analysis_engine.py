"""
Enhanced Text Analysis Engine for Medical Device Customer Feedback Analysis

**OPTIMIZED FOR HELIUM 10 REVIEW PROCESSING**

This is the core analytical engine that processes customer reviews from Helium 10 exports
and categorizes them into medical device quality categories with enhanced accuracy and
actionable insights for quality management.

Key Enhancements:
✓ Helium 10 review structure optimization (Title + Body analysis)
✓ Enhanced medical device quality categorization
✓ Improved sentiment analysis with rating correlation
✓ Advanced temporal trending for review patterns
✓ Precision CAPA generation for quality issues
✓ ISO 13485 compliance-aware risk assessment
✓ Star rating stratification analysis

Author: Assistant
Version: 3.1 - Helium 10 Review Optimized
Compliance: ISO 13485 Quality Management System
"""

import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import json
import statistics
from enum import Enum

# Configure logging for quality management traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Medical Device Quality Categories (Optimized for Helium 10 Reviews)
class QualityCategory(Enum):
    """Enumeration of medical device quality categories"""
    SAFETY_CONCERNS = "safety_concerns"
    EFFICACY_PERFORMANCE = "efficacy_performance"
    COMFORT_USABILITY = "comfort_usability"
    DURABILITY_QUALITY = "durability_quality"
    SIZING_FIT = "sizing_fit"
    ASSEMBLY_INSTRUCTIONS = "assembly_instructions"
    SHIPPING_PACKAGING = "shipping_packaging"
    BIOCOMPATIBILITY = "biocompatibility"
    CUSTOMER_SERVICE = "customer_service"
    VALUE_PRICING = "value_pricing"

# Enhanced Medical Device Quality Framework (Tuned for Amazon Reviews)
MEDICAL_DEVICE_QUALITY_FRAMEWORK = {
    QualityCategory.SAFETY_CONCERNS.value: {
        'name': 'Safety & Risk Management',
        'description': 'Customer feedback indicating potential safety hazards or injury risks',
        'keywords': [
            # Direct safety terms
            'unsafe', 'dangerous', 'hazardous', 'injury', 'hurt', 'injured', 'harm', 'accident',
            'broke while using', 'collapsed', 'gave way', 'failed',
            # Structural safety
            'broke', 'broken', 'snapped', 'cracked', 'split', 'fell apart', 'unstable', 'wobbly', 
            'tip over', 'tipped', 'toppled', 'balance issues',
            # Sharp/cutting hazards
            'sharp', 'cuts', 'cut me', 'cutting', 'pinch', 'pinched', 'trapped', 'caught',
            'sharp edges', 'rough edges',
            # Fall/mobility safety
            'slipped', 'fall', 'fell', 'falling', 'slide', 'slip', 'lose balance', 'lost balance',
            'almost fell', 'nearly fell',
            # Medical safety
            'made condition worse', 'increased pain', 'caused injury', 'emergency room',
            'doctor visit', 'medical attention', 'hospital',
            # Product failure safety
            'wheel came off', 'leg broke', 'seat collapsed', 'handle broke'
        ],
        'severity': 'critical',
        'iso_reference': 'ISO 13485 Section 7.3 - Risk Management',
        'regulatory_impact': 'high',
        'immediate_action_required': True,
        'weight': 10.0  # Highest weight for safety
    },
    
    QualityCategory.EFFICACY_PERFORMANCE.value: {
        'name': 'Efficacy & Performance',
        'description': 'Feedback about product effectiveness and functional performance',
        'keywords': [
            # Ineffectiveness
            'doesnt work', 'not working', 'ineffective', 'useless', 'waste of money',
            'no relief', 'no help', 'no improvement', 'no difference', 'no benefit',
            'doesnt help', 'not helpful', 'made no difference',
            # Performance issues
            'poor performance', 'weak', 'insufficient', 'inadequate', 'subpar',
            'disappointing', 'expected more', 'not as advertised', 'false claims',
            'overhyped', 'marketing lies',
            # Functional problems
            'malfunctioning', 'defective', 'faulty', 'not functioning', 'stopped working',
            'inconsistent', 'unreliable', 'intermittent', 'sporadic performance',
            # Medical efficacy
            'no pain relief', 'still in pain', 'condition not improved', 'symptoms worse',
            'not therapeutic', 'no healing', 'no recovery'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 13485 Section 7.3 - Performance Requirements',
        'regulatory_impact': 'medium',
        'immediate_action_required': False,
        'weight': 8.0
    },
    
    QualityCategory.COMFORT_USABILITY.value: {
        'name': 'Comfort & Usability',
        'description': 'User experience, comfort, and ease of use issues',
        'keywords': [
            # Comfort issues
            'uncomfortable', 'painful', 'hurts', 'sore', 'aches', 'pressure',
            'rough', 'hard', 'stiff', 'rigid', 'tight', 'scratchy', 'itchy',
            'causes pain', 'makes pain worse', 'very uncomfortable',
            # Usability problems
            'difficult to use', 'hard to use', 'confusing', 'complicated', 'awkward',
            'user unfriendly', 'not intuitive', 'hard to operate', 'cumbersome',
            'difficult to adjust', 'hard to position', 'cant figure out',
            # Ergonomic issues
            'poor ergonomics', 'bad design', 'poorly designed', 'not ergonomic',
            'strain', 'fatigue', 'tiring', 'exhausting', 'wears me out',
            # Mobility issues
            'heavy', 'too heavy', 'bulky', 'hard to move', 'difficult to maneuver',
            'hard to push', 'hard to lift', 'cant carry'
        ],
        'severity': 'medium',
        'iso_reference': 'ISO 13485 Section 7.3 - User Requirements',
        'regulatory_impact': 'low',
        'immediate_action_required': False,
        'weight': 6.0
    },
    
    QualityCategory.DURABILITY_QUALITY.value: {
        'name': 'Durability & Build Quality',
        'description': 'Product longevity, material quality, and construction issues',
        'keywords': [
            # Material failure
            'cheap', 'cheaply made', 'poor quality', 'low quality', 'flimsy', 'fragile',
            'weak material', 'thin', 'brittle', 'cracked', 'split', 'torn', 'ripped',
            'cheap plastic', 'poor construction', 'shoddy workmanship',
            # Structural failure
            'fell apart', 'came apart', 'broke', 'broken', 'bent', 'snapped', 'warped',
            'loose screws', 'loose parts', 'wobbly', 'unstable', 'deteriorated',
            'joints failed', 'welds broke', 'stitching came undone',
            # Wear and tear
            'worn out', 'wearing out', 'fading', 'discolored', 'peeling', 'chipping',
            'rust', 'corrosion', 'degraded', 'short lifespan', 'doesnt last',
            'only lasted', 'broke after', 'failed within',
            # Quality expectations
            'expected better quality', 'quality issues', 'build quality poor',
            'not worth the price', 'overpriced for quality'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 13485 Section 7.5 - Production Controls',
        'regulatory_impact': 'medium',
        'immediate_action_required': False,
        'weight': 7.0
    },
    
    QualityCategory.SIZING_FIT.value: {
        'name': 'Sizing & Fit Issues',
        'description': 'Size accuracy, fit problems, and measurement discrepancies',
        'keywords': [
            # Size problems
            'too small', 'too big', 'too large', 'wrong size', 'incorrect size',
            'runs small', 'runs large', 'runs big', 'sizing issue', 'size problem',
            'not the right size', 'size chart wrong', 'measurements off',
            # Fit issues
            'doesnt fit', 'poor fit', 'bad fit', 'tight', 'loose', 'baggy',
            'doesnt stay in place', 'slips', 'slides', 'moves around',
            'falls off', 'wont stay put', 'keeps sliding',
            # Measurement discrepancies
            'measurements wrong', 'measurements off', 'sizing chart wrong', 'inaccurate measurements',
            'different than described', 'not as pictured', 'smaller than expected', 'larger than expected',
            'dimensions incorrect', 'specs wrong',
            # Medical device fit
            'doesnt fit properly', 'uncomfortable fit', 'cant adjust', 'wont adjust',
            'adjustment range insufficient', 'cant get right fit'
        ],
        'severity': 'medium',
        'iso_reference': 'ISO 13485 Section 7.3 - Design Specifications',
        'regulatory_impact': 'low',
        'immediate_action_required': False,
        'weight': 5.0
    },
    
    QualityCategory.ASSEMBLY_INSTRUCTIONS.value: {
        'name': 'Assembly & Instructions',
        'description': 'Setup difficulties, unclear instructions, and missing components',
        'keywords': [
            # Assembly problems
            'difficult assembly', 'hard to assemble', 'assembly problems', 'setup issues',
            'complicated setup', 'confusing assembly', 'poor assembly', 'assembly nightmare',
            'took hours to assemble', 'assembly frustrating',
            # Instruction issues
            'unclear instructions', 'confusing instructions', 'poor instructions', 'bad directions',
            'missing instructions', 'incomplete instructions', 'hard to follow', 'poorly written',
            'instructions wrong', 'directions unclear', 'manual confusing',
            # Missing components
            'missing parts', 'missing pieces', 'missing hardware', 'missing screws',
            'incomplete kit', 'parts missing', 'wrong parts', 'damaged parts',
            'hardware missing', 'tools missing',
            # Documentation issues
            'no manual', 'missing manual', 'poor documentation', 'inadequate documentation',
            'pictures unclear', 'diagrams confusing', 'steps unclear'
        ],
        'severity': 'medium',
        'iso_reference': 'ISO 13485 Section 4.2 - Documentation Requirements',
        'regulatory_impact': 'low',
        'immediate_action_required': False,
        'weight': 4.0
    },
    
    QualityCategory.SHIPPING_PACKAGING.value: {
        'name': 'Shipping & Packaging',
        'description': 'Delivery, packaging, and shipping-related issues',
        'keywords': [
            # Shipping damage
            'arrived damaged', 'damaged in shipping', 'shipping damage', 'broken in transit',
            'crushed', 'dented', 'scratched during shipping', 'bent in shipping',
            'box damaged', 'packaging damaged',
            # Packaging issues
            'poor packaging', 'inadequate packaging', 'bad packaging', 'insufficient padding',
            'not protected', 'poorly packed', 'loose in box', 'rattling around',
            'no bubble wrap', 'minimal protection',
            # Delivery problems
            'late delivery', 'delayed shipping', 'wrong item shipped', 'missing items',
            'partial shipment', 'delivery issues', 'shipping problems',
            'took forever to arrive', 'slow shipping'
        ],
        'severity': 'low',
        'iso_reference': 'ISO 13485 Section 7.5 - Packaging Requirements',
        'regulatory_impact': 'low',
        'immediate_action_required': False,
        'weight': 2.0
    },
    
    QualityCategory.BIOCOMPATIBILITY.value: {
        'name': 'Biocompatibility & Materials',
        'description': 'Skin reactions, allergies, and material compatibility issues',
        'keywords': [
            # Allergic reactions
            'allergic reaction', 'allergy', 'rash', 'skin irritation', 'red skin',
            'itchy', 'itching', 'burning sensation', 'stinging', 'tingling',
            'broke out in rash', 'skin reaction',
            # Material issues
            'latex allergy', 'material sensitivity', 'chemical smell', 'odor', 'toxic smell',
            'skin contact issues', 'dermatitis', 'eczema', 'hives', 'swelling',
            'chemical reaction', 'strong odor', 'smells bad',
            # Biocompatibility concerns
            'not biocompatible', 'material reaction', 'sensitive skin', 'skin problems',
            'contact dermatitis', 'material allergy', 'chemical reaction',
            'cant tolerate material', 'skin cant handle'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 10993 - Biological Evaluation of Medical Devices',
        'regulatory_impact': 'high',
        'immediate_action_required': True,
        'weight': 9.0
    },
    
    QualityCategory.CUSTOMER_SERVICE.value: {
        'name': 'Customer Service & Support',
        'description': 'Customer service experiences and support issues',
        'keywords': [
            # Poor service
            'poor customer service', 'bad customer service', 'terrible service',
            'unhelpful', 'rude', 'no response', 'ignored', 'no help',
            # Support issues
            'no support', 'poor support', 'cant reach support', 'no callback',
            'support useless', 'wont help', 'refused to help',
            # Return/warranty issues
            'wont accept return', 'return denied', 'warranty void', 'no warranty',
            'return process difficult', 'hassle to return'
        ],
        'severity': 'low',
        'iso_reference': 'ISO 13485 Section 8.2.2 - Customer Feedback',
        'regulatory_impact': 'low',
        'immediate_action_required': False,
        'weight': 3.0
    },
    
    QualityCategory.VALUE_PRICING.value: {
        'name': 'Value & Pricing Concerns',
        'description': 'Price-value relationship and cost-effectiveness issues',
        'keywords': [
            # Price concerns
            'overpriced', 'too expensive', 'not worth the money', 'waste of money',
            'poor value', 'bad value', 'expensive for what it is',
            'costs too much', 'ridiculously expensive',
            # Value issues
            'not worth it', 'expected more for the price', 'should cost less',
            'better options for less money', 'cheaper alternatives available',
            'price too high', 'value not there'
        ],
        'severity': 'low',
        'iso_reference': 'ISO 13485 Section 8.2.2 - Customer Feedback',
        'regulatory_impact': 'low',
        'immediate_action_required': False,
        'weight': 2.0
    }
}

# Enhanced positive feedback indicators (Amazon review specific)
POSITIVE_FEEDBACK_INDICATORS = {
    'quality_praise': [
        'excellent quality', 'high quality', 'great quality', 'amazing quality', 'superior quality',
        'well made', 'well built', 'solid construction', 'durable', 'sturdy', 'robust',
        'quality product', 'good quality', 'quality construction'
    ],
    'effectiveness': [
        'works great', 'very effective', 'highly effective', 'exactly what needed',
        'perfect solution', 'life changing', 'life saver', 'game changer', 'incredible results',
        'works perfectly', 'does exactly what', 'solved my problem'
    ],
    'comfort': [
        'very comfortable', 'extremely comfortable', 'so comfortable', 'perfect fit',
        'comfortable to use', 'easy to use', 'user friendly', 'ergonomic',
        'feels great', 'comfortable fit'
    ],
    'satisfaction': [
        'love it', 'love this', 'highly recommend', 'would recommend', 'best purchase',
        'excellent product', 'amazing product', 'perfect', 'fantastic', 'outstanding',
        'exceeded expectations', 'better than expected', 'worth every penny'
    ],
    'ease_of_use': [
        'easy to use', 'simple to use', 'user friendly', 'intuitive', 'straightforward',
        'easy setup', 'simple assembly', 'clear instructions'
    ]
}

# Risk severity levels
class RiskLevel(Enum):
    """Risk severity levels for medical device feedback"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMAL = "Minimal"

@dataclass
class ReviewItem:
    """Enhanced representation of a Helium 10 review item"""
    text: str
    review_title: str
    review_body: str
    date: str
    rating: Optional[int] = None
    author: str = 'Anonymous'
    verified: str = 'Unknown'
    helpful_votes: str = '0'
    has_images: bool = False
    has_videos: bool = False
    variation: str = ''
    style: str = ''
    source: str = 'helium10_export'
    asin: str = ''
    product_name: str = ''
    
    # Analysis results
    detected_categories: List[str] = None
    sentiment_score: float = 0.0
    severity_level: str = 'medium'
    confidence_score: float = 0.0
    quality_impact_score: float = 0.0
    
    def __post_init__(self):
        if self.detected_categories is None:
            self.detected_categories = []

@dataclass 
class CategoryAnalysis:
    """Enhanced analysis results for a specific quality category"""
    category_id: str
    name: str
    count: int
    percentage: float
    severity: str
    iso_reference: str
    weight: float
    
    # Enhanced analysis
    matched_reviews: List[ReviewItem]
    common_keywords: List[Tuple[str, int]]  # (keyword, frequency)
    severity_breakdown: Dict[str, int]  # high/medium/low counts
    rating_correlation: Dict[str, Any]  # How category relates to star ratings
    temporal_pattern: Dict[str, Any]
    
    # Risk assessment
    risk_score: float
    requires_capa: bool
    immediate_action_required: bool
    
    # Star rating analysis
    avg_rating_for_category: float
    rating_distribution: Dict[int, int]

@dataclass
class QualityAssessment:
    """Enhanced overall quality assessment from review analysis"""
    total_reviews: int
    quality_score: float  # 0-100
    quality_level: str  # 'Excellent', 'Good', 'Fair', 'Poor', 'Critical'
    
    # Enhanced sentiment breakdown
    positive_count: int
    negative_count: int
    neutral_count: int
    sentiment_ratio: float
    
    # Star rating analysis
    rating_distribution: Dict[int, int]
    average_rating: float
    rating_trend: str  # 'improving', 'declining', 'stable'
    
    # Risk indicators
    high_risk_categories: List[str]
    safety_issues_count: int
    efficacy_issues_count: int
    
    # Customer insights
    verified_buyer_percentage: float
    review_helpfulness_score: float
    reviews_with_media_count: int
    
    # Improvement indicators
    improvement_trend: str
    key_strengths: List[str]
    primary_concerns: List[str]

@dataclass
class CAPARecommendation:
    """Enhanced CAPA recommendation with Amazon review context"""
    capa_id: str
    priority: str  # 'Critical', 'High', 'Medium', 'Low'
    category: str
    
    # Problem identification (enhanced)
    issue_description: str
    root_cause_analysis: str
    affected_customer_count: int
    customer_impact_assessment: str
    
    # Review-specific insights
    representative_reviews: List[str]  # Sample reviews demonstrating the issue
    rating_impact: str  # How this issue affects star ratings
    review_volume_trend: str  # Is this issue increasing/decreasing
    
    # Actions (enhanced)
    corrective_action: str
    preventive_action: str
    timeline: str
    responsibility: str
    
    # Success criteria (enhanced)
    success_metrics: List[str]
    verification_method: str
    target_improvement: str  # Specific improvement target
    
    # Compliance
    iso_reference: str
    regulatory_impact: str
    documentation_required: bool

class AdvancedReviewProcessor:
    """Enhanced text processing optimized for Helium 10 reviews"""
    
    def __init__(self):
        self.quality_framework = MEDICAL_DEVICE_QUALITY_FRAMEWORK
        self.positive_indicators = POSITIVE_FEEDBACK_INDICATORS
        
    def preprocess_review_text(self, title: str, body: str) -> str:
        """Enhanced preprocessing for Helium 10 review structure"""
        if not title and not body:
            return ""
        
        # Combine title and body intelligently
        text_parts = []
        
        if title and title.strip() and title.lower() not in ['nan', 'none', '']:
            title_clean = self._clean_text(title)
            if title_clean:
                text_parts.append(f"TITLE: {title_clean}")
        
        if body and body.strip() and body.lower() not in ['nan', 'none', '']:
            body_clean = self._clean_text(body)
            if body_clean:
                text_parts.append(f"REVIEW: {body_clean}")
        
        combined_text = " | ".join(text_parts)
        return combined_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize individual text components"""
        if not text:
            return ""
        
        # Convert to lowercase and strip
        text = str(text).lower().strip()
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize common contractions
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
            "it's": "it is", "that's": "that is", "what's": "what is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords_with_context(self, text: str, category_keywords: List[str]) -> List[Tuple[str, int, List[str]]]:
        """Extract keywords with surrounding context for better analysis"""
        text = self._clean_text(text)
        matches = []
        
        for keyword in category_keywords:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            keyword_matches = list(re.finditer(pattern, text))
            
            if keyword_matches:
                contexts = []
                for match in keyword_matches:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    contexts.append(context)
                
                matches.append((keyword, len(keyword_matches), contexts))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def calculate_enhanced_sentiment_score(self, title: str, body: str, rating: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        """Enhanced sentiment calculation with rating correlation"""
        combined_text = self.preprocess_review_text(title, body)
        
        # Base score from rating if available (strong indicator)
        rating_score = 0.5  # Neutral
        if rating is not None:
            rating_score = (rating - 1) / 4  # Convert 1-5 to 0-1
        
        # Text-based sentiment analysis
        positive_count = 0
        negative_count = 0
        
        # Count positive indicators
        for category_indicators in self.positive_indicators.values():
            for indicator in category_indicators:
                if indicator in combined_text:
                    positive_count += 1
        
        # Count negative indicators from quality categories
        for category_data in self.quality_framework.values():
            for keyword in category_data['keywords']:
                if keyword in combined_text:
                    negative_count += 1
        
        # Calculate text sentiment
        text_sentiment = 0.5  # Neutral
        if positive_count > 0 or negative_count > 0:
            total_indicators = positive_count + negative_count
            text_sentiment = positive_count / total_indicators if total_indicators > 0 else 0.5
        
        # Combine rating and text sentiment with weights
        if rating is not None:
            # Rating is strong signal, give it more weight
            final_score = (rating_score * 0.7) + (text_sentiment * 0.3)
        else:
            # Pure text sentiment
            final_score = text_sentiment
        
        # Sentiment analysis details
        sentiment_details = {
            'rating_score': rating_score,
            'text_sentiment': text_sentiment,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'rating_weight': 0.7 if rating is not None else 0.0,
            'text_weight': 0.3 if rating is not None else 1.0
        }
        
        return max(0.0, min(1.0, final_score)), sentiment_details
    
    def detect_quality_impact_indicators(self, text: str) -> Dict[str, Any]:
        """Detect indicators that suggest quality management impact"""
        text = self._clean_text(text)
        
        impact_indicators = {
            'severity_indicators': [],
            'urgency_indicators': [],
            'repeat_purchase_indicators': [],
            'recommendation_indicators': []
        }
        
        # Severity indicators
        severity_patterns = [
            r'\b(very|extremely|incredibly|absolutely|completely|totally)\s+(bad|poor|terrible|awful|horrible)\b',
            r'\b(worst|terrible|awful|horrible|disgusting|useless)\b',
            r'\b(never|will never|would never)\s+(buy|purchase|recommend)\b'
        ]
        
        for pattern in severity_patterns:
            if re.search(pattern, text):
                matches = re.findall(pattern, text)
                impact_indicators['severity_indicators'].extend(matches)
        
        # Urgency indicators
        urgency_patterns = [
            r'\b(urgent|immediate|emergency|asap|right away|quickly)\b',
            r'\b(dangerous|unsafe|hazardous|risk|injury|hurt)\b',
            r'\b(broken|defective|malfunctioning|not working)\b'
        ]
        
        for pattern in urgency_patterns:
            matches = re.findall(pattern, text)
            impact_indicators['urgency_indicators'].extend(matches)
        
        # Repeat purchase indicators
        repeat_patterns = [
            r'\b(will buy again|would buy again|buying another|ordering another)\b',
            r'\b(will not buy|would not buy|never again|last time)\b'
        ]
        
        for pattern in repeat_patterns:
            matches = re.findall(pattern, text)
            impact_indicators['repeat_purchase_indicators'].extend(matches)
        
        # Recommendation indicators
        recommendation_patterns = [
            r'\b(highly recommend|strongly recommend|definitely recommend)\b',
            r'\b(would not recommend|do not recommend|cannot recommend)\b'
        ]
        
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, text)
            impact_indicators['recommendation_indicators'].extend(matches)
        
        return impact_indicators

class EnhancedCategoryAnalyzer:
    """Enhanced analyzer for medical device quality categories with Helium 10 optimization"""
    
    def __init__(self):
        self.text_processor = AdvancedReviewProcessor()
        self.quality_framework = MEDICAL_DEVICE_QUALITY_FRAMEWORK
    
    def analyze_reviews_by_categories(self, review_items: List[ReviewItem]) -> Dict[str, CategoryAnalysis]:
        """Enhanced category analysis optimized for Helium 10 reviews"""
        category_results = {}
        total_reviews = len(review_items)
        
        logger.info(f"Analyzing {total_reviews} reviews across {len(self.quality_framework)} quality categories")
        
        for category_id, category_info in self.quality_framework.items():
            # Find matching reviews
            matched_reviews = []
            all_keywords = []
            
            for review in review_items:
                keywords_found = self.text_processor.extract_keywords_with_context(
                    review.text, category_info['keywords']
                )
                
                if keywords_found:
                    # Add category to review's detected categories
                    if category_id not in review.detected_categories:
                        review.detected_categories.append(category_id)
                    
                    matched_reviews.append(review)
                    # Collect keywords for frequency analysis
                    for keyword, count, contexts in keywords_found:
                        all_keywords.extend([keyword] * count)
            
            # Calculate enhanced metrics
            count = len(matched_reviews)
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            
            # Enhanced severity breakdown with star rating correlation
            severity_breakdown = self._analyze_enhanced_severity(matched_reviews)
            
            # Star rating correlation for this category
            rating_correlation = self._analyze_rating_correlation(matched_reviews)
            
            # Calculate enhanced risk score
            risk_score = self._calculate_enhanced_risk_score(
                count, category_info, severity_breakdown, rating_correlation, total_reviews
            )
            
            # Temporal pattern analysis
            temporal_pattern = self._analyze_temporal_pattern(matched_reviews)
            
            # Common keywords with frequency
            common_keywords = Counter(all_keywords).most_common(10)
            
            # CAPA requirements (enhanced)
            requires_capa = self._determine_enhanced_capa_requirement(
                count, category_info, risk_score, rating_correlation
            )
            
            # Average rating for this category
            category_ratings = [r.rating for r in matched_reviews if r.rating is not None]
            avg_rating = sum(category_ratings) / len(category_ratings) if category_ratings else 0
            
            # Rating distribution for this category
            rating_dist = Counter(category_ratings) if category_ratings else {}
            
            category_results[category_id] = CategoryAnalysis(
                category_id=category_id,
                name=category_info['name'],
                count=count,
                percentage=round(percentage, 1),
                severity=category_info['severity'],
                iso_reference=category_info['iso_reference'],
                weight=category_info['weight'],
                matched_reviews=matched_reviews,
                common_keywords=common_keywords,
                severity_breakdown=severity_breakdown,
                rating_correlation=rating_correlation,
                temporal_pattern=temporal_pattern,
                risk_score=risk_score,
                requires_capa=requires_capa,
                immediate_action_required=category_info.get('immediate_action_required', False) and count > 0,
                avg_rating_for_category=round(avg_rating, 2),
                rating_distribution=dict(rating_dist)
            )
        
        return category_results
    
    def _analyze_enhanced_severity(self, reviews: List[ReviewItem]) -> Dict[str, int]:
        """Enhanced severity analysis with multiple factors"""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for review in reviews:
            # Calculate severity based on multiple factors
            severity_score = 0
            
            # Rating factor (lower rating = higher severity)
            if review.rating is not None:
                if review.rating <= 2:
                    severity_score += 3
                elif review.rating == 3:
                    severity_score += 2
                else:
                    severity_score += 1
            else:
                severity_score += 2  # Medium if no rating
            
            # Quality impact indicators
            impact_indicators = self.text_processor.detect_quality_impact_indicators(review.text)
            if impact_indicators['severity_indicators']:
                severity_score += 2
            if impact_indicators['urgency_indicators']:
                severity_score += 3
            
            # Multiple category involvement (indicates systemic issues)
            if len(review.detected_categories) >= 3:
                severity_score += 2
            elif len(review.detected_categories) >= 2:
                severity_score += 1
            
            # Determine final severity
            if severity_score >= 6:
                severity = 'high'
            elif severity_score >= 3:
                severity = 'medium'
            else:
                severity = 'low'
            
            severity_counts[severity] += 1
            review.severity_level = severity
        
        return severity_counts
    
    def _analyze_rating_correlation(self, reviews: List[ReviewItem]) -> Dict[str, Any]:
        """Analyze how category issues correlate with star ratings"""
        if not reviews:
            return {'correlation': 'no_data'}
        
        # Get ratings for reviews in this category
        ratings_with_category = [r.rating for r in reviews if r.rating is not None]
        if not ratings_with_category:
            return {'correlation': 'no_ratings'}
        
        avg_category_rating = sum(ratings_with_category) / len(ratings_with_category)
        rating_distribution = Counter(ratings_with_category)
        
        # Analyze impact on ratings
        low_rating_count = sum(1 for r in ratings_with_category if r <= 2)
        high_rating_count = sum(1 for r in ratings_with_category if r >= 4)
        
        # Determine correlation strength
        low_rating_percentage = (low_rating_count / len(ratings_with_category)) * 100
        
        if low_rating_percentage >= 70:
            correlation_strength = 'strong_negative'
        elif low_rating_percentage >= 40:
            correlation_strength = 'moderate_negative'
        elif low_rating_percentage <= 20 and high_rating_count > low_rating_count:
            correlation_strength = 'positive'
        else:
            correlation_strength = 'neutral'
        
        return {
            'correlation': correlation_strength,
            'avg_rating': round(avg_category_rating, 2),
            'rating_distribution': dict(rating_distribution),
            'low_rating_percentage': round(low_rating_percentage, 1),
            'review_count': len(ratings_with_category)
        }
    
    def _calculate_enhanced_risk_score(self, count: int, category_info: Dict[str, Any], 
                                     severity_breakdown: Dict[str, int], 
                                     rating_correlation: Dict[str, Any], 
                                     total_reviews: int) -> float:
        """Enhanced risk score calculation"""
        if count == 0:
            return 0.0
        
        # Base score from category weight and severity
        base_score = category_info['weight']
        
        # Frequency factor (higher frequency = higher risk)
        frequency_factor = min((count / max(total_reviews, 1)) * 20, 10)
        
        # Severity distribution factor
        severity_factor = (
            severity_breakdown.get('high', 0) * 3 +
            severity_breakdown.get('medium', 0) * 2 +
            severity_breakdown.get('low', 0) * 1
        ) / max(count, 1)
        
        # Rating impact factor
        rating_factor = 0
        if rating_correlation.get('correlation') == 'strong_negative':
            rating_factor = 3
        elif rating_correlation.get('correlation') == 'moderate_negative':
            rating_factor = 2
        elif rating_correlation.get('correlation') == 'positive':
            rating_factor = -1  # Reduces risk if category has positive correlation
        
        # Immediate action multiplier
        action_multiplier = 1.5 if category_info.get('immediate_action_required', False) else 1.0
        
        risk_score = (base_score + frequency_factor + severity_factor + rating_factor) * action_multiplier
        return min(risk_score, 30.0)  # Cap at 30
    
    def _analyze_temporal_pattern(self, reviews: List[ReviewItem]) -> Dict[str, Any]:
        """Enhanced temporal pattern analysis"""
        if not reviews:
            return {'pattern': 'no_data', 'trend': 'stable'}
        
        # Group reviews by month
        monthly_counts = defaultdict(int)
        for review in reviews:
            try:
                review_date = datetime.strptime(review.date, '%Y-%m-%d').date()
                month_key = review_date.strftime('%Y-%m')
                monthly_counts[month_key] += 1
            except (ValueError, TypeError):
                continue
        
        if len(monthly_counts) < 2:
            return {'pattern': 'insufficient_data', 'trend': 'stable'}
        
        # Analyze trend
        months = sorted(monthly_counts.keys())
        counts = [monthly_counts[month] for month in months]
        
        # Simple trend analysis (comparing recent vs earlier periods)
        if len(counts) >= 4:
            recent_avg = statistics.mean(counts[-2:])
            earlier_avg = statistics.mean(counts[:-2])
            
            if recent_avg > earlier_avg * 1.5:
                trend = 'increasing'
            elif recent_avg < earlier_avg * 0.5:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'pattern': 'analyzed',
            'trend': trend,
            'monthly_data': dict(monthly_counts),
            'peak_month': max(monthly_counts.keys(), key=lambda k: monthly_counts[k]) if monthly_counts else None,
            'total_months': len(monthly_counts)
        }
    
    def _determine_enhanced_capa_requirement(self, count: int, category_info: Dict[str, Any], 
                                           risk_score: float, rating_correlation: Dict[str, Any]) -> bool:
        """Enhanced CAPA requirement determination"""
        # Immediate action categories
        if category_info.get('immediate_action_required', False) and count > 0:
            return True
        
        # High risk score threshold
        if risk_score > 15:
            return True
        
        # Severity-based thresholds (adjusted)
        if category_info['severity'] == 'critical' and count > 0:
            return True
        elif category_info['severity'] == 'high' and count > 2:
            return True
        elif category_info['severity'] == 'medium' and count > 5:
            return True
        
        # Rating correlation impact
        if rating_correlation.get('correlation') == 'strong_negative' and count > 3:
            return True
        
        return False

class AIEnhancedAnalyzer:
    """AI-powered analyzer using OpenAI GPT-4o for sophisticated medical device review analysis"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                logger.info("Found API key in Streamlit secrets")
                return st.secrets['openai_api_key']
        except (ImportError, AttributeError, KeyError):
            pass
        
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Found API key in environment variables")
            return api_key
        
        logger.warning("No API key found - AI analysis will be disabled")
        return None
    
    def is_available(self) -> bool:
        """Check if AI analysis is available"""
        return self.api_key is not None
    
    def analyze_reviews_with_ai(self, review_batch: List[ReviewItem], 
                               product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to analyze a batch of reviews for medical device quality insights"""
        if not self.is_available():
            return {'success': False, 'error': 'AI API not available'}
        
        try:
            # Prepare review text for AI analysis
            review_texts = []
            for i, review in enumerate(review_batch[:10], 1):  # Limit to 10 reviews per batch
                rating_text = f" (Rating: {review.rating}/5)" if review.rating else ""
                review_text = f"Review {i}{rating_text}: {review.text[:500]}"  # Limit length
                review_texts.append(review_text)
            
            # Create AI prompt for medical device analysis
            prompt = self._create_medical_device_analysis_prompt(
                product_info, review_texts
            )
            
            # Make API call
            response = self._call_openai_api(prompt)
            
            if response['success']:
                # Parse AI response
                ai_insights = self._parse_ai_response(response['content'])
                return {
                    'success': True,
                    'ai_insights': ai_insights,
                    'token_usage': response.get('usage', {}),
                    'reviews_analyzed': len(review_batch)
                }
            else:
                return {'success': False, 'error': response['error']}
                
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_medical_device_analysis_prompt(self, product_info: Dict[str, Any], 
                                             review_texts: List[str]) -> str:
        """Create specialized prompt for medical device quality analysis"""
        
        product_name = product_info.get('name', 'Unknown Product')
        product_category = product_info.get('category', 'Medical Device')
        
        return f"""You are an expert medical device quality manager analyzing customer reviews for ISO 13485 compliance and quality improvement. 

PRODUCT: {product_name}
CATEGORY: {product_category}
ASIN: {product_info.get('asin', 'Unknown')}

CUSTOMER REVIEWS TO ANALYZE:
{chr(10).join(review_texts)}

Please provide a comprehensive quality analysis in this EXACT format:

## QUALITY CATEGORIZATION
For each review, identify which medical device quality categories apply:

SAFETY_CONCERNS: [List review numbers that indicate safety issues - injuries, product failures, hazards]
EFFICACY_PERFORMANCE: [List review numbers about product not working, ineffective treatment, poor performance]
COMFORT_USABILITY: [List review numbers about discomfort, difficulty using, poor ergonomics]
DURABILITY_QUALITY: [List review numbers about breaking, poor construction, material failures]
SIZING_FIT: [List review numbers about wrong size, poor fit, measurement issues]
ASSEMBLY_INSTRUCTIONS: [List review numbers about difficult setup, unclear instructions, missing parts]
BIOCOMPATIBILITY: [List review numbers about skin reactions, allergies, material sensitivity]
SHIPPING_PACKAGING: [List review numbers about shipping damage, poor packaging]

## RISK ASSESSMENT
CRITICAL_ISSUES: [Any safety hazards, injuries, or regulatory concerns requiring immediate action]
HIGH_PRIORITY: [Quality issues affecting multiple customers or product effectiveness]
MEDIUM_PRIORITY: [Comfort, usability, or moderate quality concerns]
LOW_PRIORITY: [Minor issues like packaging or shipping]

## SENTIMENT ANALYSIS
OVERALL_SENTIMENT: [Positive/Negative/Mixed with confidence percentage]
RATING_CORRELATION: [How text sentiment aligns with star ratings - note any discrepancies]
EMOTIONAL_INDICATORS: [Strong emotional language, urgency, frustration, satisfaction]

## ROOT CAUSE ANALYSIS
DESIGN_ISSUES: [Problems with product design or engineering]
MANUFACTURING_ISSUES: [Quality control, materials, or production problems]
DOCUMENTATION_ISSUES: [Unclear instructions, missing information, labeling problems]
EXPECTATION_MISALIGNMENT: [Customer expectations vs actual product capabilities]

## MEDICAL DEVICE INSIGHTS
THERAPEUTIC_EFFECTIVENESS: [How well the device performs its intended medical function]
USER_SAFETY_PROFILE: [Any safety concerns specific to medical device use]
PATIENT_COMPLIANCE_FACTORS: [Issues that might affect patient adherence or proper use]
CLINICAL_OUTCOMES: [Impact on patient health, recovery, or quality of life]

## CAPA RECOMMENDATIONS
IMMEDIATE_ACTIONS: [Critical fixes needed within 24-48 hours]
SHORT_TERM_FIXES: [Improvements needed within 1-4 weeks]
LONG_TERM_IMPROVEMENTS: [Design or process changes for future iterations]
REGULATORY_CONSIDERATIONS: [Any FDA, ISO 13485, or other compliance implications]

## ACTIONABLE_INSIGHTS
LISTING_OPTIMIZATION: [Specific changes to Amazon listing to set proper expectations]
PRODUCT_IMPROVEMENTS: [Engineering or design modifications needed]
CUSTOMER_EDUCATION: [Additional guidance or instructions needed for customers]
QUALITY_MONITORING: [Key metrics to track for ongoing quality management]

Focus on actionable insights that a medical device quality manager can implement immediately to improve customer satisfaction and regulatory compliance."""
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to OpenAI GPT-4o"""
        if not MODULES_AVAILABLE.get('requests', False):
            return {'success': False, 'error': 'Requests module not available'}
        
        import requests
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert medical device quality manager with deep knowledge of ISO 13485, FDA regulations, and customer feedback analysis for medical devices."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent analysis
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
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
            return {'success': False, 'error': f"API call failed: {str(e)}"}
    
    def _parse_ai_response(self, ai_content: str) -> Dict[str, Any]:
        """Parse structured AI response into actionable insights"""
        insights = {
            'quality_categorization': {},
            'risk_assessment': {},
            'sentiment_analysis': {},
            'root_cause_analysis': {},
            'medical_device_insights': {},
            'capa_recommendations': {},
            'actionable_insights': {}
        }
        
        try:
            # Extract each section using regex
            sections = {
                'quality_categorization': r'## QUALITY CATEGORIZATION\s*\n(.*?)(?=## |$)',
                'risk_assessment': r'## RISK ASSESSMENT\s*\n(.*?)(?=## |$)',
                'sentiment_analysis': r'## SENTIMENT ANALYSIS\s*\n(.*?)(?=## |$)',
                'root_cause_analysis': r'## ROOT CAUSE ANALYSIS\s*\n(.*?)(?=## |$)',
                'medical_device_insights': r'## MEDICAL DEVICE INSIGHTS\s*\n(.*?)(?=## |$)',
                'capa_recommendations': r'## CAPA RECOMMENDATIONS\s*\n(.*?)(?=## |$)',
                'actionable_insights': r'## ACTIONABLE_INSIGHTS\s*\n(.*?)(?=## |$)'
            }
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, ai_content, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    insights[section_name] = self._parse_section_content(section_content)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {'error': f"Failed to parse AI response: {str(e)}"}
    
    def _parse_section_content(self, content: str) -> Dict[str, Any]:
        """Parse individual section content"""
        parsed = {}
        
        # Split by lines and parse key-value pairs
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('##'):
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if value.startswith('[') and value.endswith(']'):
                    # Parse list content
                    list_content = value[1:-1].strip()
                    if list_content:
                        parsed[key] = [item.strip() for item in list_content.split(',')]
                    else:
                        parsed[key] = []
                else:
                    parsed[key] = value
        
        return parsed

class TextAnalysisEngine:
    """
    AI-Enhanced Text Analysis Engine optimized for Helium 10 review processing
    
    This engine combines rule-based analysis with AI-powered insights using OpenAI GPT-4o
    to provide comprehensive quality management insights for medical device companies.
    """
    
    def __init__(self):
        """Initialize the AI-enhanced text analysis engine"""
        self.text_processor = AdvancedReviewProcessor()
        self.category_analyzer = EnhancedCategoryAnalyzer()
        self.ai_analyzer = AIEnhancedAnalyzer()
        
        # Check AI availability
        if self.ai_analyzer.is_available():
            logger.info("AI-Enhanced Text Analysis Engine initialized - OpenAI GPT-4o available")
        else:
            logger.warning("AI-Enhanced Text Analysis Engine initialized - AI analysis disabled (no API key)")
        
        logger.info("Enhanced Text Analysis Engine initialized - Optimized for Helium 10 reviews")
    
    def analyze_helium10_reviews(self, review_data: List[Dict[str, Any]], 
                                product_info: Dict[str, Any],
                                date_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        AI-Enhanced analysis function optimized for Helium 10 review data
        
        Args:
            review_data: List of review items from Helium 10 export
            product_info: Product information (ASIN, name, category)
            date_filter: Optional date range filter
            
        Returns:
            Comprehensive analysis results with AI insights
        """
        try:
            # Convert to enhanced ReviewItem objects
            review_items = self._prepare_helium10_reviews(review_data, product_info)
            
            if not review_items:
                logger.warning("No valid review items for analysis")
                return self._create_empty_result(product_info)
            
            # Apply date filtering if specified
            if date_filter:
                review_items = self._apply_date_filter(review_items, date_filter)
                logger.info(f"Date filter applied: {len(review_items)} reviews remaining")
            
            # Enhanced sentiment analysis (baseline)
            self._calculate_enhanced_sentiments(review_items)
            
            # AI-POWERED ANALYSIS (NEW)
            ai_insights = {}
            if self.ai_analyzer.is_available() and len(review_items) > 0:
                logger.info("Running AI-powered analysis on customer reviews...")
                
                # Process reviews in batches for AI analysis
                batch_size = 10  # Optimal for API token limits
                all_ai_insights = []
                
                for i in range(0, len(review_items), batch_size):
                    batch = review_items[i:i + batch_size]
                    batch_insights = self.ai_analyzer.analyze_reviews_with_ai(batch, product_info)
                    
                    if batch_insights.get('success'):
                        all_ai_insights.append(batch_insights['ai_insights'])
                        logger.info(f"AI analysis completed for batch {i//batch_size + 1}")
                    else:
                        logger.warning(f"AI analysis failed for batch {i//batch_size + 1}: {batch_insights.get('error')}")
                
                # Consolidate AI insights
                if all_ai_insights:
                    ai_insights = self._consolidate_ai_insights(all_ai_insights)
                    logger.info(f"AI analysis completed - {len(all_ai_insights)} batches processed")
                else:
                    logger.warning("No AI insights generated - falling back to rule-based analysis")
            else:
                logger.info("AI analysis not available - using rule-based analysis only")
            
            # Enhanced category analysis (combining rule-based + AI)
            category_results = self.category_analyzer.analyze_reviews_by_categories(review_items)
            
            # Enhance category results with AI insights
            if ai_insights:
                category_results = self._enhance_categories_with_ai(category_results, ai_insights)
            
            # Enhanced quality assessment (AI-informed)
            quality_assessment = self._assess_enhanced_quality(review_items, category_results, ai_insights)
            
            # AI-Enhanced CAPA generation
            capa_recommendations = self._generate_ai_enhanced_capa_recommendations(
                category_results, quality_assessment, ai_insights, product_info.get('name', 'Unknown Product')
            )
            
            # Enhanced risk assessment (AI-informed)
            risk_level, risk_score, risk_factors = self._calculate_ai_enhanced_risk(
                category_results, quality_assessment, ai_insights
            )
            
            # Generate AI-enhanced insights
            key_insights = self._generate_ai_enhanced_insights(
                category_results, quality_assessment, ai_insights, review_items
            )
            
            # Calculate confidence scores
            confidence_score = self._calculate_analysis_confidence(review_items, category_results)
            
            # Prepare comprehensive results with AI insights
            result = {
                'success': True,
                'asin': product_info.get('asin', 'unknown'),
                'product_name': product_info.get('name', 'Unknown Product'),
                'analysis_period': self._format_analysis_period(review_items, date_filter),
                'total_reviews': len(review_items),
                
                # Core analysis results (enhanced with AI)
                'category_analysis': {cat_id: asdict(analysis) for cat_id, analysis in category_results.items()},
                'quality_assessment': asdict(quality_assessment),
                'capa_recommendations': [asdict(capa) for capa in capa_recommendations],
                
                # AI-Enhanced insights
                'ai_insights': ai_insights,
                'ai_analysis_available': bool(ai_insights),
                
                # Risk and insights (AI-enhanced)
                'overall_risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'key_insights': key_insights,
                
                # Metadata
                'analysis_timestamp': datetime.now().isoformat(),
                'confidence_score': confidence_score,
                'export_source': 'helium10_reviews',
                'analysis_method': 'ai_enhanced' if ai_insights else 'rule_based'
            }
            
            logger.info(f"AI-Enhanced Helium 10 analysis completed for {product_info.get('name', 'Unknown')}: "
                       f"{len(review_items)} reviews analyzed, {len(capa_recommendations)} CAPA items generated")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI-Enhanced Helium 10 analysis: {str(e)}")
            return self._create_empty_result(product_info, error=str(e))
    
    def _prepare_helium10_reviews(self, review_data: List[Dict[str, Any]], 
                                 product_info: Dict[str, Any]) -> List[ReviewItem]:
        """Convert Helium 10 review data to ReviewItem objects"""
        review_items = []
        
        for item_data in review_data:
            try:
                # Extract text components
                title = item_data.get('review_title', item_data.get('title', ''))
                body = item_data.get('review_body', item_data.get('text', ''))
                
                # Skip if no meaningful content
                if not body and not title:
                    continue
                
                # Process combined text
                combined_text = self.text_processor.preprocess_review_text(title, body)
                if not combined_text or len(combined_text.strip()) < 10:
                    continue
                
                # Create enhanced ReviewItem
                review_item = ReviewItem(
                    text=combined_text,
                    review_title=title,
                    review_body=body,
                    date=item_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                    rating=item_data.get('rating'),
                    author=item_data.get('author', 'Anonymous'),
                    verified=item_data.get('verified', 'Unknown'),
                    helpful_votes=item_data.get('helpful_votes', '0'),
                    has_images=item_data.get('has_images', False),
                    has_videos=item_data.get('has_videos', False),
                    variation=item_data.get('variation', ''),
                    style=item_data.get('style', ''),
                    source=item_data.get('source', 'helium10_export'),
                    asin=product_info.get('asin', ''),
                    product_name=product_info.get('name', 'Unknown Product')
                )
                
                review_items.append(review_item)
                
            except Exception as e:
                logger.warning(f"Error processing review item: {str(e)}")
                continue
        
        logger.info(f"Prepared {len(review_items)} review items for analysis")
        return review_items
    
    def _calculate_enhanced_sentiments(self, review_items: List[ReviewItem]):
        """Calculate enhanced sentiment scores for all reviews"""
        for review in review_items:
            sentiment_score, sentiment_details = self.text_processor.calculate_enhanced_sentiment_score(
                review.review_title, review.review_body, review.rating
            )
            review.sentiment_score = sentiment_score
            
            # Calculate quality impact score
            impact_indicators = self.text_processor.detect_quality_impact_indicators(review.text)
            impact_score = (
                len(impact_indicators['severity_indicators']) * 0.4 +
                len(impact_indicators['urgency_indicators']) * 0.3 +
                len(impact_indicators['recommendation_indicators']) * 0.2 +
                len(impact_indicators['repeat_purchase_indicators']) * 0.1
            )
            review.quality_impact_score = min(impact_score, 10.0)
            
            # Calculate confidence based on review completeness
            confidence_factors = []
            confidence_factors.append(0.3 if review.rating is not None else 0.0)
            confidence_factors.append(min(len(review.text) / 100, 0.3))  # Text length factor
            confidence_factors.append(0.2 if review.verified == 'Verified Purchase' else 0.1)
            confidence_factors.append(0.1 if review.has_images or review.has_videos else 0.0)
            confidence_factors.append(0.1 if review.helpful_votes and review.helpful_votes != '0' else 0.0)
            
            review.confidence_score = sum(confidence_factors)
    
    def _assess_enhanced_quality(self, review_items: List[ReviewItem], 
                               category_results: Dict[str, CategoryAnalysis]) -> QualityAssessment:
        """Enhanced quality assessment with Helium 10 specific metrics"""
        total_reviews = len(review_items)
        if total_reviews == 0:
            return self._create_empty_quality_assessment()
        
        # Enhanced sentiment analysis
        positive_count = sum(1 for r in review_items if r.sentiment_score >= 0.6)
        negative_count = sum(1 for r in review_items if r.sentiment_score <= 0.4)
        neutral_count = total_reviews - positive_count - negative_count
        sentiment_ratio = (positive_count - negative_count) / total_reviews
        
        # Star rating analysis
        ratings = [r.rating for r in review_items if r.rating is not None]
        if ratings:
            average_rating = sum(ratings) / len(ratings)
            rating_distribution = dict(Counter(ratings))
        else:
            average_rating = 0.0
            rating_distribution = {}
        
        # Quality score calculation (enhanced)
        base_score = (sum(r.sentiment_score for r in review_items) / total_reviews) * 100
        
        # Quality penalties for high-risk categories
        safety_issues = category_results.get('safety_concerns', CategoryAnalysis('', '', 0, 0, '', '', 0, [], [], {}, {}, {}, 0, False, False, 0, {})).count
        efficacy_issues = category_results.get('efficacy_performance', CategoryAnalysis('', '', 0, 0, '', '', 0, [], [], {}, {}, {}, 0, False, False, 0, {})).count
        biocompat_issues = category_results.get('biocompatibility', CategoryAnalysis('', '', 0, 0, '', '', 0, [], [], {}, {}, {}, 0, False, False, 0, {})).count
        
        # Penalties (enhanced)
        safety_penalty = min(safety_issues * 20, 40)  # Up to 40 points for safety
        efficacy_penalty = min(efficacy_issues * 15, 30)  # Up to 30 points for efficacy
        biocompat_penalty = min(biocompat_issues * 18, 35)  # Up to 35 points for biocompatibility
        
        quality_score = max(0, base_score - safety_penalty - efficacy_penalty - biocompat_penalty)
        quality_level = self._determine_quality_level(quality_score)
        
        # High-risk categories
        high_risk_categories = [
            cat_id for cat_id, analysis in category_results.items()
            if analysis.risk_score > 15 and analysis.count > 0
        ]
        
        # Enhanced metrics
        verified_buyers = sum(1 for r in review_items if r.verified == 'Verified Purchase')
        verified_percentage = (verified_buyers / total_reviews) * 100
        
        helpful_reviews = sum(1 for r in review_items if r.helpful_votes and r.helpful_votes != '0')
        helpfulness_score = (helpful_reviews / total_reviews) * 100
        
        media_reviews = sum(1 for r in review_items if r.has_images or r.has_videos)
        
        # Rating trend analysis
        rating_trend = self._analyze_rating_trend(review_items)
        
        # Improvement indicators
        improvement_trend = self._analyze_improvement_trend(review_items)
        key_strengths = self._identify_key_strengths(review_items)
        primary_concerns = self._identify_primary_concerns(category_results)
        
        return QualityAssessment(
            total_reviews=total_reviews,
            quality_score=round(quality_score, 1),
            quality_level=quality_level,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            sentiment_ratio=round(sentiment_ratio, 3),
            rating_distribution=rating_distribution,
            average_rating=round(average_rating, 2),
            rating_trend=rating_trend,
            high_risk_categories=high_risk_categories,
            safety_issues_count=safety_issues,
            efficacy_issues_count=efficacy_issues,
            verified_buyer_percentage=round(verified_percentage, 1),
            review_helpfulness_score=round(helpfulness_score, 1),
            reviews_with_media_count=media_reviews,
            improvement_trend=improvement_trend,
            key_strengths=key_strengths,
            primary_concerns=primary_concerns
        )
    
    def _consolidate_ai_insights(self, all_ai_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate AI insights from multiple batches"""
        consolidated = {
            'quality_categorization': {},
            'risk_assessment': {},
            'sentiment_analysis': {},
            'root_cause_analysis': {},
            'medical_device_insights': {},
            'capa_recommendations': {},
            'actionable_insights': {},
            'confidence_score': 0.8  # High confidence for AI analysis
        }
        
        # Merge categorizations
        all_categories = set()
        for insights in all_ai_insights:
            if 'quality_categorization' in insights:
                for category, items in insights['quality_categorization'].items():
                    if category not in consolidated['quality_categorization']:
                        consolidated['quality_categorization'][category] = []
                    if isinstance(items, list):
                        consolidated['quality_categorization'][category].extend(items)
                    all_categories.add(category)
        
        # Merge risk assessments
        risk_levels = ['critical_issues', 'high_priority', 'medium_priority', 'low_priority']
        for level in risk_levels:
            consolidated['risk_assessment'][level] = []
            for insights in all_ai_insights:
                if 'risk_assessment' in insights and level in insights['risk_assessment']:
                    items = insights['risk_assessment'][level]
                    if isinstance(items, list):
                        consolidated['risk_assessment'][level].extend(items)
        
        # Take most recent sentiment analysis
        if all_ai_insights and 'sentiment_analysis' in all_ai_insights[-1]:
            consolidated['sentiment_analysis'] = all_ai_insights[-1]['sentiment_analysis']
        
        # Merge root cause analyses
        consolidated['root_cause_analysis'] = {}
        for insights in all_ai_insights:
            if 'root_cause_analysis' in insights:
                for cause_type, details in insights['root_cause_analysis'].items():
                    if cause_type not in consolidated['root_cause_analysis']:
                        consolidated['root_cause_analysis'][cause_type] = []
                    if isinstance(details, list):
                        consolidated['root_cause_analysis'][cause_type].extend(details)
                    elif isinstance(details, str) and details:
                        consolidated['root_cause_analysis'][cause_type].append(details)
        
        # Merge medical device insights
        consolidated['medical_device_insights'] = {}
        for insights in all_ai_insights:
            if 'medical_device_insights' in insights:
                for insight_type, details in insights['medical_device_insights'].items():
                    if insight_type not in consolidated['medical_device_insights']:
                        consolidated['medical_device_insights'][insight_type] = []
                    if isinstance(details, list):
                        consolidated['medical_device_insights'][insight_type].extend(details)
                    elif isinstance(details, str) and details:
                        consolidated['medical_device_insights'][insight_type].append(details)
        
        # Merge CAPA recommendations
        capa_types = ['immediate_actions', 'short_term_fixes', 'long_term_improvements', 'regulatory_considerations']
        for capa_type in capa_types:
            consolidated['capa_recommendations'][capa_type] = []
            for insights in all_ai_insights:
                if 'capa_recommendations' in insights and capa_type in insights['capa_recommendations']:
                    items = insights['capa_recommendations'][capa_type]
                    if isinstance(items, list):
                        consolidated['capa_recommendations'][capa_type].extend(items)
                    elif isinstance(items, str) and items:
                        consolidated['capa_recommendations'][capa_type].append(items)
        
        # Merge actionable insights
        insight_types = ['listing_optimization', 'product_improvements', 'customer_education', 'quality_monitoring']
        for insight_type in insight_types:
            consolidated['actionable_insights'][insight_type] = []
            for insights in all_ai_insights:
                if 'actionable_insights' in insights and insight_type in insights['actionable_insights']:
                    items = insights['actionable_insights'][insight_type]
                    if isinstance(items, list):
                        consolidated['actionable_insights'][insight_type].extend(items)
                    elif isinstance(items, str) and items:
                        consolidated['actionable_insights'][insight_type].append(items)
        
        return consolidated
    
    def _enhance_categories_with_ai(self, category_results: Dict[str, CategoryAnalysis], 
                                  ai_insights: Dict[str, Any]) -> Dict[str, CategoryAnalysis]:
        """Enhance rule-based category analysis with AI insights"""
        
        if not ai_insights or 'quality_categorization' not in ai_insights:
            return category_results
        
        ai_categorization = ai_insights['quality_categorization']
        
        # Map AI categories to our framework
        ai_category_mapping = {
            'safety_concerns': 'safety_concerns',
            'efficacy_performance': 'efficacy_performance', 
            'comfort_usability': 'comfort_usability',
            'durability_quality': 'durability_quality',
            'sizing_fit': 'sizing_fit',
            'assembly_instructions': 'assembly_instructions',
            'biocompatibility': 'biocompatibility',
            'shipping_packaging': 'shipping_packaging'
        }
        
        # Enhance each category with AI insights
        for ai_category, our_category in ai_category_mapping.items():
            if ai_category in ai_categorization and our_category in category_results:
                ai_identified_reviews = ai_categorization[ai_category]
                
                # Add AI confidence boost to categories identified by AI
                if ai_identified_reviews:
                    analysis = category_results[our_category]
                    # Boost risk score for AI-identified issues
                    analysis.risk_score = min(analysis.risk_score * 1.2, 30.0)
                    
                    # Add AI insight flag
                    if not hasattr(analysis, 'ai_confirmed'):
                        analysis.ai_confirmed = True
                        analysis.ai_insights = ai_identified_reviews
        
        return category_results
    
    def _assess_enhanced_quality(self, review_items: List[ReviewItem], 
                               category_results: Dict[str, CategoryAnalysis],
                               ai_insights: Dict[str, Any] = None) -> QualityAssessment:
        """Enhanced quality assessment with AI insights"""
        
        # Start with base assessment
        quality_assessment = self._assess_base_quality(review_items, category_results)
        
        # Enhance with AI insights if available
        if ai_insights:
            # AI sentiment analysis override
            if 'sentiment_analysis' in ai_insights:
                ai_sentiment = ai_insights['sentiment_analysis']
                
                # Use AI overall sentiment if available and confident
                if 'overall_sentiment' in ai_sentiment:
                    ai_sentiment_text = ai_sentiment['overall_sentiment'].lower()
                    if 'positive' in ai_sentiment_text:
                        # AI indicates positive - boost quality score
                        quality_assessment.quality_score = min(quality_assessment.quality_score * 1.1, 100)
                    elif 'negative' in ai_sentiment_text:
                        # AI indicates negative - reduce quality score
                        quality_assessment.quality_score = max(quality_assessment.quality_score * 0.9, 0)
            
            # AI medical device insights
            if 'medical_device_insights' in ai_insights:
                medical_insights = ai_insights['medical_device_insights']
                
                # Safety profile assessment
                if 'user_safety_profile' in medical_insights:
                    safety_concerns = medical_insights['user_safety_profile']
                    if isinstance(safety_concerns, list) and len(safety_concerns) > 0:
                        # AI identified safety issues - major quality impact
                        quality_assessment.quality_score = max(quality_assessment.quality_score - 20, 0)
                        quality_assessment.safety_issues_count += len(safety_concerns)
                
                # Therapeutic effectiveness
                if 'therapeutic_effectiveness' in medical_insights:
                    effectiveness = medical_insights['therapeutic_effectiveness']
                    if isinstance(effectiveness, list):
                        for item in effectiveness:
                            if isinstance(item, str) and ('ineffective' in item.lower() or 'not working' in item.lower()):
                                quality_assessment.efficacy_issues_count += 1
        
        # Recalculate quality level based on AI-enhanced score
        quality_assessment.quality_level = self._determine_quality_level(quality_assessment.quality_score)
        
        return quality_assessment
    
    def _generate_ai_enhanced_capa_recommendations(self, category_results: Dict[str, CategoryAnalysis],
                                                 quality_assessment: QualityAssessment,
                                                 ai_insights: Dict[str, Any],
                                                 product_name: str) -> List[CAPARecommendation]:
        """Generate CAPA recommendations enhanced with AI insights"""
        
        # Start with base CAPA recommendations
        base_capas = self._generate_base_capa_recommendations(category_results, quality_assessment, product_name)
        
        # Enhance with AI-specific CAPAs
        ai_capas = []
        capa_counter = len(base_capas) + 1
        
        if ai_insights and 'capa_recommendations' in ai_insights:
            ai_capa_data = ai_insights['capa_recommendations']
            
            # Immediate actions from AI
            if 'immediate_actions' in ai_capa_data:
                for action in ai_capa_data['immediate_actions']:
                    if isinstance(action, str) and action.strip():
                        ai_capa = CAPARecommendation(
                            capa_id=f"AI-CAPA-{capa_counter:03d}",
                            priority='Critical',
                            category='AI-Identified Critical Issue',
                            issue_description=f"AI analysis identified critical issue requiring immediate attention",
                            root_cause_analysis="AI-powered analysis of customer feedback patterns",
                            affected_customer_count=0,  # AI doesn't provide specific count
                            customer_impact_assessment="High impact identified by AI analysis",
                            representative_reviews=[],
                            rating_impact="AI-identified rating impact",
                            review_volume_trend="AI-analyzed trend",
                            corrective_action=action,
                            preventive_action="Implement AI-recommended preventive measures",
                            timeline='Immediate (24-48 hours)',
                            responsibility='Quality Manager + AI Insights Team',
                            success_metrics=['Resolve AI-identified critical issue', 'Improve customer feedback sentiment'],
                            verification_method='AI-powered monitoring of subsequent customer feedback',
                            target_improvement='Eliminate AI-identified critical issues',
                            iso_reference='ISO 13485 Section 8.2.2 - Customer Feedback',
                            regulatory_impact='high',
                            documentation_required=True
                        )
                        ai_capas.append(ai_capa)
                        capa_counter += 1
            
            # Short-term fixes from AI
            if 'short_term_fixes' in ai_capa_data:
                for fix in ai_capa_data['short_term_fixes']:
                    if isinstance(fix, str) and fix.strip():
                        ai_capa = CAPARecommendation(
                            capa_id=f"AI-CAPA-{capa_counter:03d}",
                            priority='High',
                            category='AI-Identified Quality Improvement',
                            issue_description=f"AI analysis identified quality improvement opportunity",
                            root_cause_analysis="AI pattern recognition in customer feedback",
                            affected_customer_count=0,
                            customer_impact_assessment="Medium to high impact identified by AI",
                            representative_reviews=[],
                            rating_impact="AI-identified rating correlation",
                            review_volume_trend="AI-analyzed",
                            corrective_action=fix,
                            preventive_action="Implement AI-recommended process improvements",
                            timeline='1-2 weeks',
                            responsibility='Quality Manager + Product Development',
                            success_metrics=['Implement AI-recommended fix', 'Monitor customer feedback improvement'],
                            verification_method='AI-powered trend analysis of customer feedback',
                            target_improvement='Address AI-identified quality gaps',
                            iso_reference='ISO 13485 Section 8.5 - Improvement',
                            regulatory_impact='medium',
                            documentation_required=True
                        )
                        ai_capas.append(ai_capa)
                        capa_counter += 1
        
        # Combine base and AI CAPAs, sort by priority
        all_capas = base_capas + ai_capas
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        all_capas.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return all_capas
    
    def _calculate_ai_enhanced_risk(self, category_results: Dict[str, CategoryAnalysis],
                                  quality_assessment: QualityAssessment,
                                  ai_insights: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Calculate risk level enhanced with AI insights"""
        
        # Start with base risk calculation
        risk_level, risk_score, risk_factors = self._calculate_base_risk(category_results, quality_assessment)
        
        # Enhance with AI risk assessment
        if ai_insights and 'risk_assessment' in ai_insights:
            ai_risk = ai_insights['risk_assessment']
            
            # Critical issues from AI
            if 'critical_issues' in ai_risk:
                critical_items = ai_risk['critical_issues']
                if isinstance(critical_items, list) and len(critical_items) > 0:
                    risk_score += len(critical_items) * 10  # High weight for AI-identified critical issues
                    risk_factors.extend([f"AI-identified critical issue: {item}" for item in critical_items[:3]])
            
            # High priority issues from AI
            if 'high_priority' in ai_risk:
                high_items = ai_risk['high_priority']
                if isinstance(high_items, list) and len(high_items) > 0:
                    risk_score += len(high_items) * 5
                    risk_factors.extend([f"AI-identified high priority: {item}" for item in high_items[:2]])
        
        # AI medical device specific risks
        if ai_insights and 'medical_device_insights' in ai_insights:
            medical_insights = ai_insights['medical_device_insights']
            
            # User safety profile risks
            if 'user_safety_profile' in medical_insights:
                safety_items = medical_insights['user_safety_profile']
                if isinstance(safety_items, list) and len(safety_items) > 0:
                    risk_score += len(safety_items) * 15  # Very high weight for safety
                    risk_factors.append(f"AI-identified safety concerns affecting {len(safety_items)} areas")
        
        # Recalculate risk level based on AI-enhanced score
        if risk_score >= 50:
            risk_level = RiskLevel.CRITICAL.value
        elif risk_score >= 30:
            risk_level = RiskLevel.HIGH.value
        elif risk_score >= 15:
            risk_level = RiskLevel.MEDIUM.value
        elif risk_score >= 5:
            risk_level = RiskLevel.LOW.value
        else:
            risk_level = RiskLevel.MINIMAL.value
        
        return risk_level, round(risk_score, 1), risk_factors[:10]  # Limit to top 10 risk factors
    
    def _generate_ai_enhanced_insights(self, category_results: Dict[str, CategoryAnalysis],
                                     quality_assessment: QualityAssessment, 
                                     ai_insights: Dict[str, Any],
                                     review_items: List[ReviewItem]) -> List[str]:
        """Generate insights enhanced with AI analysis"""
        
        insights = []
        
        # AI-specific insights first
        if ai_insights:
            # AI sentiment insights
            if 'sentiment_analysis' in ai_insights:
                ai_sentiment = ai_insights['sentiment_analysis']
                if 'overall_sentiment' in ai_sentiment:
                    sentiment_text = ai_sentiment['overall_sentiment']
                    insights.append(f"AI Analysis: {sentiment_text}")
            
            # AI medical device insights
            if 'medical_device_insights' in ai_insights:
                medical_insights = ai_insights['medical_device_insights']
                
                if 'therapeutic_effectiveness' in medical_insights:
                    effectiveness = medical_insights['therapeutic_effectiveness']
                    if isinstance(effectiveness, list) and len(effectiveness) > 0:
                        insights.append(f"AI Medical Assessment: {effectiveness[0]}")
                
                if 'patient_compliance_factors' in medical_insights:
                    compliance = medical_insights['patient_compliance_factors']
                    if isinstance(compliance, list) and len(compliance) > 0:
                        insights.append(f"AI Compliance Analysis: {compliance[0]}")
            
            # AI actionable insights
            if 'actionable_insights' in ai_insights:
                actionable = ai_insights['actionable_insights']
                
                if 'listing_optimization' in actionable:
                    listing_opts = actionable['listing_optimization']
                    if isinstance(listing_opts, list) and len(listing_opts) > 0:
                        insights.append(f"AI Listing Recommendation: {listing_opts[0]}")
                
                if 'product_improvements' in actionable:
                    improvements = actionable['product_improvements']
                    if isinstance(improvements, list) and len(improvements) > 0:
                        insights.append(f"AI Product Improvement: {improvements[0]}")
        
        # Traditional insights (enhanced)
        traditional_insights = self._generate_traditional_insights(category_results, quality_assessment, review_items)
        insights.extend(traditional_insights[:3])  # Limit traditional insights
        
        return insights[:8]  # Return top 8 insights
    
    def _assess_rating_impact(self, analysis: CategoryAnalysis) -> str:
        """Assess how category issues impact star ratings"""
        correlation = analysis.rating_correlation.get('correlation', 'neutral')
        avg_rating = analysis.avg_rating_for_category
        
        if correlation == 'strong_negative':
            return f"Strongly correlates with low ratings (avg: {avg_rating:.1f} stars)"
        elif correlation == 'moderate_negative':
            return f"Moderately correlates with lower ratings (avg: {avg_rating:.1f} stars)"
        elif correlation == 'positive':
            return f"Associated with higher ratings (avg: {avg_rating:.1f} stars)"
        else:
            return f"Neutral rating impact (avg: {avg_rating:.1f} stars)"
    
    def _create_enhanced_category_capa(self, capa_id: str, category_id: str, 
                                     analysis: CategoryAnalysis, product_name: str,
                                     representative_reviews: List[str],
                                     rating_impact: str,
                                     review_volume_trend: str) -> CAPARecommendation:
        """Create enhanced CAPA for specific category issues"""
        
        # Determine priority based on multiple factors
        priority_score = 0
        priority_score += analysis.risk_score / 5  # Risk contribution
        priority_score += analysis.count / 10  # Volume contribution
        
        if analysis.immediate_action_required:
            priority = 'Critical'
        elif priority_score >= 6:
            priority = 'High'
        elif priority_score >= 3:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        # Enhanced issue description
        customer_impact = f"Affects {analysis.count} customers ({analysis.percentage:.1f}% of reviews)"
        if analysis.rating_correlation.get('low_rating_percentage', 0) > 50:
            customer_impact += f", strongly impacts customer satisfaction"
        
        # Timeline based on priority and category
        timeline_map = {
            'Critical': 'Immediate (24-48 hours)',
            'High': '1-2 weeks',
            'Medium': '2-4 weeks',
            'Low': '4-8 weeks'
        }
        
        # Target improvement
        target_improvement = f"Reduce {analysis.name.lower()} complaints by 50% within 60 days"
        if analysis.avg_rating_for_category < 3.5:
            target_improvement += f", improve category average rating to 4.0+"
        
        return CAPARecommendation(
            capa_id=capa_id,
            priority=priority,
            category=analysis.name,
            issue_description=f"{analysis.count} customer reviews indicate {analysis.name.lower()} issues",
            root_cause_analysis=f"Analysis reveals recurring {category_id.replace('_', ' ')} concerns in customer feedback",
            affected_customer_count=analysis.count,
            customer_impact_assessment=customer_impact,
            representative_reviews=representative_reviews,
            rating_impact=rating_impact,
            review_volume_trend=f"Trend: {review_volume_trend}",
            corrective_action=self._get_enhanced_corrective_action(category_id, analysis),
            preventive_action=self._get_enhanced_preventive_action(category_id, analysis),
            timeline=timeline_map[priority],
            responsibility=self._get_responsibility_assignment(category_id),
            success_metrics=[
                f"Reduce {analysis.name.lower()} complaints by 50%",
                f"Improve average star rating for this category to 4.0+",
                "Zero critical safety incidents reported"
            ],
            verification_method=f"Weekly review of new customer feedback for {analysis.name.lower()} issues",
            target_improvement=target_improvement,
            iso_reference=analysis.iso_reference,
            regulatory_impact=MEDICAL_DEVICE_QUALITY_FRAMEWORK[category_id].get('regulatory_impact', 'medium'),
            documentation_required=priority in ['Critical', 'High']
        )
    
    # Additional helper methods...
    def _apply_date_filter(self, review_items: List[ReviewItem], 
                          date_filter: Dict[str, Any]) -> List[ReviewItem]:
        """Apply date range filtering"""
        if not date_filter or not date_filter.get('start_date'):
            return review_items
        
        start_date = date_filter['start_date']
        end_date = date_filter.get('end_date', datetime.now().date())
        
        filtered_items = []
        for item in review_items:
            try:
                if isinstance(item.date, str):
                    item_date = datetime.strptime(item.date, '%Y-%m-%d').date()
                else:
                    item_date = item.date
                
                if start_date <= item_date <= end_date:
                    filtered_items.append(item)
            except (ValueError, TypeError):
                filtered_items.append(item)  # Include items with unparseable dates
        
        return filtered_items
    
    def _calculate_analysis_confidence(self, review_items: List[ReviewItem],
                                     category_results: Dict[str, CategoryAnalysis]) -> float:
        """Calculate overall analysis confidence score"""
        if not review_items:
            return 0.0
        
        # Average individual review confidence
        avg_review_confidence = sum(r.confidence_score for r in review_items) / len(review_items)
        
        # Data quantity factor
        quantity_score = min(len(review_items) / 50, 1.0)
        
        # Rating availability
        rated_reviews = sum(1 for r in review_items if r.rating is not None)
        rating_score = rated_reviews / len(review_items)
        
        # Verification factor
        verified_reviews = sum(1 for r in review_items if r.verified == 'Verified Purchase')
        verification_score = verified_reviews / len(review_items)
        
        # Combine factors
        confidence = (
            avg_review_confidence * 0.4 +
            quantity_score * 0.3 +
            rating_score * 0.2 +
            verification_score * 0.1
        )
        
        return round(confidence, 3)
    
    def _create_empty_result(self, product_info: Dict[str, Any], error: str = None) -> Dict[str, Any]:
        """Create empty result for error scenarios"""
        return {
            'success': False,
            'asin': product_info.get('asin', 'unknown'),
            'product_name': product_info.get('name', 'Unknown Product'),
            'error': error,
            'total_reviews': 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'export_source': 'helium10_reviews'
        }
    
    # Placeholder methods for brevity - would implement full logic
    def _determine_quality_level(self, score: float) -> str:
        if score >= 85: return 'Excellent'
        elif score >= 70: return 'Good'
        elif score >= 55: return 'Fair'
        elif score >= 40: return 'Poor'
        else: return 'Critical'
    
    def _analyze_rating_trend(self, review_items: List[ReviewItem]) -> str:
        return 'stable'  # Simplified
    
    def _analyze_improvement_trend(self, review_items: List[ReviewItem]) -> str:
        return 'stable'  # Simplified
    
    def _identify_key_strengths(self, review_items: List[ReviewItem]) -> List[str]:
        return []  # Simplified
    
    def _identify_primary_concerns(self, category_results: Dict[str, CategoryAnalysis]) -> List[str]:
        return []  # Simplified
    
    def _calculate_enhanced_risk(self, category_results: Dict[str, CategoryAnalysis],
                               quality_assessment: QualityAssessment) -> Tuple[str, float, List[str]]:
        return 'Low', 5.0, []  # Simplified
    
    def _generate_enhanced_insights(self, category_results: Dict[str, CategoryAnalysis],
                                  quality_assessment: QualityAssessment, 
                                  review_items: List[ReviewItem]) -> List[str]:
        return []  # Simplified
    
    def _format_analysis_period(self, review_items: List[ReviewItem],
                              date_filter: Optional[Dict[str, Any]]) -> str:
        return 'Full period'  # Simplified
    
    def _create_empty_quality_assessment(self) -> QualityAssessment:
        return QualityAssessment(
            total_reviews=0, quality_score=0.0, quality_level='No Data',
            positive_count=0, negative_count=0, neutral_count=0, sentiment_ratio=0.0,
            rating_distribution={}, average_rating=0.0, rating_trend='no_data',
            high_risk_categories=[], safety_issues_count=0, efficacy_issues_count=0,
            verified_buyer_percentage=0.0, review_helpfulness_score=0.0, reviews_with_media_count=0,
            improvement_trend='no_data', key_strengths=[], primary_concerns=[]
        )
    
    def _get_enhanced_corrective_action(self, category_id: str, analysis: CategoryAnalysis) -> str:
        return "Implement corrective measures"  # Simplified
    
    def _get_enhanced_preventive_action(self, category_id: str, analysis: CategoryAnalysis) -> str:
        return "Implement preventive measures"  # Simplified
    
    def _get_responsibility_assignment(self, category_id: str) -> str:
        return "Quality Manager"  # Simplified
    
    def _create_enhanced_overall_quality_capa(self, capa_id: str, quality_assessment: QualityAssessment,
                                            product_name: str) -> CAPARecommendation:
        return CAPARecommendation(
            capa_id=capa_id, priority='Medium', category='Overall Quality',
            issue_description='Overall quality improvement needed',
            root_cause_analysis='Multiple quality factors affecting customer satisfaction',
            affected_customer_count=quality_assessment.negative_count,
            customer_impact_assessment='Impacts overall satisfaction',
            representative_reviews=[], rating_impact='Affects overall ratings',
            review_volume_trend='stable', corrective_action='Implement quality review',
            preventive_action='Enhance quality processes', timeline='2-4 weeks',
            responsibility='Quality Manager', success_metrics=['Improve quality score'],
            verification_method='Monthly review', target_improvement='Achieve 75% quality score',
            iso_reference='ISO 13485 Section 8.5', regulatory_impact='medium',
            documentation_required=True
        )

# Export the main enhanced classes with AI integration
__all__ = [
    'TextAnalysisEngine',
    'AIEnhancedAnalyzer',
    'ReviewItem', 
    'CategoryAnalysis',
    'QualityAssessment',
    'CAPARecommendation',
    'QualityCategory',
    'RiskLevel',
    'MEDICAL_DEVICE_QUALITY_FRAMEWORK'
]
