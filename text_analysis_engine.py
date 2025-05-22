"""
Text Analysis Engine for Medical Device Customer Feedback Analysis

**CORE MODULE - PRIMARY ANALYTICAL COMPONENT**

This is the main text analysis engine that processes customer comments, reviews, and 
return reasons to provide quality management insights for medical device listings.

Key Functions:
✓ Medical device-specific text categorization (safety, comfort, efficacy, etc.)
✓ Date range filtering for temporal analysis
✓ Quality risk assessment and CAPA generation
✓ ISO 13485 compliance-aware analysis
✓ Advanced NLP for customer sentiment and intent detection
✓ Temporal trend analysis and pattern recognition

Author: Assistant
Version: 3.0 - Core Text Analysis Engine
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

# Medical Device Quality Categories (ISO 13485 Aligned)
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
    REGULATORY_COMPLIANCE = "regulatory_compliance"

# Comprehensive Medical Device Quality Framework
MEDICAL_DEVICE_QUALITY_FRAMEWORK = {
    QualityCategory.SAFETY_CONCERNS.value: {
        'name': 'Safety & Risk Management',
        'description': 'Customer feedback indicating potential safety hazards or injury risks',
        'keywords': [
            # Direct safety terms
            'unsafe', 'dangerous', 'hazardous', 'injury', 'hurt', 'injured', 'harm', 'accident',
            # Structural safety
            'broke', 'broken', 'collapsed', 'fell apart', 'unstable', 'wobbly', 'tip over', 'tipped',
            # Sharp/cutting hazards
            'sharp', 'cuts', 'cut me', 'cutting', 'pinch', 'pinched', 'trapped', 'caught',
            # Fall/mobility safety
            'slipped', 'fall', 'fell', 'falling', 'slide', 'slip', 'lose balance',
            # Electrical/mechanical safety
            'shock', 'electrical', 'sparks', 'overheated', 'burning', 'smoke',
            # Emergency situations
            'emergency', 'emergency room', 'hospital', 'doctor visit', 'urgent care'
        ],
        'severity': 'critical',
        'iso_reference': 'ISO 13485 Section 7.3 - Risk Management',
        'regulatory_impact': 'high',
        'immediate_action_required': True
    },
    
    QualityCategory.EFFICACY_PERFORMANCE.value: {
        'name': 'Efficacy & Performance',
        'description': 'Feedback about product effectiveness and functional performance',
        'keywords': [
            # Ineffectiveness
            'doesnt work', 'not working', 'ineffective', 'useless', 'waste of money',
            'no relief', 'no help', 'no improvement', 'no difference', 'no benefit',
            # Performance issues
            'poor performance', 'weak', 'insufficient', 'inadequate', 'subpar',
            'disappointing', 'expected more', 'not as advertised', 'false claims',
            # Functional problems
            'malfunctioning', 'defective', 'faulty', 'not functioning', 'stopped working',
            'inconsistent', 'unreliable', 'intermittent', 'sporadic performance'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 13485 Section 7.3 - Performance Requirements',
        'regulatory_impact': 'medium',
        'immediate_action_required': False
    },
    
    QualityCategory.COMFORT_USABILITY.value: {
        'name': 'Comfort & Usability',
        'description': 'User experience, comfort, and ease of use issues',
        'keywords': [
            # Comfort issues
            'uncomfortable', 'painful', 'hurts', 'sore', 'aches', 'pressure',
            'rough', 'hard', 'stiff', 'rigid', 'tight', 'loose', 'scratchy',
            # Usability problems
            'difficult to use', 'hard to use', 'confusing', 'complicated', 'awkward',
            'user unfriendly', 'not intuitive', 'hard to operate', 'cumbersome',
            # Ergonomic issues
            'poor ergonomics', 'bad design', 'poorly designed', 'not ergonomic',
            'strain', 'fatigue', 'tiring', 'exhausting', 'heavy', 'bulky'
        ],
        'severity': 'medium',
        'iso_reference': 'ISO 13485 Section 7.3 - User Requirements',
        'regulatory_impact': 'low',
        'immediate_action_required': False
    },
    
    QualityCategory.DURABILITY_QUALITY.value: {
        'name': 'Durability & Build Quality',
        'description': 'Product longevity, material quality, and construction issues',
        'keywords': [
            # Material failure
            'cheap', 'cheaply made', 'poor quality', 'low quality', 'flimsy', 'fragile',
            'weak material', 'thin', 'brittle', 'cracked', 'split', 'torn', 'ripped',
            # Structural failure
            'fell apart', 'came apart', 'broke', 'broken', 'bent', 'snapped', 'warped',
            'loose screws', 'loose parts', 'wobbly', 'unstable', 'deteriorated',
            # Wear and tear
            'worn out', 'wearing out', 'fading', 'discolored', 'peeling', 'chipping',
            'rust', 'corrosion', 'degraded', 'short lifespan', 'doesnt last'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 13485 Section 7.5 - Production Controls',
        'regulatory_impact': 'medium',
        'immediate_action_required': False
    },
    
    QualityCategory.SIZING_FIT.value: {
        'name': 'Sizing & Fit Issues',
        'description': 'Size accuracy, fit problems, and measurement discrepancies',
        'keywords': [
            # Size problems
            'too small', 'too big', 'too large', 'wrong size', 'incorrect size',
            'runs small', 'runs large', 'runs big', 'sizing issue', 'size problem',
            # Fit issues
            'doesnt fit', 'poor fit', 'bad fit', 'tight', 'loose', 'baggy',
            'doesnt stay in place', 'slips', 'slides', 'moves around',
            # Measurement discrepancies
            'measurements wrong', 'measurements off', 'sizing chart wrong', 'inaccurate measurements',
            'different than described', 'not as pictured', 'smaller than expected', 'larger than expected'
        ],
        'severity': 'medium',
        'iso_reference': 'ISO 13485 Section 7.3 - Design Specifications',
        'regulatory_impact': 'low',
        'immediate_action_required': False
    },
    
    QualityCategory.ASSEMBLY_INSTRUCTIONS.value: {
        'name': 'Assembly & Instructions',
        'description': 'Setup difficulties, unclear instructions, and missing components',
        'keywords': [
            # Assembly problems
            'difficult assembly', 'hard to assemble', 'assembly problems', 'setup issues',
            'complicated setup', 'confusing assembly', 'poor assembly', 'assembly nightmare',
            # Instruction issues
            'unclear instructions', 'confusing instructions', 'poor instructions', 'bad directions',
            'missing instructions', 'incomplete instructions', 'hard to follow', 'poorly written',
            # Missing components
            'missing parts', 'missing pieces', 'missing hardware', 'missing screws',
            'incomplete kit', 'parts missing', 'wrong parts', 'damaged parts',
            # Documentation
            'no manual', 'missing manual', 'poor documentation', 'inadequate documentation'
        ],
        'severity': 'medium',
        'iso_reference': 'ISO 13485 Section 4.2 - Documentation Requirements',
        'regulatory_impact': 'low',
        'immediate_action_required': False
    },
    
    QualityCategory.SHIPPING_PACKAGING.value: {
        'name': 'Shipping & Packaging',
        'description': 'Delivery, packaging, and shipping-related issues',
        'keywords': [
            # Shipping damage
            'arrived damaged', 'damaged in shipping', 'shipping damage', 'broken in transit',
            'crushed', 'dented', 'scratched during shipping', 'bent in shipping',
            # Packaging issues
            'poor packaging', 'inadequate packaging', 'bad packaging', 'insufficient padding',
            'not protected', 'poorly packed', 'loose in box', 'rattling around',
            # Delivery problems
            'late delivery', 'delayed shipping', 'wrong item shipped', 'missing items',
            'partial shipment', 'delivery issues', 'shipping problems'
        ],
        'severity': 'low',
        'iso_reference': 'ISO 13485 Section 7.5 - Packaging Requirements',
        'regulatory_impact': 'low',
        'immediate_action_required': False
    },
    
    QualityCategory.BIOCOMPATIBILITY.value: {
        'name': 'Biocompatibility & Materials',
        'description': 'Skin reactions, allergies, and material compatibility issues',
        'keywords': [
            # Allergic reactions
            'allergic reaction', 'allergy', 'rash', 'skin irritation', 'red skin',
            'itchy', 'itching', 'burning sensation', 'stinging', 'tingling',
            # Material issues
            'latex allergy', 'material sensitivity', 'chemical smell', 'odor', 'toxic smell',
            'skin contact issues', 'dermatitis', 'eczema', 'hives', 'swelling',
            # Biocompatibility
            'not biocompatible', 'material reaction', 'sensitive skin', 'skin problems',
            'contact dermatitis', 'material allergy', 'chemical reaction'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 10993 - Biological Evaluation of Medical Devices',
        'regulatory_impact': 'high',
        'immediate_action_required': True
    },
    
    QualityCategory.REGULATORY_COMPLIANCE.value: {
        'name': 'Regulatory & Compliance',
        'description': 'FDA compliance, labeling, and regulatory requirement issues',
        'keywords': [
            # Regulatory mentions
            'fda approved', 'not fda approved', 'medical device', 'prescription required',
            'doctor recommendation', 'medical professional', 'healthcare provider',
            # Labeling issues
            'misleading label', 'incorrect labeling', 'false advertising', 'not as described',
            'medical claims', 'health claims', 'therapeutic claims', 'clinical claims',
            # Compliance
            'regulatory compliance', 'standards compliance', 'certification', 'approval'
        ],
        'severity': 'high',
        'iso_reference': 'ISO 13485 Section 8.2.2 - Customer Feedback',
        'regulatory_impact': 'high',
        'immediate_action_required': True
    }
}

# Positive feedback indicators
POSITIVE_FEEDBACK_INDICATORS = {
    'quality_praise': [
        'excellent quality', 'high quality', 'great quality', 'amazing quality', 'superior quality',
        'well made', 'well built', 'solid construction', 'durable', 'sturdy', 'robust'
    ],
    'effectiveness': [
        'works great', 'very effective', 'highly effective', 'exactly what needed',
        'perfect solution', 'life changing', 'life saver', 'game changer', 'incredible results'
    ],
    'comfort': [
        'very comfortable', 'extremely comfortable', 'so comfortable', 'perfect fit',
        'comfortable to use', 'easy to use', 'user friendly', 'ergonomic'
    ],
    'satisfaction': [
        'love it', 'love this', 'highly recommend', 'would recommend', 'best purchase',
        'excellent product', 'amazing product', 'perfect', 'fantastic', 'outstanding'
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
class FeedbackItem:
    """Structured representation of a single feedback item"""
    text: str
    date: str
    type: str  # 'review', 'return_reason', 'complaint', etc.
    rating: Optional[int] = None
    source: str = 'unknown'
    asin: str = ''
    product_name: str = ''
    
    # Analysis results
    detected_categories: List[str] = None
    sentiment_score: float = 0.0
    severity_level: str = 'medium'
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.detected_categories is None:
            self.detected_categories = []

@dataclass 
class CategoryAnalysis:
    """Analysis results for a specific quality category"""
    category_id: str
    name: str
    count: int
    percentage: float
    severity: str
    iso_reference: str
    
    # Detailed analysis
    matched_items: List[FeedbackItem]
    common_keywords: List[Tuple[str, int]]  # (keyword, frequency)
    severity_breakdown: Dict[str, int]  # high/medium/low counts
    temporal_pattern: Dict[str, Any]
    
    # Risk assessment
    risk_score: float
    requires_capa: bool
    immediate_action_required: bool

@dataclass
class TemporalTrend:
    """Temporal analysis of feedback patterns"""
    period: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    confidence: float
    weekly_data: Dict[str, int]
    monthly_data: Dict[str, int]
    seasonal_patterns: Dict[str, Any]
    anomalies: List[Dict[str, Any]]

@dataclass
class QualityAssessment:
    """Overall quality assessment from feedback analysis"""
    total_items: int
    quality_score: float  # 0-100
    quality_level: str  # 'Excellent', 'Good', 'Fair', 'Poor', 'Critical'
    
    # Sentiment breakdown
    positive_count: int
    negative_count: int
    neutral_count: int
    sentiment_ratio: float
    
    # Risk indicators
    high_risk_categories: List[str]
    safety_issues_count: int
    efficacy_issues_count: int
    
    # Improvement indicators
    improvement_trend: str
    key_strengths: List[str]
    primary_concerns: List[str]

@dataclass
class CAPARecommendation:
    """Corrective and Preventive Action recommendation"""
    capa_id: str
    priority: str  # 'Critical', 'High', 'Medium', 'Low'
    category: str
    
    # Problem identification
    issue_description: str
    root_cause_analysis: str
    affected_customer_count: int
    
    # Actions
    corrective_action: str
    preventive_action: str
    timeline: str
    responsibility: str
    
    # Success criteria
    success_metrics: List[str]
    verification_method: str
    
    # Compliance
    iso_reference: str
    regulatory_impact: str
    documentation_required: bool

@dataclass
class TextAnalysisResult:
    """Comprehensive text analysis result"""
    # Product information
    asin: str
    product_name: str
    analysis_period: str
    total_feedback_items: int
    
    # Category analysis
    category_results: Dict[str, CategoryAnalysis]
    uncategorized_count: int
    categorization_rate: float
    
    # Quality assessment
    quality_assessment: QualityAssessment
    
    # Temporal analysis
    temporal_trends: TemporalTrend
    
    # Risk assessment
    overall_risk_level: str
    risk_score: float
    risk_factors: List[str]
    
    # CAPA recommendations
    capa_recommendations: List[CAPARecommendation]
    
    # Insights
    key_insights: List[str]
    positive_highlights: List[str]
    improvement_opportunities: List[str]
    
    # Metadata
    analysis_timestamp: str
    confidence_score: float
    data_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class AdvancedTextProcessor:
    """Advanced text processing for medical device feedback"""
    
    def __init__(self):
        self.quality_framework = MEDICAL_DEVICE_QUALITY_FRAMEWORK
        self.positive_indicators = POSITIVE_FEEDBACK_INDICATORS
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Normalize common contractions
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, category_keywords: List[str]) -> List[Tuple[str, int]]:
        """Extract and count keyword matches"""
        text = self.preprocess_text(text)
        matches = []
        
        for keyword in category_keywords:
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text))
            if count > 0:
                matches.append((keyword, count))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def calculate_sentiment_score(self, text: str, rating: Optional[int] = None) -> float:
        """Calculate sentiment score from text and rating"""
        text = self.preprocess_text(text)
        
        # Base score from rating if available
        base_score = 0.5  # Neutral
        if rating is not None:
            base_score = (rating - 1) / 4  # Convert 1-5 to 0-1
        
        # Text-based sentiment adjustments
        positive_count = 0
        negative_count = 0
        
        # Count positive indicators
        for category_indicators in self.positive_indicators.values():
            for indicator in category_indicators:
                if indicator in text:
                    positive_count += 1
        
        # Count negative indicators from quality categories
        for category_data in self.quality_framework.values():
            for keyword in category_data['keywords']:
                if keyword in text:
                    negative_count += 1
        
        # Adjust score based on text sentiment
        text_sentiment = 0.5  # Neutral
        if positive_count > 0 or negative_count > 0:
            total_indicators = positive_count + negative_count
            text_sentiment = positive_count / total_indicators
        
        # Combine rating and text sentiment
        if rating is not None:
            # Weight: 70% rating, 30% text
            final_score = (base_score * 0.7) + (text_sentiment * 0.3)
        else:
            # Pure text sentiment
            final_score = text_sentiment
        
        return max(0.0, min(1.0, final_score))
    
    def detect_urgency_indicators(self, text: str) -> List[str]:
        """Detect urgency indicators in feedback"""
        text = self.preprocess_text(text)
        urgency_indicators = []
        
        urgent_patterns = [
            r'\b(urgent|emergency|immediate|asap|right away|quickly)\b',
            r'\b(dangerous|unsafe|hazardous|risk|injury|hurt)\b',
            r'\b(broken|defective|malfunctioning|not working)\b',
            r'\b(hospital|doctor|medical attention|emergency room)\b'
        ]
        
        for pattern in urgent_patterns:
            matches = re.findall(pattern, text)
            urgency_indicators.extend(matches)
        
        return list(set(urgency_indicators))
    
    def extract_numerical_mentions(self, text: str) -> Dict[str, List[float]]:
        """Extract numerical mentions (dates, measurements, etc.)"""
        text = self.preprocess_text(text)
        numerical_data = defaultdict(list)
        
        # Time periods
        time_patterns = [
            (r'(\d+)\s*(?:days?|weeks?|months?|years?)', 'time_periods'),
            (r'(\d+)\s*(?:hours?|minutes?)', 'usage_time'),
            (r'(\d+)\s*(?:inches?|feet?|cm|mm)', 'measurements'),
            (r'(\d+)\s*(?:lbs?|pounds?|kg|ounces?)', 'weight'),
            (r'\$(\d+(?:\.\d{2})?)', 'price_mentions')
        ]
        
        for pattern, category in time_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    numerical_data[category].append(float(match))
                except ValueError:
                    continue
        
        return dict(numerical_data)

class CategoryAnalyzer:
    """Analyzes feedback by medical device quality categories"""
    
    def __init__(self):
        self.text_processor = AdvancedTextProcessor()
        self.quality_framework = MEDICAL_DEVICE_QUALITY_FRAMEWORK
    
    def analyze_feedback_by_categories(self, feedback_items: List[FeedbackItem]) -> Dict[str, CategoryAnalysis]:
        """Analyze feedback items by quality categories"""
        category_results = {}
        total_items = len(feedback_items)
        
        for category_id, category_info in self.quality_framework.items():
            # Find matching feedback items
            matched_items = []
            all_keywords = []
            
            for item in feedback_items:
                keywords_found = self.text_processor.extract_keywords(item.text, category_info['keywords'])
                
                if keywords_found:
                    # Add category to item's detected categories
                    if category_id not in item.detected_categories:
                        item.detected_categories.append(category_id)
                    
                    matched_items.append(item)
                    all_keywords.extend([kw for kw, count in keywords_found for _ in range(count)])
            
            # Calculate metrics
            count = len(matched_items)
            percentage = (count / total_items * 100) if total_items > 0 else 0
            
            # Analyze severity breakdown
            severity_breakdown = self._analyze_severity_breakdown(matched_items)
            
            # Calculate risk score
            risk_score = self._calculate_category_risk_score(
                count, category_info['severity'], severity_breakdown, total_items
            )
            
            # Temporal pattern analysis
            temporal_pattern = self._analyze_temporal_pattern(matched_items)
            
            # Common keywords
            common_keywords = Counter(all_keywords).most_common(10)
            
            # CAPA requirements
            requires_capa = self._determine_capa_requirement(
                count, category_info['severity'], risk_score
            )
            
            category_results[category_id] = CategoryAnalysis(
                category_id=category_id,
                name=category_info['name'],
                count=count,
                percentage=round(percentage, 1),
                severity=category_info['severity'],
                iso_reference=category_info['iso_reference'],
                matched_items=matched_items,
                common_keywords=common_keywords,
                severity_breakdown=severity_breakdown,
                temporal_pattern=temporal_pattern,
                risk_score=risk_score,
                requires_capa=requires_capa,
                immediate_action_required=category_info.get('immediate_action_required', False) and count > 0
            )
        
        return category_results
    
    def _analyze_severity_breakdown(self, items: List[FeedbackItem]) -> Dict[str, int]:
        """Analyze severity breakdown of feedback items"""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for item in items:
            # Determine severity based on rating, urgency indicators, and keywords
            urgency_indicators = self.text_processor.detect_urgency_indicators(item.text)
            
            if item.rating is not None and item.rating <= 2:
                severity = 'high'
            elif urgency_indicators:
                severity = 'high'
            elif item.rating is not None and item.rating == 3:
                severity = 'medium'
            elif len(item.detected_categories) >= 2:  # Multiple quality issues
                severity = 'medium'
            else:
                severity = 'low'
            
            severity_counts[severity] += 1
            item.severity_level = severity
        
        return severity_counts
    
    def _calculate_category_risk_score(self, count: int, base_severity: str, 
                                     severity_breakdown: Dict[str, int], total_items: int) -> float:
        """Calculate risk score for a category"""
        if count == 0:
            return 0.0
        
        # Base score by severity
        base_scores = {'critical': 10, 'high': 7, 'medium': 4, 'low': 2}
        base_score = base_scores.get(base_severity, 4)
        
        # Frequency factor
        frequency_factor = min(count / max(total_items, 1) * 10, 5)
        
        # Severity distribution factor
        severity_factor = (
            severity_breakdown.get('high', 0) * 3 +
            severity_breakdown.get('medium', 0) * 2 +
            severity_breakdown.get('low', 0) * 1
        ) / max(count, 1)
        
        risk_score = base_score + frequency_factor + severity_factor
        return min(risk_score, 20.0)  # Cap at 20
    
    def _analyze_temporal_pattern(self, items: List[FeedbackItem]) -> Dict[str, Any]:
        """Analyze temporal patterns in feedback"""
        if not items:
            return {'pattern': 'no_data', 'trend': 'stable'}
        
        # Group by date
        date_counts = defaultdict(int)
        for item in items:
            try:
                item_date = datetime.strptime(item.date, '%Y-%m-%d').date()
                date_counts[item_date] += 1
            except (ValueError, TypeError):
                continue
        
        if len(date_counts) < 2:
            return {'pattern': 'insufficient_data', 'trend': 'stable'}
        
        # Analyze trend
        dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]
        
        # Simple trend analysis
        if len(counts) >= 3:
            recent_avg = statistics.mean(counts[-3:])
            earlier_avg = statistics.mean(counts[:-3]) if len(counts) > 3 else counts[0]
            
            if recent_avg > earlier_avg * 1.3:
                trend = 'increasing'
            elif recent_avg < earlier_avg * 0.7:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'pattern': 'analyzed',
            'trend': trend,
            'date_range': f"{min(dates)} to {max(dates)}",
            'peak_date': max(date_counts.keys(), key=lambda d: date_counts[d]),
            'total_days': (max(dates) - min(dates)).days + 1
        }
    
    def _determine_capa_requirement(self, count: int, severity: str, risk_score: float) -> bool:
        """Determine if CAPA is required for this category"""
        # CAPA thresholds
        if severity == 'critical' and count > 0:
            return True
        elif severity == 'high' and count > 1:
            return True
        elif severity == 'medium' and count > 3:
            return True
        elif risk_score > 10:
            return True
        
        return False

class TemporalAnalyzer:
    """Analyzes temporal trends and patterns in feedback"""
    
    def analyze_temporal_trends(self, feedback_items: List[FeedbackItem], 
                              date_filter: Optional[Dict[str, Any]] = None) -> TemporalTrend:
        """Analyze temporal trends in feedback data"""
        
        if not feedback_items:
            return self._create_empty_trend()
        
        # Parse dates and group by time periods
        daily_counts = defaultdict(int)
        weekly_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        
        valid_dates = []
        
        for item in feedback_items:
            try:
                item_date = datetime.strptime(item.date, '%Y-%m-%d').date()
                valid_dates.append(item_date)
                
                # Daily counts
                daily_counts[item_date] += 1
                
                # Weekly counts (Monday as start of week)
                week_start = item_date - timedelta(days=item_date.weekday())
                week_key = week_start.strftime('%Y-W%U')
                weekly_counts[week_key] += 1
                
                # Monthly counts
                month_key = item_date.strftime('%Y-%m')
                monthly_counts[month_key] += 1
                
            except (ValueError, TypeError):
                continue
        
        if not valid_dates:
            return self._create_empty_trend()
        
        # Determine analysis period
        min_date = min(valid_dates)
        max_date = max(valid_dates)
        period = f"{min_date} to {max_date}"
        
        # Analyze trend direction
        trend_direction, confidence = self._analyze_trend_direction(daily_counts, valid_dates)
        
        # Detect seasonal patterns
        seasonal_patterns = self._detect_seasonal_patterns(monthly_counts)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(daily_counts)
        
        return TemporalTrend(
            period=period,
            trend_direction=trend_direction,
            confidence=confidence,
            weekly_data=dict(weekly_counts),
            monthly_data=dict(monthly_counts),
            seasonal_patterns=seasonal_patterns,
            anomalies=anomalies
        )
    
    def _analyze_trend_direction(self, daily_counts: Dict[date, int], 
                               valid_dates: List[date]) -> Tuple[str, float]:
        """Analyze overall trend direction"""
        if len(valid_dates) < 7:
            return 'insufficient_data', 0.0
        
        # Sort dates and get counts
        sorted_dates = sorted(daily_counts.keys())
        counts = [daily_counts[d] for d in sorted_dates]
        
        # Split into two halves for comparison
        mid_point = len(counts) // 2
        first_half_avg = statistics.mean(counts[:mid_point]) if mid_point > 0 else 0
        second_half_avg = statistics.mean(counts[mid_point:]) if len(counts) > mid_point else 0
        
        if first_half_avg == 0 and second_half_avg == 0:
            return 'stable', 0.5
        
        # Calculate percentage change
        if first_half_avg > 0:
            change_ratio = (second_half_avg - first_half_avg) / first_half_avg
        else:
            change_ratio = 1.0 if second_half_avg > 0 else 0.0
        
        # Determine trend
        if change_ratio > 0.2:
            trend = 'increasing'
            confidence = min(abs(change_ratio), 1.0)
        elif change_ratio < -0.2:
            trend = 'decreasing'
            confidence = min(abs(change_ratio), 1.0)
        else:
            trend = 'stable'
            confidence = 1.0 - abs(change_ratio)
        
        return trend, round(confidence, 2)
    
    def _detect_seasonal_patterns(self, monthly_counts: Dict[str, int]) -> Dict[str, Any]:
        """Detect seasonal patterns in monthly data"""
        if len(monthly_counts) < 6:
            return {'pattern_detected': False, 'confidence': 0.0}
        
        # Group by month (ignoring year)
        month_totals = defaultdict(int)
        for month_key, count in monthly_counts.items():
            try:
                month_num = int(month_key.split('-')[1])
                month_totals[month_num] += count
            except (ValueError, IndexError):
                continue
        
        if len(month_totals) < 3:
            return {'pattern_detected': False, 'confidence': 0.0}
        
        # Simple seasonality detection
        values = list(month_totals.values())
        max_month = max(month_totals.keys(), key=lambda k: month_totals[k])
        min_month = min(month_totals.keys(), key=lambda k: month_totals[k])
        
        variation_coefficient = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else 0
        
        return {
            'pattern_detected': variation_coefficient > 0.3,
            'confidence': min(variation_coefficient, 1.0),
            'peak_month': max_month,
            'low_month': min_month,
            'variation_coefficient': round(variation_coefficient, 3)
        }
    
    def _detect_anomalies(self, daily_counts: Dict[date, int]) -> List[Dict[str, Any]]:
        """Detect anomalous days with unusual feedback volume"""
        if len(daily_counts) < 7:
            return []
        
        counts = list(daily_counts.values())
        mean_count = statistics.mean(counts)
        
        if len(counts) < 2:
            return []
        
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0
        threshold = mean_count + (2 * std_count)  # 2 standard deviations
        
        anomalies = []
        for date_key, count in daily_counts.items():
            if count > threshold and count > mean_count * 2:
                anomalies.append({
                    'date': date_key.strftime('%Y-%m-%d'),
                    'count': count,
                    'expected_count': round(mean_count, 1),
                    'severity': 'high' if count > mean_count * 3 else 'medium'
                })
        
        return sorted(anomalies, key=lambda x: x['count'], reverse=True)[:5]  # Top 5
    
    def _create_empty_trend(self) -> TemporalTrend:
        """Create empty trend for no data scenarios"""
        return TemporalTrend(
            period='No data period',
            trend_direction='no_data',
            confidence=0.0,
            weekly_data={},
            monthly_data={},
            seasonal_patterns={'pattern_detected': False},
            anomalies=[]
        )

class QualityAssessor:
    """Assesses overall quality from feedback analysis"""
    
    def assess_quality(self, feedback_items: List[FeedbackItem], 
                      category_results: Dict[str, CategoryAnalysis]) -> QualityAssessment:
        """Assess overall quality from feedback analysis"""
        
        total_items = len(feedback_items)
        if total_items == 0:
            return self._create_empty_assessment()
        
        # Calculate sentiment breakdown
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        total_sentiment = 0
        
        for item in feedback_items:
            sentiment = item.sentiment_score
            total_sentiment += sentiment
            
            if sentiment >= 0.6:
                positive_count += 1
            elif sentiment <= 0.4:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall quality score
        avg_sentiment = total_sentiment / total_items
        
        # Factor in category analysis
        high_risk_categories = [
            cat_id for cat_id, analysis in category_results.items()
            if analysis.risk_score > 10 and analysis.count > 0
        ]
        
        safety_issues = category_results.get('safety_concerns', CategoryAnalysis(
            '', '', 0, 0, '', '', [], [], {}, {}, 0, False, False
        )).count
        
        efficacy_issues = category_results.get('efficacy_performance', CategoryAnalysis(
            '', '', 0, 0, '', '', [], [], {}, {}, 0, False, False
        )).count
        
        # Quality score calculation (0-100)
        base_score = avg_sentiment * 100
        
        # Penalties for high-risk issues
        safety_penalty = min(safety_issues * 15, 30)  # Max 30 point penalty
        efficacy_penalty = min(efficacy_issues * 10, 20)  # Max 20 point penalty
        high_risk_penalty = len(high_risk_categories) * 5  # 5 points per high-risk category
        
        quality_score = max(0, base_score - safety_penalty - efficacy_penalty - high_risk_penalty)
        
        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)
        
        # Calculate sentiment ratio
        sentiment_ratio = (positive_count - negative_count) / total_items if total_items > 0 else 0
        
        # Identify improvement trends and key insights
        improvement_trend = self._analyze_improvement_trend(feedback_items)
        key_strengths = self._identify_key_strengths(feedback_items)
        primary_concerns = self._identify_primary_concerns(category_results)
        
        return QualityAssessment(
            total_items=total_items,
            quality_score=round(quality_score, 1),
            quality_level=quality_level,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            sentiment_ratio=round(sentiment_ratio, 3),
            high_risk_categories=[cat_id for cat_id in high_risk_categories],
            safety_issues_count=safety_issues,
            efficacy_issues_count=efficacy_issues,
            improvement_trend=improvement_trend,
            key_strengths=key_strengths,
            primary_concerns=primary_concerns
        )
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level from score"""
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 55:
            return 'Fair'
        elif score >= 40:
            return 'Poor'
        else:
            return 'Critical'
    
    def _analyze_improvement_trend(self, feedback_items: List[FeedbackItem]) -> str:
        """Analyze if quality is improving over time"""
        if len(feedback_items) < 10:
            return 'insufficient_data'
        
        # Sort by date
        dated_items = []
        for item in feedback_items:
            try:
                item_date = datetime.strptime(item.date, '%Y-%m-%d').date()
                dated_items.append((item_date, item.sentiment_score))
            except (ValueError, TypeError):
                continue
        
        if len(dated_items) < 10:
            return 'insufficient_data'
        
        dated_items.sort(key=lambda x: x[0])
        
        # Compare first and last thirds
        third = len(dated_items) // 3
        early_sentiment = statistics.mean([score for _, score in dated_items[:third]])
        late_sentiment = statistics.mean([score for _, score in dated_items[-third:]])
        
        change = late_sentiment - early_sentiment
        
        if change > 0.1:
            return 'improving'
        elif change < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_key_strengths(self, feedback_items: List[FeedbackItem]) -> List[str]:
        """Identify key product strengths from positive feedback"""
        strengths = []
        
        # Count positive indicators
        positive_mentions = defaultdict(int)
        
        for item in feedback_items:
            if item.sentiment_score >= 0.6:  # Positive feedback
                text = item.text.lower()
                
                # Check for positive indicators
                for category, indicators in POSITIVE_FEEDBACK_INDICATORS.items():
                    for indicator in indicators:
                        if indicator in text:
                            positive_mentions[indicator] += 1
        
        # Get top strengths
        top_strengths = sorted(positive_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for strength, count in top_strengths:
            if count >= 2:  # At least 2 mentions
                strengths.append(f"{strength.title()} ({count} mentions)")
        
        return strengths
    
    def _identify_primary_concerns(self, category_results: Dict[str, CategoryAnalysis]) -> List[str]:
        """Identify primary quality concerns"""
        concerns = []
        
        # Sort categories by risk score
        sorted_categories = sorted(
            [(cat_id, analysis) for cat_id, analysis in category_results.items()],
            key=lambda x: x[1].risk_score,
            reverse=True
        )
        
        for cat_id, analysis in sorted_categories[:5]:  # Top 5 concerns
            if analysis.count > 0 and analysis.risk_score > 5:
                concern_text = f"{analysis.name}: {analysis.count} issues"
                if analysis.severity in ['critical', 'high']:
                    concern_text += f" ({analysis.severity} severity)"
                concerns.append(concern_text)
        
        return concerns
    
    def _create_empty_assessment(self) -> QualityAssessment:
        """Create empty assessment for no data"""
        return QualityAssessment(
            total_items=0,
            quality_score=0.0,
            quality_level='No Data',
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            sentiment_ratio=0.0,
            high_risk_categories=[],
            safety_issues_count=0,
            efficacy_issues_count=0,
            improvement_trend='no_data',
            key_strengths=[],
            primary_concerns=[]
        )

class CAPAGenerator:
    """Generates CAPA (Corrective and Preventive Action) recommendations"""
    
    def __init__(self):
        self.quality_framework = MEDICAL_DEVICE_QUALITY_FRAMEWORK
    
    def generate_capa_recommendations(self, category_results: Dict[str, CategoryAnalysis],
                                    quality_assessment: QualityAssessment,
                                    product_name: str) -> List[CAPARecommendation]:
        """Generate CAPA recommendations based on analysis results"""
        
        recommendations = []
        capa_counter = 1
        
        # Generate CAPA for high-risk categories
        for cat_id, analysis in category_results.items():
            if analysis.requires_capa:
                capa = self._create_category_capa(
                    f"CAPA-{capa_counter:03d}",
                    cat_id,
                    analysis,
                    product_name
                )
                recommendations.append(capa)
                capa_counter += 1
        
        # Generate overall quality CAPA if needed
        if quality_assessment.quality_score < 60:
            capa = self._create_overall_quality_capa(
                f"CAPA-{capa_counter:03d}",
                quality_assessment,
                product_name
            )
            recommendations.append(capa)
            capa_counter += 1
        
        # Generate safety-specific CAPA if safety issues exist
        if quality_assessment.safety_issues_count > 0:
            capa = self._create_safety_capa(
                f"CAPA-{capa_counter:03d}",
                quality_assessment,
                product_name
            )
            recommendations.append(capa)
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return recommendations
    
    def _create_category_capa(self, capa_id: str, category_id: str, 
                            analysis: CategoryAnalysis, product_name: str) -> CAPARecommendation:
        """Create CAPA for specific category issues"""
        
        category_info = self.quality_framework[category_id]
        
        # Determine priority
        if analysis.immediate_action_required:
            priority = 'Critical'
        elif analysis.severity in ['critical', 'high'] and analysis.count > 2:
            priority = 'High'
        elif analysis.risk_score > 10:
            priority = 'High'
        elif analysis.count > 5:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        # Generate specific actions based on category
        corrective_action, preventive_action = self._get_category_specific_actions(category_id, analysis)
        
        # Timeline based on priority
        timeline_map = {
            'Critical': 'Immediate (24-48 hours)',
            'High': '1-2 weeks',
            'Medium': '2-4 weeks',
            'Low': '4-8 weeks'
        }
        
        return CAPARecommendation(
            capa_id=capa_id,
            priority=priority,
            category=analysis.name,
            issue_description=f"{analysis.count} customer feedback items indicate {analysis.name.lower()} issues affecting product quality",
            root_cause_analysis=f"Analysis of customer feedback reveals recurring {category_id.replace('_', ' ')} concerns",
            affected_customer_count=analysis.count,
            corrective_action=corrective_action,
            preventive_action=preventive_action,
            timeline=timeline_map[priority],
            responsibility=self._get_responsibility_assignment(category_id),
            success_metrics=[
                f"Reduce {analysis.name.lower()} complaints by 50% within next review period",
                f"Achieve customer satisfaction score of 80%+ for {analysis.name.lower()}",
                "Zero critical safety incidents reported"
            ],
            verification_method=f"Monthly review of customer feedback for {analysis.name.lower()} issues",
            iso_reference=category_info['iso_reference'],
            regulatory_impact=category_info.get('regulatory_impact', 'medium'),
            documentation_required=priority in ['Critical', 'High']
        )
    
    def _get_category_specific_actions(self, category_id: str, 
                                     analysis: CategoryAnalysis) -> Tuple[str, str]:
        """Get category-specific corrective and preventive actions"""
        
        action_map = {
            'safety_concerns': (
                'Immediate product safety review and customer notification if required; investigate reported safety incidents',
                'Implement enhanced safety testing protocols and clearer safety warnings in product documentation'
            ),
            'efficacy_performance': (
                'Review product performance specifications and conduct user testing to validate effectiveness claims',
                'Enhance quality control testing and update product documentation with realistic performance expectations'
            ),
            'comfort_usability': (
                'Conduct ergonomic assessment and user experience testing with target demographic',
                'Implement design improvements and provide better user guidance and training materials'
            ),
            'durability_quality': (
                'Review manufacturing processes and incoming material quality; inspect current inventory',
                'Implement additional quality checkpoints in production and supplier quality agreements'
            ),
            'sizing_fit': (
                'Update product listings with more detailed size charts and measurements; review sizing accuracy',
                'Implement size recommendation tools and clearer sizing guidance for customers'
            ),
            'assembly_instructions': (
                'Revise assembly instructions with clearer diagrams and step-by-step photos',
                'Implement user testing of assembly instructions before product launch'
            ),
            'shipping_packaging': (
                'Review packaging adequacy and shipping partner performance; improve protective packaging',
                'Implement packaging testing protocols and shipping quality monitoring'
            ),
            'biocompatibility': (
                'Investigate material composition and conduct biocompatibility testing; notify affected customers',
                'Implement enhanced material screening and biocompatibility testing protocols'
            ),
            'regulatory_compliance': (
                'Review regulatory compliance status and update labeling/claims as needed',
                'Implement regulatory compliance review process for all marketing materials and claims'
            )
        }
        
        return action_map.get(category_id, (
            'Investigate reported issues and implement corrective measures',
            'Implement preventive measures to avoid recurrence of similar issues'
        ))
    
    def _get_responsibility_assignment(self, category_id: str) -> str:
        """Get responsibility assignment based on category"""
        
        responsibility_map = {
            'safety_concerns': 'Quality Manager + Engineering + Regulatory Affairs',
            'efficacy_performance': 'Engineering + Quality Assurance + Clinical Affairs',
            'comfort_usability': 'Design Team + User Experience + Clinical Affairs',
            'durability_quality': 'Manufacturing Manager + Quality Assurance + Supplier Quality',
            'sizing_fit': 'Product Management + Design Team + Marketing',
            'assembly_instructions': 'Technical Writing + Customer Experience + Quality',
            'shipping_packaging': 'Operations + Packaging Engineering + Logistics',
            'biocompatibility': 'Regulatory Affairs + Materials Engineering + Quality',
            'regulatory_compliance': 'Regulatory Affairs + Quality Manager + Legal'
        }
        
        return responsibility_map.get(category_id, 'Quality Manager + Product Management')
    
    def _create_overall_quality_capa(self, capa_id: str, quality_assessment: QualityAssessment,
                                   product_name: str) -> CAPARecommendation:
        """Create CAPA for overall quality improvement"""
        
        return CAPARecommendation(
            capa_id=capa_id,
            priority='High' if quality_assessment.quality_score < 50 else 'Medium',
            category='Overall Product Quality',
            issue_description=f"Overall quality score is {quality_assessment.quality_score}%, below acceptable threshold of 70%",
            root_cause_analysis='Multiple quality categories showing customer dissatisfaction requiring systematic improvement',
            affected_customer_count=quality_assessment.negative_count,
            corrective_action='Comprehensive review of top customer complaints and implementation of rapid resolution plan',
            preventive_action='Implement proactive customer feedback monitoring and quality improvement process',
            timeline='2-4 weeks',
            responsibility='Quality Manager + Product Management + Customer Success',
            success_metrics=[
                'Achieve overall quality score above 75% within 60 days',
                'Reduce negative feedback ratio to below 15%',
                'Implement monthly quality review process'
            ],
            verification_method='Monthly quality score calculation and customer feedback review',
            iso_reference='ISO 13485 Section 8.5 - Improvement',
            regulatory_impact='medium',
            documentation_required=True
        )
    
    def _create_safety_capa(self, capa_id: str, quality_assessment: QualityAssessment,
                          product_name: str) -> CAPARecommendation:
        """Create CAPA specifically for safety issues"""
        
        return CAPARecommendation(
            capa_id=capa_id,
            priority='Critical',
            category='Product Safety',
            issue_description=f"{quality_assessment.safety_issues_count} safety-related concerns identified in customer feedback",
            root_cause_analysis='Customer feedback indicates potential safety hazards requiring immediate investigation',
            affected_customer_count=quality_assessment.safety_issues_count,
            corrective_action='Immediate investigation of all safety concerns; customer notification and product recall assessment',
            preventive_action='Implement enhanced safety risk assessment and monitoring protocols',
            timeline='Immediate (24-48 hours)',
            responsibility='Quality Manager + Engineering + Regulatory Affairs + Legal',
            success_metrics=[
                'Zero additional safety incidents reported',
                'Complete investigation of all reported safety concerns',
                'Implementation of enhanced safety monitoring'
            ],
            verification_method='Daily safety incident monitoring and weekly safety review meetings',
            iso_reference='ISO 13485 Section 8.2.2 - Customer Feedback and ISO 14971 - Risk Management',
            regulatory_impact='high',
            documentation_required=True
        )

class TextAnalysisEngine:
    """
    Main Text Analysis Engine - Core analytical component
    
    This is the primary engine that coordinates all text analysis functionality
    for medical device customer feedback interpretation and quality management.
    """
    
    def __init__(self):
        """Initialize the text analysis engine with all components"""
        self.text_processor = AdvancedTextProcessor()
        self.category_analyzer = CategoryAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.quality_assessor = QualityAssessor()
        self.capa_generator = CAPAGenerator()
        
        logger.info("Text Analysis Engine initialized - Core analytical component ready")
    
    def analyze_customer_feedback(self, feedback_data: List[Dict[str, Any]], 
                                product_info: Dict[str, Any],
                                date_filter: Optional[Dict[str, Any]] = None) -> TextAnalysisResult:
        """
        Main analysis function - comprehensive customer feedback analysis
        
        Args:
            feedback_data: List of customer feedback items (reviews, returns, complaints)
            product_info: Product information (ASIN, name, category)
            date_filter: Optional date range filter for temporal analysis
            
        Returns:
            TextAnalysisResult: Comprehensive analysis with quality insights and CAPA recommendations
        """
        
        try:
            # Convert raw feedback data to structured FeedbackItem objects
            feedback_items = self._prepare_feedback_items(feedback_data, product_info)
            
            if not feedback_items:
                logger.warning("No valid feedback items for analysis")
                return self._create_empty_result(product_info)
            
            # Apply date filtering if specified
            if date_filter:
                feedback_items = self._apply_date_filter(feedback_items, date_filter)
                logger.info(f"Date filter applied: {len(feedback_items)} items remaining")
            
            # Calculate sentiment scores for all items
            self._calculate_sentiment_scores(feedback_items)
            
            # Perform category analysis
            category_results = self.category_analyzer.analyze_feedback_by_categories(feedback_items)
            
            # Calculate categorization statistics
            categorized_items = sum(len(item.detected_categories) for item in feedback_items)
            uncategorized_count = sum(1 for item in feedback_items if not item.detected_categories)
            categorization_rate = (len(feedback_items) - uncategorized_count) / len(feedback_items) * 100 if feedback_items else 0
            
            # Perform temporal analysis
            temporal_trends = self.temporal_analyzer.analyze_temporal_trends(feedback_items, date_filter)
            
            # Assess overall quality
            quality_assessment = self.quality_assessor.assess_quality(feedback_items, category_results)
            
            # Calculate overall risk assessment
            risk_level, risk_score, risk_factors = self._calculate_overall_risk(category_results, quality_assessment)
            
            # Generate CAPA recommendations
            capa_recommendations = self.capa_generator.generate_capa_recommendations(
                category_results, quality_assessment, product_info.get('name', 'Unknown Product')
            )
            
            # Generate insights and recommendations
            key_insights = self._generate_key_insights(category_results, quality_assessment, temporal_trends)
            positive_highlights = self._extract_positive_highlights(feedback_items, quality_assessment)
            improvement_opportunities = self._identify_improvement_opportunities(category_results, quality_assessment)
            
            # Calculate confidence and data quality scores
            confidence_score = self._calculate_confidence_score(feedback_items, category_results)
            data_quality_score = self._calculate_data_quality_score(feedback_items)
            
            # Format analysis period
            analysis_period = self._format_analysis_period(feedback_items, date_filter)
            
            # Create comprehensive result
            result = TextAnalysisResult(
                asin=product_info.get('asin', 'unknown'),
                product_name=product_info.get('name', 'Unknown Product'),
                analysis_period=analysis_period,
                total_feedback_items=len(feedback_items),
                category_results=category_results,
                uncategorized_count=uncategorized_count,
                categorization_rate=round(categorization_rate, 1),
                quality_assessment=quality_assessment,
                temporal_trends=temporal_trends,
                overall_risk_level=risk_level,
                risk_score=risk_score,
                risk_factors=risk_factors,
                capa_recommendations=capa_recommendations,
                key_insights=key_insights,
                positive_highlights=positive_highlights,
                improvement_opportunities=improvement_opportunities,
                analysis_timestamp=datetime.now().isoformat(),
                confidence_score=confidence_score,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"Text analysis completed for {product_info.get('name', 'Unknown')}: "
                       f"{len(feedback_items)} items analyzed, {len(capa_recommendations)} CAPA items generated")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            return self._create_empty_result(product_info, error=str(e))
    
    def _prepare_feedback_items(self, feedback_data: List[Dict[str, Any]], 
                              product_info: Dict[str, Any]) -> List[FeedbackItem]:
        """Convert raw feedback data to structured FeedbackItem objects"""
        
        feedback_items = []
        
        for item_data in feedback_data:
            try:
                # Extract and validate required fields
                text = item_data.get('text', '').strip()
                if not text:
                    continue
                
                # Preprocess text
                processed_text = self.text_processor.preprocess_text(text)
                if not processed_text:
                    continue
                
                # Create FeedbackItem
                feedback_item = FeedbackItem(
                    text=processed_text,
                    date=item_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                    type=item_data.get('type', 'unknown'),
                    rating=item_data.get('rating'),
                    source=item_data.get('source', 'unknown'),
                    asin=product_info.get('asin', ''),
                    product_name=product_info.get('name', 'Unknown Product')
                )
                
                feedback_items.append(feedback_item)
                
            except Exception as e:
                logger.warning(f"Error processing feedback item: {str(e)}")
                continue
        
        logger.info(f"Prepared {len(feedback_items)} feedback items for analysis")
        return feedback_items
    
    def _apply_date_filter(self, feedback_items: List[FeedbackItem], 
                          date_filter: Dict[str, Any]) -> List[FeedbackItem]:
        """Apply date range filtering to feedback items"""
        
        if not date_filter or not date_filter.get('start_date'):
            return feedback_items
        
        start_date = date_filter['start_date']
        end_date = date_filter.get('end_date', datetime.now().date())
        
        filtered_items = []
        
        for item in feedback_items:
            try:
                if isinstance(item.date, str):
                    item_date = datetime.strptime(item.date, '%Y-%m-%d').date()
                else:
                    item_date = item.date
                
                if start_date <= item_date <= end_date:
                    filtered_items.append(item)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse date {item.date}: {str(e)}")
                # Include items with unparseable dates to avoid data loss
                filtered_items.append(item)
        
        return filtered_items
    
    def _calculate_sentiment_scores(self, feedback_items: List[FeedbackItem]):
        """Calculate sentiment scores for all feedback items"""
        
        for item in feedback_items:
            sentiment_score = self.text_processor.calculate_sentiment_score(item.text, item.rating)
            item.sentiment_score = sentiment_score
            
            # Calculate confidence score based on text length and rating availability
            text_length_factor = min(len(item.text) / 100, 1.0)  # Up to 100 chars = full confidence
            rating_factor = 0.3 if item.rating is not None else 0.0
            item.confidence_score = (text_length_factor * 0.7) + rating_factor
    
    def _calculate_overall_risk(self, category_results: Dict[str, CategoryAnalysis],
                              quality_assessment: QualityAssessment) -> Tuple[str, float, List[str]]:
        """Calculate overall risk level and factors"""
        
        risk_score = 0.0
        risk_factors = []
        
        # Safety risk (highest weight)
        safety_analysis = category_results.get('safety_concerns')
        if safety_analysis and safety_analysis.count > 0:
            risk_score += safety_analysis.count * 15  # 15 points per safety issue
            risk_factors.append(f"{safety_analysis.count} safety concerns identified in customer feedback")
        
        # Efficacy risk
        efficacy_analysis = category_results.get('efficacy_performance')
        if efficacy_analysis and efficacy_analysis.count > 2:
            risk_score += efficacy_analysis.count * 5
            risk_factors.append(f"{efficacy_analysis.count} efficacy/performance issues reported")
        
        # Quality score risk
        if quality_assessment.quality_score < 50:
            quality_risk = (50 - quality_assessment.quality_score) / 2
            risk_score += quality_risk
            risk_factors.append(f"Low overall quality score ({quality_assessment.quality_score:.1f}%)")
        
        # High-risk category count
        high_risk_count = len(quality_assessment.high_risk_categories)
        if high_risk_count > 0:
            risk_score += high_risk_count * 3
            risk_factors.append(f"{high_risk_count} high-risk quality categories identified")
        
        # Negative feedback ratio
        if quality_assessment.sentiment_ratio < -0.2:
            risk_score += abs(quality_assessment.sentiment_ratio) * 10
            risk_factors.append(f"High negative feedback ratio ({quality_assessment.sentiment_ratio:.1%})")
        
        # Determine risk level
        if risk_score >= 40:
            risk_level = RiskLevel.CRITICAL.value
        elif risk_score >= 25:
            risk_level = RiskLevel.HIGH.value
        elif risk_score >= 15:
            risk_level = RiskLevel.MEDIUM.value
        elif risk_score >= 5:
            risk_level = RiskLevel.LOW.value
        else:
            risk_level = RiskLevel.MINIMAL.value
        
        return risk_level, round(risk_score, 1), risk_factors
    
    def _generate_key_insights(self, category_results: Dict[str, CategoryAnalysis],
                             quality_assessment: QualityAssessment, 
                             temporal_trends: TemporalTrend) -> List[str]:
        """Generate key insights from analysis"""
        
        insights = []
        
        # Top category insights
        top_categories = sorted(
            [(cat_id, analysis) for cat_id, analysis in category_results.items()],
            key=lambda x: x[1].risk_score,
            reverse=True
        )[:3]
        
        for cat_id, analysis in top_categories:
            if analysis.count > 0:
                insights.append(
                    f"{analysis.name} is the primary concern with {analysis.count} feedback items "
                    f"({analysis.percentage}% of total feedback)"
                )
        
        # Quality insights
        if quality_assessment.quality_score >= 80:
            insights.append(f"Strong overall quality performance (score: {quality_assessment.quality_score}%)")
        elif quality_assessment.quality_score < 60:
            insights.append(f"Quality concerns requiring attention (score: {quality_assessment.quality_score}%)")
        
        # Temporal insights
        if temporal_trends.trend_direction == 'increasing':
            insights.append("Feedback volume is increasing - monitor for emerging quality issues")
        elif temporal_trends.trend_direction == 'decreasing':
            insights.append("Feedback volume is decreasing - potential quality improvements")
        
        # Safety insights
        if quality_assessment.safety_issues_count > 0:
            insights.append(f"CRITICAL: {quality_assessment.safety_issues_count} safety-related issues require immediate attention")
        
        return insights[:5]  # Top 5 insights
    
    def _extract_positive_highlights(self, feedback_items: List[FeedbackItem],
                                   quality_assessment: QualityAssessment) -> List[str]:
        """Extract positive highlights from feedback"""
        
        highlights = []
        
        # Positive feedback ratio
        if quality_assessment.positive_count > 0:
            positive_ratio = quality_assessment.positive_count / quality_assessment.total_items
            if positive_ratio >= 0.6:
                highlights.append(f"{positive_ratio:.1%} of customers provided positive feedback")
        
        # Key strengths from quality assessment
        highlights.extend(quality_assessment.key_strengths[:3])
        
        # High-rating feedback highlights
        high_rating_items = [item for item in feedback_items if item.rating and item.rating >= 5]
        if len(high_rating_items) >= 5:
            highlights.append(f"{len(high_rating_items)} customers gave 5-star ratings")
        
        return highlights[:5]
    
    def _identify_improvement_opportunities(self, category_results: Dict[str, CategoryAnalysis],
                                          quality_assessment: QualityAssessment) -> List[str]:
        """Identify specific improvement opportunities"""
        
        opportunities = []
        
        # Category-specific opportunities
        for cat_id, analysis in category_results.items():
            if analysis.count > 0 and analysis.risk_score > 5:
                opportunity = f"Address {analysis.name.lower()} issues to improve customer satisfaction"
                if analysis.count > 3:
                    opportunity += f" (affects {analysis.count} customers)"
                opportunities.append(opportunity)
        
        # Overall quality opportunities
        if quality_assessment.quality_score < 80:
            opportunities.append("Implement systematic quality improvement program")
        
        # Temporal opportunities
        if quality_assessment.improvement_trend == 'declining':
            opportunities.append("Investigate recent quality decline and implement corrective measures")
        
        return opportunities[:5]
    
    def _calculate_confidence_score(self, feedback_items: List[FeedbackItem],
                                  category_results: Dict[str, CategoryAnalysis]) -> float:
        """Calculate overall analysis confidence score"""
        
        if not feedback_items:
            return 0.0
        
        # Base confidence from data quantity
        quantity_score = min(len(feedback_items) / 50, 1.0)  # 50 items = full quantity confidence
        
        # Text quality score
        avg_text_length = sum(len(item.text) for item in feedback_items) / len(feedback_items)
        text_quality_score = min(avg_text_length / 100, 1.0)  # 100 chars average = full text confidence
        
        # Categorization success rate
        categorized_items = sum(1 for item in feedback_items if item.detected_categories)
        categorization_score = categorized_items / len(feedback_items)
        
        # Date availability
        valid_dates = sum(1 for item in feedback_items if item.date and item.date != datetime.now().strftime('%Y-%m-%d'))
        date_score = valid_dates / len(feedback_items)
        
        # Weighted average
        confidence = (
            quantity_score * 0.3 +
            text_quality_score * 0.3 +
            categorization_score * 0.25 +
            date_score * 0.15
        )
        
        return round(confidence, 3)
    
    def _calculate_data_quality_score(self, feedback_items: List[FeedbackItem]) -> float:
        """Calculate data quality score"""
        
        if not feedback_items:
            return 0.0
        
        # Check various quality indicators
        has_ratings = sum(1 for item in feedback_items if item.rating is not None)
        has_dates = sum(1 for item in feedback_items if item.date)
        has_sources = sum(1 for item in feedback_items if item.source != 'unknown')
        has_adequate_text = sum(1 for item in feedback_items if len(item.text) >= 20)
        
        # Calculate quality score
        rating_score = has_ratings / len(feedback_items)
        date_score = has_dates / len(feedback_items)
        source_score = has_sources / len(feedback_items)
        text_score = has_adequate_text / len(feedback_items)
        
        # Weighted average
        quality_score = (
            rating_score * 0.25 +
            date_score * 0.25 +
            source_score * 0.2 +
            text_score * 0.3
        )
        
        return round(quality_score, 3)
    
    def _format_analysis_period(self, feedback_items: List[FeedbackItem],
                              date_filter: Optional[Dict[str, Any]]) -> str:
        """Format the analysis period string"""
        
        if not feedback_items:
            return 'No data period'
        
        if date_filter:
            start_date = date_filter.get('start_date')
            end_date = date_filter.get('end_date', datetime.now().date())
            return f"{start_date} to {end_date}"
        
        # Extract date range from data
        dates = []
        for item in feedback_items:
            try:
                if isinstance(item.date, str):
                    parsed_date = datetime.strptime(item.date, '%Y-%m-%d').date()
                else:
                    parsed_date = item.date
                dates.append(parsed_date)
            except (ValueError, TypeError):
                continue
        
        if dates:
            return f"{min(dates)} to {max(dates)}"
        else:
            return 'Date range unknown'
    
    def _create_empty_result(self, product_info: Dict[str, Any], error: str = None) -> TextAnalysisResult:
        """Create empty result for error scenarios"""
        
        error_insights = [error] if error else []
        
        return TextAnalysisResult(
            asin=product_info.get('asin', 'unknown'),
            product_name=product_info.get('name', 'Unknown Product'),
            analysis_period='No data period',
            total_feedback_items=0,
            category_results={},
            uncategorized_count=0,
            categorization_rate=0.0,
            quality_assessment=QualityAssessment(
                total_items=0, quality_score=0.0, quality_level='No Data',
                positive_count=0, negative_count=0, neutral_count=0, sentiment_ratio=0.0,
                high_risk_categories=[], safety_issues_count=0, efficacy_issues_count=0,
                improvement_trend='no_data', key_strengths=[], primary_concerns=[]
            ),
            temporal_trends=TemporalTrend(
                period='No data period', trend_direction='no_data', confidence=0.0,
                weekly_data={}, monthly_data={}, seasonal_patterns={}, anomalies=[]
            ),
            overall_risk_level=RiskLevel.MINIMAL.value,
            risk_score=0.0,
            risk_factors=[],
            capa_recommendations=[],
            key_insights=error_insights,
            positive_highlights=[],
            improvement_opportunities=[],
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=0.0,
            data_quality_score=0.0
        )

# Export the main engine and supporting classes
__all__ = [
    'TextAnalysisEngine',
    'TextAnalysisResult', 
    'CategoryAnalysis',
    'QualityAssessment',
    'CAPARecommendation',
    'FeedbackItem',
    'QualityCategory',
    'RiskLevel',
    'MEDICAL_DEVICE_QUALITY_FRAMEWORK'
]
