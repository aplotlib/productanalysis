"""
Amazon Medical Device Text Analysis Optimizer - Main Application

**PRIMARY FOCUS: Customer Feedback Text Analysis & Quality Management**

Designed specifically for listing managers who need:
✓ Deep text analysis of customer comments and reviews
✓ Date range filtering for temporal analysis  
✓ Medical device quality categorization (comfort, assembly, safety, etc.)
✓ CAPA recommendations based on customer feedback patterns
✓ Actionable insights for listing optimization

Architecture: Text Analysis Engine → AI Enhancement → Quality Management Dashboard

Author: Assistant
Version: 3.0 - Text Analysis Focused
Compliance: ISO 13485 Quality Management Awareness
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import traceback
import io
import re
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

# Configure logging for quality management traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('medical_device_analysis.log', mode='a')
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

# Import custom modules
upload_handler_module, upload_available = safe_import('upload_handler')
ai_analysis_module, ai_available = safe_import('enhanced_ai_analysis')
dashboard_module, dashboard_available = safe_import('dashboard')

if upload_available:
    from upload_handler import UploadHandler, UploadError, DataValidationError, MEDICAL_DEVICE_CATEGORIES
else:
    logger.error("Upload handler module not available")
    
if ai_available:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AnalysisResult
else:
    logger.error("AI analysis module not available")

if dashboard_available:
    from dashboard import ProfessionalDashboard
else:
    logger.error("Dashboard module not available")

# Application configuration focused on text analysis
APP_CONFIG = {
    'title': 'Medical Device Customer Feedback Analyzer',
    'version': '3.0',
    'description': 'Text analysis and quality management for medical device customer feedback',
    'support_email': 'support@medical-device-analyzer.com',
    'max_products_analysis': 100,
    'session_timeout_hours': 6,
    'focus': 'Customer Text Analysis & Quality Management',
    'compliance': 'ISO 13485 Aware'
}

# Medical Device Quality Categories (ISO 13485 aligned)
QUALITY_CATEGORIES = {
    'safety_concerns': {
        'name': 'Safety & Risk',
        'keywords': ['unsafe', 'dangerous', 'injury', 'hurt', 'hazard', 'broken', 'sharp', 'cuts', 'unstable', 'tip over', 'fall'],
        'severity': 'high',
        'iso_ref': 'ISO 13485 Section 7.3 - Risk Management'
    },
    'comfort_usability': {
        'name': 'Comfort & Usability',
        'keywords': ['uncomfortable', 'painful', 'hurts', 'sore', 'hard', 'rough', 'difficult', 'confusing', 'awkward'],
        'severity': 'medium',
        'iso_ref': 'ISO 13485 Section 7.3 - User Requirements'
    },
    'assembly_instructions': {
        'name': 'Assembly & Instructions',
        'keywords': ['assembly', 'instructions', 'directions', 'setup', 'confusing', 'unclear', 'missing parts', 'difficult setup'],
        'severity': 'medium',
        'iso_ref': 'ISO 13485 Section 4.2 - Documentation Requirements'
    },
    'durability_quality': {
        'name': 'Durability & Quality',
        'keywords': ['broke', 'broken', 'cheap', 'flimsy', 'fell apart', 'cracked', 'tore', 'bent', 'snapped', 'poor quality'],
        'severity': 'high',
        'iso_ref': 'ISO 13485 Section 7.5 - Production Controls'
    },
    'sizing_fit': {
        'name': 'Sizing & Fit',
        'keywords': ['too small', 'too big', 'wrong size', 'doesnt fit', 'tight', 'loose', 'sizing chart', 'measurements'],
        'severity': 'medium',
        'iso_ref': 'ISO 13485 Section 7.3 - Design Specifications'
    },
    'efficacy_performance': {
        'name': 'Efficacy & Performance',
        'keywords': ['doesnt work', 'ineffective', 'no relief', 'no help', 'not working', 'useless', 'no improvement'],
        'severity': 'high',
        'iso_ref': 'ISO 13485 Section 7.3 - Performance Requirements'
    },
    'shipping_packaging': {
        'name': 'Shipping & Packaging',
        'keywords': ['damaged shipping', 'poor packaging', 'arrived broken', 'shipping damage', 'packaging issues'],
        'severity': 'low',
        'iso_ref': 'ISO 13485 Section 7.5 - Packaging Requirements'
    }
}

# Date range options for temporal analysis
DATE_FILTER_OPTIONS = {
    'last_7_days': {'days': 7, 'label': 'Last 7 Days'},
    'last_30_days': {'days': 30, 'label': 'Last 30 Days'},
    'last_90_days': {'days': 90, 'label': 'Last 90 Days'},
    'last_180_days': {'days': 180, 'label': 'Last 6 Months'},
    'last_365_days': {'days': 365, 'label': 'Last 12 Months'},
    'custom': {'days': None, 'label': 'Custom Date Range'}
}

# Example data focused on customer feedback text analysis
EXAMPLE_FEEDBACK_DATA = {
    'products': [
        {
            'asin': 'B0DT7NW5VY',
            'name': 'Vive Tri-Rollator with Seat and Storage',
            'category': 'Mobility Aids',
            'sku': 'VH-TRI-001',
            'current_return_rate': 4.9,
            'total_feedback_items': 156,
            'analysis_period': '2024-01-01 to 2024-11-20'
        },
        {
            'asin': 'B0DT8XYZ123', 
            'name': 'Premium Shower Chair with Back Support',
            'category': 'Bathroom Safety',
            'sku': 'VH-SHW-234',
            'current_return_rate': 4.0,
            'total_feedback_items': 98,
            'analysis_period': '2024-01-01 to 2024-11-20'
        }
    ],
    'customer_feedback': {
        'B0DT7NW5VY': [
            {
                'date': '2024-11-15',
                'type': 'review',
                'rating': 2,
                'text': 'The wheels started squeaking loudly after just 2 weeks of use. Very annoying and seems cheaply made.',
                'category_flags': ['durability_quality']
            },
            {
                'date': '2024-11-12',
                'type': 'return_reason',
                'text': 'Too heavy for elderly user to maneuver easily. Difficult to lift over thresholds.',
                'category_flags': ['comfort_usability', 'sizing_fit']
            },
            {
                'date': '2024-11-10',
                'type': 'review',
                'rating': 5,
                'text': 'This rollator is amazing! Very stable and the seat is so comfortable. Great for my daily walks.',
                'category_flags': ['positive_feedback']
            },
            {
                'date': '2024-11-08',
                'type': 'return_reason',
                'text': 'Assembly instructions were confusing and missing parts diagram. Could not complete setup.',
                'category_flags': ['assembly_instructions']
            }
        ],
        'B0DT8XYZ123': [
            {
                'date': '2024-11-14',
                'type': 'review',
                'rating': 4,
                'text': 'Good chair but the legs could be more stable on wet surfaces. Overall satisfied with purchase.',
                'category_flags': ['safety_concerns', 'comfort_usability']
            },
            {
                'date': '2024-11-11',
                'type': 'return_reason',
                'text': 'Chair legs not stable enough, felt unsafe in shower. Safety concern for elderly parent.',
                'category_flags': ['safety_concerns']
            }
        ]
    }
}

@dataclass
class TextAnalysisResult:
    """Structured container for text analysis results"""
    asin: str
    product_name: str
    analysis_period: str
    total_feedback_items: int
    
    # Category breakdown
    category_analysis: Dict[str, Any]
    
    # Temporal trends
    trend_analysis: Dict[str, Any]
    
    # Quality insights
    quality_assessment: Dict[str, Any]
    
    # CAPA recommendations
    capa_recommendations: List[Dict[str, Any]]
    
    # Risk assessment
    risk_level: str
    risk_factors: List[str]
    
    # Success metrics
    positive_indicators: List[str]
    improvement_opportunities: List[str]
    
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class TextAnalysisEngine:
    """
    Core Text Analysis Engine - Primary component for customer feedback interpretation
    
    This is the main analytical engine that processes customer comments, reviews, and
    return reasons to provide quality management insights for medical device listings.
    """
    
    def __init__(self):
        self.quality_categories = QUALITY_CATEGORIES
        logger.info("Text Analysis Engine initialized - Core component ready")
    
    def analyze_customer_feedback(self, feedback_data: List[Dict[str, Any]], 
                                date_filter: Optional[Dict[str, Any]] = None) -> TextAnalysisResult:
        """
        Main text analysis function - analyzes customer feedback for quality insights
        
        Args:
            feedback_data: List of customer feedback items (reviews, returns, etc.)
            date_filter: Optional date range filter for temporal analysis
            
        Returns:
            TextAnalysisResult: Comprehensive analysis with quality insights
        """
        
        if not feedback_data:
            logger.warning("No feedback data provided for analysis")
            return self._create_empty_result()
        
        # Filter data by date if specified
        filtered_data = self._apply_date_filter(feedback_data, date_filter) if date_filter else feedback_data
        
        logger.info(f"Analyzing {len(filtered_data)} feedback items (filtered from {len(feedback_data)} total)")
        
        # Perform comprehensive text analysis
        category_analysis = self._analyze_by_quality_categories(filtered_data)
        trend_analysis = self._analyze_temporal_trends(filtered_data)
        quality_assessment = self._assess_quality_indicators(filtered_data, category_analysis)
        capa_recommendations = self._generate_capa_recommendations(category_analysis, quality_assessment)
        risk_assessment = self._assess_risk_level(category_analysis, quality_assessment)
        improvement_analysis = self._identify_improvement_opportunities(category_analysis, filtered_data)
        
        # Extract product info from first item
        product_info = self._extract_product_info(feedback_data)
        
        return TextAnalysisResult(
            asin=product_info.get('asin', 'unknown'),
            product_name=product_info.get('name', 'Unknown Product'),
            analysis_period=self._format_analysis_period(filtered_data, date_filter),
            total_feedback_items=len(filtered_data),
            category_analysis=category_analysis,
            trend_analysis=trend_analysis,
            quality_assessment=quality_assessment,
            capa_recommendations=capa_recommendations,
            risk_level=risk_assessment['level'],
            risk_factors=risk_assessment['factors'],
            positive_indicators=improvement_analysis['positive'],
            improvement_opportunities=improvement_analysis['opportunities']
        )
    
    def _apply_date_filter(self, feedback_data: List[Dict[str, Any]], 
                          date_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply date range filtering to feedback data"""
        
        if not date_filter or not date_filter.get('start_date'):
            return feedback_data
        
        start_date = date_filter['start_date']
        end_date = date_filter.get('end_date', datetime.now().date())
        
        filtered_data = []
        for item in feedback_data:
            item_date_str = item.get('date', '')
            if item_date_str:
                try:
                    # Handle various date formats
                    if isinstance(item_date_str, str):
                        item_date = datetime.strptime(item_date_str, '%Y-%m-%d').date()
                    else:
                        item_date = item_date_str
                    
                    if start_date <= item_date <= end_date:
                        filtered_data.append(item)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse date {item_date_str}: {str(e)}")
                    # Include items with unparseable dates to avoid data loss
                    filtered_data.append(item)
        
        logger.info(f"Date filter applied: {len(filtered_data)} items remain from {len(feedback_data)} total")
        return filtered_data
    
    def _analyze_by_quality_categories(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback by medical device quality categories"""
        
        category_results = {}
        total_items = len(feedback_data)
        
        for category_id, category_info in self.quality_categories.items():
            category_matches = []
            category_keywords = category_info['keywords']
            
            # Find feedback items that match this category
            for item in feedback_data:
                text = item.get('text', '').lower()
                
                # Check for keyword matches
                matches = []
                for keyword in category_keywords:
                    if keyword in text:
                        matches.append(keyword)
                
                if matches:
                    category_matches.append({
                        'item': item,
                        'matched_keywords': matches,
                        'match_strength': len(matches)
                    })
            
            # Calculate category metrics
            category_count = len(category_matches)
            category_percentage = (category_count / total_items * 100) if total_items > 0 else 0
            
            # Analyze severity and patterns
            severity_breakdown = self._analyze_category_severity(category_matches, category_info)
            common_patterns = self._extract_common_patterns(category_matches)
            
            category_results[category_id] = {
                'name': category_info['name'],
                'count': category_count,
                'percentage': round(category_percentage, 1),
                'severity': category_info['severity'],
                'iso_reference': category_info['iso_ref'],
                'matches': category_matches,
                'severity_breakdown': severity_breakdown,
                'common_patterns': common_patterns,
                'requires_capa': category_count > 0 and category_info['severity'] in ['high', 'medium']
            }
        
        # Add overall summary
        total_categorized = sum(result['count'] for result in category_results.values())
        uncategorized_count = total_items - total_categorized
        
        category_results['summary'] = {
            'total_feedback_items': total_items,
            'total_categorized': total_categorized,
            'uncategorized_count': uncategorized_count,
            'categorization_rate': round((total_categorized / total_items * 100), 1) if total_items > 0 else 0
        }
        
        return category_results
    
    def _analyze_temporal_trends(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in customer feedback"""
        
        # Group feedback by date
        date_groups = defaultdict(list)
        for item in feedback_data:
            item_date = item.get('date', '')
            if item_date:
                try:
                    if isinstance(item_date, str):
                        parsed_date = datetime.strptime(item_date, '%Y-%m-%d').date()
                    else:
                        parsed_date = item_date
                    
                    date_groups[parsed_date].append(item)
                except (ValueError, TypeError):
                    date_groups['unknown'].append(item)
        
        # Calculate weekly trends
        weekly_trends = self._calculate_weekly_trends(date_groups)
        
        # Identify trend patterns
        trend_patterns = self._identify_trend_patterns(weekly_trends)
        
        return {
            'daily_breakdown': dict(date_groups),
            'weekly_trends': weekly_trends,
            'trend_patterns': trend_patterns,
            'analysis_span_days': self._calculate_analysis_span(date_groups)
        }
    
    def _assess_quality_indicators(self, feedback_data: List[Dict[str, Any]], 
                                 category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality indicators from feedback analysis"""
        
        total_items = len(feedback_data)
        
        # Calculate positive vs negative feedback ratio
        positive_count = 0
        negative_count = 0
        
        for item in feedback_data:
            rating = item.get('rating')
            text = item.get('text', '').lower()
            
            # Determine sentiment
            if rating:
                if rating >= 4:
                    positive_count += 1
                elif rating <= 2:
                    negative_count += 1
            else:
                # Text-based sentiment for non-rated feedback (returns, etc.)
                positive_indicators = ['great', 'excellent', 'love', 'perfect', 'amazing', 'wonderful']
                negative_indicators = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'broken']
                
                if any(indicator in text for indicator in positive_indicators):
                    positive_count += 1
                elif any(indicator in text for indicator in negative_indicators):
                    negative_count += 1
        
        # Quality score calculation
        if total_items > 0:
            positive_ratio = positive_count / total_items
            negative_ratio = negative_count / total_items
            quality_score = (positive_ratio - negative_ratio + 1) / 2 * 100  # Normalize to 0-100
        else:
            quality_score = 50  # Neutral if no data
        
        # Risk indicators from category analysis
        high_risk_categories = [cat for cat, data in category_analysis.items() 
                              if cat != 'summary' and data.get('severity') == 'high' and data.get('count', 0) > 0]
        
        return {
            'total_feedback_analyzed': total_items,
            'positive_feedback_count': positive_count,
            'negative_feedback_count': negative_count,
            'neutral_feedback_count': total_items - positive_count - negative_count,
            'positive_ratio': round(positive_ratio * 100, 1) if total_items > 0 else 0,
            'negative_ratio': round(negative_ratio * 100, 1) if total_items > 0 else 0,
            'quality_score': round(quality_score, 1),
            'quality_level': self._determine_quality_level(quality_score),
            'high_risk_categories': high_risk_categories,
            'requires_immediate_action': len(high_risk_categories) > 0
        }
    
    def _generate_capa_recommendations(self, category_analysis: Dict[str, Any], 
                                     quality_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate CAPA (Corrective and Preventive Action) recommendations"""
        
        capa_recommendations = []
        
        # Safety-related CAPA (highest priority)
        safety_issues = category_analysis.get('safety_concerns', {})
        if safety_issues.get('count', 0) > 0:
            capa_recommendations.append({
                'priority': 'Critical',
                'category': 'Safety & Risk Management',
                'issue': f"{safety_issues['count']} safety concerns identified in customer feedback",
                'corrective_action': 'Immediate product safety review and customer notification if required',
                'preventive_action': 'Implement enhanced safety testing protocols and clearer safety warnings',
                'timeline': 'Immediate (24-48 hours)',
                'iso_reference': 'ISO 13485 Section 8.2.2 - Customer Feedback',
                'responsibility': 'Quality Manager + Engineering',
                'success_metric': 'Zero safety-related customer complaints in next 30 days'
            })
        
        # Quality/Durability CAPA
        durability_issues = category_analysis.get('durability_quality', {})
        if durability_issues.get('count', 0) > 2:  # Threshold for action
            capa_recommendations.append({
                'priority': 'High',
                'category': 'Product Quality & Durability',
                'issue': f"{durability_issues['count']} durability/quality complaints affecting customer satisfaction",
                'corrective_action': 'Review manufacturing processes and incoming material quality',
                'preventive_action': 'Implement additional quality checkpoints in production',
                'timeline': '1-2 weeks',
                'iso_reference': 'ISO 13485 Section 7.5 - Production Controls',
                'responsibility': 'Manufacturing Manager + Quality Assurance',
                'success_metric': 'Reduce durability complaints by 50% in next 60 days'
            })
        
        # Assembly/Instructions CAPA
        assembly_issues = category_analysis.get('assembly_instructions', {})
        if assembly_issues.get('count', 0) > 1:
            capa_recommendations.append({
                'priority': 'Medium',
                'category': 'Documentation & User Experience',
                'issue': f"{assembly_issues['count']} customer complaints about assembly difficulty or unclear instructions",
                'corrective_action': 'Revise assembly instructions with clearer diagrams and step-by-step photos',
                'preventive_action': 'User testing of assembly instructions before product launch',
                'timeline': '2-3 weeks',
                'iso_reference': 'ISO 13485 Section 4.2 - Documentation Requirements',
                'responsibility': 'Technical Writing + Customer Experience',
                'success_metric': 'Improve assembly instruction rating to 4.0+ stars'
            })
        
        # Sizing/Fit CAPA
        sizing_issues = category_analysis.get('sizing_fit', {})
        if sizing_issues.get('count', 0) > 1:
            capa_recommendations.append({
                'priority': 'Medium',
                'category': 'Product Specifications & Listing Accuracy',
                'issue': f"{sizing_issues['count']} customer complaints about sizing/fit expectations",
                'corrective_action': 'Update product listings with more detailed size charts and measurements',
                'preventive_action': 'Add size recommendation wizard and clearer size guidance',
                'timeline': '1 week',
                'iso_reference': 'ISO 13485 Section 7.3 - Design Specifications',
                'responsibility': 'Product Management + Marketing',
                'success_metric': 'Reduce size-related returns by 30% in next 45 days'
            })
        
        # Overall quality score CAPA
        if quality_assessment.get('quality_score', 50) < 60:
            capa_recommendations.append({
                'priority': 'High',
                'category': 'Overall Customer Satisfaction',
                'issue': f"Overall quality score is {quality_assessment['quality_score']:.1f}%, below acceptable threshold",
                'corrective_action': 'Comprehensive review of top customer complaints and rapid resolution plan',
                'preventive_action': 'Implement proactive customer feedback monitoring and response system',
                'timeline': '2-4 weeks',
                'iso_reference': 'ISO 13485 Section 8.5 - Improvement',
                'responsibility': 'Quality Manager + Customer Success',
                'success_metric': 'Achieve quality score above 75% within 60 days'
            })
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        capa_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return capa_recommendations
    
    def _assess_risk_level(self, category_analysis: Dict[str, Any], 
                          quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level based on feedback analysis"""
        
        risk_factors = []
        risk_score = 0
        
        # Safety risk assessment (highest weight)
        safety_count = category_analysis.get('safety_concerns', {}).get('count', 0)
        if safety_count > 0:
            risk_score += safety_count * 10  # High weight for safety
            risk_factors.append(f"{safety_count} safety concerns identified")
        
        # Quality risk assessment
        durability_count = category_analysis.get('durability_quality', {}).get('count', 0)
        if durability_count > 2:
            risk_score += durability_count * 3
            risk_factors.append(f"Multiple quality/durability complaints ({durability_count})")
        
        # Overall quality score risk
        quality_score = quality_assessment.get('quality_score', 50)
        if quality_score < 50:
            risk_score += (50 - quality_score) / 2
            risk_factors.append(f"Low overall quality score ({quality_score:.1f}%)")
        
        # Negative feedback ratio risk
        negative_ratio = quality_assessment.get('negative_ratio', 0)
        if negative_ratio > 20:
            risk_score += negative_ratio / 5
            risk_factors.append(f"High negative feedback ratio ({negative_ratio:.1f}%)")
        
        # Determine risk level
        if risk_score >= 30:
            risk_level = 'Critical'
        elif risk_score >= 15:
            risk_level = 'High'
        elif risk_score >= 5:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'level': risk_level,
            'score': round(risk_score, 1),
            'factors': risk_factors,
            'requires_immediate_action': risk_level in ['Critical', 'High']
        }
    
    def _identify_improvement_opportunities(self, category_analysis: Dict[str, Any], 
                                          feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify positive indicators and improvement opportunities"""
        
        positive_indicators = []
        improvement_opportunities = []
        
        # Analyze positive feedback patterns
        positive_feedback = [item for item in feedback_data 
                           if item.get('rating', 0) >= 4 or 'positive_feedback' in item.get('category_flags', [])]
        
        if positive_feedback:
            # Extract common positive themes
            positive_text = ' '.join([item.get('text', '') for item in positive_feedback]).lower()
            positive_keywords = ['great', 'excellent', 'love', 'perfect', 'amazing', 'comfortable', 'stable', 'easy']
            
            for keyword in positive_keywords:
                if keyword in positive_text:
                    count = positive_text.count(keyword)
                    if count > 1:
                        positive_indicators.append(f"Customers appreciate {keyword} aspects ({count} mentions)")
        
        # Identify improvement opportunities from category analysis
        for category_id, category_data in category_analysis.items():
            if category_id == 'summary':
                continue
                
            count = category_data.get('count', 0)
            if count > 0:
                category_name = category_data['name']
                if category_data['severity'] in ['high', 'medium']:
                    improvement_opportunities.append({
                        'category': category_name,
                        'impact': 'High' if category_data['severity'] == 'high' else 'Medium',
                        'description': f"Address {count} customer concerns in {category_name.lower()}",
                        'potential_benefit': f"Could improve customer satisfaction and reduce returns"
                    })
        
        return {
            'positive': positive_indicators[:5],  # Top 5 positive indicators
            'opportunities': improvement_opportunities[:5]  # Top 5 opportunities
        }
    
    def _analyze_category_severity(self, category_matches: List[Dict], category_info: Dict) -> Dict[str, Any]:
        """Analyze severity breakdown within a category"""
        
        if not category_matches:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for match in category_matches:
            item = match['item']
            rating = item.get('rating', 3)
            
            # Determine severity based on rating and match strength
            if rating <= 2 or match['match_strength'] >= 3:
                severity_counts['high'] += 1
            elif rating == 3 or match['match_strength'] == 2:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1
        
        return severity_counts
    
    def _extract_common_patterns(self, category_matches: List[Dict]) -> List[str]:
        """Extract common patterns within a category"""
        
        if not category_matches:
            return []
        
        # Collect all matched keywords
        all_keywords = []
        for match in category_matches:
            all_keywords.extend(match['matched_keywords'])
        
        # Count frequency and return most common
        keyword_counts = Counter(all_keywords)
        common_patterns = [f"{keyword} ({count} mentions)" 
                          for keyword, count in keyword_counts.most_common(3)]
        
        return common_patterns
    
    def _calculate_weekly_trends(self, date_groups: Dict) -> Dict[str, Any]:
        """Calculate weekly trends from daily data"""
        
        weekly_data = defaultdict(int)
        
        for date_key, items in date_groups.items():
            if date_key == 'unknown':
                continue
                
            # Get week number
            week_start = date_key - timedelta(days=date_key.weekday())
            week_key = week_start.strftime('%Y-W%U')
            weekly_data[week_key] += len(items)
        
        return dict(weekly_data)
    
    def _identify_trend_patterns(self, weekly_trends: Dict[str, int]) -> Dict[str, Any]:
        """Identify patterns in weekly trends"""
        
        if len(weekly_trends) < 2:
            return {'pattern': 'insufficient_data', 'description': 'Need more time periods for trend analysis'}
        
        values = list(weekly_trends.values())
        
        # Calculate trend direction
        recent_avg = np.mean(values[-2:]) if len(values) >= 2 else values[-1]
        earlier_avg = np.mean(values[:-2]) if len(values) > 2 else values[0]
        
        if recent_avg > earlier_avg * 1.2:
            pattern = 'increasing'
            description = 'Feedback volume is increasing (potential concern)'
        elif recent_avg < earlier_avg * 0.8:
            pattern = 'decreasing'
            description = 'Feedback volume is decreasing (positive trend)'
        else:
            pattern = 'stable'
            description = 'Feedback volume is relatively stable'
        
        return {
            'pattern': pattern,
            'description': description,
            'recent_average': round(recent_avg, 1),
            'earlier_average': round(earlier_avg, 1)
        }
    
    def _calculate_analysis_span(self, date_groups: Dict) -> int:
        """Calculate the span of analysis in days"""
        
        valid_dates = [date_key for date_key in date_groups.keys() if date_key != 'unknown']
        
        if len(valid_dates) < 2:
            return 1
        
        min_date = min(valid_dates)
        max_date = max(valid_dates)
        
        return (max_date - min_date).days + 1
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level from score"""
        
        if quality_score >= 80:
            return 'Excellent'
        elif quality_score >= 70:
            return 'Good'
        elif quality_score >= 60:
            return 'Fair'
        elif quality_score >= 50:
            return 'Poor'
        else:
            return 'Critical'
    
    def _extract_product_info(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract product information from feedback data"""
        
        if not feedback_data:
            return {}
        
        # Try to get product info from first item
        first_item = feedback_data[0]
        return {
            'asin': first_item.get('asin', 'unknown'),
            'name': first_item.get('product_name', 'Unknown Product')
        }
    
    def _format_analysis_period(self, filtered_data: List[Dict[str, Any]], 
                               date_filter: Optional[Dict[str, Any]]) -> str:
        """Format the analysis period string"""
        
        if not filtered_data:
            return 'No data period'
        
        if date_filter:
            start_date = date_filter.get('start_date')
            end_date = date_filter.get('end_date', datetime.now().date())
            return f"{start_date} to {end_date}"
        
        # Extract date range from data
        dates = []
        for item in filtered_data:
            item_date = item.get('date')
            if item_date:
                try:
                    if isinstance(item_date, str):
                        parsed_date = datetime.strptime(item_date, '%Y-%m-%d').date()
                    else:
                        parsed_date = item_date
                    dates.append(parsed_date)
                except (ValueError, TypeError):
                    continue
        
        if dates:
            return f"{min(dates)} to {max(dates)}"
        else:
            return 'Date range unknown'
    
    def _create_empty_result(self) -> TextAnalysisResult:
        """Create empty result for when no data is available"""
        
        return TextAnalysisResult(
            asin='unknown',
            product_name='No Data Available',
            analysis_period='No data period',
            total_feedback_items=0,
            category_analysis={'summary': {'total_feedback_items': 0, 'total_categorized': 0}},
            trend_analysis={'pattern': 'no_data'},
            quality_assessment={'quality_score': 0, 'quality_level': 'No Data'},
            capa_recommendations=[],
            risk_level='Unknown',
            risk_factors=[],
            positive_indicators=[],
            improvement_opportunities=[]
        )

class SafeDataProcessor:
    """Enhanced data processor focused on text analysis workflow"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.upload_handler = None
        self.text_analysis_engine = TextAnalysisEngine()
        self.ai_analyzer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with error handling"""
        try:
            if upload_available:
                self.upload_handler = UploadHandler()
                logger.info("Upload handler initialized for text analysis workflow")
            
            if ai_available:
                self.ai_analyzer = EnhancedAIAnalyzer()
                logger.info("AI analyzer initialized for enhanced text analysis")
                
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
    
    def process_uploaded_data(self, uploaded_data: Dict[str, Any], 
                            date_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process uploaded data with focus on customer feedback extraction"""
        
        with self.lock:
            try:
                processed_data = {
                    'products': [],
                    'customer_feedback': {},  # Main focus: customer feedback by ASIN
                    'text_analysis_results': {},
                    'processing_summary': {},
                    'date_filter_applied': date_filter
                }
                
                # Process structured data (mainly for product metadata)
                if 'structured_data' in uploaded_data:
                    df = uploaded_data['structured_data']
                    logger.info(f"Processing {len(df)} products for metadata")
                    
                    for _, row in df.iterrows():
                        try:
                            product = {
                                'asin': str(row.get('ASIN', '')).strip(),
                                'name': str(row.get('Product Name', f"Product {row.get('ASIN', 'Unknown')}")),
                                'category': str(row.get('Category', 'Other')),
                                'sku': str(row.get('SKU', '')),
                                'current_return_rate': self._calculate_return_rate(row),
                                'metadata': {
                                    'star_rating': self._safe_convert_numeric(row.get('Star Rating')),
                                    'total_reviews': self._safe_convert_int(row.get('Total Reviews')),
                                    'average_price': self._safe_convert_numeric(row.get('Average Price'))
                                }
                            }
                            
                            if product['asin']:
                                processed_data['products'].append(product)
                                # Initialize feedback collection for this product
                                processed_data['customer_feedback'][product['asin']] = []
                                
                        except Exception as e:
                            logger.error(f"Error processing product row: {str(e)}")
                            continue
                    
                    processed_data['processing_summary']['products_processed'] = len(processed_data['products'])
                
                # Process customer feedback (primary focus)
                feedback_count = 0
                
                # Process manual reviews as feedback
                if 'manual_reviews' in uploaded_data:
                    for asin, reviews in uploaded_data['manual_reviews'].items():
                        if asin not in processed_data['customer_feedback']:
                            processed_data['customer_feedback'][asin] = []
                        
                        for review in reviews:
                            feedback_item = {
                                'type': 'review',
                                'text': review.get('review_text', ''),
                                'rating': review.get('rating'),
                                'date': review.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'source': 'manual_entry',
                                'asin': asin
                            }
                            processed_data['customer_feedback'][asin].append(feedback_item)
                            feedback_count += 1
                
                # Process manual returns as feedback
                if 'manual_returns' in uploaded_data:
                    for asin, returns in uploaded_data['manual_returns'].items():
                        if asin not in processed_data['customer_feedback']:
                            processed_data['customer_feedback'][asin] = []
                        
                        for return_item in returns:
                            feedback_item = {
                                'type': 'return_reason',
                                'text': return_item.get('return_reason', ''),
                                'date': return_item.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'source': 'manual_entry',
                                'asin': asin
                            }
                            processed_data['customer_feedback'][asin].append(feedback_item)
                            feedback_count += 1
                
                # Process extracted documents for feedback
                if 'documents' in uploaded_data:
                    doc_feedback = self._process_document_feedback(uploaded_data['documents'])
                    
                    for asin, feedback_items in doc_feedback.items():
                        if asin not in processed_data['customer_feedback']:
                            processed_data['customer_feedback'][asin] = []
                        
                        processed_data['customer_feedback'][asin].extend(feedback_items)
                        feedback_count += len(feedback_items)
                
                processed_data['processing_summary']['total_feedback_items'] = feedback_count
                
                # Run text analysis on each product's feedback (core functionality)
                for asin, feedback_items in processed_data['customer_feedback'].items():
                    if feedback_items:
                        try:
                            # Get product info for analysis
                            product_info = next((p for p in processed_data['products'] if p['asin'] == asin), 
                                              {'asin': asin, 'name': f'Product {asin}'})
                            
                            # Add product info to each feedback item
                            for item in feedback_items:
                                item['product_name'] = product_info['name']
                                item['asin'] = asin
                            
                            # Run core text analysis
                            text_analysis_result = self.text_analysis_engine.analyze_customer_feedback(
                                feedback_items, date_filter
                            )
                            
                            processed_data['text_analysis_results'][asin] = text_analysis_result
                            
                            logger.info(f"Text analysis completed for {asin}: {len(feedback_items)} items analyzed")
                            
                        except Exception as e:
                            logger.error(f"Text analysis failed for {asin}: {str(e)}")
                            continue
                
                processed_data['processing_summary']['text_analysis_completed'] = len(processed_data['text_analysis_results'])
                
                logger.info(f"Data processing complete: {processed_data['processing_summary']}")
                return processed_data
                
            except Exception as e:
                logger.error(f"Error processing uploaded data: {str(e)}")
                raise Exception(f"Failed to process uploaded data: {str(e)}")
    
    def _calculate_return_rate(self, row: pd.Series) -> float:
        """Calculate return rate from row data"""
        try:
            sales_30d = self._safe_convert_int(row.get('Last 30 Days Sales', 0))
            returns_30d = self._safe_convert_int(row.get('Last 30 Days Returns', 0))
            
            if sales_30d > 0:
                return round((returns_30d / sales_30d) * 100, 2)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _safe_convert_numeric(self, value, default=None):
        """Safely convert values to numeric"""
        if pd.isna(value) or value == '' or value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_convert_int(self, value, default=0):
        """Safely convert values to integer"""
        numeric_val = self._safe_convert_numeric(value, default)
        try:
            return int(numeric_val) if numeric_val is not None else default
        except (ValueError, TypeError):
            return default
    
    def _process_document_feedback(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Process extracted feedback from documents"""
        
        feedback_by_asin = defaultdict(list)
        
        for doc in documents:
            if not doc.get('success') or not doc.get('structured_data'):
                continue
            
            structured_data = doc['structured_data']
            content_type = doc.get('content_type', '')
            
            # Extract ASIN
            asin = doc.get('asin') or structured_data.get('primary_asin')
            if not asin and 'detected_asins' in structured_data:
                asin = structured_data['detected_asins'][0]
            
            if not asin:
                logger.warning(f"No ASIN found for document {doc.get('filename', 'unknown')}")
                continue
            
            # Process reviews
            if content_type == 'Product Reviews' and 'reviews' in structured_data:
                for review in structured_data['reviews']:
                    feedback_item = {
                        'type': 'review',
                        'text': review.get('review_text', ''),
                        'rating': review.get('rating'),
                        'date': datetime.now().strftime('%Y-%m-%d'),  # Would extract from doc if available
                        'source': f"document_{doc.get('filename', 'unknown')}",
                        'asin': asin
                    }
                    feedback_by_asin[asin].append(feedback_item)
            
            # Process returns
            elif content_type == 'Return Reports' and 'returns' in structured_data:
                for return_item in structured_data['returns']:
                    feedback_item = {
                        'type': 'return_reason',
                        'text': return_item.get('return_reason', ''),
                        'date': datetime.now().strftime('%Y-%m-%d'),  # Would extract from doc if available
                        'source': f"document_{doc.get('filename', 'unknown')}",
                        'asin': asin
                    }
                    feedback_by_asin[asin].append(feedback_item)
        
        return dict(feedback_by_asin)
    
    def run_ai_enhancement(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI analysis to enhance text analysis results"""
        
        if not self.ai_analyzer:
            logger.warning("AI analyzer not available for enhancement")
            return {}
        
        try:
            ai_results = {}
            text_analysis_results = processed_data.get('text_analysis_results', {})
            customer_feedback = processed_data.get('customer_feedback', {})
            products = processed_data.get('products', [])
            
            # Check API status
            api_status = self.ai_analyzer.get_api_status()
            if not api_status.get('available', False):
                logger.warning(f"AI API not available: {api_status.get('error', 'Unknown error')}")
                return {}
            
            logger.info(f"Running AI enhancement for {len(text_analysis_results)} products")
            
            for asin, text_result in text_analysis_results.items():
                try:
                    # Get product info
                    product = next((p for p in products if p['asin'] == asin), 
                                 {'asin': asin, 'name': f'Product {asin}', 'category': 'Other'})
                    
                    # Get feedback items
                    feedback_items = customer_feedback.get(asin, [])
                    
                    if not feedback_items:
                        logger.debug(f"No feedback items for AI enhancement: {asin}")
                        continue
                    
                    # Convert feedback to format expected by AI analyzer
                    reviews = [item for item in feedback_items if item['type'] == 'review']
                    returns = [item for item in feedback_items if item['type'] == 'return_reason']
                    
                    # Run comprehensive AI analysis
                    analysis_results = self.ai_analyzer.analyze_product_comprehensive(
                        product, reviews, returns
                    )
                    
                    ai_results[asin] = analysis_results
                    logger.debug(f"AI enhancement completed for {asin}")
                    
                except Exception as e:
                    logger.error(f"AI enhancement failed for {asin}: {str(e)}")
                    continue
            
            logger.info(f"AI enhancement completed for {len(ai_results)} products")
            return ai_results
            
        except Exception as e:
            logger.error(f"Error running AI enhancement: {str(e)}")
            return {}

class SessionManager:
    """Enhanced session management for text analysis workflow"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state for text analysis workflow"""
        default_state = {
            # Core data storage
            'uploaded_data': {},
            'processed_data': {},
            'text_analysis_results': {},
            'ai_enhancement_results': {},
            
            # UI state
            'current_tab': 0,
            'selected_product': None,
            'show_example_data': False,
            
            # Date filtering state
            'date_filter_enabled': False,
            'date_filter_option': 'last_30_days',
            'custom_start_date': None,
            'custom_end_date': None,
            'active_date_filter': None,
            
            # Processing state
            'data_processed': False,
            'text_analysis_complete': False,
            'ai_enhancement_complete': False,
            'processing_locked': False,
            
            # Module status
            'module_status': MODULES_LOADED.copy(),
            'api_status': {'available': False, 'error': 'Not tested'},
            
            # Timestamps
            'session_start': datetime.now(),
            'last_activity': datetime.now(),
            
            # Settings
            'auto_run_text_analysis': True,
            'auto_run_ai_enhancement': False,
            'show_debug_info': False,
            'focus_mode': 'text_analysis',  # Core focus
            
            # Quality management
            'capa_tracking_enabled': True,
            'risk_assessment_enabled': True,
            'iso_compliance_mode': True,
            
            # Error tracking
            'error_count': 0,
            'last_error': None
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        logger.info("Session state initialized for text analysis workflow")
    
    @staticmethod
    def update_activity():
        """Update last activity timestamp"""
        st.session_state.last_activity = datetime.now()
    
    @staticmethod
    def get_active_date_filter() -> Optional[Dict[str, Any]]:
        """Get active date filter configuration"""
        
        if not st.session_state.get('date_filter_enabled', False):
            return None
        
        filter_option = st.session_state.get('date_filter_option', 'last_30_days')
        
        if filter_option == 'custom':
            start_date = st.session_state.get('custom_start_date')
            end_date = st.session_state.get('custom_end_date')
            
            if start_date and end_date:
                return {
                    'start_date': start_date,
                    'end_date': end_date,
                    'label': f"Custom: {start_date} to {end_date}"
                }
            else:
                return None
        else:
            # Predefined date range
            if filter_option in DATE_FILTER_OPTIONS:
                days = DATE_FILTER_OPTIONS[filter_option]['days']
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days)
                
                return {
                    'start_date': start_date,
                    'end_date': end_date,
                    'label': DATE_FILTER_OPTIONS[filter_option]['label']
                }
        
        return None
    
    @staticmethod
    def load_example_data():
        """Load example data focused on customer feedback analysis"""
        try:
            logger.info("Loading example customer feedback data")
            
            # Convert example data to expected format
            st.session_state.uploaded_data = {
                'structured_data': pd.DataFrame(EXAMPLE_FEEDBACK_DATA['products']),
                'manual_reviews': {},
                'manual_returns': {}
            }
            
            # Convert customer feedback to manual entry format
            for asin, feedback_items in EXAMPLE_FEEDBACK_DATA['customer_feedback'].items():
                reviews = []
                returns = []
                
                for item in feedback_items:
                    if item['type'] == 'review':
                        reviews.append({
                            'rating': item.get('rating'),
                            'review_text': item['text'],
                            'date': item['date'],
                            'asin': asin
                        })
                    elif item['type'] == 'return_reason':
                        returns.append({
                            'return_reason': item['text'],
                            'date': item['date'],
                            'asin': asin
                        })
                
                if reviews:
                    st.session_state.uploaded_data['manual_reviews'][asin] = reviews
                if returns:
                    st.session_state.uploaded_data['manual_returns'][asin] = returns
            
            st.session_state.show_example_data = True
            logger.info("Example customer feedback data loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load example data: {str(e)}")
            st.error(f"Failed to load example data: {str(e)}")

class ApplicationController:
    """Main application controller focused on text analysis workflow"""
    
    def __init__(self):
        self.data_processor = SafeDataProcessor()
        self.dashboard = None
        
        # Initialize dashboard if available
        if dashboard_available:
            try:
                self.dashboard = ProfessionalDashboard()
                logger.info("Dashboard initialized for text analysis workflow")
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
        """Handle different types of data uploads with text analysis focus"""
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
                    if 'structured_data' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['structured_data'] = result['data']
                    else:
                        existing_df = st.session_state.uploaded_data['structured_data']
                        combined_df = pd.concat([existing_df, result['data']], ignore_index=True)
                        st.session_state.uploaded_data['structured_data'] = combined_df
                    
                    st.success(f"✅ Successfully uploaded {filename}")
                    success = True
                else:
                    errors = result.get('errors', ['Unknown error'])
                    st.error(f"❌ Upload failed: {'; '.join([str(e) for e in errors])}")
            
            elif upload_type == 'manual_entry':
                result = self.data_processor.upload_handler.process_manual_entry(data)
                
                if result['success']:
                    manual_data = result['data']
                    
                    # Add to structured data
                    df_row = {
                        'ASIN': manual_data['asin'],
                        'Product Name': manual_data.get('product_name', ''),
                        'Category': manual_data.get('category', ''),
                        'SKU': manual_data.get('sku', ''),
                        'Last 30 Days Sales': manual_data.get('sales_30d', 0),
                        'Last 30 Days Returns': manual_data.get('returns_30d', 0)
                    }
                    
                    if 'structured_data' not in st.session_state.uploaded_data:
                        st.session_state.uploaded_data['structured_data'] = pd.DataFrame([df_row])
                    else:
                        existing_df = st.session_state.uploaded_data['structured_data']
                        # Remove existing entry with same ASIN if present
                        existing_df = existing_df[existing_df['ASIN'] != manual_data['asin']]
                        new_df = pd.concat([existing_df, pd.DataFrame([df_row])], ignore_index=True)
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
    
    def process_data(self) -> bool:
        """Process all uploaded data with text analysis focus"""
        try:
            SessionManager.update_activity()
            
            if not st.session_state.uploaded_data:
                st.warning("No data to process. Please upload data first.")
                return False
            
            # Get active date filter
            date_filter = SessionManager.get_active_date_filter()
            
            # Set processing lock
            st.session_state.processing_locked = True
            
            try:
                with st.spinner("Processing data and analyzing customer feedback..."):
                    processed_data = self.data_processor.process_uploaded_data(
                        st.session_state.uploaded_data, date_filter
                    )
                    st.session_state.processed_data = processed_data
                    st.session_state.text_analysis_results = processed_data.get('text_analysis_results', {})
                    st.session_state.data_processed = True
                    st.session_state.text_analysis_complete = True
                    st.session_state.active_date_filter = date_filter
                
                # Auto-run AI enhancement if enabled
                if st.session_state.auto_run_ai_enhancement and processed_data.get('text_analysis_results'):
                    self.run_ai_enhancement()
                
                summary = processed_data.get('processing_summary', {})
                feedback_count = summary.get('total_feedback_items', 0)
                analysis_count = summary.get('text_analysis_completed', 0)
                
                success_msg = f"✅ Text analysis complete! Processed {feedback_count} feedback items across {analysis_count} products"
                if date_filter:
                    success_msg += f"\n📅 Date filter applied: {date_filter['label']}"
                
                st.success(success_msg)
                return True
                
            finally:
                # Always release lock
                st.session_state.processing_locked = False
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            st.error(f"❌ Data processing failed: {str(e)}")
            st.session_state.processing_locked = False
            return False
    
    def run_ai_enhancement(self) -> bool:
        """Run AI enhancement on text analysis results"""
        try:
            SessionManager.update_activity()
            
            if not st.session_state.api_status.get('available', False):
                st.error("❌ AI enhancement not available. Please check your API configuration.")
                return False
            
            if not st.session_state.text_analysis_complete or not st.session_state.processed_data:
                st.warning("Please run text analysis first before AI enhancement.")
                return False
            
            with st.spinner("Running AI enhancement on text analysis... This may take a few minutes."):
                ai_results = self.data_processor.run_ai_enhancement(st.session_state.processed_data)
                st.session_state.ai_enhancement_results = ai_results
                st.session_state.ai_enhancement_complete = True
            
            if ai_results:
                st.success(f"✅ AI enhancement complete for {len(ai_results)} products")
                return True
            else:
                st.warning("No AI enhancement results generated. Check that you have customer feedback data.")
                return False
            
        except Exception as e:
            logger.error(f"Error running AI enhancement: {str(e)}")
            st.error(f"❌ AI enhancement failed: {str(e)}")
            return False
    
    def run_application(self):
        """Main application entry point focused on text analysis workflow"""
        try:
            # Dashboard available - proceed with text analysis focused rendering
            if not self.dashboard:
                self._render_minimal_interface()
                return
            
            # Handle example data loading
            if st.session_state.get('load_example', False) or st.session_state.show_example_data:
                SessionManager.load_example_data()
                if st.session_state.uploaded_data and not st.session_state.data_processed:
                    self.process_data()
                st.session_state['load_example'] = False
            
            # Render main text analysis dashboard
            try:
                self.dashboard.render_text_analysis_dashboard()
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
        st.title("🏥 Medical Device Customer Feedback Analyzer")
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
                if st.button("Process Data & Analyze Text"):
                    self.process_data()
    
    def _render_error_interface(self, error_message: str):
        """Render error interface when critical errors occur"""
        st.error("🚨 Critical Application Error")
        st.error("The application encountered a critical error and cannot continue.")
        st.error(f"Error: {error_message}")
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **System Information:**
            - App Version: {APP_CONFIG['version']}
            - Focus: {APP_CONFIG['focus']}
            - Compliance: {APP_CONFIG['compliance']}
            
            **Module Status:**
            """)
            
            for module, available in st.session_state.module_status.items():
                status = "Available" if available else "Not Available"
                st.markdown(f"- {module}: {status}")
            
            st.markdown(f"""
            **Support:** {APP_CONFIG['support_email']}
            """)
    
    def _handle_processing_triggers(self):
        """Handle background processing triggers"""
        try:
            # Auto-process data if new uploads detected
            if (st.session_state.uploaded_data and 
                not st.session_state.data_processed and 
                not st.session_state.get('processing_locked', False)):
                
                logger.info("Auto-processing new data for text analysis")
                if self.process_data():
                    st.rerun()
            
            # Handle date filter changes
            if 'date_filter_changed' in st.session_state:
                del st.session_state['date_filter_changed']
                
                if st.session_state.data_processed:
                    logger.info("Re-processing data due to date filter change")
                    if self.process_data():
                        st.rerun()
                        
        except Exception as e:
            logger.error(f"Error in processing triggers: {str(e)}")

def main():
    """Application entry point focused on customer feedback text analysis"""
    try:
        # Set Streamlit page config for text analysis workflow
        st.set_page_config(
            page_title=APP_CONFIG['title'],
            page_icon="🔍",
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
        - Focus: {APP_CONFIG['focus']}
        """)

if __name__ == "__main__":
    main()
