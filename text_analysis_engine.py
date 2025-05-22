"""
Text Analysis Engine for Medical Device Customer Feedback - FIXED VERSION

**STABLE & ACCURATE ANALYSIS ENGINE**

Core analytical engine for processing customer reviews and generating
medical device quality insights with AI enhancement.

Author: Assistant
Version: 4.0 - Production Stable
"""

import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import json
import statistics

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

# Check for AI enhancement
requests, has_requests = safe_import('requests')

# Medical Device Quality Categories (Simplified & Accurate)
QUALITY_CATEGORIES = {
    'safety_concerns': {
        'name': 'Safety & Risk',
        'keywords': [
            'unsafe', 'dangerous', 'injury', 'hurt', 'broke', 'broken', 'unstable',
            'tip over', 'fall', 'collapsed', 'sharp', 'cuts', 'hazard', 'accident'
        ],
        'severity': 'critical',
        'weight': 10.0
    },
    'efficacy_performance': {
        'name': 'Effectiveness & Performance',
        'keywords': [
            'doesnt work', 'not working', 'ineffective', 'useless', 'no relief',
            'no help', 'waste of money', 'not helpful', 'disappointing', 'poor performance'
        ],
        'severity': 'high',
        'weight': 8.0
    },
    'comfort_usability': {
        'name': 'Comfort & Usability',
        'keywords': [
            'uncomfortable', 'painful', 'hurts', 'difficult to use', 'hard to use',
            'awkward', 'confusing', 'user unfriendly', 'heavy', 'bulky'
        ],
        'severity': 'medium',
        'weight': 6.0
    },
    'durability_quality': {
        'name': 'Durability & Quality',
        'keywords': [
            'cheap', 'poor quality', 'flimsy', 'fell apart', 'broke', 'broken',
            'cheap plastic', 'shoddy', 'low quality', 'doesnt last'
        ],
        'severity': 'high',
        'weight': 7.0
    },
    'sizing_fit': {
        'name': 'Sizing & Fit',
        'keywords': [
            'too small', 'too big', 'wrong size', 'doesnt fit', 'tight', 'loose',
            'sizing chart wrong', 'measurements off', 'runs small', 'runs large'
        ],
        'severity': 'medium',
        'weight': 5.0
    },
    'assembly_instructions': {
        'name': 'Assembly & Instructions',
        'keywords': [
            'difficult assembly', 'confusing instructions', 'missing parts',
            'unclear directions', 'hard to assemble', 'poor instructions'
        ],
        'severity': 'medium',
        'weight': 4.0
    }
}

# Positive indicators
POSITIVE_INDICATORS = [
    'excellent', 'amazing', 'great', 'love it', 'perfect', 'fantastic',
    'highly recommend', 'best purchase', 'works great', 'comfortable',
    'easy to use', 'good quality', 'well made', 'sturdy', 'durable'
]

class TextProcessor:
    """Enhanced text processing for medical device reviews"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize contractions
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "can't": "cannot", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, keywords: List[str]) -> List[Tuple[str, int]]:
        """Extract keywords and their frequencies"""
        text = TextProcessor.clean_text(text)
        found_keywords = []
        
        for keyword in keywords:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text))
            if matches > 0:
                found_keywords.append((keyword, matches))
        
        return sorted(found_keywords, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def calculate_sentiment_score(text: str, rating: Optional[int] = None) -> float:
        """Calculate sentiment score (0-1 scale)"""
        text = TextProcessor.clean_text(text)
        
        # Count positive and negative indicators
        positive_count = sum(1 for indicator in POSITIVE_INDICATORS if indicator in text)
        
        negative_count = 0
        for category_data in QUALITY_CATEGORIES.values():
            for keyword in category_data['keywords']:
                if keyword in text:
                    negative_count += 1
        
        # Text-based sentiment
        text_sentiment = 0.5  # Neutral
        if positive_count > 0 or negative_count > 0:
            total = positive_count + negative_count
            text_sentiment = positive_count / total if total > 0 else 0.5
        
        # Rating-based sentiment (if available)
        if rating is not None:
            rating_sentiment = (rating - 1) / 4  # Convert 1-5 to 0-1
            # Combine with 70% rating weight, 30% text weight
            return (rating_sentiment * 0.7) + (text_sentiment * 0.3)
        
        return text_sentiment

class CategoryAnalyzer:
    """Analyze reviews by medical device quality categories"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.categories = QUALITY_CATEGORIES
    
    def analyze_reviews(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reviews and categorize by quality issues"""
        if not reviews:
            return self._empty_analysis()
        
        total_reviews = len(reviews)
        category_results = {}
        
        logger.info(f"Analyzing {total_reviews} reviews across {len(self.categories)} categories")
        
        for category_id, category_info in self.categories.items():
            matched_reviews = []
            all_keywords = []
            
            for review in reviews:
                text = review.get('text', '')
                keywords_found = self.text_processor.extract_keywords(text, category_info['keywords'])
                
                if keywords_found:
                    matched_reviews.append(review)
                    all_keywords.extend([kw for kw, count in keywords_found for _ in range(count)])
            
            count = len(matched_reviews)
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            
            # Calculate severity breakdown
            severity_breakdown = self._calculate_severity(matched_reviews)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(count, category_info, total_reviews)
            
            # Top keywords
            keyword_freq = Counter(all_keywords).most_common(5)
            
            category_results[category_id] = {
                'name': category_info['name'],
                'count': count,
                'percentage': round(percentage, 1),
                'severity': category_info['severity'],
                'weight': category_info['weight'],
                'matched_reviews': matched_reviews,
                'top_keywords': keyword_freq,
                'severity_breakdown': severity_breakdown,
                'risk_score': risk_score,
                'requires_action': count > 0 and category_info['severity'] in ['critical', 'high']
            }
        
        return category_results
    
    def _calculate_severity(self, reviews: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate severity breakdown for matched reviews"""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for review in reviews:
            rating = review.get('rating')
            
            if rating is not None:
                if rating <= 2:
                    severity_counts['high'] += 1
                elif rating == 3:
                    severity_counts['medium'] += 1
                else:
                    severity_counts['low'] += 1
            else:
                severity_counts['medium'] += 1  # Default for unrated
        
        return severity_counts
    
    def _calculate_risk_score(self, count: int, category_info: Dict[str, Any], total_reviews: int) -> float:
        """Calculate risk score for category"""
        if count == 0:
            return 0.0
        
        # Base score from category weight
        base_score = category_info['weight']
        
        # Frequency factor
        frequency_factor = min((count / max(total_reviews, 1)) * 20, 10)
        
        # Severity multiplier
        severity_multiplier = {
            'critical': 2.0,
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }.get(category_info['severity'], 1.0)
        
        risk_score = (base_score + frequency_factor) * severity_multiplier
        return min(risk_score, 30.0)  # Cap at 30
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for no data"""
        return {cat_id: {
            'name': cat_info['name'],
            'count': 0,
            'percentage': 0,
            'severity': cat_info['severity'],
            'weight': cat_info['weight'],
            'matched_reviews': [],
            'top_keywords': [],
            'severity_breakdown': {'high': 0, 'medium': 0, 'low': 0},
            'risk_score': 0.0,
            'requires_action': False
        } for cat_id, cat_info in self.categories.items()}

class QualityAssessment:
    """Assess overall quality from review analysis"""
    
    @staticmethod
    def assess_quality(reviews: List[Dict[str, Any]], category_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality from reviews and category analysis"""
        if not reviews:
            return QualityAssessment._empty_assessment()
        
        total_reviews = len(reviews)
        
        # Calculate sentiment scores
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        ratings = []
        
        for review in reviews:
            # Calculate sentiment
            sentiment = TextProcessor.calculate_sentiment_score(
                review.get('text', ''), 
                review.get('rating')
            )
            sentiment_scores.append(sentiment)
            
            # Categorize sentiment
            if sentiment >= 0.6:
                positive_count += 1
            elif sentiment <= 0.4:
                negative_count += 1
            else:
                neutral_count += 1
            
            # Collect ratings
            if review.get('rating') is not None:
                ratings.append(review['rating'])
        
        # Calculate overall quality score
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        quality_score = avg_sentiment * 100
        
        # Apply penalties for high-risk categories
        safety_issues = category_results.get('safety_concerns', {}).get('count', 0)
        efficacy_issues = category_results.get('efficacy_performance', {}).get('count', 0)
        
        # Penalties
        safety_penalty = min(safety_issues * 20, 40)
        efficacy_penalty = min(efficacy_issues * 15, 30)
        
        quality_score = max(0, quality_score - safety_penalty - efficacy_penalty)
        
        # Quality level
        quality_level = QualityAssessment._determine_quality_level(quality_score)
        
        # Rating analysis
        rating_distribution = dict(Counter(ratings)) if ratings else {}
        average_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # High-risk categories
        high_risk_categories = [
            cat_id for cat_id, cat_data in category_results.items()
            if cat_data.get('requires_action', False)
        ]
        
        return {
            'total_reviews': total_reviews,
            'quality_score': round(quality_score, 1),
            'quality_level': quality_level,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_ratio': round((positive_count - negative_count) / total_reviews, 3),
            'rating_distribution': rating_distribution,
            'average_rating': round(average_rating, 2),
            'high_risk_categories': high_risk_categories,
            'safety_issues_count': safety_issues,
            'efficacy_issues_count': efficacy_issues
        }
    
    @staticmethod
    def _determine_quality_level(score: float) -> str:
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
    
    @staticmethod
    def _empty_assessment() -> Dict[str, Any]:
        """Return empty assessment"""
        return {
            'total_reviews': 0,
            'quality_score': 0.0,
            'quality_level': 'No Data',
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'sentiment_ratio': 0.0,
            'rating_distribution': {},
            'average_rating': 0.0,
            'high_risk_categories': [],
            'safety_issues_count': 0,
            'efficacy_issues_count': 0
        }

class CAPAGenerator:
    """Generate CAPA (Corrective and Preventive Action) recommendations"""
    
    @staticmethod
    def generate_capa_recommendations(category_results: Dict[str, Any], 
                                    quality_assessment: Dict[str, Any],
                                    product_name: str) -> List[Dict[str, Any]]:
        """Generate CAPA recommendations based on analysis"""
        recommendations = []
        capa_counter = 1
        
        # Safety CAPA (highest priority)
        safety_data = category_results.get('safety_concerns', {})
        if safety_data.get('count', 0) > 0:
            recommendations.append({
                'capa_id': f"CAPA-{capa_counter:03d}",
                'priority': 'Critical',
                'category': 'Safety & Risk Management',
                'issue_description': f"{safety_data['count']} safety concerns identified in customer feedback",
                'corrective_action': 'Immediate safety review and customer notification if required',
                'preventive_action': 'Implement enhanced safety testing and clearer safety warnings',
                'timeline': 'Immediate (24-48 hours)',
                'responsibility': 'Quality Manager + Engineering',
                'success_metrics': ['Zero safety incidents in next 30 days'],
                'affected_customers': safety_data['count']
            })
            capa_counter += 1
        
        # Efficacy CAPA
        efficacy_data = category_results.get('efficacy_performance', {})
        if efficacy_data.get('count', 0) > 2:
            recommendations.append({
                'capa_id': f"CAPA-{capa_counter:03d}",
                'priority': 'High',
                'category': 'Product Effectiveness',
                'issue_description': f"{efficacy_data['count']} effectiveness complaints affecting customer satisfaction",
                'corrective_action': 'Review product specifications and customer expectations',
                'preventive_action': 'Improve product listing accuracy and customer education',
                'timeline': '1-2 weeks',
                'responsibility': 'Product Manager + Marketing',
                'success_metrics': ['Reduce effectiveness complaints by 50%'],
                'affected_customers': efficacy_data['count']
            })
            capa_counter += 1
        
        # Quality CAPA
        quality_data = category_results.get('durability_quality', {})
        if quality_data.get('count', 0) > 2:
            recommendations.append({
                'capa_id': f"CAPA-{capa_counter:03d}",
                'priority': 'High',
                'category': 'Product Quality & Durability',
                'issue_description': f"{quality_data['count']} quality/durability complaints",
                'corrective_action': 'Review manufacturing processes and material quality',
                'preventive_action': 'Implement additional quality checkpoints',
                'timeline': '2-3 weeks',
                'responsibility': 'Manufacturing + Quality Assurance',
                'success_metrics': ['Reduce quality complaints by 40%'],
                'affected_customers': quality_data['count']
            })
            capa_counter += 1
        
        # Assembly CAPA
        assembly_data = category_results.get('assembly_instructions', {})
        if assembly_data.get('count', 0) > 1:
            recommendations.append({
                'capa_id': f"CAPA-{capa_counter:03d}",
                'priority': 'Medium',
                'category': 'Documentation & Instructions',
                'issue_description': f"{assembly_data['count']} assembly/instruction complaints",
                'corrective_action': 'Revise assembly instructions with clearer diagrams',
                'preventive_action': 'User testing of instructions before product launch',
                'timeline': '2-4 weeks',
                'responsibility': 'Technical Writing + Customer Experience',
                'success_metrics': ['Improve instruction clarity rating to 4.0+'],
                'affected_customers': assembly_data['count']
            })
            capa_counter += 1
        
        # Overall quality CAPA
        if quality_assessment.get('quality_score', 50) < 60:
            recommendations.append({
                'capa_id': f"CAPA-{capa_counter:03d}",
                'priority': 'High',
                'category': 'Overall Customer Satisfaction',
                'issue_description': f"Overall quality score {quality_assessment['quality_score']:.1f}% below target",
                'corrective_action': 'Comprehensive review of top customer complaints',
                'preventive_action': 'Implement proactive customer feedback monitoring',
                'timeline': '3-4 weeks',
                'responsibility': 'Quality Manager + Customer Success',
                'success_metrics': ['Achieve quality score above 75%'],
                'affected_customers': quality_assessment.get('negative_count', 0)
            })
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations

class AIEnhancer:
    """AI enhancement for text analysis (optional)"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.available = bool(self.api_key and has_requests)
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or secrets"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                return st.secrets['openai_api_key']
        except:
            pass
        
        try:
            import os
            return os.environ.get('OPENAI_API_KEY')
        except:
            pass
        
        return None
    
    def is_available(self) -> bool:
        """Check if AI enhancement is available"""
        return self.available
    
    def enhance_analysis(self, reviews: List[Dict[str, Any]], 
                        product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis with AI insights"""
        if not self.is_available():
            return {'ai_analysis_available': False}
        
        try:
            # Create summary of reviews for AI
            review_summaries = []
            for i, review in enumerate(reviews[:10], 1):  # Limit to 10 reviews
                rating_text = f" (Rating: {review['rating']}/5)" if review.get('rating') else ""
                text = review.get('text', '')[:300]  # Limit text length
                review_summaries.append(f"Review {i}{rating_text}: {text}")
            
            # Create AI prompt
            prompt = self._create_analysis_prompt(product_info, review_summaries)
            
            # Make API call
            response = self._call_openai_api(prompt)
            
            if response.get('success'):
                ai_insights = self._parse_ai_response(response['content'])
                ai_insights['ai_analysis_available'] = True
                return ai_insights
            else:
                return {'ai_analysis_available': False, 'ai_error': response.get('error')}
                
        except Exception as e:
            logger.error(f"AI enhancement error: {str(e)}")
            return {'ai_analysis_available': False, 'ai_error': str(e)}
    
    def _create_analysis_prompt(self, product_info: Dict[str, Any], 
                              review_summaries: List[str]) -> str:
        """Create AI analysis prompt"""
        product_name = product_info.get('name', 'Unknown Product')
        
        return f"""Analyze these medical device customer reviews for quality insights:

PRODUCT: {product_name}
CATEGORY: Medical Device

REVIEWS:
{chr(10).join(review_summaries)}

Provide analysis in this format:

## OVERALL SENTIMENT
[Positive/Negative/Mixed] - [confidence percentage]

## KEY ISSUES
[List top 3 issues mentioned across reviews]

## SAFETY CONCERNS
[Any safety-related issues - mark as CRITICAL if found]

## RECOMMENDATIONS
[Top 3 actionable recommendations for improvement]

## LISTING IMPROVEMENTS
[Specific suggestions for Amazon listing optimization]

Focus on medical device quality and safety. Be specific and actionable."""
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to OpenAI"""
        if not has_requests:
            return {'success': False, 'error': 'Requests module not available'}
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are an expert medical device quality analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content']
                }
            else:
                return {
                    'success': False,
                    'error': f"API error {response.status_code}"
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        insights = {}
        
        # Extract sections using regex
        sections = {
            'overall_sentiment': r'## OVERALL SENTIMENT\s*\n(.*?)(?=## |$)',
            'key_issues': r'## KEY ISSUES\s*\n(.*?)(?=## |$)',
            'safety_concerns': r'## SAFETY CONCERNS\s*\n(.*?)(?=## |$)',
            'recommendations': r'## RECOMMENDATIONS\s*\n(.*?)(?=## |$)',
            'listing_improvements': r'## LISTING IMPROVEMENTS\s*\n(.*?)(?=## |$)'
        }
        
        for section_name, pattern in sections.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                insights[section_name] = match.group(1).strip()
        
        return insights

class TextAnalysisEngine:
    """Main text analysis engine with AI enhancement"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.category_analyzer = CategoryAnalyzer()
        self.quality_assessor = QualityAssessment()
        self.capa_generator = CAPAGenerator()
        self.ai_enhancer = AIEnhancer()
        
        logger.info("Text Analysis Engine initialized - Production stable version")
    
    def analyze_helium10_reviews(self, review_data: List[Dict[str, Any]], 
                                product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis function for Helium 10 reviews"""
        try:
            if not review_data:
                return self._create_empty_result(product_info)
            
            logger.info(f"Analyzing {len(review_data)} reviews for {product_info.get('name', 'Unknown')}")
            
            # Prepare reviews
            reviews = self._prepare_reviews(review_data, product_info)
            
            # Category analysis
            category_results = self.category_analyzer.analyze_reviews(reviews)
            
            # Quality assessment
            quality_assessment = self.quality_assessor.assess_quality(reviews, category_results)
            
            # CAPA recommendations
            capa_recommendations = self.capa_generator.generate_capa_recommendations(
                category_results, quality_assessment, product_info.get('name', 'Unknown Product')
            )
            
            # Risk assessment
            risk_level, risk_factors = self._assess_risk(category_results, quality_assessment)
            
            # AI enhancement (optional)
            ai_insights = self.ai_enhancer.enhance_analysis(reviews, product_info)
            
            # Compile results
            result = {
                'success': True,
                'asin': product_info.get('asin', 'unknown'),
                'product_name': product_info.get('name', 'Unknown Product'),
                'total_reviews': len(reviews),
                'analysis_timestamp': datetime.now().isoformat(),
                
                # Core analysis
                'category_analysis': category_results,
                'quality_assessment': quality_assessment,
                'capa_recommendations': capa_recommendations,
                
                # Risk assessment
                'overall_risk_level': risk_level,
                'risk_factors': risk_factors,
                
                # AI insights
                'ai_insights': ai_insights,
                'ai_analysis_available': ai_insights.get('ai_analysis_available', False),
                
                # Metadata
                'analysis_method': 'enhanced_text_analysis',
                'engine_version': '4.0'
            }
            
            logger.info(f"Analysis completed: {len(capa_recommendations)} CAPA items, Risk: {risk_level}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'asin': product_info.get('asin', 'unknown'),
                'product_name': product_info.get('name', 'Unknown Product'),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _prepare_reviews(self, review_data: List[Dict[str, Any]], 
                        product_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare and standardize review data"""
        prepared_reviews = []
        
        for item in review_data:
            try:
                # Extract text
                text = item.get('text', '')
                if not text:
                    # Try to construct from title and body
                    title = item.get('review_title', '')
                    body = item.get('review_body', '')
                    if title and body:
                        text = f"{title} | {body}"
                    elif body:
                        text = body
                    elif title:
                        text = title
                
                if not text or len(text.strip()) < 5:
                    continue
                
                # Standardize review
                review = {
                    'text': text,
                    'rating': item.get('rating'),
                    'date': item.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'author': item.get('author', 'Anonymous'),
                    'verified': item.get('verified', 'Unknown'),
                    'source': item.get('source', 'unknown'),
                    'asin': item.get('asin', product_info.get('asin', 'unknown'))
                }
                
                prepared_reviews.append(review)
                
            except Exception as e:
                logger.warning(f"Error preparing review: {str(e)}")
                continue
        
        return prepared_reviews
    
    def _assess_risk(self, category_results: Dict[str, Any], 
                    quality_assessment: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Assess overall risk level"""
        risk_factors = []
        risk_score = 0
        
        # Safety risk (highest weight)
        safety_count = category_results.get('safety_concerns', {}).get('count', 0)
        if safety_count > 0:
            risk_score += safety_count * 15
            risk_factors.append(f"{safety_count} safety concerns identified")
        
        # Efficacy risk
        efficacy_count = category_results.get('efficacy_performance', {}).get('count', 0)
        if efficacy_count > 2:
            risk_score += efficacy_count * 5
            risk_factors.append(f"Multiple effectiveness complaints ({efficacy_count})")
        
        # Quality score risk
        quality_score = quality_assessment.get('quality_score', 50)
        if quality_score < 50:
            risk_score += (50 - quality_score) / 2
            risk_factors.append(f"Low quality score ({quality_score:.1f}%)")
        
        # Negative feedback ratio
        negative_ratio = abs(quality_assessment.get('sentiment_ratio', 0))
        if negative_ratio > 0.3:
            risk_score += negative_ratio * 20
            risk_factors.append(f"High negative feedback ratio ({negative_ratio:.1%})")
        
        # Determine risk level
        if risk_score >= 30:
            risk_level = 'Critical'
        elif risk_score >= 15:
            risk_level = 'High'
        elif risk_score >= 8:
            risk_level = 'Medium'
        elif risk_score >= 3:
            risk_level = 'Low'
        else:
            risk_level = 'Minimal'
        
        return risk_level, risk_factors
    
    def _create_empty_result(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty result for no data scenarios"""
        return {
            'success': False,
            'error': 'No review data available',
            'asin': product_info.get('asin', 'unknown'),
            'product_name': product_info.get('name', 'Unknown Product'),
            'total_reviews': 0,
            'analysis_timestamp': datetime.now().isoformat()
        }

# Export main class
__all__ = ['TextAnalysisEngine', 'QUALITY_CATEGORIES']
