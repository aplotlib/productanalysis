"""
Composite Scoring System for Amazon Medical Device Listing Optimizer

This module calculates comprehensive 0-100 performance scores for each product based on:
- Sales performance and velocity
- Return rates and patterns
- Customer satisfaction (star ratings, review sentiment)
- Review volume and engagement
- Profit margins and financial health
- Category benchmarking and competitive positioning

Author: Assistant
Version: 2.0
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Scoring weights and parameters
SCORING_WEIGHTS = {
    'sales_performance': 0.25,      # 25% - Sales velocity and volume
    'return_rate': 0.20,            # 20% - Return rate optimization
    'customer_satisfaction': 0.20,  # 20% - Star ratings and sentiment
    'review_engagement': 0.15,      # 15% - Review volume and quality
    'profitability': 0.10,          # 10% - Margins and financial health
    'competitive_position': 0.10    # 10% - Category performance vs competitors
}

# Category benchmarks for medical devices (industry averages)
CATEGORY_BENCHMARKS = {
    'Mobility Aids': {
        'avg_return_rate': 6.5,
        'avg_star_rating': 4.1,
        'avg_reviews_per_month': 8,
        'avg_profit_margin': 45
    },
    'Bathroom Safety': {
        'avg_return_rate': 4.2,
        'avg_star_rating': 4.3,
        'avg_reviews_per_month': 12,
        'avg_profit_margin': 52
    },
    'Pain Relief': {
        'avg_return_rate': 8.1,
        'avg_star_rating': 3.9,
        'avg_reviews_per_month': 15,
        'avg_profit_margin': 38
    },
    'Orthopedic Support': {
        'avg_return_rate': 7.3,
        'avg_star_rating': 4.0,
        'avg_reviews_per_month': 18,
        'avg_profit_margin': 41
    },
    'Blood Pressure Monitors': {
        'avg_return_rate': 5.8,
        'avg_star_rating': 4.2,
        'avg_reviews_per_month': 22,
        'avg_profit_margin': 35
    },
    'Diabetes Care': {
        'avg_return_rate': 4.9,
        'avg_star_rating': 4.4,
        'avg_reviews_per_month': 25,
        'avg_profit_margin': 48
    },
    'Default': {  # Fallback for unknown categories
        'avg_return_rate': 6.0,
        'avg_star_rating': 4.1,
        'avg_reviews_per_month': 15,
        'avg_profit_margin': 42
    }
}

# Performance thresholds for scoring
PERFORMANCE_THRESHOLDS = {
    'excellent': {'min_score': 85, 'color': '#22C55E', 'label': 'Excellent'},
    'good': {'min_score': 70, 'color': '#3B82F6', 'label': 'Good'},
    'average': {'min_score': 55, 'color': '#F59E0B', 'label': 'Average'},
    'needs_improvement': {'min_score': 40, 'color': '#EF4444', 'label': 'Needs Improvement'},
    'critical': {'min_score': 0, 'color': '#DC2626', 'label': 'Critical'}
}

@dataclass
class ProductMetrics:
    """Container for all product performance metrics"""
    asin: str
    name: str
    category: str
    
    # Sales metrics
    sales_30d: int
    sales_365d: Optional[int] = None
    sales_velocity: Optional[float] = None  # Sales per day
    
    # Return metrics  
    returns_30d: int
    returns_365d: Optional[int] = None
    return_rate_30d: float = 0.0
    return_rate_365d: Optional[float] = None
    
    # Customer satisfaction
    star_rating: Optional[float] = None
    total_reviews: Optional[int] = None
    review_sentiment_score: Optional[float] = None  # From AI analysis
    
    # Financial metrics
    average_price: Optional[float] = None
    cost_per_unit: Optional[float] = None
    profit_margin: Optional[float] = None
    
    # Engagement metrics
    reviews_per_month: Optional[float] = None
    recent_review_trend: Optional[str] = None  # 'improving', 'stable', 'declining'
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from base data"""
        # Calculate return rates
        if self.sales_30d > 0:
            self.return_rate_30d = (self.returns_30d / self.sales_30d) * 100
        
        if self.sales_365d and self.sales_365d > 0 and self.returns_365d is not None:
            self.return_rate_365d = (self.returns_365d / self.sales_365d) * 100
        
        # Calculate sales velocity (sales per day)
        self.sales_velocity = self.sales_30d / 30
        
        # Calculate profit margin if cost data available
        if self.average_price and self.cost_per_unit:
            self.profit_margin = ((self.average_price - self.cost_per_unit) / self.average_price) * 100
        
        # Calculate reviews per month estimate
        if self.total_reviews and self.sales_365d:
            # Estimate based on review ratio (rough approximation)
            review_ratio = self.total_reviews / max(self.sales_365d, 1)
            self.reviews_per_month = (self.sales_30d * review_ratio) * (30/30)  # Reviews per 30 days

@dataclass
class ComponentScore:
    """Individual component score with details"""
    component: str
    raw_score: float
    weighted_score: float
    weight: float
    benchmark_comparison: str
    performance_level: str
    improvement_potential: float
    key_drivers: List[str]
    
@dataclass
class CompositeScore:
    """Complete composite score with breakdown"""
    asin: str
    product_name: str
    category: str
    
    # Main score
    composite_score: float
    performance_level: str
    score_color: str
    
    # Component scores
    component_scores: Dict[str, ComponentScore]
    
    # Benchmarking
    category_ranking: Optional[str] = None
    improvement_priority: List[str] = None
    
    # Trends and insights
    score_trend: Optional[str] = None  # 'improving', 'stable', 'declining'
    risk_factors: List[str] = None
    strengths: List[str] = None
    
    # Financial impact
    revenue_impact: Optional[float] = None
    potential_savings: Optional[float] = None
    
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class PerformanceCalculator:
    """Calculates individual performance component scores"""
    
    @staticmethod
    def calculate_sales_performance_score(metrics: ProductMetrics, category_benchmark: Dict[str, float]) -> ComponentScore:
        """Calculate sales performance score (0-100)"""
        
        # Base score on sales velocity and volume
        sales_velocity = metrics.sales_velocity or 0
        
        # Normalize sales velocity (assuming 10+ sales/day is excellent for medical devices)
        velocity_score = min(sales_velocity / 10 * 100, 100)
        
        # Volume consistency bonus (if 365d data available)
        consistency_bonus = 0
        if metrics.sales_365d:
            expected_annual = metrics.sales_30d * 12
            actual_annual = metrics.sales_365d
            consistency_ratio = min(actual_annual / max(expected_annual, 1), 1.5)
            consistency_bonus = min(consistency_ratio * 20, 20)  # Up to 20 point bonus
        
        raw_score = min(velocity_score + consistency_bonus, 100)
        
        # Determine performance level and key drivers
        if raw_score >= 80:
            performance_level = "Excellent"
            key_drivers = ["Strong daily sales velocity", "Consistent performance"]
        elif raw_score >= 60:
            performance_level = "Good"
            key_drivers = ["Solid sales volume", "Room for velocity improvement"]
        elif raw_score >= 40:
            performance_level = "Average"
            key_drivers = ["Moderate sales", "Needs velocity boost"]
        else:
            performance_level = "Needs Improvement"
            key_drivers = ["Low sales velocity", "Volume concerns"]
        
        # Benchmark comparison
        benchmark_comparison = "Above average" if sales_velocity > 5 else "Below average"
        
        return ComponentScore(
            component="Sales Performance",
            raw_score=raw_score,
            weighted_score=raw_score * SCORING_WEIGHTS['sales_performance'],
            weight=SCORING_WEIGHTS['sales_performance'],
            benchmark_comparison=benchmark_comparison,
            performance_level=performance_level,
            improvement_potential=100 - raw_score,
            key_drivers=key_drivers
        )
    
    @staticmethod
    def calculate_return_rate_score(metrics: ProductMetrics, category_benchmark: Dict[str, float]) -> ComponentScore:
        """Calculate return rate score (0-100, higher is better)"""
        
        return_rate = metrics.return_rate_30d
        benchmark_return_rate = category_benchmark.get('avg_return_rate', 6.0)
        
        # Score calculation: excellent return rate (0-2%), good (2-5%), average (5-8%), poor (8%+)
        if return_rate <= 2.0:
            raw_score = 100
            performance_level = "Excellent"
            key_drivers = ["Exceptional return rate", "High customer satisfaction"]
        elif return_rate <= 4.0:
            raw_score = 85
            performance_level = "Good"
            key_drivers = ["Low return rate", "Good product-market fit"]
        elif return_rate <= 6.0:
            raw_score = 70
            performance_level = "Average"
            key_drivers = ["Moderate returns", "Industry average"]
        elif return_rate <= 10.0:
            raw_score = 50
            performance_level = "Needs Improvement"
            key_drivers = ["High return rate", "Customer expectations mismatch"]
        else:
            raw_score = 25
            performance_level = "Critical"
            key_drivers = ["Very high returns", "Major product/listing issues"]
        
        # Adjust based on category benchmark
        if return_rate < benchmark_return_rate * 0.7:
            benchmark_comparison = "Significantly better than category average"
            raw_score = min(raw_score + 10, 100)
        elif return_rate < benchmark_return_rate:
            benchmark_comparison = "Better than category average"
            raw_score = min(raw_score + 5, 100)
        elif return_rate > benchmark_return_rate * 1.5:
            benchmark_comparison = "Significantly worse than category average"
            raw_score = max(raw_score - 15, 0)
        else:
            benchmark_comparison = "About category average"
        
        return ComponentScore(
            component="Return Rate",
            raw_score=raw_score,
            weighted_score=raw_score * SCORING_WEIGHTS['return_rate'],
            weight=SCORING_WEIGHTS['return_rate'],
            benchmark_comparison=benchmark_comparison,
            performance_level=performance_level,
            improvement_potential=100 - raw_score,
            key_drivers=key_drivers
        )
    
    @staticmethod
    def calculate_customer_satisfaction_score(metrics: ProductMetrics, category_benchmark: Dict[str, float]) -> ComponentScore:
        """Calculate customer satisfaction score based on ratings and sentiment"""
        
        star_rating = metrics.star_rating or 0
        sentiment_score = metrics.review_sentiment_score or 0  # From AI analysis (0-100)
        benchmark_rating = category_benchmark.get('avg_star_rating', 4.1)
        
        # Star rating component (0-100)
        star_component = (star_rating / 5.0) * 100 if star_rating > 0 else 50
        
        # Sentiment component (from AI analysis)
        sentiment_component = sentiment_score if sentiment_score > 0 else 50
        
        # Weighted combination (star rating 70%, sentiment 30%)
        raw_score = (star_component * 0.7) + (sentiment_component * 0.3)
        
        # Performance level determination
        if raw_score >= 90:
            performance_level = "Excellent"
            key_drivers = ["Outstanding customer satisfaction", "Positive review sentiment"]
        elif raw_score >= 80:
            performance_level = "Good"
            key_drivers = ["Good customer satisfaction", "Mostly positive feedback"]
        elif raw_score >= 70:
            performance_level = "Average"
            key_drivers = ["Average satisfaction", "Mixed customer feedback"]
        elif raw_score >= 60:
            performance_level = "Needs Improvement"
            key_drivers = ["Below average satisfaction", "Negative feedback patterns"]
        else:
            performance_level = "Critical"
            key_drivers = ["Poor customer satisfaction", "Significant issues reported"]
        
        # Benchmark comparison
        if star_rating > benchmark_rating + 0.3:
            benchmark_comparison = "Significantly above category average"
        elif star_rating > benchmark_rating:
            benchmark_comparison = "Above category average"
        elif star_rating < benchmark_rating - 0.3:
            benchmark_comparison = "Below category average"
        else:
            benchmark_comparison = "About category average"
        
        return ComponentScore(
            component="Customer Satisfaction",
            raw_score=raw_score,
            weighted_score=raw_score * SCORING_WEIGHTS['customer_satisfaction'],
            weight=SCORING_WEIGHTS['customer_satisfaction'],
            benchmark_comparison=benchmark_comparison,
            performance_level=performance_level,
            improvement_potential=100 - raw_score,
            key_drivers=key_drivers
        )
    
    @staticmethod
    def calculate_review_engagement_score(metrics: ProductMetrics, category_benchmark: Dict[str, float]) -> ComponentScore:
        """Calculate review engagement score based on volume and recency"""
        
        total_reviews = metrics.total_reviews or 0
        reviews_per_month = metrics.reviews_per_month or 0
        benchmark_monthly = category_benchmark.get('avg_reviews_per_month', 15)
        
        # Volume score (based on total reviews)
        if total_reviews >= 500:
            volume_score = 100
        elif total_reviews >= 200:
            volume_score = 85
        elif total_reviews >= 100:
            volume_score = 70
        elif total_reviews >= 50:
            volume_score = 55
        elif total_reviews >= 20:
            volume_score = 40
        else:
            volume_score = 25
        
        # Velocity score (based on reviews per month)
        velocity_ratio = reviews_per_month / max(benchmark_monthly, 1)
        velocity_score = min(velocity_ratio * 100, 100)
        
        # Combined score (volume 60%, velocity 40%)
        raw_score = (volume_score * 0.6) + (velocity_score * 0.4)
        
        # Performance level determination
        if raw_score >= 85:
            performance_level = "Excellent"
            key_drivers = ["High review volume", "Strong customer engagement"]
        elif raw_score >= 70:
            performance_level = "Good"
            key_drivers = ["Good review volume", "Decent engagement"]
        elif raw_score >= 55:
            performance_level = "Average"
            key_drivers = ["Moderate review activity", "Average engagement"]
        else:
            performance_level = "Needs Improvement"
            key_drivers = ["Low review volume", "Limited customer engagement"]
        
        # Benchmark comparison
        if reviews_per_month > benchmark_monthly * 1.3:
            benchmark_comparison = "Above category average engagement"
        elif reviews_per_month > benchmark_monthly * 0.7:
            benchmark_comparison = "About category average"
        else:
            benchmark_comparison = "Below category average engagement"
        
        return ComponentScore(
            component="Review Engagement",
            raw_score=raw_score,
            weighted_score=raw_score * SCORING_WEIGHTS['review_engagement'],
            weight=SCORING_WEIGHTS['review_engagement'],
            benchmark_comparison=benchmark_comparison,
            performance_level=performance_level,
            improvement_potential=100 - raw_score,
            key_drivers=key_drivers
        )
    
    @staticmethod
    def calculate_profitability_score(metrics: ProductMetrics, category_benchmark: Dict[str, float]) -> ComponentScore:
        """Calculate profitability score based on margins and financial health"""
        
        profit_margin = metrics.profit_margin or 0
        benchmark_margin = category_benchmark.get('avg_profit_margin', 42)
        average_price = metrics.average_price or 0
        
        # Base score on profit margin
        if profit_margin >= 60:
            margin_score = 100
        elif profit_margin >= 50:
            margin_score = 85
        elif profit_margin >= 40:
            margin_score = 70
        elif profit_margin >= 30:
            margin_score = 55
        elif profit_margin >= 20:
            margin_score = 40
        else:
            margin_score = 25
        
        # Price positioning score (assuming $30-150 is optimal range for medical devices)
        if 30 <= average_price <= 150:
            price_score = 100
        elif 20 <= average_price <= 200:
            price_score = 80
        elif 10 <= average_price <= 300:
            price_score = 60
        else:
            price_score = 40
        
        # Combined score (margin 80%, price positioning 20%)
        raw_score = (margin_score * 0.8) + (price_score * 0.2)
        
        # Performance level determination
        if raw_score >= 85:
            performance_level = "Excellent"
            key_drivers = ["High profit margins", "Optimal price positioning"]
        elif raw_score >= 70:
            performance_level = "Good"
            key_drivers = ["Good profitability", "Reasonable pricing"]
        elif raw_score >= 55:
            performance_level = "Average"
            key_drivers = ["Average margins", "Standard pricing"]
        else:
            performance_level = "Needs Improvement"
            key_drivers = ["Low profitability", "Pricing concerns"]
        
        # Benchmark comparison
        if profit_margin > benchmark_margin + 10:
            benchmark_comparison = "Significantly above category margins"
        elif profit_margin > benchmark_margin:
            benchmark_comparison = "Above category average"
        elif profit_margin < benchmark_margin - 10:
            benchmark_comparison = "Below category margins"
        else:
            benchmark_comparison = "About category average"
        
        return ComponentScore(
            component="Profitability",
            raw_score=raw_score,
            weighted_score=raw_score * SCORING_WEIGHTS['profitability'],
            weight=SCORING_WEIGHTS['profitability'],
            benchmark_comparison=benchmark_comparison,
            performance_level=performance_level,
            improvement_potential=100 - raw_score,
            key_drivers=key_drivers
        )
    
    @staticmethod
    def calculate_competitive_position_score(metrics: ProductMetrics, category_data: List[ProductMetrics]) -> ComponentScore:
        """Calculate competitive position score within category"""
        
        if not category_data or len(category_data) < 2:
            # Default score if no competitive data
            return ComponentScore(
                component="Competitive Position",
                raw_score=60,
                weighted_score=60 * SCORING_WEIGHTS['competitive_position'],
                weight=SCORING_WEIGHTS['competitive_position'],
                benchmark_comparison="Insufficient competitive data",
                performance_level="Average",
                improvement_potential=40,
                key_drivers=["No competitive comparison available"]
            )
        
        # Calculate percentiles across key metrics
        sales_percentile = PerformanceCalculator._calculate_percentile(
            metrics.sales_30d, [p.sales_30d for p in category_data]
        )
        
        return_rate_percentile = 100 - PerformanceCalculator._calculate_percentile(
            metrics.return_rate_30d, [p.return_rate_30d for p in category_data]
        )  # Invert because lower return rate is better
        
        rating_percentile = PerformanceCalculator._calculate_percentile(
            metrics.star_rating or 0, [p.star_rating or 0 for p in category_data]
        )
        
        # Combined competitive score
        raw_score = (sales_percentile * 0.4) + (return_rate_percentile * 0.3) + (rating_percentile * 0.3)
        
        # Performance level determination
        if raw_score >= 80:
            performance_level = "Excellent"
            key_drivers = ["Top performer in category", "Strong competitive position"]
        elif raw_score >= 60:
            performance_level = "Good"
            key_drivers = ["Above average in category", "Competitive"]
        elif raw_score >= 40:
            performance_level = "Average"
            key_drivers = ["Middle of category", "Room for improvement"]
        else:
            performance_level = "Needs Improvement"
            key_drivers = ["Below category average", "Competitive disadvantage"]
        
        benchmark_comparison = f"Top {100-raw_score:.0f}% of category"
        
        return ComponentScore(
            component="Competitive Position",
            raw_score=raw_score,
            weighted_score=raw_score * SCORING_WEIGHTS['competitive_position'],
            weight=SCORING_WEIGHTS['competitive_position'],
            benchmark_comparison=benchmark_comparison,
            performance_level=performance_level,
            improvement_potential=100 - raw_score,
            key_drivers=key_drivers
        )
    
    @staticmethod
    def _calculate_percentile(value: float, data_list: List[float]) -> float:
        """Calculate percentile rank of value in dataset"""
        if not data_list:
            return 50  # Default middle percentile
        
        data_array = np.array([x for x in data_list if x is not None])
        if len(data_array) == 0:
            return 50
        
        percentile = (np.sum(data_array <= value) / len(data_array)) * 100
        return percentile

class CompositeScoreCalculator:
    """Main class for calculating composite scores"""
    
    def __init__(self):
        self.performance_calculator = PerformanceCalculator()
    
    def calculate_product_score(self, product_data: Dict[str, Any], 
                              category_data: Optional[List[Dict[str, Any]]] = None,
                              ai_analysis_results: Optional[Dict[str, Any]] = None) -> CompositeScore:
        """Calculate complete composite score for a product"""
        
        try:
            # Convert to ProductMetrics object
            metrics = self._create_product_metrics(product_data, ai_analysis_results)
            
            # Get category benchmark
            category_benchmark = CATEGORY_BENCHMARKS.get(
                metrics.category, CATEGORY_BENCHMARKS['Default']
            )
            
            # Convert category data to ProductMetrics if available
            category_metrics = []
            if category_data:
                for cat_product in category_data:
                    cat_metrics = self._create_product_metrics(cat_product)
                    category_metrics.append(cat_metrics)
            
            # Calculate individual component scores
            component_scores = {}
            
            # Sales Performance
            component_scores['sales_performance'] = self.performance_calculator.calculate_sales_performance_score(
                metrics, category_benchmark
            )
            
            # Return Rate
            component_scores['return_rate'] = self.performance_calculator.calculate_return_rate_score(
                metrics, category_benchmark
            )
            
            # Customer Satisfaction
            component_scores['customer_satisfaction'] = self.performance_calculator.calculate_customer_satisfaction_score(
                metrics, category_benchmark
            )
            
            # Review Engagement
            component_scores['review_engagement'] = self.performance_calculator.calculate_review_engagement_score(
                metrics, category_benchmark
            )
            
            # Profitability
            component_scores['profitability'] = self.performance_calculator.calculate_profitability_score(
                metrics, category_benchmark
            )
            
            # Competitive Position
            component_scores['competitive_position'] = self.performance_calculator.calculate_competitive_position_score(
                metrics, category_metrics
            )
            
            # Calculate composite score
            composite_score = sum(score.weighted_score for score in component_scores.values())
            
            # Determine performance level and color
            performance_level, score_color = self._get_performance_level(composite_score)
            
            # Generate insights and recommendations
            improvement_priority = self._generate_improvement_priority(component_scores)
            risk_factors = self._identify_risk_factors(component_scores, metrics)
            strengths = self._identify_strengths(component_scores)
            
            # Calculate financial impact
            revenue_impact, potential_savings = self._calculate_financial_impact(metrics, component_scores)
            
            return CompositeScore(
                asin=metrics.asin,
                product_name=metrics.name,
                category=metrics.category,
                composite_score=round(composite_score, 1),
                performance_level=performance_level,
                score_color=score_color,
                component_scores=component_scores,
                improvement_priority=improvement_priority,
                risk_factors=risk_factors,
                strengths=strengths,
                revenue_impact=revenue_impact,
                potential_savings=potential_savings
            )
            
        except Exception as e:
            logger.error(f"Error calculating composite score for {product_data.get('asin', 'unknown')}: {str(e)}")
            raise
    
    def calculate_category_scores(self, products_data: List[Dict[str, Any]], 
                                category: Optional[str] = None) -> Dict[str, CompositeScore]:
        """Calculate scores for multiple products in a category"""
        
        category_scores = {}
        
        # Filter by category if specified
        if category:
            products_data = [p for p in products_data if p.get('category') == category]
        
        # Calculate individual scores
        for product_data in products_data:
            try:
                score = self.calculate_product_score(product_data, products_data)
                category_scores[product_data['asin']] = score
            except Exception as e:
                logger.error(f"Failed to calculate score for {product_data.get('asin')}: {str(e)}")
        
        return category_scores
    
    def generate_portfolio_insights(self, scores: Dict[str, CompositeScore]) -> Dict[str, Any]:
        """Generate portfolio-level insights and recommendations"""
        
        if not scores:
            return {"error": "No scores available for analysis"}
        
        score_values = [score.composite_score for score in scores.values()]
        
        portfolio_insights = {
            'summary': {
                'total_products': len(scores),
                'average_score': round(np.mean(score_values), 1),
                'median_score': round(np.median(score_values), 1),
                'score_range': {
                    'min': round(min(score_values), 1),
                    'max': round(max(score_values), 1)
                }
            },
            'performance_distribution': {},
            'top_performers': [],
            'priority_improvements': [],
            'category_analysis': {},
            'risk_assessment': {}
        }
        
        # Performance distribution
        for threshold_name, threshold_data in PERFORMANCE_THRESHOLDS.items():
            count = sum(1 for score in score_values if score >= threshold_data['min_score'])
            portfolio_insights['performance_distribution'][threshold_name] = {
                'count': count,
                'percentage': round((count / len(scores)) * 100, 1)
            }
        
        # Top performers (top 20% or at least top 3)
        sorted_scores = sorted(scores.values(), key=lambda x: x.composite_score, reverse=True)
        top_count = max(3, len(sorted_scores) // 5)
        portfolio_insights['top_performers'] = [
            {
                'asin': score.asin,
                'name': score.product_name,
                'score': score.composite_score,
                'strengths': score.strengths[:2]  # Top 2 strengths
            }
            for score in sorted_scores[:top_count]
        ]
        
        # Priority improvements (bottom 20% or lowest 3)
        bottom_count = max(3, len(sorted_scores) // 5)
        priority_products = sorted_scores[-bottom_count:]
        portfolio_insights['priority_improvements'] = [
            {
                'asin': score.asin,
                'name': score.product_name,
                'score': score.composite_score,
                'top_priority': score.improvement_priority[0] if score.improvement_priority else 'General improvement needed',
                'potential_impact': score.revenue_impact
            }
            for score in priority_products
        ]
        
        # Category analysis
        category_data = defaultdict(list)
        for score in scores.values():
            category_data[score.category].append(score.composite_score)
        
        for category, scores_list in category_data.items():
            portfolio_insights['category_analysis'][category] = {
                'average_score': round(np.mean(scores_list), 1),
                'product_count': len(scores_list),
                'performance': 'Above Average' if np.mean(scores_list) > np.mean(score_values) else 'Below Average'
            }
        
        return portfolio_insights
    
    def _create_product_metrics(self, product_data: Dict[str, Any], 
                              ai_analysis: Optional[Dict[str, Any]] = None) -> ProductMetrics:
        """Create ProductMetrics object from product data"""
        
        # Extract sentiment score from AI analysis if available
        sentiment_score = None
        if ai_analysis and 'review_analysis' in ai_analysis:
            # Extract sentiment from AI analysis results
            sentiment_score = 75  # Placeholder - would extract from actual AI results
        
        metrics = ProductMetrics(
            asin=product_data.get('asin', ''),
            name=product_data.get('name', 'Unknown Product'),
            category=product_data.get('category', 'Other'),
            sales_30d=product_data.get('sales_30d', 0),
            sales_365d=product_data.get('sales_365d'),
            returns_30d=product_data.get('returns_30d', 0),
            returns_365d=product_data.get('returns_365d'),
            star_rating=product_data.get('star_rating'),
            total_reviews=product_data.get('total_reviews'),
            average_price=product_data.get('average_price'),
            cost_per_unit=product_data.get('cost_per_unit'),
            review_sentiment_score=sentiment_score
        )
        
        # Calculate derived metrics
        metrics.calculate_derived_metrics()
        
        return metrics
    
    def _get_performance_level(self, score: float) -> Tuple[str, str]:
        """Get performance level and color for score"""
        
        for threshold_name, threshold_data in PERFORMANCE_THRESHOLDS.items():
            if score >= threshold_data['min_score']:
                return threshold_data['label'], threshold_data['color']
        
        return 'Critical', '#DC2626'
    
    def _generate_improvement_priority(self, component_scores: Dict[str, ComponentScore]) -> List[str]:
        """Generate prioritized list of improvement areas"""
        
        # Sort components by improvement potential and weight
        improvements = []
        for component, score in component_scores.items():
            impact_score = score.improvement_potential * score.weight
            improvements.append((component, impact_score, score.performance_level))
        
        # Sort by impact score (descending)
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        # Return prioritized list with focus on lowest performing areas
        priority_list = []
        for component, impact, performance in improvements:
            if performance in ['Critical', 'Needs Improvement']:
                priority_list.append(f"{component} - {performance} (High Impact)")
            elif performance == 'Average':
                priority_list.append(f"{component} - {performance} (Medium Impact)")
        
        return priority_list[:5]  # Top 5 priorities
    
    def _identify_risk_factors(self, component_scores: Dict[str, ComponentScore], 
                             metrics: ProductMetrics) -> List[str]:
        """Identify key risk factors for the product"""
        
        risks = []
        
        # High return rate risk
        if metrics.return_rate_30d > 8:
            risks.append(f"High return rate ({metrics.return_rate_30d:.1f}%) - Customer satisfaction issues")
        
        # Low sales volume risk
        if metrics.sales_30d < 30:
            risks.append("Low sales volume - Market demand or visibility issues")
        
        # Poor customer satisfaction risk
        if metrics.star_rating and metrics.star_rating < 3.8:
            risks.append(f"Low star rating ({metrics.star_rating:.1f}) - Product quality concerns")
        
        # Low profitability risk
        if metrics.profit_margin and metrics.profit_margin < 25:
            risks.append(f"Low profit margin ({metrics.profit_margin:.1f}%) - Financial sustainability risk")
        
        # Low review engagement
        if metrics.total_reviews and metrics.total_reviews < 20:
            risks.append("Low review volume - Limited social proof and feedback")
        
        return risks[:5]  # Top 5 risks
    
    def _identify_strengths(self, component_scores: Dict[str, ComponentScore]) -> List[str]:
        """Identify key strengths for the product"""
        
        strengths = []
        
        for component, score in component_scores.items():
            if score.performance_level in ['Excellent', 'Good']:
                key_driver = score.key_drivers[0] if score.key_drivers else "Strong performance"
                strengths.append(f"{component}: {key_driver}")
        
        return strengths[:5]  # Top 5 strengths
    
    def _calculate_financial_impact(self, metrics: ProductMetrics, 
                                  component_scores: Dict[str, ComponentScore]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate potential financial impact of improvements"""
        
        if not metrics.average_price or not metrics.sales_30d:
            return None, None
        
        monthly_revenue = metrics.sales_30d * metrics.average_price
        
        # Calculate potential revenue impact from improvements
        revenue_impact = 0
        
        # Return rate improvement impact
        return_score = component_scores.get('return_rate')
        if return_score and return_score.performance_level in ['Critical', 'Needs Improvement']:
            # Estimate 10-20% sales increase from return rate improvement
            revenue_impact += monthly_revenue * 0.15
        
        # Customer satisfaction improvement impact
        satisfaction_score = component_scores.get('customer_satisfaction')
        if satisfaction_score and satisfaction_score.performance_level in ['Critical', 'Needs Improvement']:
            # Estimate 5-15% sales increase from satisfaction improvement
            revenue_impact += monthly_revenue * 0.10
        
        # Calculate potential cost savings from return reduction
        potential_savings = 0
        if metrics.returns_30d > 0:
            # Estimate $20 cost per return (processing, shipping, etc.)
            current_return_cost = metrics.returns_30d * 20
            # Potential 30-50% reduction in returns with optimization
            potential_savings = current_return_cost * 0.4
        
        return round(revenue_impact, 2) if revenue_impact > 0 else None, round(potential_savings, 2) if potential_savings > 0 else None

# Export functionality
class ScoreExporter:
    """Export scoring results to various formats"""
    
    @staticmethod
    def export_to_dataframe(scores: Dict[str, CompositeScore]) -> pd.DataFrame:
        """Export scores to pandas DataFrame"""
        
        export_data = []
        
        for asin, score in scores.items():
            row = {
                'ASIN': score.asin,
                'Product_Name': score.product_name,
                'Category': score.category,
                'Composite_Score': score.composite_score,
                'Performance_Level': score.performance_level,
                'Sales_Performance': score.component_scores['sales_performance'].raw_score,
                'Return_Rate_Score': score.component_scores['return_rate'].raw_score,
                'Customer_Satisfaction': score.component_scores['customer_satisfaction'].raw_score,
                'Review_Engagement': score.component_scores['review_engagement'].raw_score,
                'Profitability': score.component_scores['profitability'].raw_score,
                'Competitive_Position': score.component_scores['competitive_position'].raw_score,
                'Top_Priority': score.improvement_priority[0] if score.improvement_priority else '',
                'Revenue_Impact': score.revenue_impact,
                'Potential_Savings': score.potential_savings,
                'Risk_Count': len(score.risk_factors) if score.risk_factors else 0,
                'Strength_Count': len(score.strengths) if score.strengths else 0
            }
            
            export_data.append(row)
        
        return pd.DataFrame(export_data)
    
    @staticmethod
    def export_detailed_report(scores: Dict[str, CompositeScore], 
                             portfolio_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Export comprehensive scoring report"""
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_products': len(scores),
                'scoring_version': '2.0'
            },
            'portfolio_summary': portfolio_insights,
            'individual_scores': {asin: score.to_dict() for asin, score in scores.items()},
            'scoring_methodology': {
                'weights': SCORING_WEIGHTS,
                'benchmarks': CATEGORY_BENCHMARKS,
                'thresholds': PERFORMANCE_THRESHOLDS
            }
        }

# Main scoring system class
class CompositeScoring:
    """Main class for the composite scoring system"""
    
    def __init__(self):
        self.calculator = CompositeScoreCalculator()
        self.exporter = ScoreExporter()
    
    def score_single_product(self, product_data: Dict[str, Any], 
                           category_data: Optional[List[Dict[str, Any]]] = None,
                           ai_analysis: Optional[Dict[str, Any]] = None) -> CompositeScore:
        """Score a single product"""
        return self.calculator.calculate_product_score(product_data, category_data, ai_analysis)
    
    def score_product_portfolio(self, products_data: List[Dict[str, Any]]) -> Tuple[Dict[str, CompositeScore], Dict[str, Any]]:
        """Score entire product portfolio"""
        scores = self.calculator.calculate_category_scores(products_data)
        insights = self.calculator.generate_portfolio_insights(scores)
        return scores, insights
    
    def get_scoring_weights(self) -> Dict[str, float]:
        """Get current scoring weights"""
        return SCORING_WEIGHTS.copy()
    
    def get_category_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get category benchmarks"""
        return CATEGORY_BENCHMARKS.copy()
    
    def export_scores(self, scores: Dict[str, CompositeScore], 
                     format_type: str = 'dataframe') -> Any:
        """Export scores in specified format"""
        
        if format_type == 'dataframe':
            return self.exporter.export_to_dataframe(scores)
        elif format_type == 'detailed_report':
            insights = self.calculator.generate_portfolio_insights(scores)
            return self.exporter.export_detailed_report(scores, insights)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

# Export main classes
__all__ = [
    'CompositeScoring',
    'CompositeScore', 
    'ComponentScore',
    'ProductMetrics',
    'SCORING_WEIGHTS',
    'CATEGORY_BENCHMARKS',
    'PERFORMANCE_THRESHOLDS'
]
