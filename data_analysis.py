import pandas as pd
import numpy as np
from collections import Counter
import json
import re
import logging
import io

logger = logging.getLogger(__name__)

# Check if visualization dependencies are available
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    logger.info("Plotly is available for visualization")
except ImportError:
    logger.warning("Plotly is not available for visualization")

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default if divisor is zero."""
    try:
        if b == 0:
            return default
        return a / b
    except Exception as e:
        logger.error(f"Error in safe_divide: {str(e)}")
        return default

def calculate_product_metrics(product_data):
    """Calculate key metrics from product data."""
    metrics = {}
    
    # Basic metrics
    metrics['total_sales_30d'] = product_data.get('sales_30d', 0)
    metrics['total_returns_30d'] = product_data.get('returns_30d', 0)
    
    # Calculate return rate
    if metrics['total_sales_30d'] > 0:
        metrics['return_rate_30d'] = (metrics['total_returns_30d'] / metrics['total_sales_30d']) * 100
    else:
        metrics['return_rate_30d'] = 0
    
    # 365-day metrics if available
    if product_data.get('sales_365d') is not None:
        metrics['total_sales_365d'] = product_data.get('sales_365d', 0)
        metrics['total_returns_365d'] = product_data.get('returns_365d', 0)
        
        if metrics['total_sales_365d'] > 0:
            metrics['return_rate_365d'] = (metrics['total_returns_365d'] / metrics['total_sales_365d']) * 100
        else:
            metrics['return_rate_365d'] = 0
    
    # Star rating if available
    metrics['star_rating'] = product_data.get('star_rating')
    metrics['total_reviews'] = product_data.get('total_reviews')
    
    return metrics

def analyze_return_reasons(returns_data):
    """Analyze return reasons from structured data."""
    if not returns_data or len(returns_data) == 0:
        return {
            "total_returns": 0,
            "reason_counts": {},
            "common_phrases": {}
        }
    
    # Count return reasons
    if isinstance(returns_data, pd.DataFrame):
        if 'return_reason' in returns_data.columns:
            reason_counts = returns_data['return_reason'].value_counts().to_dict()
        else:
            reason_counts = {}
    else:
        # If it's a list of dictionaries
        reasons = [item.get('return_reason', '') for item in returns_data if item.get('return_reason')]
        reason_counts = dict(Counter(reasons))
    
    # Extract common phrases from comments
    comments = []
    if isinstance(returns_data, pd.DataFrame):
        if 'buyer_comment' in returns_data.columns:
            comments = returns_data['buyer_comment'].dropna().tolist()
    else:
        comments = [item.get('buyer_comment', '') for item in returns_data if item.get('buyer_comment')]
    
    # Analyze comments to find common phrases
    comment_text = ' '.join(comments).lower()
    
    # Define phrases to look for
    phrase_patterns = [
        r'too small', r'not stable', r'unstable', r'difficult to assemble', 
        r'hard to put together', r'quality', r'damaged', r'defective',
        r'uncomfortable', r'doesn\'t work', r'broke', r'fell apart',
        r'wrong size', r'too big', r'heavy', r'lightweight'
    ]
    
    # Count occurrences
    common_phrases = {}
    for pattern in phrase_patterns:
        count = len(re.findall(pattern, comment_text))
        if count > 0:
            common_phrases[pattern] = count
    
    return {
        "total_returns": len(returns_data),
        "reason_counts": reason_counts,
        "common_phrases": common_phrases
    }

def analyze_reviews(reviews_data):
    """Analyze review content and ratings."""
    if not reviews_data or len(reviews_data) == 0:
        return {
            "total_reviews": 0,
            "average_rating": None,
            "rating_distribution": {},
            "sentiment": {},
            "common_topics": {}
        }
    
    # Initialize counters
    total_reviews = 0
    ratings_sum = 0
    rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Process reviews based on data structure
    if isinstance(reviews_data, pd.DataFrame):
        if 'rating' in reviews_data.columns:
            # Count ratings
            valid_ratings = reviews_data['rating'].dropna()
            total_reviews = len(valid_ratings)
            ratings_sum = valid_ratings.sum()
            rating_distribution = valid_ratings.value_counts().to_dict()
            
            # Simple sentiment analysis based on ratings
            sentiment = {
                "positive": len(reviews_data[reviews_data['rating'] >= 4]),
                "neutral": len(reviews_data[reviews_data['rating'] == 3]),
                "negative": len(reviews_data[reviews_data['rating'] <= 2])
            }
    else:
        # If it's a list structure
        ratings = []
        for review in reviews_data:
            if isinstance(review, dict) and 'rating' in review:
                rating = review['rating']
                if rating is not None:
                    ratings.append(rating)
                    rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        total_reviews = len(ratings)
        ratings_sum = sum(ratings) if ratings else 0
        
        # Simple sentiment analysis
        sentiment = {
            "positive": sum(1 for r in ratings if r >= 4),
            "neutral": sum(1 for r in ratings if r == 3),
            "negative": sum(1 for r in ratings if r <= 2)
        }
    
    # Calculate average rating
    average_rating = ratings_sum / total_reviews if total_reviews > 0 else None
    
    # Analyze review text for common topics
    common_topics = {}
    review_text = ''
    
    if isinstance(reviews_data, pd.DataFrame):
        if 'review_text' in reviews_data.columns:
            review_text = ' '.join(reviews_data['review_text'].dropna().astype(str))
    else:
        review_texts = [review.get('review_text', '') for review in reviews_data if isinstance(review, dict)]
        review_text = ' '.join(review_texts)
    
    # Define topics to look for
    topic_patterns = [
        r'quality', r'comfort', r'size', r'price', r'value', 
        r'durability', r'assembly', r'stability', r'weight',
        r'easy to use', r'difficult', r'recommend', 
        r'wheels', r'brakes', r'seat', r'folding', 
        r'lightweight', r'heavy', r'instruction'
    ]
    
    # Count occurrences
    for pattern in topic_patterns:
        count = len(re.findall(pattern, review_text.lower()))
        if count > 0:
            common_topics[pattern] = count
    
    return {
        "total_reviews": total_reviews,
        "average_rating": average_rating,
        "rating_distribution": rating_distribution,
        "sentiment": sentiment,
        "common_topics": common_topics
    }

def create_return_reasons_chart(return_analysis):
    """Create a pie chart of return reasons."""
    if not PLOTLY_AVAILABLE:
        return None
        
    reason_counts = return_analysis.get('reason_counts', {})
    if not reason_counts:
        return None
    
    # Sort and get top reasons
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    top_reasons = dict(sorted_reasons[:6])  # Top 6 reasons
    
    # If there are more, add an "Other" category
    if len(sorted_reasons) > 6:
        other_count = sum(count for _, count in sorted_reasons[6:])
        top_reasons["Other"] = other_count
    
    # Create pie chart
    fig = px.pie(
        values=list(top_reasons.values()),
        names=list(top_reasons.keys()),
        title="Return Reasons",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Return Reason",
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

def create_sentiment_chart(review_analysis):
    """Create a sentiment chart from review analysis."""
    if not PLOTLY_AVAILABLE:
        return None
        
    sentiment = review_analysis.get('sentiment', {})
    if not sentiment:
        return None
    
    # Create data for chart
    sentiment_df = pd.DataFrame({
        'Sentiment': list(sentiment.keys()),
        'Count': list(sentiment.values())
    })
    
    # Custom color map
    color_map = {'positive': 'green', 'neutral': 'gold', 'negative': 'red'}
    
    # Create bar chart
    fig = px.bar(
        sentiment_df, 
        x='Sentiment', 
        y='Count',
        title="Review Sentiment",
        color='Sentiment',
        color_discrete_map=color_map
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Number of Reviews",
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

def create_rating_distribution_chart(review_analysis):
    """Create a chart showing rating distribution."""
    if not PLOTLY_AVAILABLE:
        return None
        
    rating_dist = review_analysis.get('rating_distribution', {})
    if not rating_dist:
        return None
    
    # Create data for chart
    rating_df = pd.DataFrame({
        'Rating': [f"{k} ★" for k in rating_dist.keys()],
        'Count': list(rating_dist.values())
    })
    
    # Sort by rating
    rating_df['Sort'] = rating_df['Rating'].str.extract(r'(\d)').astype(int)
    rating_df = rating_df.sort_values('Sort', ascending=False)
    
    # Create color map
    colors = {
        "5 ★": "darkgreen",
        "4 ★": "lightgreen", 
        "3 ★": "gold",
        "2 ★": "orange",
        "1 ★": "red"
    }
    
    # Create bar chart
    fig = px.bar(
        rating_df,
        x='Rating',
        y='Count',
        title="Rating Distribution",
        color='Rating',
        color_discrete_map=colors
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Number of Reviews",
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

def create_topics_chart(review_analysis):
    """Create a chart showing common topics in reviews."""
    if not PLOTLY_AVAILABLE:
        return None
        
    topics = review_analysis.get('common_topics', {})
    if not topics:
        return None
    
    # Sort and get top topics
    sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
    top_topics = dict(sorted_topics[:10])  # Top 10 topics
    
    # Create data for chart
    topics_df = pd.DataFrame({
        'Topic': list(top_topics.keys()),
        'Mentions': list(top_topics.values())
    })
    
    # Create bar chart
    fig = px.bar(
        topics_df,
        x='Mentions',
        y='Topic',
        title="Common Topics in Reviews",
        orientation='h',
        color='Mentions',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Number of Mentions",
        yaxis_title=None,
        margin=dict(t=50, b=0, l=0, r=30)
    )
    
    return fig

def generate_report_data(product_info, returns_analysis, reviews_analysis):
    """Generate structured data for a report."""
    report_data = {
        "product_info": {
            "name": product_info.get('name', 'Unknown Product'),
            "asin": product_info.get('asin', 'Unknown ASIN'),
            "sku": product_info.get('sku', 'N/A'),
            "category": product_info.get('category', 'Medical Device')
        },
        "sales_metrics": {
            "sales_30d": product_info.get('sales_30d', 0),
            "returns_30d": product_info.get('returns_30d', 0),
            "return_rate_30d": product_info.get('return_rate_30d', 0),
            "sales_365d": product_info.get('sales_365d'),
            "returns_365d": product_info.get('returns_365d'),
            "return_rate_365d": product_info.get('return_rate_365d')
        },
        "review_metrics": {
            "total_reviews": reviews_analysis.get('total_reviews', 0),
            "average_rating": reviews_analysis.get('average_rating'),
            "rating_distribution": reviews_analysis.get('rating_distribution', {}),
            "sentiment": reviews_analysis.get('sentiment', {})
        },
        "return_analysis": {
            "total_returns": returns_analysis.get('total_returns', 0),
            "reason_counts": returns_analysis.get('reason_counts', {}),
            "common_phrases": returns_analysis.get('common_phrases', {})
        },
        "review_analysis": {
            "common_topics": reviews_analysis.get('common_topics', {})
        }
    }
    
    return report_data

def create_excel_report(report_data):
    """Create Excel report from structured report data."""
    # Create a writer object
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Product Info Sheet
        product_df = pd.DataFrame([report_data['product_info']])
        product_df.to_excel(writer, sheet_name='Product Info', index=False)
        
        # Metrics Sheet
        metrics = {
            **report_data['sales_metrics'],
            'total_reviews': report_data['review_metrics']['total_reviews'],
            'average_rating': report_data['review_metrics']['average_rating']
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Return Reasons Sheet
        if report_data['return_analysis']['reason_counts']:
            reasons_df = pd.DataFrame([
                {'Reason': k, 'Count': v} 
                for k, v in report_data['return_analysis']['reason_counts'].items()
            ])
            reasons_df.to_excel(writer, sheet_name='Return Reasons', index=False)
        
        # Rating Distribution Sheet
        if report_data['review_metrics']['rating_distribution']:
            ratings_df = pd.DataFrame([
                {'Rating': k, 'Count': v} 
                for k, v in report_data['review_metrics']['rating_distribution'].items()
            ])
            ratings_df.to_excel(writer, sheet_name='Rating Distribution', index=False)
        
        # Common Topics Sheet
        if report_data['review_analysis']['common_topics']:
            topics_df = pd.DataFrame([
                {'Topic': k, 'Mentions': v} 
                for k, v in report_data['review_analysis']['common_topics'].items()
            ])
            topics_df.to_excel(writer, sheet_name='Common Topics', index=False)
        
        # Format workbook
        workbook = writer.book
        
        # Add a header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply the header format to each sheet
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col_num, value in enumerate(pd.DataFrame([{}]).columns.values):
                worksheet.write(0, col_num, value, header_format)
    
    # Return the Excel binary data
    output.seek(0)
    return output
