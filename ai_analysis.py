"""
AI Analysis module for Amazon Medical Device Listing Optimizer

This module contains functions for performing AI-based analysis of
Amazon medical device product reviews, return reasons, listings, and other data.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing OpenAI for SDK fallback if needed
try:
    import openai
    from openai import OpenAI  # Import OpenAI client for newer SDK
    HAS_OPENAI = True
except ImportError:
    logger.warning("OpenAI module not available")
    HAS_OPENAI = False

# Amazon-specific keywords for listing optimization
AMAZON_KEYWORDS = {
    'listing_quality': ['title', 'bullet', 'description', 'feature', 'benefit', 'keyword', 'search term'],
    'customer_concerns': ['size', 'fit', 'quality', 'durability', 'comfort', 'broken', 'difficult'],
    'competitive_factors': ['price', 'value', 'better', 'cheaper', 'competitor', 'alternative', 'similar'],
    'purchase_factors': ['decision', 'purchase', 'bought', 'chose', 'recommend', 'satisfied', 'happy'],
    'return_factors': ['return', 'refund', 'sent back', 'disappointed', 'expected', 'not as described'],
    'image_factors': ['picture', 'photo', 'image', 'looks different', 'see', 'shown', 'display'],
    'ecommerce_specific': ['shipping', 'packaging', 'arrived', 'delivery', 'box', 'amazon']
}

# Medical device specific keywords for analysis
MEDICAL_KEYWORDS = {
    'safety': ['safe', 'unsafe', 'injury', 'hurt', 'hazard', 'danger', 'risk', 'warning', 'precaution'],
    'efficacy': ['effective', 'ineffective', 'work', 'doesn\'t work', 'helped', 'improve', 'benefit'],
    'comfort': ['comfortable', 'uncomfortable', 'pain', 'painful', 'soft', 'hard', 'irritate', 'sore'],
    'durability': ['durable', 'broke', 'broken', 'sturdy', 'flimsy', 'quality', 'lasting', 'cheaply made'],
    'usability': ['easy', 'difficult', 'complicated', 'simple', 'intuitive', 'confusing', 'instructions'],
    'fit': ['fit', 'size', 'small', 'large', 'tight', 'loose', 'adjustable', 'measurement'],
    'regulatory': ['fda', 'approved', 'certified', 'medical grade', 'compliant', 'regulation', 'complies']
}

def get_api_key():
    """
    Get the OpenAI API key from various sources
    
    Returns:
    - API key string or None
    """
    # First, try to get from streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "openai_api_key" in st.secrets:
            return st.secrets["openai_api_key"]
    except (ImportError, AttributeError):
        pass
    
    # Then try environment variable as fallback
    return os.environ.get("OPENAI_API_KEY")

def call_openai_api(messages, model="gpt-4o", temperature=0.1, max_tokens=1000):
    """
    Make a direct call to the OpenAI API
    
    Parameters:
    - messages: List of message dictionaries
    - model: Model name to use
    - temperature: Temperature setting
    - max_tokens: Maximum tokens in the response
    
    Returns:
    - API response text or error message
    """
    api_key = get_api_key()
    
    if not api_key:
        logger.error("OpenAI API key not found")
        return None
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return None

def analyze_reviews_with_ai(reviews):
    """
    Analyze a list of product reviews using AI
    
    Parameters:
    - reviews: List of dictionaries containing review data
    
    Returns:
    - Dictionary with analysis results
    """
    try:
        # Prepare review text
        review_texts = []
        for review in reviews[:20]:  # Limit to 20 reviews to avoid token limits
            rating = review.get('rating', 'Unknown')
            text = review.get('review_text', '')
            review_texts.append(f"Rating: {rating} - {text}")
        
        review_content = "\n\n".join(review_texts)
        
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."
        user_prompt = f"""Analyze the following Amazon medical device reviews. 
        Identify key issues, patterns, and insights related to:
        1. Overall customer sentiment
        2. Product quality and durability concerns
        3. Fit, comfort and usability issues
        4. Listing accuracy (how well the listing matched the actual product)
        5. Features customers liked most
        6. Features customers disliked most
        7. Common questions or confusion points
        
        For each category, provide specific examples from the reviews.
        Then provide an overall assessment of how the listing could be improved based on this feedback.
        
        Reviews:
        {review_content}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        analysis = call_openai_api(messages)
        
        if not analysis:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract themes and issues from the analysis
        sentiment = extract_theme_from_analysis(analysis, "Overall customer sentiment")
        quality_issues = extract_theme_from_analysis(analysis, "Product quality and durability concerns")
        fit_issues = extract_theme_from_analysis(analysis, "Fit, comfort and usability issues")
        listing_accuracy = extract_theme_from_analysis(analysis, "Listing accuracy")
        liked_features = extract_theme_from_analysis(analysis, "Features customers liked most")
        disliked_features = extract_theme_from_analysis(analysis, "Features customers disliked most")
        confusion_points = extract_theme_from_analysis(analysis, "Common questions or confusion points")
        
        return {
            "success": True,
            "full_analysis": analysis,
            "themes": {
                "sentiment": sentiment,
                "quality_issues": quality_issues,
                "fit_issues": fit_issues,
                "listing_accuracy": listing_accuracy,
                "liked_features": liked_features,
                "disliked_features": disliked_features,
                "confusion_points": confusion_points
            }
        }
    except Exception as e:
        logger.error(f"Error in AI review analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def extract_theme_from_analysis(analysis_text, theme_name):
    """
    Extract specific theme sections from the AI analysis text
    
    Parameters:
    - analysis_text: The full text of the AI analysis
    - theme_name: The name of the theme to extract
    
    Returns:
    - Extracted text for the theme or None if not found
    """
    # Look for sections that begin with the theme name
    pattern = rf"{theme_name}:?(.*?)(?:\n\d+\.|$)"
    match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    return None

def analyze_returns_with_ai(return_reasons):
    """
    Analyze a list of return reasons using AI
    
    Parameters:
    - return_reasons: List of return reason strings
    
    Returns:
    - Dictionary with analysis results
    """
    try:
        # Prepare return reason text
        return_text = "\n".join([f"- {reason}" for reason in return_reasons[:30]])  # Limit to 30 returns
        
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."
        user_prompt = f"""Analyze the following return reasons for an Amazon medical device product.
        Categorize the returns into these groups:
        1. Product defects/quality issues
        2. Size/fit issues
        3. Comfort issues
        4. Performance/efficacy issues
        5. Listing accuracy problems (images, description)
        6. Preference/expectation mismatch
        7. User error or misunderstanding
        
        For each category, provide the count and percentage of returns, with specific examples.
        Then provide specific recommendations for how to improve the product listing to reduce these types of returns.
        
        Return Reasons:
        {return_text}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        analysis = call_openai_api(messages)
        
        if not analysis:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract categories from the analysis
        defect_issues = extract_theme_from_analysis(analysis, "Product defects/quality issues")
        size_issues = extract_theme_from_analysis(analysis, "Size/fit issues")
        comfort_issues = extract_theme_from_analysis(analysis, "Comfort issues")
        performance_issues = extract_theme_from_analysis(analysis, "Performance/efficacy issues")
        listing_issues = extract_theme_from_analysis(analysis, "Listing accuracy problems")
        expectation_issues = extract_theme_from_analysis(analysis, "Preference/expectation mismatch")
        user_error = extract_theme_from_analysis(analysis, "User error or misunderstanding")
        
        return {
            "success": True,
            "full_analysis": analysis,
            "categories": {
                "defects": defect_issues,
                "size": size_issues,
                "comfort": comfort_issues,
                "performance": performance_issues,
                "listing_accuracy": listing_issues,
                "expectations": expectation_issues,
                "user_error": user_error
            }
        }
    except Exception as e:
        logger.error(f"Error in return reason analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def analyze_listing_optimization(product_info):
    """
    Analyze and provide recommendations for Amazon listing optimization
    
    Parameters:
    - product_info: Dictionary with product details
    
    Returns:
    - Dictionary with optimization recommendations
    """
    try:
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."
        user_prompt = f"""Analyze this Amazon medical device product and provide actionable 
        recommendations to optimize the listing for higher conversion rates and reduced returns.
        
        Product name: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        Description: {product_info.get('description', '')}
        30-Day Return Rate: {product_info.get('return_rate_30d', 'N/A')}%
        Star Rating: {product_info.get('star_rating', 'N/A')}
        
        Provide the following:
        1. Title optimization recommendations (for better CTR and keyword relevance)
        2. Bullet points strategy (highlight key benefits and features)
        3. Description improvements (storytelling and problem-solution format)
        4. Image optimization suggestions (specific shots and demonstrations needed)
        5. Keywords to target for this specific medical device
        6. A+ Content recommendations (if applicable)
        7. Common customer questions to address proactively
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        analysis = call_openai_api(messages, max_tokens=1200)
        
        if not analysis:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract sections from the analysis
        title_recommendations = extract_theme_from_analysis(analysis, "Title optimization recommendations")
        bullet_strategy = extract_theme_from_analysis(analysis, "Bullet points strategy")
        description_improvements = extract_theme_from_analysis(analysis, "Description improvements")
        image_suggestions = extract_theme_from_analysis(analysis, "Image optimization suggestions")
        keywords = extract_theme_from_analysis(analysis, "Keywords to target")
        a_plus_recommendations = extract_theme_from_analysis(analysis, "A+ Content recommendations")
        common_questions = extract_theme_from_analysis(analysis, "Common customer questions")
        
        return {
            "success": True,
            "full_analysis": analysis,
            "sections": {
                "title_recommendations": title_recommendations,
                "bullet_strategy": bullet_strategy,
                "description_improvements": description_improvements,
                "image_suggestions": image_suggestions,
                "keywords": keywords,
                "a_plus_recommendations": a_plus_recommendations,
                "common_questions": common_questions
            }
        }
    except Exception as e:
        logger.error(f"Error in listing optimization analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_improvement_recommendations(product_info, reviews_data, returns_data, sales_data=None):
    """
    Generate AI-powered recommendations for product improvements
    
    Parameters:
    - product_info: Dictionary with product details
    - reviews_data: List of review dictionaries
    - returns_data: List of return reason dictionaries
    - sales_data: Optional sales and returns metrics
    
    Returns:
    - Dictionary with recommendations
    """
    try:
        # Prepare review text
        review_texts = []
        for review in reviews_data[:15]:  # Limit to 15 reviews
            rating = review.get('rating', 'Unknown')
            text = review.get('review_text', '')
            review_texts.append(f"Rating: {rating} - {text}")
        
        review_content = "\n".join(review_texts)
        
        # Prepare return reason text
        return_reasons = []
        for ret in returns_data[:15]:  # Limit to 15 return reasons
            reason = ret.get('return_reason', '')
            return_reasons.append(f"- {reason}")
        
        return_content = "\n".join(return_reasons)
        
        # Calculate return rate
        return_rate = None
        if sales_data:
            sales = sales_data.get('sales_30d', 0)
            returns = sales_data.get('returns_30d', 0)
            if sales > 0:
                return_rate = (returns / sales) * 100
        
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist who helps medical device companies maximize their e-commerce sales and minimize returns."
        user_prompt = f"""As an Amazon listing optimization specialist for medical devices, analyze the following 
        product data and provide actionable recommendations:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        """
        
        if return_rate is not None:
            user_prompt += f"Return Rate: {return_rate:.2f}%\n"
        
        user_prompt += f"""
        Customer Reviews:
        {review_content}
        
        Return Reasons:
        {return_content}
        
        Please provide:
        1. Top 3-5 product improvement recommendations based on customer feedback
        2. Specific listing improvements to reduce return rate
        3. New or improved image recommendations based on customer confusion
        4. Keywords and features to emphasize based on positive reviews
        5. Features that need better explanation in the listing
        6. Competitive differentiators to highlight more prominently
        
        For each recommendation, include specific actionable steps for implementation.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        recommendations = call_openai_api(messages, max_tokens=1200, temperature=0.2)
        
        if not recommendations:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract sections from the recommendations
        product_improvements = extract_theme_from_analysis(recommendations, "Top 3-5 product improvement recommendations")
        listing_improvements = extract_theme_from_analysis(recommendations, "Specific listing improvements to reduce return rate")
        image_recommendations = extract_theme_from_analysis(recommendations, "New or improved image recommendations")
        positive_keywords = extract_theme_from_analysis(recommendations, "Keywords and features to emphasize")
        explanation_needs = extract_theme_from_analysis(recommendations, "Features that need better explanation")
        competitive_differentiators = extract_theme_from_analysis(recommendations, "Competitive differentiators")
        
        return {
            "success": True,
            "full_recommendations": recommendations,
            "sections": {
                "product_improvements": product_improvements,
                "listing_improvements": listing_improvements,
                "image_recommendations": image_recommendations,
                "positive_keywords": positive_keywords,
                "explanation_needs": explanation_needs,
                "competitive_differentiators": competitive_differentiators
            }
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {"success": False, "error": str(e)}

def analyze_competitive_positioning(product_info, competitors_data):
    """
    Analyze competitive positioning and provide recommendations
    
    Parameters:
    - product_info: Dictionary with product details
    - competitors_data: List of competitor dictionaries
    
    Returns:
    - Dictionary with competitive analysis
    """
    try:
        # Prepare competitor text
        competitor_text = "\n".join([f"- {comp.get('name', '')} (ASIN: {comp.get('asin', 'Unknown')})" 
                                    for comp in competitors_data[:10]])  # Limit to 10 competitors
        
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon marketplace strategist specializing in competitive analysis and differentiation for medical device products."
        user_prompt = f"""Perform a competitive analysis for this Amazon medical device and its competitors:
        
        Your Product: {product_info.get('name', 'Unknown')} (ASIN: {product_info.get('asin', 'Unknown')})
        Category: {product_info.get('category', 'Medical Device')}
        
        Competitors:
        {competitor_text}
        
        Provide the following analysis:
        1. Competitive positioning strategy
        2. Key differentiators to highlight in your listing
        3. Price positioning recommendations
        4. Feature comparison strategy
        5. Unique selling propositions to emphasize
        6. Weaknesses of competitors to address
        7. Customer pain points competitors aren't solving
        
        Focus on how to optimize the Amazon listing to stand out from these specific competitors.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        analysis = call_openai_api(messages, max_tokens=1200, temperature=0.2)
        
        if not analysis:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract sections from the analysis
        positioning = extract_theme_from_analysis(analysis, "Competitive positioning strategy")
        differentiators = extract_theme_from_analysis(analysis, "Key differentiators")
        price_positioning = extract_theme_from_analysis(analysis, "Price positioning")
        feature_comparison = extract_theme_from_analysis(analysis, "Feature comparison strategy")
        selling_propositions = extract_theme_from_analysis(analysis, "Unique selling propositions")
        competitor_weaknesses = extract_theme_from_analysis(analysis, "Weaknesses of competitors")
        pain_points = extract_theme_from_analysis(analysis, "Customer pain points")
        
        return {
            "success": True,
            "full_analysis": analysis,
            "sections": {
                "positioning": positioning,
                "differentiators": differentiators,
                "price_positioning": price_positioning,
                "feature_comparison": feature_comparison,
                "selling_propositions": selling_propositions,
                "competitor_weaknesses": competitor_weaknesses,
                "pain_points": pain_points
            }
        }
    except Exception as e:
        logger.error(f"Error in competitive analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_title_optimization(product_info, current_title=""):
    """
    Generate an optimized Amazon product title
    
    Parameters:
    - product_info: Dictionary with product details
    - current_title: Current product title if available
    
    Returns:
    - Dictionary with optimized title
    """
    try:
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist for medical devices."
        user_prompt = f"""Create an optimized Amazon title for this medical device product:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        Current Title: {current_title}
        
        Follow Amazon's best practices:
        - Maximum 200 characters
        - Include key search terms
        - Format: Brand + Model + Type + Key Features/Benefits
        - No promotional language like "best" or "top-rated"
        - No special characters beyond basic punctuation
        
        Provide just the optimized title text.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        optimized_title = call_openai_api(messages, max_tokens=200, temperature=0.2)
        
        if not optimized_title:
            return {"success": False, "error": "Failed to get API response"}
        
        # Clean up the title (remove quotes if present, etc.)
        optimized_title = optimized_title.strip('"\'')
        
        return {
            "success": True,
            "optimized_title": optimized_title,
            "character_count": len(optimized_title)
        }
    except Exception as e:
        logger.error(f"Error generating optimized title: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_bullet_points(product_info, current_bullets=None):
    """
    Generate optimized bullet points for Amazon listing
    
    Parameters:
    - product_info: Dictionary with product details
    - current_bullets: List of current bullet points if available
    
    Returns:
    - Dictionary with optimized bullet points
    """
    try:
        # Format current bullets if provided
        current_bullets_text = ""
        if current_bullets and isinstance(current_bullets, list):
            current_bullets_text = "\n".join([f"- {bullet}" for bullet in current_bullets])
        
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist for medical devices."
        user_prompt = f"""Create 5 optimized bullet points for this Amazon medical device listing:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        Description: {product_info.get('description', '')}
        
        Current Bullet Points:
        {current_bullets_text if current_bullets_text else "None provided"}
        
        Follow Amazon's best practices:
        - Focus on benefits, not just features
        - Address customer pain points and questions
        - Include relevant keywords
        - Start with capital letters
        - Keep each point under 200 characters
        - Format as complete sentences
        - No promotional language like "best" or "top-rated"
        
        Include one bullet point specifically addressing quality/durability and one addressing comfort/ease of use.
        
        Provide 5 bullet points formatted with a dash at the beginning of each line.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        bullet_points_text = call_openai_api(messages, max_tokens=500, temperature=0.2)
        
        if not bullet_points_text:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract bullet points from the response
        bullet_pattern = r"[-•]\s*(.+)"
        matches = re.findall(bullet_pattern, bullet_points_text)
        
        # If no matches found, try splitting by newlines
        if not matches:
            bullet_points = [line.strip() for line in bullet_points_text.split('\n') if line.strip()]
        else:
            bullet_points = [match.strip() for match in matches]
        
        return {
            "success": True,
            "bullet_points": bullet_points,
            "count": len(bullet_points)
        }
    except Exception as e:
        logger.error(f"Error generating bullet points: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_product_description(product_info, current_description=""):
    """
    Generate an optimized product description for Amazon listing
    
    Parameters:
    - product_info: Dictionary with product details
    - current_description: Current product description if available
    
    Returns:
    - Dictionary with optimized description
    """
    try:
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist for medical devices."
        user_prompt = f"""Create an optimized Amazon product description for this medical device:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        Current Description: {current_description}
        
        Follow Amazon's best practices:
        - Write in HTML format with paragraph tags (<p>) and line breaks
        - Expand on features and benefits beyond the bullet points
        - Include keywords naturally throughout the text
        - Address potential customer questions and objections
        - Describe specific use cases and scenarios
        - Highlight quality, comfort, ease of use, and durability
        - Avoid excessive capitalization and promotional language
        
        The description should be 3-5 paragraphs and include HTML formatting.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        description = call_openai_api(messages, max_tokens=800, temperature=0.2)
        
        if not description:
            return {"success": False, "error": "Failed to get API response"}
        
        return {
            "success": True,
            "description": description,
            "html_formatted": "<p>" in description.lower(),
            "character_count": len(description)
        }
    except Exception as e:
        logger.error(f"Error generating product description: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_keywords(product_info):
    """
    Generate keyword recommendations for Amazon listing
    
    Parameters:
    - product_info: Dictionary with product details
    
    Returns:
    - Dictionary with keyword recommendations
    """
    try:
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon SEO specialist for medical devices with deep knowledge of search patterns and keyword optimization."
        user_prompt = f"""Generate keyword recommendations for this Amazon medical device listing:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        Description: {product_info.get('description', '')}
        
        Please provide:
        1. Primary keywords (5-7 most important search terms)
        2. Secondary keywords (8-10 additional relevant terms)
        3. Long-tail keyword phrases (5-7 specific search phrases)
        4. Backend keywords (for Amazon backend search terms field)
        5. Competitor keywords (terms used by top competitors)
        
        Focus on medical terminology, symptoms, conditions, and use cases relevant to this product.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        keyword_analysis = call_openai_api(messages, max_tokens=800, temperature=0.2)
        
        if not keyword_analysis:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract sections
        primary_keywords = extract_theme_from_analysis(keyword_analysis, "Primary keywords")
        secondary_keywords = extract_theme_from_analysis(keyword_analysis, "Secondary keywords")
        long_tail = extract_theme_from_analysis(keyword_analysis, "Long-tail keyword phrases")
        backend_keywords = extract_theme_from_analysis(keyword_analysis, "Backend keywords")
        competitor_keywords = extract_theme_from_analysis(keyword_analysis, "Competitor keywords")
        
        return {
            "success": True,
            "full_analysis": keyword_analysis,
            "sections": {
                "primary_keywords": primary_keywords,
                "secondary_keywords": secondary_keywords,
                "long_tail": long_tail,
                "backend_keywords": backend_keywords,
                "competitor_keywords": competitor_keywords
            }
        }
    except Exception as e:
        logger.error(f"Error generating keywords: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_image_recommendations(product_info):
    """
    Generate image recommendations for Amazon listing
    
    Parameters:
    - product_info: Dictionary with product details
    
    Returns:
    - Dictionary with image recommendations
    """
    try:
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist for medical devices with expertise in product photography and image optimization."
        user_prompt = f"""Create detailed image recommendations for this Amazon medical device listing:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        Description: {product_info.get('description', '')}
        
        Provide the following:
        1. Specific types of images needed for this product (7-9 total images)
        2. Key features that should be highlighted in close-ups
        3. How to show the product in use (lifestyle images)
        4. Size reference recommendations
        5. Any infographics that would help explain benefits
        6. How to visually address common customer questions/concerns
        
        Be specific to this type of medical device and focus on images that would increase conversion rate.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        image_recommendations = call_openai_api(messages, max_tokens=800, temperature=0.2)
        
        if not image_recommendations:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract sections
        image_types = extract_theme_from_analysis(image_recommendations, "Specific types of images needed")
        feature_highlights = extract_theme_from_analysis(image_recommendations, "Key features that should be highlighted")
        lifestyle_images = extract_theme_from_analysis(image_recommendations, "How to show the product in use")
        size_reference = extract_theme_from_analysis(image_recommendations, "Size reference recommendations")
        infographics = extract_theme_from_analysis(image_recommendations, "Any infographics")
        address_concerns = extract_theme_from_analysis(image_recommendations, "How to visually address common customer questions")
        
        return {
            "success": True,
            "full_recommendations": image_recommendations,
            "sections": {
                "image_types": image_types,
                "feature_highlights": feature_highlights,
                "lifestyle_images": lifestyle_images,
                "size_reference": size_reference,
                "infographics": infographics,
                "address_concerns": address_concerns
            }
        }
    except Exception as e:
        logger.error(f"Error generating image recommendations: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_return_reduction_plan(product_info, return_analysis):
    """
    Generate a plan to reduce return rates
    
    Parameters:
    - product_info: Dictionary with product details
    - return_analysis: Analysis of return reasons
    
    Returns:
    - Dictionary with return reduction plan
    """
    try:
        # Format return categories if available
        return_categories = ""
        if isinstance(return_analysis, dict) and 'categories' in return_analysis:
            for category, content in return_analysis['categories'].items():
                if content:
                    return_categories += f"{category}: {content}\n\n"
        
        # Create the prompt for the AI
        system_prompt = "You are an expert Amazon listing optimization specialist with deep expertise in reducing return rates for medical devices."
        user_prompt = f"""Create an actionable return reduction plan for this Amazon medical device:
        
        Product: {product_info.get('name', 'Unknown')} (ASIN: {product_info.get('asin', 'Unknown')})
        Category: {product_info.get('category', 'Medical Device')}
        Current Return Rate: {product_info.get('return_rate_30d', 0):.2f}%
        
        Return Analysis:
        {return_categories if return_categories else "Not available"}
        
        Please provide:
        1. Immediate listing changes to reduce returns (top 3 priorities)
        2. Product improvement recommendations
        3. Packaging/instructions improvements
        4. Customer expectation management strategies
        5. Expected impact on return rate for each recommendation
        
        The plan should be specific, actionable, and prioritized by impact.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API
        return_plan = call_openai_api(messages, max_tokens=1000, temperature=0.2)
        
        if not return_plan:
            return {"success": False, "error": "Failed to get API response"}
        
        # Extract sections
        listing_changes = extract_theme_from_analysis(return_plan, "Immediate listing changes")
        product_improvements = extract_theme_from_analysis(return_plan, "Product improvement recommendations")
        packaging_improvements = extract_theme_from_analysis(return_plan, "Packaging/instructions improvements")
        expectation_management = extract_theme_from_analysis(return_plan, "Customer expectation management strategies")
        impact_assessment = extract_theme_from_analysis(return_plan, "Expected impact")
        
        return {
            "success": True,
            "full_plan": return_plan,
            "sections": {
                "listing_changes": listing_changes,
                "product_improvements": product_improvements,
                "packaging_improvements": packaging_improvements,
                "expectation_management": expectation_management,
                "impact_assessment": impact_assessment
            }
        }
    except Exception as e:
        logger.error(f"Error generating return reduction plan: {str(e)}")
        return {"success": False, "error": str(e)}

# Non-AI backup functions for when API calls fail
def keyword_based_review_analysis(reviews):
    """
    Perform basic keyword-based analysis of reviews when AI is not available
    
    Parameters:
    - reviews: List of review dictionaries
    
    Returns:
    - Dictionary with analysis results
    """
    results = {
        'total_reviews': len(reviews),
        'average_rating': 0,
        'keyword_analysis': {}
    }
    
    # Calculate average rating
    total_rating = 0
    rated_reviews = 0
    
    for review in reviews:
        if 'rating' in review and review['rating']:
            try:
                rating = float(review['rating'])
                total_rating += rating
                rated_reviews += 1
            except (ValueError, TypeError):
                pass
    
    if rated_reviews > 0:
        results['average_rating'] = total_rating / rated_reviews
    
    # Perform keyword analysis for both medical and Amazon keywords
    for category, keywords in {**MEDICAL_KEYWORDS, **AMAZON_KEYWORDS}.items():
        category_count = 0
        for review in reviews:
            review_text = review.get('review_text', '').lower()
            if any(keyword.lower() in review_text for keyword in keywords):
                category_count += 1
        
        results['keyword_analysis'][category] = {
            'count': category_count,
            'percentage': (category_count / len(reviews) * 100) if len(reviews) > 0 else 0
        }
    
    return results

def basic_return_analysis(return_reasons):
    """
    Perform basic analysis of return reasons when AI is not available
    
    Parameters:
    - return_reasons: List of return reason dictionaries
    
    Returns:
    - Dictionary with analysis results
    """
    results = {
        'total_returns': len(return_reasons),
        'common_reasons': {},
        'keyword_analysis': {}
    }
    
    # Count common reasons
    reason_counts = {}
    
    for item in return_reasons:
        reason = item.get('return_reason', '').strip()
        if reason:
            if reason in reason_counts:
                reason_counts[reason] += 1
            else:
                reason_counts[reason] = 1
    
    # Sort by frequency
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    results['common_reasons'] = dict(sorted_reasons[:10])  # Top 10 reasons
    
    # Perform keyword analysis for both medical and Amazon keywords
    for category, keywords in {**MEDICAL_KEYWORDS, **AMAZON_KEYWORDS}.items():
        category_count = 0
        for item in return_reasons:
            reason_text = item.get('return_reason', '').lower()
            if any(keyword.lower() in reason_text for keyword in keywords):
                category_count += 1
        
        results['keyword_analysis'][category] = {
            'count': category_count,
            'percentage': (category_count / len(return_reasons) * 100) if len(return_reasons) > 0 else 0
        }
    
    return results

def generate_basic_recommendations(product_info, review_analysis, return_analysis):
    """
    Generate basic recommendations when AI is not available
    
    Parameters:
    - product_info: Dictionary with product details
    - review_analysis: Results from review analysis
    - return_analysis: Results from return analysis
    
    Returns:
    - List of recommendation dictionaries
    """
    recommendations = []
    
    # Check return rate
    return_rate = product_info.get('return_rate_30d', 0)
    if return_rate > 8:
        recommendations.append({
            "issue": "High return rate",
            "recommendation": "Investigate the top return reasons and address product quality issues",
            "priority": "High"
        })
    
    # Check rating
    avg_rating = product_info.get('star_rating', 0)
    if avg_rating and avg_rating < 4.0:
        recommendations.append({
            "issue": "Low star rating",
            "recommendation": "Address common complaints in negative reviews",
            "priority": "High"
        })
    
    # Check keyword analysis from reviews
    if isinstance(review_analysis, dict) and 'keyword_analysis' in review_analysis:
        for category, data in review_analysis['keyword_analysis'].items():
            if data['percentage'] > 20:  # If more than 20% of reviews mention this category
                recommendations.append({
                    "issue": f"High frequency of {category} concerns",
                    "recommendation": f"Review and address {category} issues mentioned in customer feedback",
                    "priority": "Medium"
                })
    
    # Check common return reasons
    if isinstance(return_analysis, dict) and 'common_reasons' in return_analysis and return_analysis['common_reasons']:
        top_reason = list(return_analysis['common_reasons'].keys())[0]
        recommendations.append({
            "issue": f"Most common return reason: {top_reason}",
            "recommendation": "Address this specific issue in the product description and images",
            "priority": "High"
        })
    
    # Amazon listing optimization recommendations
    recommendations.append({
        "issue": "Title optimization",
        "recommendation": "Include primary keywords, features and benefits in product title",
        "priority": "Medium"
    })
    
    recommendations.append({
        "issue": "Bullet point improvement",
        "recommendation": "Focus on benefits rather than features in bullet points",
        "priority": "Medium"
    })
    
    recommendations.append({
        "issue": "Image optimization",
        "recommendation": "Include lifestyle images, size references, and feature close-ups",
        "priority": "High"
    })
    
    recommendations.append({
        "issue": "Documentation and instructions",
        "recommendation": "Ensure clear usage instructions are visible in images and description",
        "priority": "Medium"
    })
    
    return recommendations
