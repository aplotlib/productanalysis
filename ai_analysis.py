"""
AI Analysis module for Medical Device Review Analysis Tool

This module contains functions for performing AI-based analysis of
medical device product reviews, return reasons, and other data.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing OpenAI - for AI analysis
try:
    import openai
    from openai import OpenAI  # Import OpenAI client for newer SDK
    HAS_OPENAI = True
except ImportError:
    logger.warning("OpenAI module not available")
    HAS_OPENAI = False

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

# FDA Device classes and typical products
FDA_DEVICE_CLASSES = {
    'Class I': [
        'Mobility Aids (Basic canes, basic walkers)',
        'Bathroom Safety (Basic shower chairs, toilet safety frames)',
        'Elastic Bandages',
        'Manual Wheelchairs',
        'Basic Orthopedic Supports',
        'Basic Dressings'
    ],
    'Class II': [
        'Blood Pressure Monitors',
        'Powered Wheelchairs',
        'CPAP Devices',
        'Infusion Pumps',
        'Powered Mobility Aids',
        'Pulse Oximeters',
        'Digital Thermometers'
    ],
    'Class III': [
        'Implantable Devices',
        'Life-Supporting Devices',
        'Life-Sustaining Devices',
        'Automated External Defibrillators (AEDs)'
    ]
}

def get_openai_client(api_key=None):
    """
    Get the OpenAI client based on available SDK version
    
    Parameters:
    - api_key: OpenAI API key (optional if set in environment or streamlit secrets)
    
    Returns:
    - Client object or None, and a boolean indicating if it's the newer SDK
    """
    # Check for API key from different sources
    if not api_key:
        # Try to get from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        
        # Try to get from streamlit secrets if available and api_key is still None
        try:
            import streamlit as st
            if api_key is None and hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except (ImportError, AttributeError):
            pass
    
    if not api_key:
        logger.warning("OpenAI API key not found")
        return None, False
    
    # Check which version of the OpenAI SDK is being used
    if hasattr(openai, 'OpenAI'):  # New SDK (v1.0.0+)
        client = OpenAI(api_key=api_key)
        return client, True
    else:  # Legacy SDK (<v1.0.0)
        openai.api_key = api_key
        return openai, False

def analyze_reviews_with_ai(reviews, api_key=None):
    """
    Analyze a list of product reviews using AI
    
    Parameters:
    - reviews: List of dictionaries containing review data
    - api_key: OpenAI API key (optional if set in environment)
    
    Returns:
    - Dictionary with analysis results
    """
    if not HAS_OPENAI:
        return {"error": "OpenAI module not available"}
    
    client, is_new_sdk = get_openai_client(api_key)
    if client is None:
        return {"error": "OpenAI API key not set"}
    
    try:
        # Prepare review text
        review_texts = []
        for review in reviews[:20]:  # Limit to 20 reviews to avoid token limits
            rating = review.get('rating', 'Unknown')
            text = review.get('review_text', '')
            review_texts.append(f"Rating: {rating} - {text}")
        
        review_content = "\n\n".join(review_texts)
        
        # Create the prompt for the AI
        system_prompt = "You are an expert in medical device quality analysis with deep knowledge of FDA regulations, customer feedback interpretation, and product quality improvement."
        user_prompt = f"""Analyze the following medical device product reviews. 
        Identify key issues, patterns, and insights related to:
        1. Safety concerns
        2. Efficacy/effectiveness
        3. Comfort/ergonomics
        4. Durability/quality
        5. Ease of use
        6. Size/fit issues
        7. Potential regulatory concerns
        
        For each category, provide specific examples from the reviews.
        Then provide an overall assessment of the product based on these reviews.
        
        Reviews:
        {review_content}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API based on SDK version
        if is_new_sdk:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            analysis = response.choices[0].message.content
        else:
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            analysis = response.choices[0].message.content
        
        # Extract themes and issues from the analysis
        safety_issues = extract_theme_from_analysis(analysis, "Safety concerns")
        efficacy_issues = extract_theme_from_analysis(analysis, "Efficacy/effectiveness")
        comfort_issues = extract_theme_from_analysis(analysis, "Comfort/ergonomics")
        durability_issues = extract_theme_from_analysis(analysis, "Durability/quality")
        usability_issues = extract_theme_from_analysis(analysis, "Ease of use")
        fit_issues = extract_theme_from_analysis(analysis, "Size/fit issues")
        regulatory_issues = extract_theme_from_analysis(analysis, "Potential regulatory concerns")
        
        return {
            "success": True,
            "full_analysis": analysis,
            "themes": {
                "safety": safety_issues,
                "efficacy": efficacy_issues,
                "comfort": comfort_issues,
                "durability": durability_issues,
                "usability": usability_issues,
                "fit": fit_issues,
                "regulatory": regulatory_issues
            }
        }
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
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

def analyze_returns_with_ai(return_reasons, api_key=None):
    """
    Analyze a list of return reasons using AI
    
    Parameters:
    - return_reasons: List of return reason strings
    - api_key: OpenAI API key (optional if set in environment)
    
    Returns:
    - Dictionary with analysis results
    """
    if not HAS_OPENAI:
        return {"error": "OpenAI module not available"}
    
    client, is_new_sdk = get_openai_client(api_key)
    if client is None:
        return {"error": "OpenAI API key not set"}
    
    try:
        # Prepare return reason text
        return_text = "\n".join([f"- {reason}" for reason in return_reasons[:30]])  # Limit to 30 returns
        
        # Create the prompt for the AI
        system_prompt = "You are an expert in medical device quality analysis with deep knowledge of product returns, root cause analysis, and quality improvement."
        user_prompt = f"""Analyze the following return reasons for a medical device product.
        Categorize the returns into these groups:
        1. Product defects/quality issues
        2. Size/fit issues
        3. Comfort issues
        4. Performance/efficacy issues
        5. User error/misunderstanding
        6. Preference/expectation mismatch
        7. Potential safety concerns
        
        For each category, provide the count and percentage of returns, with specific examples.
        Then provide an overall assessment of return patterns and recommended actions.
        
        Return Reasons:
        {return_text}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API based on SDK version
        if is_new_sdk:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
            analysis = response.choices[0].message.content
        else:
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
            analysis = response.choices[0].message.content
        
        # Extract categories from the analysis
        defect_issues = extract_theme_from_analysis(analysis, "Product defects/quality issues")
        size_issues = extract_theme_from_analysis(analysis, "Size/fit issues")
        comfort_issues = extract_theme_from_analysis(analysis, "Comfort issues")
        performance_issues = extract_theme_from_analysis(analysis, "Performance/efficacy issues")
        user_error = extract_theme_from_analysis(analysis, "User error/misunderstanding")
        expectation_issues = extract_theme_from_analysis(analysis, "Preference/expectation mismatch")
        safety_concerns = extract_theme_from_analysis(analysis, "Potential safety concerns")
        
        return {
            "success": True,
            "full_analysis": analysis,
            "categories": {
                "defects": defect_issues,
                "size": size_issues,
                "comfort": comfort_issues,
                "performance": performance_issues,
                "user_error": user_error,
                "expectations": expectation_issues,
                "safety": safety_concerns
            }
        }
    except Exception as e:
        logger.error(f"Error in return analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def classify_medical_device_with_ai(product_name, category, description="", api_key=None):
    """
    Classify a medical device according to FDA risk classes using AI
    
    Parameters:
    - product_name: Name of the product
    - category: Product category
    - description: Product description
    - api_key: OpenAI API key (optional if set in environment)
    
    Returns:
    - Dictionary with classification results
    """
    if not HAS_OPENAI:
        return {"error": "OpenAI module not available"}
    
    client, is_new_sdk = get_openai_client(api_key)
    if client is None:
        return {"error": "OpenAI API key not set"}
    
    try:
        # Create the prompt for the AI
        system_prompt = "You are an expert regulatory affairs specialist with deep knowledge of FDA medical device classifications, regulatory pathways, and compliance requirements."
        user_prompt = f"""Classify the following medical device according to FDA risk classes (I, II, or III).
        
        Product Name: {product_name}
        Category: {category}
        Description: {description}
        
        For each FDA class, provide:
        1. Class designation (I, II, or III)
        2. Confidence level (Low, Medium, High)
        3. Rationale for classification
        4. Typical regulatory requirements for this type of device
        5. Whether the device would likely be exempt from 510(k) requirements
        6. Whether the device is likely OTC (over-the-counter) or Rx (prescription)
        
        FDA Device Class Examples:
        - Class I (low risk): elastic bandages, examination gloves, hand-held surgical instruments
        - Class II (moderate risk): powered wheelchairs, infusion pumps, surgical drapes
        - Class III (high risk): implantable pacemakers, implantable defibrillators

        Provide your final classification with confidence level and key regulatory considerations.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API based on SDK version
        if is_new_sdk:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
            analysis = response.choices[0].message.content
        else:
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
            analysis = response.choices[0].message.content
        
        # Try to extract the final classification
        class_match = re.search(r"Class (I|II|III)", analysis)
        fda_class = class_match.group(0) if class_match else "Unknown"
        
        # Try to extract the confidence level
        confidence_match = re.search(r"(Low|Medium|High) confidence", analysis, re.IGNORECASE)
        confidence = confidence_match.group(1) if confidence_match else "Unknown"
        
        # Try to extract OTC/Rx status
        otc_match = re.search(r"(OTC|over-the-counter)", analysis, re.IGNORECASE)
        rx_match = re.search(r"(Rx|prescription)", analysis, re.IGNORECASE)
        status = "OTC" if otc_match else "Rx" if rx_match else "Unknown"
        
        return {
            "success": True,
            "full_analysis": analysis,
            "classification": {
                "fda_class": fda_class,
                "confidence": confidence,
                "otc_rx_status": status
            }
        }
    except Exception as e:
        logger.error(f"Error in device classification: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_improvement_recommendations(product_info, reviews_data, returns_data, sales_data=None, api_key=None):
    """
    Generate AI-powered recommendations for product improvements
    
    Parameters:
    - product_info: Dictionary with product details
    - reviews_data: List of review dictionaries
    - returns_data: List of return reason dictionaries
    - sales_data: Optional sales and returns metrics
    - api_key: OpenAI API key (optional if set in environment)
    
    Returns:
    - Dictionary with recommendations
    """
    if not HAS_OPENAI:
        return {"error": "OpenAI module not available"}
    
    client, is_new_sdk = get_openai_client(api_key)
    if client is None:
        return {"error": "OpenAI API key not set"}
    
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
        system_prompt = "You are an expert in medical device quality improvement with deep knowledge of FDA regulations, product development, quality systems, and risk management."
        user_prompt = f"""As a medical device quality expert, analyze the following product data and provide
        actionable improvement recommendations:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        FDA Class: {product_info.get('device_class', 'Unknown')}
        """
        
        if return_rate is not None:
            user_prompt += f"Return Rate: {return_rate:.2f}%\n"
        
        user_prompt += f"""
        Customer Reviews:
        {review_content}
        
        Return Reasons:
        {return_content}
        
        Based on this data, provide:
        1. Top 5 actionable improvement recommendations, ranked by potential impact
        2. Quality concerns that require immediate investigation, if any
        3. Regulatory compliance considerations or risks
        4. User experience/usability improvements
        5. Packaging or instruction improvements
        6. Competitive differentiation opportunities
        
        For each recommendation, include:
        - Specific action
        - Expected impact
        - Implementation difficulty (Low, Medium, High)
        - Priority level (Low, Medium, High)
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API based on SDK version
        if is_new_sdk:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
                temperature=0.2
            )
            recommendations = response.choices[0].message.content
        else:
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
                temperature=0.2
            )
            recommendations = response.choices[0].message.content
        
        # Extract sections from the recommendations
        top_recommendations = extract_theme_from_analysis(recommendations, "Top 5 actionable improvement recommendations")
        quality_concerns = extract_theme_from_analysis(recommendations, "Quality concerns")
        regulatory_considerations = extract_theme_from_analysis(recommendations, "Regulatory compliance considerations")
        usability_improvements = extract_theme_from_analysis(recommendations, "User experience/usability improvements")
        packaging_improvements = extract_theme_from_analysis(recommendations, "Packaging or instruction improvements")
        competitive_opportunities = extract_theme_from_analysis(recommendations, "Competitive differentiation opportunities")
        
        return {
            "success": True,
            "full_recommendations": recommendations,
            "sections": {
                "top_recommendations": top_recommendations,
                "quality_concerns": quality_concerns,
                "regulatory_considerations": regulatory_considerations,
                "usability_improvements": usability_improvements,
                "packaging_improvements": packaging_improvements,
                "competitive_opportunities": competitive_opportunities
            }
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_regulatory_assessment(product_info, reviews_data, api_key=None):
    """
    Generate AI-powered regulatory compliance assessment
    
    Parameters:
    - product_info: Dictionary with product details
    - reviews_data: List of review dictionaries
    - api_key: OpenAI API key (optional if set in environment)
    
    Returns:
    - Dictionary with regulatory assessment
    """
    if not HAS_OPENAI:
        return {"error": "OpenAI module not available"}
    
    client, is_new_sdk = get_openai_client(api_key)
    if client is None:
        return {"error": "OpenAI API key not set"}
    
    try:
        # Prepare review text
        review_texts = []
        for review in reviews_data[:15]:  # Limit to 15 reviews
            rating = review.get('rating', 'Unknown')
            text = review.get('review_text', '')
            review_texts.append(f"Rating: {rating} - {text}")
        
        review_content = "\n".join(review_texts)
        
        # Create the prompt for the AI
        system_prompt = "You are an expert in medical device regulatory compliance with deep knowledge of FDA regulations, QSR requirements, MDR reporting, and labeling/marketing requirements."
        user_prompt = f"""As a medical device regulatory expert, assess potential regulatory compliance concerns
        for the following product based on customer reviews:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Medical Device')}
        FDA Class: {product_info.get('device_class', 'Unknown')}
        
        Customer Reviews:
        {review_content}
        
        Provide a regulatory assessment covering:
        1. Potential labeling/marketing claim concerns
        2. Adverse events or safety issues that may require reporting
        3. Quality System Regulation (QSR) considerations
        4. Post-market surveillance implications
        5. Overall regulatory risk assessment (Low, Medium, High)
        
        For each identified concern, provide:
        - Specific issue description
        - Relevant regulation/guidance
        - Recommended corrective action
        - Priority level (Low, Medium, High)
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the OpenAI API based on SDK version
        if is_new_sdk:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            assessment = response.choices[0].message.content
        else:
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            assessment = response.choices[0].message.content
        
        # Extract risk level
        risk_match = re.search(r"Overall regulatory risk assessment: (Low|Medium|High)", assessment)
        risk_level = risk_match.group(1) if risk_match else "Unknown"
        
        # Extract sections from the assessment
        labeling_concerns = extract_theme_from_analysis(assessment, "Potential labeling/marketing claim concerns")
        adverse_events = extract_theme_from_analysis(assessment, "Adverse events or safety issues")
        qsr_considerations = extract_theme_from_analysis(assessment, "Quality System Regulation")
        surveillance_implications = extract_theme_from_analysis(assessment, "Post-market surveillance implications")
        
        return {
            "success": True,
            "full_assessment": assessment,
            "risk_level": risk_level,
            "sections": {
                "labeling_concerns": labeling_concerns,
                "adverse_events": adverse_events,
                "qsr_considerations": qsr_considerations,
                "surveillance_implications": surveillance_implications
            }
        }
    except Exception as e:
        logger.error(f"Error generating regulatory assessment: {str(e)}")
        return {"success": False, "error": str(e)}

# Non-AI backup functions for when AI is not available
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
    
    # Perform keyword analysis
    for category, keywords in MEDICAL_KEYWORDS.items():
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
    
    # Perform keyword analysis
    for category, keywords in MEDICAL_KEYWORDS.items():
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

def estimate_device_class(product_name, category):
    """
    Estimate the FDA device class based on product name and category
    when AI is not available
    
    Parameters:
    - product_name: Name of the product
    - category: Product category
    
    Returns:
    - Dictionary with estimated classification
    """
    product_name_lower = product_name.lower()
    category_lower = category.lower()
    
    # Check for Class I indicators
    class_i_indicators = ['cane', 'walker', 'basic', 'bandage', 'dressing', 'support',
                         'bath', 'shower', 'toilet', 'manual wheelchair']
    
    # Check for Class II indicators
    class_ii_indicators = ['blood pressure', 'powered', 'electronic', 'monitor', 'digital',
                          'pump', 'cpap', 'oximeter', 'thermometer', 'nebulizer']
    
    # Check for Class III indicators (unlikely on Amazon but included for completeness)
    class_iii_indicators = ['implant', 'implantable', 'life-sustaining', 'life-supporting',
                           'defibrillator', 'pacemaker']
    
    # Count indicators in product name and category
    class_i_count = sum(1 for indicator in class_i_indicators if indicator in product_name_lower or indicator in category_lower)
    class_ii_count = sum(1 for indicator in class_ii_indicators if indicator in product_name_lower or indicator in category_lower)
    class_iii_count = sum(1 for indicator in class_iii_indicators if indicator in product_name_lower or indicator in category_lower)
    
    # Determine class based on highest count
    if class_iii_count > 0:
        # Class III devices are unlikely to be sold on Amazon, so add extra validation
        if class_iii_count > class_ii_count and class_iii_count > class_i_count:
            return {
                "fda_class": "Class III",
                "confidence": "Low",
                "otc_rx_status": "Rx",
                "note": "Class III estimation has low confidence. These devices typically require prescription and are rarely sold on Amazon."
            }
    
    if class_ii_count > class_i_count:
        return {
            "fda_class": "Class II",
            "confidence": "Medium",
            "otc_rx_status": "OTC" if "otc" in product_name_lower or "over the counter" in product_name_lower else "Unknown",
            "note": "Class II estimation based on product keywords. Many Class II devices can be sold OTC."
        }
    
    # Default to Class I
    return {
        "fda_class": "Class I",
        "confidence": "Medium",
        "otc_rx_status": "OTC",
        "note": "Class I estimation based on product keywords. Most Class I devices are exempt from 510(k) requirements."
    }

def generate_basic_recommendations(product_info, review_analysis, return_analysis):
    """
    Generate basic recommendations when AI is not available
    
    Parameters:
    - product_info: Dictionary with product details
    - review_analysis: Results from review analysis
    - return_analysis: Results from return analysis
    
    Returns:
    - Dictionary with basic recommendations
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
    if 'keyword_analysis' in review_analysis:
        for category, data in review_analysis['keyword_analysis'].items():
            if data['percentage'] > 20:  # If more than 20% of reviews mention this category
                recommendations.append({
                    "issue": f"High frequency of {category} concerns",
                    "recommendation": f"Review and address {category} issues mentioned in customer feedback",
                    "priority": "Medium"
                })
    
    # Check common return reasons
    if 'common_reasons' in return_analysis and return_analysis['common_reasons']:
        top_reason = list(return_analysis['common_reasons'].keys())[0]
        recommendations.append({
            "issue": f"Most common return reason: {top_reason}",
            "recommendation": "Address this specific issue in the product design or documentation",
            "priority": "High"
        })
    
    # Add general recommendations
    recommendations.append({
        "issue": "General product improvement",
        "recommendation": "Review all customer feedback for recurring themes and improvement opportunities",
        "priority": "Medium"
    })
    
    recommendations.append({
        "issue": "Documentation and instructions",
        "recommendation": "Ensure clear, comprehensive instructions, especially for assembly and usage",
        "priority": "Medium"
    })
    
    if product_info.get('device_class', '') in ('Class II', 'Class III'):
        recommendations.append({
            "issue": "Regulatory compliance",
            "recommendation": "Ensure all marketing claims and product descriptions comply with FDA regulations",
            "priority": "High"
        })
    
    return recommendations
