# Simplified Medical Device Review Analyzer - Listing Optimization Focus

import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import json

class ListingOptimizationAnalyzer:
    def __init__(self):
        self.ai_available = self._check_ai_availability()
    
    def _check_ai_availability(self):
        """Check if AI is available and working"""
        try:
            # Test AI connection
            import openai
            api_key = st.secrets.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                return False
            
            openai.api_key = api_key
            # Quick test call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except:
            return False
    
    def analyze_reviews_ai(self, reviews_df, product_info):
        """AI-powered analysis of reviews for listing optimization"""
        if not self.ai_available:
            return None
        
        # Prepare review text for AI
        review_texts = []
        for _, row in reviews_df.iterrows():
            text = f"Rating: {row.get('rating', 'N/A')}\n"
            text += f"Title: {row.get('title', '')}\n"
            text += f"Review: {row.get('body', '')}\n"
            review_texts.append(text)
        
        # Combine reviews (limit for token management)
        combined_reviews = "\n---\n".join(review_texts[:50])  # Limit to 50 reviews
        
        prompt = f"""
        Analyze these Amazon reviews for a medical device product for listing optimization purposes.
        
        Product: {product_info.get('name', 'Unknown Product')}
        Total Reviews Analyzed: {len(review_texts)}
        
        Reviews:
        {combined_reviews}
        
        Provide analysis in this JSON format:
        {{
            "listing_optimization": {{
                "title_improvements": ["suggestion 1", "suggestion 2"],
                "bullet_point_improvements": ["improvement 1", "improvement 2"],
                "image_recommendations": ["recommendation 1", "recommendation 2"],
                "description_enhancements": ["enhancement 1", "enhancement 2"]
            }},
            "review_categories": {{
                "positive_themes": {{"theme": "description", "count": number, "examples": ["example"]}},
                "negative_themes": {{"theme": "description", "count": number, "examples": ["example"]}},
                "feature_mentions": {{"feature": "description", "sentiment": "positive/negative", "count": number}}
            }},
            "quantitative_summary": {{
                "overall_sentiment": "positive/negative/mixed",
                "key_metrics": {{"satisfaction_score": 0-100, "recommendation_likelihood": 0-100}},
                "priority_issues": ["issue 1", "issue 2"],
                "competitive_advantages": ["advantage 1", "advantage 2"]
            }},
            "actionable_insights": {{
                "immediate_actions": ["action 1", "action 2"],
                "content_updates": ["update 1", "update 2"],
                "customer_pain_points": ["pain point 1", "pain point 2"],
                "selling_points_to_emphasize": ["point 1", "point 2"]
            }}
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use GPT-4 for better analysis
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return {"success": True, "analysis": result, "ai_powered": True}
            
        except Exception as e:
            st.error(f"AI Analysis failed: {str(e)}")
            return None
    
    def basic_fallback_analysis(self, reviews_df):
        """Basic analysis when AI is unavailable"""
        if reviews_df.empty:
            return {"success": False, "error": "No data to analyze"}
        
        # Basic metrics
        total_reviews = len(reviews_df)
        avg_rating = reviews_df['rating'].mean() if 'rating' in reviews_df.columns else 0
        
        # Simple sentiment based on ratings
        positive_reviews = len(reviews_df[reviews_df['rating'] >= 4]) if 'rating' in reviews_df.columns else 0
        negative_reviews = len(reviews_df[reviews_df['rating'] <= 2]) if 'rating' in reviews_df.columns else 0
        
        # Basic keyword analysis
        all_text = ' '.join(reviews_df['body'].fillna('').astype(str)).lower()
        
        positive_keywords = ['great', 'excellent', 'love', 'perfect', 'amazing', 'wonderful']
        negative_keywords = ['terrible', 'awful', 'broken', 'poor', 'disappointing', 'waste']
        
        positive_mentions = sum(all_text.count(word) for word in positive_keywords)
        negative_mentions = sum(all_text.count(word) for word in negative_keywords)
        
        return {
            "success": True,
            "ai_powered": False,
            "basic_analysis": {
                "total_reviews": total_reviews,
                "average_rating": round(avg_rating, 2),
                "positive_reviews": positive_reviews,
                "negative_reviews": negative_reviews,
                "positive_sentiment_indicators": positive_mentions,
                "negative_sentiment_indicators": negative_mentions,
                "sentiment_ratio": positive_mentions / max(negative_mentions, 1)
            }
        }

def main():
    st.set_page_config(
        page_title="Amazon Review Analyzer - Listing Optimization",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Amazon Review Analyzer")
    st.markdown("**Listing Optimization Tool** - Upload Helium 10 exports for AI-powered insights")
    
    analyzer = ListingOptimizationAnalyzer()
    
    # AI Status indicator
    if analyzer.ai_available:
        st.success("🤖 AI Analysis Available")
    else:
        st.error("🤖 AI Analysis Unavailable - Basic analysis mode only")
        st.caption("Configure OpenAI API key for full AI-powered insights")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Helium 10 Review Export (CSV/Excel)",
        type=['csv', 'xlsx'],
        help="Upload the review export file from Helium 10"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} reviews")
            
            # Show data preview
            with st.expander("📄 Data Preview"):
                st.dataframe(df.head())
            
            # Product info extraction (adapt based on Helium 10 format)
            product_info = {
                'name': df.get('Product Title', ['Unknown Product']).iloc[0] if 'Product Title' in df.columns else 'Unknown Product',
                'asin': df.get('ASIN', ['Unknown']).iloc[0] if 'ASIN' in df.columns else 'Unknown'
            }
            
            if st.button("🚀 Analyze Reviews", type="primary"):
                if analyzer.ai_available:
                    # AI Analysis
                    with st.spinner("🤖 Running AI analysis..."):
                        result = analyzer.analyze_reviews_ai(df, product_info)
                    
                    if result and result.get('success'):
                        display_ai_results(result['analysis'])
                    else:
                        st.warning("AI analysis failed, running basic analysis...")
                        basic_result = analyzer.basic_fallback_analysis(df)
                        display_basic_results(basic_result)
                else:
                    # Basic fallback
                    with st.spinner("📊 Running basic analysis..."):
                        result = analyzer.basic_fallback_analysis(df)
                    display_basic_results(result)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def display_ai_results(analysis):
    """Display AI-powered analysis results"""
    st.markdown("## 🤖 AI Analysis Results")
    
    # Listing Optimization Tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Listing Optimization", 
        "📊 Review Categories", 
        "📈 Quantitative Summary",
        "💡 Actionable Insights"
    ])
    
    with tab1:
        st.markdown("### Listing Optimization Recommendations")
        opt = analysis.get('listing_optimization', {})
        
        if opt.get('title_improvements'):
            st.markdown("**Title Improvements:**")
            for improvement in opt['title_improvements']:
                st.markdown(f"• {improvement}")
        
        if opt.get('bullet_point_improvements'):
            st.markdown("**Bullet Point Improvements:**")
            for improvement in opt['bullet_point_improvements']:
                st.markdown(f"• {improvement}")
        
        if opt.get('image_recommendations'):
            st.markdown("**Image Recommendations:**")
            for rec in opt['image_recommendations']:
                st.markdown(f"• {rec}")
    
    with tab2:
        st.markdown("### Review Categories")
        categories = analysis.get('review_categories', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Themes:**")
            for theme, data in categories.get('positive_themes', {}).items():
                st.markdown(f"• **{theme}**: {data.get('count', 0)} mentions")
        
        with col2:
            st.markdown("**Negative Themes:**")
            for theme, data in categories.get('negative_themes', {}).items():
                st.markdown(f"• **{theme}**: {data.get('count', 0)} mentions")
    
    with tab3:
        st.markdown("### Quantitative Summary")
        summary = analysis.get('quantitative_summary', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment = summary.get('overall_sentiment', 'Unknown')
            st.metric("Overall Sentiment", sentiment)
        
        with col2:
            satisfaction = summary.get('key_metrics', {}).get('satisfaction_score', 0)
            st.metric("Satisfaction Score", f"{satisfaction}/100")
        
        with col3:
            recommendation = summary.get('key_metrics', {}).get('recommendation_likelihood', 0)
            st.metric("Recommendation Likelihood", f"{recommendation}/100")
    
    with tab4:
        st.markdown("### Actionable Insights")
        insights = analysis.get('actionable_insights', {})
        
        if insights.get('immediate_actions'):
            st.markdown("**Immediate Actions:**")
            for action in insights['immediate_actions']:
                st.markdown(f"• {action}")
        
        if insights.get('selling_points_to_emphasize'):
            st.markdown("**Selling Points to Emphasize:**")
            for point in insights['selling_points_to_emphasize']:
                st.markdown(f"• {point}")

def display_basic_results(result):
    """Display basic fallback analysis"""
    st.warning("🤖 AI Analysis Unavailable - Showing Basic Analysis")
    st.caption("Configure OpenAI API key for detailed AI-powered insights")
    
    if not result.get('success'):
        st.error("Analysis failed")
        return
    
    basic = result['basic_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", basic['total_reviews'])
    
    with col2:
        st.metric("Average Rating", basic['average_rating'])
    
    with col3:
        st.metric("Positive Reviews", basic['positive_reviews'])
    
    with col4:
        st.metric("Negative Reviews", basic['negative_reviews'])
    
    st.info("💡 **Upgrade to AI Analysis**: Configure OpenAI API key for detailed categorization, listing optimization recommendations, and actionable insights.")

if __name__ == "__main__":
    main()
