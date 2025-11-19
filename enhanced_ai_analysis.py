import os
import pandas as pd
import google.generativeai as genai
# import openai # Uncomment if using OpenAI

class AIAnalysisEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analyze_feedback(self, df, text_column='Customer Feedback'):
        """
        Sends batches of customer feedback to AI to categorize return reasons.
        """
        if not self.api_key or text_column not in df.columns:
            return "AI Analysis Unavailable: Missing API Key or Text Column."

        insights = []
        
        # Process in chunks to avoid token limits
        feedback_samples = df[text_column].dropna().head(50).tolist()
        
        prompt = f"""
        Analyze the following list of customer return comments for medical mobility products.
        Categorize the main issues (e.g., Sizing, Damaged in Transit, Defective, Confusion).
        Provide a summary of the top 3 pain points.
        
        Comments:
        {feedback_samples}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error during AI analysis: {str(e)}"

    def analyze_helpdesk_trends(self, df):
        """
        Reads Helpdesk subjects to find common issues.
        """
        if 'Subject' not in df.columns:
            return "No 'Subject' column found for analysis."
            
        subjects = df['Subject'].dropna().tail(50).tolist()
        
        prompt = f"""
        These are recent helpdesk ticket subjects. 
        Identify if there is a surge in a specific type of request (e.g., LTL shipping, spare parts, returns).
        
        Subjects:
        {subjects}
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing tickets: {str(e)}"
