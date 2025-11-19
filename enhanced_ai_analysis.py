import os
import json
import streamlit as st
import google.generativeai as genai

# Optional: Import OpenAI if installed
try:
    import openai
except ImportError:
    openai = None

class IntelligenceEngine:
    """
    The AI Core. Handles text generation, vision analysis, and CAPA drafting.
    """
    def __init__(self):
        self.client = None
        self.provider = None
        self.model_name = None
        self.available = False
        self.connection_error = None

    def _get_key(self, names, manual_key=None):
        if manual_key: return manual_key
        # Check Streamlit secrets and Environment variables
        for name in names:
            if hasattr(st, "secrets") and name in st.secrets: return st.secrets[name]
            if name in os.environ: return os.environ[name]
        return None

    def configure_client(self, provider_choice, manual_key_input=None):
        self.available = False
        self.connection_error = None
        
        try:
            if "Gemini" in provider_choice:
                api_key = self._get_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], manual_key_input)
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai
                    self.provider = "Google Gemini"
                    self.model_name = "gemini-1.5-flash" if "Flash" in provider_choice else "gemini-1.5-pro"
                    self.available = True
                else:
                    self.connection_error = "Missing Google API Key"

            elif "GPT" in provider_choice:
                if not openai:
                    self.connection_error = "OpenAI library not installed."
                    return
                
                api_key = self._get_key(["OPENAI_API_KEY"], manual_key_input)
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.provider = "OpenAI"
                    self.model_name = "gpt-4o" if "4o" in provider_choice else "gpt-4o-mini"
                    self.available = True
                else:
                    self.connection_error = "Missing OpenAI API Key"
                    
        except Exception as e:
            self.connection_error = f"Connection Failed: {str(e)}"
            self.available = False

    def generate(self, prompt, temperature=0.3):
        if not self.available: return f"⚠️ AI Offline: {self.connection_error}"
        try:
            if "Gemini" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content(prompt, generation_config={'temperature': temperature})
                return response.text
            elif "OpenAI" in self.provider:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e: 
            return f"Generation Error: {str(e)}"

    def analyze_image(self, image, prompt):
        """Analyzes uploaded images (Vision)."""
        if not self.available: return f"AI Offline: {self.connection_error}"
        try:
            if "Gemini" in self.provider:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content([prompt, image])
                return response.text
            elif "OpenAI" in self.provider:
                # Simplified OpenAI Vision implementation could go here
                return "Vision analysis currently optimized for Gemini."
        except Exception as e: 
            return f"Vision Error: {str(e)}"

    def generate_capa_draft(self, context):
        """Generates a structured CAPA JSON."""
        prompt = f"""
        You are a Quality Assurance Expert. Create a CAPA investigation JSON for:
        Product: {context.get('product')}
        Issue: {context.get('issue')}
        
        Return ONLY valid JSON with these keys:
        - issue_description
        - root_cause_analysis (detailed)
        - immediate_action
        - corrective_action
        - effectiveness_check
        """
        res = self.generate(prompt, temperature=0.5)
        try:
            # Clean markdown code blocks if present
            if "```json" in res: 
                res = res.split("```json")[1].split("```")[0]
            elif "```" in res:
                res = res.split("```")[1].split("```")[0]
            return json.loads(res)
        except: 
            return {"issue_description": res, "root_cause_analysis": "Failed to parse JSON"}
