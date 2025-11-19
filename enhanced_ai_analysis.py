import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

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

# Check for dependencies
requests, has_requests = safe_import('requests')

# API Configuration
API_TIMEOUT = 60  # Increased for longer plan generation
MAX_RETRIES = 2
MAX_TOKENS = 2500

class APIClient:
    """Robust OpenAI API client with error handling"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                for key_name in ["openai_api_key", "OPENAI_API_KEY"]:
                    if key_name in st.secrets:
                        return st.secrets[key_name]
        except:
            pass
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        return None
    
    def is_available(self) -> bool:
        return bool(self.api_key and has_requests)
    
    def call_api(self, messages: List[Dict[str, str]], 
                model: str = "gpt-4o",
                temperature: float = 0.3,
                max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """Make API call with retry logic"""
        
        if not self.is_available():
            return {"success": False, "error": "API not available", "result": None}
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "result": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {})
                    }
                elif response.status_code == 429:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {"success": False, "error": f"API error {response.status_code}", "result": None}
                    
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"success": False, "error": str(e), "result": None}
        
        return {"success": False, "error": "Max retries exceeded", "result": None}

# --- NEW PROJECT PLANNING CLASSES ---

class ProjectPlanTemplates:
    """Templates for generating Quality Project Plan documents based on uploaded QPP PDFs"""
    
    @staticmethod
    def create_comprehensive_plan_prompt(product_name, goal, timeline) -> str:
        return f"""
        Act as a Medical Device Quality Project Manager. Create a "Comprehensive Quality Project Plan" (QPP) for:
        Product: {product_name}
        Goal: {goal}
        Launch Timeline: {timeline}

        Generate these 5 specific parts based on Juran, Deming, and Ohno principles. Use professional technical language.

        **PART 1: Juran "Fitness for Use" Charter**
        - **Device Description:** Brief overview and Intended Use.
        - **User Profile:** Who is the primary user?
        - **User Needs & CTQs:** List 3 key User Needs linked to 3 measurable Design Inputs (Critical-to-Quality specs).
        - **Regulatory:** Target Markets (US/EU) and likely Device Classification.

        **PART 2: Deming "System of Profound Knowledge"**
        - **Project Risk Assessment:** List 3 risks to the *project* (resources, timeline, scope) and specific mitigations.
        - **Supplier Qualification:** Identify 2 critical component types and their required qualification method (e.g., On-site Audit, First Article Inspection).

        **PART 3: Ohno "Lean Design" Execution**
        - **Usability (URRA):** List 2 potential misuse scenarios (Genchi Genbutsu).
        - **Design FMEA:** List the TOP 3 high-risk failure modes with estimated Severity, Occurrence, Detection, and RPN scores.

        **PART 4: ASQ "Quality Toolbox" V&V**
        - **V&V Strategy:** List 3 key validation tests required (e.g., Biocompatibility ISO 10993, Drop Test, Sterility).
        - **Process FMEA:** List 2 potential manufacturing process risks and controls.

        **PART 5: PMI Design Transfer**
        - **Residual Risk:** Write a formal "Statement of Residual Risk" for the final report.
        - **Post-Market Surveillance:** List 3 specific data sources to monitor after launch.

        Output strictly in Markdown format. Use bold headers for the Parts.
        """

    @staticmethod
    def create_streamlined_plan_prompt(product_name, goal, timeline) -> str:
        return f"""
        Act as a Medical Device Quality Lead. Create a "Streamlined Quality Project Plan" for a low-risk/non-critical device:
        Product: {product_name}
        Goal: {goal}
        Timeline: {timeline}

        Generate this specific 3-section structure suitable for a Fast Track project:

        **1. Device Charter**
        - **Intended Use & Classification:** Brief statement.
        - **Top 3 Critical-to-Quality (CTQ) Requirements:** Specific, testable metrics (e.g., weight load, accuracy, material strength).

        **2. Core Risk Management Summary**
        - Create a table of the "Top 3 Risks & Mitigations" covering Design, Usability, or Process risks.

        **3. V&V & Transfer Summary**
        - **Validation Checklist:** Bullet points of key activities (e.g., User Testing, Packaging Validation).
        - **Readiness Statement:** A formal statement confirming design transfer readiness.

        Output strictly in Markdown. Keep it concise, actionable, and focused on speed without sacrificing safety.
        """

class ProjectPlanGenerator:
    """Generates full project documentation using AI"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.templates = ProjectPlanTemplates()

    def generate_plan(self, product_name, goal, timeline, mode="critical"):
        """Generates the plan based on the selected mode"""
        
        if mode == "critical":
            prompt = self.templates.create_comprehensive_plan_prompt(product_name, goal, timeline)
            system_role = "You are a Regulatory Affairs and Quality Assurance Director with expertise in ISO 13485 and FDA QSR."
        else:
            prompt = self.templates.create_streamlined_plan_prompt(product_name, goal, timeline)
            system_role = "You are a Product Development Lead focused on agile execution and efficient quality compliance."

        response = self.api_client.call_api([
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ])

        return {
            "success": response['success'],
            "content": response['result'] if response['success'] else "Error generating plan.",
            "mode": mode
        }

# Keep existing classes to ensure backward compatibility if needed
class EnhancedAIAnalyzer:
    def __init__(self):
        self.api_client = APIClient()
    
    def get_api_status(self):
        status = self.api_client.is_available()
        return {"available": status}

# Export main classes
__all__ = ['EnhancedAIAnalyzer', 'APIClient', 'ProjectPlanGenerator']
