class ProjectPlanTemplates:
    """Templates for generating Quality Project Plan documents based on uploaded QPP PDFs"""
    
    @staticmethod
    def create_comprehensive_plan_prompt(product_name, goal, timeline) -> str:
        return f"""
        Act as a Medical Device Quality Project Manager. Create a "Comprehensive Quality Project Plan" (QPP) for:
        Product: {product_name}
        Goal: {goal}
        Launch Timeline: {timeline}

        You must generate the content for these 5 specific parts based on Juran, Deming, and Ohno principles:

        **PART 1: Juran "Fitness for Use" Charter**
        - Define Intended Use & User Profile.
        - List 3 User Needs (UN) linked to 3 measurable Design Inputs/CTQs (DI).
        - Regulatory Assessment: Target Markets (assume US/EU) and Classification.

        **PART 2: Deming "System of Profound Knowledge"**
        - Project Risk Assessment: List 3 risks to the *project timeline/budget* (not device risks) and mitigations.
        - Key Supplier Qualification: Identify 2 critical component types and required qualification (e.g., Audit, FAI).

        **PART 3: Ohno "Lean Design" Execution**
        - Usability (URRA): List 2 potential user misuse scenarios.
        - Design FMEA (dFMEA): List the TOP 3 high-risk failure modes with Severity, Occurrence, Detection, and RPN.

        **PART 4: ASQ "Quality Toolbox" V&V**
        - V&V Strategy: List 3 key validation tests required (e.g., Biocompatibility, Drop Test, Sterility).
        - Process FMEA (pFMEA): List 2 manufacturing process risks.

        **PART 5: PMI Design Transfer**
        - Write a "Statement of Residual Risk" for the final report.
        - Post-Market Surveillance: List 3 data sources to monitor after launch.

        Output strictly in Markdown. Use bold headers for Parts.
        """

    @staticmethod
    def create_streamlined_plan_prompt(product_name, goal, timeline) -> str:
        return f"""
        Act as a Medical Device Quality Lead. Create a "Streamlined Quality Project Plan" for a low-risk device:
        Product: {product_name}
        Goal: {goal}
        Timeline: {timeline}

        Generate this specific 3-section structure:

        **1. Device Charter**
        - Intended Use & Classification.
        - Top 3 Critical-to-Quality (CTQ) Requirements (Measurable specs).

        **2. Core Risk Management Summary**
        - Combine dFMEA and URRA into a single table of the "Top 3 Risks & Mitigations".

        **3. V&V & Transfer Summary**
        - Checklist of Key Validation Activities (The "Proof").
        - Design Transfer Readiness Statement.

        Output strictly in Markdown. Keep it concise and actionable.
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
            system_role = "You are a Regulatory Affairs and Quality Assurance Director."
        else:
            prompt = self.templates.create_streamlined_plan_prompt(product_name, goal, timeline)
            system_role = "You are a Product Development Lead focused on speed and efficiency."

        response = self.api_client.call_api([
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ], max_tokens=2500) # Increased for full plan

        return {
            "success": response['success'],
            "content": response['result'] if response['success'] else "Error generating plan.",
            "mode": mode
        }
