# Add to imports
from enhanced_ai_analysis import ProjectPlanGenerator

def render_project_planner():
    st.markdown("## 🚀 Quality Project Command Center")
    st.markdown("Execute a Quality Project Plan (QPP) aligned with ISO 13485 & FDA QSR.")

    # --- PART 0: THE APPLICABILITY SCREEN (Screening/Gate) ---
    with st.expander("📋 Part 0: QPP Screening Gate", expanded=True):
        st.info("Answer these questions to determine if a **Critical Path (Full)** or **Fast Track (Streamlined)** plan is required.")
        
        col1, col2 = st.columns(2)
        with col1:
            q1 = st.checkbox("1. Is the device sterile?")
            q2 = st.checkbox("2. Is it Class I w/ special controls, Class II, or higher?")
            q3 = st.checkbox("3. Is it an active instrument or software-driven?")
        with col2:
            q4 = st.checkbox("4. Is it a mobility item (supports weight/movement)?")
            q5 = st.checkbox("5. Does it have complex moving parts (e.g., knee brace)?")
            q6 = st.checkbox("6. Is it high financial risk?")

        # Logic from "Product Changes...pdf"
        is_critical = any([q1, q2, q3, q4, q5, q6])
        
        if is_critical:
            recommended_mode = "Critical Path (Comprehensive)"
            mode_color = "red"
            mode_key = "critical"
            st.markdown(f"### 🔴 Recommendation: **{recommended_mode}**")
            st.caption("Based on your inputs, this project requires the Full QPP (Parts 1-5) for regulatory compliance.")
        else:
            recommended_mode = "Fast Track (Streamlined)"
            mode_color = "green"
            mode_key = "fast_track"
            st.markdown(f"### 🟢 Recommendation: **{recommended_mode}**")
            st.caption("This qualifies for the Streamlined QPP.")

    # --- PROJECT INPUTS ---
    st.markdown("---")
    st.markdown("### 📝 Project Definition")
    
    with st.form("project_input_form"):
        # Allow override
        selected_mode = st.radio("Confirm Planning Mode:", 
                                ["Critical Path (Comprehensive)", "Fast Track (Streamlined)"],
                                index=0 if is_critical else 1)
        
        col1, col2 = st.columns(2)
        with col1:
            product_name = st.text_input("Product Name", placeholder="e.g., Post-Op Shoe V2")
            timeline = st.text_input("Target Timeline", placeholder="e.g., Q3 2025 Launch")
        
        goal = st.text_area("Primary Goal / Problem Statement", 
                           placeholder="e.g., Reduce return rate to <5% by fixing sizing chart issues.")
        
        submitted = st.form_submit_button("✨ Generate Quality Plan")

    # --- GENERATION LOGIC ---
    if submitted and product_name and goal:
        # Check for API Key
        ai_status = check_ai_status()
        if not ai_status.get('available'):
            st.error("❌ OpenAI API Key required for plan generation.")
            return

        # Map selection to internal key
        final_mode = "critical" if "Critical" in selected_mode else "fast_track"

        with st.spinner(f"🤖 acting as Quality Manager... Generating {selected_mode} QPP..."):
            try:
                # Initialize Generator
                generator = ProjectPlanGenerator(st.session_state.ai_analyzer.api_client)
                
                # Generate
                plan_result = generator.generate_plan(product_name, goal, timeline, final_mode)
                
                if plan_result['success']:
                    st.session_state.generated_plan = plan_result
                    st.success("✅ Quality Project Plan Generated!")
                else:
                    st.error("Failed to generate plan.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # --- DISPLAY RESULTS ---
    if 'generated_plan' in st.session_state:
        plan = st.session_state.generated_plan
        content = plan['content']
        
        st.markdown("---")
        st.markdown(f"## 📄 {plan['mode'].title().replace('_', ' ')} Quality Plan")
        
        # Display content in a clean container
        with st.container():
            st.markdown(content)
        
        # Download Button
        st.download_button(
            label="📥 Download QPP as Markdown",
            data=content,
            file_name=f"QPP_{product_name.replace(' ', '_')}.md",
            mime="text/markdown"
        )

def main():
    # ... [Keep existing Config/CSS] ...
    
    # Initialize session
    initialize_session_state()

    # Sidebar Navigation
    with st.sidebar:
        st.title("Navigate")
        app_mode = st.radio("Go to:", ["Review Analyzer", "Quality Command Center"])
        
        st.markdown("---")
        # [Keep existing Sidebar AI Status]
        display_modern_sidebar()

    # Routing
    if app_mode == "Review Analyzer":
        # ... [Existing Main Logic for Review Analysis] ...
        if st.session_state.current_step == 'upload':
            handle_modern_file_upload()
        elif st.session_state.current_step == 'analysis':
            display_modern_data_summary()
        elif st.session_state.current_step == 'results':
            display_modern_analysis_results()

    elif app_mode == "Quality Command Center":
        render_project_planner()

# ... [Keep existing main execution block] ...
