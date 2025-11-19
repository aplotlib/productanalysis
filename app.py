import streamlit as st
import pandas as pd
from odoo_processor import OdooProcessor
from return_processor import ReturnReportProcessor
from enhanced_ai_analysis import AIAnalysisEngine

def main():
    st.title("Product Analysis Dashboard 2.0")
    
    # Initialize Processors
    odoo_proc = OdooProcessor()
    return_proc = ReturnReportProcessor()
    ai_engine = AIAnalysisEngine() # Ensure GEMINI_API_KEY is in .env or secrets

    uploaded_files = st.file_uploader("Upload Odoo Exports (Inventory, Helpdesk, Returns)", accept_multiple_files=True)

    if uploaded_files:
        inventory_data = None
        helpdesk_data = None
        return_data = None

        for file in uploaded_files:
            # 1. Identify and Process Inventory File
            if "Inventory" in file.name:
                st.info(f"Processing Inventory Master: {file.name}")
                inventory_data = odoo_proc.load_inventory(file.getvalue())
                st.dataframe(inventory_data.head())

            # 2. Identify and Process Hierarchical Pivot Report
            elif "Return Report" in file.name or "Pivot" in file.name:
                st.info(f"Processing Pivot Return Report: {file.name}")
                return_data = return_proc.process(file.getvalue())
                st.success(f"Successfully flattened {len(return_data)} return rows.")
                st.dataframe(return_data.head())

            # 3. Identify and Process Helpdesk Tickets
            elif "Helpdesk" in file.name:
                st.info(f"Processing Helpdesk Tickets: {file.name}")
                helpdesk_data = odoo_proc.process_helpdesk(file.getvalue())
                st.dataframe(helpdesk_data.head())

        # --- Visualizations & AI Insights ---
        
        if return_data is not None:
            st.subheader("Return Analytics")
            # Simple Bar Chart of Returns by SKU
            sku_counts = return_data.groupby('SKU')['Total Returns'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(sku_counts)
            
        if helpdesk_data is not None:
            st.subheader("AI Helpdesk Analysis")
            if st.button("Analyze Ticket Trends"):
                with st.spinner("Consulting Gemini..."):
                    insight = ai_engine.analyze_helpdesk_trends(helpdesk_data)
                    st.markdown(insight)

if __name__ == "__main__":
    main()
