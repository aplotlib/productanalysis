import streamlit as st
import pandas as pd
from odoo_processor import OdooProcessor
from return_processor import ReturnReportProcessor

# --- CONFIGURATION ---
st.set_page_config(page_title="Product Intelligence HQ", layout="wide")

# Custom CSS for "Powerful" Look
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 { color: #1e293b; }
    h3 { color: #334155; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 Product Intelligence HQ")
    st.markdown("### Centralized Quality & Performance Analytics")

    # Initialize Processors
    odoo_proc = OdooProcessor()
    return_proc = ReturnReportProcessor()

    # --- SIDEBAR: UPLOAD CENTER ---
    with st.sidebar:
        st.header("Data Ingestion")
        st.info("Upload your Odoo exports here. The app will automatically link them.")
        
        uploaded_files = st.file_uploader(
            "Drop Files (Inventory, Returns, Helpdesk)", 
            accept_multiple_files=True,
            type=['csv', 'xlsx']
        )

    if not uploaded_files:
        st.warning("waiting for data... Upload 'Inventory Forecast', 'Helpdesk Tickets', or 'Return Reports' to begin.")
        st.stop()

    # --- DATA PROCESSING ENGINE ---
    inventory_df = pd.DataFrame()
    helpdesk_df = pd.DataFrame()
    returns_df = pd.DataFrame()

    # 1. Process Inventory FIRST (Master Key)
    for file in uploaded_files:
        if "Inventory" in file.name:
            with st.spinner("Indexing Inventory..."):
                inventory_df = odoo_proc.load_inventory_master(file.getvalue())
            st.toast(f"Inventory Loaded: {len(inventory_df)} SKUs", icon="✅")

    # 2. Process Other Files
    for file in uploaded_files:
        try:
            if "Inventory" in file.name:
                continue # Already done
            
            elif "Return" in file.name or "Pivot" in file.name:
                returns_df = return_proc.process(file.getvalue())
                st.toast(f"Parsed Returns: {len(returns_df)} records", icon="📉")

            elif "Helpdesk" in file.name:
                helpdesk_df = odoo_proc.process_helpdesk(file.getvalue())
                st.toast(f"Parsed Tickets: {len(helpdesk_df)} tickets", icon="🎫")
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    # --- THE INTELLIGENCE DASHBOARD ---
    
    # 1. Top Level Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Active SKUs", len(inventory_df) if not inventory_df.empty else "0")
    col2.metric("Total Returns Logged", int(returns_df['Total Returns'].sum()) if not returns_df.empty else "0")
    col3.metric("Support Tickets", len(helpdesk_df) if not helpdesk_df.empty else "0")

    st.divider()

    # 2. The "Master Merge" View
    if not inventory_df.empty and (not helpdesk_df.empty or not returns_df.empty):
        st.subheader("🔥 High-Risk Products (Combined Data)")
        
        # Merge data
        master_data = odoo_proc.merge_data(inventory_df, helpdesk_df, returns_df)
        
        if not master_data.empty:
            # Calculate a "Risk Score"
            # Simple logic: Returns + Tickets = Pain
            master_data['Risk Score'] = master_data['Ticket Count'] + master_data['Total Returns']
            
            # Show Top 10 Worst Products
            top_risk = master_data.sort_values('Risk Score', ascending=False).head(10)
            
            # Clean up columns for display
            display_cols = ['Product SKU', 'Product Title', 'Ticket Count', 'Total Returns', 'Risk Score', 'On Hand']
            # Only show columns that actually exist
            final_cols = [c for c in display_cols if c in top_risk.columns]
            
            st.dataframe(
                top_risk[final_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Visual Chart
            st.bar_chart(top_risk.set_index('Product SKU')['Risk Score'])
            
        else:
            st.info("Data merged but result was empty. Check SKU matching.")

    # 3. Deep Dive Tabs
    tab1, tab2 = st.tabs(["🎫 Helpdesk Intelligence", "📉 Return Analytics"])
    
    with tab1:
        if not helpdesk_df.empty:
            st.dataframe(helpdesk_df)
            st.caption("Note: 'Inferred_SKU' is extracted from Ticket Subjects or Odoo IDs.")
        else:
            st.markdown("*Upload Helpdesk file to see ticket analysis*")

    with tab2:
        if not returns_df.empty:
            st.dataframe(returns_df)
        else:
            st.markdown("*Upload Pivot Return Report to see return analysis*")

if __name__ == "__main__":
    main()
