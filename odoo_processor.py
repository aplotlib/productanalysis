import pandas as pd
import numpy as np
import re
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdooProcessor:
    """
    The logic core: Cleans Odoo's messy exports and links Ticket IDs to real SKUs.
    """
    def __init__(self):
        # Maps Odoo internal ID (e.g., '52') to SKU (e.g., 'MOB1001')
        self.product_id_map = {} 

    @staticmethod
    def smart_read(file_content):
        """Robustly reads CSV or Excel bytes."""
        try:
            # 1. Try reading as CSV first
            return pd.read_csv(io.BytesIO(file_content))
        except:
            # 2. Fallback to Excel
            return pd.read_excel(io.BytesIO(file_content))

    def load_inventory_master(self, file_content):
        """
        Loads the Inventory Forecast file to create the Master Key (SKU <-> ID).
        """
        try:
            df = self.smart_read(file_content)
            df.columns = [c.strip() for c in df.columns]
            
            # Attempt to map IDs if an ID column exists
            if 'ID' in df.columns and 'SKU' in df.columns:
                 self.product_id_map = pd.Series(df.SKU.values, index=df.ID.astype(str).values).to_dict()

            if 'SKU' in df.columns:
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Inventory Load Error: {e}")
            return pd.DataFrame()

    def process_helpdesk(self, file_content):
        """
        Ingests Helpdesk Tickets and extracts the Product SKU.
        """
        df = self.smart_read(file_content)
        if df.empty: return pd.DataFrame()

        # Identify Columns
        product_col = next((c for c in df.columns if 'Products' in c or 'product' in c), None)
        subject_col = next((c for c in df.columns if 'Subject' in c or 'subject' in c), None)

        clean_rows = []
        for _, row in df.iterrows():
            raw_prod = str(row.get(product_col, ''))
            subject = str(row.get(subject_col, ''))
            
            sku = "UNKNOWN"
            odoo_id = None

            # STRATEGY 1: Clean the Odoo Relation String
            if "helpdesk.ticket.products," in raw_prod:
                try:
                    # Extract '52' from "helpdesk.ticket.products,52"
                    odoo_id = raw_prod.split(',')[1]
                except:
                    pass

            # STRATEGY 2: Regex Match from Subject Line (e.g., "(MOB1027BLU)")
            # Look for pattern: 3-4 Caps, optional hyphen, numbers
            match = re.search(r'\b([A-Z]{3,4}-?[0-9]{3,5}[A-Z0-9]*)\b', subject)
            if match:
                sku = match.group(1)

            clean_rows.append({
                'Ticket ID': row.get('Ticket IDs Sequence', 'N/A'),
                'Date': row.get('Created on', pd.NaT),
                'Subject': subject,
                'Odoo_ID_Raw': odoo_id,
                'Inferred_SKU': sku, # The extracted SKU
                'Priority': row.get('Priority', 'Low')
            })
            
        return pd.DataFrame(clean_rows)

    def merge_data(self, inventory_df, helpdesk_df, returns_df):
        """
        Fuses the three datasets into a single 'Product Performance' table.
        """
        if inventory_df is None or inventory_df.empty:
            return pd.DataFrame()

        master = inventory_df.copy()
        
        # Standardize Master SKU column
        if 'SKU' in master.columns:
            master.rename(columns={'SKU': 'Product SKU'}, inplace=True)

        # 1. Merge Helpdesk Ticket Counts
        if not helpdesk_df.empty:
            ticket_counts = helpdesk_df[helpdesk_df['Inferred_SKU'] != 'UNKNOWN']
            ticket_counts = ticket_counts['Inferred_SKU'].value_counts().reset_index()
            ticket_counts.columns = ['Product SKU', 'Ticket Count']
            
            master = pd.merge(master, ticket_counts, on='Product SKU', how='left')
            master['Ticket Count'] = master['Ticket Count'].fillna(0)

        # 2. Merge Returns (if available)
        if not returns_df.empty and 'SKU' in returns_df.columns:
            return_counts = returns_df.groupby('SKU')['Total Returns'].sum().reset_index()
            return_counts.rename(columns={'SKU': 'Product SKU'}, inplace=True)
            
            master = pd.merge(master, return_counts, on='Product SKU', how='left')
            master['Total Returns'] = master['Total Returns'].fillna(0)

        return master
