import pandas as pd
import numpy as np
import re
import io
import logging
from datetime import datetime, timedelta
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdooProcessor:
    """
    Advanced processor for Odoo Sales, Returns, and Helpdesk exports.
    Includes logic for 'XLOOKUP' style matching and SKU normalization.
    """

    @staticmethod
    def normalize_sku(sku):
        """
        Aggressively cleans SKUs to ensure matches between disparate systems.
        Removes common suffixes, special chars, and casing issues.
        """
        if pd.isna(sku):
            return "UNKNOWN"
        
        sku = str(sku).upper().strip()
        
        # Remove Common Odoo/Amazon Suffixes that might cause mismatches
        sku = re.sub(r'(-FBA|-NEW|-REF|-US|-CA)$', '', sku)
        
        # Remove non-alphanumeric characters (optional, depends on SKU strictness)
        # keeping hyphens as they are common in SKUs
        sku = re.sub(r'[^A-Z0-9\-]', '', sku)
        
        return sku

    @staticmethod
    def extract_parent_sku(sku):
        """Extracts the base product SKU from a variant SKU."""
        # Assumes parent is the part before the first hyphen if it starts with letters
        # Adjust regex based on your specific SKU naming convention
        match = re.match(r"^([A-Z]+[0-9]+)", sku)
        return match.group(1) if match else sku

    @staticmethod
    def smart_read_excel(file) -> pd.DataFrame:
        """
        Scans the first 20 rows of an Excel file to find the actual header row.
        Solves the issue of Odoo reports having metadata lines at the top.
        """
        try:
            # Read first few rows without header
            preview = pd.read_excel(file, header=None, nrows=20)
            
            # Keywords to identify the header row
            header_keywords = ['sku', 'default code', 'product', 'reference', 'order', 'quantity', 'sales', 'ticket', 'subject']
            
            header_idx = 0
            max_matches = 0
            
            for idx, row in preview.iterrows():
                # Convert row to string and check for keyword density
                row_str = row.astype(str).str.lower().tolist()
                matches = sum(1 for kw in header_keywords if any(kw in cell for cell in row_str))
                
                if matches > max_matches:
                    max_matches = matches
                    header_idx = idx
            
            # Reset file pointer and read with correct header
            file.seek(0)
            df = pd.read_excel(file, header=header_idx)
            
            # Normalize column names
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error reading excel: {e}")
            return pd.DataFrame()

    def process_sales_file(self, file):
        """Process Odoo Sales/Forecast File."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Identify key columns
        sku_col = next((c for c in df.columns if c in ['sku', 'default_code', 'product_reference', 'internal_reference']), None)
        qty_col = next((c for c in df.columns if any(x in c for x in ['total', 'quantity', 'qty', 'forecast'])), None)
        product_col = next((c for c in df.columns if 'product' in c), None)

        if not sku_col or not qty_col:
            logger.warning("Could not identify SKU or Quantity column in Sales file.")
            return pd.DataFrame()

        # Clean Data
        df['clean_sku'] = df[sku_col].apply(self.normalize_sku)
        df['parent_sku'] = df['clean_sku'].apply(self.extract_parent_sku)
        
        # Convert qty to numeric, handle Odoo's sometimes messy formatting
        df['sales_qty'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
        
        # Aggregate by SKU (handling duplicates/multiple orders)
        agg_df = df.groupby(['clean_sku', 'parent_sku']).agg({
            'sales_qty': 'sum',
            product_col: 'first' if product_col else lambda x: 'Unknown'
        }).reset_index()
        
        agg_df.rename(columns={product_col: 'product_name'}, inplace=True)
        return agg_df

    def process_returns_file(self, file):
        """Process Odoo Pivot Return Report."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Identify key columns
        sku_col = next((c for c in df.columns if c in ['sku', 'default_code', 'product']), None)
        reason_col = next((c for c in df.columns if 'reason' in c), None)
        qty_col = next((c for c in df.columns if any(x in c for x in ['qty', 'quantity', 'count'])), None)
        
        # If Pivot table (Dates across top), melt it? 
        # Assuming flat export for robustness, but if pivot, we might need unstack logic.
        # For now, we assume standard list view export from Odoo.

        if not sku_col:
            return pd.DataFrame()

        df['clean_sku'] = df[sku_col].apply(self.normalize_sku)
        
        # If no qty column, assume 1 row = 1 return
        if not qty_col:
            df['return_qty'] = 1
        else:
            df['return_qty'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1)

        df['return_reason'] = df[reason_col].fillna('Unspecified') if reason_col else 'Unspecified'

        # Aggregate
        agg_df = df.groupby(['clean_sku', 'return_reason']).agg({
            'return_qty': 'sum'
        }).reset_index()

        return agg_df

    def process_helpdesk_file(self, file):
        """Process Helpdesk Ticket Export."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Columns: Subject, Description, maybe Product/SKU
        sku_col = next((c for c in df.columns if c in ['sku', 'product']), None)
        text_cols = [c for c in df.columns if any(x in c for x in ['subject', 'description', 'name', 'content'])]

        clean_data = []
        for _, row in df.iterrows():
            text_content = " | ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
            
            sku = "UNKNOWN"
            if sku_col and pd.notna(row[sku_col]):
                sku = self.normalize_sku(row[sku_col])
            else:
                # Attempt to extract SKU from text if column missing
                # Look for pattern like "ABC1234"
                match = re.search(r'\b([A-Z]{3,4}[0-9]{3,5})\b', text_content)
                if match:
                    sku = match.group(1)

            clean_data.append({
                'clean_sku': sku,
                'ticket_text': text_content,
                'ticket_date': datetime.now() # Placeholder if date col missing
            })

        return pd.DataFrame(clean_data)

    def merge_datasets(self, sales_df, returns_df, helpdesk_df=None):
        """
        Performs the 'XLOOKUP' logic to merge Sales and Returns.
        Calculates Return Rates and matches Helpdesk data.
        """
        if sales_df.empty and returns_df.empty:
            return pd.DataFrame()

        # 1. Master SKU List (Union of both)
        all_skus = pd.concat([
            sales_df['clean_sku'] if not sales_df.empty else pd.Series(),
            returns_df['clean_sku'] if not returns_df.empty else pd.Series()
        ]).unique()

        master_df = pd.DataFrame({'clean_sku': all_skus})

        # 2. Merge Sales (Left Join)
        if not sales_df.empty:
            # Aggregate sales to distinct SKU first
            sales_agg = sales_df.groupby('clean_sku').agg({
                'sales_qty': 'sum',
                'product_name': 'first',
                'parent_sku': 'first'
            }).reset_index()
            master_df = pd.merge(master_df, sales_agg, on='clean_sku', how='left')
        
        # 3. Merge Returns (Left Join)
        if not returns_df.empty:
            # Aggregate returns total per SKU
            returns_agg = returns_df.groupby('clean_sku')['return_qty'].sum().reset_index()
            master_df = pd.merge(master_df, returns_agg, on='clean_sku', how='left')
            
            # Capture top return reasons
            top_reasons = returns_df.groupby('clean_sku')['return_reason'].apply(lambda x: list(x.unique())[:3]).reset_index()
            master_df = pd.merge(master_df, top_reasons, on='clean_sku', how='left')

        # 4. Fill NaNs
        master_df['sales_qty'] = master_df['sales_qty'].fillna(0)
        master_df['return_qty'] = master_df['return_qty'].fillna(0)
        master_df['product_name'] = master_df['product_name'].fillna('Unknown Product')
        
        # 5. Calculate Metrics
        master_df['return_rate'] = (master_df['return_qty'] / master_df['sales_qty'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 6. Create Categories if missing
        if 'parent_sku' not in master_df.columns:
             master_df['parent_sku'] = master_df['clean_sku'].apply(self.extract_parent_sku)
        
        # 7. Attach Helpdesk Context (Count of tickets)
        if helpdesk_df is not None and not helpdesk_df.empty:
            ticket_counts = helpdesk_df.groupby('clean_sku').size().reset_index(name='ticket_count')
            master_df = pd.merge(master_df, ticket_counts, on='clean_sku', how='left')
            master_df['ticket_count'] = master_df['ticket_count'].fillna(0)
        
        # Sort by volume (Impact)
        master_df = master_df.sort_values('sales_qty', ascending=False)

        return master_df
