import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime

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
        Removes common suffixes (-FBA, -REF), special chars, and casing issues.
        """
        if pd.isna(sku):
            return "UNKNOWN"
        
        sku = str(sku).upper().strip()
        
        # Remove Common Odoo/Amazon Suffixes that might cause mismatches
        sku = re.sub(r'(-FBA|-NEW|-REF|-US|-CA|-DS)$', '', sku)
        
        # Remove non-alphanumeric characters (optional, depends on SKU strictness)
        # We keep hyphens as they are common in SKUs like 'SUP-101'
        sku = re.sub(r'[^A-Z0-9\-]', '', sku)
        
        return sku

    @staticmethod
    def extract_parent_sku(sku):
        """Extracts the base product SKU from a variant SKU (e.g., SUP-101-BLU -> SUP-101)."""
        match = re.match(r"^([A-Z]+[0-9]+)", sku)
        return match.group(1) if match else sku

    @staticmethod
    def get_category(sku):
        """Heuristic category extraction from SKU prefix."""
        if pd.isna(sku): return "Other"
        match = re.match(r"^([A-Z]+)", str(sku))
        return match.group(1) if match else "Other"

    @staticmethod
    def smart_read_excel(file) -> pd.DataFrame:
        """
        Scans the first 20 rows of an Excel file to find the actual header row.
        Solves the issue of Odoo reports having metadata lines at the top.
        """
        try:
            # Read first few rows without header to scan content
            preview = pd.read_excel(file, header=None, nrows=20)
            
            # Keywords to identify the header row in Odoo exports
            header_keywords = ['sku', 'default code', 'product', 'reference', 'order', 'quantity', 'sales', 'ticket', 'subject', 'qty']
            
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
            
            # Normalize column names (strip whitespace, lower case)
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('/', '_') for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error reading excel: {e}")
            return pd.DataFrame()

    def process_sales_file(self, file):
        """Process Odoo Sales/Forecast File."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Fuzzy Column Matching
        sku_col = next((c for c in df.columns if c in ['sku', 'default_code', 'product_reference', 'internal_reference']), None)
        qty_col = next((c for c in df.columns if any(x in c for x in ['total', 'quantity', 'qty', 'forecast', 'volume'])), None)
        product_col = next((c for c in df.columns if 'product' in c), None)

        if not sku_col or not qty_col:
            logger.warning("Could not identify SKU or Quantity column in Sales file.")
            return pd.DataFrame()

        # Clean Data
        df['clean_sku'] = df[sku_col].apply(self.normalize_sku)
        df['parent_sku'] = df['clean_sku'].apply(self.extract_parent_sku)
        
        # Convert qty to numeric, handle Odoo's potential string formatting
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
        sku_col = next((c for c in df.columns if c in ['sku', 'default_code', 'product', 'variant']), None)
        reason_col = next((c for c in df.columns if any(x in c for x in ['reason', 'comment', 'note'])), None)
        qty_col = next((c for c in df.columns if any(x in c for x in ['qty', 'quantity', 'count'])), None)
        
        if not sku_col: return pd.DataFrame()

        df['clean_sku'] = df[sku_col].apply(self.normalize_sku)
        
        # If no qty column, assume 1 row = 1 return
        if not qty_col:
            df['return_qty'] = 1
        else:
            df['return_qty'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1)

        df['return_reason'] = df[reason_col].fillna('Unspecified') if reason_col else 'Unspecified'

        # Aggregate returns by SKU and Reason
        # This preserves the granular reason data while allowing total sums
        return df

    def process_helpdesk_file(self, file):
        """Process Helpdesk Ticket Export."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Columns: Subject, Description, maybe Product/SKU
        sku_col = next((c for c in df.columns if c in ['sku', 'product', 'product_template']), None)
        text_cols = [c for c in df.columns if any(x in c for x in ['subject', 'description', 'name', 'content', 'message'])]

        clean_data = []
        for _, row in df.iterrows():
            # Combine all text fields for analysis
            text_content = " | ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
            
            sku = "UNKNOWN"
            if sku_col and pd.notna(row[sku_col]):
                sku = self.normalize_sku(row[sku_col])
            else:
                # Attempt to extract SKU from text if column missing (Regex search for SKU pattern)
                # Pattern: 3-4 letters, 3-5 digits (e.g., SUP101, LVA2024)
                match = re.search(r'\b([A-Z]{3,4}-?[0-9]{3,5})\b', text_content)
                if match:
                    sku = self.normalize_sku(match.group(1))

            clean_data.append({
                'clean_sku': sku,
                'ticket_text': text_content,
                'ticket_date': datetime.now() # Placeholder if date col missing
            })

        return pd.DataFrame(clean_data)

    def merge_datasets(self, sales_df, returns_df, helpdesk_df=None):
        """
        The 'Deep Logic' engine: 
        1. Creates a master SKU list.
        2. XLOOKUPs sales, returns, and tickets to that master list.
        3. Calculates rates and extracts top issues.
        """
        if sales_df.empty and returns_df.empty:
            return pd.DataFrame()

        # 1. Master SKU List (Union of Sales and Returns)
        all_skus = pd.concat([
            sales_df['clean_sku'] if not sales_df.empty else pd.Series(),
            returns_df['clean_sku'] if not returns_df.empty else pd.Series()
        ]).unique()

        master_df = pd.DataFrame({'clean_sku': all_skus})

        # 2. Merge Sales (Left Join)
        if not sales_df.empty:
            sales_agg = sales_df.groupby('clean_sku').agg({
                'sales_qty': 'sum',
                'product_name': 'first',
                'parent_sku': 'first'
            }).reset_index()
            master_df = pd.merge(master_df, sales_agg, on='clean_sku', how='left')
        
        # 3. Merge Returns (Left Join)
        if not returns_df.empty:
            # Total Returns count
            returns_agg = returns_df.groupby('clean_sku')['return_qty'].sum().reset_index()
            master_df = pd.merge(master_df, returns_agg, on='clean_sku', how='left')
            
            # Top 3 Return Reasons (List aggregation)
            top_reasons = returns_df.groupby('clean_sku')['return_reason'].apply(
                lambda x: list(x.value_counts().index[:3])
            ).reset_index()
            master_df = pd.merge(master_df, top_reasons, on='clean_sku', how='left')

        # 4. Fill NaNs
        master_df['sales_qty'] = master_df['sales_qty'].fillna(0)
        master_df['return_qty'] = master_df['return_qty'].fillna(0)
        master_df['product_name'] = master_df['product_name'].fillna('Unknown Product')
        
        # 5. Calculate Metrics
        # Return Rate logic: Returns / Sales * 100
        master_df['return_rate'] = (master_df['return_qty'] / master_df['sales_qty'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Handle Edge Case: Returns exist but Sales are 0 (likely data gap or old sales)
        # We mark these as 100% return rate to flag them, or cap them.
        mask_zero_sales = (master_df['sales_qty'] == 0) & (master_df['return_qty'] > 0)
        master_df.loc[mask_zero_sales, 'return_rate'] = 100.0
        
        # 6. Create Categories/Parents if missing
        if 'parent_sku' not in master_df.columns or master_df['parent_sku'].isna().all():
             master_df['parent_sku'] = master_df['clean_sku'].apply(self.extract_parent_sku)
        
        master_df['Category'] = master_df['clean_sku'].apply(self.get_category)
        
        # 7. Attach Helpdesk Context
        if helpdesk_df is not None and not helpdesk_df.empty:
            ticket_counts = helpdesk_df.groupby('clean_sku').size().reset_index(name='ticket_count')
            master_df = pd.merge(master_df, ticket_counts, on='clean_sku', how='left')
            master_df['ticket_count'] = master_df['ticket_count'].fillna(0)
        
        # Sort by volume (Business Impact)
        master_df = master_df.sort_values('sales_qty', ascending=False)

        return master_df
