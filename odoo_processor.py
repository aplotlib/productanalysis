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
    Robust data processor for Odoo exports. 
    Handles fuzzy matching, financial estimation, and multi-source merging.
    """

    @staticmethod
    def normalize_sku(sku):
        """Standardizes SKUs by removing common platform suffixes."""
        if pd.isna(sku): return "UNKNOWN"
        sku = str(sku).upper().strip()
        # Remove standard suffixes to ensure 'PRO-123-FBA' matches 'PRO-123'
        sku = re.sub(r'(-FBA|-NEW|-REF|-US|-CA|-DS|-WHT|-BLU|-RED)$', '', sku)
        sku = re.sub(r'[^A-Z0-9\-]', '', sku)
        return sku

    @staticmethod
    def extract_parent_sku(sku):
        """Groups variants (e.g., sizes/colors) into parent families."""
        match = re.match(r"^([A-Z]+[0-9]+)", sku)
        return match.group(1) if match else sku

    @staticmethod
    def get_category(sku):
        """Auto-categorizes products based on SKU prefixes."""
        if pd.isna(sku): return "General"
        match = re.match(r"^([A-Z]+)", str(sku))
        prefix_map = {
            'SUP': 'Supports', 'MOB': 'Mobility', 'BATH': 'Bath Safety', 
            'LVA': 'Daily Living', 'DME': 'Equipment', 'RHB': 'Rehab'
        }
        prefix = match.group(1) if match else "Uncategorized"
        return prefix_map.get(prefix, prefix)

    @staticmethod
    def smart_read_excel(file) -> pd.DataFrame:
        """Scans file content to find the true header row, ignoring metadata."""
        try:
            preview = pd.read_excel(file, header=None, nrows=20)
            keywords = ['sku', 'default code', 'product', 'reference', 'order', 'qty', 'sales', 'ticket', 'count', 'measure']
            
            header_idx = 0
            max_matches = 0
            
            for idx, row in preview.iterrows():
                row_str = row.astype(str).str.lower().tolist()
                matches = sum(1 for kw in keywords if any(kw in cell for cell in row_str))
                if matches > max_matches:
                    max_matches = matches
                    header_idx = idx
            
            file.seek(0)
            df = pd.read_excel(file, header=header_idx)
            # Standardize headers to snake_case
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('/', '_').replace('.', '') for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Read Error: {e}")
            return pd.DataFrame()

    def process_sales_file(self, file):
        """Ingests Sales/Forecast data to establish baseline volume."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Dynamic Column Mapping
        sku_col = next((c for c in df.columns if c in ['sku', 'default_code', 'product_reference', 'internal_reference']), None)
        qty_col = next((c for c in df.columns if any(x in c for x in ['total', 'qty', 'quantity', 'forecast', 'demand'])), None)
        price_col = next((c for c in df.columns if any(x in c for x in ['price', 'unit_price', 'cost', 'value'])), None)
        prod_col = next((c for c in df.columns if 'product' in c), None)

        if not sku_col or not qty_col: return pd.DataFrame()

        df['clean_sku'] = df[sku_col].apply(self.normalize_sku)
        df['sales_qty'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
        
        # Revenue Estimation (Default to $30 if no price found)
        if price_col:
            df['unit_price'] = pd.to_numeric(df[price_col], errors='coerce').fillna(30.0)
        else:
            df['unit_price'] = 30.0

        agg = df.groupby('clean_sku').agg({
            'sales_qty': 'sum',
            'unit_price': 'mean',
            prod_col: 'first' if prod_col else lambda x: 'Unknown'
        }).reset_index()

        agg['est_revenue'] = agg['sales_qty'] * agg['unit_price']
        agg.rename(columns={prod_col: 'product_name'}, inplace=True)
        return agg

    def process_returns_file(self, file):
        """Ingests Return Reports to identify defect rates and reasons."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        sku_col = next((c for c in df.columns if c in ['sku', 'default_code', 'product', 'variant']), None)
        reason_col = next((c for c in df.columns if any(x in c for x in ['reason', 'comment', 'note'])), None)
        qty_col = next((c for c in df.columns if any(x in c for x in ['qty', 'quantity', 'count'])), None)

        if not sku_col: return pd.DataFrame()

        df['clean_sku'] = df[sku_col].apply(self.normalize_sku)
        df['return_qty'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1) if qty_col else 1
        df['return_reason'] = df[reason_col].fillna('Unspecified') if reason_col else 'Unspecified'
        
        return df

    def process_helpdesk_file(self, file):
        """Ingests Support Tickets to correlate complaints with returns."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        sku_col = next((c for c in df.columns if c in ['sku', 'product']), None)
        text_cols = [c for c in df.columns if any(x in c for x in ['subject', 'desc', 'content', 'message', 'name'])]

        clean = []
        for _, row in df.iterrows():
            txt = " | ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
            sku = "UNKNOWN"
            
            # Try explicit column first
            if sku_col and pd.notna(row[sku_col]):
                sku = self.normalize_sku(row[sku_col])
            
            # Fallback: Regex extraction from subject line
            if sku == "UNKNOWN":
                match = re.search(r'\b([A-Z]{3,4}-?[0-9]{3,5})\b', txt)
                if match: sku = self.normalize_sku(match.group(1))

            clean.append({'clean_sku': sku, 'ticket_text': txt})
            
        return pd.DataFrame(clean)

    def merge_datasets(self, sales_df, returns_df, helpdesk_df=None):
        """
        Core Logic: 
        1. XLOOKUPs Sales, Returns, and Tickets into a single master view.
        2. Calculates Return Rate % and Financial Loss $.
        """
        
        # 1. Build SKU Master List
        skus = pd.Series(dtype='object')
        if not sales_df.empty: skus = pd.concat([skus, sales_df['clean_sku']])
        if not returns_df.empty: skus = pd.concat([skus, returns_df['clean_sku']])
        all_skus = skus.dropna().unique()
        
        master = pd.DataFrame({'clean_sku': all_skus})

        # 2. Merge Sales (Financial Baseline)
        if not sales_df.empty:
            master = pd.merge(master, sales_df[['clean_sku', 'sales_qty', 'est_revenue', 'product_name']], on='clean_sku', how='left')
        
        # 3. Merge Returns (Risk Overlay)
        if not returns_df.empty:
            ret_agg = returns_df.groupby('clean_sku')['return_qty'].sum().reset_index()
            master = pd.merge(master, ret_agg, on='clean_sku', how='left')
            
            # Aggregate Top 3 Reasons
            top_reasons = returns_df.groupby('clean_sku')['return_reason'].apply(
                lambda x: list(x.value_counts().index[:3])
            ).reset_index()
            master = pd.merge(master, top_reasons, on='clean_sku', how='left')

        # 4. Safety Defaults (Prevent crashes on empty columns)
        cols_defaults = {
            'sales_qty': 0, 'return_qty': 0, 'est_revenue': 0.0, 
            'product_name': 'Unknown Product', 'return_reason': []
        }
        for col, val in cols_defaults.items():
            if col not in master.columns: master[col] = val
            master[col] = master[col].fillna(val)

        # 5. Metrics Calculation
        # Return Rate
        master['return_rate'] = (master['return_qty'] / master['sales_qty'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        master.loc[(master['sales_qty'] == 0) & (master['return_qty'] > 0), 'return_rate'] = 100.0
        
        # Financial Loss Calculation
        # (Avg Price * Returns)
        avg_price = (master['est_revenue'] / master['sales_qty']).replace([np.inf, -np.inf], 0).fillna(0)
        master['lost_revenue'] = master['return_qty'] * avg_price

        # 6. Merge Helpdesk (Volume Check)
        if helpdesk_df is not None and not helpdesk_df.empty:
            counts = helpdesk_df.groupby('clean_sku').size().reset_index(name='ticket_count')
            master = pd.merge(master, counts, on='clean_sku', how='left')
            master['ticket_count'] = master['ticket_count'].fillna(0)
        else:
            master['ticket_count'] = 0

        # 7. Categorization
        master['parent_sku'] = master['clean_sku'].apply(self.extract_parent_sku)
        master['category'] = master['clean_sku'].apply(self.get_category)

        # Sort by Financial Impact
        master = master.sort_values('lost_revenue', ascending=False)
        
        return master
