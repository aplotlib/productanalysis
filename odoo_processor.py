import pandas as pd
import numpy as np
import re
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdooProcessor:
    """
    Robust data processor for Odoo exports. 
    Handles fuzzy matching, financial estimation, and multi-source merging.
    """

    def __init__(self):
        # Dictionary to map Odoo ID (e.g., '52') to SKU (e.g., 'MOB1001')
        self.product_id_map = {} 

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
            # Handle bytes input (from Streamlit)
            if isinstance(file, bytes):
                file = io.BytesIO(file)
                
            # Try reading as CSV first (common for Odoo exports despite .xlsx name)
            try:
                return pd.read_csv(file)
            except:
                file.seek(0)
            
            # Fallback to Excel sniffing
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

    def load_product_master(self, file):
        """
        Optional: Load a Product Export that has 'ID' and 'Internal Reference'.
        This allows perfect linking of Helpdesk Tickets (ID-based) to Inventory (SKU-based).
        """
        df = self.smart_read_excel(file)
        if df.empty: return
        
        # Find ID and SKU columns
        id_col = next((c for c in df.columns if c == 'id' or c == 'external_id'), None)
        sku_col = next((c for c in df.columns if 'reference' in c or 'code' in c), None)
        
        if id_col and sku_col:
            # Create a map: '52' -> 'MOB1001'
            self.product_id_map = pd.Series(
                df[sku_col].values, index=df[id_col].astype(str).values
            ).to_dict()
            logger.info(f"Loaded {len(self.product_id_map)} product ID mappings.")

    def process_helpdesk_file(self, file):
        """Ingests Support Tickets to correlate complaints with returns."""
        df = self.smart_read_excel(file)
        if df.empty: return pd.DataFrame()

        # Identify Columns
        # Helpdesk exports often have 'Products' column with format "helpdesk.ticket.products,52"
        product_rel_col = next((c for c in df.columns if 'products' in c), None)
        text_cols = [c for c in df.columns if any(x in c for x in ['subject', 'desc', 'content', 'message', 'name'])]

        clean_rows = []
        for _, row in df.iterrows():
            txt = " | ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
            sku = "UNKNOWN"
            odoo_id = None

            # 1. Extract Odoo ID from Relation String (e.g., "helpdesk.ticket.products,52")
            if product_rel_col and pd.notna(row[product_rel_col]):
                val = str(row[product_rel_col])
                if "," in val:
                    parts = val.split(',')
                    if len(parts) > 1 and parts[1].isdigit():
                        odoo_id = parts[1]

            # 2. Try Mapping ID -> SKU (Best Method)
            if odoo_id and odoo_id in self.product_id_map:
                sku = self.normalize_sku(self.product_id_map[odoo_id])
            
            # 3. Fallback: Regex extraction from Subject/Text (e.g. finds "MOB1027")
            if sku == "UNKNOWN":
                # Look for patterns like MOB1001 or SUP2047
                match = re.search(r'\b([A-Z]{3,4}-?[0-9]{3,5}[A-Z0-9]*)\b', txt)
                if match: 
                    sku = self.normalize_sku(match.group(1))
            
            clean_rows.append({
                'ticket_id': row.get('ticket_ids_sequence', 'N/A'),
                'date': row.get('created_on', None),
                'subject': txt,
                'odoo_product_id': odoo_id,
                'clean_sku': sku,
                'priority': row.get('priority', 'Low')
            })
            
        return pd.DataFrame(clean_rows)

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
            # Filter out UNKNOWN skus to avoid skewing data
            valid_tickets = helpdesk_df[helpdesk_df['clean_sku'] != 'UNKNOWN']
            counts = valid_tickets.groupby('clean_sku').size().reset_index(name='ticket_count')
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
