import pandas as pd
import re
import io

class ReturnReportProcessor:
    def __init__(self):
        # Regex to identify the "Month Year" rows (e.g., "August 2020")
        self.date_pattern = re.compile(r'^\s*[A-Z][a-z]+ \d{4}$')
        # Regex to identify Product rows (e.g., "[MOB1001] Folding Cane")
        self.product_pattern = re.compile(r'^\s*\[(.*?)\]\s*(.*)')

    def process(self, file_content):
        """
        Parses the hierarchical Odoo Pivot Report for Returns.
        """
        # Read without header because headers are multi-row and messy
        if isinstance(file_content, bytes):
            df = pd.read_csv(io.BytesIO(file_content), header=None)
        else:
            df = pd.read_csv(file_content, header=None)

        cleaned_data = []
        current_date = None
        
        # Iterate row by row to handle the hierarchy
        for index, row in df.iterrows():
            col0_text = str(row[0])
            
            # 1. Detect Date Header (Top level hierarchy)
            if self.date_pattern.match(col0_text):
                current_date = col0_text.strip()
                continue
            
            # 2. Detect Product Row (Nested hierarchy)
            match = self.product_pattern.search(col0_text)
            if match and current_date:
                sku = match.group(1)
                product_name = match.group(2)
                
                # Map columns based on your file structure:
                # Col 1: B2B, Col 2: FBM, Col 3: Shopify (Indices 1, 2, 3)
                # Using pd.to_numeric to handle empty strings/NaNs safely
                def get_val(idx):
                    try:
                        val = row[idx]
                        return float(val) if pd.notna(val) and str(val).strip() != '' else 0.0
                    except:
                        return 0.0

                b2b_qty = get_val(1)
                fbm_qty = get_val(2)
                shopify_qty = get_val(3)
                
                # Only add if there is return data
                if b2b_qty + fbm_qty + shopify_qty > 0:
                    cleaned_data.append({
                        'Date': current_date,
                        'SKU': sku,
                        'Product Name': product_name,
                        'B2B Returns': b2b_qty,
                        'FBM Returns': fbm_qty,
                        'Shopify Returns': shopify_qty,
                        'Total Returns': b2b_qty + fbm_qty + shopify_qty
                    })

        return pd.DataFrame(cleaned_data)
