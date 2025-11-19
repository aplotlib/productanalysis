import pandas as pd
import re
import io

class ReturnReportProcessor:
    """
    Handles complex hierarchical Odoo Pivot Exports.
    """
    def process(self, file_content):
        # Read without header to handle the nested structure
        df = pd.read_csv(io.BytesIO(file_content), header=None)
        
        cleaned_data = []
        current_date = None
        
        # Regex to detect "Month Year" rows (e.g., "August 2020")
        date_pattern = re.compile(r'^\s*[A-Z][a-z]+ \d{4}$')
        
        for index, row in df.iterrows():
            col0 = str(row[0])
            
            # 1. Detect Date Group
            if date_pattern.match(col0):
                current_date = col0.strip()
                continue
                
            # 2. Detect Product Row: "     [MOB1001] Folding Cane"
            # We look for the brackets [SKU]
            if "[" in col0 and "]" in col0:
                try:
                    # Extract SKU between brackets
                    start = col0.find("[") + 1
                    end = col0.find("]")
                    sku = col0[start:end]
                    product_name = col0[end+1:].strip()
                    
                    # Assuming columns: 0=Name, 1=B2B, 2=FBM, 3=Shopify (Adjust if needed based on file)
                    total = 0
                    # Sum specific columns if they exist and are numeric
                    for i in [1, 2, 3]: 
                        if i < len(row) and pd.notna(row[i]):
                            try:
                                total += float(row[i])
                            except:
                                pass
                    
                    if total > 0:
                        cleaned_data.append({
                            'Date': current_date,
                            'SKU': sku,
                            'Product Name': product_name,
                            'Total Returns': total
                        })
                except:
                    continue

        return pd.DataFrame(cleaned_data)
