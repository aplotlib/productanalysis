import pandas as pd
import io

def create_import_template():
    """Create a sample import template for users to download."""
    # Create sample data
    data = {
        "SKU": ["MOB1116BLU", "BAT2234RED", "KOMF352WHT"],
        "ASIN*": ["B0DT7NW5VY", "B0DT8XYZ123", "B08CK7MN45"],
        "Product Name": ["Tri-Rollator With Seat", "Vive Shower Chair", "Comfort Cushion"],
        "Category": ["Mobility Aids", "Bathroom Safety", "Comfort Products"],
        "Last 30 Days Sales*": [491, 325, 278],
        "Last 30 Days Returns*": [10, 8, 5],
        "Last 365 Days Sales": [5840, 3900, 2950],
        "Last 365 Days Returns": [67, 45, 29],
        "Star Rating": [3.9, 4.2, 4.5],
        "Total Reviews": [20, 35, 42],
        "Average Price": [89.99, 59.99, 34.99]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create Excel binary
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Import Template', index=False)
        
        # Format the Excel sheet
        workbook = writer.book
        worksheet = writer.sheets['Import Template']
        
        # Format for headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Make mandatory columns stand out
        required_format = workbook.add_format({
            'fg_color': '#FFEB9C',
            'border': 1
        })
        
        # Format required columns
        required_cols = ['ASIN*', 'Last 30 Days Sales*', 'Last 30 Days Returns*']
        for col in required_cols:
            col_idx = df.columns.get_loc(col)
            worksheet.set_column(col_idx, col_idx, None, required_format)
        
        # Add a legend for mandatory fields
        worksheet.write(len(df) + 2, 0, "* Mandatory fields", header_format)
        
        # Add instructions
        instructions = [
            "Instructions for using this template:",
            "1. Fill in all mandatory fields (marked with *)",
            "2. ASIN is the Amazon Standard Identification Number",
            "3. SKU is your internal Stock Keeping Unit code (optional)",
            "4. Include as many optional fields as possible for better analysis",
            "5. Save as CSV or Excel file for import into the Product Review Analysis Tool"
        ]
        
        for i, instruction in enumerate(instructions):
            worksheet.write(len(df) + 4 + i, 0, instruction)
        
        # Set column widths
        for i, col in enumerate(df.columns):
            column_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_width)
        
    return output.getvalue()

# The template can be used in the main application for download
