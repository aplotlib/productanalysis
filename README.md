# Medical Device Customer Feedback Analyzer

**Version 4.0 - Production Stable**

A powerful yet simple tool for analyzing Amazon review data with AI-powered insights specifically designed for medical device companies. Upload review data and receive categorized, qualitative, and quantitative analysis with actionable CAPA recommendations.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Set your OpenAI API key (required for AI analysis)
export OPENAI_API_KEY="your-api-key-here"

# Alternative: Create .streamlit/secrets.toml
mkdir .streamlit
echo 'openai_api_key = "your-api-key-here"' > .streamlit/secrets.toml
```

### 3. Run the Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
medical-device-analyzer/
├── app.py                    # Main application (Fixed v4.0)
├── upload_handler.py         # File upload processing (Helium 10 optimized)
├── text_analysis_engine.py   # Core analysis engine (AI-enhanced)
├── enhanced_ai_analysis.py   # OpenAI GPT-4o integration
├── dashboard.py             # Simple dashboard components
├── requirements.txt         # Dependencies (stable versions)
└── README.md               # This file
```

## 🎯 Key Features

### ✅ **Stable & Accurate**
- Robust error handling throughout
- Graceful degradation when dependencies unavailable
- Production-ready code with comprehensive logging

### 📊 **Helium 10 Optimized**
- Automatic detection of Helium 10 review exports
- ASIN extraction from filenames
- Accurate date parsing and review processing

### 🤖 **AI-Powered Analysis**
- OpenAI GPT-4o integration for deep insights
- Medical device quality categorization
- Safety risk assessment
- Actionable CAPA recommendations

### 🏥 **Medical Device Focus**
- ISO 13485 compliance awareness
- Quality management categories
- Risk assessment framework
- CAPA generation for quality improvement

## 📤 Supported File Formats

### **Primary: Helium 10 Review Exports**
- **Format:** CSV files with Helium 10 structure
- **Filename Pattern:** `{ASIN}  {Product_Name}  {Date}.csv`
- **Example:** `B00TZ73MUY  Vive Rollator Walker  20250122.csv`
- **Columns:** Date, Author, Verified, Helpful, Title, Body, Rating, Images, Videos, etc.

### **Secondary: Generic Review Data**
- CSV or Excel files with review content
- Automatic column mapping for text, ratings, dates
- Flexible format support

## 🔧 Configuration

### **Required Environment Variables**
```bash
OPENAI_API_KEY=your-openai-api-key  # For AI analysis features
```

### **Optional Configuration**
- **File Size Limit:** 50MB (configurable in upload_handler.py)
- **AI Model:** GPT-4o (configurable in enhanced_ai_analysis.py)
- **Analysis Timeout:** 30 seconds (configurable)

## 📊 Analysis Output

### **Quality Categories**
- Safety & Risk Management
- Effectiveness & Performance  
- Comfort & Usability
- Durability & Quality
- Sizing & Fit Issues
- Assembly & Instructions

### **AI Insights**
- Overall sentiment analysis
- Safety concern identification
- Top quality issues
- Immediate action recommendations
- Listing improvement suggestions

### **CAPA Recommendations**
- Priority-based action items
- Corrective and preventive actions
- Timeline and responsibility assignments
- Success metrics and verification

### **Risk Assessment**
- Overall risk level (Critical/High/Medium/Low)
- Specific risk factors
- Safety issue identification
- Quality score calculation

## 🚀 Usage Workflow

### 1. **Upload Data**
- Upload Helium 10 review export (preferred)
- Or use generic CSV/Excel with review data
- Or try the built-in example data

### 2. **Process & Analyze**
- Automatic format detection
- Text analysis and categorization
- AI-powered insight generation
- Risk assessment calculation

### 3. **Review Results**
- Quality category breakdown
- CAPA recommendations
- AI insights and suggestions
- Risk assessment summary

### 4. **Export & Act**
- Download summary reports
- Export CAPA action lists
- Implement quality improvements

## 🔍 Example Helium 10 File

```csv
Date,Author,Verified,Helpful,Title,Body,Rating,Images,Videos,URL,Variation,Style
"January 15, 2024","Sarah M.","Verified Purchase","2","Great product but assembly difficult","This rollator is very stable and helps me walk confidently. However, the assembly instructions were confusing and some screws were missing.",4,"","","","",""
"January 10, 2024","John D.","Verified Purchase","0","Broke after one week","The wheel came off after just one week of light use. Very disappointed with the quality.",1,"","","","",""
```

## 🛠️ Troubleshooting

### **Common Issues**

**1. "AI analysis not available"**
- Ensure OPENAI_API_KEY is set correctly
- Check API key has sufficient credits
- Verify internet connection

**2. "Upload failed"** 
- Check file format is CSV or Excel
- Ensure file size under 50MB
- Verify file has review content

**3. "No analysis results"**
- Ensure uploaded file has text content in reviews
- Check date formats are recognizable
- Verify file has actual review data

### **System Status Check**
The app displays module availability in the sidebar:
- ✅ Upload Handler - File processing
- ✅ Text Analysis Engine - Core analysis
- ✅ AI Analyzer - AI enhancement

## 🔒 Data Privacy & Security

- **No Data Storage:** Files processed in memory only
- **API Security:** OpenAI API calls use secure HTTPS
- **Local Processing:** Core analysis runs locally
- **No Data Persistence:** Session data cleared on restart

## 📈 Performance Notes

- **Recommended:** 10-500 reviews per file for optimal processing
- **AI Analysis:** Limited to 15 reviews per batch for API efficiency
- **Processing Time:** 10-30 seconds typical for 100 reviews
- **Memory Usage:** ~100MB typical for large files

## 🤝 Support & Development

### **System Requirements**
- Python 3.8+
- 4GB RAM minimum
- Internet connection (for AI features)
- Modern web browser

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug mode
streamlit run app.py --logger.level=debug

# Environment variables for development
export STREAMLIT_ENV=development
export OPENAI_API_KEY=your-key
```

### **Deployment Options**

**Streamlit Cloud (Recommended):**
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Add `openai_api_key` to Streamlit secrets
4. Deploy automatically

**Local Server:**
```bash
streamlit run app.py --server.port 8501
```

**Docker (Optional):**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📋 Version History

### **Version 4.0 (Current) - Production Stable**
- ✅ Complete system stability overhaul
- ✅ Enhanced Helium 10 processing
- ✅ Robust error handling throughout
- ✅ Simplified dependencies
- ✅ AI integration improvements
- ✅ Medical device focus enhancement

### **Key Improvements in v4.0:**
- **Stability:** Comprehensive error handling and graceful degradation
- **Accuracy:** Enhanced text processing and categorization
- **Usability:** Simplified interface with clear workflow
- **Performance:** Optimized for speed and reliability
- **Medical Focus:** ISO 13485 awareness and medical device categories

## 📞 Contact & Feedback

For technical support or feature requests, the application includes comprehensive logging to help diagnose issues. Check the browser console and application logs for detailed error information.
contact: alexander.popoff@vivehealth.com
---

**Built for ecomm listing managers and medical device quality managers who need actionable insights from customer feedback.**
