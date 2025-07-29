# Water & Diesel Usage Analytics Application

A comprehensive Streamlit application for analyzing water and diesel usage data from PDF reports, featuring detailed analytics, trend analysis, and professional PDF report generation.

## 🚀 Enhanced Features

### 📊 Comprehensive Analytics Summary
- **Detailed Metric Performance**: In-depth analysis of each metric with percentage changes, absolute differences, and impact classifications
- **Cost Impact Analysis**: Total fuel cost calculations with year-over-year comparisons
- **Usage Efficiency Patterns**: Volatility analysis and peak usage identification
- **Key Insights & Trends**: Automated identification of critical changes requiring attention
- **Strategic Recommendations**: AI-generated actionable recommendations based on data patterns

### 📈 Advanced Statistical Insights
- **Volatility Analysis**: Coefficient of variation calculations for all metrics
- **Correlation Analysis**: Relationship strength between different usage metrics
- **Monthly Patterns**: Seasonal trend identification across multiple months
- **Operational Efficiency**: Year-over-year efficiency ratio calculations

### 📄 Professional PDF Reports
- **Enhanced Formatting**: Structured layout with proper headers and sections
- **Executive Summary**: Comprehensive multi-page summary with detailed insights
- **Visual Charts**: High-quality trend graphs with data labels
- **Detailed Tables**: Professional formatting with centered alignment

## 📋 Features Overview

- **PDF Data Extraction**: Automatically extracts structured data from WATER.DIESEL PDF files
- **Interactive Visualizations**: Trend graphs with data labels and trendlines
- **Month Selection**: Filter analysis by specific months
- **Data Export**: Download processed data as CSV
- **Professional Reports**: Generate comprehensive PDF reports with insights

## 🛠️ Installation

1. **Clone or download the application files**

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas matplotlib streamlit fpdf2 PyMuPDF
   ```

## 🚀 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload your WATER.DIESEL PDF file** using the file uploader

3. **Select months** to analyze (default: all available months)

4. **View the comprehensive analytics:**
   - **Trend Graphs**: Interactive charts with data labels
   - **Raw Data Table**: Complete dataset view
   - **Comprehensive Analytics Summary**: Detailed insights and recommendations
   - **Advanced Statistical Insights**: Volatility and correlation analysis
   - **Summary Metrics Table**: Year-over-year comparison

5. **Download results:**
   - **CSV Export**: Raw data for further analysis
   - **PDF Report**: Professional report with all insights and charts

## 📊 Sample Analysis Output

The enhanced summary provides:

```
COMPREHENSIVE WATER & DIESEL USAGE ANALYSIS
Analysis Period: Jan, Feb, Mar across 2024, 2025
Total Records Analyzed: 6 monthly data points

DETAILED METRIC PERFORMANCE:
• Water Used (L): Increased by 15.3% (moderate change)
  - 2024: 3600.00 total, 1200.00 monthly average
  - 2025: 4150.00 total, 1383.33 monthly average
  - Absolute change: 550.00 units

COST IMPACT ANALYSIS:
• Total fuel costs increased by 12.4%
• Net cost difference: R531.25

KEY INSIGHTS & TRENDS:
• Notable: Water Used (L) changed by 15.3% - monitor closely
• Both diesel usage and costs increased significantly

STRATEGIC RECOMMENDATIONS:
1. Implement cost control measures for fuel expenses
2. Continue monthly monitoring to identify seasonal patterns
3. Consider implementing usage forecasting for better budget planning
```

## 🔧 Technical Details

### Data Processing
- **Regex Pattern Matching**: Extracts structured data from PDF text
- **Data Validation**: Ensures data integrity and consistency
- **Statistical Analysis**: Advanced calculations for insights

### Visualization
- **Matplotlib Integration**: High-quality charts with customization
- **Trend Analysis**: Automatic trendline generation
- **Data Labels**: Clear value display on all charts

### Report Generation
- **FPDF2**: Professional PDF creation with advanced formatting
- **Multi-page Layout**: Structured report organization
- **Image Integration**: High-resolution chart embedding

## 📁 File Structure

```
├── app.py                 # Main Streamlit application
├── README.md             # This documentation file
└── venv/                 # Virtual environment (after setup)
```

## 🔍 Key Functions

- `extract_table_from_pdf()`: PDF text extraction
- `parse_usage_data()`: Data parsing and structuring
- `calculate_summary_table()`: Statistical calculations
- `generate_summary_paragraph()`: Comprehensive analysis generation
- `calculate_additional_insights()`: Advanced statistical insights
- `generate_pdf_report()`: Professional PDF creation

## 🎯 Use Cases

- **Operational Management**: Monitor resource usage patterns
- **Cost Control**: Track and analyze fuel expenses
- **Trend Analysis**: Identify seasonal variations and patterns
- **Budget Planning**: Forecast future usage and costs
- **Performance Monitoring**: Track efficiency improvements

## 🔧 Customization

The application can be easily customized for different data formats or additional metrics by modifying the regex patterns in `parse_usage_data()` and updating the analysis functions accordingly.

## 📞 Support

For technical support or feature requests, please refer to the code documentation and comments within the application files.