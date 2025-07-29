import streamlit as st
import pandas as pd
import fitz
import re
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Water & Diesel Usage Comparison", layout="wide")
st.title("📊 Water & Diesel Usage Analytics Comparison")

uploaded_file = st.file_uploader("Upload WATER.DIESEL PDF", type="pdf")

def extract_table_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_usage_data(text):
    text = re.sub(r"R(\d{1,3})\s(\d{3}\.\d{2})", r"R\1\2", text)
    pattern = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(2[4-5])\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+R(\d+\.\d{2})\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+R(\d+\.\d{2})"
    )
    data = []
    for match in pattern.finditer(text):
        month, year = match.group(1), f"20{match.group(2)}"
        data.append({
            "Month": month,
            "Year": int(year),
            "Water Used (L)": int(match.group(5)),
            "Diesel Used (L)": int(match.group(8)),
            "Diesel Cost (R)": float(match.group(9)),
            "Generator Used (L)": int(match.group(12)),
            "Generator Cost (R)": float(match.group(13)),
        })
    return pd.DataFrame(data)

def calculate_summary_table(df):
    metrics = ["Water Used (L)", "Diesel Used (L)", "Diesel Cost (R)",
               "Generator Used (L)", "Generator Cost (R)"]
    summary = []
    for metric in metrics:
        y24 = df[df["Year"] == 2024][metric]
        y25 = df[df["Year"] == 2025][metric]
        total_2024, avg_2024 = y24.sum(), y24.mean()
        total_2025, avg_2025 = y25.sum(), y25.mean()
        pct_change = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 else 0
        summary.append({
            "Metric": metric,
            "2024 Total": total_2024,
            "2024 Avg": avg_2024,
            "2025 Total": total_2025,
            "2025 Avg": avg_2025,
            "% Change": pct_change
        })
    return pd.DataFrame(summary)

def generate_summary_paragraph(summary_df, df):
    """Generate a comprehensive, detailed summary with insights and recommendations."""
    lines = []
    
    # Header with time period
    months_analyzed = sorted(df["Month"].unique())
    years_analyzed = sorted(df["Year"].unique())
    time_period = f"{', '.join(months_analyzed)} across {', '.join(map(str, years_analyzed))}"
    
    lines.append(f"COMPREHENSIVE WATER & DIESEL USAGE ANALYSIS")
    lines.append(f"Analysis Period: {time_period}")
    lines.append(f"Total Records Analyzed: {len(df)} monthly data points")
    lines.append("")
    
    # Detailed metric analysis
    lines.append("DETAILED METRIC PERFORMANCE:")
    
    for _, row in summary_df.iterrows():
        metric = row['Metric']
        change = row['% Change']
        total_2024 = row['2024 Total']
        total_2025 = row['2025 Total']
        avg_2024 = row['2024 Avg']
        avg_2025 = row['2025 Avg']
        
        direction = "increased" if change > 0 else "decreased"
        impact = "significant" if abs(change) > 20 else "moderate" if abs(change) > 10 else "minimal"
        
        lines.append(f"• {metric}:")
        lines.append(f"  - {direction.capitalize()} by {abs(change):.1f}% ({impact} change)")
        lines.append(f"  - 2024: {total_2024:.2f} total, {avg_2024:.2f} monthly average")
        lines.append(f"  - 2025: {total_2025:.2f} total, {avg_2025:.2f} monthly average")
        lines.append(f"  - Absolute change: {total_2025 - total_2024:.2f} units")
        lines.append("")
    
    # Cost analysis
    diesel_cost_change = summary_df[summary_df['Metric'] == 'Diesel Cost (R)']['% Change'].iloc[0]
    generator_cost_change = summary_df[summary_df['Metric'] == 'Generator Cost (R)']['% Change'].iloc[0]
    total_cost_2024 = summary_df[summary_df['Metric'] == 'Diesel Cost (R)']['2024 Total'].iloc[0] + summary_df[summary_df['Metric'] == 'Generator Cost (R)']['2024 Total'].iloc[0]
    total_cost_2025 = summary_df[summary_df['Metric'] == 'Diesel Cost (R)']['2025 Total'].iloc[0] + summary_df[summary_df['Metric'] == 'Generator Cost (R)']['2025 Total'].iloc[0]
    total_cost_change = ((total_cost_2025 - total_cost_2024) / total_cost_2024 * 100) if total_cost_2024 > 0 else 0
    
    lines.append("COST IMPACT ANALYSIS:")
    lines.append(f"• Total fuel costs {'increased' if total_cost_change > 0 else 'decreased'} by {abs(total_cost_change):.1f}%")
    lines.append(f"• 2024 total costs: R{total_cost_2024:.2f}")
    lines.append(f"• 2025 total costs: R{total_cost_2025:.2f}")
    lines.append(f"• Net cost difference: R{total_cost_2025 - total_cost_2024:.2f}")
    lines.append("")
    
    # Usage efficiency analysis
    water_change = summary_df[summary_df['Metric'] == 'Water Used (L)']['% Change'].iloc[0]
    diesel_usage_change = summary_df[summary_df['Metric'] == 'Diesel Used (L)']['% Change'].iloc[0]
    
    lines.append("EFFICIENCY & USAGE PATTERNS:")
    if abs(water_change) > abs(diesel_usage_change):
        lines.append(f"• Water usage shows higher volatility ({abs(water_change):.1f}%) compared to diesel usage ({abs(diesel_usage_change):.1f}%)")
    else:
        lines.append(f"• Diesel usage shows higher volatility ({abs(diesel_usage_change):.1f}%) compared to water usage ({abs(water_change):.1f}%)")
    
    # Peak usage analysis
    if not df.empty:
        peak_water_month = df.loc[df['Water Used (L)'].idxmax()]
        peak_diesel_month = df.loc[df['Diesel Used (L)'].idxmax()]
        lines.append(f"• Peak water usage: {peak_water_month['Water Used (L)']} L in {peak_water_month['Month']} {peak_water_month['Year']}")
        lines.append(f"• Peak diesel usage: {peak_diesel_month['Diesel Used (L)']} L in {peak_diesel_month['Month']} {peak_diesel_month['Year']}")
    lines.append("")
    
    # Trend insights
    lines.append("KEY INSIGHTS & TRENDS:")
    insights = []
    
    for _, row in summary_df.iterrows():
        metric = row['Metric']
        change = row['% Change']
        
        if abs(change) > 25:
            insights.append(f"Critical: {metric} shows major change of {change:.1f}% - requires immediate attention")
        elif abs(change) > 15:
            insights.append(f"Notable: {metric} changed by {change:.1f}% - monitor closely")
    
    if diesel_cost_change > 10 and diesel_usage_change > 10:
        insights.append("Both diesel usage and costs increased significantly - potential inefficiency or increased demand")
    elif diesel_cost_change > 10 and diesel_usage_change < 5:
        insights.append("Diesel costs rose faster than usage - likely due to price increases")
    
    if water_change > 20:
        insights.append("Substantial water usage change may indicate operational changes or seasonal factors")
    
    for insight in insights:
        lines.append(f"• {insight}")
    
    if not insights:
        lines.append("• Usage patterns appear stable with no critical changes requiring immediate action")
    lines.append("")
    
    # Recommendations
    lines.append("STRATEGIC RECOMMENDATIONS:")
    recommendations = []
    
    if total_cost_change > 10:
        recommendations.append("Implement cost control measures for fuel expenses")
    
    if diesel_usage_change > 15:
        recommendations.append("Review diesel consumption patterns and explore efficiency improvements")
    
    if generator_cost_change > 20:
        recommendations.append("Evaluate generator usage optimization and maintenance schedules")
    
    if abs(water_change) > 20:
        recommendations.append("Investigate water usage variations and implement conservation measures if needed")
    
    # General recommendations
    recommendations.extend([
        "Continue monthly monitoring to identify seasonal patterns",
        "Consider implementing usage forecasting for better budget planning",
        "Establish baseline metrics for performance benchmarking"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
    
    lines.append("")
    lines.append("This analysis provides actionable insights for operational efficiency and cost management.")
    
    return "\n".join(lines)

def calculate_additional_insights(df):
    """Calculate additional statistical insights and seasonal patterns."""
    insights = {}
    
    if len(df) > 1:
        # Volatility analysis (coefficient of variation)
        for metric in ["Water Used (L)", "Diesel Used (L)", "Diesel Cost (R)", "Generator Used (L)", "Generator Cost (R)"]:
            cv = (df[metric].std() / df[metric].mean()) * 100 if df[metric].mean() > 0 else 0
            insights[f"{metric}_volatility"] = cv
        
        # Correlation analysis
        correlations = {}
        correlations['water_diesel'] = df["Water Used (L)"].corr(df["Diesel Used (L)"])
        correlations['diesel_cost'] = df["Diesel Used (L)"].corr(df["Diesel Cost (R)"])
        correlations['generator_usage_cost'] = df["Generator Used (L)"].corr(df["Generator Cost (R)"])
        insights['correlations'] = correlations
        
        # Monthly patterns (if multiple months available)
        if len(df["Month"].unique()) > 1:
            monthly_avg = df.groupby("Month")[["Water Used (L)", "Diesel Used (L)"]].mean()
            insights['monthly_patterns'] = monthly_avg.to_dict()
        
        # Year-over-year efficiency
        if len(df["Year"].unique()) > 1:
            efficiency_2024 = df[df["Year"] == 2024]["Water Used (L)"].sum() / max(df[df["Year"] == 2024]["Diesel Used (L)"].sum(), 1)
            efficiency_2025 = df[df["Year"] == 2025]["Water Used (L)"].sum() / max(df[df["Year"] == 2025]["Diesel Used (L)"].sum(), 1)
            insights['efficiency_ratio_change'] = ((efficiency_2025 - efficiency_2024) / efficiency_2024 * 100) if efficiency_2024 > 0 else 0
    
    return insights

def plot_metric_with_trend(df, metric):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = df[metric].plot(kind="bar", ax=ax)

    # ✅ Add value labels to each bar
    for container in bars.containers:
        bars.bar_label(container, fmt="%.0f", padding=3, fontsize=8)

    for year in [2024, 2025]:
        if year in df[metric].columns:
            values = df[metric][year].values.astype(float)
            ax.plot(range(len(values)), values, marker='o', linestyle='--', label=f"{year} Trend")

    ax.set_title(f"{metric} Comparison with Trendlines")
    ax.set_ylabel(metric)
    ax.set_xlabel("Month")
    ax.legend(title="Year")
    plt.tight_layout()
    return fig

def generate_pdf_report(figures, summary_df, summary_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Water & Diesel Analytics Summary Report", ln=True, align='C')
    pdf.ln(10)
    
    # Detailed summary section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 10)
    
    # Split summary text into lines and handle each line
    summary_lines = summary_text.split('\n')
    for line in summary_lines:
        if line.strip():  # Only process non-empty lines
            # Handle encoding issues
            clean_line = line.encode("latin-1", "ignore").decode("latin-1")
            
            # Check if line is a header (all caps or starts with specific patterns)
            if (line.isupper() and len(line) > 10) or line.startswith("COMPREHENSIVE"):
                pdf.ln(3)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, clean_line, ln=True)
                pdf.set_font("Arial", "", 10)
            elif line.startswith("Analysis Period:") or line.startswith("Total Records"):
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 5, clean_line, ln=True)
                pdf.set_font("Arial", "", 10)
            elif line.startswith("•"):
                # Bullet points
                pdf.cell(0, 5, clean_line, ln=True)
            elif line.startswith("  -"):
                # Sub-bullet points with indentation
                pdf.cell(10, 5, "", ln=False)  # Indent
                pdf.cell(0, 5, clean_line.strip(), ln=True)
            elif line[0].isdigit() and line[1] == ".":
                # Numbered recommendations
                pdf.cell(0, 5, clean_line, ln=True)
            else:
                # Regular text
                pdf.cell(0, 5, clean_line, ln=True)
        else:
            # Empty line - add small space
            pdf.ln(2)
    
    pdf.ln(10)

    # Summary table section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detailed Metrics Table", ln=True)
    pdf.ln(5)
    
    # Table header
    pdf.set_font("Arial", "B", 9)
    headers = ["Metric", "2024 Total", "2024 Avg", "2025 Total", "2025 Avg", "% Change"]
    col_widths = [50, 28, 23, 28, 23, 23]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align='C')
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", "", 8)
    for _, row in summary_df.iterrows():
        values = [
            row["Metric"][:20] + "..." if len(row["Metric"]) > 20 else row["Metric"],  # Truncate long names
            f"{row['2024 Total']:.1f}",
            f"{row['2024 Avg']:.1f}",
            f"{row['2025 Total']:.1f}",
            f"{row['2025 Avg']:.1f}",
            f"{row['% Change']:.1f}%"
        ]
        for i, v in enumerate(values):
            pdf.cell(col_widths[i], 7, str(v), border=1, align='C')
        pdf.ln()

    # Add graphs on separate pages
    for title, fig in figures:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Chart: {title}", ln=True)
        pdf.ln(5)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format="png", dpi=150, bbox_inches='tight')
            tmpfile.flush()
            
            # Center the image
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
        os.unlink(tmpfile.name)

    return pdf.output(dest="S").encode("latin-1")

# --- MAIN ---
if uploaded_file:
    raw_text = extract_table_from_pdf(uploaded_file)

    with st.expander("🔍 View Raw Extracted Text"):
        st.text(raw_text[:3000])

    df = parse_usage_data(raw_text)

    if df.empty:
        st.error("⚠️ Could not extract structured data.")
    else:
        df["Month"] = pd.Categorical(df["Month"], categories=[
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ], ordered=True)
        df.sort_values(["Year", "Month"], inplace=True)

        # Month selector
        available_months = df["Month"].unique().tolist()
        selected_months = st.multiselect(
            "Select month(s) to compare",
            options=available_months,
            default=available_months
        )
        df = df[df["Month"].isin(selected_months)]

        if df.empty:
            st.warning("No data available for selected month(s).")
        else:
            pivot_df = df.pivot(index="Month", columns="Year", values=[
                "Water Used (L)", "Diesel Used (L)", "Diesel Cost (R)",
                "Generator Used (L)", "Generator Cost (R)"
            ])
            pivot_df = pivot_df.reindex(selected_months)

            st.subheader("📈 Trend Graphs with Values")
            figures = []
            for metric in pivot_df.columns.levels[0]:
                fig = plot_metric_with_trend(pivot_df, metric)
                st.pyplot(fig)
                figures.append((metric, fig))

            st.subheader("📋 Raw Data Table")
            st.dataframe(df)

            summary_df = calculate_summary_table(df)
            additional_insights = calculate_additional_insights(df)
            summary_text = generate_summary_paragraph(summary_df, df)
            
            st.subheader("📑 Comprehensive Analytics Summary")
            
            # Display the detailed summary with proper formatting
            formatted_summary = summary_text.replace('\n', '\n\n')  # Add extra line breaks for markdown
            st.markdown(formatted_summary)
            
            # Display additional statistical insights
            if additional_insights:
                st.subheader("📈 Advanced Statistical Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Volatility Analysis (Coefficient of Variation)**")
                    volatility_data = []
                    for key, value in additional_insights.items():
                        if key.endswith('_volatility'):
                            metric_name = key.replace('_volatility', '').replace('(L)', '').replace('(R)', '')
                            volatility_data.append({
                                'Metric': metric_name,
                                'Volatility %': f"{value:.1f}%",
                                'Stability': 'High' if value < 15 else 'Medium' if value < 30 else 'Low'
                            })
                    if volatility_data:
                        st.dataframe(pd.DataFrame(volatility_data), use_container_width=True)
                
                with col2:
                    if 'correlations' in additional_insights:
                        st.write("**Correlation Analysis**")
                        corr_data = []
                        correlations = additional_insights['correlations']
                        corr_mapping = {
                            'water_diesel': 'Water vs Diesel Usage',
                            'diesel_cost': 'Diesel Usage vs Cost',
                            'generator_usage_cost': 'Generator Usage vs Cost'
                        }
                        for key, value in correlations.items():
                            if not pd.isna(value):
                                strength = 'Strong' if abs(value) > 0.7 else 'Medium' if abs(value) > 0.4 else 'Weak'
                                corr_data.append({
                                    'Relationship': corr_mapping.get(key, key),
                                    'Correlation': f"{value:.3f}",
                                    'Strength': strength
                                })
                        if corr_data:
                            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
                
                if 'efficiency_ratio_change' in additional_insights:
                    st.write(f"**Operational Efficiency Change:** {additional_insights['efficiency_ratio_change']:.1f}%")
            
            st.subheader("📊 Summary Metrics Table")
            st.dataframe(summary_df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, file_name="water_diesel_data.csv")

            pdf = generate_pdf_report(figures, summary_df, summary_text)
            st.download_button("📄 Download PDF Report", data=pdf, file_name="water_diesel_report.pdf", mime="application/pdf")
