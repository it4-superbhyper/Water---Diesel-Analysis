import streamlit as st
import pandas as pd
import fitz
import re
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import tempfile
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Water & Diesel Usage Comparison", layout="wide")
st.title("📊 Advanced Water & Diesel Usage Analytics")

# Sidebar for analysis options
st.sidebar.header("🔧 Analysis Options")
show_advanced = st.sidebar.checkbox("Show Advanced Analytics", value=True)
show_forecasting = st.sidebar.checkbox("Show Forecasting", value=True)
show_anomalies = st.sidebar.checkbox("Show Anomaly Detection", value=True)
show_correlations = st.sidebar.checkbox("Show Correlation Analysis", value=True)

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
            "Month": match.group(1),
            "Year": int(year),
            "Water Used (L)": int(match.group(5)),
            "Diesel Used (L)": int(match.group(8)),
            "Diesel Cost (R)": float(match.group(9)),
            "Generator Used (L)": int(match.group(12)),
            "Generator Cost (R)": float(match.group(13)),
        })
    return pd.DataFrame(data)

def calculate_efficiency_metrics(df):
    """Calculate efficiency and ratio metrics"""
    metrics_df = df.copy()
    
    # Cost per liter calculations
    metrics_df['Diesel Cost per L'] = metrics_df['Diesel Cost (R)'] / metrics_df['Diesel Used (L)']
    metrics_df['Generator Cost per L'] = metrics_df['Generator Cost (R)'] / metrics_df['Generator Used (L)']
    
    # Usage ratios
    metrics_df['Water to Diesel Ratio'] = metrics_df['Water Used (L)'] / metrics_df['Diesel Used (L)']
    metrics_df['Generator to Diesel Ratio'] = metrics_df['Generator Used (L)'] / metrics_df['Diesel Used (L)']
    
    # Total costs and usage
    metrics_df['Total Fuel Cost (R)'] = metrics_df['Diesel Cost (R)'] + metrics_df['Generator Cost (R)']
    metrics_df['Total Fuel Used (L)'] = metrics_df['Diesel Used (L)'] + metrics_df['Generator Used (L)']
    
    # Efficiency score (lower is better)
    metrics_df['Cost Efficiency Score'] = metrics_df['Total Fuel Cost (R)'] / metrics_df['Water Used (L)']
    
    return metrics_df

def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis"""
    stats_results = {}
    
    numeric_cols = ['Water Used (L)', 'Diesel Used (L)', 'Diesel Cost (R)', 
                   'Generator Used (L)', 'Generator Cost (R)']
    
    # Basic statistics
    stats_results['descriptive'] = df[numeric_cols].describe()
    
    # Normality tests
    normality_results = {}
    for col in numeric_cols:
        stat, p_value = stats.shapiro(df[col].dropna())
        normality_results[col] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
    stats_results['normality'] = normality_results
    
    # Year-over-year comparison using t-test
    if len(df['Year'].unique()) >= 2:
        comparison_results = {}
        for col in numeric_cols:
            years = sorted(df['Year'].unique())
            if len(years) >= 2:
                data_year1 = df[df['Year'] == years[0]][col].dropna()
                data_year2 = df[df['Year'] == years[1]][col].dropna()
                if len(data_year1) > 0 and len(data_year2) > 0:
                    stat, p_value = stats.ttest_ind(data_year1, data_year2)
                    comparison_results[col] = {
                        'statistic': stat, 
                        'p_value': p_value, 
                        'significant_change': p_value < 0.05,
                        'year1_mean': data_year1.mean(),
                        'year2_mean': data_year2.mean()
                    }
        stats_results['year_comparison'] = comparison_results
    
    return stats_results

def calculate_correlations(df):
    """Calculate correlation matrix for usage metrics"""
    numeric_cols = ['Water Used (L)', 'Diesel Used (L)', 'Diesel Cost (R)', 
                   'Generator Used (L)', 'Generator Cost (R)']
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Calculate p-values for correlations
    p_values = np.zeros((len(numeric_cols), len(numeric_cols)))
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i != j:
                _, p_val = pearsonr(df[col1].dropna(), df[col2].dropna())
                p_values[i, j] = p_val
            else:
                p_values[i, j] = 0
    
    return correlation_matrix, p_values

def detect_anomalies(df):
    """Detect anomalies using statistical methods"""
    numeric_cols = ['Water Used (L)', 'Diesel Used (L)', 'Generator Used (L)']
    anomalies = {}
    
    for col in numeric_cols:
        data = df[col].values.reshape(-1, 1)
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df[col]))
        z_anomalies = df[z_scores > 2.5].index.tolist()
        
        # IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        anomalies[col] = {
            'z_score_anomalies': z_anomalies,
            'iqr_anomalies': iqr_anomalies,
            'combined_anomalies': list(set(z_anomalies + iqr_anomalies))
        }
    
    return anomalies

def simple_forecast(df, periods=3):
    """Simple linear regression forecast"""
    forecasts = {}
    numeric_cols = ['Water Used (L)', 'Diesel Used (L)', 'Generator Used (L)']
    
    # Create time index
    df_sorted = df.sort_values(['Year', 'Month'])
    df_sorted['time_index'] = range(len(df_sorted))
    
    for col in numeric_cols:
        if len(df_sorted) >= 3:  # Need at least 3 points for meaningful forecast
            X = df_sorted['time_index'].values.reshape(-1, 1)
            y = df_sorted[col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future periods
            future_indices = np.arange(len(df_sorted), len(df_sorted) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_indices)
            
            # Calculate R² score
            r2_score = model.score(X, y)
            
            forecasts[col] = {
                'values': forecast_values,
                'r2_score': r2_score,
                'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing'
            }
    
    return forecasts

def create_interactive_charts(df):
    """Create interactive Plotly charts"""
    charts = {}
    
    # Time series chart
    fig_ts = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Water Usage', 'Diesel Usage', 'Generator Usage', 'Costs'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Add traces for different metrics
    df_sorted = df.sort_values(['Year', 'Month'])
    x_labels = [f"{row['Month']} {row['Year']}" for _, row in df_sorted.iterrows()]
    
    fig_ts.add_trace(go.Scatter(x=x_labels, y=df_sorted['Water Used (L)'], 
                               name='Water (L)', line=dict(color='blue')), row=1, col=1)
    fig_ts.add_trace(go.Scatter(x=x_labels, y=df_sorted['Diesel Used (L)'], 
                               name='Diesel (L)', line=dict(color='red')), row=1, col=2)
    fig_ts.add_trace(go.Scatter(x=x_labels, y=df_sorted['Generator Used (L)'], 
                               name='Generator (L)', line=dict(color='green')), row=2, col=1)
    fig_ts.add_trace(go.Scatter(x=x_labels, y=df_sorted['Diesel Cost (R)'], 
                               name='Diesel Cost', line=dict(color='orange')), row=2, col=2)
    fig_ts.add_trace(go.Scatter(x=x_labels, y=df_sorted['Generator Cost (R)'], 
                               name='Generator Cost', line=dict(color='purple')), row=2, col=2)
    
    fig_ts.update_layout(height=600, title_text="Usage and Cost Trends Over Time")
    charts['time_series'] = fig_ts
    
    # Efficiency scatter plot
    metrics_df = calculate_efficiency_metrics(df)
    fig_efficiency = px.scatter(metrics_df, 
                               x='Water Used (L)', 
                               y='Total Fuel Cost (R)',
                               size='Total Fuel Used (L)',
                               color='Year',
                               hover_data=['Month', 'Cost Efficiency Score'],
                               title='Water Usage vs Total Fuel Cost (Size = Total Fuel Used)')
    charts['efficiency'] = fig_efficiency
    
    return charts

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

def generate_summary_paragraph(summary_df):
    lines = []
    for _, row in summary_df.iterrows():
        direction = "increased" if row["% Change"] > 0 else "decreased"
        lines.append(f"{row['Metric']} {direction} by {abs(row['% Change']):.1f}%")
    paragraph = "In the selected months, " + "; ".join(lines) + "."
    return paragraph

def plot_metric_with_trend(df, metric):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = df[metric].plot(kind="bar", ax=ax)

    # Add value labels to each bar
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

def generate_advanced_insights(df, stats_results, anomalies, forecasts):
    """Generate advanced insights and recommendations"""
    insights = []
    
    # Efficiency insights
    metrics_df = calculate_efficiency_metrics(df)
    avg_efficiency = metrics_df['Cost Efficiency Score'].mean()
    best_efficiency_month = metrics_df.loc[metrics_df['Cost Efficiency Score'].idxmin()]
    
    insights.append(f"💡 **Efficiency Analysis**: Average cost efficiency is R{avg_efficiency:.2f} per liter of water. "
                   f"Best efficiency was in {best_efficiency_month['Month']} {best_efficiency_month['Year']} "
                   f"(R{best_efficiency_month['Cost Efficiency Score']:.2f}/L).")
    
    # Trend insights
    for col, forecast in forecasts.items():
        trend_direction = forecast['trend']
        confidence = "high" if forecast['r2_score'] > 0.7 else "moderate" if forecast['r2_score'] > 0.4 else "low"
        insights.append(f"📈 **{col} Forecast**: {trend_direction.title()} trend with {confidence} confidence "
                       f"(R² = {forecast['r2_score']:.3f}). Next 3 months predicted: "
                       f"{', '.join([f'{v:.0f}L' for v in forecast['values']])}")
    
    # Anomaly insights
    total_anomalies = sum(len(anomaly_info['combined_anomalies']) for anomaly_info in anomalies.values())
    if total_anomalies > 0:
        insights.append(f"⚠️ **Anomaly Detection**: Found {total_anomalies} anomalous usage patterns that may need investigation.")
    
    # Statistical significance insights
    if 'year_comparison' in stats_results:
        significant_changes = [col for col, result in stats_results['year_comparison'].items() 
                             if result['significant_change']]
        if significant_changes:
            insights.append(f"📊 **Statistical Significance**: Significant year-over-year changes detected in: "
                           f"{', '.join(significant_changes)}")
    
    return insights

def generate_pdf_report(figures, summary_df, summary_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Water & Diesel Analytics Summary", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, summary_text.encode("latin-1", "ignore").decode("latin-1"))
    pdf.ln(4)

    # Table header
    pdf.set_font("Arial", "B", 10)
    headers = ["Metric", "2024 Total", "2024 Avg", "2025 Total", "2025 Avg", "% Change"]
    col_widths = [55, 30, 25, 30, 25, 25]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1)
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", "", 10)
    for _, row in summary_df.iterrows():
        values = [
            row["Metric"],
            f"{row['2024 Total']:.2f}",
            f"{row['2024 Avg']:.2f}",
            f"{row['2025 Total']:.2f}",
            f"{row['2025 Avg']:.2f}",
            f"{row['% Change']:.2f}%"
        ]
        for i, v in enumerate(values):
            pdf.cell(col_widths[i], 8, str(v), border=1)
        pdf.ln()

    # Add graphs
    for title, fig in figures:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format="png")
            tmpfile.flush()
            pdf.image(tmpfile.name, x=10, y=30, w=190)
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
        df_filtered = df[df["Month"].isin(selected_months)]

        if df_filtered.empty:
            st.warning("No data available for selected month(s).")
        else:
            # Calculate enhanced metrics
            metrics_df = calculate_efficiency_metrics(df_filtered)
            
            # Create tabs for different analysis sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Advanced Analytics", "🔍 Statistical Analysis", "🎯 Insights", "📋 Raw Data"])
            
            with tab1:
                st.subheader("📈 Interactive Visualizations")
                
                # Create interactive charts
                charts = create_interactive_charts(df_filtered)
                st.plotly_chart(charts['time_series'], use_container_width=True)
                st.plotly_chart(charts['efficiency'], use_container_width=True)
                
                # Traditional matplotlib charts
                pivot_df = df_filtered.pivot(index="Month", columns="Year", values=[
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
            
            with tab2:
                if show_advanced:
                    st.subheader("⚡ Efficiency Metrics")
                    
                    # Display efficiency metrics
                    efficiency_cols = ['Month', 'Year', 'Cost Efficiency Score', 'Water to Diesel Ratio', 
                                     'Generator to Diesel Ratio', 'Total Fuel Cost (R)', 'Total Fuel Used (L)']
                    st.dataframe(metrics_df[efficiency_cols])
                    
                    # Efficiency trends
                    fig_eff = px.line(metrics_df, x='Month', y='Cost Efficiency Score', color='Year',
                                     title='Cost Efficiency Trend (Lower is Better)')
                    st.plotly_chart(fig_eff, use_container_width=True)
                    
                if show_forecasting:
                    st.subheader("🔮 Usage Forecasting")
                    forecasts = simple_forecast(df_filtered)
                    
                    forecast_cols = st.columns(3)
                    for i, (metric, forecast) in enumerate(forecasts.items()):
                        with forecast_cols[i % 3]:
                            st.metric(
                                f"{metric} Trend",
                                f"{forecast['trend'].title()}",
                                f"R² = {forecast['r2_score']:.3f}"
                            )
                            st.write("Next 3 periods:")
                            for j, val in enumerate(forecast['values'], 1):
                                st.write(f"Period {j}: {val:.0f}L")
                
                if show_anomalies:
                    st.subheader("🚨 Anomaly Detection")
                    anomalies = detect_anomalies(df_filtered)
                    
                    for metric, anomaly_info in anomalies.items():
                        if anomaly_info['combined_anomalies']:
                            st.warning(f"**{metric}**: {len(anomaly_info['combined_anomalies'])} anomalies detected at indices: {anomaly_info['combined_anomalies']}")
                        else:
                            st.success(f"**{metric}**: No anomalies detected")
            
            with tab3:
                if show_correlations:
                    st.subheader("🔗 Correlation Analysis")
                    correlation_matrix, p_values = calculate_correlations(df_filtered)
                    
                    # Create correlation heatmap
                    fig_corr, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, ax=ax)
                    ax.set_title('Correlation Matrix of Usage Metrics')
                    st.pyplot(fig_corr)
                    
                    # Show strongest correlations
                    st.subheader("🎯 Strongest Correlations")
                    corr_pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_val = correlation_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:  # Only show strong correlations
                                corr_pairs.append({
                                    'Metric 1': correlation_matrix.columns[i],
                                    'Metric 2': correlation_matrix.columns[j],
                                    'Correlation': corr_val,
                                    'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                                })
                    
                    if corr_pairs:
                        st.dataframe(pd.DataFrame(corr_pairs))
                    else:
                        st.info("No strong correlations found (|r| > 0.5)")
                
                st.subheader("📊 Statistical Summary")
                stats_results = perform_statistical_analysis(df_filtered)
                
                # Descriptive statistics
                st.write("**Descriptive Statistics:**")
                st.dataframe(stats_results['descriptive'])
                
                # Normality test results
                if 'normality' in stats_results:
                    st.write("**Normality Tests (Shapiro-Wilk):**")
                    normality_df = pd.DataFrame.from_dict(stats_results['normality'], orient='index')
                    st.dataframe(normality_df)
                
                # Year comparison results
                if 'year_comparison' in stats_results:
                    st.write("**Year-over-Year Statistical Comparison (T-test):**")
                    comparison_df = pd.DataFrame.from_dict(stats_results['year_comparison'], orient='index')
                    st.dataframe(comparison_df)
            
            with tab4:
                st.subheader("🎯 Advanced Insights & Recommendations")
                
                # Generate insights
                stats_results = perform_statistical_analysis(df_filtered)
                anomalies = detect_anomalies(df_filtered)
                forecasts = simple_forecast(df_filtered)
                insights = generate_advanced_insights(df_filtered, stats_results, anomalies, forecasts)
                
                for insight in insights:
                    st.markdown(insight)
                
                # Cost optimization recommendations
                st.subheader("💰 Cost Optimization Recommendations")
                avg_diesel_cost = df_filtered['Diesel Cost (R)'].mean()
                avg_generator_cost = df_filtered['Generator Cost (R)'].mean()
                
                if avg_generator_cost > avg_diesel_cost:
                    savings_potential = avg_generator_cost - avg_diesel_cost
                    st.info(f"💡 **Generator Optimization**: Generator costs are R{savings_potential:.2f} higher on average than diesel costs. Consider optimizing generator usage.")
                
                # Seasonal patterns
                if len(df_filtered) >= 6:  # Need sufficient data for seasonal analysis
                    seasonal_analysis = df_filtered.groupby('Month')[['Water Used (L)', 'Diesel Used (L)', 'Generator Used (L)']].mean()
                    peak_usage_month = seasonal_analysis.sum(axis=1).idxmax()
                    st.info(f"📅 **Seasonal Pattern**: Peak usage typically occurs in {peak_usage_month}. Plan maintenance and procurement accordingly.")
            
            with tab5:
                st.subheader("📋 Enhanced Data Tables")
                
                # Original summary
                summary_df = calculate_summary_table(df_filtered)
                summary_text = generate_summary_paragraph(summary_df)
                
                st.subheader("📑 Executive Summary")
                st.text(summary_text)
                st.dataframe(summary_df)
                
                # Enhanced metrics table
                st.subheader("⚡ Efficiency & Ratio Metrics")
                st.dataframe(metrics_df)
                
                # Raw data
                st.subheader("📊 Raw Usage Data")
                st.dataframe(df_filtered)
                
                # Download options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df_filtered.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", csv, file_name="water_diesel_data.csv")
                
                with col2:
                    enhanced_csv = metrics_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Enhanced CSV", enhanced_csv, file_name="enhanced_water_diesel_data.csv")
                
                with col3:
                    pdf = generate_pdf_report(figures, summary_df, summary_text)
                    st.download_button("📄 Download PDF Report", data=pdf, file_name="water_diesel_report.pdf", mime="application/pdf")
