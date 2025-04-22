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
            summary_text = generate_summary_paragraph(summary_df)

            st.subheader("📑 Auto Summary")
            st.text(summary_text)
            st.dataframe(summary_df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, file_name="water_diesel_data.csv")

            pdf = generate_pdf_report(figures, summary_df, summary_text)
            st.download_button("📄 Download PDF Report", data=pdf, file_name="water_diesel_report.pdf", mime="application/pdf")
