import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import parser
from dotenv import load_dotenv
import os
import re
from langchain_groq import ChatGroq

# 💡 Custom logic from agents.py
from agents import (
    df_to_summary_string,
    extract_total_cost,
    extract_average_cost,
    detect_anomalies,
    suggest_optimizations,
    cost_forecast
)

# --- App Config ---
st.set_page_config(page_title="AWS S3 Cost Analyzer", layout="centered")
st.title("AWS S3 Cost Intelligence")

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Set up LLM ---
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

# --- Upload + Input ---
uploaded_file = st.file_uploader("Upload AWS S3 CSV", type="csv")
user_input = st.text_input("Enter Month and Year (e.g. 'March 2025')")

# --- Analyze Button ---
if st.button("Analyze"):
    if not uploaded_file:
        st.warning("Please upload a CSV file first.")
        st.stop()

    if not user_input:
        st.warning("Please enter a valid month and year like 'March 2025'.")
        st.stop()

    # Parse date input
    try:
        cleaned_input = user_input.strip().replace("\n", " ")
        parsed_date = parser.parse(cleaned_input)
        month, year = parsed_date.month, parsed_date.year
    except (ValueError, TypeError):
        st.error("Invalid date format. Please enter like 'March 2025'")
        st.stop()

    # Read and clean file
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Cost'])
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
        df = df.dropna(subset=['Cost'])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Filter by month and year
    df = df[(df['Date'].dt.month == month) & (df['Date'].dt.year == year)]

    if df.empty:
        st.warning("No cost data found for that month/year.")
    else:
        st.success(f"Found {len(df)} records for {user_input}")
        summary_text = df_to_summary_string(df)

        st.metric("Total Cost", f"${extract_total_cost(df):,.2f}")
        st.metric("Avg Daily Cost", f"${extract_average_cost(df):,.2f}")

        st.subheader("Anomalies")
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            st.markdown("No anomalies detected.")
        else:
            for _, row in anomalies.iterrows():
                st.markdown(f"- **{row['Date']}** →  ${row['Anomalous Cost']:.2f}")

        st.subheader("Optimization Tips")
        tips = suggest_optimizations(df)
        for i, tip in enumerate(tips, 1):
            st.markdown(f"**{i}.** {tip}")

        # Forecast
# --- Forecast Section ---
st.subheader("Forecast (Next Month)")

try:
    # Convert parsed_date (from user input) into string format for function
    start_date_str = parsed_date.strftime("%Y-%m-%d")

    result = cost_forecast(df, start_date_str)
    forecast_text = result.get("forecast")
    forecast_data = result.get("raw_prediction")

    if forecast_data is None:
        st.warning(forecast_text or "Forecast could not be generated.")
    else:
        # 🔍 Show forecast date range for debug
        forecast_start = parsed_date.replace(day=1) + relativedelta(months=1)  # 1st April
        forecast_end = (forecast_start + relativedelta(months=1)) - timedelta(days=1)  # 30th April
        st.info(f"🕵️ Forecast Window: {forecast_start} to {forecast_end}")

        # 💰 Show predicted total cost
        predicted_total = forecast_data['yhat'].sum()
        st.metric("Predicted 30-Day S3 Cost", f"${predicted_total:,.2f}")

        # 🧠 LLM Summary
        st.markdown(forecast_text)

        # 📊 Optional daily view
        # st.dataframe(forecast_data[['ds', 'yhat']], use_container_width=True)
except Exception as e:
    st.error(f"Forecast error: {e}")
