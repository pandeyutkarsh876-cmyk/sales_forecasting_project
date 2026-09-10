import streamlit as st
import pandas as pd

st.title("Sales Forecasting Dashboard")

from pathlib import Path

csv_path = Path(__file__).parent / "sales_data.csv"
df = pd.read_csv(csv_path)

st.subheader("Sales Data")
st.dataframe(df.head())

st.subheader("Revenue Trend")
st.line_chart(df["revenue"])
