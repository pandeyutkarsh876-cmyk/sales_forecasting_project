import streamlit as st
import pandas as pd

st.title("Sales Forecasting Dashboard")

df = pd.read_csv("sales_data.csv")

st.subheader("Sales Data")
st.dataframe(df.head())

st.subheader("Revenue Trend")
st.line_chart(df["revenue"])
