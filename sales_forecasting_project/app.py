import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sales Forecasting", layout="wide")

df = pd.read_csv("../data/sales_data.csv")
df["date"] = pd.to_datetime(df["date"])

st.title("📊 Sales Forecasting Dashboard")

region = st.sidebar.selectbox("Region", ["All"] + list(df["region"].unique()))
product = st.sidebar.selectbox("Product", ["All"] + list(df["product_line"].unique()))

if region != "All":
    df = df[df["region"] == region]
if product != "All":
    df = df[df["product_line"] == product]

c1, c2, c3 = st.columns(3)
c1.metric("Total Revenue", f"₹{df['revenue'].sum():,.0f}")
c2.metric("Units Sold", int(df["units_sold"].sum()))
c3.metric("Avg Price", f"₹{df['unit_price'].mean():.2f}")

st.subheader("Revenue Trend")
st.line_chart(df.groupby("date")["revenue"].sum())

st.subheader("Sales Data")
st.dataframe(df, use_container_width=True)
