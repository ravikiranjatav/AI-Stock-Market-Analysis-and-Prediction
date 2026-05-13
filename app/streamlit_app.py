import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------
# PAGE TITLE
# -------------------------

st.title("📈 AI Stock Market Dashboard")

st.write("Financial Analysis & ML Prediction System")

# -------------------------
# STOCK INPUT
# -------------------------

stock = st.sidebar.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

# -------------------------
# DOWNLOAD DATA
# -------------------------

df = yf.download(
    stock,
    start="2020-01-01"
)

# -------------------------
# SHOW DATA
# -------------------------

st.subheader("Stock Data")

st.dataframe(df.tail())

# -------------------------
# CLOSE PRICE CHART
# -------------------------

st.subheader("Closing Price Trend")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(df['Close'])

ax.set_title(f"{stock} Closing Price")

st.pyplot(fig)

# -------------------------
# MOVING AVERAGES
# -------------------------

df['SMA_20'] = df['Close'].rolling(20).mean()

df['SMA_50'] = df['Close'].rolling(50).mean()

st.subheader("Moving Averages")

fig2, ax2 = plt.subplots(figsize=(12,6))

ax2.plot(df['Close'], label='Close')

ax2.plot(df['SMA_20'], label='SMA 20')

ax2.plot(df['SMA_50'], label='SMA 50')

ax2.legend()

st.pyplot(fig2)

# -------------------------
# DAILY RETURNS
# -------------------------

df['Daily_Return'] = df['Close'].pct_change()

st.subheader("Daily Returns")

fig3, ax3 = plt.subplots(figsize=(12,6))

ax3.plot(df['Daily_Return'])

st.pyplot(fig3)

# -------------------------
# METRICS
# -------------------------

st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    round(df['Close'].iloc[-1], 2)
)

col2.metric(
    "Highest Price",
    round(df['High'].max(), 2)
)

col3.metric(
    "Lowest Price",
    round(df['Low'].min(), 2)
)