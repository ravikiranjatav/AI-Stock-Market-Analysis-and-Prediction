import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as pltimport streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from nsepython import *
# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="AI Stock Market Dashboard",
    layout="wide"
)

# -------------------------
# RSI FUNCTION
# -------------------------

def calculate_rsi(data, window=14):

    delta = data['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi

# -------------------------
# MACD FUNCTION
# -------------------------

def calculate_macd(data):

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()

    exp2 = data['Close'].ewm(span=26, adjust=False).mean()

    macd = exp1 - exp2

    signal = macd.ewm(span=9, adjust=False).mean()

    histogram = macd - signal

    return macd, signal, histogram

# -------------------------
# PAGE TITLE
# -------------------------

st.title("📈 AI Stock Market Dashboard")

st.write("Financial Analysis & AI Prediction System")

# -------------------------
# STOCK LIST
# -------------------------

# -------------------------
# ALL NSE STOCKS
# -------------------------

all_nse_stocks = nse_eq_symbols()

# ADD .NS FOR YFINANCE
nse_stocks = {
    stock: f"{stock}.NS"
    for stock in all_nse_stocks
}

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("📊 NSE Dashboard")

selected_company = st.sidebar.selectbox(
    "Select NSE Company",
    sorted(list(nse_stocks.keys()))
)
stock = nse_stocks[selected_company]

# -------------------------
# MARKET OVERVIEW DATA
# -------------------------

market_data = []

for company, ticker in list(nse_stocks.items())[:20]:

    temp_df = yf.download(
        ticker,
        period="5d",
        progress=False,
        auto_adjust=True
    )

    if not temp_df.empty:

        latest_close = float(temp_df['Close'].iloc[-1])

        previous_close = float(temp_df['Close'].iloc[-2])

        percent_change = (
            (latest_close - previous_close)
            / previous_close
        ) * 100

        market_data.append({
            "Company": company,
            "Price": round(latest_close, 2),
            "Change %": round(percent_change, 2)
        })

market_df = pd.DataFrame(market_data)

# -------------------------
# TOP GAINERS / LOSERS
# -------------------------

top_gainers = market_df.sort_values(
    by="Change %",
    ascending=False
).head(5)

top_losers = market_df.sort_values(
    by="Change %",
    ascending=True
).head(5)

st.subheader("📊 Market Overview")

col_gainer, col_loser = st.columns(2)

with col_gainer:

    st.success("🚀 Top Gainers")

    st.dataframe(
        top_gainers,
        use_container_width=True
    )

with col_loser:

    st.error("📉 Top Losers")

    st.dataframe(
        top_losers,
        use_container_width=True
    )

# -------------------------
# DOWNLOAD STOCK DATA
# -------------------------

df = yf.download(
    stock,
    start="2020-01-01",
    auto_adjust=True
)

# FIX MULTIINDEX
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# REMOVE EMPTY ROWS
df.dropna(inplace=True)

# -------------------------
# INDICATORS
# -------------------------

df['RSI'] = calculate_rsi(df)

df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df)

df['SMA_20'] = df['Close'].rolling(20).mean()

df['SMA_50'] = df['Close'].rolling(50).mean()

df['Daily_Return'] = df['Close'].pct_change()

# REMOVE NaN AGAIN
df.dropna(inplace=True)

# -------------------------
# AI FEATURES
# -------------------------

df['Target'] = df['Close'].shift(-1)

features = [
    'Open',
    'High',
    'Low',
    'Volume',
    'SMA_20',
    'SMA_50',
    'RSI'
]

ai_df = df.dropna()

X = ai_df[features]

y = ai_df['Target']

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# RANDOM FOREST MODEL
# -------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# PREDICTIONS
# -------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

accuracy = r2_score(y_test, predictions) * 100

# -------------------------
# NEXT DAY PREDICTION
# -------------------------

latest_data = X.iloc[-1:]

next_day_prediction = model.predict(latest_data)

predicted_price = round(float(next_day_prediction[0]), 2)

# -------------------------
# BUY / SELL SIGNALS
# -------------------------

latest_rsi = df['RSI'].iloc[-1]

latest_macd = df['MACD'].iloc[-1]

latest_signal = df['Signal'].iloc[-1]

current_price = float(df['Close'].iloc[-1])

trading_signal = "HOLD"

if latest_rsi < 30 and latest_macd > latest_signal:
    trading_signal = "BUY"

elif latest_rsi > 70 and latest_macd < latest_signal:
    trading_signal = "SELL"

# -------------------------
# STOCK DATA
# -------------------------

st.subheader("📋 Stock Data")

st.dataframe(df.tail())

# -------------------------
# KEY METRICS
# -------------------------

st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

current_price_display = round(
    float(df['Close'].iloc[-1]), 2
)

highest_price = round(
    float(df['High'].max()), 2
)

lowest_price = round(
    float(df['Low'].min()), 2
)

col1.metric(
    "Current Price",
    f"₹ {current_price_display}"
)

col2.metric(
    "Highest Price",
    f"₹ {highest_price}"
)

col3.metric(
    "Lowest Price",
    f"₹ {lowest_price}"
)

# -------------------------
# AI PREDICTION
# -------------------------

st.subheader("🤖 AI Prediction")

col4, col5, col6 = st.columns(3)

col4.metric(
    "Predicted Next Day Price",
    f"₹ {predicted_price}"
)

col5.metric(
    "Model MAE",
    round(mae, 2)
)

col6.metric(
    "Model Accuracy",
    f"{round(accuracy,2)}%"
)

# -------------------------
# TRADING SIGNAL
# -------------------------

st.subheader("📢 AI Trading Signal")

if trading_signal == "BUY":

    st.success(
        f"✅ BUY SIGNAL for {selected_company}"
    )

elif trading_signal == "SELL":

    st.error(
        f"❌ SELL SIGNAL for {selected_company}"
    )

else:

    st.warning(
        f"⚠️ HOLD Signal for {selected_company}"
    )

# -------------------------
# MARKET STATUS
# -------------------------

price_change = predicted_price - current_price

if price_change > 0:

    st.success(
        f"📈 AI predicts upside movement of ₹ {round(price_change,2)}"
    )

else:

    st.error(
        f"📉 AI predicts downside movement of ₹ {round(abs(price_change),2)}"
    )

# -------------------------
# CANDLESTICK CHART
# -------------------------

st.subheader("🕯️ Candlestick Chart")

fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        )
    ]
)

fig_candle.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='SMA 20'
    )
)

fig_candle.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        name='SMA 50'
    )
)

fig_candle.update_layout(
    title=f"{stock} Candlestick Chart",
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    height=700
)

st.plotly_chart(
    fig_candle,
    use_container_width=True
)

# -------------------------
# CLOSING PRICE CHART
# -------------------------

st.subheader("📈 Closing Price Trend")

fig = px.line(
    df,
    x=df.index,
    y='Close',
    title=f"{stock} Closing Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------
# MOVING AVERAGES
# -------------------------

st.subheader("📊 Moving Averages")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Close Price'
    )
)

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='SMA 20'
    )
)

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        name='SMA 50'
    )
)

fig2.update_layout(height=500)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -------------------------
# DAILY RETURNS
# -------------------------

st.subheader("📉 Daily Returns")

fig3 = px.line(
    df,
    x=df.index,
    y='Daily_Return',
    title="Daily Returns"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# -------------------------
# RSI CHART
# -------------------------

st.subheader("📍 RSI Indicator")

fig_rsi = px.line(
    df,
    x=df.index,
    y='RSI',
    title='RSI Indicator'
)

fig_rsi.add_hline(y=70)

fig_rsi.add_hline(y=30)

st.plotly_chart(
    fig_rsi,
    use_container_width=True
)

# -------------------------
# MACD CHART
# -------------------------

st.subheader("📊 MACD Indicator")

fig_macd = go.Figure()

fig_macd.add_trace(
    go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD'
    )
)

fig_macd.add_trace(
    go.Scatter(
        x=df.index,
        y=df['Signal'],
        mode='lines',
        name='Signal'
    )
)

fig_macd.add_trace(
    go.Bar(
        x=df.index,
        y=df['Histogram'],
        name='Histogram'
    )
)

fig_macd.update_layout(
    title='MACD Indicator',
    xaxis_title='Date',
    yaxis_title='Value',
    height=500
)

st.plotly_chart(
    fig_macd,
    use_container_width=True
)

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
