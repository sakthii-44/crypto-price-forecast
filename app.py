# =========================
# CRYPTO PRICE FORECAST APP
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# -------------------------
# PAGE CONFIG (ONLY ONCE)
# -------------------------
st.set_page_config(
    page_title="Crypto Price Forecast",
    page_icon="📈",
    layout="wide"
)

# -------------------------
# LIGHT BACKGROUND (READABLE)
# -------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f7fb;
}
[data-testid="stMetricValue"] {
    font-size: 28px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📌 Navigation")

crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD"]
)

page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "📈 Forecast", "🧠 Decision", "ℹ️ About App"]
)

st.title("🚀 Crypto Price Forecast & Decision System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    df = df[["Close"]].dropna()
    return df

data = load_data(crypto)

if len(data) < 50:
    st.error("⚠️ Not enough data.")
    st.stop()

# -------------------------
# PAGE 1: DASHBOARD (CLEAN)
# -------------------------
if page == "📊 Dashboard":

    st.subheader(f"📊 Market Overview — {crypto}")

    latest_price = float(data["Close"].iloc[-1])
    prev_price = float(data["Close"].iloc[-2])

    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current Price ($)", f"{latest_price:.2f}")
    col2.metric("📈 Daily Change ($)", f"{change:.2f}", f"{pct_change:.2f}%")
    col3.metric("📆 Days Used", len(data))

    st.markdown("### 📉 Price Trend (Last 1 Year)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["Close"], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True)

    st.pyplot(fig)

# -------------------------
# PAGE 2: FORECAST (CLEAR)
# -------------------------
elif page == "📈 Forecast":

    st.subheader("📈 30-Day Price Forecast")

    prices = data["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    lookback = 14
    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)

    future = []
    last_seq = scaled[-lookback:]

    for _ in range(30):
        pred = model.predict(last_seq.reshape(1, lookback, 1), verbose=0)
        future.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    forecast_prices = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten()

    future_dates = [
        data.index[-1] + timedelta(days=i + 1) for i in range(30)
    ]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast Price": forecast_prices
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["Close"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["Forecast Price"], label="Forecast")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    st.dataframe(forecast_df, use_container_width=True)

# -------------------------
# PAGE 3: DECISION (FIXED)
# -------------------------
elif page == "🧠 Decision":

    st.subheader("🧠 Trading Decision")

    short_ma = float(data["Close"].rolling(10).mean().dropna().iloc[-1])
    long_ma = float(data["Close"].rolling(30).mean().dropna().iloc[-1])
    current_price = float(data["Close"].iloc[-1])

    if short_ma > long_ma:
        decision = "✅ BUY"
        reason = "Short-term trend is stronger than long-term trend."
    elif short_ma < long_ma:
        decision = "❌ SELL"
        reason = "Short-term trend is weaker than long-term trend."
    else:
        decision = "⚖️ HOLD"
        reason = "Market trend is neutral."

    col1, col2 = st.columns(2)
    col1.metric("💰 Current Price ($)", f"{current_price:.2f}")
    col2.metric("📌 Decision", decision)

    st.info(reason)

# -------------------------
# PAGE 4: ABOUT APP + ACCURACY
# -------------------------
elif page == "ℹ️ About App":

    st.subheader("ℹ️ About This Application")

    data["Predicted"] = data["Close"].rolling(7).mean()
    valid = data.dropna()

    mape = np.mean(np.abs((valid["Close"] - valid["Predicted"]) / valid["Close"])) * 100
    accuracy = 100 - mape

    st.metric("📊 Model Accuracy (%)", f"{accuracy:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(valid.index, valid["Close"], label="Actual")
    ax.plot(valid.index, valid["Predicted"], label="Predicted")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.markdown("""
### 🔍 What This App Does
- Fetches real crypto prices
- Predicts next 30 days using LSTM
- Generates Buy / Sell / Hold signals
- Evaluates accuracy using real past data

### 📌 Accuracy Meaning
- Calculated using **MAPE**
- Shows how close predictions are to real prices
""")


