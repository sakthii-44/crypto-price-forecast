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
# SIDEBAR
# -------------------------
st.sidebar.title("📌 Navigation")

crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD"]
)

page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "📈 Forecast", "🧠 Decision", "ℹ️ App Info"]
)

st.title("🚀 Crypto Price Forecast & Decision System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    df = df[["Close"]].dropna()
    return df

data = load_data(crypto)

if data.empty:
    st.error("⚠️ Failed to load market data.")
    st.stop()

# -------------------------
# DASHBOARD PAGE
# -------------------------
if page == "📊 Dashboard":

    st.subheader(f"📊 Market Overview — {crypto}")

    latest_price = float(data["Close"].iloc[-1])
    prev_price = float(data["Close"].iloc[-2])

    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Latest Price ($)", f"{latest_price:.2f}")
    col2.metric("📉 Daily Change ($)", f"{change:.2f}", f"{pct_change:.2f}%")
    col3.metric("📆 Data Points", len(data))

    st.markdown("### 📉 Price Trend (1 Year)")
    fig, ax = plt.subplots()
    ax.plot(data.index, data["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    st.pyplot(fig)

# -------------------------
# FORECAST PAGE
# -------------------------
elif page == "📈 Forecast":

    st.subheader("📈 30-Day Price Forecast (LSTM)")

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
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

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

    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df, use_container_width=True)

# -------------------------
# DECISION PAGE (ERROR-FREE)
# -------------------------
elif page == "🧠 Decision":

    st.subheader("🧠 Trading Decision")

    short_ma = data["Close"].rolling(10).mean().iloc[-1]
    long_ma = data["Close"].rolling(30).mean().iloc[-1]
    current_price = data["Close"].iloc[-1]

    if short_ma > long_ma:
        decision = "✅ BUY"
        reason = "Short-term trend is above long-term trend"
    elif short_ma < long_ma:
        decision = "❌ SELL"
        reason = "Short-term trend is below long-term trend"
    else:
        decision = "⚖️ HOLD"
        reason = "Market trend is neutral"

    st.metric("💰 Current Price ($)", f"{current_price:.2f}")
    st.metric("📊 Decision", decision)
    st.info(f"📖 Reason: {reason}")

# -------------------------
# APP INFO PAGE (ACCURACY + CHART)
# -------------------------
elif page == "ℹ️ App Info":

    st.subheader("ℹ️ Application Overview")

    st.markdown("""
    This application helps users **analyze cryptocurrency trends** and
    **forecast future prices** using deep learning.
    """)

    st.markdown("### 📊 Model Accuracy (Backtesting)")

    # Use last 30 days as test data
    test_days = 30
    prices = data["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    lookback = 14

    for i in range(lookback, len(scaled) - test_days):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # Predict last 30 days
    test_inputs = scaled[-(test_days + lookback):-test_days]
    preds = []

    for i in range(test_days):
        pred = model.predict(test_inputs.reshape(1, lookback, 1), verbose=0)
        preds.append(pred[0, 0])
        test_inputs = np.append(test_inputs[1:], pred, axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actual = prices[-test_days:].flatten()

    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    accuracy = 100 - mape

    st.success(f"✅ Forecast Accuracy: **{accuracy:.2f}%** (Last 30 Days)")

    acc_df = pd.DataFrame({
        "Date": data.index[-test_days:],
        "Actual Price": actual,
        "Predicted Price": preds
    })

    st.line_chart(acc_df.set_index("Date"))

    st.markdown("""
    ### 🔍 Key Highlights
    - Uses **LSTM Neural Networks**
    - Trained on **1 year of historical data**
    - Accuracy measured using **MAPE**
    - Designed for **trend understanding**, not instant trading
    """)
