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
    ["📊 Dashboard", "📈 Forecast", "🧠 Decision", "ℹ️ About"]
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

    X, y = [], []
    lookback = 14

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
    )

    future_dates = [
        data.index[-1] + timedelta(days=i + 1) for i in range(30)
    ]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast Price": forecast_prices.flatten()
    })

    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df, use_container_width=True)

# -------------------------
# DECISION PAGE
# -------------------------
elif page == "🧠 Decision":

    st.subheader("🧠 Trading Decision")

    short_ma = data["Close"].rolling(10).mean().iloc[-1]
    long_ma = data["Close"].rolling(30).mean().iloc[-1]
    current_price = data["Close"].iloc[-1]

    if short_ma > long_ma:
        decision = "✅ BUY"
        reason = "Short-term trend above long-term trend"
    elif short_ma < long_ma:
        decision = "❌ SELL"
        reason = "Short-term trend below long-term trend"
    else:
        decision = "⚖️ HOLD"
        reason = "Market is neutral"

    st.metric("📌 Current Price ($)", f"{current_price:.2f}")
    st.metric("📊 Decision", decision)
    st.info(f"📖 Reason: {reason}")

# -------------------------
# ABOUT PAGE
# -------------------------
elif page == "ℹ️ About":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### Crypto Price Forecast & Decision System
    **Features**
    - Real-time crypto price tracking
    - LSTM-based forecasting
    - Buy / Sell / Hold decision system
    - Multi-page Streamlit app

    **Tech Stack**
    - Python
    - Streamlit
    - TensorFlow (LSTM)
    - Yahoo Finance API

    ⚠️ This project is for educational purposes only.
    """)

    st.success("Built by Sakthi Sowmiya 💙")


