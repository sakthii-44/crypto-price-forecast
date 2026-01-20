# ==============================
# CRYPTO PRICE FORECAST APP
# ==============================

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Crypto Price Forecast & Decision",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# LIGHT GREEN BACKGROUND
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
}
h1, h2, h3 {
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# CONSTANTS
# ==============================
CURRENCY = "usd"
HIST_DAYS = 120
FORECAST_DAYS = 30
LOOKBACK = 14
EPOCHS = 25

# ==============================
# FETCH COINS (SEARCHABLE)
# ==============================
@st.cache_data
def get_all_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    data = requests.get(url).json()
    return {coin["id"]: coin["symbol"].upper() for coin in data}

COINS = get_all_coins()

# ==============================
# FETCH PRICE DATA
# ==============================
def fetch_crypto_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": CURRENCY, "days": HIST_DAYS}
    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)
    return df[["price"]]

# ==============================
# TRAIN + FORECAST
# ==============================
def train_and_forecast(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=EPOCHS, batch_size=16, verbose=0)

    last_seq = scaled[-LOOKBACK:]
    future_scaled = []

    for _ in range(FORECAST_DAYS):
        pred = model.predict(last_seq.reshape(1, LOOKBACK, 1), verbose=0)
        future_scaled.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    return future_prices

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📌 Navigation")

coin_id = st.sidebar.selectbox(
    "🔍 Search Cryptocurrency",
    options=list(COINS.keys()),
    format_func=lambda x: f"{COINS[x]} ({x})"
)

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Forecast", "Decision", "About App"]
)

# ==============================
# LOAD DATA
# ==============================
df = fetch_crypto_data(coin_id)
current_price = df["price"].iloc[-1]

future_prices = train_and_forecast(df)
future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=FORECAST_DAYS
)

# ==============================
# PAGE 1: DASHBOARD
# ==============================
if page == "Dashboard":
    st.title("🚀 Crypto Price Forecast & Decision System")

    col1, col2 = st.columns(2)
    col1.metric("💰 Current Price ($)", f"{current_price:.2f}")
    col2.metric("🪙 Coin", COINS[coin_id])

    fig, ax = plt.subplots()
    ax.plot(df.index, df["price"])
    ax.set_title("Historical Price (Last 120 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)

# ==============================
# PAGE 2: FORECAST
# ==============================
elif page == "Forecast":
    st.title("📈 Price Forecast (Next 30 Days)")

    fig, ax = plt.subplots()
    ax.plot(future_dates, future_prices, marker="o")
    ax.set_title("Forecasted Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)

# ==============================
# PAGE 3: DECISION (FULL LOGIC)
# ==============================
elif page == "Decision":
    st.title("🧠 Trading Decision")

    buy_price = current_price
    future_df = pd.DataFrame({
        "date": future_dates,
        "price": future_prices
    })
    future_df["profit_pct"] = (future_df["price"] - buy_price) / buy_price * 100

    worst_day = future_df.loc[future_df["profit_pct"].idxmin()]
    profitable_days = future_df[future_df["profit_pct"] > 0]

    if not profitable_days.empty:
        best_day = profitable_days.iloc[0]
        strategy = "WAIT → BUY → SELL"
        decision = "BUY"
    else:
        best_day = None
        strategy = "SELL or HOLD"
        decision = "SELL"

    st.metric("📌 Decision", decision)
    st.write(f"**Buy Price (Today):** ${buy_price:.4f}")
    st.write(
        f"**Worst Sell Date:** {worst_day['date'].date()} → "
        f"${worst_day['price']:.4f} ({worst_day['profit_pct']:.2f}%)"
    )

    if best_day is not None:
        st.write(
            f"**Best Buy Date:** {best_day['date'].date()} → "
            f"${best_day['price']:.4f} (+{best_day['profit_pct']:.2f}%)"
        )
    else:
        st.error("❌ No profitable date predicted")

    st.success(f"📊 Strategy: {strategy}")

# ==============================
# PAGE 4: ABOUT APP + ACCURACY
# ==============================
elif page == "About App":
    st.title("ℹ️ About This App")

    st.write("""
This application uses **LSTM deep learning** to forecast cryptocurrency prices
based on historical time-series data.

Instead of traditional accuracy, we measure **directional accuracy**
(UP / DOWN prediction correctness).
    """)

    actual_direction = np.sign(np.diff(df["price"].values))
    predicted_direction = np.sign(np.diff(future_prices[:len(actual_direction)]))

    directional_accuracy = (
        (actual_direction == predicted_direction).sum()
        / len(actual_direction)
    ) * 100

    st.metric("📊 Directional Accuracy", f"{directional_accuracy:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(["Accuracy"], [directional_accuracy])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    st.pyplot(fig, use_container_width=True)


