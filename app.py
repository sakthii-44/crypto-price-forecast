import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crypto Price Forecast", layout="centered")

# ---------------- LIGHT THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------- CONSTANTS ----------------
HIST_DAYS = 120
FORECAST_DAYS = 30
LOOKBACK = 14
EPOCHS = 10   # 🔥 reduced
CURRENCY = "usd"

# ---------------- API FUNCTIONS ----------------
@st.cache_data
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 20}
    data = requests.get(url).json()
    return {coin["id"]: coin["symbol"].upper() for coin in data}

@st.cache_data
def fetch_price_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": CURRENCY, "days": HIST_DAYS}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data["prices"], columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("date")[["price"]]

# ---------------- MODEL (CACHED) ----------------
@st.cache_resource
def forecast_prices(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(32, input_shape=(LOOKBACK, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=EPOCHS, batch_size=16, verbose=0)

    last_seq = scaled[-LOOKBACK:]
    future = []

    for _ in range(FORECAST_DAYS):
        pred = model.predict(last_seq.reshape(1, LOOKBACK, 1), verbose=0)
        future.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    return scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")

coins = get_coins()
coin = st.sidebar.selectbox(
    "🔍 Select Cryptocurrency",
    list(coins.keys()),
    format_func=lambda x: f"{coins[x]} ({x})"
)

page = st.sidebar.radio("Go to", ["Dashboard", "Forecast", "Decision", "About App"])

# ---------------- DATA LOAD ----------------
df = fetch_price_data(coin)
future_prices = forecast_prices(df)
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)
today_price = df["price"].iloc[-1]

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🚀 Crypto Price Dashboard")
    st.metric("Current Price ($)", f"{today_price:.2f}")
    st.line_chart(df["price"])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FORECAST ----------------
elif page == "Forecast":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📈 30-Day Forecast")
    fig, ax = plt.subplots()
    ax.plot(future_dates, future_prices, marker="o")
    ax.set_ylabel("USD")
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DECISION ----------------
elif page == "Decision":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🧠 Trade Decision")

    future_df = pd.DataFrame({"price": future_prices}, index=future_dates)
    future_df["profit"] = (future_df["price"] - today_price) / today_price * 100

    worst = future_df.iloc[future_df["profit"].idxmin()]
    profitable = future_df[future_df["profit"] > 0]

    st.write(f"**Buy Price:** ${today_price:.4f}")
    st.write(f"**Worst Case:** {worst.name.date()} → {worst.profit:.2f}%")

    if not profitable.empty:
        best = profitable.iloc[0]
        st.success(f"📈 Best Buy Date: {best.name.date()} (+{best.profit:.2f}%)")
        st.success("Strategy: WAIT → BUY → SELL")
    else:
        st.warning("❌ No profitable window predicted")
        st.warning("Strategy: SELL or HOLD")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif page == "About App":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About App")

    actual = np.sign(np.diff(df["price"].values))
    predicted = np.sign(np.diff(future_prices[:len(actual)]))
    accuracy = (actual == predicted).mean() * 100

    st.metric("Directional Accuracy", f"{accuracy:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(["Accuracy"], [accuracy])
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    st.info("LSTM-based crypto price forecasting using historical market data.")
    st.markdown("</div>", unsafe_allow_html=True)


