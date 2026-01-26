import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Crypto Forecast – LSTM",
    page_icon="🚀",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (PRO UI)
# -------------------------------------------------
st.markdown("""
<style>
.card {
    background: linear-gradient(145deg,#0b1220,#050816);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(74,222,222,0.08);
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
.sub {
    color:#9CA3AF;
}
.glow {
    color:#4ADEDE;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# COINGECKO FUNCTIONS
# -------------------------------------------------
@st.cache_data
def get_coin_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    return requests.get(url).json()

@st.cache_data
def get_price_history(coin_id, days=120):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    data = requests.get(url, params=params).json()
    prices = [p[1] for p in data["prices"]]
    return np.array(prices).reshape(-1, 1)

# -------------------------------------------------
# LSTM TRAINING
# -------------------------------------------------
@st.cache_resource
def train_lstm(prices):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1],1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(25):
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        progress.progress((epoch+1)/25)
        status.info(f"Training LSTM Model... Epoch {epoch+1}/25")
        time.sleep(0.05)

    return model, scaler

# -------------------------------------------------
# FORECAST
# -------------------------------------------------
def forecast_price(model, scaler, prices):
    last_60 = prices[-60:]
    scaled = scaler.transform(last_60)
    X = np.array([scaled])
    preds = []

    for _ in range(30):
        p = model.predict(X, verbose=0)
        preds.append(p[0][0])
        X = np.append(X[:,1:,:], [[p]], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1,1))

# -------------------------------------------------
# SIDEBAR NAV
# -------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["Dashboard","Forecast","Decision","About"])

coins = get_coin_list()
coin_name = st.sidebar.selectbox(
    "🔍 Search Cryptocurrency",
    options=coins,
    format_func=lambda x: x["name"]
)
coin_id = coin_name["id"]

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
prices = get_price_history(coin_id)
model, scaler = train_lstm(prices)
future = forecast_price(model, scaler, prices)

current_price = prices[-1][0]
pred_price = future[-1][0]
expected_return = ((pred_price-current_price)/current_price)*100

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
if page == "Dashboard":
    st.markdown("## 🚀 Crypto Forecast Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><div class='sub'>Current Price</div><div class='metric'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='sub'>LSTM Prediction</div><div class='metric'>{expected_return:+.2f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><div class='sub'>Model</div><div class='metric glow'>LSTM</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><div class='sub'>Forecast Days</div><div class='metric'>30</div></div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(prices, label="Historical", color="#4ADEDE")
    ax.set_title("Price History (120 Days)")
    ax.legend()
    st.pyplot(fig)

# -------------------------------------------------
# FORECAST PAGE
# -------------------------------------------------
elif page == "Forecast":
    st.markdown("## 📈 LSTM 30-Day Forecast")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(prices, label="Historical")
    ax.plot(range(len(prices), len(prices)+30), future, label="Forecast")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"""
    <div class="card">
        <div class="sub">Starting Price</div>
        <div class="metric">${current_price:,.2f}</div>
        <br>
        <div class="sub">Predicted Price (30d)</div>
        <div class="metric">${pred_price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# DECISION PAGE
# -------------------------------------------------
elif page == "Decision":
    decision = "HOLD"
    if expected_return > 1:
        decision = "BUY"
    elif expected_return < -1:
        decision = "SELL"

    st.markdown("## 🧠 Trading Decision")

    st.markdown(f"""
    <div class="card">
        <h1 class="glow">{decision}</h1>
        <p>Expected Return: {expected_return:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------
else:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    <div class="card">
    This application uses a **Long Short-Term Memory (LSTM)** neural network  
    trained on **120 days of historical crypto prices** to forecast the next **30 days**.

    • Real-time CoinGecko data  
    • 2-layer LSTM (64 → 32)  
    • Adam Optimizer + MSE Loss  
    • Streamlit-optimized (low memory)
    </div>
    """, unsafe_allow_html=True)


