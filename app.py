import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crypto Price Forecast & Decision System",
    page_icon="📊",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📈 Forecast", "🤖 Trading Decision", "ℹ️ About"]
)

crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD"]
)

# ---------------- DATA FUNCTION ----------------
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, period="6mo", interval="1d")
    data = data[['Close']].dropna()
    return data

data = load_data(crypto)

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.title("🚀 Crypto Price Forecast & Decision Support System")

    col1, col2, col3 = st.columns(3)

    col1.metric("📅 Data Period", "Last 6 Months")
    col2.metric("💱 Selected Crypto", crypto)
    col3.metric("📈 Latest Price ($)", f"{data['Close'].iloc[-1]:.2f}")

    st.subheader("📊 Closing Price Trend")
    st.line_chart(data['Close'])

    st.success("This system forecasts crypto prices and provides BUY / SELL decisions using Deep Learning (LSTM).")

# ---------------- FORECAST PAGE ----------------
elif page == "📈 Forecast":
    st.title("📈 Crypto Price Forecast")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    lookback = 14
    X, y = [], []

    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    with st.spinner("Training LSTM model..."):
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)

    last_seq = scaled_data[-lookback:]
    future = []

    for _ in range(7):
        pred = model.predict(last_seq.reshape(1, lookback, 1), verbose=0)
        future.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Price ($)": future_prices.flatten()
    })

    st.subheader("🔮 7-Day Price Forecast")
    st.dataframe(forecast_df, use_container_width=True)

    st.line_chart(forecast_df["Predicted Price ($)"])

# ---------------- DECISION PAGE ----------------
elif page == "🤖 Trading Decision":
    st.title("🤖 AI-Based Trading Decision")

    last_price = data['Close'].iloc[-1]
    avg_future = np.mean(data['Close'].tail(5))

    if avg_future > last_price * 1.02:
        decision = "BUY 🟢"
        color = "green"
        reason = "Expected upward trend"
    elif avg_future < last_price * 0.98:
        decision = "SELL 🔴"
        color = "red"
        reason = "Expected downward trend"
    else:
        decision = "HOLD 🟡"
        color = "orange"
        reason = "Market is stable"

    st.markdown(f"## 🧠 Decision: :{color}[{decision}]")
    st.write(f"**Reason:** {reason}")
    st.write(f"**Current Price:** ${last_price:.2f}")
    st.write(f"**Recent Average:** ${avg_future:.2f}")

    st.warning("⚠️ This is an educational prediction system, not financial advice.")

# ---------------- ABOUT PAGE ----------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 📌 Project Title
    **Crypto Price Forecast & Trading Decision System**

    ### 🧠 Technologies Used
    - Python
    - Streamlit
    - LSTM (Deep Learning)
    - Yahoo Finance API

    ### 🎯 Objective
    To forecast cryptocurrency prices and assist users with BUY / SELL / HOLD decisions.

    ### 👩‍💻 Developed By
    **Sakthi Sowmiya**

    ### 🏫 Use Case
    - Final Year Project
    - Internship Project
    - Portfolio Demonstration
    """)

    st.success("Thank you for using this application!")
