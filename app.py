import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crypto Forecast",
    page_icon="🚀",
    layout="wide"
)

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8fff1, #f4fff9);
}
.metric {
    padding: 20px;
    border-radius: 12px;
    background: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    text-align: center;
}
h1, h2, h3 {
    color: #064e3b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 Crypto Forecast")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Forecast", "Decision", "About"]
)

coin_symbol = st.sidebar.text_input(
    "🔍 Search Coin (Yahoo Symbol)",
    value="BTC-USD"
).upper()

# ---------------- DATA FETCH ----------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="6mo")
    df = df.reset_index()
    return df

data = load_data(coin_symbol)

# ---------------- HELPERS ----------------
def train_model(prices):
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 Crypto Dashboard")

    if data.empty:
        st.error("Invalid coin symbol")
    else:
        current_price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2]
        change = ((current_price - prev_price) / prev_price) * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"<div class='metric'><h4>Price</h4><h2>${current_price:.2f}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric'><h4>24h Change</h4><h2>{change:.2f}%</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric'><h4>High</h4><h2>${data['High'].max():.2f}</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric'><h4>Low</h4><h2>${data['Low'].min():.2f}</h2></div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Price"
        ))
        fig.update_layout(
            title="Price History",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECAST ----------------
elif page == "Forecast":
    st.title("📈 30-Day Price Forecast")

    prices = data["Close"].values
    model = train_model(prices)

    future_days = 30
    future_X = np.arange(len(prices), len(prices) + future_days).reshape(-1, 1)
    forecast = model.predict(future_X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=prices,
        mode="lines",
        name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=pd.date_range(data["Date"].iloc[-1], periods=future_days),
        y=forecast,
        mode="lines",
        name="Forecast"
    ))

    fig.update_layout(
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Predicted price after 30 days: **${forecast[-1]:.2f}**")

# ---------------- DECISION ----------------
elif page == "Decision":
    st.title("🤖 Trade Decision")

    prices = data["Close"].values
    model = train_model(prices)

    predicted_price = model.predict([[len(prices) + 30]])[0]
    current_price = prices[-1]
    expected_return = ((predicted_price - current_price) / current_price) * 100

    if expected_return > 5:
        decision = "BUY 🟢"
    elif expected_return < -5:
        decision = "SELL 🔴"
    else:
        decision = "HOLD 🟡"

    st.markdown(f"""
    <div class="metric">
        <h2>{decision}</h2>
        <p>Expected Return: <b>{expected_return:.2f}%</b></p>
        <p>Current Price: ${current_price:.2f}</p>
        <p>Predicted Price: ${predicted_price:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About This App")

    prices = data["Close"].values
    X = np.arange(len(prices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X[:-30], prices[:-30])
    preds = model.predict(X[-30:])

    accuracy = 100 - mean_absolute_percentage_error(prices[-30:], preds) * 100

    st.metric("📌 Model Accuracy", f"{accuracy:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prices[-30:], name="Actual"))
    fig.add_trace(go.Scatter(y=preds, name="Predicted"))
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 🔍 How it works
    - Uses historical price data
    - Applies Linear Regression
    - Forecasts short-term trends
    - Provides decision support (BUY / HOLD / SELL)

    ⚠️ This app is for **learning & analysis**, not financial advice.
    """)



