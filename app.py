import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🚀 Crypto Forecast & Decision",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- APP BACKGROUND -----------------
page_bg_img = """
<style>
body {
background-color: #f0f2f6;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------- APP SIDEBAR -----------------
st.sidebar.title("Crypto Forecast System")
page = st.sidebar.radio("Navigate", ["Home", "Forecast", "Trading Decision", "App Info"])

# ----------------- GLOBAL VARIABLES -----------------
CURRENCY = "usd"
HIST_DAYS = 120
FORECAST_DAYS = 30
LOOKBACK = 14
EPOCHS = 25

# ----------------- HELPER FUNCTIONS -----------------
def get_top_coins(limit=10):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    data = requests.get(url, params=params).json()
    return {coin["id"]: coin["symbol"].upper() for coin in data}

COINS = get_top_coins(limit=10)

def fetch_crypto_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": CURRENCY, "days": HIST_DAYS}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["date", "price"]]
    df.set_index("date", inplace=True)
    return df

def train_and_forecast(df):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(df[["price"]])
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_prices)):
        X.append(scaled_prices[i-LOOKBACK:i])
        y.append(scaled_prices[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=EPOCHS, batch_size=16, verbose=0)

    last_seq = scaled_prices[-LOOKBACK:]
    future_scaled = []
    for _ in range(FORECAST_DAYS):
        pred = model.predict(last_seq.reshape(1, LOOKBACK, 1), verbose=0)
        future_scaled.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    return future_prices

# ----------------- HOME PAGE -----------------
if page == "Home":
    st.title("🚀 Crypto Price Forecast & Decision System")
    st.markdown("""
    Welcome to the Crypto Forecast & Decision System! 💹

    - **Forecast upcoming crypto prices** for top coins.
    - **Visualize trends** with clear charts.
    - **Get trading suggestions** based on predicted profit/loss.
    """)

# ----------------- FORECAST PAGE -----------------
elif page == "Forecast":
    st.title("📊 Price Forecast")
    coin_choice = st.selectbox("Select a cryptocurrency:", list(COINS.keys()), format_func=lambda x: COINS[x])
    df = fetch_crypto_data(coin_choice)
    st.subheader(f"📈 Historical Prices - Last {HIST_DAYS} Days")
    st.line_chart(df["price"])

    future_prices = train_and_forecast(df)
    future_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    future_df = pd.DataFrame({"Date": future_dates, "Forecasted Price": future_prices})
    
    st.subheader(f"🔮 Forecast - Next {FORECAST_DAYS} Days")
    st.line_chart(future_df.set_index("Date"))

# ----------------- TRADING DECISION PAGE -----------------
elif page == "Trading Decision":
    st.title("🧠 Trading Decision")

    coin_choice = st.selectbox("Select a cryptocurrency for analysis:", list(COINS.keys()), format_func=lambda x: COINS[x])
    df = fetch_crypto_data(coin_choice)
    hist = df[df.index <= pd.Timestamp.today()]
    future_prices = train_and_forecast(hist)
    future_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    future_df = pd.DataFrame({"date": future_dates, "predicted_price": future_prices})
    
    # Moving averages for decision
    short_ma = hist["price"].rolling(5).mean()
    long_ma = hist["price"].rolling(20).mean()
    # Compare last values only
    decision_ma = "BUY" if short_ma.iloc[-1] > long_ma.iloc[-1] else "SELL"

    buy_price = hist["price"].iloc[-1]
    future_df["profit_pct"] = (future_df["predicted_price"] - buy_price) / buy_price * 100
    worst_day = future_df.loc[future_df["profit_pct"].idxmin()]
    profit_days = future_df[future_df["profit_pct"] > 0]
    best_day = profit_days.iloc[0] if not profit_days.empty else None
    final_price = future_prices[-1]
    final_profit = (final_price - buy_price) / buy_price * 100

    # ----------------- OUTPUT -----------------
    st.markdown("### 🚀 Trade Decision Summary")
    st.markdown(f"💰 **Buy Price (Today):** ${buy_price:.4f}")
    st.markdown(f"📉 **Worst Sell Date:** {worst_day['date'].date()} → ${worst_day['predicted_price']:.4f} ({worst_day['profit_pct']:.2f}% loss)")
    
    if best_day is not None:
        st.markdown(f"📈 **Best Buy Date:** {best_day['date'].date()} → ${best_day['predicted_price']:.4f} (+{best_day['profit_pct']:.2f}%)")
        st.markdown("🔄 **Strategy:** WAIT → BUY on best date → SELL later")
    else:
        st.markdown("❌ **No profitable date predicted**")
        st.markdown("📉 **Strategy:** SELL now or HOLD")

    st.markdown(f"🔮 **Predicted Price on Last Forecast Day:** ${final_price:.4f}")
    st.markdown(f"💹 **Expected Return:** {final_profit:.2f}%")
    st.markdown(f"🧠 **Moving Average Strategy:** {decision_ma}")

# ----------------- APP INFO PAGE -----------------
elif page == "App Info":
    st.title("📌 App Information")
    st.markdown("""
    This app is a **real-time cryptocurrency forecasting & trading assistant**.

    **Features**:
    - Forecasts crypto prices for the next 30 days using **LSTM neural networks**.
    - Provides **profit/loss predictions** and **trading suggestions**.
    - Calculates **moving average strategy** for better trade timing.
    - Visualizes **historical trends** and **future forecasts** with interactive charts.
    """)

    st.subheader("📊 Forecast Accuracy Example")

    # Use first coin as example for accuracy demonstration
    coin_choice = list(COINS.keys())[0]
    df = fetch_crypto_data(coin_choice)
    hist = df[df.index <= pd.Timestamp.today()]

    # Split last 30 days for test
    test_days = 30
    test_hist = hist[-(LOOKBACK + test_days):]  # need LOOKBACK for input
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(test_hist[["price"]])

    # Prepare sequences for testing
    X_test, y_test = [], []
    for i in range(LOOKBACK, len(scaled_prices)):
        X_test.append(scaled_prices[i-LOOKBACK:i])
        y_test.append(scaled_prices[i])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Train model on earlier data
    train_hist = hist[:-test_days]
    future_pred = train_and_forecast(train_hist)[:test_days]  # predict next 30 days
    y_true = test_hist["price"].values[LOOKBACK:]

    # Calculate MAPE and accuracy
    mape = np.mean(np.abs((y_true - future_pred) / y_true)) * 100
    accuracy = 100 - mape

    st.markdown(f"✅ **Model Forecast Accuracy:** {accuracy:.2f}% (based on last 30 days)")

    # Plot chart
    st.subheader("📈 Actual vs Predicted Prices (Last 30 Days)")
    chart_df = pd.DataFrame({
        "Date": test_hist.index[LOOKBACK:],
        "Actual Price": y_true,
        "Predicted Price": future_pred
    })
    st.line_chart(chart_df.set_index("Date"))

    st.subheader("💡 How to Use")
    st.markdown("""
    1. Navigate to **Forecast** to view predicted crypto prices.
    2. Navigate to **Trading Decision** to get a buy/sell strategy.
    3. Use the charts to **visually inspect trends** and forecast.
    4. Make your trading decisions based on **expected returns** and **best/worst dates**.
    """)

    st.subheader("✨ Additional Details")
    st.markdown("""
    - Supports top 10 cryptocurrencies by market cap.
    - Uses **MinMax scaling** and **LSTM neural networks** for prediction.
    - Interactive and user-friendly interface with **graphs and summaries**.
    """)


