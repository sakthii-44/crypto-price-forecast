import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crypto Price Forecast", layout="centered")

st.title("📈 Crypto Price Forecasting using LSTM")
st.write("Forecast next 30 days of crypto prices")

# ---------------- USER INPUT ----------------
crypto = st.selectbox(
    "Select Cryptocurrency",
    ["bitcoin", "ethereum", "binancecoin", "ripple", "cardano"]
)

days_to_forecast = st.slider("Forecast days", 7, 60, 30)

# ---------------- FETCH DATA ----------------
@st.cache_data
def fetch_crypto_data(coin, days=180):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)
    data = response.json()
    prices = [price[1] for price in data["prices"]]
    return pd.DataFrame(prices, columns=["Close"])

df = fetch_crypto_data(crypto)

st.subheader("📊 Historical Price Data")
st.line_chart(df)

# ---------------- PREPROCESS ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

LOOKBACK = 14
X, y = [], []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i - LOOKBACK:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ---------------- MODEL ----------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

with st.spinner("🔄 Training LSTM model..."):
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

st.success("✅ Model trained successfully!")

# ---------------- FORECAST ----------------
last_sequence = scaled_data[-LOOKBACK:]
future_predictions = []

for _ in range(days_to_forecast):
    pred = model.predict(last_sequence.reshape(1, LOOKBACK, 1), verbose=0)
    future_predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred, axis=0)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# ---------------- PLOT ----------------
st.subheader("🔮 Price Forecast")

plt.figure(figsize=(10, 4))
plt.plot(df.values, label="Historical Price")
plt.plot(
    range(len(df), len(df) + days_to_forecast),
    future_predictions,
    label="Forecast",
    linestyle="dashed"
)
plt.legend()
st.pyplot(plt)

# ---------------- TABLE ----------------
forecast_df = pd.DataFrame(
    future_predictions,
    columns=["Predicted Price (USD)"]
)

st.subheader("📅 Forecasted Prices")
st.dataframe(forecast_df)
