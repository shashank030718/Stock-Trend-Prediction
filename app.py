import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("stock_dl_model.h5")

model = load_lstm_model()

# App title
st.title("📈 Stock Price Prediction App")

# Sidebar input
stock = st.text_input("Enter Stock Ticker", value="POWERGRID.NS")

# Date range
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2025, 5, 22)

# Load stock data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start, end=end)
    return df

df = load_data(stock)

if df.empty:
    st.error("No data found. Please enter a valid ticker.")
    st.stop()

st.subheader("Raw Data")
st.dataframe(df.tail())

# EMA calculations
ema20 = df['Close'].ewm(span=20, adjust=False).mean()
ema50 = df['Close'].ewm(span=50, adjust=False).mean()
ema100 = df['Close'].ewm(span=100, adjust=False).mean()
ema200 = df['Close'].ewm(span=200, adjust=False).mean()

# Train-test split
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare input for model
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Prediction
y_predicted = model.predict(x_test)

# Inverse scaling
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot 1: EMA 20 & 50
st.subheader("📊 Closing Price vs Time (20 & 50 Days EMA)")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(df['Close'], label='Closing Price')
ax1.plot(ema20, label='EMA 20')
ax1.plot(ema50, label='EMA 50')
ax1.set_xlabel("Time")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# Plot 2: EMA 100 & 200
st.subheader("📊 Closing Price vs Time (100 & 200 Days EMA)")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df['Close'], label='Closing Price')
ax2.plot(ema100, label='EMA 100')
ax2.plot(ema200, label='EMA 200')
ax2.set_xlabel("Time")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# Plot 3: Predicted vs Actual
st.subheader("🔮 Prediction vs Actual Prices")
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(y_test, label='Actual Price')
ax3.plot(y_predicted, label='Predicted Price')
ax3.set_xlabel("Time")
ax3.set_ylabel("Price")
ax3.legend()
st.pyplot(fig3)

# Summary statistics
st.subheader("📋 Descriptive Statistics")
st.dataframe(df.describe())

# CSV Download
st.subheader("⬇️ Download Stock Dataset")
csv = df.to_csv(index=True).encode()
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=f'{stock}_dataset.csv',
    mime='text/csv',
)


