from flask import Flask, jsonify, request, render_template_string
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
import requests
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

API_KEY = "Y96NS3HCGNHAWTXV"

# Function to fetch real-time stock data
def get_stock_data(symbol, interval="5min"):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": "full"
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Debugging: Print API response
    print("API Response:", data)

    if f"Time Series ({interval})" in data:
        df = pd.DataFrame.from_dict(data[f"Time Series ({interval})"], orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df
    else:
        return None  # API error or invalid stock symbol

# ARIMA Model Prediction
def predict_arima(symbol):
    df = get_stock_data(symbol)
    if df is None or df.empty:
        return "Error fetching data"

    df_arima = df[["Close"]].copy()
    df_arima.index = pd.to_datetime(df_arima.index)
    model = auto_arima(df_arima, seasonal=False, stepwise=True)
    forecast = model.predict(n_periods=10)  # Predict next 10 intervals
    return forecast.tolist()

# LSTM Model Prediction
def predict_lstm(symbol):
    df = get_stock_data(symbol)
    if df is None or df.empty:
        return "Error fetching data"

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["Close"]])

    def create_sequences(data, seq_length=50):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
        return np.array(X)

    seq_length = 50
    X = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X[:-10], scaled_data[seq_length:-10], epochs=10, batch_size=32)

    future_data = X[-1].reshape((1, seq_length, 1))
    predicted_price = model.predict(future_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    return float(predicted_price[0][0])

# ---------------------- Flask Routes ----------------------
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Price Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            input { padding: 10px; font-size: 18px; margin-right: 10px; }
            button { padding: 10px 20px; font-size: 18px; background-color: #007BFF; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Welcome to the Stock Price Prediction System</h1>
        <p>Enter a stock symbol to get predictions:</p>
        <form action="/predict" method="get">
            <input type="text" name="symbol" placeholder="Enter Stock Symbol" required>
            <button type="submit">Get Predictions</button>
        </form>
    </body>
    </html>
    """)

@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "").strip().upper()  # Convert input to uppercase
    if not symbol:
        return jsonify({"error": "Stock symbol is required!"})

    arima_pred = predict_arima(symbol)
    lstm_pred = predict_lstm(symbol)

    return jsonify({
        "Stock Symbol": symbol,
        "ARIMA_Predictions": arima_pred,
        "LSTM_Prediction": lstm_pred,
        "direct_link": f"https://stockpriceprediction-2gxw.onrender.com/predict?symbol={symbol}"
    })

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
