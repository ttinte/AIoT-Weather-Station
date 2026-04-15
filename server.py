from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import os

import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("weather_lstm_model.h5")

# API KEY
API_KEY = os.environ.get("MY_API_KEY")

# Firebase init
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aiotnhom2-default-rtdb.firebaseio.com'
})

@app.route("/")
def home():
    return "Server is running!"

@app.route("/predict")
def predict():
    # 🔐 check API key
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    # 🔥 lấy data từ Firebase
    ref = db.reference("weather_stations/Weather_station_1/latest")
    data = ref.get()

    # parse dữ liệu
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    pressure = data.get("pressure")

    # ⚠️ đúng thứ tự input model của bạn
    x = np.array([temperature, humidity, pressure]).reshape(1, -1)

    pred = model.predict(x).tolist()

    return jsonify({
        "input": data,
        "prediction": pred
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)