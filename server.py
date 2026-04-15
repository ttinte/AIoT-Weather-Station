from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model("weather_lstm_model.h5")

@app.route("/")
def home():
    return "Server is running!"

@app.route("/predict")
def predict():
    data = [30, 80, 1012]
    x = np.array(data).reshape(1, -1)
    pred = model.predict(x).tolist()
    return jsonify(pred)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # QUAN TRỌNG
    app.run(host="0.0.0.0", port=port)
