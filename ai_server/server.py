from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import time
import threading
import joblib

import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)
CORS(app)

# 1. Load model AI
model = tf.keras.models.load_model("weather_lstm_model.h5")

# 2. Load Scaler
scaler = joblib.load("weather_scaler.pkl")

# 3. Load API KEY From Render Environment Variables
API_KEY = os.environ.get("MY_API_KEY")

# 4. Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aiotnhom2-80e7a-default-rtdb.firebaseio.com/'
})

WINDOW_SIZE = model.input_shape[1] or 36

SAMPLE_INTERVAL_MINUTES = 10
FORECAST_HORIZON_STEPS = 2
HORIZON_MINUTES = SAMPLE_INTERVAL_MINUTES * FORECAST_HORIZON_STEPS

STATION_PATH = "weather_stations/Weather_station_1"

_predict_lock = threading.Lock()
_listener_registration = None
_last_processed_source_timestamp = None


def run_prediction():
    ref = db.reference(f"{STATION_PATH}/readings")
    data_dict = ref.order_by_key().limit_to_last(2).get()

    if not data_dict:
        return None

    sequence = []
    for date_key in sorted(data_dict.keys()):
        day_data = data_dict[date_key]
        if not isinstance(day_data, dict):
            continue
        for time_key in sorted(day_data.keys()):
            val = day_data[time_key]
            try:
                sequence.append([
                    float(val.get("temperature", 0.0)),
                    float(val.get("humidity", 0.0)),
                    float(val.get("pressure", 0.0)),
                    float(val.get("rain", 0.0)),
                ])
            except AttributeError:
                continue

    if len(sequence) < WINDOW_SIZE:
        return None

    with _predict_lock:
        scaled_input = scaler.transform(sequence[-WINDOW_SIZE:])
        x = np.array(scaled_input).reshape(1, WINDOW_SIZE, 4)
        scaled_pred = model.predict(x)
        final = scaler.inverse_transform(scaled_pred)[0]

    return {
        "temperature": round(float(final[0]), 2),
        "humidity": round(float(final[1]), 2),
        "pressure": round(float(final[2]), 2),
        "rain": 1.0 if final[3] > 0.5 else 0.0,
    }


def save_forecast(result, source_sec):
    target_sec = source_sec + HORIZON_MINUTES * 60
    target_ms = target_sec * 1000
    date_str = time.strftime("%Y-%m-%d", time.gmtime(target_sec))
    path = f"{STATION_PATH}/forecast/{date_str}/{target_ms}"

    ref = db.reference(path)
    if ref.get() is not None:
        return None

    record = dict(result)
    record["timestamp"] = target_sec
    record["source_timestamp"] = source_sec
    record["horizon_minutes"] = HORIZON_MINUTES
    record["created_at"] = int(time.time())
    ref.set(record)
    return path


def on_latest_update(event):
    global _last_processed_source_timestamp

    data = event.data
    if not isinstance(data, dict):
        return

    raw_ts = data.get("timestamp")
    if raw_ts is None:
        return
    raw_ts = int(raw_ts)
    source_sec = raw_ts // 1000 if raw_ts >= 1_000_000_000_000 else raw_ts

    if source_sec == _last_processed_source_timestamp:
        return
    _last_processed_source_timestamp = source_sec

    try:
        result = run_prediction()
        if result is None:
            print(f"[auto-predict] Chưa đủ {WINDOW_SIZE} bản ghi, bỏ qua.")
            return
        path = save_forecast(result, source_sec)
        if path:
            print(f"[auto-predict] Đã ghi forecast tại {path}: {result}")
        else:
            print(f"[auto-predict] Forecast cho source={source_sec} đã tồn tại, bỏ qua.")
    except Exception as e:
        print(f"[auto-predict] Lỗi: {e}")


def start_listener():
    global _listener_registration
    _listener_registration = db.reference(f"{STATION_PATH}/latest").listen(on_latest_update)
    print(f"[auto-predict] Đang lắng nghe 'latest' | WINDOW_SIZE={WINDOW_SIZE} | HORIZON={HORIZON_MINUTES} phút")


@app.route("/")
def home():
    return "Server AI dự báo thời tiết đang hoạt động!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    client_key = request.headers.get("x-api-key")

    # Debug Log:
    print(f"Key server đang có: '{API_KEY}'")
    print(f"Key Postman gửi lên: '{client_key}'")

    # Check API Key
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        result = run_prediction()
        if result is None:
            return jsonify({"error": f"Không đủ dữ liệu để dự báo (cần ít nhất {WINDOW_SIZE} bản ghi)."}), 400

        return jsonify({
            "status": "success",
            "prediction": [[result["temperature"], result["humidity"],
                            result["pressure"], result["rain"]]],
            "forecast": result
        })

    except Exception as e:
        # Bắt lỗi hệ thống để server không bị sập
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    start_listener()
    # Tắt debug mode khi chạy trên Render
    app.run(host="0.0.0.0", port=port, debug=False)
