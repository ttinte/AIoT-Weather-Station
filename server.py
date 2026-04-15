from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import os

import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# 1. Load model AI
model = tf.keras.models.load_model("weather_lstm_model.h5")

# 2. Lấy API KEY từ biến môi trường của Render
API_KEY = os.environ.get("MY_API_KEY")

# 3. Khởi tạo kết nối Firebase
# Đảm bảo bạn đã dán nội dung Service Account vào Secret Files trên Render với tên firebase_key.json
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aiotnhom2-default-rtdb.firebaseio.com/'
})

@app.route("/")
def home():
    return "Server AI dự báo thời tiết đang hoạt động!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Lấy 10 bản ghi cuối cùng từ nhánh lịch sử
        ref = db.reference("weather_stations/Weather_station_1/history")
        data_dict = ref.order_by_key().limit_to_last(10).get()

        if not data_dict or len(data_dict) < 10:
             return jsonify({"error": "Chưa đủ 10 bản ghi dữ liệu lịch sử để dự báo"}), 400

        # Trích xuất dữ liệu thành mảng
        sequence = []
        for key, val in data_dict.items():
            temp = float(val.get("temperature", 0.0))
            hum = float(val.get("humidity", 0.0))
            press = float(val.get("pressure", 0.0))
            rain = float(val.get("rain", 0.0)) # Bổ sung thêm biến thứ 4
            sequence.append([temp, hum, press, rain])

        # Chuyển thành Numpy array với kích thước (1, 10, 4)
        x = np.array(sequence).reshape(1, 10, 4)

        # Chạy dự đoán
        pred = model.predict(x).tolist()

        return jsonify({
            "status": "success",
            "prediction": pred
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Tắt debug mode khi chạy trên Render để tránh rò rỉ bảo mật
    app.run(host="0.0.0.0", port=port, debug=False)