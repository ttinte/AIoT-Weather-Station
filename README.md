## 📁 Project Structure

```text
AIOT-WEATHER/
│
├── ai_server/
│   ├── server.py              # Flask API server
│   ├── model.h5               # Trained LSTM model (ignored)
│   ├── scaler.pkl             # Data scaler (ignored)
│   └── requirements.txt       # Python dependencies
│
├── WeatherPredictTraining/
│   ├── 01_check_timeseries.py # Data analysis & visualization
│   ├── train_lstm_model.py    # Train LSTM model
│   ├── weather_fixed_pro.csv  # Dataset
│   ├── weather_lstm_model.h5  # Trained model (ignored)
│   ├── weather_scaler.pkl     # Scaler (ignored)
│   └── Predict_plot.png       # Prediction result (ignored)
│
├── web_dashboard/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── firebase.js
│   ├── map.js
│   ├── lang.js
│   └── logo.png
│
├── esp32_node/
│   └── main.ino               # ESP32 firmware
│
├── .gitignore
└── README.md
