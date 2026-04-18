## 📁 Project Structure

```text
AIoT-WEATHER-Station/
│
├── ai_server/
│   ├── server.py              # Flask API server
│   ├── model.h5               # Trained LSTM model
│   ├── scaler.pkl             # Data scaler
│   └── requirements.txt       # Python dependencies
│
├── WeatherPredictTraining/
│   ├── 01_check_timeseries.py # Missing value and outliers detection
│   ├── train_lstm_model.py    # Train model
│   ├── weather_fixed_pro.csv  # Dataset
│   ├── weather_lstm_model.h5  # Trained LSTM model (ignored)
│   ├── weather_scaler.pkl     # Data Scaler (ignored)
│   └── Predict_plot.png       # Prediction result
│
├── web_dashboard/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── firebase.js
│   ├── map.js
│   ├── lang.js
│   └── logo.png
│   └── dashboard.html
│
├── firmware/                  # ESP32 firmware (PlatformIO)
│   ├── include/               # Header files (.h)
│   ├── lib/                   # External libraries
│   ├── src/                   # Main source code (.cpp / .ino)
│   ├── test/                  # Unit tests
│   ├── .gitignore
│   └── platformio.ini         # PlatformIO config
│
├── .gitignore
└── README.md
```
