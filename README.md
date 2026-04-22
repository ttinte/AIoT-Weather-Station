## 📁 Project Structure

```text
AIoT-WEATHER-Station/
│
├── ai_server/
│   ├── server.py              # Flask API server
│   ├── model.h5               # Trained LSTM model (from WeatherPredictTraining)
│   ├── scaler.pkl             # Data scaler (from WeatherPredictTraining)
│   └── requirements.txt       # Python dependencies
│
├── WeatherPredictTraining/
│   ├── raw/                   # Raw JSON data
│   │
│   ├── 00_exportcsv.py        # Convert JSON → CSV
│   ├── 01_check_timeseries.py # Detect missing data & outliers
│   ├── 02_fixdata.py          # Seasonal imputation
│   ├── 03_train_lstm_model.py # Train LSTM model
│   ├── 04_collect_data.py     # Weather data from Meteo API
│   ├── 05_compare_data.py     # Compare IoT sensor data with Meteo data
│   │
│   └── Predict_plot.png       
│
├── web_dashboard/
│   ├── index.html             # Login Page
│   ├── dashboard.html         # Dashboard
│   │
│   ├── config.js              # ⚠️ Firebase/API config (ignored)
│   ├── firebase.js            # Connect Firebase
│   ├── script.js              # Logic dashboard + AI call
│   ├── map.js
│   ├── lang.js
│   │
│   ├── style.css
│   └── logo.png
│
├── firmware/                  # ESP32 firmware (PlatformIO)
│   ├── include/               # Header files (.h)
│   ├── src/                   # Main source code (.cpp)
│   └── platformio.ini         # PlatformIO config
│
├── .gitignore
└── README.md
```
