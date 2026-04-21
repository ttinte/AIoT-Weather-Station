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
│   ├── raw/                   # Raw JSON data
│   │
│   ├── 00_exportcsv.py        # Convert JSON → CSV
│   ├── 01_check_timeseries.py # Detect missing data & outliers
│   ├── 02_fixdata.py          # Seasonal imputation
│   ├── 03_train_lstm_model.py # Train LSTM model
│   ├── 04_collect_data.py     # Weather data from Meteo API
│   ├── 05_compare_data.py     # Compare IoT sensor data with Meteo data
│   │
│   ├── weather_fixed_pro.csv  
│   ├── comparison_result.csv  
│   ├── Predict_plot.png       
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
│   ├── lib/                   # External libraries
│   ├── src/                   # Main source code (.cpp / .ino)
│   ├── test/                  # Unit tests
│   ├── .gitignore
│   └── platformio.ini         # PlatformIO config
│
├── .gitignore
└── README.md
```
