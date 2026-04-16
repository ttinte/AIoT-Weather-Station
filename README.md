AIoT-WEATHER-Station/
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
│   ├── index.html             # Main dashboard UI
│   ├── script.js              # Frontend logic
│   ├── style.css              # Styling
│   ├── firebase.js            # Firebase config
│   ├── map.js                 # Map integration
│   ├── lang.js                # Multi-language support
│   └── logo.png               # UI asset
│
├── esp32_node/
│   └── main.ino               # ESP32 firmware (data collection & send to server)
│
├── .gitignore
└── README.md
