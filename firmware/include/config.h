#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

#ifndef WIFI_SSID
#define WIFI_SSID "QuocQuang"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "21112005"
#endif

// Station ID selects the Firebase namespace for this physical device.
#ifndef STATION_ID
#define STATION_ID "Weather_station_1"
#endif

// Keep DEVICE_ID aligned by default so existing code remains compatible.
#ifndef DEVICE_ID
#define DEVICE_ID STATION_ID
#endif

// Station metadata is stored once under /weather_stations/{STATION_ID}/info.
#ifndef STATION_LOCATION
#define STATION_LOCATION "HCM City, Vietnam"
#endif

#ifndef API_KEY
#define API_KEY "YOUR_API_KEY"
#endif

#ifndef DATABASE_URL
#define DATABASE_URL "https://aiotnhom2-80e7a-default-rtdb.firebaseio.com"
#endif

#ifndef DHT_GPIO
#define DHT_GPIO 16
#endif

#ifndef RAIN_SENSOR_GPIO
#define RAIN_SENSOR_GPIO 34
#endif

#ifndef STATUS_LED_GPIO
#define STATUS_LED_GPIO 2
#endif

namespace config
{
  // Avoid ESP32 boot-strap pins for sensors by default.
  constexpr uint8_t DHT_PIN = DHT_GPIO;
  // DHT22 sensor type ID used by the DHT library.
  constexpr uint8_t DHT_TYPE = 22;
  // GPIO34 is input-only and requires an external pull-up/down resistor.
  constexpr uint8_t RAIN_SENSOR_PIN = RAIN_SENSOR_GPIO;
  constexpr uint8_t STATUS_LED_PIN = STATUS_LED_GPIO;

  // Centralize Firebase path fragments so uploader.cpp can build paths consistently.
  constexpr char FIREBASE_ROOT_PATH[] = "/weather_stations";
  constexpr char FIREBASE_INFO_SUFFIX[] = "/info";
  constexpr char FIREBASE_LATEST_SUFFIX[] = "/latest";
  constexpr char FIREBASE_READINGS_SUFFIX[] = "/readings/";

  constexpr unsigned long UPLOAD_INTERVAL_MS = 600000;
  constexpr unsigned long WIFI_RETRY_INTERVAL_MS = 5000;
  constexpr unsigned long FIREBASE_RETRY_INTERVAL_MS = 5000;
  constexpr unsigned long FIREBASE_INIT_RETRY_MAX_MS = 60000;
  constexpr unsigned long FIREBASE_UPLOAD_RETRY_INITIAL_MS = 2000;
  constexpr unsigned long FIREBASE_UPLOAD_RETRY_MAX_MS = 60000;
  constexpr unsigned long FIREBASE_AUTH_LOG_INTERVAL_MS = 15000;
  constexpr unsigned long HEALTH_LOG_INTERVAL_MS = 60000;
  constexpr unsigned long UPLOADER_TASK_INTERVAL_MS = 250;
  constexpr unsigned long QUEUE_OVERFLOW_LOG_INTERVAL_MS = 30000;
  constexpr unsigned long WIFI_RECOVERY_RESTART_MS = 15UL * 60UL * 1000UL;
  constexpr unsigned long UPLOAD_STALL_RESTART_MS = 30UL * 60UL * 1000UL;
  constexpr unsigned long RECOVERY_ACTION_COOLDOWN_MS = 60000;
  constexpr unsigned long RESTART_MIN_INTERVAL_MS = 60UL * 60UL * 1000UL;
} // namespace config

#endif // CONFIG_H
