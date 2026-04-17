#include "sensor_manager.h"

#include <DHT.h>
#include <Adafruit_BMP280.h>
#include <Wire.h>

#include "config.h"

namespace
{
  // Common rain sensor modules drive the digital pin low when water is detected.
  constexpr int RAIN_DETECTED_STATE = LOW;
  constexpr uint8_t RAIN_SAMPLE_COUNT = 5;
  constexpr uint8_t RAIN_SAMPLE_THRESHOLD = 3;

  DHT g_dht(config::DHT_PIN, config::DHT_TYPE);
  Adafruit_BMP280 g_bmp;
} // namespace

bool SensorManager::begin()
{
  g_dht.begin();
  bmpReady_ = g_bmp.begin(0x76);
  if (!bmpReady_)
  {
    bmpReady_ = g_bmp.begin(0x77);
  }

  // GPIO34 is input-only on ESP32 and has no internal pull-up/down support.
  // Use an external resistor on the rain sensor signal line.
  pinMode(config::RAIN_SENSOR_PIN, INPUT);

  if (!bmpReady_)
  {
    Serial.println("Warning: BMP280 not detected, uploads will be skipped until the sensor responds");
  }

  return true;
}

bool SensorManager::pressureAvailable() const
{
  return bmpReady_;
}

void SensorManager::setTimestampProvider(TimestampProvider provider)
{
  timestampProvider_ = provider;
}

bool SensorManager::sample(SensorReading &outReading)
{
  const float humidity = g_dht.readHumidity();
  const float temperature = g_dht.readTemperature();

  if (isnan(humidity) || isnan(temperature))
  {
    Serial.println("DHT read failed: temperature or humidity is NaN");
    return false;
  }

  if (humidity < 0.0f || humidity > 100.0f)
  {
    Serial.printf("DHT read failed: humidity out of range %.2f%%\n", humidity);
    return false;
  }

  if (!bmpReady_)
  {
    Serial.println("BMP280 read failed: sensor not initialized");
    return false;
  }

  const float pressurePa = g_bmp.readPressure();
  if (isnan(pressurePa) || pressurePa <= 0.0f)
  {
    Serial.printf("BMP280 read failed: invalid pressure %.2f Pa\n", pressurePa);
    return false;
  }

  const float pressureHpa = pressurePa / 100.0f;

  uint8_t rainHits = 0;
  for (uint8_t i = 0; i < RAIN_SAMPLE_COUNT; ++i)
  {
    if (digitalRead(config::RAIN_SENSOR_PIN) == RAIN_DETECTED_STATE)
    {
      rainHits++;
    }
  }
  const bool isRaining = rainHits >= RAIN_SAMPLE_THRESHOLD;

  outReading.timestamp = timestampProvider_ != nullptr ? timestampProvider_() : 0ULL;
  outReading.temperatureC = temperature;
  outReading.humidityPct = humidity;
  outReading.pressureHpa = pressureHpa;
  outReading.isRaining = isRaining;

  return true;
}
