#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H

#include <Arduino.h>

using TimestampProvider = uint64_t (*)();

struct SensorReading
{
  uint64_t timestamp;
  float temperatureC;
  float humidityPct;
  float pressureHpa;
  bool isRaining;
};

class SensorManager
{
public:
  bool begin();
  bool sample(SensorReading &outReading);
  bool pressureAvailable() const;
  void setTimestampProvider(TimestampProvider provider);

private:
  bool bmpReady_ = false;
  TimestampProvider timestampProvider_ = nullptr;
};

#endif // SENSOR_MANAGER_H
