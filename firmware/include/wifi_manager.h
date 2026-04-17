#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <Arduino.h>

class WiFiManager
{
public:
  void begin();
  void update(unsigned long nowMs);
  bool isConnected() const;
  bool timeReady() const;
  uint64_t currentUnixTime() const;
  bool justConnected() const;
  bool justDisconnected() const;
  bool hasEverConnected() const;
  unsigned long disconnectedDurationMs(unsigned long nowMs) const;
  void forceReconnect();

private:
  bool isTimeValid_() const;

  unsigned long lastConnectAttemptMs_ = 0;
  unsigned long lastNtpAttemptMs_ = 0;
  unsigned long disconnectedSinceMs_ = 0;
  bool wasConnected_ = false;
  bool justConnected_ = false;
  bool justDisconnected_ = false;
  bool hasEverConnected_ = false;
};

#endif // WIFI_MANAGER_H
