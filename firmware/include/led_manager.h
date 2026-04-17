#ifndef LED_MANAGER_H
#define LED_MANAGER_H

#include <Arduino.h>

class LedManager
{
public:
  void begin();
  void update(unsigned long nowMs, bool wifiConnected);

private:
  unsigned long blinkIntervalMs_(bool wifiConnected) const;

  unsigned long lastToggleMs_ = 0;
  bool ledState_ = false;
  bool initialized_ = false;
};

#endif // LED_MANAGER_H
