#include "led_manager.h"

#include "config.h"

namespace
{
  constexpr unsigned long CONNECTED_TOGGLE_INTERVAL_MS = 500;
  constexpr unsigned long DISCONNECTED_TOGGLE_INTERVAL_MS = 100;
} // namespace

void LedManager::begin()
{
  pinMode(config::STATUS_LED_PIN, OUTPUT);
  digitalWrite(config::STATUS_LED_PIN, LOW);
  lastToggleMs_ = millis();
  ledState_ = false;
  initialized_ = true;
}

void LedManager::update(unsigned long nowMs, bool wifiConnected)
{
  if (!initialized_)
  {
    return;
  }

  if (nowMs - lastToggleMs_ < blinkIntervalMs_(wifiConnected))
  {
    return;
  }

  lastToggleMs_ = nowMs;
  ledState_ = !ledState_;
  digitalWrite(config::STATUS_LED_PIN, ledState_ ? HIGH : LOW);
}

unsigned long LedManager::blinkIntervalMs_(bool wifiConnected) const
{
  return wifiConnected ? CONNECTED_TOGGLE_INTERVAL_MS : DISCONNECTED_TOGGLE_INTERVAL_MS;
}
