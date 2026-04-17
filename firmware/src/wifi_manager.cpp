#include "wifi_manager.h"

#include <WiFi.h>
#include <time.h>

#include "config.h"

namespace
{
  constexpr unsigned long NTP_RETRY_INTERVAL_MS = 30000;
  constexpr time_t MIN_VALID_UNIX_TIME = 1700000000;
} // namespace

void WiFiManager::begin()
{
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  // Let the network stack run immediately after starting station mode.
  yield();
  lastConnectAttemptMs_ = millis();
  disconnectedSinceMs_ = lastConnectAttemptMs_;
  wasConnected_ = false;
  justConnected_ = false;
  justDisconnected_ = false;
  hasEverConnected_ = false;
}

void WiFiManager::update(unsigned long nowMs)
{
  const bool connected = WiFi.status() == WL_CONNECTED;

  justConnected_ = false;
  justDisconnected_ = false;

  if (connected && !wasConnected_)
  {
    justConnected_ = true;
    lastNtpAttemptMs_ = 0;
    disconnectedSinceMs_ = 0;
    hasEverConnected_ = true;
    const IPAddress ip = WiFi.localIP();
    Serial.printf("WiFi connected. IP: %u.%u.%u.%u\n", ip[0], ip[1], ip[2], ip[3]);
  }
  else if (!connected && wasConnected_)
  {
    justDisconnected_ = true;
    disconnectedSinceMs_ = nowMs;
    Serial.println("WiFi disconnected");
  }

  wasConnected_ = connected;

  if (!connected)
  {
    if (nowMs - lastConnectAttemptMs_ >= config::WIFI_RETRY_INTERVAL_MS)
    {
      WiFi.disconnect(false, false);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      // Yield after the reconnect request so long-lived runs stay watchdog-friendly.
      yield();
      lastConnectAttemptMs_ = nowMs;
      Serial.println("WiFi reconnect attempt");
    }
    return;
  }

  if (!isTimeValid_() && (lastNtpAttemptMs_ == 0 || nowMs - lastNtpAttemptMs_ >= NTP_RETRY_INTERVAL_MS))
  {
    configTime(0, 0, "pool.ntp.org", "time.nist.gov", "time.google.com");
    lastNtpAttemptMs_ = nowMs;
    Serial.println("NTP sync requested");
  }
}

bool WiFiManager::isConnected() const
{
  return WiFi.status() == WL_CONNECTED;
}

bool WiFiManager::timeReady() const
{
  return isTimeValid_();
}

bool WiFiManager::justConnected() const
{
  return justConnected_;
}

bool WiFiManager::justDisconnected() const
{
  return justDisconnected_;
}

bool WiFiManager::hasEverConnected() const
{
  return hasEverConnected_;
}

unsigned long WiFiManager::disconnectedDurationMs(unsigned long nowMs) const
{
  if (WiFi.status() == WL_CONNECTED || disconnectedSinceMs_ == 0)
  {
    return 0;
  }

  return nowMs - disconnectedSinceMs_;
}

void WiFiManager::forceReconnect()
{
  WiFi.disconnect(false, false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  yield();
  lastConnectAttemptMs_ = millis();
  if (disconnectedSinceMs_ == 0)
  {
    disconnectedSinceMs_ = lastConnectAttemptMs_;
  }
  Serial.println("WiFi forced reconnect");
}

uint64_t WiFiManager::currentUnixTime() const
{
  const time_t now = time(nullptr);
  if (now >= MIN_VALID_UNIX_TIME)
  {
    return static_cast<uint64_t>(now);
  }

  return 0ULL;
}

bool WiFiManager::isTimeValid_() const
{
  return time(nullptr) >= MIN_VALID_UNIX_TIME;
}
