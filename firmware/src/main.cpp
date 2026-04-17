#include <Arduino.h>
#include <esp_system.h>
#include <esp_task_wdt.h>
#include <WiFi.h>
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

#include "config.h"
#include "led_manager.h"
#include "sensor_manager.h"
#include "uploader.h"
#include "wifi_manager.h"

namespace
{
  constexpr uint8_t LCD_I2C_ADDRESS = 0x27;
  constexpr uint8_t LCD_COLUMNS = 16;
  constexpr uint8_t LCD_ROWS = 2;
  constexpr unsigned long SENSOR_POLL_INTERVAL_MS = 2500;
  constexpr unsigned long STATUS_LOG_INTERVAL_MS = 10000;
  constexpr uint32_t LOOP_WATCHDOG_TIMEOUT_SECONDS = 10;
  constexpr unsigned long HEAP_LOG_INTERVAL_MS = 60UL * 60UL * 1000UL;

  SensorManager g_sensorManager;
  WiFiManager g_wifiManager;
  LedManager g_ledManager;
  Uploader g_uploader;
  LiquidCrystal_I2C g_lcd(LCD_I2C_ADDRESS, LCD_COLUMNS, LCD_ROWS);
  SensorReading g_latestReading{};
  unsigned long g_lastSensorPollMs = 0;
  unsigned long g_lastUploadMs = 0;
  unsigned long g_lastClockWaitLogMs = 0;
  unsigned long g_lastHealthLogMs = 0;
  unsigned long g_lastHeapLogMs = 0;
  unsigned long g_lastRecoveryActionMs = 0;
  char g_lastLcdLines[LCD_ROWS][LCD_COLUMNS + 1] = {};
  RTC_DATA_ATTR uint32_t g_bootCount = 0;
  RTC_DATA_ATTR uint32_t g_softwareRestartCount = 0;
  bool g_restartEscalationDisabled = false;
  bool g_hasValidReading = false;

  const char *resetReasonToString(esp_reset_reason_t reason)
  {
    switch (reason)
    {
    case ESP_RST_POWERON:
      return "power_on";
    case ESP_RST_EXT:
      return "external";
    case ESP_RST_SW:
      return "software";
    case ESP_RST_PANIC:
      return "panic";
    case ESP_RST_INT_WDT:
      return "int_wdt";
    case ESP_RST_TASK_WDT:
      return "task_wdt";
    case ESP_RST_WDT:
      return "other_wdt";
    case ESP_RST_DEEPSLEEP:
      return "deepsleep";
    case ESP_RST_BROWNOUT:
      return "brownout";
    case ESP_RST_SDIO:
      return "sdio";
    default:
      return "unknown";
    }
  }

  uint64_t currentUnixTimeProvider()
  {
    return g_wifiManager.currentUnixTime();
  }

  void writeLcdLine(uint8_t row, const char *text)
  {
    char paddedLine[LCD_COLUMNS + 1];
    size_t index = 0;
    for (; index < LCD_COLUMNS && text[index] != '\0'; ++index)
    {
      paddedLine[index] = text[index];
    }
    for (; index < LCD_COLUMNS; ++index)
    {
      paddedLine[index] = ' ';
    }
    paddedLine[LCD_COLUMNS] = '\0';

    if (strncmp(g_lastLcdLines[row], paddedLine, LCD_COLUMNS + 1) == 0)
    {
      return;
    }

    memcpy(g_lastLcdLines[row], paddedLine, LCD_COLUMNS + 1);

    g_lcd.setCursor(0, row);
    g_lcd.print(paddedLine);
  }

  void renderReadingOnLcd(const SensorReading &reading)
  {
    char line1[LCD_COLUMNS + 1];
    char line2[LCD_COLUMNS + 1];

    const int humidityPct = constrain(static_cast<int>(reading.humidityPct + 0.5f), 0, 100);
    const int pressureHpa = static_cast<int>(reading.pressureHpa + 0.5f);
    const char *rainText = reading.isRaining ? "YES" : "NO";

    // Keep text compact so both values remain visible on a 16x2 display.
    snprintf(line1, sizeof(line1), "Temp:%4.1fC %3d%%", reading.temperatureC, humidityPct);
    snprintf(line2, sizeof(line2), "Rain:%-3s %4dhPa", rainText, pressureHpa);

    writeLcdLine(0, line1);
    writeLcdLine(1, line2);
  }

  void renderStatusOnLcd(const char *line1, const char *line2)
  {
    writeLcdLine(0, line1);
    writeLcdLine(1, line2);
  }

  void initializeDisplay()
  {
    Wire.begin();
    g_lcd.init();
    g_lcd.backlight();
    renderStatusOnLcd("Weather station", "Starting...");
  }

  bool sampleSensors()
  {
    SensorReading reading{};
    if (!g_sensorManager.sample(reading))
    {
      g_hasValidReading = false;
      renderStatusOnLcd("Sensor sample", "failed");
      return false;
    }

    g_latestReading = reading;
    g_hasValidReading = true;
    renderReadingOnLcd(g_latestReading);
    return true;
  }

  void initializeSubsystems()
  {
    g_wifiManager.begin();
    g_uploader.begin();
    g_sensorManager.setTimestampProvider(currentUnixTimeProvider);
    g_sensorManager.begin();
  }

  void initializeWatchdog()
  {
    const esp_err_t initResult = esp_task_wdt_init(LOOP_WATCHDOG_TIMEOUT_SECONDS, true);
    if (initResult != ESP_OK && initResult != ESP_ERR_INVALID_STATE)
    {
      Serial.printf("Task watchdog init failed: %d\n", static_cast<int>(initResult));
      return;
    }

    const esp_err_t addResult = esp_task_wdt_add(nullptr);
    if (addResult != ESP_OK && addResult != ESP_ERR_INVALID_ARG)
    {
      Serial.printf("Task watchdog add failed: %d\n", static_cast<int>(addResult));
      return;
    }

    Serial.printf("Task watchdog enabled: %u seconds\n", static_cast<unsigned>(LOOP_WATCHDOG_TIMEOUT_SECONDS));
  }

  void renderConnectivityStatus()
  {
    if (g_wifiManager.justConnected())
    {
      renderStatusOnLcd("WiFi connected", g_wifiManager.timeReady() ? "Upload ready" : "Syncing clock");
    }
    else if (g_wifiManager.justDisconnected())
    {
      renderStatusOnLcd("WiFi lost", "Reconnecting...");
    }
  }

  void handlePeriodicUpload(unsigned long nowMs)
  {
    if (nowMs - g_lastUploadMs < config::UPLOAD_INTERVAL_MS)
    {
      return;
    }

    g_lastUploadMs = nowMs;

    if (!g_hasValidReading)
    {
      Serial.println("Upload skipped: latest sensor reading is invalid");
      return;
    }

    if (!g_wifiManager.timeReady())
    {
      if (g_lastClockWaitLogMs == 0 || nowMs - g_lastClockWaitLogMs >= STATUS_LOG_INTERVAL_MS)
      {
        Serial.println("Upload deferred: waiting for NTP time sync");
        g_lastClockWaitLogMs = nowMs;
      }
      return;
    }

    if (g_uploader.enqueue(g_latestReading))
    {
      Serial.printf(
          "Queued Firebase sample ts=%llu T=%.2fC H=%.2f%% P=%.2fhPa Rain=%s pending=%u\n",
          static_cast<unsigned long long>(g_latestReading.timestamp),
          g_latestReading.temperatureC,
          g_latestReading.humidityPct,
          g_latestReading.pressureHpa,
          g_latestReading.isRaining ? "YES" : "NO",
          static_cast<unsigned>(g_uploader.pendingCount()));
      return;
    }

    Serial.printf(
        "Skipping upload: ts=%llu pending=%u\n",
        static_cast<unsigned long long>(g_latestReading.timestamp),
        static_cast<unsigned>(g_uploader.pendingCount()));
  }

  void handlePeriodicSampling(unsigned long nowMs)
  {
    if (nowMs - g_lastSensorPollMs < SENSOR_POLL_INTERVAL_MS)
    {
      return;
    }

    g_lastSensorPollMs = nowMs;
    if (!sampleSensors())
    {
      Serial.println("Sensor sample failed");
    }
  }

  void logHeapUsage(unsigned long nowMs)
  {
    if (g_lastHeapLogMs != 0 && nowMs - g_lastHeapLogMs < HEAP_LOG_INTERVAL_MS)
    {
      return;
    }

    g_lastHeapLogMs = nowMs;
    Serial.printf("Heap check: free=%u min=%u\n",
                  static_cast<unsigned>(ESP.getFreeHeap()),
                  static_cast<unsigned>(ESP.getMinFreeHeap()));
  }

  void logHealth(unsigned long nowMs)
  {
    if (g_lastHealthLogMs != 0 && nowMs - g_lastHealthLogMs < config::HEALTH_LOG_INTERVAL_MS)
    {
      return;
    }

    g_lastHealthLogMs = nowMs;
    Serial.printf(
        "Health: heap=%u min_heap=%u wifi=%s time=%s firebase=%s pending=%u dropped=%u stalled=%lu last_ok=%lu\n",
        static_cast<unsigned>(ESP.getFreeHeap()),
        static_cast<unsigned>(ESP.getMinFreeHeap()),
        g_wifiManager.isConnected() ? "up" : "down",
        g_wifiManager.timeReady() ? "ready" : "syncing",
        g_uploader.waitingForAuth() ? "auth" : (g_uploader.firebaseReady() ? "ready" : "idle"),
        static_cast<unsigned>(g_uploader.pendingCount()),
        static_cast<unsigned>(g_uploader.droppedCount()),
        static_cast<unsigned long>(g_uploader.stalledDurationMs()),
        static_cast<unsigned long>(g_uploader.lastSuccessfulUploadMs()));
    if (g_wifiManager.isConnected())
    {
      Serial.printf("WiFi RSSI: %d dBm\n", WiFi.RSSI());
    }
  }

  void recoverIfSystemStalled(unsigned long nowMs)
  {
    const bool restartForWifiLoss =
        g_wifiManager.hasEverConnected() &&
        g_wifiManager.disconnectedDurationMs(nowMs) >= config::WIFI_RECOVERY_RESTART_MS;
    const bool restartForUploadStall =
        g_wifiManager.isConnected() &&
        g_wifiManager.timeReady() &&
        g_uploader.hasUploadStalled();

    if (!restartForWifiLoss && !restartForUploadStall)
    {
      return;
    }

    if (g_lastRecoveryActionMs != 0 && nowMs - g_lastRecoveryActionMs < config::RECOVERY_ACTION_COOLDOWN_MS)
    {
      return;
    }

    g_lastRecoveryActionMs = nowMs;

    if (restartForWifiLoss)
    {
      Serial.printf(
          "Recovery: WiFi outage %lu ms, forcing reconnect\n",
          static_cast<unsigned long>(g_wifiManager.disconnectedDurationMs(nowMs)));
      g_wifiManager.forceReconnect();
      return;
    }

    Serial.printf(
        "Recovery: upload stalled for %lu ms, resetting uploader session\n",
        static_cast<unsigned long>(g_uploader.stalledDurationMs()));
    g_uploader.requestRecovery();
    Serial.println("Upload recovery requested without restarting ESP32");
  }
} // namespace

void setup()
{
  const esp_reset_reason_t resetReason = esp_reset_reason();
  ++g_bootCount;
  if (resetReason == ESP_RST_SW)
  {
    ++g_softwareRestartCount;
  }
  else
  {
    g_softwareRestartCount = 0;
  }

  g_restartEscalationDisabled = g_softwareRestartCount >= 3;
  Serial.begin(115200);
  Serial.printf(
      "Booting environmental monitoring station. boot=%lu reset=%s\n",
      static_cast<unsigned long>(g_bootCount),
      resetReasonToString(resetReason));
  if (g_restartEscalationDisabled)
  {
    Serial.printf(
        "Software restart escalation disabled after %lu consecutive software resets\n",
        static_cast<unsigned long>(g_softwareRestartCount));
  }

  initializeDisplay();
  g_ledManager.begin();
  initializeWatchdog();

  initializeSubsystems();
  if (!g_sensorManager.pressureAvailable())
  {
    Serial.println("BMP280 unavailable");
    renderStatusOnLcd("Pressure sensor", "not detected");
  }

  if (!sampleSensors())
  {
    renderStatusOnLcd("Sensor sample", "failed");
  }

  const unsigned long nowMs = millis();
  g_lastSensorPollMs = nowMs;
  g_lastUploadMs = nowMs;
}

void loop()
{
  const unsigned long nowMs = millis();

  g_wifiManager.update(nowMs);
  g_ledManager.update(nowMs, g_wifiManager.isConnected());
  g_uploader.update(nowMs, g_wifiManager.isConnected(), g_wifiManager.timeReady());
  renderConnectivityStatus();
  handlePeriodicSampling(nowMs);
  handlePeriodicUpload(nowMs);

  logHealth(nowMs);
  logHeapUsage(nowMs);
  recoverIfSystemStalled(nowMs);

  esp_task_wdt_reset();
  yield();
}
