#ifndef UPLOADER_H
#define UPLOADER_H

#include <Arduino.h>
#include <Firebase_ESP_Client.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "config.h"
#include "sensor_manager.h"

class Uploader
{
public:
  void begin();
  bool enqueue(const SensorReading &reading);
  void update(unsigned long nowMs, bool wifiConnected, bool timeReady);
  size_t pendingCount() const;
  size_t droppedCount() const;
  unsigned long lastSuccessfulUploadMs() const;
  unsigned long lastQueueProgressMs() const;
  unsigned long stalledDurationMs() const;
  bool hasUploadStalled() const;
  bool waitingForAuth() const;
  bool firebaseReady() const;
  void requestRecovery();

private:
  static void uploadTaskEntry_(void *context);
  void initializeFirebaseConfig_();
  void resetState_();
  void snapshotTaskInputs_(bool &wifiConnected, bool &timeReady, bool &recoveryRequested);
  bool waitForConnectivity_(unsigned long nowMs, bool wifiConnected, bool timeReady);
  void uploadTask_();
  bool ensureFirebaseReady_(unsigned long nowMs);
  bool fetchQueuedReading_(SensorReading &reading, unsigned long nowMs);
  bool waitForUploadBackoff_(unsigned long nowMs) const;
  bool uploadReading_(const SensorReading &reading, unsigned long nowMs);
  bool isFirebaseAuthPending_() const;
  void handleAuthPendingUpload_();
  void handleUploadBackoff_(unsigned long nowMs);
  void handleSuccessfulUpload_(const SensorReading &reading, unsigned long nowMs);
  void updateStallState_(bool canAttemptUpload, unsigned long nowMs);
  bool peekNextReading_(SensorReading &outReading) const;
  void popUploadedReading_(const SensorReading &uploadedReading, unsigned long nowMs);

  static constexpr size_t kQueueSize = 48;
  static constexpr uint32_t kUploadTaskStackBytes = 8192;

  FirebaseData fbdo_;
  FirebaseAuth auth_;
  FirebaseConfig firebaseConfig_;
  SensorReading queue_[kQueueSize];
  size_t head_ = 0;
  size_t tail_ = 0;
  size_t count_ = 0;
  size_t droppedCount_ = 0;
  bool firebaseStarted_ = false;
  bool wifiConnected_ = false;
  bool timeReady_ = false;
  bool firebaseReady_ = false;
  bool authPending_ = false;
  unsigned long lastFirebaseAttemptMs_ = 0;
  unsigned long lastUploadAttemptMs_ = 0;
  unsigned long lastSuccessfulUploadMs_ = 0;
  unsigned long lastQueueProgressMs_ = 0;
  unsigned long pendingSinceMs_ = 0;
  unsigned long stallSinceMs_ = 0;
  unsigned long lastQueueOverflowLogMs_ = 0;
  unsigned long nextFirebaseAttemptMs_ = 0;
  unsigned long nextUploadAttemptMs_ = 0;
  unsigned long firebaseRetryIntervalMs_ = config::FIREBASE_RETRY_INTERVAL_MS;
  unsigned long uploadRetryIntervalMs_ = config::FIREBASE_UPLOAD_RETRY_INITIAL_MS;
  unsigned long lastAuthLogMs_ = 0;
  bool recoveryRequested_ = false;
  portMUX_TYPE queueMux_ = portMUX_INITIALIZER_UNLOCKED;
  TaskHandle_t uploadTaskHandle_ = nullptr;
};

#endif // UPLOADER_H
