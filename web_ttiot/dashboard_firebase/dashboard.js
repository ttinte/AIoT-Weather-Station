// ===============================
// FIREBASE IMPORT
// ===============================

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getDatabase,
  ref,
  onValue
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

// ===============================
// CẤU HÌNH FIREBASE
// ===============================

const firebaseConfig = {
  apiKey: "AIzaSyBuot6MSqGdTXu19kEMfONUvIpid323Fj4",
  authDomain: "aiotnhom2.firebaseapp.com",
  databaseURL: "https://aiotnhom2-default-rtdb.firebaseio.com",
  projectId: "aiotnhom2",
  storageBucket: "aiotnhom2.firebasestorage.app",
  messagingSenderId: "6813545233",
  appId: "1:6813545233:web:48d4e7ca32c2e6250592b0",
  measurementId: "G-TJPHJ5LF6T"
};



// Đường dẫn này phải trùng với nhánh latest mà ESP32 đang ghi dữ liệu.
// Bạn cần sửa dòng này theo cây dữ liệu trong Realtime Database.
const FIREBASE_LATEST_PATH = "weather_stations/Weather_station_1/latest";

// Khởi tạo Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// ===============================
// LẤY PHẦN TỬ HTML
// ===============================

const temperatureValue = document.getElementById("temperatureValue");
const humidityValue = document.getElementById("humidityValue");
const rainValue = document.getElementById("rainValue");
const pressureValue = document.getElementById("pressureValue");
const weatherConclusion = document.getElementById("weatherConclusion");

// ===============================
// BIỂU ĐỒ LỊCH SỬ CẢM BIẾN
// ===============================

const ctx = document.getElementById("sensorChart");

const sensorChart = new Chart(ctx, {
  type: "line",

  data: {
    labels: [],

    datasets: [
      {
        label: "Nhiệt độ (°C)",
        data: [],
        borderColor: "#ef4444",
        backgroundColor: "rgba(239, 68, 68, 0.12)",
        borderWidth: 2,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 6,
        fill: false
      },

      {
        label: "Độ ẩm (%)",
        data: [],
        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.12)",
        borderWidth: 2,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 6,
        fill: false
      },

      {
        label: "Áp suất (hPa)",
        data: [],
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.12)",
        borderWidth: 2,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 6,
        fill: false,
        yAxisID: "y1"
      }
    ]
  },

  options: {
    responsive: true,
    maintainAspectRatio: false,

    interaction: {
      mode: "index",
      intersect: false
    },

    plugins: {
      legend: {
        position: "top",
        labels: {
          usePointStyle: true,
          boxWidth: 8,
          font: {
            size: 13
          }
        }
      },

      tooltip: {
        backgroundColor: "#111827",
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        padding: 12,
        cornerRadius: 8
      }
    },

    scales: {
      y: {
        beginAtZero: false,
        grid: {
          color: "#e5e7eb"
        },
        ticks: {
          color: "#6b7280"
        }
      },

      y1: {
        position: "right",
        beginAtZero: false,
        grid: {
          drawOnChartArea: false
        },
        ticks: {
          color: "#6b7280"
        }
      },

      x: {
        grid: {
          color: "#f1f5f9"
        },
        ticks: {
          color: "#6b7280"
        }
      }
    }
  }
});

// ===============================
// HÀM ĐỊNH DẠNG THỜI GIAN
// ===============================

function formatTime(timestamp) {
  if (!timestamp) {
    return new Date().toLocaleTimeString("vi-VN", {
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  let time = Number(timestamp);

  // Nếu timestamp là giây thì đổi sang mili giây
  if (time < 1000000000000) {
    time = time * 1000;
  }

  return new Date(time).toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit"
  });
}

// ===============================
// CẬP NHẬT KẾT LUẬN THỜI TIẾT
// ===============================

function updateWeatherConclusion(data) {
  const temperature = Number(data.temperature);
  const humidity = Number(data.humidity);
  const rain = Number(data.rain);

  let conclusion = "";

  if (rain === 1) {
    conclusion = "Hiện tại khu vực trạm đo đang ghi nhận có mưa. Cần chú ý khi di chuyển ngoài trời.";
  } else if (temperature >= 35) {
    conclusion = "Thời tiết hiện tại khá nóng, nhiệt độ đang ở mức cao. Nên hạn chế hoạt động ngoài trời trong thời gian dài.";
  } else if (humidity >= 80) {
    conclusion = "Độ ẩm đang ở mức cao, không khí có thể khá oi và dễ xuất hiện mưa.";
  } else {
    conclusion = "Thời tiết hiện tại tương đối ổn định, chưa ghi nhận mưa tại khu vực trạm đo.";
  }

  weatherConclusion.textContent = conclusion;
}

// ===============================
// CẬP NHẬT BIỂU ĐỒ
// ===============================

function updateChart(data) {
  const timeLabel = formatTime(data.timestamp);

  const temperature = Number(data.temperature);
  const humidity = Number(data.humidity);
  const pressure = Number(data.pressure);

  if (
    Number.isNaN(temperature) ||
    Number.isNaN(humidity) ||
    Number.isNaN(pressure)
  ) {
    console.log("Dữ liệu biểu đồ không hợp lệ.");
    return;
  }

  sensorChart.data.labels.push(timeLabel);
  sensorChart.data.datasets[0].data.push(temperature);
  sensorChart.data.datasets[1].data.push(humidity);
  sensorChart.data.datasets[2].data.push(pressure);

  // Giữ tối đa 10 điểm gần nhất trên biểu đồ
  const maxPoints = 10;

  if (sensorChart.data.labels.length > maxPoints) {
    sensorChart.data.labels.shift();

    sensorChart.data.datasets.forEach((dataset) => {
      dataset.data.shift();
    });
  }

  sensorChart.update();
}

// ===============================
// ĐỌC DỮ LIỆU MỚI NHẤT TỪ FIREBASE
// ===============================

const latestRef = ref(database, FIREBASE_LATEST_PATH);

onValue(latestRef, (snapshot) => {
  const data = snapshot.val();

  if (!data) {
    console.log("Chưa có dữ liệu từ Firebase.");
    return;
  }

  console.log("Dữ liệu cảm biến:", data);

  const temperature = Number(data.temperature);
  const humidity = Number(data.humidity);
  const pressure = Number(data.pressure);
  const rain = Number(data.rain);

  if (!Number.isNaN(temperature)) {
    temperatureValue.textContent = `${temperature.toFixed(1)} °C`;
  } else {
    temperatureValue.textContent = "-- °C";
  }

  if (!Number.isNaN(humidity)) {
    humidityValue.textContent = `${humidity.toFixed(1)} %`;
  } else {
    humidityValue.textContent = "-- %";
  }

  if (!Number.isNaN(pressure)) {
    pressureValue.textContent = `${pressure.toFixed(1)} hPa`;
  } else {
    pressureValue.textContent = "-- hPa";
  }

  if (!Number.isNaN(rain)) {
    rainValue.textContent = rain === 1 ? "Có" : "Không";
  } else {
    rainValue.textContent = "--";
  }

  updateWeatherConclusion(data);
  updateChart(data);
});

// ===============================
// ACTIVE MENU SIDEBAR
// ===============================

const menuItems = document.querySelectorAll(".menu-item");

menuItems.forEach((item) => {
  item.addEventListener("click", function () {
    menuItems.forEach((menu) => {
      menu.classList.remove("active");
    });

    this.classList.add("active");
  });
});