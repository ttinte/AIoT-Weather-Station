// ===============================
// FIREBASE IMPORT
// ===============================
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getDatabase, ref, onValue } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

const firebaseConfig = {
  apiKey: "AIzaSyBuot6MSqGdTXu19kEMfONUvIpid323Fj4",
  authDomain: "aiotnhom2.firebaseapp.com",
  databaseURL: "https://aiotnhom2-80e7a-default-rtdb.firebaseio.com",
  projectId: "aiotnhom2",
  storageBucket: "aiotnhom2.firebasestorage.app",
  messagingSenderId: "6813545233",
  appId: "1:6813545233:web:48d4e7ca32c2e6250592b0",
  measurementId: "G-TJPHJ5LF6T"
};

const FIREBASE_LATEST_PATH = "weather_stations/Weather_station_1/readings/2026-06-18";

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
const forecastHistory = document.getElementById("forecastHistory");

// ===============================
// CẤU HÌNH API AI CỦA TÍN
// ===============================
const API_CONFIG = {
    BASE_URL: "https://aiot-nhom2-ai.onrender.com", // Cập nhật link server Render vào đây
    API_KEY: "nhom2_aiot"                            // Cập nhật API Key vào đây
};

let historyForecastArray = []; // Mảng lưu trữ lịch sử dự báo

// ===============================
// HÀM GỌI API DỰ BÁO AI
// ===============================
// ===============================
// HÀM GỌI API DỰ BÁO AI
// ===============================
async function getForecastAI(temp, hum, rain) {
    try {
        const requestBody = { temperature: temp, humidity: hum, rain: rain };

        const response = await fetch(`${API_CONFIG.BASE_URL}/predict`, {
            method: "POST", 
            headers: {
                "Content-Type": "application/json",
                "x-api-key": API_CONFIG.API_KEY
            },
            body: JSON.stringify(requestBody) 
        });

        const data = await response.json();

        // NẾU AI BÁO THIẾU DỮ LIỆU (LỖI 400)
        if (!response.ok) {
            if (response.status === 400 && data.error) {
                // In thẳng câu "Không đủ dữ liệu..." ra màn hình
                return `Trạng thái: ${data.error}`; 
            }
            throw new Error(`Lỗi HTTP! Trạng thái: ${response.status}`);
        }

        // NẾU ĐỦ DATA VÀ DỰ BÁO THÀNH CÔNG
        if (data.status === "success" && data.prediction && data.prediction.length > 0) {
            const predValues = data.prediction[0];
            const nextTemp = predValues[0]; 
            const nextHum = predValues[1];  
            const nextPres = predValues[2] || 0; 
            const nextRain = predValues[3] || 0; 
            const rainText = nextRain >= 0.5 ? "Có mưa" : "Không mưa"; 

            return `Dự báo 20 phút sau: ${nextTemp.toFixed(1)}°C, Ẩm ${nextHum.toFixed(1)}%, Áp suất ${nextPres.toFixed(1)}hPa, ${rainText}`;
        }

        return "AI không trả về dữ liệu hợp lệ.";
    } catch (error) {
        console.error("Lỗi khi kết nối AI:", error);
        return "Mất kết nối với AI. Đang chờ phản hồi từ máy chủ...";
    }
}

// Cập nhật giao diện mảng Lịch sử dự báo
function updateHistoryUI(timeStr, forecastStr) {
    historyForecastArray.unshift({ time: timeStr, text: forecastStr });
    
    // Giữ tối đa 5 bản ghi dự báo gần nhất
    if (historyForecastArray.length > 5) {
        historyForecastArray.pop();
    }

    forecastHistory.innerHTML = "";
    historyForecastArray.forEach(item => {
        const div = document.createElement("div");
        div.className = "forecast-item";
        // Chỉnh lại CSS lưới cho mục lịch sử dự báo
        div.style.gridTemplateColumns = "60px 1fr";
        
        div.innerHTML = `
            <span>${item.time}</span>
            <span><i class="fa-solid fa-robot"></i> ${item.text}</span>
        `;
        forecastHistory.appendChild(div);
    });
}

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
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { position: "top", labels: { usePointStyle: true, boxWidth: 8, font: { size: 13 } } },
      tooltip: { backgroundColor: "#111827", titleColor: "#ffffff", bodyColor: "#ffffff", padding: 12, cornerRadius: 8 }
    },
    scales: {
      y: { beginAtZero: false, grid: { color: "#e5e7eb" }, ticks: { color: "#6b7280" } },
      y1: { position: "right", beginAtZero: false, grid: { drawOnChartArea: false }, ticks: { color: "#6b7280" } },
      x: { grid: { color: "#f1f5f9" }, ticks: { color: "#6b7280" } }
    }
  }
});

// ===============================
// HÀM ĐỊNH DẠNG THỜI GIAN
// ===============================
function formatTime(timestamp) {
  if (!timestamp) {
    return new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
  }
  let time = Number(timestamp);
  if (time < 1000000000000) { time = time * 1000; }
  return new Date(time).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
}

// ===============================
// ĐỌC DỮ LIỆU TỪ FIREBASE
// ===============================
const latestRef = ref(database, FIREBASE_LATEST_PATH);

onValue(latestRef, async (snapshot) => {
  const dataNode = snapshot.val();
  if (!dataNode) return;

  sensorChart.data.labels = [];
  sensorChart.data.datasets[0].data = [];
  sensorChart.data.datasets[1].data = [];
  sensorChart.data.datasets[2].data = [];

  let lastData = null;

  Object.keys(dataNode).forEach((timestampKey) => {
    const data = dataNode[timestampKey];
    
    let timeLabel = (data.timestamp && !isNaN(Number(data.timestamp))) 
      ? formatTime(data.timestamp) 
      : new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }); 

    const tempVal = !Number.isNaN(Number(data.temperature)) ? Number(data.temperature) : null;
    const humVal = !Number.isNaN(Number(data.humidity)) ? Number(data.humidity) : null;
    const pressVal = !Number.isNaN(Number(data.pressure)) ? Number(data.pressure) : null;

    if (tempVal !== null || humVal !== null || pressVal !== null) {
      sensorChart.data.labels.push(timeLabel);
      sensorChart.data.datasets[0].data.push(tempVal);
      sensorChart.data.datasets[1].data.push(humVal);
      sensorChart.data.datasets[2].data.push(pressVal);
    }
    lastData = data;
  });

  const maxPoints = 50;
  while (sensorChart.data.labels.length > maxPoints) {
    sensorChart.data.labels.shift();
    sensorChart.data.datasets.forEach((dataset) => dataset.data.shift());
  }
  sensorChart.update();

  if (lastData) {
    const temp = Number(lastData.temperature);
    const hum = Number(lastData.humidity);
    const press = Number(lastData.pressure);
    const rain = Number(lastData.rain);

    temperatureValue.textContent = !Number.isNaN(temp) ? `${temp.toFixed(1)} °C` : "-- °C";
    humidityValue.textContent = !Number.isNaN(hum) ? `${hum.toFixed(1)} %` : "-- %";
    pressureValue.textContent = !Number.isNaN(press) ? `${press.toFixed(1)} hPa` : "-- hPa";
    rainValue.textContent = !Number.isNaN(rain) ? (rain === 1 ? "Có" : "Không") : "--";

    // ----------------------------------------
    // TÍCH HỢP AI VÀO KẾT LUẬN & LỊCH SỬ
    // ----------------------------------------
    weatherConclusion.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> AI đang phân tích dữ liệu...`;
    
    // Gọi API của server AI
    const aiResult = await getForecastAI(temp, hum, rain);
    
    // Đẩy kết quả ra khung kết luận chính
    weatherConclusion.innerHTML = `<strong>Kết quả:</strong> ${aiResult}`;

    // Nạp kết quả vào mảng lịch sử dự báo
    const timeStr = formatTime(lastData.timestamp);
    updateHistoryUI(timeStr, aiResult);
  }
});