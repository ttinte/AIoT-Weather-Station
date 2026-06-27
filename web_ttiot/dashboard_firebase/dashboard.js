// ===============================
// FIREBASE IMPORT
// ===============================
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getDatabase, ref, onValue, query, orderByKey, limitToLast } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

// Cấu hình mới nhất đã được cập nhật
const firebaseConfig = {
  apiKey: "AIzaSyCEZa07j4UMaaHLhboLotGh8IFCs3bOciM",
  authDomain: "aiotnhom2-80e7a.firebaseapp.com",
  databaseURL: "https://aiotnhom2-80e7a-default-rtdb.firebaseio.com",
  projectId: "aiotnhom2-80e7a",
  storageBucket: "aiotnhom2-80e7a.firebasestorage.app",
  messagingSenderId: "72429002560",
  appId: "1:72429002560:web:9d970722994a70a61ba4a5",
  measurementId: "G-TXP8W6MQ0T"
};

const FIREBASE_LATEST_PATH = "weather_stations/Weather_station_1/latest";

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
    BASE_URL: "https://weatherpredict-model-5.onrender.com",
    API_KEY: "nhom2"
};

let historyForecastArray = []; // Mảng lưu trữ lịch sử dự báo

// ===============================
// HÀM GỌI API DỰ BÁO AI
// ===============================
// ===============================
// HÀM GỌI API DỰ BÁO AI
// ===============================
async function getForecastAI(temp, hum, rain, pressure) {
    try {
        const requestBody = {
            temperature: temp,
            humidity: hum,
            rain: rain,
            pressure: pressure
        };

        const response = await fetch(`${API_CONFIG.BASE_URL}/predict`, {
            method: "POST", 
            headers: {
                "Content-Type": "application/json",
                "x-api-key": API_CONFIG.API_KEY
            },
            body: JSON.stringify(requestBody) 
        });

        const data = await response.json().catch(() => ({}));

        // NẾU AI BÁO THIẾU DỮ LIỆU (LỖI 400)
        if (!response.ok) {
            if (response.status === 400 && data.error) {
                // In thẳng câu "Không đủ dữ liệu..." ra màn hình
                return `Trạng thái: ${data.error}`; 
            }
            const message = data.error || data.message || response.statusText || "Không rõ lỗi";
            return `AI lỗi HTTP ${response.status}: ${message}`;
        }

        // NẾU ĐỦ DATA VÀ DỰ BÁO THÀNH CÔNG
        if (data.status === "success") {
            const forecast = data.forecast || {};
            const predValues = Array.isArray(data.prediction) && Array.isArray(data.prediction[0])
                ? data.prediction[0]
                : [];

            const nextTemp = Number(forecast.temperature ?? predValues[0]);
            const nextHum = Number(forecast.humidity ?? predValues[1]);
            const nextPres = Number(forecast.pressure ?? predValues[2]);
            const nextRain = Number(forecast.rain ?? predValues[3]);

            if ([nextTemp, nextHum, nextPres, nextRain].some(Number.isNaN)) {
                console.warn("AI trả về dữ liệu thiếu hoặc sai định dạng:", data);
                return "AI trả về dữ liệu chưa đủ để hiển thị.";
            }

            const rainText = nextRain >= 0.5 ? "Có mưa" : "Không mưa"; 

            return `Dự báo 20 phút sau: ${nextTemp.toFixed(1)}°C, Ẩm ${nextHum.toFixed(1)}%, Áp suất ${nextPres.toFixed(1)}hPa, ${rainText}`;
        }

        return "AI không trả về dữ liệu hợp lệ.";
    } catch (error) {
        console.error("Lỗi khi kết nối AI:", error);
        const detail = error && error.message ? ` (${error.message})` : "";
        return `Mất kết nối với AI${detail}.`;
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
// ĐƯỜNG DẪN FIREBASE CỦA READINGS
// ===============================
function getCurrentDateStr() {
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, '0');
  const dd = String(today.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}
const FIREBASE_READINGS_PATH = `weather_stations/Weather_station_1/readings/${getCurrentDateStr()}`;

// ===============================
// 1. ĐỌC 'LATEST' ĐỂ HIỂN THỊ 4 Ô THÔNG SỐ & DỰ BÁO AI
// ===============================
const latestRef = ref(database, FIREBASE_LATEST_PATH);

onValue(latestRef, async (snapshot) => {
  const data = snapshot.val();
  if (!data) return; // Nếu chưa có dữ liệu thì bỏ qua

  // Lấy dữ liệu trực tiếp từ object latest
  const temp = Number(data.temperature);
  const hum = Number(data.humidity);
  const press = Number(data.pressure);
  const rain = Number(data.rain);
  const timestamp = data.timestamp;


  // Cập nhật 4 ô giao diện
  temperatureValue.textContent = !Number.isNaN(temp) ? `${temp.toFixed(1)} °C` : "-- °C";
  humidityValue.textContent = !Number.isNaN(hum) ? `${hum.toFixed(1)} %` : "-- %";
  pressureValue.textContent = !Number.isNaN(press) ? `${press.toFixed(1)} hPa` : "-- hPa";
  rainValue.textContent = !Number.isNaN(rain) ? (rain >= 0.5 ? "Có" : "Không") : "--";

    // ... (code gán 4 ô thông số của bạn ở đây) ...

  // =========================================
  // LOGIC CẢNH BÁO CHỚP ĐỎ (Lấy từ Setting)
  // =========================================
  const savedSettings = JSON.parse(localStorage.getItem('iotWeatherSettings'));
  
  // Lấy ngưỡng (nếu chưa cài thì lấy mặc định 35 độ và 40%)
  const maxTemp = savedSettings && savedSettings.tempThreshold ? Number(savedSettings.tempThreshold) : 35;
  const minHumid = savedSettings && savedSettings.humidThreshold ? Number(savedSettings.humidThreshold) : 40;
  const isAlertEnabled = savedSettings && savedSettings.webAlert !== undefined ? savedSettings.webAlert : true;

  // Kiểm tra: Bật cảnh báo (Toggle = true) VÀ (Nhiệt độ > Max HOẶC Độ ẩm < Min)
  if (isAlertEnabled && (temp > maxTemp || hum < minHumid)) {
      document.body.classList.add('alert-active');
  } else {
      document.body.classList.remove('alert-active');
  }
  
  
  // Gọi AI xử lý dữ liệu mới nhất
  weatherConclusion.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> AI đang phân tích dữ liệu...`;
  
  const aiResult = await getForecastAI(temp, hum, rain, press);
  weatherConclusion.innerHTML = `<strong>Kết quả:</strong> ${aiResult}`;

  const timeStr = formatTime(timestamp);
  updateHistoryUI(timeStr, aiResult);
});

// ===============================
// ĐỌC CÀI ĐẶT TỪ BỘ NHỚ (LOCALSTORAGE)
// ===============================
const savedSettings = JSON.parse(localStorage.getItem('iotWeatherSettings'));

// Nếu có cài đặt thì lấy số bản ghi, không thì mặc định là 50
const recordLimit = savedSettings && savedSettings.chartLimit ? parseInt(savedSettings.chartLimit) : 50;

// Cập nhật "Tên Trạm Giám Sát" lên Badge góc phải màn hình
if (savedSettings && savedSettings.stationName) {
    const stationBadgeSpan = document.querySelector('.station-badge span');
    if (stationBadgeSpan) {
        stationBadgeSpan.textContent = savedSettings.stationName;
    }
}

// ===============================
// 2. ĐỌC 'READINGS' ĐỂ VẼ BIỂU ĐỒ LỊCH SỬ
// ===============================
// Thay vì fix cứng số 50, ta truyền biến recordLimit vào limitToLast()
const readingsRef = query(
  ref(database, FIREBASE_READINGS_PATH), 
  orderByKey(), 
  limitToLast(recordLimit) 
);

onValue(readingsRef, (snapshot) => {
  if (!snapshot.exists()) return;

  // Xóa dữ liệu cũ trên biểu đồ trước khi nạp dữ liệu mới
  sensorChart.data.labels = [];
  sensorChart.data.datasets[0].data = [];
  sensorChart.data.datasets[1].data = [];
  sensorChart.data.datasets[2].data = [];

  // Duyệt qua từng mốc thời gian trong ngày
  snapshot.forEach((childSnapshot) => {
    const data = childSnapshot.val();
    
    // Key của childSnapshot chính là timestamp
    const timestamp = data.timestamp || childSnapshot.key;
    let timeLabel = formatTime(timestamp);

    const tempVal = !Number.isNaN(Number(data.temperature)) ? Number(data.temperature) : null;
    const humVal = !Number.isNaN(Number(data.humidity)) ? Number(data.humidity) : null;
    const pressVal = !Number.isNaN(Number(data.pressure)) ? Number(data.pressure) : null;

    if (tempVal !== null || humVal !== null || pressVal !== null) {
      sensorChart.data.labels.push(timeLabel);
      sensorChart.data.datasets[0].data.push(tempVal);
      sensorChart.data.datasets[1].data.push(humVal);
      sensorChart.data.datasets[2].data.push(pressVal);
    }
  });

  // Render lại biểu đồ
  sensorChart.update();
});
