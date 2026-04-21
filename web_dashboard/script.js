function login() {
    const usernameInput = document.getElementById("username");
    const passwordInput = document.getElementById("password");
    const errorElement = document.getElementById("error");

    if (!usernameInput || !passwordInput || !errorElement) {
        return;
    }

    let u = usernameInput.value.trim();
    let p = passwordInput.value;

    if (u === "nhom2" && p === "nhom2") {
        window.location = "dashboard.html";
    } else {
        errorElement.innerText = "Sai tài khoản";
    }
}

function logout() {
    window.location = "index.html";
}

window.login = login;
window.logout = logout;

/* Theme toggle */
function toggleTheme() {
    const nextTheme = document.body.classList.contains("light") ? "dark" : "light";
    applyTheme(nextTheme);
}

window.toggleTheme = toggleTheme;

let chart;
let unsubscribeWeather = null;

function getStoredPreference(key, fallback) {
    return localStorage.getItem(key) || fallback;
}

function applyTheme(theme) {
    document.body.classList.toggle("light", theme === "light");
    localStorage.setItem("dashboardTheme", theme);
}

function initializeTheme() {
    applyTheme(getStoredPreference("dashboardTheme", "dark"));
}
// Hàm gọi API sang server AI của nhóm
async function getForecastAI(temp, hum, rain) {
    try {
        // Chuẩn bị body dữ liệu gửi cho AI dự đoán (Format tham khảo)
        const requestBody = {
            temperature: temp,
            humidity: hum,
            rain: rain
        };

        // Bắn request sang server Render
        const response = await fetch(`${API_CONFIG.BASE_URL}/predict`, {
            method: "POST", 
            headers: {
                "Content-Type": "application/json",
                "x-api-key": API_CONFIG.API_KEY
            },
            body: JSON.stringify(requestBody) 
        });

        if (!response.ok) {
            throw new Error(`Lỗi HTTP! Trạng thái: ${response.status}`);
        }

        const data = await response.json();

        // Bóc tách dữ liệu theo đúng format
        if (data.status === "success" && data.prediction && data.prediction.length > 0) {
            const predValues = data.prediction[0];
            
            // Lấy 2 thông số cũ
            const nextTemp = predValues[0]; // Vị trí 0: Nhiệt độ
            const nextHum = predValues[1];  // Vị trí 1: Độ ẩm
            
            // Lấy thêm 2 thông số mới (dùng || 0 để lỡ AI chưa update thì web không bị lỗi NaN)
            const nextPres = predValues[2] || 0; // Vị trí 2: Áp suất
            const nextRain = predValues[3] || 0; // Vị trí 3: Mưa (Giả sử trả về số 0 hoặc 1)

            // Do cảm biến mưa của bạn xài chữ Yes/No, nên ta quy đổi kết quả AI ra chữ cho đẹp:
            const rainText = nextRain >= 0.5 ? "Có" : "Không"; 

            // Trả về câu văn hoàn chỉnh
            return `AI dự báo 20 phút sau: ${nextTemp.toFixed(1)}°C, Ẩm ${nextHum.toFixed(1)}%, Áp suất ${nextPres.toFixed(1)}hPa, Mưa: ${rainText}`;
        }

        return "AI không trả về dữ liệu hợp lệ.";

    } catch (error) {
        console.error("Lỗi khi kết nối AI:", error);
        // Trả về chữ mặc định nếu server AI sập hoặc gọi không được
        return window.getTranslatedText ? window.getTranslatedText("noData") : "Mất kết nối với AI";
    }
}
function renderEmptyDashboard(message) {
    document.getElementById("temp").innerText = "--";
    document.getElementById("humidity").innerText = "--";
    document.getElementById("rain").innerText = "--";
    document.getElementById("pressure").innerText = "--";
    document.getElementById("forecastText").innerText = message;
    document.getElementById("forecastHistory").innerHTML = "";

    if (chart) {
        chart.data.labels = [];
        chart.data.datasets.forEach((dataset) => {
            dataset.data = [];
        });
        chart.update("none");
    }
}

// --- THAY THẾ HÀM initChart ---
function initChart() {
    let ctx = document.getElementById("chart").getContext("2d");

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Temperature (°C)",
                    data: [],
                    borderColor: "#ff6b6b",
                    yAxisID: 'y', 
                    tension: 0.4
                },
                {
                    label: "Humidity (%)",
                    data: [],
                    borderColor: "#4dabf7",
                    yAxisID: 'y', 
                    tension: 0.4
                },
                {
                    label: "Pressure (hPa)",
                    data: [],
                    borderColor: "#fcc419",
                    yAxisID: 'y_pressure', 
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            
            // Tắt các chấm tròn mặc định để đồ thị mượt hơn
            elements: {
                point: {
                    radius: 0, 
                    hitRadius: 10, 
                    hoverRadius: 5 
                }
            },

            scales: {
                x: { 
                    ticks: { 
                        color: "#a0aec0",
                        // Giới hạn số lượng nhãn trục X tránh bị chồng chữ
                        maxTicksLimit: 12, 
                        maxRotation: 45    
                    } 
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    suggestedMin: 0,
                    suggestedMax: 100,
                    title: { display: true, text: 'Temp & Hum', color: '#a0aec0' },
                    ticks: { color: "#a0aec0" }
                },
                y_pressure: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    suggestedMin: 980, 
                    suggestedMax: 1020,
                    title: { display: true, text: 'Pressure (hPa)', color: '#a0aec0' },
                    grid: { drawOnChartArea: false }, 
                    ticks: { color: "#a0aec0" }
                }
            }
        }
    });
}
let historyForecast = [];

function formatSensorValue(value, unit) {
    return Number.isFinite(value) ? value.toFixed(1) + " " + unit : "--";
}

function formatChartTime(timestamp) {
    if (!timestamp) {
        return "--:--:--";
    }

    return new Date(timestamp).toLocaleTimeString("vi-VN", { hour12: false });
}


function renderChart(readings) {
    chart.data.labels = readings.map((reading) => formatChartTime(reading.timestamp));
    chart.data.datasets[0].data = readings.map((reading) => reading.temperature);
    chart.data.datasets[1].data = readings.map((reading) => reading.humidity);
    
    // Đã xóa dòng đẩy dữ liệu Mưa
    
    // Áp suất giờ sẽ là datasets[2] (thay vì 3 như trước)
    chart.data.datasets[2].data = readings.map((reading) => reading.pressure);
    chart.update("none");
}

async function renderDashboardData(latestReading, readings) {
    if (!latestReading) {
        renderEmptyDashboard(getTranslatedText("noData"));
        return;
    }

    document.getElementById("temp").innerText = formatSensorValue(latestReading.temperature, "°C");
    document.getElementById("humidity").innerText = formatSensorValue(latestReading.humidity, "%");
    document.getElementById("pressure").innerText = formatSensorValue(latestReading.pressure, "hPa");

    // XỬ LÝ RIÊNG CHO CẢM BIẾN MƯA (Không dùng mm nữa)
// XỬ LÝ RIÊNG CHO CẢM BIẾN MƯA (Tuyệt chiêu tự động hóa bằng data-lang)
    const rainEl = document.getElementById("rain");
    if (rainEl) {
        // Gán thẻ data-lang động. Nếu mưa thì gắn "rainYes", không mưa gắn "rainNo"
        const rainStatus = latestReading.rain === "yes" ? "rainYes" : "rainNo";
        rainEl.setAttribute("data-lang", rainStatus);
        
        // Cập nhật chữ hiển thị ngay lúc nhận data (dùng ngôn ngữ hiện tại)
        rainEl.innerText = getTranslatedText(rainStatus);
    }

    renderChart(readings);

    // Gọi AI thay vì if-else
    document.getElementById("forecastText").innerText = getTranslatedText("aiThinking");
    let forecast = await getForecastAI(latestReading.temperature, latestReading.humidity, latestReading.rain);

    document.getElementById("forecastText").innerText = forecast;
    
    // Đẩy lịch sử AI vào danh sách
    let timeStr = formatChartTime(latestReading.timestamp);
    historyForecast.unshift(timeStr + " - AI: " + forecast);
    if(historyForecast.length > 5) historyForecast.pop();

    let list = document.getElementById("forecastHistory");
    list.innerHTML = "";
    historyForecast.forEach((item) => {
        let li = document.createElement("li");
        li.innerText = item;
        list.appendChild(li);
    });
}

function initDashboardRealtime() {
    if (!window.firebaseWeatherApi || typeof window.firebaseWeatherApi.subscribeToWeatherReadings !== "function") {
        renderEmptyDashboard(getTranslatedText("fbError"));
        return;
    }

    unsubscribeWeather = window.firebaseWeatherApi.subscribeToWeatherReadings({
        limit: 720,
        onData: ({ latest, history }) => {
            renderDashboardData(latest, history);
        },
        onError: () => {
            renderEmptyDashboard("Unable to load Firebase data");
        }
    });
}

function initDashboardControls() {
    const btnTheme = document.getElementById("btn-theme");
    if (btnTheme) {
        btnTheme.addEventListener("click", () => {
            const nextTheme = document.body.classList.contains("light") ? "dark" : "light";
            applyTheme(nextTheme);
        });
    }

    const btnLangVi = document.getElementById("btn-lang-vi");
    if (btnLangVi) {
        btnLangVi.addEventListener("click", () => setLang("vi"));
    }

    const btnLangEn = document.getElementById("btn-lang-en");
    if (btnLangEn) {
        btnLangEn.addEventListener("click", () => setLang("en"));
    }

    const btnLogout = document.getElementById("btn-logout");
    if (btnLogout) {
        btnLogout.addEventListener("click", logout);
    }
}

function initLoginInteractions() {
    const usernameInput = document.getElementById("username");
    const passwordInput = document.getElementById("password");

    [usernameInput, passwordInput].forEach((input) => {
        if (!input) {
            return;
        }

        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                login();
            }
        });
    });
}

document.addEventListener("DOMContentLoaded", function() {
    initializeTheme();
    initLoginInteractions();

    if (document.getElementById("chart")) {
        initChart();
        initDashboardRealtime();
        initDashboardControls();
    }
});

window.addEventListener("beforeunload", function() {
    if (typeof unsubscribeWeather === "function") {
        unsubscribeWeather();
    }
});
// --- XỬ LÝ CẬP NHẬT GIAO DIỆN ĐỘNG KHI ĐỔI NGÔN NGỮ ---
window.addEventListener('languageChanged', () => {
    
// 1. Dịch lại nhãn của biểu đồ Chart.js và vẽ lại
    if (chart) {
        chart.data.datasets[0].label = getTranslatedText("chartTemp");
        chart.data.datasets[1].label = getTranslatedText("chartHum");
        
        // Đã xóa dòng dịch Mưa
        
        // Cập nhật lại số thứ tự cho Áp suất thành [2]
        chart.data.datasets[2].label = getTranslatedText("chartPres"); 

        chart.options.scales.y.title.text = getTranslatedText("axisTempHum");
        chart.options.scales.y_pressure.title.text = getTranslatedText("chartPres");
        
        // Đã xóa các cấu hình đổi chữ Yes/No của trục y_rain

        chart.update("none"); 
    }

    // 2. Dịch lại chữ đang hiển thị dở dang (AI is thinking...)
    const forecastEl = document.getElementById("forecastText");
    if (forecastEl) {
        const currentText = forecastEl.innerText;
        // Nếu đang hiển thị chữ chờ AI thì đổi ngôn ngữ ngay lập tức
        if (currentText === "AI is thinking..." || currentText === "AI đang suy nghĩ...") {
            forecastEl.innerText = getTranslatedText("aiThinking");
        } else if (currentText === "Waiting for data..." || currentText === "Đang đợi dữ liệu...") {
            forecastEl.innerText = getTranslatedText("noData");
        }
        // Lưu ý: Nếu AI đã trả ra kết quả dự báo thời tiết (ví dụ: "Trời có thể mưa"), 
        // thì ta không dịch đè lên kết quả đó. Lần data mới tới, AI sẽ tự trả về ngôn ngữ mới.
    }
});
import { API_CONFIG } from "./config.js";