const langData = {
    en: {
        temp: "Temperature",
        humidity: "Humidity",
        rain: "Rain",
        pressure: "Pressure",
        history: "Sensor History",
        forecast: "Weather Conclusion",
        oldForecast: "Forecast History",
        mapTitle: "Weather Map",
        logout: "Logout",
        aiThinking: "AI is thinking...",
        noData: "Waiting for data...", // Đã sửa cho khớp
        fbError: "Firebase is not configured",
        rainYes: "Yes",
        rainNo: "No",
        popupContent: "<b>IoT Weather Station</b><br>Station is active.",
        
        // TỪ VỰNG CHO BIỂU ĐỒ
        chartTemp: "Temperature (°C)",
        chartHum: "Humidity (%)",
        chartRain: "Rain (Status)",
        chartPres: "Pressure (hPa)",
        axisTempHum: "Temp & Hum"
    },

    vi: {
        temp: "Nhiệt độ",
        humidity: "Độ ẩm",
        rain: "Mưa",
        pressure: "Áp suất",
        history: "Lịch sử cảm biến",
        forecast: "Kết luận thời tiết",
        oldForecast: "Lịch sử dự báo",
        mapTitle: "Bản đồ thời tiết",
        logout: "Đăng xuất",
        aiThinking: "AI đang suy nghĩ...",
        noData: "Đang đợi dữ liệu...", // Đã sửa cho khớp
        fbError: "Chưa cấu hình Firebase",
        rainYes: "Có",
        rainNo: "Không",
        popupContent: "<b>Trạm Thời Tiết IoT</b><br>Trạm đang hoạt động.",
        
        // TỪ VỰNG CHO BIỂU ĐỒ
        chartTemp: "Nhiệt độ (°C)",
        chartHum: "Độ ẩm (%)",
        chartRain: "Mưa (Trạng thái)",
        chartPres: "Áp suất (hPa)",
        axisTempHum: "Nhiệt độ & Độ ẩm"
    }
};
// ... (Phần code bên dưới của lang.js giữ nguyên)

let currentLang = "vi"; // Biến toàn cục lưu ngôn ngữ hiện tại

// Hàm hỗ trợ lấy chữ dịch cho file script.js
function getTranslatedText(key) {
    return langData[currentLang][key] || key;
}

function setLang(lang) {
    currentLang = langData[lang] ? lang : "vi";

    // 1. Dịch các thẻ có data-lang (HTML cứng)
    document.querySelectorAll("[data-lang]").forEach((el) => {
        const key = el.getAttribute("data-lang");
        el.innerText = langData[currentLang][key] || el.innerText;
    });

    document.documentElement.lang = currentLang;
    localStorage.setItem("dashboardLanguage", currentLang);

    // 2. Kích hoạt sự kiện để các file JS khác (như map, bảng điều khiển) tự cập nhật
    window.dispatchEvent(new Event('languageChanged'));
}

window.setLang = setLang;
window.getTranslatedText = getTranslatedText; // Export hàm cho script.js dùng

document.addEventListener("DOMContentLoaded", () => {
    const savedLang = localStorage.getItem("dashboardLanguage");
    setLang(savedLang || "vi");
});