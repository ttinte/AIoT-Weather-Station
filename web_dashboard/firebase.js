
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getDatabase, onValue, ref, get, query, limitToLast, orderByKey } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

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

// Khởi tạo Firebase
const app = initializeApp(firebaseConfig); //khởi tạo ứng dụng Firebase
const database = getDatabase(app); //tạo kết nối đến cơ sở dữ liệu 

// ĐƯỜNG DẪN CHUẨN THEO FIREBASE CỦA NHÓM
const LATEST_PATH = "weather_stations/Weather_station_1/latest";

function toNumber(value, fallback = null) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeReading(id, reading) {
    let rawTimestamp = toNumber(reading.timestamp);
    let finalTimestamp = rawTimestamp === null ? null : (rawTimestamp > 9999999999 ? rawTimestamp : rawTimestamp * 1000);

    // XỬ LÝ MƯA LINH HOẠT HƠN: Nhận cả số (0/1) lẫn chữ
    let rainVal = reading.rain;
    if (typeof rainVal === 'number') {
        rainVal = rainVal > 0 ? "yes" : "no";
    } else if (typeof rainVal === 'string') {
        rainVal = rainVal.toLowerCase();
    } else {
        rainVal = "no"; // Mặc định nếu mất tín hiệu mưa thì cho là không mưa
    }

    return {
        id,
        temperature: toNumber(reading.temperature),
        humidity: toNumber(reading.humidity),
        rain: rainVal,
        pressure: toNumber(reading.pressure),
        timestamp: finalTimestamp
    };
}

// BỘ LỌC NỚI LỎNG HƠN
function isValidReading(reading) {
    // 1. Vẫn chặn gắt nếu không có thời gian
    if (!reading.timestamp || reading.timestamp <= 0) return false;

    // 2. Chỉ chặn nếu mất hoàn toàn Nhiệt độ HOẶC Độ ẩm (Áp suất có thể bị lỗi trả về 0 nên châm chước)
    if (reading.temperature === null || reading.humidity === null) return false;

    // 3. Ngưỡng thực tế (Nới rộng ra một chút)
    if (reading.temperature < -50 || reading.temperature > 100) return false; 
    if (reading.humidity < 0 || reading.humidity > 100) return false;
    
    // Tạm thời TẮT kiểm tra ngưỡng áp suất khắt khe, vì nhiều mạch ESP32 trả về 0 nếu chưa kết nối cảm biến
    // if (reading.pressure < 800 || reading.pressure > 1200) return false; 

    // Mưa đã được hàm normalize gánh, nên chắc chắn là "yes" hoặc "no" rồi, không cần check ở đây nữa.

    return true; 
}
// Hàm hỗ trợ lấy ngày hôm nay theo format YYYY-MM-DD để tìm đúng thư mục Firebase
function getTodayString() {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`; 
}

function subscribeToWeatherReadings({ limit = 15, onData, onError }) {
    if (!database) {
        if (typeof onError === "function") onError(new Error("Firebase config is incomplete"));
        return () => {};
    }

    let localHistory = [];
    const latestRef = ref(database, LATEST_PATH);
    
    // Tự động dò tìm thư mục ngày hôm nay (VD: "2026-04-16")
    const todayStr = getTodayString(); 
    //const todayStr = "2026-04-17";
    // Mẹo: Nếu bạn muốn test dữ liệu ngày cũ trong ảnh, đổi thành: const todayStr = "2026-04-04";
    
    const READINGS_PATH = `weather_stations/Weather_station_1/readings/${todayStr}`;
    
    // Tạo câu lệnh: "Vào nhánh ngày hôm nay, sắp xếp theo thời gian, lấy 15 điểm cuối cùng"
    const historyQuery = query(ref(database, READINGS_PATH), orderByKey(), limitToLast(limit));

    // BƯỚC 1: KÉO LỊCH SỬ THẬT TỪ FIREBASE 1 LẦN DUY NHẤT LÚC MỞ WEB
// BƯỚC 1: KÉO LỊCH SỬ THẬT TỪ FIREBASE 1 LẦN DUY NHẤT LÚC MỞ WEB
    get(historyQuery).then((snapshot) => {
        if (snapshot.exists()) {
            snapshot.forEach((childSnapshot) => {
                const reading = normalizeReading(childSnapshot.val().timestamp, childSnapshot.val());
                
                // THÊM KIỂM TRA TẠI ĐÂY: Chỉ push vào mảng nếu dữ liệu sạch
                if (isValidReading(reading)) {
                    localHistory.push(reading);
                }
            });
        }

        // BƯỚC 2: BẮT ĐẦU NGHE DỮ LIỆU REALTIME TỪ NHÁNH 'LATEST'
        onValue(latestRef, (snapshot) => {
            const data = snapshot.val();
            if (data) {
                const reading = normalizeReading(data.timestamp, data);

                // THÊM KIỂM TRA TẠI ĐÂY: Nếu dữ liệu mới rác, lập tức thoát (return) không xử lý tiếp
                if (!isValidReading(reading)) {
                    console.warn("Đã loại bỏ dữ liệu cảm biến không hợp lệ:", data);
                    return; 
                }

                // Kiểm tra xem điểm này đã vẽ chưa để chống trùng lặp dữ liệu nối tiếp
                const isDuplicate = localHistory.length > 0 && localHistory[localHistory.length - 1].timestamp === reading.timestamp;

                if (!isDuplicate) {
                    localHistory.push(reading);
                    if (localHistory.length > limit) {
                        localHistory.shift(); // Xóa bớt điểm cũ nhất nếu vượt quá 15 điểm
                    }
                }

                // Gửi dữ liệu ra cho file script.js vẽ
                if (typeof onData === "function") {
                    onData({
                        latest: reading,
                        history: localHistory
                    });
                }
            }
        }, (error) => {
            // ... (Giữ nguyên đoạn này)
            if (typeof onError === "function") onError(error);
        });

    }).catch((error) => {
        console.error("Lỗi khi tải lịch sử Firebase:", error);
    });

    return () => {}; // Dummy unsubscribe cho gọn
}

window.firebaseWeatherApi = {
    hasConfiguredFirebase: true,
    subscribeToWeatherReadings
};
import "./script.js";
