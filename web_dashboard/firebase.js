
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getDatabase, onValue, ref, get, query, limitToLast, orderByKey } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

const firebaseConfig = {
    apiKey: "AIzaSyBuot6MSqGdTXu19kEMfONUvIpid323Fj4",
    authDomain: "aiotnhom2.firebaseapp.com",
    databaseURL: "https://aiotnhom2-default-rtdb.firebaseio.com",
    projectId: "aiotnhom2",
    storageBucket: "aiotnhom2.appspot.com",
};

// Khởi tạo Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// ĐƯỜNG DẪN CHUẨN THEO FIREBASE CỦA NHÓM
const LATEST_PATH = "weather_stations/Weather_station_1/latest";

function toNumber(value, fallback = null) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeReading(id, reading) {
    return {
        id,
        temperature: toNumber(reading.temperature),
        humidity: toNumber(reading.humidity),
        rain: reading.rain, // GIỮ NGUYÊN ĐỂ NHẬN CHỮ "yes" HOẶC "no"
        pressure: toNumber(reading.pressure),
        timestamp: toNumber(reading.timestamp) > 9999999999 ? toNumber(reading.timestamp) : toNumber(reading.timestamp) * 1000
    };
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
    //const todayStr = getTodayString(); 
    const todayStr =  "2026-04-04";
    // Mẹo: Nếu bạn muốn test dữ liệu ngày cũ trong ảnh, đổi thành: const todayStr = "2026-04-04";
    
    const READINGS_PATH = `weather_stations/Weather_station_1/readings/${todayStr}`;
    
    // Tạo câu lệnh: "Vào nhánh ngày hôm nay, sắp xếp theo thời gian, lấy 15 điểm cuối cùng"
    const historyQuery = query(ref(database, READINGS_PATH), orderByKey(), limitToLast(limit));

    // BƯỚC 1: KÉO LỊCH SỬ THẬT TỪ FIREBASE 1 LẦN DUY NHẤT LÚC MỞ WEB
    get(historyQuery).then((snapshot) => {
        if (snapshot.exists()) {
            snapshot.forEach((childSnapshot) => {
                const reading = normalizeReading(childSnapshot.val().timestamp, childSnapshot.val());
                localHistory.push(reading);
            });
        }

        // BƯỚC 2: BẮT ĐẦU NGHE DỮ LIỆU REALTIME TỪ NHÁNH 'LATEST'
        onValue(latestRef, (snapshot) => {
            const data = snapshot.val();
            if (data) {
                const reading = normalizeReading(data.timestamp, data);

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
