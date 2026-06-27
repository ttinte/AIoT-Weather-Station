import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-auth.js";
// 1. Thêm import thư viện Realtime Database
import { getDatabase, ref, get, child } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-database.js";

// Cấu hình Firebase
const firebaseConfig = {
    apiKey: "AIzaSyCEZa07j4UMaaHLhboLotGh8IFCs3bOciM",
    authDomain: "aiotnhom2-80e7a.firebaseapp.com",
    databaseURL: "https://aiotnhom2-80e7a-default-rtdb.firebaseio.com",
    projectId: "aiotnhom2-80e7a",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getDatabase(app); // Khởi tạo database

// 2. Kiểm tra đăng nhập và Lấy tên hiển thị
onAuthStateChanged(auth, async (user) => {
    if (!user) {
        // Nếu chưa đăng nhập thì đuổi về trang login
        window.location.href = "../login/login.html";
    } else {
        try {
            // Truy cập vào database, tìm nhánh users tương ứng với UID của người đang đăng nhập
            const dbRef = ref(db);
            const snapshot = await get(child(dbRef, `users/${user.uid}`));
            
            if (snapshot.exists()) {
                // Lấy ra họ tên (fullName) đã lưu lúc đăng ký
                const userData = snapshot.val();
                const fullName = userData.fullName;
                
                // Hiển thị tên thật lên màn hình
                document.getElementById('userNameDisplay').innerText = fullName;
                
                // Tạo Avatar xịn sò: Cắt lấy chữ cái đầu tiên của Tên để hiển thị
                // Ví dụ: "Hà Thúc Tín" -> Cắt lấy chữ "Tín" -> Lấy chữ "T"
                const nameParts = fullName.trim().split(' ');
                const lastName = nameParts[nameParts.length - 1]; 
                document.getElementById('userAvatarDisplay').innerText = lastName.charAt(0).toUpperCase();
            } else {
                document.getElementById('userNameDisplay').innerText = "Sinh viên UTE";
            }
        } catch (error) {
            console.error("Lỗi khi tải thông tin user:", error);
            document.getElementById('userNameDisplay').innerText = "Lỗi kết nối";
        }
    }
});

// Giữ nguyên hiệu ứng thu gọn menu
document.querySelector('.collapse-btn')?.addEventListener('click', () => {
  document.querySelector('.sidebar')?.classList.toggle('collapsed');
});