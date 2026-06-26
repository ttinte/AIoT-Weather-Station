// --- IMPORT FIREBASE SDK ---
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, sendPasswordResetEmail } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-auth.js";
// 1. Thay đổi import: Chuyển sang dùng Realtime Database
import { getDatabase, ref, set, get, child } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-database.js";

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
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
// 2. Khởi tạo dịch vụ Realtime Database thay vì Firestore
const db = getDatabase(app);

// --- LẤY CÁC PHẦN TỬ HTML ---
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const forgotForm = document.getElementById('forgotForm');
const formTitle = document.getElementById('formTitle');

// Các nút chuyển đổi form
const btnShowRegister = document.getElementById('btnShowRegister');
const btnShowLogin = document.getElementById('btnShowLogin');
const linkForgotPass = document.getElementById('linkForgotPass');
const linkBackToLogin = document.getElementById('linkBackToLogin');

let isLogin = true;

// --- 1. CHUYỂN ĐỔI GIỮA ĐĂNG NHẬP VÀ ĐĂNG KÝ ---
btnShowRegister.addEventListener('click', function(e) {
    e.preventDefault();
    loginForm.classList.add('hidden');
    forgotForm.classList.add('hidden');
    registerForm.classList.remove('hidden');
    formTitle.innerText = "ĐĂNG KÝ";
});

// Chuyển sang Đăng nhập
btnShowLogin.addEventListener('click', function(e) {
    e.preventDefault();
    registerForm.classList.add('hidden');
    forgotForm.classList.add('hidden');
    loginForm.classList.remove('hidden');
    formTitle.innerText = "ĐĂNG NHẬP";
});

// --- 2. XỬ LÝ ĐĂNG KÝ LÊN FIREBASE ---
registerForm.addEventListener('submit', async function(event) {
    event.preventDefault(); 
    
    const fullName = document.getElementById('regName').value.trim();
    const email = document.getElementById('regEmail').value.trim();
    const password = document.getElementById('regPass').value;

    // Kiểm tra đuôi email trường
    if (!email.endsWith("@student.hcmute.edu.vn")) {
        alert("Vui lòng sử dụng email sinh viên trường để đăng ký!");
        return;
    }

    if (password.length < 6) {
        alert("Mật khẩu phải từ 6 ký tự trở lên!");
        return;
    }

    try {
        // Bước A: Tạo tài khoản trên Firebase Authentication
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Bước B: Lưu thông tin bổ sung (Họ tên) vào Realtime Database
        // 3. Sử dụng cấu trúc set() và ref() của Realtime DB
        await set(ref(db, 'users/' + user.uid), {
            fullName: fullName,
            email: email,
            createdAt: new Date().toISOString()
        });

        alert("Đăng ký tài khoản sinh viên thành công!");
        registerForm.reset(); 
        btnShowLogin.click(); // Sửa lỗi gọi sai tên biến nút bấm
        
    } catch (error) {
        console.error(error);
        if (error.code === 'auth/email-already-in-use') {
            alert("Email này đã được đăng ký tài khoản trước đó!");
        } else {
            alert("Đăng ký thất bại: " + error.message);
        }
    }
});

// --- 3. XỬ LÝ ĐĂNG NHẬP BẰNG FIREBASE ---
loginForm.addEventListener('submit', async function(event) {
    event.preventDefault(); 
    const emailInput = document.getElementById('loginUser').value.trim();
    const passInput = document.getElementById('loginPass').value;

    try {
        // Gửi yêu cầu đăng nhập lên Firebase
        const userCredential = await signInWithEmailAndPassword(auth, emailInput, passInput);
        const user = userCredential.user;

        // Lấy thử họ tên sinh viên từ Realtime Database ra để chào mừng
        // 4. Sử dụng cấu trúc get() và child() của Realtime DB
        const dbRef = ref(db);
        const snapshot = await get(child(dbRef, `users/${user.uid}`));
        
        if (snapshot.exists()) {
            alert(`Đăng nhập thành công! Chào sinh viên: ${snapshot.val().fullName}`);
        } else {
            alert("Đăng nhập thành công!");
        }

        // Chuyển hướng trang sau khi đăng nhập thành công
        window.location.href = "../overview/overview.html"; 
        
    } catch (error) {
        console.error(error);
        alert("Sai tài khoản hoặc mật khẩu! Vui lòng kiểm tra lại.");
    }
});

// --- 4. GIAO DIỆN QUÊN MẬT KHẨU ---
linkForgotPass.addEventListener('click', function(e) {
    e.preventDefault();
    loginForm.classList.add('hidden');
    registerForm.classList.add('hidden');
    forgotForm.classList.remove('hidden');
    formTitle.innerText = "KHÔI PHỤC MẬT KHẨU";
});

linkBackToLogin.addEventListener('click', function(e) {
    e.preventDefault();
    forgotForm.classList.add('hidden');
    loginForm.classList.remove('hidden');
    formTitle.innerText = "ĐĂNG NHẬP";
});

// XỬ LÝ GỬI LINK KHÔI PHỤC MẬT KHẨU
const btnSendResetLink = document.getElementById('btnSendResetLink');

btnSendResetLink.addEventListener('click', async function() {
    const resetEmail = document.getElementById('forgotEmail').value.trim();

    if (!resetEmail) {
        alert("Vui lòng nhập email sinh viên của bạn!");
        return;
    }

    try {
        // 1. Khóa nút bấm ngay lập tức để tránh click đúp
        btnSendResetLink.disabled = true;
        btnSendResetLink.classList.add('btn-disabled');
        btnSendResetLink.innerText = "Đang gửi...";

        // 2. Gọi hàm Firebase gửi mail
// Cấu hình đường link quay về web sau khi đổi mật khẩu xong
const actionCodeSettings = {
    url: 'https://aiotnhom2-80e7a.web.app/login/login.html' 
};

// Truyền thêm cấu hình này vào lệnh gửi mail
await sendPasswordResetEmail(auth, resetEmail, actionCodeSettings);
        alert(`Đã gửi liên kết! Vui lòng kiểm tra hộp thư đến/thư rác của email: ${resetEmail}`);
        document.getElementById('forgotEmail').value = ""; // Xóa trắng ô nhập
        
        // 3. Bắt đầu đếm ngược 60s
        let timeLeft = 60;
        btnSendResetLink.innerText = `Gửi lại sau ${timeLeft}s`;

        const timer = setInterval(() => {
            timeLeft--;
            btnSendResetLink.innerText = `Gửi lại sau ${timeLeft}s`;

            if (timeLeft <= 0) {
                // Hết 60s thì mở khóa nút lại như cũ
                clearInterval(timer);
                btnSendResetLink.disabled = false;
                btnSendResetLink.classList.remove('btn-disabled');
                btnSendResetLink.innerText = "Gửi liên kết khôi phục";
            }
        }, 1000);

    } catch (error) {
        console.error(error);
        // Nếu có lỗi (ví dụ sai email) thì mở khóa nút ngay lập tức
        btnSendResetLink.disabled = false;
        btnSendResetLink.classList.remove('btn-disabled');
        btnSendResetLink.innerText = "Gửi liên kết khôi phục";
        
        if (error.code === 'auth/user-not-found') {
            alert("Email này chưa được đăng ký trên hệ thống!");
        } else if (error.code === 'auth/invalid-email') {
            alert("Định dạng email không hợp lệ!");
        } else if (error.code === 'auth/too-many-requests') {
            alert("Hệ thống đang quá tải yêu cầu. Vui lòng chờ một lát rồi thử lại!");
        } else {
            alert("Lỗi: " + error.message);
        }
    }
});