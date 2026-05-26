// Đợi cho HTML tải xong toàn bộ mới chạy script
document.addEventListener('DOMContentLoaded', () => {
    
    // 1. KẾT NỐI VỚI CÁC THÀNH PHẦN TRÊN GIAO DIỆN (DOM Elements)
    const tempInput = document.getElementById('temp-threshold');
    const humidInput = document.getElementById('humid-threshold');
// Thay đổi dòng: const buzzerToggle = document.getElementById('buzzer-toggle');
// Thành dòng này:
const webAlertToggle = document.getElementById('web-alert-toggle');
    const themeToggle = document.getElementById('theme-toggle');
    const chartLimit = document.getElementById('chart-limit');
    const stationName = document.getElementById('station-name');

    const saveBtn = document.getElementById('save-btn');
    const resetBtn = document.getElementById('reset-btn');

    // 2. CẤU HÌNH MẶC ĐỊNH CỦA HỆ THỐNG
const defaultSettings = {
        tempThreshold: 35,
        humidThreshold: 40,
        webAlert: true, // Đã đổi tên ở đây
        theme: 'light',
        chartLimit: '20',
        stationName: 'Trạm Thời Tiết Nhóm 2 - HCMUTE',

    };

    // 3. HÀM TẢI CÀI ĐẶT TỪ BỘ NHỚ TRÌNH DUYỆT (LocalStorage)
    function loadSettings() {
        // Lấy dữ liệu đã lưu, nếu chưa có thì dùng cấu hình mặc định
        const savedData = localStorage.getItem('iotWeatherSettings');
        const settings = savedData ? JSON.parse(savedData) : defaultSettings;

        // Đổ dữ liệu vào các ô input trên giao diện
        tempInput.value = settings.tempThreshold;
        humidInput.value = settings.humidThreshold;
        webAlertToggle.checked = settings.webAlert;
        themeToggle.checked = (settings.theme === 'dark');
        chartLimit.value = settings.chartLimit;
        stationName.value = settings.stationName;


        // Kích hoạt giao diện Tối/Sáng theo dữ liệu đã lưu
        applyTheme(settings.theme);
    }

    // 4. HÀM XỬ LÝ CHUYỂN ĐỔI GIAO DIỆN DARK/LIGHT MODE
    function applyTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }

    // Bắt sự kiện khi người dùng gạt nút Dark Mode (Giao diện đổi ngay lập tức)
    themeToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            applyTheme('dark');
        } else {
            applyTheme('light');
        }
    });

    // 5. BẮT SỰ KIỆN KHI BẤM NÚT "LƯU CẤU HÌNH"
    saveBtn.addEventListener('click', () => {
        // Gom tất cả thông số hiện tại vào một Object
        const currentSettings = {
            tempThreshold: tempInput.value,
            humidThreshold: humidInput.value,
            webAlert: webAlertToggle.checked,
            theme: themeToggle.checked ? 'dark' : 'light',
            chartLimit: chartLimit.value,
            stationName: stationName.value,
        };

        // Lưu Object này vào LocalStorage dưới dạng chuỗi JSON
        localStorage.setItem('iotWeatherSettings', JSON.stringify(currentSettings));
        
        // Hiển thị thông báo thành công (Có thể thay bằng thư viện SweetAlert cho đẹp hơn)
        alert('✅ Đã lưu cấu hình hệ thống thành công! Các trang khác sẽ tự động cập nhật.');
    });

  // 6. BẮT SỰ KIỆN KHI BẤM NÚT "KHÔI PHỤC MẶC ĐỊNH"
    resetBtn.addEventListener('click', () => {
        const confirmReset = confirm('⚠️ Bạn có chắc chắn muốn khôi phục toàn bộ về cài đặt gốc không?');
        
        if (confirmReset) {
            // Ghi đè bộ nhớ bằng cấu hình mặc định
            localStorage.setItem('iotWeatherSettings', JSON.stringify(defaultSettings));
            
            // Tải lại giao diện
            loadSettings();
            
            alert('🔄 Đã khôi phục cài đặt mặc định.');
        }
    }); // <--- THÊM DẤU ĐÓNG NÀY ĐỂ KẾT THÚC SỰ KIỆN NÚT RESET

    // =========================================
    // XỬ LÝ POPUP MODAL THÀNH VIÊN NHÓM
    // =========================================
    const teamModal = document.getElementById('team-modal');
    const memberItems = document.querySelectorAll('.member-item');
    const closeModalBtn = document.querySelector('.close-modal');

    // Các phần tử thông tin trong Modal
    const modalAvatar = document.getElementById('modal-avatar');
    const modalName = document.getElementById('modal-name');
    const modalMssv = document.getElementById('modal-mssv');
    const modalRole = document.getElementById('modal-role');
    const modalFb = document.getElementById('modal-fb');

    // Bắt sự kiện click vào từng thành viên
    memberItems.forEach(item => {
        item.addEventListener('click', () => {
            // Đọc các thuộc tính "data-" từ HTML
            const name = item.getAttribute('data-name');
            const mssv = item.getAttribute('data-mssv');
            const role = item.getAttribute('data-role');
            const fb = item.getAttribute('data-fb');
            const avatar = item.getAttribute('data-avatar');

            // Gán dữ liệu vào bảng Modal
            modalName.textContent = name;
            modalMssv.textContent = `MSSV: ${mssv}`;
            modalRole.textContent = role;
            modalFb.setAttribute('href', fb);
            modalAvatar.setAttribute('src', avatar);

            // Hiển thị Modal lên màn hình (flex giúp căn giữa)
            teamModal.style.display = 'flex';
        });
    });

    // Bấm vào nút dấu "X" để đóng modal
    closeModalBtn.addEventListener('click', () => {
        teamModal.style.display = 'none';
    });

    // Bấm ra ngoài vùng hộp thoại (vùng nền mờ) cũng tự đóng modal
    window.addEventListener('click', (e) => {
        if (e.target === teamModal) {
            teamModal.style.display = 'none';
        }
    });

    // 7. CHẠY HÀM TẢI CÀI ĐẶT NGAY KHI MỞ TRANG
    loadSettings();
});
