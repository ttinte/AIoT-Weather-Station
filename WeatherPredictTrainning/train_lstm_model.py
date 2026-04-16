import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- BƯỚC 1: ĐỌC DỮ LIỆU ---
print("Đang đọc dữ liệu từ file CSV...")
df = pd.read_csv('weather_fixed_pro.csv')

# Khai báo tên cột chính xác chữ thường (như bạn đã check bằng lệnh head)
features = ['temperature', 'humidity', 'pressure', 'rain']
dataset = df[features].values

# --- BƯỚC 2: LÀM SẠCH DỮ LIỆU ---
# Dùng ffill để điền giá trị của ô trước đó vào các ô trống (NaN) nếu có
print("Đang làm sạch dữ liệu...")
df_clean = pd.DataFrame(dataset).ffill().values

# --- BƯỚC 3: CHUẨN HÓA DỮ LIỆU ---
# Đưa tất cả các thông số (Nhiệt độ, Độ ẩm, Áp suất...) về thang đo từ 0 đến 1
print("Đang chuẩn hóa dữ liệu...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_clean)

joblib.dump(scaler, 'weather_scaler.pkl')
print("=> Đã lưu file weather_scaler.pkl thành công!")

# --- BƯỚC 4: TẠO CỬA SỔ TRƯỢT (DỰ BÁO XA 1 TIẾNG) ---
# --- BƯỚC 4: TẠO CỬA SỔ TRƯỢT (TẦM NHÌN 6 TIẾNG, DỰ BÁO 1 TIẾNG) ---
def create_sliding_window(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        # Input X: Gom toàn bộ dữ liệu của n mẫu quá khứ
        X.append(data[i : (i + window_size)])
        # Label Y: Lấy điểm ở tương lai cách đó 1 tiếng
        y.append(data[i + window_size + forecast_horizon - 1])
        
    return np.array(X), np.array(y)

# ĐỘ RỘNG CỬA SỔ: 36 mẫu (Tương đương 6 tiếng lịch sử)
WINDOW_SIZE = 36

# ĐỘ TRỄ DỰ BÁO: 2 bước (Tương đương 1 tiếng tương lai)
FORECAST_HORIZON = 2 

X, y = create_sliding_window(scaled_data, WINDOW_SIZE, FORECAST_HORIZON)

# --- BƯỚC 5: CHIA TẬP TRAIN & TEST ---
# Chia 80% dữ liệu để huấn luyện, 20% để kiểm thử độ chính xác
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- IN KẾT QUẢ ĐỂ KIỂM TRA ---
print("\n--- HOÀN TẤT TIỀN XỬ LÝ ---")
print("Kích thước X_train (Số mẫu, Window Size, Số thông số):", X_train.shape)
print("Kích thước y_train (Số mẫu, Số thông số):", y_train.shape)
print("Kích thước X_test:", X_test.shape)
print("Kích thước y_test:", y_test.shape)
# --- BƯỚC 6: XÂY DỰNG MÔ HÌNH LSTM ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("\nĐang khởi tạo kiến trúc mạng LSTM...")
model = Sequential()

# Lớp LSTM với 32 nơ-ron để trích xuất đặc trưng thời gian
model.add(LSTM(units=32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

# Lớp Dropout ngắt ngẫu nhiên 20% nơ-ron để tránh học vẹt (overfitting)
model.add(Dropout(0.2))

# Lớp Dense cuối cùng với 4 nơ-ron đầu ra (dự đoán 4 thông số cùng lúc)
model.add(Dense(units=4))

# Biên dịch mô hình với thuật toán tối ưu Adam và hàm mất mát MSE
model.compile(optimizer='adam', loss='mean_squared_error')

# --- BƯỚC 7: HUẤN LUYỆN (TRAINING) ---
print("\n--- BẮT ĐẦU HUẤN LUYỆN MODEL ---")
# Cho AI học 50 vòng (epochs), mỗi vòng nhìn 16 mẫu dữ liệu (batch_size)
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=16, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# --- BƯỚC 8: LƯU MÔ HÌNH ---
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Lưu model với tên mới để không đè mất cái model 10 phút hôm trước
model.save('weather_lstm_model.keras')
print("\nĐã huấn luyện xong! Mô hình dự báo 1 TIẾNG được lưu tại 'weather_lstm_model.keras'")

# --- BƯỚC 9: ĐÁNH GIÁ MÔ HÌNH ---
print("\n--- ĐANG ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---")
# Cho AI dự đoán thử trên tập dữ liệu kiểm thử (X_test)model
predictions = model.predict(X_test)

# Dịch ngược dữ liệu từ [0, 1] về lại giá trị vật lý thực tế
y_test_actual = scaler.inverse_transform(y_test)
predictions_actual = scaler.inverse_transform(predictions)

# Tính toán sai số MAE và RMSE cho toàn bộ các thông số
mae = mean_absolute_error(y_test_actual, predictions_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))

print(f"Sai số tuyệt đối trung bình (MAE): {mae:.2f}")
print(f"Sai số bình phương trung bình căn (RMSE): {rmse:.2f}")

# --- BƯỚC 10 (NÂNG CẤP): VẼ CẢ 4 BIỂU ĐỒ ---
import matplotlib.pyplot as plt

print("\nĐang vẽ biểu đồ cho cả 4 thông số...")

# XỬ LÝ RIÊNG CHO CỘT MƯA (Cột số 3, vì mảng đếm từ 0)
# Ép xác suất dự đoán thành giá trị nhị phân 0 hoặc 1 (Ngưỡng 0.5)
predictions_actual[:, 3] = np.where(predictions_actual[:, 3] > 0.5, 1, 0)

# Tạo một bức ảnh lớn, chia làm 4 hàng (4 khung hình con)
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

# Tên các thông số để gắn lên biểu đồ
feature_names = ['Nhiệt độ (°C)', 'Độ ẩm (%)', 'Áp suất (hPa)', 'Trạng thái Mưa (0: Không, 1: Có)']
colors = ['red', 'blue', 'green', 'purple']

for i in range(4):
    # Vẽ đường thực tế
    axes[i].plot(y_test_actual[:, i], label=f'{feature_names[i]} Thực tế', color='black', linewidth=1.5)
    # Vẽ đường dự đoán
    axes[i].plot(predictions_actual[:, i], label=f'{feature_names[i]} Dự đoán', color=colors[i], linestyle='dashed', linewidth=1.5)
    
    axes[i].set_title(f'Thực tế vs Dự đoán: {feature_names[i]}')
    axes[i].set_xlabel('Thời gian (Các bước kiểm thử)')
    axes[i].set_ylabel(feature_names[i])
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

# Tự động căn chỉnh khoảng cách giữa 4 khung hình cho đẹp
plt.tight_layout()

# Lưu thành file ảnh mới
plt.savefig('Predict_plot.png')
print("saved file 'Predict_plot.png'!")
