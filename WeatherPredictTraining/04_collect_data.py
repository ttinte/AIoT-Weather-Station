import pandas as pd
import requests

# ==========================================
# 1. TẢI DỮ LIỆU TỪ NGUỒN UY TÍN (1 GIỜ/LẦN)
# ==========================================
lat, lon = 10.902, 106.762
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date=2026-03-29&end_date=2026-04-04&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation&timezone=Asia%2FBangkok"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    df_ref = pd.DataFrame({
        'datetime': pd.to_datetime(data['hourly']['time']),
        'temp_ref': data['hourly']['temperature_2m'],
        'humid_ref': data['hourly']['relative_humidity_2m'],
        'press_ref': data['hourly']['surface_pressure'],
        'rain_ref': data['hourly']['precipitation']
    })
    print("Đã tải xong dữ liệu chuẩn (API).")
else:
    print("Lỗi tải API!")

# ==========================================
# 2. XỬ LÝ DỮ LIỆU CỦA BẠN (LẤY GIÁ TRỊ ĐẦU TIÊN MỖI GIỜ)
# ==========================================
# Đọc file dữ liệu của bạn
df_my_data = pd.read_csv('weather_fixed_pro.csv')

# Chuyển cột datetime sang định dạng thời gian và đặt làm index
df_my_data['datetime'] = pd.to_datetime(df_my_data['datetime'])
df_my_data.set_index('datetime', inplace=True)

# Lấy giá trị ĐẦU TIÊN ('first') xuất hiện trong mỗi khung 1 giờ ('1H')
df_my_hourly = df_my_data.resample('1h').agg({
    'temperature': 'first',
    'humidity': 'first',
    'pressure': 'first',
    'rain': 'first'
}).reset_index()

# Đổi tên cột để dễ phân biệt
df_my_hourly.rename(columns={
    'temperature': 'temp_my',
    'humidity': 'humid_my',
    'pressure': 'press_my',
    'rain': 'rain_my'
}, inplace=True)
print("Đã lọc xong dữ liệu tự đo (giữ lại giá trị đầu tiên của mỗi giờ).")

# ==========================================
# 3. GHÉP 2 BẢNG LẠI ĐỂ SO SÁNH TRỰC TIẾP
# ==========================================
# Ghép dựa trên cột thời gian (datetime)
df_comparison = pd.merge(df_my_hourly, df_ref, on='datetime', how='inner')

# Lưu kết quả ra file mới 
df_comparison.to_csv('comparison_result.csv', index=False)
print("Thành công! Hãy mở file 'comparison_result.csv' để xem kết quả so sánh.")