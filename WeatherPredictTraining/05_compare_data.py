import pandas as pd
import matplotlib.pyplot as plt

# Đọc file dữ liệu đã ghép
df = pd.read_csv('comparison_result.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Thiết lập kích thước biểu đồ
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 1. Biểu đồ Nhiệt độ
ax1.plot(df['datetime'], df['temp_my'], label='Dữ liệu tự đo', color='red', marker='o', markersize=4)
ax1.plot(df['datetime'], df['temp_ref'], label='Dữ liệu API', color='blue', linestyle='--')
ax1.set_ylabel('Nhiệt độ (°C)')
ax1.set_title('So sánh Nhiệt độ (Temperature)')
ax1.legend()
ax1.grid(True)

# 2. Biểu đồ Độ ẩm
ax2.plot(df['datetime'], df['humid_my'], label='Dữ liệu tự đo', color='green', marker='o', markersize=4)
ax2.plot(df['datetime'], df['humid_ref'], label='Dữ liệu API', color='orange', linestyle='--')
ax2.set_ylabel('Độ ẩm (%)')
ax2.set_title('So sánh Độ ẩm (Humidity)')
ax2.legend()
ax2.grid(True)

# 3. Biểu đồ Áp suất
ax3.plot(df['datetime'], df['press_my'], label='Dữ liệu tự đo', color='purple', marker='o', markersize=4)
ax3.plot(df['datetime'], df['press_ref'], label='Dữ liệu API', color='brown', linestyle='--')
ax3.set_ylabel('Áp suất (hPa)')
ax3.set_title('So sánh Áp suất (Pressure)')
ax3.legend()
ax3.grid(True)

plt.xlabel('Thời gian')
plt.tight_layout()
plt.show()