import json
import pandas as pd

file_path = 'rawdata.json'
print("Đang xử lý data, lấy 1 mẫu mỗi 10 phút và vá lỗi rớt mạng...")

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    readings = data['weather_stations']['Weather_station_1']['readings']

    for date_key, timestamps_data in readings.items():
        for ts_key, metrics in timestamps_data.items():
            raw_timestamp_ms = int(ts_key)
            record = {
                'timestamp_ms': raw_timestamp_ms,
                'temperature': metrics.get('temperature', None),
                'humidity': metrics.get('humidity', None),
                'pressure': metrics.get('pressure', None),
                'rain': metrics.get('rain', None)
            }
            records.append(record)

    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.drop(columns=['timestamp_ms'], inplace=True)
    df.set_index('datetime', inplace=True)

    # 1. Gom 10 phút sử dụng first()
    df_10min = df.resample('10min').first()

    df_10min = df_10min.ffill(limit=2) 
    
    df_10min = df_10min.dropna(how='all')

    df_10min = df_10min.reset_index()

    output_file = 'raw/weather_fixed.csv'
    df_10min.to_csv(output_file, index=False)
    
    print(f"Done {len(df_10min)} ")

except Exception as e:
    print(f"Ngoại lệ: {e}")
