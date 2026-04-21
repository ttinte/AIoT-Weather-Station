import os
import pandas as pd

CSV_PATH = "raw/weather.csv"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

EXPECTED_INTERVAL_SEC = 600 
TEMP_MIN = 0
TEMP_MAX = 60 

try:
    df = pd.read_csv(CSV_PATH)
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["dt_sec"] = df["datetime"].diff().dt.total_seconds()

    gaps = df[df["dt_sec"] > EXPECTED_INTERVAL_SEC * 1.5]
    
    outliers = df[(df["temperature"] < TEMP_MIN) | (df["temperature"] > TEMP_MAX)]

    print("  Data reports:")
    print(f" Samples: {len(df)}")
    print(f" Time:: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f" Gaps) {len(gaps)}")
    print(f" Outliers): {len(outliers)}")
    print("\n🔍 Detail:")
    print(gaps[['datetime', 'dt_sec']])

    gaps.to_csv(os.path.join(LOG_DIR, "gaps.csv"), index=False)
    outliers.to_csv(os.path.join(LOG_DIR, "outliers.csv"), index=False)

    print(f"\n Logs Done! Saved: '{LOG_DIR}'")

except FileNotFoundError:
    print(f"file '{CSV_PATH}' not found")
except KeyError as e:
    print(f"Not found column {e}.")
