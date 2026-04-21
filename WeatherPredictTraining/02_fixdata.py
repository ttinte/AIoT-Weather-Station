import pandas as pd

CSV_PATH = "raw/weather.csv"
OUTPUT_PATH = "weather_fixed_pro.csv"

print("Fixing data using Seasonal Imputation...")

try:
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    

    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10min')
    df_full = df.reindex(full_time_range)

    df_filled = df_full.fillna(df_full.shift(144))
    df_filled = df_filled.fillna(df_full.shift(288))
    df_filled = df_filled.reset_index()
    df_filled.rename(columns={'index': 'datetime'}, inplace=True)

    df_filled = df_filled.dropna(how='all', subset=['temperature', 'humidity'])
    df_filled.to_csv(OUTPUT_PATH, index=False)

    print(f"Fixed Data. saved: '{OUTPUT_PATH}'")

except Exception as e:
    print(f"Err: {e}")
