import pandas as pd

CSV_PATH = "raw/weather.csv"
OUTPUT_PATH = "weather_fixed_pro.csv"

print("Fixing data using Seasonal Imputation + remove gap...")

try:
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    df["diff"] = df["datetime"].diff().dt.total_seconds()
    
    GAP_THRESHOLD = 6 * 3600 
    
    gap_idx = df[df["diff"] > GAP_THRESHOLD].index
    
    if len(gap_idx) > 0:
        cut_idx = gap_idx[0]
        print(f"Found gap at index {cut_idx}, cutting data before it...")
        df = df.loc[cut_idx:].copy()
        
    df = df.drop(columns=["diff"])
    df = df.set_index("datetime").sort_index()
    df = df.drop_duplicates()

    full_time_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='10min'
    )
    
    df_full = df.reindex(full_time_range)
    
    # SEASONAL IMPUTATION
    df_filled = df_full.fillna(df_full.shift(144)) # 1 ngày
    df_filled = df_filled.fillna(df_full.shift(288)) # 2 ngày
    
    df_filled = df_filled.reset_index()
    df_filled.rename(columns={'index': 'datetime'}, inplace=True)
    
    df_filled = df_filled.dropna(subset=['temperature', 'humidity'])
    
    df_filled.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Fixed Data saved: '{OUTPUT_PATH}'")
    print(f"Total rows: {len(df_filled)}")

except Exception as e:
    print(f"Err: {e}")