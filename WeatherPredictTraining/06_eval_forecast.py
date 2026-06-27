# -*- coding: utf-8 -*-
# Đánh giá vận hành thực tế (operational) của mô hình LSTM dự báo thời tiết.
# Dữ liệu lấy từ exp/ (export Firebase): forecast do ai_server sinh (dự báo trước 20 phút)
# và readings là số đo cảm biến thực tế. Ghép cặp forecast <-> readings để đo sai số thật.
# Baselines: Persistence (naive) và Open-Meteo (tham chiếu ngoài).
# Đầu ra: hình PNG + bảng .tex (booktabs) + CSV + số liệu in ra console.

import os
import json
import warnings

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# --- Đường dẫn ---
HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "exp")
FORECAST_JSON = os.path.join(EXP_DIR, "aiotnhom2-80e7a-default-rtdb-forecast-export.json")
READINGS_JSON = os.path.join(EXP_DIR, "aiotnhom2-80e7a-default-rtdb-readings-export.json")

RESULTS_DIR = os.path.join(HERE, "exp_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OM_CACHE = os.path.join(HERE, "exp_openmeteo_cache.csv")

# --- Cấu hình ---
# Vị trí trạm (giống script 04)
LAT, LON = 10.902, 106.762
MATCH_TOL = 60  # dung sai ghép cặp theo thời gian (giây), readings ~60s/mẫu

# (cột, tên hiển thị, đơn vị)
VARS = [
    ("temperature", "Nhiệt độ", "°C"),
    ("humidity", "Độ ẩm", "%"),
    ("pressure", "Áp suất", "hPa"),
]

# Font hỗ trợ tiếng Việt
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ---------------------------------------------------------------------------
# 1. Đọc dữ liệu Firebase (cấu trúc lồng {ngày: {ms: {...}}})
# ---------------------------------------------------------------------------
def load_firebase_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for _day, recs in data.items():
        if not isinstance(recs, dict):
            continue
        for _ms, rec in recs.items():
            if isinstance(rec, dict) and "timestamp" in rec:
                rows.append(rec)
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


# ---------------------------------------------------------------------------
# 2. Ghép forecast <-> readings: actual (tại thời điểm đích) + persistence (tại gốc)
# ---------------------------------------------------------------------------
def build_matched(fc, rd):
    fc = fc.copy()
    fc["fid"] = np.arange(len(fc))

    read_cols = ["timestamp"] + [v for v, _, _ in VARS]
    rd_small = rd[read_cols].copy()

    # actual: reading gần nhất quanh forecast.timestamp (giá trị thực ở 20' sau)
    rd_actual = rd_small.rename(columns={v: f"actual_{v}" for v, _, _ in VARS})
    m_actual = pd.merge_asof(
        fc.sort_values("timestamp"),
        rd_actual.sort_values("timestamp"),
        on="timestamp", direction="nearest", tolerance=MATCH_TOL,
    )

    # persistence: reading tại source_timestamp (giá trị "hiện tại" lúc dự báo)
    rd_pers = rd_small.rename(
        columns={"timestamp": "pers_ts", **{v: f"pers_{v}" for v, _, _ in VARS}}
    )
    m_pers = pd.merge_asof(
        fc.sort_values("source_timestamp"),
        rd_pers.sort_values("pers_ts"),
        left_on="source_timestamp", right_on="pers_ts",
        direction="nearest", tolerance=MATCH_TOL,
    )[["fid"] + [f"pers_{v}" for v, _, _ in VARS]]

    df = m_actual.merge(m_pers, on="fid", how="left")

    # forecast (LSTM) đổi tên cho rõ
    df = df.rename(columns={v: f"lstm_{v}" for v, _, _ in VARS})

    # giữ các hàng ghép đủ actual + persistence
    need = [f"actual_{v}" for v, _, _ in VARS] + [f"pers_{v}" for v, _, _ in VARS]
    df = df.dropna(subset=need).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Tính các chỉ số đánh giá
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    me = np.mean(y_pred - y_true)  # bias
    r2 = r2_score(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "ME": me, "R2": r2, "r": r}


def compute_metric_table(df):
    rows = []
    for v, name, unit in VARS:
        yt = df[f"actual_{v}"]
        for model, pred_col in [("LSTM", f"lstm_{v}"), ("Persistence", f"pers_{v}")]:
            m = metrics(yt, df[pred_col])
            rows.append({"Biến": f"{name} ({unit})", "Mô hình": model, **m})
    return pd.DataFrame(rows)


def compute_skill(df):
    rows = []
    for v, name, unit in VARS:
        yt = df[f"actual_{v}"]
        rmse_lstm = np.sqrt(mean_squared_error(yt, df[f"lstm_{v}"]))
        rmse_pers = np.sqrt(mean_squared_error(yt, df[f"pers_{v}"]))
        ss = 1 - rmse_lstm / rmse_pers if rmse_pers > 0 else np.nan
        rows.append({
            "Biến": f"{name} ({unit})",
            "RMSE_LSTM": rmse_lstm,
            "RMSE_Persistence": rmse_pers,
            "Skill Score": ss,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Open-Meteo (tham chiếu ngoài, độ phân giải giờ) - tùy chọn
# ---------------------------------------------------------------------------
def fetch_openmeteo(start_date, end_date):
    if os.path.exists(OM_CACHE):
        print(f"[Open-Meteo] Dùng cache: {OM_CACHE}")
        df = pd.read_csv(OM_CACHE, parse_dates=["dt"])
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        return df
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation"
        "&timezone=GMT"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        h = resp.json()["hourly"]
        df = pd.DataFrame({
            "dt": pd.to_datetime(h["time"], utc=True),
            "om_temperature": h["temperature_2m"],
            "om_humidity": h["relative_humidity_2m"],
            "om_pressure": h["surface_pressure"],
        })
        df.to_csv(OM_CACHE, index=False)
        print(f"[Open-Meteo] Đã tải {len(df)} mốc giờ, cache vào {OM_CACHE}")
        return df
    except Exception as e:
        print(f"[Open-Meteo] Bỏ qua (lỗi: {e})")
        return None


def openmeteo_compare(rd, om):
    """So sánh hourly: actual (đo) vs Open-Meteo cho từng biến."""
    rd_h = rd.set_index("dt")[[v for v, _, _ in VARS]].resample("1h").mean()
    rd_h.columns = [f"actual_{v}" for v, _, _ in VARS]
    om_h = om.set_index("dt")
    merged = rd_h.join(om_h, how="inner").dropna()
    rows = []
    for v, name, unit in VARS:
        m = metrics(merged[f"actual_{v}"], merged[f"om_{v}"])
        rows.append({"Biến": f"{name} ({unit})", "Mô hình": "Open-Meteo", **m})
    return pd.DataFrame(rows), merged


# ---------------------------------------------------------------------------
# 5. Vẽ hình
# ---------------------------------------------------------------------------
def plot_timeseries(df, om_merged=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    for ax, (v, name, unit) in zip(axes, VARS):
        ax.plot(df["dt"], df[f"actual_{v}"], color="black", lw=1.3, label="Thực tế")
        ax.plot(df["dt"], df[f"lstm_{v}"], color="tab:red", lw=1.1,
                ls="--", label="LSTM dự báo (20')")
        if om_merged is not None and f"om_{v}" in om_merged:
            ax.plot(om_merged.index, om_merged[f"om_{v}"], color="tab:blue",
                    lw=1.0, ls=":", marker=".", ms=4, label="Open-Meteo (giờ)")
        ax.set_title(f"{name} ({unit}): Thực tế vs Dự báo")
        ax.set_ylabel(f"{name} ({unit})")
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Thời gian (UTC)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_timeseries.png"), dpi=200)
    plt.close(fig)


def plot_scatter(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (v, name, unit) in zip(axes, VARS):
        yt, yp = df[f"actual_{v}"], df[f"lstm_{v}"]
        ax.scatter(yt, yp, s=8, alpha=0.4, color="tab:red")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        r2 = r2_score(yt, yp)
        ax.set_title(f"{name} ({unit})\n$R^2$ = {r2:.3f}")
        ax.set_xlabel("Thực tế")
        ax.set_ylabel("LSTM dự báo")
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_scatter.png"), dpi=200)
    plt.close(fig)


def plot_residual_hist(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (v, name, unit) in zip(axes, VARS):
        res = df[f"lstm_{v}"] - df[f"actual_{v}"]
        ax.hist(res, bins=40, color="tab:red", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.axvline(res.mean(), color="tab:blue", ls="--", lw=1.2,
                   label=f"Bias = {res.mean():.2f}")
        ax.set_title(f"Sai số {name} ({unit})")
        ax.set_xlabel(f"Dự báo − Thực tế ({unit})")
        ax.set_ylabel("Tần suất")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_residual_hist.png"), dpi=200)
    plt.close(fig)


def plot_metrics_bar(mt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_names = [("MAE", "MAE"), ("RMSE", "RMSE")]
    x = np.arange(len(VARS))
    w = 0.35
    labels = [f"{n}" for _, n, _ in VARS]
    for ax, (mkey, mlabel) in zip(axes[:2], metric_names):
        lstm = [mt[(mt["Biến"].str.startswith(n)) & (mt["Mô hình"] == "LSTM")][mkey].values[0]
                for _, n, _ in VARS]
        pers = [mt[(mt["Biến"].str.startswith(n)) & (mt["Mô hình"] == "Persistence")][mkey].values[0]
                for _, n, _ in VARS]
        ax.bar(x - w / 2, lstm, w, label="LSTM", color="tab:red")
        ax.bar(x + w / 2, pers, w, label="Persistence", color="tab:gray")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{mlabel} theo biến")
        ax.set_ylabel(mlabel)
        ax.legend(fontsize=9)
    # Skill score
    ax = axes[2]
    ss = []
    for _, n, _ in VARS:
        rl = mt[(mt["Biến"].str.startswith(n)) & (mt["Mô hình"] == "LSTM")]["RMSE"].values[0]
        rp = mt[(mt["Biến"].str.startswith(n)) & (mt["Mô hình"] == "Persistence")]["RMSE"].values[0]
        ss.append(1 - rl / rp if rp > 0 else np.nan)
    colors = ["tab:green" if s > 0 else "tab:orange" for s in ss]
    ax.bar(x, ss, 0.5, color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Skill Score so với Persistence")
    ax.set_ylabel("SS = 1 − RMSE$_{LSTM}$/RMSE$_{Pers}$")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_metrics_bar.png"), dpi=200)
    plt.close(fig)


def plot_error_by_hour(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    hour = df["dt"].dt.hour
    for ax, (v, name, unit) in zip(axes, VARS):
        ae = (df[f"lstm_{v}"] - df[f"actual_{v}"]).abs()
        by_hour = ae.groupby(hour).mean()
        ax.bar(by_hour.index, by_hour.values, color="tab:red", alpha=0.8)
        ax.set_title(f"MAE {name} theo giờ (UTC)")
        ax.set_xlabel("Giờ trong ngày (UTC)")
        ax.set_ylabel(f"MAE ({unit})")
        ax.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_error_by_hour.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Xuất bảng CSV
# ---------------------------------------------------------------------------
def save_table(df, name, caption=None, fmt=None):
    df.to_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Đọc dữ liệu exp/ ===")
    fc = load_firebase_json(FORECAST_JSON)
    rd = load_firebase_json(READINGS_JSON)
    print(f"Forecast: {len(fc)} bản ghi | Readings: {len(rd)} bản ghi")

    df = build_matched(fc, rd)
    print(f"Số cặp ghép đủ (actual + persistence): {len(df)}")

    start = fc["dt"].min().strftime("%Y-%m-%d")
    end = fc["dt"].max().strftime("%Y-%m-%d")

    # --- Bảng mô tả dữ liệu ---
    rain_note = "0 (không có sự kiện mưa)"
    dataset = pd.DataFrame([
        {"Mục": "Khoảng thời gian", "Giá trị": f"{start} đến {end} (UTC)"},
        {"Mục": "Số bản ghi forecast", "Giá trị": str(len(fc))},
        {"Mục": "Số bản ghi readings", "Giá trị": str(len(rd))},
        {"Mục": "Tần suất lấy mẫu", "Giá trị": "~60 giây"},
        {"Mục": "Tầm dự báo (horizon)", "Giá trị": "20 phút"},
        {"Mục": "Số cặp ghép hợp lệ", "Giá trị": str(len(df))},
        {"Mục": "Biến đánh giá", "Giá trị": "Nhiệt độ, Độ ẩm, Áp suất"},
        {"Mục": "Lượng mưa (rain)", "Giá trị": rain_note},
    ])
    save_table(dataset, "table_dataset", "Tổng quan tập dữ liệu vận hành (exp/)")

    # --- Bảng metric chính (minute-level) ---
    mt = compute_metric_table(df)
    fmt_metric = {"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}",
                  "ME": "{:.3f}", "R2": "{:.3f}", "r": "{:.3f}"}
    save_table(mt, "table_metrics",
               "Chỉ số đánh giá LSTM vs Persistence (đối chiếu số đo thực tế tại t+20')",
               fmt_metric)

    # --- Bảng skill score ---
    sk = compute_skill(df)
    save_table(sk, "table_skill",
               "Skill Score của LSTM so với baseline Persistence",
               {"RMSE_LSTM": "{:.3f}", "RMSE_Persistence": "{:.3f}", "Skill Score": "{:.3f}"})

    # --- Open-Meteo (tùy chọn) ---
    om = fetch_openmeteo(start, end)
    om_merged = None
    if om is not None:
        om_tbl, om_merged = openmeteo_compare(rd, om)
        save_table(om_tbl, "table_openmeteo",
                   "So sánh số đo thực tế với Open-Meteo (độ phân giải giờ)", fmt_metric)

    # --- Hình ---
    print("=== Vẽ hình ===")
    plot_timeseries(df, om_merged)
    plot_scatter(df)
    plot_residual_hist(df)
    plot_metrics_bar(mt)
    plot_error_by_hour(df)

    # --- CSV cặp dữ liệu ---
    keep = ["timestamp", "dt", "source_timestamp"]
    keep += [c for c in df.columns if c.startswith(("actual_", "lstm_", "pers_"))]
    df[keep].to_csv(os.path.join(RESULTS_DIR, "eval_results.csv"), index=False)

    # --- Tóm tắt console + ghi file summary.txt ---
    summary_lines = []
    summary_lines.append("=== KẾT QUẢ TÓM TẮT ===")
    summary_lines.append(f"Khoảng thời gian: {start} đến {end} (UTC)")
    summary_lines.append(f"Số bản ghi forecast: {len(fc)} | readings: {len(rd)} | cặp ghép: {len(df)}")
    summary_lines.append("")
    summary_lines.append("--- Chỉ số đánh giá (LSTM vs Persistence) ---")
    summary_lines.append(mt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_lines.append("")
    summary_lines.append("--- Skill Score ---")
    summary_lines.append(sk.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if om is not None:
        summary_lines.append("")
        summary_lines.append("--- Open-Meteo so với thực tế (hourly) ---")
        summary_lines.append(om_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_lines.append("")
    summary_lines.append(f"Hình PNG : {os.path.relpath(RESULTS_DIR)}/eval_*.png")
    summary_lines.append(f"Bảng CSV : {os.path.relpath(RESULTS_DIR)}/table_*.csv")
    summary_lines.append(f"Dữ liệu  : {os.path.relpath(RESULTS_DIR)}/eval_results.csv")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n[OK] Tất cả kết quả đã lưu vào: {os.path.relpath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
