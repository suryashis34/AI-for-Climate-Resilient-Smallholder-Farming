"""Step 5b: seasonal weather features for EVERY district, year and season (not only years with yield rows).

Used by step 8 to compute proper 'normal' weather (a long climatology instead of 2-3 yield years).
Uses exactly the same season windows and feature definitions as 05_merge.py.
Output: data/weather_features_all.csv
"""
import numpy as np
import pandas as pd
from config import *

RAIN_DAY_MM, HOT_DAY_C = 2.5, 35.0


def window(year_start, season):
    y = int(year_start)
    return {"Kharif": (f"{y}-06-01", f"{y}-10-31"),
            "Rabi": (f"{y}-11-01", f"{y+1}-03-31"),
            "Summer": (f"{y+1}-04-01", f"{y+1}-06-30")}[season]


wx = pd.read_csv(DATA_DIR / "nasa_power_daily.csv", parse_dates=["date"]).sort_values(["district_name", "date"])
rows = []
for district, w in wx.groupby("district_name"):
    w = w.set_index("date")
    for y in range(2013, 2025):
        for season in ("Kharif", "Rabi", "Summer"):
            a, b = window(y, season)
            seg = w.loc[a:b]
            expected = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
            if len(seg) < 0.9 * expected:
                continue
            f = {"district_name": district, "year_start": y, "season": season,
                 "rain_total_mm": seg["PRECTOTCORR"].sum(),
                 "rain_days": (seg["PRECTOTCORR"] >= RAIN_DAY_MM).sum(),
                 "tmax_mean": seg["T2M_MAX"].mean(), "tmin_mean": seg["T2M_MIN"].mean(),
                 "tmean": seg["T2M"].mean(), "hot_days": (seg["T2M_MAX"] > HOT_DAY_C).sum(),
                 "rh_mean": seg["RH2M"].mean(), "solar_mean": seg["ALLSKY_SFC_SW_DWN"].mean()}
            for tag, part in zip(["early", "mid", "late"], np.array_split(np.arange(len(seg)), 3)):
                s = seg.iloc[part]
                f[f"rain_{tag}"] = s["PRECTOTCORR"].sum()
                f[f"tmax_{tag}"] = s["T2M_MAX"].mean()
            rows.append(f)

out = pd.DataFrame(rows)
out.to_csv(DATA_DIR / "weather_features_all.csv", index=False)
print(f"Saved data/weather_features_all.csv: {len(out)} rows, {out.district_name.nunique()} districts, "
      f"years {out.year_start.min()}-{out.year_start.max()}")
print("Rows per season:", out.season.value_counts().to_dict())

# Cross-check against the features already used for modelling (must match exactly).
try:
    mt = pd.read_csv(DATA_DIR / "model_ready.csv")
    cols = [c for c in out.columns if c not in ("district_name", "year_start", "season")]
    m = mt.merge(out, on=["district_name", "year_start", "season"], suffixes=("", "_c"))
    diff = max(float((m[c] - m[c + "_c"]).abs().max()) for c in cols if c in mt.columns)
    print(f"Cross-check vs model_ready.csv: {len(m)} rows matched, max abs difference = {diff:.2e}")
except FileNotFoundError:
    pass
