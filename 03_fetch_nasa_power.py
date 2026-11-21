"""Step 3: download daily weather per district centroid from the NASA POWER API (no login needed).

Safe to re-run: finished downloads are cached in data/raw/ and skipped.
If a request fails with a 422 or timeout, lower CHUNK_YEARS to 1.
"""
import json
import time
import pandas as pd
import requests
from config import *

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN"]
CHUNK_YEARS = 4

coords = pd.read_csv(DATA_DIR / "district_coords.csv").dropna(subset=["lat", "lon"])
y0, y1 = int(WEATHER_START[:4]), int(WEATHER_END[:4])
chunks = []
for s in range(y0, y1 + 1, CHUNK_YEARS):
    e = min(s + CHUNK_YEARS - 1, y1)
    start = max(pd.Timestamp(WEATHER_START), pd.Timestamp(f"{s}-01-01"))
    end = min(pd.Timestamp(WEATHER_END), pd.Timestamp(f"{e}-12-31"))
    chunks.append((start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))


def fetch(lat, lon, start, end, tries=4):
    q = {"parameters": ",".join(PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
         "start": start, "end": end, "format": "JSON"}
    for i in range(tries):
        try:
            r = requests.get(API, params=q, timeout=120)
            r.raise_for_status()
            return r.json()["properties"]["parameter"]
        except Exception as ex:
            wait = 5 * (i + 1)
            print(f"   retry {i+1} after error: {ex}; waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed: {lat},{lon},{start}-{end}")


frames = []
for _, row in coords.iterrows():
    name = row.district_name
    safe = name.replace(" ", "_").replace("/", "-")
    parts = []
    for start, end in chunks:
        cache = RAW_DIR / f"power_{safe}_{start}_{end}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data = fetch(row.lat, row.lon, start, end)
            cache.write_text(json.dumps(data))
            time.sleep(1)
        parts.append(pd.DataFrame(data))
    d = pd.concat(parts)
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    d = d.replace(-999, pd.NA).astype(float)   # -999 is POWER's missing-value code
    d.insert(0, "district_name", name)
    frames.append(d.rename_axis("date").reset_index())
    print(f"done {name}: {len(d)} days, missing cells: {int(d[PARAMS].isna().sum().sum())}")

out = pd.concat(frames, ignore_index=True)
out.to_csv(DATA_DIR / "nasa_power_daily.csv", index=False)
print("Saved data/nasa_power_daily.csv", out.shape)
