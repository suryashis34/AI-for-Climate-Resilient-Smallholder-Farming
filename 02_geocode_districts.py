"""Step 2: get a latitude/longitude for every district (OpenStreetMap Nominatim, free, 1 request/second).

Run on your own computer. Then OPEN data/district_coords.csv and eyeball every row:
geocoders sometimes match a town with the same name in another state.
Fix wrong rows by hand and save; later scripts read that file as-is.
"""
import time
import pandas as pd
import requests
from config import *

URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode(q):
    name = q.split(",")[0].replace(" district", "").strip()
    r = requests.get(URL, params={"name": name, "count": 10, "country_code": "IN",
                                  "language": "en", "format": "json"}, timeout=30)
    r.raise_for_status()
    for hit in r.json().get("results", []):
        if hit.get("admin1", "").lower() == STATE.lower():
            return hit["latitude"], hit["longitude"], f'{hit["name"]}, {hit.get("admin2","")}, {hit["admin1"]}'
    return None, None, ""


districts = pd.read_csv(DATA_DIR / "districts.csv")
rows = []
for name in districts.district_name:
    lat, lon, shown = geocode(f"{name} district, {STATE}, India")
    time.sleep(1.1)
    if lat is None:
        lat, lon, shown = geocode(f"{name}, {STATE}, India")
        time.sleep(1.1)
    rows.append({"district_name": name, "lat": lat, "lon": lon, "matched_as": shown})
    print(f"{name:25s} {lat} {lon}  <- {shown[:70]}")

out = pd.DataFrame(rows)
out.to_csv(DATA_DIR / "district_coords.csv", index=False)
print("\nMissing coordinates:", out.lat.isna().sum(), "-> fill these in by hand.")
print("Saved data/district_coords.csv")
