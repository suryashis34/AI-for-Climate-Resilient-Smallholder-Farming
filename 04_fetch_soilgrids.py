"""Step 4: soil properties per district from the ISRIC SoilGrids REST API.

The public API allows only about 5 calls per minute, so this sleeps between calls.
With SOIL_POINTS = 5 and ~38 districts expect roughly 40 minutes; set SOIL_POINTS = 1 in
config.py for ~8 minutes. Finished points are cached, so you can stop and re-run safely.

If the API is down (it is a best-effort service), use the fallback in the README notes:
Google Earth Engine's OpenLandMap soil layers, or the SoilGrids rasters on files.isric.org.
"""
import json
import time
import pandas as pd
import requests
from config import *

API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ["phh2o", "soc", "sand", "clay", "nitrogen", "cec"]
DEPTHS = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}     # label -> thickness for weighting
SLEEP = 13  # seconds between calls


def points(lat, lon):
    pts = [(lat, lon)]
    if SOIL_POINTS >= 5:
        d = SOIL_OFFSET_DEG
        pts += [(lat + d, lon), (lat - d, lon), (lat, lon + d), (lat, lon - d)]
    return pts


def query(lat, lon, tries=4):
    q = [("lat", lat), ("lon", lon), ("value", "mean")]
    q += [("property", p) for p in PROPS] + [("depth", d) for d in DEPTHS]
    for i in range(tries):
        try:
            r = requests.get(API, params=q, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"   HTTP {r.status_code}")
        except requests.exceptions.RequestException as ex:
            print(f"   network error: {type(ex).__name__}")
        wait = 20 * (i + 1)
        print(f"   waiting {wait}s before retry")
        time.sleep(wait)
    return None


def parse(js):
    """Return {property: depth-weighted mean in conventional units} for one point."""
    out = {}
    for layer in js["properties"]["layers"]:
        dfac = layer["unit_measure"]["d_factor"]
        num = den = 0.0
        for dep in layer["depths"]:
            v = dep["values"].get("mean")
            w = DEPTHS.get(dep["label"])
            if v is not None and w:
                num += (v / dfac) * w
                den += w
        out[layer["name"]] = num / den if den else None
    return out


coords = pd.read_csv(DATA_DIR / "district_coords.csv").dropna(subset=["lat", "lon"])
rows = []
for _, row in coords.iterrows():
    samples = []
    for k, (la, lo) in enumerate(points(row.lat, row.lon)):
        cache = RAW_DIR / f"soil_{row.district_name.replace(' ', '_')}_{k}.json"
        if cache.exists():
            js = json.loads(cache.read_text())
        else:
            js = query(la, lo)
            time.sleep(SLEEP)
            if js is None:
                continue
            cache.write_text(json.dumps(js))
        samples.append(parse(js))
    s = pd.DataFrame(samples)
    rec = {"district_name": row.district_name, "soil_points_used": len(s)}
    for p in PROPS:
        rec[f"soil_{p}"] = s[p].mean() if p in s and s[p].notna().any() else None
    rows.append(rec)
    print(f"done {row.district_name}: {len(s)} points")

out = pd.DataFrame(rows)
out.to_csv(DATA_DIR / "soilgrids_district.csv", index=False)
print("Units: pH (pH units), soc (g/kg), sand/clay (%), nitrogen (g/kg), cec (cmol/kg)")
print("Saved data/soilgrids_district.csv")
