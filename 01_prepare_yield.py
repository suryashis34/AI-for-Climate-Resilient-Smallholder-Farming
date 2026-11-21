"""Step 1: filter the APY file to one crop and state, clean it, list the districts."""
import numpy as np
import pandas as pd
from config import *

df = pd.read_csv(YIELD_CSV)
df = df[
    (df.state_name.str.strip().str.lower() == STATE.lower())
    & (df.crop_name.str.strip().str.lower() == CROP.lower())
].copy()
n_start = len(df)

df["year_start"] = df["year"].str[:4].astype(int)
df["district_name"] = df["district_name"].str.strip()

# Cleaning: missing or non-positive values, tiny areas.
df = df.dropna(subset=["area", "production", "yield"])
df = df[(df["yield"] > 0) & (df["area"] > 0)]
df = df[df["area"] >= MIN_AREA_HA]
n_clean = len(df)

# Consistency check: yield (kg/ha) should equal production (t) / area (ha) * 1000.
calc = df["production"] / df["area"] * 1000
df["yield_mismatch"] = ((calc - df["yield"]).abs() / df["yield"]) > 0.05

# Outlier flag: robust z-score within each district and season.
grp = df.groupby(["district_name", "season"])["yield"]
med = grp.transform("median")
mad = grp.transform(lambda s: (s - s.median()).abs().median()).astype(float).replace(0, np.nan)
df["robust_z"] = 0.6745 * (df["yield"] - med) / mad
df["outlier_flag"] = df["robust_z"].abs() > 5

df = df.sort_values(["district_name", "season", "year_start"])
df.to_csv(DATA_DIR / "yield_clean.csv", index=False)

districts = (
    df.groupby(["district_name", "district_code"], dropna=False)
    .agg(rows=("id", "count"), first_year=("year_start", "min"), last_year=("year_start", "max"))
    .reset_index()
)
districts.to_csv(DATA_DIR / "districts.csv", index=False)

print(f"{STATE} / {CROP}: {n_start} raw rows -> {n_clean} after cleaning")
print(f"Districts: {len(districts)} | years: {df.year_start.min()}-{df.year_start.max()}")
print("Rows per season:", df.season.value_counts().to_dict())
print(f"Yield/production mismatch rows: {int(df.yield_mismatch.sum())}")
print(f"Outlier-flagged rows (kept, review them): {int(df.outlier_flag.sum())}")
print("Saved data/yield_clean.csv and data/districts.csv")
