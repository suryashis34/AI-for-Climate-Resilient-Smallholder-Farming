"""Shared settings. Change CROP / STATE here and every script follows."""
from pathlib import Path

CROP = "Rice"
STATE = "Tamil Nadu"
YIELD_CSV = "cropwise-area-production-yield-upag.csv"

# Rows with a tiny sown area give noisy yields; drop them (hectares).
MIN_AREA_HA = 50

# Weather download range: covers 2013-14 Kharif start to 2024-25 Summer end.
WEATHER_START = "2013-06-01"
WEATHER_END = "2025-06-30"

# Soil sampling: 1 = centroid only (fast), 5 = centroid + 4 offsets (better district average).
SOIL_POINTS = 5
SOIL_OFFSET_DEG = 0.08  # about 9 km

# Temporal split on the agricultural year's start year (2013 means 2013-2014).
TRAIN_LAST_YEAR = 2021
VAL_YEAR = 2022          # test = everything after

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
