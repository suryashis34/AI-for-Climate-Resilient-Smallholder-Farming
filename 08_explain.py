"""Step 8 (v2): ablations with confidence intervals, SHAP explanations, and anomaly-based advisory signals.

Needs:  pip install shap xgboost matplotlib     and   python 05b_climatology.py   run once first.
Uses XGBoost (falls back to Random Forest if xgboost is missing). Clean rows only.

What changed from v1 (problems found in review):
  * SHAP is now also expressed RELATIVE TO EACH DISTRICT-SEASON'S USUAL LEVEL ("anomaly SHAP"), so a
    signal only says what THIS year's departure from normal is associated with.
  * "Normal" weather comes from the full 2013-2022 climatology (05b), not from 2-3 yield years.
  * Advisory text: weather only, only when |z| >= 0.5. Soil is reported as district background, not advice.
  * Ablations carry district-level bootstrap 95% CIs (random-seed spread alone understates uncertainty).
  * Fixed duplicate 'season' CSV header; figures redrawn (dot plot, honest axis).
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import DATA_DIR, VAL_YEAR

warnings.filterwarnings("ignore")
try:
    import shap
except ImportError:
    raise SystemExit("Please run:  pip install shap")
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("xgboost not installed -> using Random Forest instead (pip install xgboost for the paper version)\n")

SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 2000
Z_MIN = 0.5            # a weather feature is reported only if it is at least this many SDs from normal
MIN_CLIM_YEARS = 5
TARGET = "yield"
cfg = json.load(open(DATA_DIR / "features.json"))
FEATURES, WEATHER, SOIL = cfg["features"], cfg["weather"], cfg["soil"]

LABEL = {
    "rain_total_mm": "Season rainfall", "rain_days": "Rainy days", "tmax_mean": "Mean max temperature",
    "tmin_mean": "Mean min temperature", "tmean": "Mean temperature", "hot_days": "Days above 35 C",
    "rh_mean": "Relative humidity", "solar_mean": "Solar radiation", "rain_early": "Early-season rainfall",
    "tmax_early": "Early-season max temperature", "rain_mid": "Mid-season rainfall",
    "tmax_mid": "Mid-season max temperature", "rain_late": "Late-season rainfall",
    "tmax_late": "Late-season max temperature", "soil_phh2o": "Soil pH", "soil_soc": "Soil organic carbon",
    "soil_clay": "Soil clay content", "lag_yield": "Last year's yield", "season_code": "Season", "area": "Sown area",
}
GROUP = {**{f: "weather" for f in WEATHER}, **{f: "soil" for f in SOIL},
         "lag_yield": "last-year yield", "season_code": "season effect"}

# ---------------------------------------------------------------- data
df = pd.read_csv(DATA_DIR / "model_ready.csv")
df = df[~df.suspect_fill].copy()
tr, va = df[df.split == "train"], df[df.split == "val"]
trva = pd.concat([tr, va])
te = df[(df.split == "test") & df.has_lag].copy()
va_s = va[va.has_lag]
print(f"clean rows: train {len(tr)}, val {len(va)}, test scored {len(te)}\n")

try:
    clim = pd.read_csv(DATA_DIR / "weather_features_all.csv")
except FileNotFoundError:
    raise SystemExit("Missing data/weather_features_all.csv -> run  python 05b_climatology.py  first")
clim = clim[clim.year_start <= VAL_YEAR]               # normals use pre-test years only
g = clim.groupby(["district_name", "season"])
CM, CS, CN = g[WEATHER].mean(), g[WEATHER].std(), g.size()


def make(seed, depth):
    if HAVE_XGB:
        return XGBRegressor(n_estimators=300, max_depth=depth, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=1)
    return RandomForestRegressor(n_estimators=300, max_depth=depth, min_samples_leaf=3,
                                 random_state=seed, n_jobs=-1)


def metrics(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


depths = (3, 5) if HAVE_XGB else (6, 10)
val_rmse = {d: metrics(va_s[TARGET], make(0, d).fit(tr[FEATURES], tr[TARGET]).predict(va_s[FEATURES]))["RMSE"]
            for d in depths}
DEPTH = min(val_rmse, key=val_rmse.get)
print(f"chosen max_depth = {DEPTH} (validation RMSE {val_rmse[DEPTH]:.0f})\n")

# ---------------------------------------------------------------- ablations + bootstrap
VARIANTS = {
    "Full model": FEATURES,
    "No weather": [f for f in FEATURES if f not in WEATHER],
    "No soil": [f for f in FEATURES if f not in SOIL],
    "No last-year yield": [f for f in FEATURES if f != "lag_yield"],
    "Weather + season only": WEATHER + ["season_code"],
    "Last-year yield + season only": ["lag_yield", "season_code"],
    "Full + sown area": FEATURES + ["area"],
}
P, rows = {}, []
for name, feats in VARIANTS.items():
    runs, preds = [], []
    for s in SEEDS:
        m = make(s, DEPTH).fit(trva[feats], trva[TARGET])
        p = m.predict(te[feats])
        preds.append(p)
        runs.append(metrics(te[TARGET], p))
    P[name] = np.mean(preds, axis=0)                  # 5-seed ensemble prediction
    r = pd.DataFrame(runs)
    rows.append({"variant": name, "n_features": len(feats), "RMSE_seed_mean": r.RMSE.mean(),
                 "RMSE_seed_sd": r.RMSE.std(), "MAE": r.MAE.mean(), "R2": r.R2.mean()})
P["Last-year baseline"] = te["lag_yield"].to_numpy()
rows.append({"variant": "Last-year baseline", "n_features": np.nan, **{
    "RMSE_seed_mean": metrics(te[TARGET], P["Last-year baseline"])["RMSE"], "RMSE_seed_sd": 0.0,
    "MAE": metrics(te[TARGET], P["Last-year baseline"])["MAE"], "R2": metrics(te[TARGET], P["Last-year baseline"])["R2"]}})
abl = pd.DataFrame(rows)

y = te[TARGET].to_numpy()
dist = te.district_name.to_numpy()
U = np.unique(dist)
by_d = {u: np.where(dist == u)[0] for u in U}
rng = np.random.default_rng(0)
samples = [np.concatenate([by_d[u] for u in rng.choice(U, len(U))]) for _ in range(N_BOOT)]
rm = lambda p, ii: float(np.sqrt(np.mean((p[ii] - y[ii]) ** 2)))
allidx = np.arange(len(y))
BOOT = {k: np.array([rm(v, ii) for ii in samples]) for k, v in P.items()}
ci = {}
for k in P:
    ens = rm(P[k], allidx)
    d_boot = BOOT[k] - BOOT["Full model"]
    ci[k] = {"RMSE_ensemble": ens, "RMSE_lo": np.percentile(BOOT[k], 2.5), "RMSE_hi": np.percentile(BOOT[k], 97.5),
             "dRMSE_vs_full": ens - rm(P["Full model"], allidx),
             "dRMSE_lo": np.percentile(d_boot, 2.5), "dRMSE_hi": np.percentile(d_boot, 97.5)}
abl = abl.merge(pd.DataFrame(ci).T.rename_axis("variant").reset_index(), on="variant")
abl["significant_vs_full"] = np.where(abl.variant == "Full model", "",
                                      np.where((abl.dRMSE_lo > 0) | (abl.dRMSE_hi < 0), "yes", "no"))
abl.to_csv(DATA_DIR / "ablation_results.csv", index=False)
pd.DataFrame(P, index=te.index).assign(actual=y, district_name=dist, year_start=te.year_start.to_numpy(),
                                       season=te.season.to_numpy()).to_csv(DATA_DIR / "ablation_predictions.csv", index=False)
show = abl[["variant", "RMSE_seed_mean", "RMSE_seed_sd", "RMSE_ensemble", "RMSE_lo", "RMSE_hi",
            "dRMSE_vs_full", "dRMSE_lo", "dRMSE_hi", "significant_vs_full"]]
print(f"=== ABLATIONS (test, {len(te)} rows; CIs = district bootstrap, {N_BOOT} resamples) ===")
print(show.round(0).to_string(index=False))
print("dRMSE = variant minus full model; 'significant' = 95% CI of the difference excludes 0.\n")

# ---------------------------------------------------------------- SHAP (raw and anomaly)
model = make(42, DEPTH).fit(trva[FEATURES], trva[TARGET])
explainer = shap.TreeExplainer(model)
S = pd.DataFrame(np.asarray(explainer.shap_values(te[FEATURES])), columns=FEATURES, index=te.index)
S_tr = pd.DataFrame(np.asarray(explainer.shap_values(trva[FEATURES])), columns=FEATURES, index=trva.index)

# Each district-season's usual SHAP level (its 'normal' contribution). Subtracting it leaves the
# contribution of THIS year's departure from normal.
M = S_tr.groupby([trva.district_name, trva.season]).mean()
Ms = S_tr.groupby(trva.season).mean()
keys = pd.MultiIndex.from_frame(te[["district_name", "season"]])
Bm = np.array(M.reindex(keys).to_numpy(), dtype=float)
miss = np.isnan(Bm).any(axis=1)
if miss.any():
    Bm[miss] = Ms.loc[te.season.to_numpy()[miss]].to_numpy()
A = S - Bm                                             # anomaly SHAP, same units (kg/ha)

meta = pd.DataFrame({"district_name": te.district_name, "year_start": te.year_start, "season": te.season})
S.join(meta).to_csv(DATA_DIR / "shap_values_test.csv", index=False)
A.join(meta).to_csv(DATA_DIR / "shap_anomaly_test.csv", index=False)

glob = pd.DataFrame({"mean_abs_shap": S.abs().mean(), "mean_abs_anomaly_shap": A.abs().mean()})
glob["group"] = glob.index.map(GROUP)
glob["label"] = glob.index.map(LABEL)
glob = glob.sort_values("mean_abs_shap", ascending=False)
glob.to_csv(DATA_DIR / "shap_global.csv")
print("=== FEATURE IMPORTANCE (kg/ha): overall vs relative to district normal ===")
print(glob.head(12).round(1).to_string())

grp = pd.DataFrame({k: S[[f for f in FEATURES if GROUP[f] == k]].sum(axis=1) for k in sorted(set(GROUP.values()))})
gs = grp.abs().groupby(te.season).mean().round(1)
gs.loc["All"] = grp.abs().mean().round(1)
gs.index.name = "test_season"
gs.to_csv(DATA_DIR / "shap_group_by_season.csv")
print("\n=== GROUP CONTRIBUTION, mean |SHAP| (kg/ha) ===")
print(gs.to_string())

imp = {yr: S[te.year_start == yr].abs().mean() for yr in sorted(te.year_start.unique())}
if len(imp) == 2:
    a, b = imp.values()
    print(f"\nStability across test years (raw SHAP): Spearman rho = {spearmanr(a, b)[0]:.2f}; "
          f"top-5 overlap = {len(set(a.nlargest(5).index) & set(b.nlargest(5).index))}/5")

# ---------------------------------------------------------------- z-scores from long climatology
mu = CM.reindex(keys).to_numpy()
sd = CS.reindex(keys).to_numpy()
n = CN.reindex(keys).to_numpy().astype(float)
Zv = (te[WEATHER].to_numpy() - mu) / np.where(sd > 0, sd, np.nan)
Zv[np.isnan(n) | (n < MIN_CLIM_YEARS)] = np.nan
Z = pd.DataFrame(Zv, columns=WEATHER, index=te.index)
print(f"\nClimatology years per district-season: min {int(np.nanmin(n))}, max |z| in test = {np.nanmax(np.abs(Zv)):.1f}")

# ---------------------------------------------------------------- effect direction (anomaly SHAP vs z)
dirs = []
for s in ["Kharif", "Rabi", "Summer", "All"]:
    mask = (te.season == s).to_numpy() if s != "All" else np.ones(len(te), bool)
    for f in WEATHER:
        pair = pd.DataFrame({"z": Z.loc[mask, f], "s": A.loc[mask, f]}).dropna()
        if len(pair) > 5:
            rho = spearmanr(pair.z, pair.s)[0]
            dirs.append({"season": s, "feature": f, "n": len(pair), "spearman_z_vs_anomaly_shap": rho,
                         "strength": "weak/noise (|rho|<0.3)" if abs(rho) < 0.3 else "moderate+"})
dirs = pd.DataFrame(dirs)
dirs.to_csv(DATA_DIR / "effect_direction.csv", index=False)
print("\n=== EFFECT DIRECTION (rho of z vs anomaly SHAP; + means above-normal raises predicted yield) ===")
print(dirs.pivot(index="feature", columns="season", values="spearman_z_vs_anomaly_shap")
      [["Kharif", "Rabi", "Summer", "All"]].round(2).to_string())
print(f"Cells with |rho| >= 0.3: {int((dirs.strength == 'moderate+').sum())} of {len(dirs)}; "
      "interpret signs only where consistent across seasons.")

# ---------------------------------------------------------------- advisory (weather only, anomalies only)
district_norm = trva.groupby(["district_name", "season"])[TARGET].mean().reindex(keys).to_numpy()
fallback = te.season.map(trva.groupby("season")[TARGET].mean()).to_numpy()
district_norm = np.where(np.isnan(district_norm), fallback, district_norm)
pred = np.asarray(P["Full model"], dtype=float)   # 5-seed ensemble = same numbers as the ablation table
out = []
for i, (idx, row) in enumerate(te.iterrows()):
    cand = [(f, Z.loc[idx, f], A.loc[idx, f]) for f in WEATHER if np.isfinite(Z.loc[idx, f]) and abs(Z.loc[idx, f]) >= Z_MIN]
    cand.sort(key=lambda t: -abs(t[2]))
    msgs = [f"{LABEL[f]} {'above' if z > 0 else 'below'} the district's normal (z={z:+.1f}): "
            f"{'+' if a > 0 else '-'}{abs(a):.0f} kg/ha vs a typical year" for f, z, a in cand[:3]]
    la = float(A.loc[idx, "lag_yield"])
    carry = (f"Last year's yield ({row.lag_yield:.0f}) was {'above' if row.lag_yield > district_norm[i] else 'below'} the district's normal "
             f"({district_norm[i]:.0f}): {'+' if la > 0 else '-'}{abs(la):.0f} kg/ha carried over into this prediction") if np.isfinite(row.lag_yield) else ""
    out.append({"district": row.district_name, "year": row.year_start, "season": row.season,
                "actual": row[TARGET], "predicted": float(pred[i]), "district_normal_yield": float(district_norm[i]),
                "last_year_yield": row.lag_yield, "carryover_signal": carry, "n_unusual_weather_features": len(cand),
                "signal_1": msgs[0] if len(msgs) > 0 else "No unusual weather this season; prediction follows last year's yield and the district's usual level" if not msgs else "",
                "signal_2": msgs[1] if len(msgs) > 1 else "", "signal_3": msgs[2] if len(msgs) > 2 else ""})
adv = pd.DataFrame(out)
adv.to_csv(DATA_DIR / "advisory_test.csv", index=False)
print(f"\n=== ADVISORY CHECK === rows with no unusual weather: {int((adv.n_unusual_weather_features == 0).sum())} of {len(adv)}; "
      f"all signals have |z| >= {Z_MIN}")
adv["abs_err"] = (adv.actual - adv.predicted).abs()
print("Example (largest prediction errors):")
for _, r in adv.sort_values("abs_err", ascending=False).head(3).iterrows():
    print(f"\n{r.district} {r.year} {r.season}: actual {r.actual:.0f}, predicted {r.predicted:.0f} "
          f"(district normal {r.district_normal_yield:.0f}, last year {r.last_year_yield:.0f})")
    for k in ("carryover_signal", "signal_1", "signal_2", "signal_3"):
        if r[k]:
            print("  -", r[k])
print("\nNOTE: predicted = 5-seed ensemble; signal sizes (kg/ha) come from the seed-42 model, so they explain the prediction only approximately.")
print("NOTE: signals describe what the MODEL associates with yield, not proven causes. Soil is district background, not advice.")

# ---------------------------------------------------------------- figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    a2 = abl.sort_values("RMSE_ensemble", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ypos = np.arange(len(a2))
    ax.errorbar(a2.RMSE_ensemble, ypos, xerr=[a2.RMSE_ensemble - a2.RMSE_lo, a2.RMSE_hi - a2.RMSE_ensemble],
                fmt="o", color="#2F5C99", ecolor="#7A9CC6", capsize=3)
    ax.axvline(a2.loc[a2.variant == "Last-year baseline", "RMSE_ensemble"].iloc[0], color="#C44E52", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(a2.variant)
    ax.set_xlabel("Test RMSE (kg/ha), lower is better; bars = 95% district-bootstrap CI")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "fig_ablation.png", dpi=300)
    plt.close()

    colors = {"weather": "#4C72B0", "soil": "#8C6D31", "last-year yield": "#C44E52", "season effect": "#7F7F7F"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title in ((axes[0], "mean_abs_shap", "Overall contribution"),
                           (axes[1], "mean_abs_anomaly_shap", "Contribution relative to district normal")):
        gg = glob.sort_values(col, ascending=False).head(12).iloc[::-1]
        ax.barh(gg.label, gg[col], color=[colors[x] for x in gg.group])
        ax.set_xlabel("Mean |SHAP| (kg/ha)")
        ax.set_title(title, fontsize=10)
        ax.legend(handles=[Patch(color=colors[k], label=k) for k in colors if k in set(gg.group)],
                  frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "fig_shap_global.png", dpi=300)
    plt.close()
    print("Saved fig_ablation.png and fig_shap_global.png")
except ImportError:
    print("matplotlib not installed -> figures skipped (pip install matplotlib)")
print("Saved results to data/")
