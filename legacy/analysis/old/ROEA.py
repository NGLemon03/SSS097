# analysis/ROEA.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from analysis import config as cfg
# Load configuration
DATA_DIR = cfg.DATA_DIR
TICKER = cfg.TICKER
RESULT_DIR = cfg.RESULT_DIR
import SSSv095a1 as SSS


# Setup matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ensure plot directory exists
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# 1. Read and clean data
file_path = RESULT_DIR / "wf_grid_ALL_00631L.TW.csv"
df = pd.read_csv(file_path)
df = df[pd.to_numeric(df["ROI_wf"], errors="coerce").notnull()]
df["ROI_wf"] = df["ROI_wf"].astype(float)

# Define grouping parameter columns and convert to string
group_cols = [
    'strategy', 'data_source', 'linlen', 'smaalen', 'devwin',
    'buy_mult', 'sell_mult', 'rma_len', 'dev_len', 'prom_factor',
    'min_dist', 'cooldown', 'prom_q', 'offsets', 'vol_window', 'cooldown_days'
]
df[group_cols] = df[group_cols].astype(str)

# 2. Identify unique parameter groups present in all three periods
param_counts = df.groupby(group_cols)["period"].nunique().reset_index(name="period_count")
print("Unique parameter groups:", len(param_counts))
full_params = param_counts[param_counts["period_count"] == 3][group_cols]
print("Groups present in all 3 periods:", len(full_params))

df_full = df.merge(full_params, on=group_cols, how="inner")
print("Rows in full-period dataset:", len(df_full))

# 3. Calculate average ROI and total ROI across all periods
agg_full = df_full.groupby(group_cols).agg(
    ROI_total=('ROI', 'first'),
    ROI_mean=('ROI_wf', 'mean')
).reset_index()
print("Aggregated full-period groups:", len(agg_full))

# Single period data
period_single = "2021-08-19_2025-05-15"
df_single = df[df["period"] == period_single]
df_single = df_single.merge(full_params, on=group_cols, how="inner")
agg_single = df_single.groupby(group_cols).agg(
    ROI_total=('ROI', 'first'),
    ROI_mean=('ROI_wf', 'mean')
).reset_index()
print("Aggregated single-period groups:", len(agg_single))

# Define plotting and stats function
def plot_and_stats(x, y, title):
    if len(x) < 2:
        print(f"Not enough points to plot: {title}")
        return None, None
    corr, p = pearsonr(x, y)
    # Linear regression residual-based anomaly detection
    slope, intercept, _, _, _ = linregress(x, y)
    res = y - (intercept + slope * x)
    z = (res - res.mean()) / res.std()
    anomalies = z.abs() > 2
    outlier_pct = anomalies.sum() / len(x) * 100

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.4, label="All")
    plt.scatter(x[anomalies], y[anomalies], edgecolors='black', label=f"Outliers ({outlier_pct:.1f}%)")
    plt.xlabel("ROI_mean")
    plt.ylabel("ROI_total")
    plt.title(f"{title}\nPearson r={corr:.3f}, p={p:.3g}")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / f'{title.replace(":", "_").replace(" ", "_")}.png')
    plt.close()

    return corr, outlier_pct

# 1. Total ROI vs ROI_mean, full periods
plot_and_stats(
    agg_full["ROI_mean"].astype(float), agg_full["ROI_total"].astype(float),
    "Total ROI vs ROI_mean (Full Periods)"
)

# 2. Total ROI vs ROI_mean, 2021-08-19_2025-05-15
plot_and_stats(
    agg_single["ROI_mean"].astype(float), agg_single["ROI_total"].astype(float),
    f"Total ROI vs ROI_mean ({period_single})"
)

# 3. Per strategy ROI_mean, full periods
for strat in agg_full["strategy"].unique():
    sub = agg_full[agg_full["strategy"] == strat]
    plot_and_stats(
        sub["ROI_mean"].astype(float), sub["ROI_total"].astype(float),
        f"{strat}: Total ROI vs ROI_mean (Full)"
    )

# 4. Per strategy ROI_mean, single period
for strat in agg_single["strategy"].unique():
    sub = agg_single[agg_single["strategy"] == strat]
    plot_and_stats(
        sub["ROI_mean"].astype(float), sub["ROI_total"].astype(float),
        f"{strat}: Total ROI vs ROI_mean ({period_single})"
    )

# 5. Stats for each strategy & data_source
stats_list = []
for (strat, src), group in agg_full.groupby(["strategy", "data_source"]):
    if len(group) < 2:
        continue
    corr, _ = pearsonr(group["ROI_mean"].astype(float), group["ROI_total"].astype(float))
    slope, intercept, _, _, _ = linregress(group["ROI_mean"].astype(float), group["ROI_total"].astype(float))
    res = group["ROI_total"].astype(float) - (intercept + slope * group["ROI_mean"].astype(float))
    z = (res - res.mean()) / res.std()
    anomalies = z.abs() > 2
    outlier_pct = anomalies.sum() / len(group) * 100
    stats_list.append({
        "strategy": strat,
        "data_source": src,
        "pearson_r": corr,
        "outlier_pct": outlier_pct
    })
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(DATA_DIR / "roi_stats.csv", index=False)

# 6. Random parameter comparison
for strat in agg_full["strategy"].unique():
    sub = agg_full[agg_full["strategy"] == strat]
    best_return = sub["ROI_total"].max()
    df_raw, df_fac = SSS.load_data_wrapper(TICKER, start_date="2010-06-01", smaa_source=sub["data_source"].iloc[0])
    try:
        rand = SSS.optimize_parameters(
            df_raw, df_fac, strategy_type=strat,
            cooldown=3, cost=0, restrict_loss=False,
            mode="random", n_samples=1000
        )
        p_val = (rand['total_return'] > best_return).mean()
        print(f"{strat}: Random params win rate p={p_val:.3f}")
        # Save random parameter results
        rand.to_csv(PLOT_DIR / f"random_params_{strat}.csv", index=False)
    except Exception as e:
        print(f"{strat}: Random params comparison failed: {e}")