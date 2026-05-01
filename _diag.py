import pandas as pd
df = pd.read_parquet(r'data/processed/train_features.parquet')
susp = ['vis_m', 'wind_mps', 'vis_km', 'wind_knots']
for c in susp:
    print(f"{c:12s} in cols? {c in df.columns}")
print()
print("correlations among the 4:")
print(df[['vis_m', 'vis_km', 'wind_mps', 'wind_knots']].corr().round(4))
print()
print("vis_m head:", df['vis_m'].head(5).tolist())
print("vis_km head:", df['vis_km'].head(5).tolist())
print("wind_mps head:", df['wind_mps'].head(5).tolist())
print("wind_knots head:", df['wind_knots'].head(5).tolist())
print()
# Is vis_m == vis_km * 1000? Is wind_mps == wind_knots * 0.5144?
ratio_v = (df['vis_m'] / df['vis_km'].replace(0, pd.NA)).dropna()
ratio_w = (df['wind_mps'] / df['wind_knots'].replace(0, pd.NA)).dropna()
print(f"vis_m / vis_km: median={ratio_v.median():.3f}  std={ratio_v.std():.3f}")
print(f"wind_mps / wind_knots: median={ratio_w.median():.4f}  std={ratio_w.std():.4f}")
