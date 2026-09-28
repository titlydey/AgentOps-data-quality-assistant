import numpy as np, pandas as pd
rng = np.random.default_rng(0)
df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60).astype(str),
                   "revenue": rng.normal(1000, 50, 60).round(2), "expenses": rng.normal(700, 40, 60).round(2)})
df.loc[[5, 17], "revenue"] = [9000, 20]      # injected anomalies
df.loc[[9, 30], "expenses"] = None           # injected missing values
df.loc[12, "date"] = "not-a-date"            # injected invalid date
df = pd.concat([df, df.iloc[[3]]])           # injected duplicate
df.to_csv("data/sample.csv", index=False)
