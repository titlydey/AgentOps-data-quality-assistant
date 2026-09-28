import re
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest

def data_quality(path="data/sample.csv"):
    df = pd.read_csv(path)
    rep = {"rows": len(df), "duplicates": int(df.duplicated().sum()),
           "missing": {k: int(v) for k, v in df.isna().sum().items() if v}}
    if "date" in df:
        rep["invalid_dates"] = int(pd.to_datetime(df["date"], errors="coerce").isna().sum())
    num = df.select_dtypes("number").dropna()
    flags = IsolationForest(contamination=0.05, random_state=0).fit_predict(num)
    rep["anomalous_rows"] = [int(i) for i in num.index[flags == -1]]
    return rep

def github_repo(repo):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):  # validate before calling out
        raise ValueError("invalid repo name")
    r = requests.get(f"https://api.github.com/repos/{repo}", timeout=10)
    r.raise_for_status()
    d = r.json()
    return {k: d.get(k) for k in ("full_name", "stargazers_count", "forks_count", "open_issues_count", "language")}
