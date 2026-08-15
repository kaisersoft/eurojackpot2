
import io
from pathlib import Path
import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/rescue3dcom-hub/lotto-data/main/eurojackpot.csv"
OUT = Path("data/eurojackpot_history.csv")

r = requests.get(URL, timeout=30, headers={"User-Agent":"Eurojackpot-Predictor-Updater/2.1"})
r.raise_for_status()
x = pd.read_csv(io.BytesIO(r.content), header=None)
x = x.iloc[:, :9]
x.columns = ["draw_no","Datum","zahl1","zahl2","zahl3","zahl4","zahl5","euro1","euro2"]
x["Datum"] = pd.to_datetime(x["Datum"], dayfirst=True, errors="coerce")
for c in ["zahl1","zahl2","zahl3","zahl4","zahl5","euro1","euro2"]:
    x[c] = pd.to_numeric(x[c], errors="coerce")
x = x.dropna()
x = x[
    x[["zahl1","zahl2","zahl3","zahl4","zahl5"]].apply(lambda r: len(set(r))==5 and all(1<=v<=50 for v in r),axis=1)
    & x[["euro1","euro2"]].apply(lambda r: len(set(r))==2 and all(1<=v<=12 for v in r),axis=1)
]
for c in ["zahl1","zahl2","zahl3","zahl4","zahl5"]:
    x[c] = x[c].astype(int)
x["euro1"], x["euro2"] = x[["euro1","euro2"]].min(axis=1), x[["euro1","euro2"]].max(axis=1)
for i in range(len(x)):
    vals=sorted(x.loc[i,["zahl1","zahl2","zahl3","zahl4","zahl5"]].tolist())
    for j,c in enumerate(["zahl1","zahl2","zahl3","zahl4","zahl5"]): x.loc[i,c]=vals[j]
x["Datum"] = x["Datum"].dt.strftime("%Y-%m-%d")
x = x[["Datum","zahl1","zahl2","zahl3","zahl4","zahl5","euro1","euro2"]].drop_duplicates().sort_values("Datum")
OUT.parent.mkdir(exist_ok=True)
x.to_csv(OUT,sep=";",index=False)
print(f"Updated {len(x)} draws; latest={x['Datum'].max()}")
