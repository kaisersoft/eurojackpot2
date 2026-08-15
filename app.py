
import io
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Eurojackpot Predictor 2.1", page_icon="🎯", layout="wide")

DATA_FILE = Path("data/eurojackpot_history.csv")
HISTORY_FILE = Path("data/prediction_history.csv")
DATA_URL = "https://raw.githubusercontent.com/rescue3dcom-hub/lotto-data/main/eurojackpot.csv"

MAIN = [f"zahl{i}" for i in range(1, 6)]
EURO = ["euro1", "euro2"]

@st.cache_data(ttl=3600)
def load_repo_data():
    return pd.read_csv(DATA_FILE, sep=";")

def normalize(df):
    df = df.copy()
    for c in MAIN + EURO:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=MAIN + EURO)
    for c in MAIN + EURO:
        df[c] = df[c].astype(int)
    df = df[
        df[MAIN].apply(lambda r: len(set(r)) == 5 and all(1 <= x <= 50 for x in r), axis=1)
        & df[EURO].apply(lambda r: len(set(r)) == 2 and all(1 <= x <= 12 for x in r), axis=1)
    ]
    if "Datum" in df:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna(subset=["Datum"]).sort_values("Datum")
    return df.drop_duplicates(subset=["Datum"] + MAIN + EURO).reset_index(drop=True)

def refresh_from_url():
    r = requests.get(DATA_URL, timeout=30, headers={"User-Agent":"Eurojackpot-Predictor/2.1"})
    r.raise_for_status()
    x = pd.read_csv(io.BytesIO(r.content), header=None)
    if x.shape[1] < 9:
        raise ValueError("Remote data has an unexpected format.")
    x = x.iloc[:, :9]
    x.columns = ["draw_no","Datum"] + MAIN + EURO
    x["Datum"] = pd.to_datetime(x["Datum"], dayfirst=True, errors="coerce")
    return normalize(x)

def freq(df, cols):
    return Counter(df[cols].to_numpy().ravel())

def gaps(df, cols, maximum):
    a = df[cols].to_numpy()
    out = {}
    for n in range(1, maximum + 1):
        ix = np.where((a == n).any(axis=1))[0]
        out[n] = len(df)-1-ix[-1] if len(ix) else len(df)
    return out

def pair_freq(df, cols):
    c = Counter()
    for row in df[cols].to_numpy():
        c.update(combinations(sorted(row), 2))
    return c

def triple_freq(df, cols):
    c = Counter()
    for row in df[cols].to_numpy():
        c.update(combinations(sorted(row), 3))
    return c

def features(df):
    mf, ef = freq(df, MAIN), freq(df, EURO)
    mg, eg = gaps(df, MAIN, 50), gaps(df, EURO, 12)
    mp, ep = pair_freq(df, MAIN), pair_freq(df, EURO)
    mt, et = triple_freq(df, MAIN), triple_freq(df, EURO)
    recent = df.tail(min(52, len(df)))
    rm, re = freq(recent, MAIN), freq(recent, EURO)

    dm, de = Counter(), Counter()
    decay = max(10, len(df) * 0.15)
    for i, row in enumerate(df[MAIN].to_numpy()):
        w = math.exp((i - len(df) + 1) / decay)
        for n in row: dm[int(n)] += w
    for i, row in enumerate(df[EURO].to_numpy()):
        w = math.exp((i - len(df) + 1) / decay)
        for n in row: de[int(n)] += w

    return mf, ef, mg, eg, mp, ep, mt, et, rm, re, dm, de

def score(combo, f, g, pairs, triples, recent, decay, draws, mode):
    pair = sum(pairs[p] for p in combinations(sorted(combo),2))
    triple = sum(triples[t] for t in combinations(sorted(combo),3))
    fs = sum(f[x] for x in combo) / max(draws,1)
    gs = sum(min(g[x],30) for x in combo) / (30*len(combo))
    rs = sum(recent[x] for x in combo) / max(draws,1)
    ds = sum(decay[x] for x in combo)
    odd = 1 - abs(sum(x % 2 for x in combo)-len(combo)/2)/(len(combo)/2)
    spread = (max(combo)-min(combo))/49
    lowhigh = 1 - abs(sum(x <= 25 for x in combo)-len(combo)/2)/(len(combo)/2)

    if mode == "frequency": return fs
    if mode == "overdue": return gs
    if mode == "cold": return 1-fs
    if mode == "pairs": return pair/max(draws,1)
    if mode == "recent": return rs
    if mode == "decay": return ds
    if mode == "structure": return .35*odd+.35*spread+.30*lowhigh
    return (.22*fs + .16*gs + .18*rs + .12*pair/max(draws,1)
            + .08*triple/max(draws,1) + .10*odd + .07*spread + .07*lowhigh)

MODES = ["balanced","frequency","overdue","cold","pairs","recent","decay","structure"]

def predict(df, n, simulations, seed, modes):
    rng = random.Random(seed)
    mf,ef,mg,eg,mp,ep,mt,et,rm,re,dm,de = features(df)
    candidates = []
    for _ in range(simulations):
        m = tuple(sorted(rng.sample(range(1,51),5)))
        e = tuple(sorted(rng.sample(range(1,13),2)))
        sm = [score(m,mf,mg,mp,mt,rm,dm,len(df),mode) for mode in modes]
        se = [score(e,ef,eg,ep,et,re,de,len(df),mode) for mode in modes]
        candidates.append((float(np.mean(sm)+np.mean(se)),m,e))
    candidates.sort(reverse=True)
    selected=[]
    for item in candidates:
        if all(len(set(item[1]) & set(x[1])) <= 3 for x in selected):
            selected.append(item)
        if len(selected) >= n: break
    return selected

def save_predictions(pred):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank,(s,m,e) in enumerate(pred,1):
        rows.append({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "main": " ".join(map(str,m)),
            "euro": " ".join(map(str,e)),
            "score": s
        })
    new = pd.DataFrame(rows)
    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE)
        new = pd.concat([old,new], ignore_index=True)
    new.to_csv(HISTORY_FILE,index=False)

st.title("🎯 Eurojackpot Predictor 2.1")
st.caption("Statistisches Ranking und Backtesting – keine echte Vorhersage zukünftiger Zufallsziehungen.")

df = normalize(load_repo_data())

with st.sidebar:
    st.header("Daten")
    st.write(f"**{len(df):,} Ziehungen**")
    st.write(f"Letzte Daten: **{df.Datum.max().date()}**")
    if st.button("🔄 Datenfeed prüfen"):
        try:
            remote = refresh_from_url()
            added = len(remote) - len(df)
            st.success(f"Remote-Daten geladen: {len(remote)} Ziehungen. Differenz: {added:+d}.")
        except Exception as e:
            st.error(f"Update fehlgeschlagen: {e}")

    n = st.slider("Anzahl Tipps",1,30,10)
    sims = st.slider("Simulationen",2000,150000,30000,2000)
    seed = st.number_input("Seed",0,99999999,42)
    modes = st.multiselect("Modelle", MODES,
        default=["balanced","frequency","overdue","pairs","recent","decay","structure"])

if not modes:
    st.error("Mindestens ein Modell auswählen.")
    st.stop()

tab1,tab2,tab3,tab4=st.tabs(["🎯 Predictor","📊 Statistik","🧪 Backtesting","🗂️ History"])

with tab1:
    if st.button("🚀 V2.1 berechnen", type="primary"):
        pred = predict(df,n,sims,int(seed),modes)
        out = pd.DataFrame([
            {"Rang":i+1,"Hauptzahlen":" ".join(map(str,m)),
             "Eurozahlen":" ".join(map(str,e)),"Score":round(s,6)}
            for i,(s,m,e) in enumerate(pred)
        ])
        st.dataframe(out,hide_index=True,use_container_width=True)
        save_predictions(pred)
        st.download_button("⬇️ CSV",out.to_csv(index=False).encode(),"predictions_v2_1.csv","text/csv")

with tab2:
    mf,ef,mg,eg,mp,ep,*_ = features(df)
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Hauptzahlen")
        st.dataframe(pd.DataFrame({
            "Zahl":range(1,51),
            "Häufigkeit":[mf[x] for x in range(1,51)],
            "Gap":[mg[x] for x in range(1,51)]
        }).sort_values("Häufigkeit",ascending=False),hide_index=True,use_container_width=True)
    with c2:
        st.subheader("Eurozahlen")
        st.dataframe(pd.DataFrame({
            "Eurozahl":range(1,13),
            "Häufigkeit":[ef[x] for x in range(1,13)],
            "Gap":[eg[x] for x in range(1,13)]
        }).sort_values("Häufigkeit",ascending=False),hide_index=True,use_container_width=True)
    st.subheader("Häufigste Zahlenpaare")
    st.dataframe(pd.DataFrame([
        {"Paar":f"{a}-{b}","Treffer":v} for (a,b),v in mp.most_common(25)
    ]),hide_index=True,use_container_width=True)

with tab3:
    st.write("Walk-forward-Test: Jede Testziehung wird nur mit Daten trainiert, die davor lagen.")
    train = st.slider("Trainingsfenster",20,min(150,max(20,len(df)-5)),52)
    tests = st.slider("Testziehungen",5,min(100,max(5,len(df)-train)),30)
    if st.button("🧪 Test starten"):
        records=[]
        start=max(train,len(df)-tests)
        for i in range(start,len(df)):
            pred=predict(df.iloc[:i],10,2500,int(seed)+i,modes)
            actual=set(df.iloc[i][MAIN])
            euro=set(df.iloc[i][EURO])
            hits=[len(actual&set(x[1]))+len(euro&set(x[2])) for x in pred]
            records.append({"Datum":df.iloc[i]["Datum"],"Beste Treffer":max(hits),"Ø Treffer Top10":np.mean(hits)})
        bt=pd.DataFrame(records)
        st.metric("Ø Treffer Top-10",round(bt["Ø Treffer Top10"].mean(),3))
        st.metric("Bestes Ergebnis",int(bt["Beste Treffer"].max()))
        st.dataframe(bt,use_container_width=True,hide_index=True)

with tab4:
    if HISTORY_FILE.exists():
        st.dataframe(pd.read_csv(HISTORY_FILE).tail(200),use_container_width=True,hide_index=True)
    else:
        st.info("Noch keine gespeicherten Predictions.")

with st.expander("Methodik & Datenherkunft"):
    st.markdown("""
**Modelle:** Häufigkeit, Overdue, Cold, Paare, Triples, Recent, exponentieller Decay, Struktur und Balanced Ensemble.

**Regimewechsel:** Eurojackpot startete am 23.03.2012. Am 10.10.2014 wurde von 2 aus 8 auf 2 aus 10 Eurozahlen erweitert; am 25.03.2022 kamen Eurozahlen 11 und 12 hinzu.

**Daten:** Die mitgelieferte CSV wurde aus einem öffentlich verfügbaren historischen Datensatz normalisiert. WestLotto stellt ebenfalls historische Gewinnzahlen und Downloads bereit. Für eine vollständig offizielle Produktionspipeline sollte der WestLotto-Download regelmäßig manuell bzw. über einen stabilen offiziellen Endpunkt gegengeprüft werden.

**Wichtig:** WestLotto weist selbst darauf hin, dass die Wahrscheinlichkeit einer auswählbaren Zahl nicht davon abhängt, wann sie zuletzt gezogen wurde. Scores sind daher Rankingwerte und keine echten Gewinnwahrscheinlichkeiten.
""")
