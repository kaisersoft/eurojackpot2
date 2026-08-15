"""
Eurojackpot Predictor 2.2
Umschalter: Eurojackpot (Default) | LOTTO 6aus49
Getrennte Zahlenräume, Historien und Modelle.
Statistisches Ranking – keine echte Vorhersage.
"""

import io
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Eurojackpot Predictor 2.2", page_icon="🎯", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Spiel-Konfiguration
# ---------------------------------------------------------------------------
GAMES = {
    "eurojackpot": {
        "label": "Eurojackpot",
        "main_cols": [f"zahl{i}" for i in range(1, 6)],
        "bonus_cols": ["euro1", "euro2"],
        "main_count": 5,
        "main_max": 50,
        "bonus_count": 2,
        "bonus_max": 12,
        "bonus_label": "Eurozahlen",
        "bonus_range_label": "1–12",
        "local_file": DATA_DIR / "eurojackpot_history.csv",
        "history_file": DATA_DIR / "prediction_history_ej.csv",
        "data_url": "https://raw.githubusercontent.com/rescue3dcom-hub/lotto-data/main/eurojackpot.csv",
        "draw_weekdays": [1, 4],  # Di, Fr
        "draw_names": {1: "Dienstag", 4: "Freitag"},
    },
    "6aus49": {
        "label": "LOTTO 6aus49",
        "main_cols": [f"zahl{i}" for i in range(1, 7)],
        "bonus_cols": ["superzahl"],
        "main_count": 6,
        "main_max": 49,
        "bonus_count": 1,
        "bonus_max": 9,  # Superzahl 0–9
        "bonus_label": "Superzahl",
        "bonus_range_label": "0–9",
        "local_file": DATA_DIR / "lotto6aus49_history.csv",
        "history_file": DATA_DIR / "prediction_history_6aus49.csv",
        "data_url": "https://raw.githubusercontent.com/daowa89/lottery-archive/main/de/lotto_6aus49/results.csv",
        "draw_weekdays": [2, 5],  # Mi, Sa
        "draw_names": {2: "Mittwoch", 5: "Samstag"},
    },
}

MODES = ["balanced", "frequency", "overdue", "cold", "pairs", "recent", "decay", "structure"]


# ---------------------------------------------------------------------------
# Daten laden / normalisieren
# ---------------------------------------------------------------------------
def normalize_ej(df: pd.DataFrame) -> pd.DataFrame:
    cfg = GAMES["eurojackpot"]
    main, bonus = cfg["main_cols"], cfg["bonus_cols"]
    df = df.copy()
    for c in main + bonus:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=main + bonus)
    for c in main + bonus:
        df[c] = df[c].astype(int)
    df = df[
        df[main].apply(lambda r: len(set(r)) == 5 and all(1 <= x <= 50 for x in r), axis=1)
        & df[bonus].apply(lambda r: len(set(r)) == 2 and all(1 <= x <= 12 for x in r), axis=1)
    ]
    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna(subset=["Datum"]).sort_values("Datum")
    return df.drop_duplicates(subset=["Datum"] + main + bonus).reset_index(drop=True)


def normalize_6aus49(df: pd.DataFrame) -> pd.DataFrame:
    """Erwartet Spalten: date/Datum, n1..n6 oder zahl1..zahl6, superzahl."""
    df = df.copy()
    # Spalten vereinheitlichen
    rename = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if "date" in cols_lower:
        rename[cols_lower["date"]] = "Datum"
    for i in range(1, 7):
        for key in (f"n{i}", f"zahl{i}"):
            if key in cols_lower:
                rename[cols_lower[key]] = f"zahl{i}"
    if "superzahl" in cols_lower:
        rename[cols_lower["superzahl"]] = "superzahl"
    df = df.rename(columns=rename)

    main = [f"zahl{i}" for i in range(1, 7)]
    need = main + ["superzahl", "Datum"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"6aus49-Daten: Spalte '{c}' fehlt.")

    for c in main + ["superzahl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Superzahl kann in frühen Jahren fehlen
    df = df.dropna(subset=main)
    df["superzahl"] = df["superzahl"].fillna(-1).astype(int)
    for c in main:
        df[c] = df[c].astype(int)

    df = df[
        df[main].apply(lambda r: len(set(r)) == 6 and all(1 <= x <= 49 for x in r), axis=1)
    ]
    # Nur Ziehungen mit gültiger Superzahl (0–9) für Scoring behalten
    df_valid_sz = df[df["superzahl"].between(0, 9)].copy()
    if len(df_valid_sz) < 50:
        # Fallback: alle behalten, Superzahl 0 setzen wo fehlend
        df.loc[~df["superzahl"].between(0, 9), "superzahl"] = 0
        df_valid_sz = df

    df_valid_sz["Datum"] = pd.to_datetime(df_valid_sz["Datum"], errors="coerce")
    df_valid_sz = df_valid_sz.dropna(subset=["Datum"]).sort_values("Datum")
    return df_valid_sz.drop_duplicates(subset=["Datum"] + main + ["superzahl"]).reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_local(game_key: str):
    cfg = GAMES[game_key]
    path = cfg["local_file"]
    if not path.exists():
        return None
    if game_key == "eurojackpot":
        # Originalformat mit ;
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path)
        return normalize_ej(df)
    else:
        df = pd.read_csv(path)
        return normalize_6aus49(df)


def refresh_from_url(game_key: str) -> pd.DataFrame:
    cfg = GAMES[game_key]
    r = requests.get(
        cfg["data_url"],
        timeout=30,
        headers={"User-Agent": "Eurojackpot-Predictor/2.2"},
    )
    r.raise_for_status()

    if game_key == "eurojackpot":
        x = pd.read_csv(io.BytesIO(r.content), header=None)
        if x.shape[1] < 9:
            raise ValueError("Remote Eurojackpot-Daten unerwartetes Format.")
        x = x.iloc[:, :9]
        x.columns = ["draw_no", "Datum"] + cfg["main_cols"] + cfg["bonus_cols"]
        x["Datum"] = pd.to_datetime(x["Datum"], dayfirst=True, errors="coerce")
        df = normalize_ej(x)
    else:
        x = pd.read_csv(io.BytesIO(r.content))
        df = normalize_6aus49(x)

    # lokal speichern
    cfg["local_file"].parent.mkdir(parents=True, exist_ok=True)
    if game_key == "eurojackpot":
        df.to_csv(cfg["local_file"], sep=";", index=False)
    else:
        df.to_csv(cfg["local_file"], index=False)
    return df


def ensure_data(game_key: str) -> pd.DataFrame:
    df = load_local(game_key)
    if df is not None and len(df) > 0:
        return df
    # Versuch Remote
    try:
        return refresh_from_url(game_key)
    except Exception as e:
        st.error(f"Keine lokalen Daten und Remote-Laden fehlgeschlagen: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Features & Scoring (spielunabhängig über Config)
# ---------------------------------------------------------------------------
def freq(df, cols):
    return Counter(df[cols].to_numpy().ravel())


def gaps(df, cols, maximum, minimum=1):
    a = df[cols].to_numpy()
    out = {}
    for n in range(minimum, maximum + 1):
        ix = np.where((a == n).any(axis=1))[0]
        out[n] = len(df) - 1 - ix[-1] if len(ix) else len(df)
    return out


def pair_freq(df, cols):
    c = Counter()
    for row in df[cols].to_numpy():
        c.update(combinations(sorted(int(x) for x in row), 2))
    return c


def triple_freq(df, cols):
    c = Counter()
    for row in df[cols].to_numpy():
        c.update(combinations(sorted(int(x) for x in row), 3))
    return c


def features(df, cfg):
    main, bonus = cfg["main_cols"], cfg["bonus_cols"]
    mf = freq(df, main)
    bf = freq(df, bonus)
    mg = gaps(df, main, cfg["main_max"], 1)
    # Superzahl 0–9 vs Euro 1–12
    b_min = 0 if cfg["bonus_max"] == 9 else 1
    bg = gaps(df, bonus, cfg["bonus_max"], b_min)
    mp = pair_freq(df, main)
    bp = pair_freq(df, bonus) if cfg["bonus_count"] >= 2 else Counter()
    mt = triple_freq(df, main)
    bt = Counter()
    recent = df.tail(min(52, len(df)))
    rm = freq(recent, main)
    rb = freq(recent, bonus)

    dm, db = Counter(), Counter()
    decay = max(10, len(df) * 0.15)
    for i, row in enumerate(df[main].to_numpy()):
        w = math.exp((i - len(df) + 1) / decay)
        for n in row:
            dm[int(n)] += w
    for i, row in enumerate(df[bonus].to_numpy()):
        w = math.exp((i - len(df) + 1) / decay)
        for n in np.atleast_1d(row):
            db[int(n)] += w

    return mf, bf, mg, bg, mp, bp, mt, bt, rm, rb, dm, db


def score_combo(combo, f, g, pairs, triples, recent, decay, draws, mode, main_max):
    combo = tuple(sorted(combo))
    pair = sum(pairs[p] for p in combinations(combo, 2)) if pairs else 0
    triple = sum(triples[t] for t in combinations(combo, 3)) if triples and len(combo) >= 3 else 0
    fs = sum(f[x] for x in combo) / max(draws, 1)
    gs = sum(min(g.get(x, draws), 30) for x in combo) / (30 * len(combo))
    rs = sum(recent[x] for x in combo) / max(draws, 1)
    ds = sum(decay[x] for x in combo)
    odd = 1 - abs(sum(x % 2 for x in combo) - len(combo) / 2) / (len(combo) / 2)
    spread = (max(combo) - min(combo)) / max(main_max - 1, 1)
    mid = main_max / 2
    lowhigh = 1 - abs(sum(x <= mid for x in combo) - len(combo) / 2) / (len(combo) / 2)

    if mode == "frequency":
        return fs
    if mode == "overdue":
        return gs
    if mode == "cold":
        return 1 - fs
    if mode == "pairs":
        return pair / max(draws, 1)
    if mode == "recent":
        return rs
    if mode == "decay":
        return ds
    if mode == "structure":
        return 0.35 * odd + 0.35 * spread + 0.30 * lowhigh
    # balanced
    return (
        0.22 * fs
        + 0.16 * gs
        + 0.18 * rs
        + 0.12 * pair / max(draws, 1)
        + 0.08 * triple / max(draws, 1)
        + 0.10 * odd
        + 0.07 * spread
        + 0.07 * lowhigh
    )


def predict(df, cfg, n, simulations, seed, modes):
    rng = random.Random(seed)
    mf, bf, mg, bg, mp, bp, mt, bt, rm, rb, dm, db = features(df, cfg)
    main_count = cfg["main_count"]
    main_max = cfg["main_max"]
    bonus_count = cfg["bonus_count"]
    bonus_max = cfg["bonus_max"]
    b_min = 0 if bonus_max == 9 else 1

    candidates = []
    for _ in range(simulations):
        m = tuple(sorted(rng.sample(range(1, main_max + 1), main_count)))
        if bonus_count == 2:
            b = tuple(sorted(rng.sample(range(b_min, bonus_max + 1), bonus_count)))
        else:
            b = (rng.randint(b_min, bonus_max),)

        sm = [score_combo(m, mf, mg, mp, mt, rm, dm, len(df), mode, main_max) for mode in modes]
        sb = [
            score_combo(b, bf, bg, bp, bt, rb, db, len(df), mode, bonus_max) for mode in modes
        ]
        candidates.append((float(np.mean(sm) + np.mean(sb)), m, b))

    candidates.sort(reverse=True)
    selected = []
    overlap_limit = 3 if main_count == 5 else 4
    for item in candidates:
        if all(len(set(item[1]) & set(x[1])) <= overlap_limit for x in selected):
            selected.append(item)
        if len(selected) >= n:
            break
    return selected


def save_predictions(pred, cfg):
    path = cfg["history_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (s, m, b) in enumerate(pred, 1):
        rows.append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rank": rank,
                "main": " ".join(map(str, m)),
                "bonus": " ".join(map(str, b)),
                "score": s,
            }
        )
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def next_draw_dates(weekdays, n=3):
    out = []
    d = date.today()
    for _ in range(60):
        if d.weekday() in weekdays:
            out.append(d)
            if len(out) >= n:
                break
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎯 Eurojackpot Predictor 2.2")
st.caption(
    "Statistisches Ranking & Backtesting – keine echte Vorhersage. "
    "Eurojackpot und 6aus49 mit getrennten Daten und Zahlenräumen."
)

game_key = st.radio(
    "Spielmodus",
    options=["eurojackpot", "6aus49"],
    format_func=lambda k: GAMES[k]["label"],
    index=0,
    horizontal=True,
    help="Eurojackpot: 5 aus 50 + 2 Eurozahlen. 6aus49: 6 aus 49 + Superzahl.",
)
cfg = GAMES[game_key]

# Nächste Ziehungen (Info, keine 3 Score-Slots)
draws = next_draw_dates(cfg["draw_weekdays"], n=3)
draw_str = " · ".join(
    f"{cfg['draw_names'].get(d.weekday(), d.strftime('%a'))} {d.strftime('%d.%m.')}" for d in draws
)
st.info(
    f"**{cfg['label']}** · Hauptzahlen: {cfg['main_count']} aus {cfg['main_max']} · "
    f"{cfg['bonus_label']}: {cfg['bonus_range_label']} · Nächste Ziehungen: {draw_str}"
)

df = ensure_data(game_key)

with st.sidebar:
    st.header("Daten")
    st.write(f"**Spiel:** {cfg['label']}")
    st.write(f"**{len(df):,} Ziehungen**")
    st.write(f"Letzte Daten: **{df.Datum.max().date()}**")
    if st.button("🔄 Datenfeed prüfen / aktualisieren"):
        try:
            remote = refresh_from_url(game_key)
            load_local.clear()
            st.success(f"Remote geladen: {len(remote)} Ziehungen.")
            st.rerun()
        except Exception as e:
            st.error(f"Update fehlgeschlagen: {e}")

    n = st.slider("Anzahl Tipps", 1, 30, 10)
    sims = st.slider("Simulationen", 2000, 150000, 30000, 2000)
    seed = st.number_input("Seed", 0, 99999999, 42)
    modes = st.multiselect(
        "Modelle",
        MODES,
        default=["balanced", "frequency", "overdue", "pairs", "recent", "decay", "structure"],
    )

if not modes:
    st.error("Mindestens ein Modell auswählen.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictor", "📊 Statistik", "🧪 Backtesting", "🗂️ History"])

with tab1:
    if st.button("🚀 Berechnen", type="primary"):
        pred = predict(df, cfg, n, sims, int(seed), modes)
        rows = []
        for i, (s, m, b) in enumerate(pred):
            row = {
                "Rang": i + 1,
                "Hauptzahlen": " ".join(map(str, m)),
                cfg["bonus_label"]: " ".join(map(str, b)),
                "Score": round(s, 6),
            }
            rows.append(row)
        out = pd.DataFrame(rows)
        st.dataframe(out, hide_index=True, use_container_width=True)
        save_predictions(pred, cfg)
        st.download_button(
            "⬇️ CSV",
            out.to_csv(index=False).encode(),
            f"predictions_{game_key}.csv",
            "text/csv",
        )

with tab2:
    mf, bf, mg, bg, mp, bp, *_ = features(df, cfg)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hauptzahlen")
        st.dataframe(
            pd.DataFrame(
                {
                    "Zahl": range(1, cfg["main_max"] + 1),
                    "Häufigkeit": [mf[x] for x in range(1, cfg["main_max"] + 1)],
                    "Gap": [mg[x] for x in range(1, cfg["main_max"] + 1)],
                }
            ).sort_values("Häufigkeit", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.subheader(cfg["bonus_label"])
        b_min = 0 if cfg["bonus_max"] == 9 else 1
        st.dataframe(
            pd.DataFrame(
                {
                    cfg["bonus_label"]: range(b_min, cfg["bonus_max"] + 1),
                    "Häufigkeit": [bf[x] for x in range(b_min, cfg["bonus_max"] + 1)],
                    "Gap": [bg[x] for x in range(b_min, cfg["bonus_max"] + 1)],
                }
            ).sort_values("Häufigkeit", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Häufigste Zahlenpaare (Hauptzahlen)")
    st.dataframe(
        pd.DataFrame(
            [{"Paar": f"{a}-{b}", "Treffer": v} for (a, b), v in mp.most_common(25)]
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab3:
    st.write("Walk-forward-Test: Jede Testziehung nur mit Daten davor.")
    train = st.slider("Trainingsfenster", 20, min(150, max(20, len(df) - 5)), 52)
    tests = st.slider("Testziehungen", 5, min(100, max(5, len(df) - train)), 30)
    if st.button("🧪 Test starten"):
        records = []
        start = max(train, len(df) - tests)
        main, bonus = cfg["main_cols"], cfg["bonus_cols"]
        for i in range(start, len(df)):
            pred = predict(df.iloc[:i], cfg, 10, 2500, int(seed) + i, modes)
            actual_m = set(int(x) for x in df.iloc[i][main].tolist())
            raw_b = df.iloc[i][bonus]
            if isinstance(raw_b, pd.Series):
                actual_b = set(int(x) for x in raw_b.tolist())
            else:
                actual_b = {int(raw_b)}
            hits = [len(actual_m & set(x[1])) + len(actual_b & set(x[2])) for x in pred]
            records.append(
                {
                    "Datum": df.iloc[i]["Datum"],
                    "Beste Treffer": max(hits) if hits else 0,
                    "Ø Treffer Top10": float(np.mean(hits)) if hits else 0,
                }
            )
        bt = pd.DataFrame(records)
        st.metric("Ø Treffer Top-10", round(bt["Ø Treffer Top10"].mean(), 3))
        st.metric("Bestes Ergebnis", int(bt["Beste Treffer"].max()))
        st.dataframe(bt, use_container_width=True, hide_index=True)

with tab4:
    hf = cfg["history_file"]
    if hf.exists():
        st.dataframe(pd.read_csv(hf).tail(200), use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine gespeicherten Predictions für dieses Spiel.")

with st.expander("Methodik & Datenherkunft"):
    st.markdown(
        f"""
**Aktueller Modus:** {cfg['label']}

| | Eurojackpot | 6aus49 |
|--|-------------|--------|
| Hauptzahlen | 5 aus 50 | 6 aus 49 |
| Zusatz | 2 Eurozahlen (1–12) | 1 Superzahl (0–9) |
| Ziehungen | Di + Fr | Mi + Sa |
| Daten-URL | rescue3dcom-hub/lotto-data | daowa89/lottery-archive |

**Modelle:** Häufigkeit, Overdue, Cold, Paare, Recent, Decay, Struktur, Balanced.

**Wichtig:** Scores sind Rankingwerte, keine Gewinnwahrscheinlichkeiten. Zufallsziehungen sind nicht vorhersagbar.
"""
    )
