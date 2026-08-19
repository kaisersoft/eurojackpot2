"""
Eurojackpot Predictor 2.3
Umschalter: Eurojackpot (Default) | LOTTO 6aus49

Änderungen gegenüber 2.2:
- Gewichtetes / paar-gestütztes Sampling statt Blind-Monte-Carlo
- Modi getrennt (Ensemble), nicht gegeneinander gemittelt
- Eigenes Bonus-/Superzahl-Scoring
- Walk-forward-Backtest mit Zufalls-Baseline
- Fehlende Superzahlen nicht auf 0 gesetzt
- Seed an nächstes Ziehungsdatum
- History mit Spiel, Seed, Modellen, Datenstand
- Datenqualität, letzte Ziehung, Diversitätsfilter

Statistisches Ranking – keine echte Vorhersage.
"""

from __future__ import annotations

import io
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Eurojackpot Predictor 2.3",
    page_icon="🎯",
    layout="wide",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

GAMES = {
    "eurojackpot": {
        "label": "Eurojackpot",
        "main_cols": [f"zahl{i}" for i in range(1, 6)],
        "bonus_cols": ["euro1", "euro2"],
        "main_count": 5,
        "main_max": 50,
        "bonus_count": 2,
        "bonus_max": 12,
        "bonus_min": 1,
        "bonus_label": "Eurozahlen",
        "bonus_range_label": "1–12",
        "local_file": DATA_DIR / "eurojackpot_history.csv",
        "history_file": DATA_DIR / "prediction_history_ej.csv",
        "data_url": "https://raw.githubusercontent.com/rescue3dcom-hub/lotto-data/main/eurojackpot.csv",
        "draw_weekdays": [1, 4],
        "draw_names": {1: "Dienstag", 4: "Freitag"},
        "overlap_limit": 2,
    },
    "6aus49": {
        "label": "LOTTO 6aus49",
        "main_cols": [f"zahl{i}" for i in range(1, 7)],
        "bonus_cols": ["superzahl"],
        "main_count": 6,
        "main_max": 49,
        "bonus_count": 1,
        "bonus_max": 9,
        "bonus_min": 0,
        "bonus_label": "Superzahl",
        "bonus_range_label": "0–9",
        "local_file": DATA_DIR / "lotto6aus49_history.csv",
        "history_file": DATA_DIR / "prediction_history_6aus49.csv",
        "data_url": "https://raw.githubusercontent.com/daowa89/lottery-archive/main/de/lotto_6aus49/results.csv",
        "draw_weekdays": [2, 5],
        "draw_names": {2: "Mittwoch", 5: "Samstag"},
        "overlap_limit": 3,
    },
}

MODES = [
    "balanced",
    "frequency",
    "overdue",
    "cold",
    "pairs",
    "recent",
    "decay",
    "structure",
]

MODE_HELP = {
    "balanced": "Mischung aus Häufigkeit, Recency, Gap und Struktur – nur innerhalb dieses Modus.",
    "frequency": "Oft gezogene Zahlen.",
    "overdue": "Lange nicht gezogene Zahlen (Heuristik, kein Rand).",
    "cold": "Selten gezogene Zahlen.",
    "pairs": "Historisch häufige Zahlenpaare.",
    "recent": "Häufigkeit der letzten ~52 Ziehungen.",
    "decay": "Exponentiell gewichtete jüngere Ziehungen.",
    "structure": "Gerade/ungerade, Spread, Low/High der Combo.",
}


# ---------------------------------------------------------------------------
# Daten
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
    """Hauptzahlen immer; Superzahl nur behalten, wenn 0–9. Kein Fill mit 0."""
    df = df.copy()
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
    need = main + ["Datum"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"6aus49-Daten: Spalte '{c}' fehlt.")
    if "superzahl" not in df.columns:
        df["superzahl"] = np.nan

    for c in main:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["superzahl"] = pd.to_numeric(df["superzahl"], errors="coerce")
    df = df.dropna(subset=main)
    for c in main:
        df[c] = df[c].astype(int)

    df = df[df[main].apply(lambda r: len(set(r)) == 6 and all(1 <= x <= 49 for x in r), axis=1)]
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
    df = df.dropna(subset=["Datum"]).sort_values("Datum")
    df = df.drop_duplicates(subset=["Datum"] + main).reset_index(drop=True)
    df["_sz_valid"] = df["superzahl"].between(0, 9)
    return df


def data_quality_report(df: pd.DataFrame, cfg: dict) -> list[str]:
    notes = []
    if df is None or len(df) == 0:
        return ["Keine Ziehungen geladen."]
    dates = df["Datum"].sort_values()
    notes.append(f"{len(df)} gültige Ziehungen ({dates.min().date()} – {dates.max().date()}).")
    if dates.duplicated().any():
        notes.append(f"Warnung: {int(dates.duplicated().sum())} doppelte Datumsangaben.")
    deltas = dates.diff().dt.days.dropna()
    if len(deltas) and (deltas > 21).any():
        notes.append(f"Warnung: {int((deltas > 21).sum())} Lücke(n) > 21 Tage in der Historie.")
    if cfg["bonus_count"] == 1 and "_sz_valid" in df.columns:
        missing = int((~df["_sz_valid"]).sum())
        if missing:
            notes.append(f"{missing} Ziehungen ohne gültige Superzahl – SZ-Statistik nur über gültige.")
    return notes


@st.cache_data(ttl=3600)
def load_local(game_key: str):
    cfg = GAMES[game_key]
    path = cfg["local_file"]
    if not path.exists():
        return None
    if game_key == "eurojackpot":
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path)
        return normalize_ej(df)
    df = pd.read_csv(path)
    return normalize_6aus49(df)


def refresh_from_url(game_key: str) -> pd.DataFrame:
    cfg = GAMES[game_key]
    r = requests.get(
        cfg["data_url"],
        timeout=30,
        headers={"User-Agent": "Eurojackpot-Predictor/2.3"},
    )
    r.raise_for_status()

    if game_key == "eurojackpot":
        x = pd.read_csv(io.BytesIO(r.content), header=None)
        if x.shape[1] < 9:
            raise ValueError("Remote Eurojackpot-Daten: unerwartetes Format.")
        x = x.iloc[:, :9]
        x.columns = ["draw_no", "Datum"] + cfg["main_cols"] + cfg["bonus_cols"]
        x["Datum"] = pd.to_datetime(x["Datum"], dayfirst=True, errors="coerce")
        df = normalize_ej(x)
    else:
        x = pd.read_csv(io.BytesIO(r.content))
        df = normalize_6aus49(x)

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
    try:
        return refresh_from_url(game_key)
    except Exception as e:
        st.error(f"Keine lokalen Daten und Remote-Laden fehlgeschlagen: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def freq(df, cols):
    return Counter(int(x) for x in df[cols].to_numpy().ravel())


def gaps(df, cols, maximum, minimum=1):
    a = df[cols].to_numpy()
    out = {}
    for n in range(minimum, maximum + 1):
        ix = np.where((a == n).any(axis=1))[0]
        out[n] = int(len(df) - 1 - ix[-1]) if len(ix) else int(len(df))
    return out


def pair_freq(df, cols):
    c = Counter()
    for row in df[cols].to_numpy():
        vals = sorted(int(x) for x in np.atleast_1d(row))
        if len(vals) >= 2:
            c.update(combinations(vals, 2))
    return c


def triple_freq(df, cols):
    c = Counter()
    for row in df[cols].to_numpy():
        vals = sorted(int(x) for x in np.atleast_1d(row))
        if len(vals) >= 3:
            c.update(combinations(vals, 3))
    return c


def features(df, cfg):
    main, bonus = cfg["main_cols"], cfg["bonus_cols"]
    b_min, b_max = cfg["bonus_min"], cfg["bonus_max"]

    df_main = df
    if "_sz_valid" in df.columns:
        df_bonus = df[df["_sz_valid"]].copy()
        if len(df_bonus) == 0:
            df_bonus = df
    else:
        df_bonus = df

    mf = freq(df_main, main)
    bf = freq(df_bonus, bonus)
    mg = gaps(df_main, main, cfg["main_max"], 1)
    bg = gaps(df_bonus, bonus, b_max, b_min)
    mp = pair_freq(df_main, main)
    bp = pair_freq(df_bonus, bonus) if cfg["bonus_count"] >= 2 else Counter()
    mt = triple_freq(df_main, main)

    recent_n = min(52, len(df_main))
    rm = freq(df_main.tail(recent_n), main)
    rb = freq(df_bonus.tail(min(52, len(df_bonus))), bonus)

    dm, db = Counter(), Counter()
    decay_m = max(10.0, len(df_main) * 0.15)
    for i, row in enumerate(df_main[main].to_numpy()):
        w = math.exp((i - len(df_main) + 1) / decay_m)
        for n in row:
            dm[int(n)] += w
    decay_b = max(10.0, len(df_bonus) * 0.15)
    for i, row in enumerate(df_bonus[bonus].to_numpy()):
        w = math.exp((i - len(df_bonus) + 1) / decay_b)
        for n in np.atleast_1d(row):
            db[int(n)] += w

    last_main = set()
    last_bonus = set()
    if len(df_main):
        last_main = {int(x) for x in df_main.iloc[-1][main].tolist()}
    if len(df_bonus):
        raw = df_bonus.iloc[-1][bonus]
        if isinstance(raw, pd.Series):
            last_bonus = {int(x) for x in raw.tolist()}
        else:
            last_bonus = {int(raw)}

    return {
        "mf": mf,
        "bf": bf,
        "mg": mg,
        "bg": bg,
        "mp": mp,
        "bp": bp,
        "mt": mt,
        "rm": rm,
        "rb": rb,
        "dm": dm,
        "db": db,
        "n_main": len(df_main),
        "n_bonus": len(df_bonus),
        "last_main": last_main,
        "last_bonus": last_bonus,
        "last_date": df_main["Datum"].max() if len(df_main) else None,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _norm_decay(decay: Counter, combo) -> float:
    if not combo:
        return 0.0
    mx = max(decay.values()) if decay else 1.0
    return sum(decay[x] for x in combo) / (mx * len(combo) + 1e-12)


def score_main_combo(combo, feat, mode, main_max, damp_last: set | None):
    combo = tuple(sorted(combo))
    f, g, pairs, triples = feat["mf"], feat["mg"], feat["mp"], feat["mt"]
    recent, decay, draws = feat["rm"], feat["dm"], feat["n_main"]
    pair = sum(pairs[p] for p in combinations(combo, 2)) if pairs else 0
    triple = sum(triples[t] for t in combinations(combo, 3)) if triples and len(combo) >= 3 else 0
    fs = sum(f[x] for x in combo) / max(draws, 1)
    gs = sum(min(g.get(x, draws), 30) for x in combo) / (30 * len(combo))
    rs = sum(recent[x] for x in combo) / max(draws, 1)
    ds = _norm_decay(decay, combo)
    odd = 1 - abs(sum(x % 2 for x in combo) - len(combo) / 2) / max(len(combo) / 2, 1e-9)
    spread = (max(combo) - min(combo)) / max(main_max - 1, 1)
    mid = main_max / 2
    lowhigh = 1 - abs(sum(x <= mid for x in combo) - len(combo) / 2) / max(len(combo) / 2, 1e-9)

    if mode == "frequency":
        s = fs
    elif mode == "overdue":
        s = gs
    elif mode == "cold":
        s = 1 - fs
    elif mode == "pairs":
        s = pair / max(draws, 1)
    elif mode == "recent":
        s = rs
    elif mode == "decay":
        s = ds
    elif mode == "structure":
        s = 0.35 * odd + 0.35 * spread + 0.30 * lowhigh
    else:
        s = (
            0.26 * fs
            + 0.12 * gs
            + 0.20 * rs
            + 0.14 * pair / max(draws, 1)
            + 0.08 * triple / max(draws, 1)
            + 0.08 * odd
            + 0.06 * spread
            + 0.06 * lowhigh
        )

    if damp_last:
        overlap = len(set(combo) & damp_last)
        s *= 0.92 ** overlap
    return float(s)


def score_bonus_combo(combo, feat, mode):
    """Nur Frequenz / Gap / Recency / Decay – keine Struktur."""
    combo = tuple(sorted(int(x) for x in combo))
    f, g, recent, decay, draws = feat["bf"], feat["bg"], feat["rb"], feat["db"], feat["n_bonus"]
    pairs = feat["bp"]
    fs = sum(f[x] for x in combo) / max(draws, 1)
    gs = sum(min(g.get(x, draws), 30) for x in combo) / (30 * max(len(combo), 1))
    rs = sum(recent[x] for x in combo) / max(draws, 1)
    ds = _norm_decay(decay, combo)
    pair = sum(pairs[p] for p in combinations(combo, 2)) if pairs and len(combo) >= 2 else 0

    if mode == "frequency":
        return float(fs)
    if mode == "overdue":
        return float(gs)
    if mode == "cold":
        return float(1 - fs)
    if mode == "pairs":
        return float(pair / max(draws, 1)) if len(combo) >= 2 else float(fs)
    if mode == "recent":
        return float(rs)
    if mode == "decay":
        return float(ds)
    if mode == "structure":
        return float(0.5 * fs + 0.5 * rs)
    return float(0.34 * fs + 0.18 * gs + 0.28 * rs + 0.20 * ds)


def number_weights(values, f, g, recent, decay, draws, mode) -> np.ndarray:
    w = []
    mx = max(decay.values()) if decay else 1.0
    for x in values:
        fs = f[x] / max(draws, 1)
        gs = min(g.get(x, draws), 30) / 30.0
        rs = recent[x] / max(draws, 1)
        ds = decay[x] / (mx + 1e-12)
        if mode == "frequency":
            val = fs
        elif mode == "overdue":
            val = gs
        elif mode == "cold":
            val = 1.0 - fs
        elif mode == "recent":
            val = rs
        elif mode == "decay":
            val = ds
        elif mode == "pairs":
            val = 0.5 * fs + 0.5 * rs
        elif mode == "structure":
            val = 1.0
        else:
            val = 0.35 * fs + 0.20 * gs + 0.25 * rs + 0.20 * ds
        w.append(max(float(val), 1e-9))
    arr = np.asarray(w, dtype=float)
    arr /= arr.sum()
    return arr


def _sample_combo(rng: np.random.Generator, values: np.ndarray, k: int, p: np.ndarray):
    pick = rng.choice(values, size=k, replace=False, p=p)
    return tuple(sorted(int(x) for x in pick))


def _sample_from_pairs(rng, values, k, p, top_pairs):
    if not top_pairs or k < 2:
        return _sample_combo(rng, values, k, p)
    a, b = top_pairs[int(rng.integers(0, len(top_pairs)))]
    chosen = {int(a), int(b)}
    rest_vals = np.array([x for x in values if int(x) not in chosen], dtype=int)
    if len(rest_vals) < k - 2:
        return _sample_combo(rng, values, k, p)
    rest_p = np.array([p[int(np.where(values == x)[0][0])] for x in rest_vals], dtype=float)
    rest_p = np.maximum(rest_p, 1e-12)
    rest_p /= rest_p.sum()
    extra = rng.choice(rest_vals, size=k - 2, replace=False, p=rest_p)
    chosen.update(int(x) for x in extra)
    return tuple(sorted(chosen))


def generate_candidates(rng, cfg, feat, mode, n_cands, damp_last: bool):
    main_vals = np.arange(1, cfg["main_max"] + 1)
    bonus_vals = np.arange(cfg["bonus_min"], cfg["bonus_max"] + 1)
    pm = number_weights(
        main_vals, feat["mf"], feat["mg"], feat["rm"], feat["dm"], feat["n_main"], mode
    )
    pb = number_weights(
        bonus_vals, feat["bf"], feat["bg"], feat["rb"], feat["db"], feat["n_bonus"], mode
    )
    top_pairs = [p for p, _ in feat["mp"].most_common(50)]
    last_m = feat["last_main"] if damp_last else None

    out = []
    seen = set()
    for i in range(n_cands):
        r = i / max(n_cands, 1)
        if mode == "pairs" and top_pairs and r < 0.55:
            m = _sample_from_pairs(rng, main_vals, cfg["main_count"], pm, top_pairs)
        elif r < 0.72:
            m = _sample_combo(rng, main_vals, cfg["main_count"], pm)
        elif r < 0.88 and top_pairs:
            m = _sample_from_pairs(rng, main_vals, cfg["main_count"], pm, top_pairs)
        else:
            m = tuple(sorted(int(x) for x in rng.choice(main_vals, cfg["main_count"], replace=False)))

        if cfg["bonus_count"] == 2:
            b = _sample_combo(rng, bonus_vals, 2, pb)
        else:
            b = (int(rng.choice(bonus_vals, p=pb)),)

        key = (m, b)
        if key in seen:
            continue
        seen.add(key)
        sm = score_main_combo(m, feat, mode, cfg["main_max"], last_m)
        sb = score_bonus_combo(b, feat, mode)
        out.append((sm + 0.35 * sb, m, b, mode))
    return out


def select_diverse(candidates, n, overlap_limit, max_pair_reuse=2):
    selected = []
    pair_count = Counter()
    for item in candidates:
        combo = item[1]
        if any(len(set(combo) & set(s[1])) > overlap_limit for s in selected):
            continue
        pairs = list(combinations(combo, 2))
        if selected and any(pair_count[p] >= max_pair_reuse for p in pairs):
            continue
        selected.append(item)
        for p in pairs:
            pair_count[p] += 1
        if len(selected) >= n:
            return selected
    for item in candidates:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= n:
            break
    return selected


def predict(df, cfg, n, simulations, seed, modes, damp_last=False):
    rng = np.random.default_rng(int(seed))
    feat = features(df, cfg)
    if not modes:
        modes = ["balanced"]

    per_mode = max(400, simulations // max(len(modes), 1))
    pool = []
    for mode in modes:
        pool.extend(generate_candidates(rng, cfg, feat, mode, per_mode, damp_last))

    # Pro Modus vorab die besten, dann mischen – kein Mittelwert über widersprüchliche Heuristiken
    interleaved = []
    by_mode = {m: [] for m in modes}
    for item in pool:
        by_mode[item[3]].append(item)
    for m in modes:
        by_mode[m].sort(key=lambda x: x[0], reverse=True)

    max_len = max((len(v) for v in by_mode.values()), default=0)
    for i in range(max_len):
        for m in modes:
            if i < len(by_mode[m]):
                interleaved.append(by_mode[m][i])

    return select_diverse(interleaved, n, cfg["overlap_limit"])


def random_tickets(rng, cfg, n):
    main_vals = np.arange(1, cfg["main_max"] + 1)
    bonus_vals = np.arange(cfg["bonus_min"], cfg["bonus_max"] + 1)
    out = []
    for _ in range(n):
        m = tuple(sorted(int(x) for x in rng.choice(main_vals, cfg["main_count"], replace=False)))
        if cfg["bonus_count"] == 2:
            b = tuple(sorted(int(x) for x in rng.choice(bonus_vals, 2, replace=False)))
        else:
            b = (int(rng.integers(cfg["bonus_min"], cfg["bonus_max"] + 1)),)
        out.append((0.0, m, b, "random"))
    return out


def save_predictions(pred, cfg, game_key, seed, modes, sims, as_of):
    path = cfg["history_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    created = datetime.now(timezone.utc).isoformat()
    for rank, item in enumerate(pred, 1):
        s, m, b = item[0], item[1], item[2]
        mode = item[3] if len(item) > 3 else ""
        rows.append(
            {
                "created_at": created,
                "game": game_key,
                "as_of_draw": pd.Timestamp(as_of).date().isoformat() if as_of is not None else "",
                "seed": int(seed),
                "sims": int(sims),
                "modes": ",".join(modes),
                "rank": rank,
                "source_mode": mode,
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


def next_draw_dates(weekdays, n=3, start=None):
    out = []
    d = start or date.today()
    for _ in range(60):
        if d.weekday() in weekdays:
            out.append(d)
            if len(out) >= n:
                break
        d += timedelta(days=1)
    return out


def default_seed_from_draw(weekdays) -> int:
    nxt = next_draw_dates(weekdays, n=1)
    if not nxt:
        return int(date.today().strftime("%Y%m%d"))
    return int(nxt[0].strftime("%Y%m%d"))


def last_draw_text(df, cfg) -> str:
    if df is None or len(df) == 0:
        return "–"
    row = df.iloc[-1]
    mains = " ".join(str(int(row[c])) for c in cfg["main_cols"])
    raw = row[cfg["bonus_cols"]]
    if isinstance(raw, pd.Series):
        if "_sz_valid" in df.columns and not bool(row.get("_sz_valid", True)):
            bonus = "–"
        else:
            bonus = " ".join(str(int(x)) for x in raw.tolist())
    else:
        bonus = str(int(raw))
    return f"{row['Datum'].date()} · {mains} · {cfg['bonus_label']} {bonus}"


def expected_main_hits(cfg) -> float:
    k, n = cfg["main_count"], cfg["main_max"]
    return k * k / n


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎯 Eurojackpot Predictor 2.3")
st.caption(
    "Statistisches Ranking & Backtesting – keine Gewinnwahrscheinlichkeit. "
    "Höherer Score heißt nur: besser passend zur gewählten Heuristik. "
    "Zufallsziehungen sind nicht vorhersagbar."
)

game_key = st.radio(
    "Spielmodus",
    options=["eurojackpot", "6aus49"],
    format_func=lambda k: GAMES[k]["label"],
    index=0,
    horizontal=True,
)
cfg = GAMES[game_key]

draws = next_draw_dates(cfg["draw_weekdays"], n=3)
draw_str = " · ".join(
    f"{cfg['draw_names'].get(d.weekday(), d.strftime('%a'))} {d.strftime('%d.%m.')}" for d in draws
)
st.info(
    f"**{cfg['label']}** · Hauptzahlen: {cfg['main_count']} aus {cfg['main_max']} · "
    f"{cfg['bonus_label']}: {cfg['bonus_range_label']} · Nächste Ziehungen: {draw_str}"
)

df = ensure_data(game_key)
exp_hits = expected_main_hits(cfg)

with st.sidebar:
    st.header("Daten")
    st.write(f"**Spiel:** {cfg['label']}")
    st.write(f"**{len(df):,} Ziehungen**")
    st.write(f"Letzte Ziehung: **{last_draw_text(df, cfg)}**")
    for note in data_quality_report(df, cfg):
        st.caption(note)

    if st.button("🔄 Datenfeed prüfen / aktualisieren"):
        try:
            remote = refresh_from_url(game_key)
            load_local.clear()
            st.success(f"Remote geladen: {len(remote)} Ziehungen.")
            st.rerun()
        except Exception as e:
            st.error(f"Update fehlgeschlagen: {e}")

    n = st.slider("Anzahl Tipps", 1, 30, 10)
    sims = st.slider(
        "Kandidaten pro Modell",
        1000,
        40000,
        8000,
        1000,
        help="Smarte Stichprobe (gewichtet + Paare), kein Blind-Monte-Carlo über den ganzen Raum.",
    )
    auto_seed = default_seed_from_draw(cfg["draw_weekdays"])
    seed = st.number_input(
        "Seed (Standard: nächstes Ziehungsdatum)",
        0,
        99999999,
        auto_seed,
        help="Gleicher Tag + gleiches Spiel → gleiche Tipps. Nächster Ziehungstag → neuer Seed.",
    )
    modes = st.multiselect(
        "Modelle (getrennt, nicht gemittelt)",
        MODES,
        default=["balanced", "frequency", "pairs", "recent", "decay"],
        help="Jeder Modus erzeugt eigene Kandidaten. Die Tipps werden abwechselnd und divers ausgewählt.",
    )
    damp_last = st.checkbox(
        "Letzte Ziehung leicht dämpfen",
        value=False,
        help="Nur Komfort für Nutzer, kein statistischer Vorteil.",
    )

if not modes:
    st.error("Mindestens ein Modell auswählen.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictor", "📊 Statistik", "🧪 Backtesting", "🗂️ History"])

with tab1:
    st.markdown(
        "Kandidaten werden **pro Modell gewichtet** gezogen (Häufigkeit/Gap/Paare), "
        "dann rangiert und auf Überschneidung begrenzt. "
        f"Zufallserwartung Hauptzahlen-Hits/Tipp: **{exp_hits:.2f}**."
    )
    if st.button("🚀 Berechnen", type="primary"):
        pred = predict(df, cfg, n, sims, int(seed), modes, damp_last=damp_last)
        rows = []
        for i, item in enumerate(pred):
            s, m, b, src = item
            rows.append(
                {
                    "Rang": i + 1,
                    "Hauptzahlen": " ".join(map(str, m)),
                    cfg["bonus_label"]: " ".join(map(str, b)),
                    "Modell": src,
                    "Score": round(s, 6),
                }
            )
        out = pd.DataFrame(rows)
        st.dataframe(out, hide_index=True, use_container_width=True)
        save_predictions(pred, cfg, game_key, seed, modes, sims, df["Datum"].max())
        st.download_button(
            "⬇️ CSV",
            out.to_csv(index=False).encode(),
            f"predictions_{game_key}_{date.today().isoformat()}.csv",
            "text/csv",
        )

with tab2:
    feat = features(df, cfg)
    last_m = feat["last_main"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hauptzahlen")
        main_tbl = pd.DataFrame(
            {
                "Zahl": range(1, cfg["main_max"] + 1),
                "Häufigkeit": [feat["mf"][x] for x in range(1, cfg["main_max"] + 1)],
                "Gap": [feat["mg"][x] for x in range(1, cfg["main_max"] + 1)],
                "Letzte": ["●" if x in last_m else "" for x in range(1, cfg["main_max"] + 1)],
            }
        )
        st.bar_chart(main_tbl.set_index("Zahl")["Häufigkeit"], height=220)
        st.dataframe(
            main_tbl.sort_values("Häufigkeit", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.subheader(cfg["bonus_label"])
        b_min, b_max = cfg["bonus_min"], cfg["bonus_max"]
        bonus_tbl = pd.DataFrame(
            {
                cfg["bonus_label"]: range(b_min, b_max + 1),
                "Häufigkeit": [feat["bf"][x] for x in range(b_min, b_max + 1)],
                "Gap": [feat["bg"][x] for x in range(b_min, b_max + 1)],
                "Letzte": [
                    "●" if x in feat["last_bonus"] else "" for x in range(b_min, b_max + 1)
                ],
            }
        )
        st.bar_chart(bonus_tbl.set_index(cfg["bonus_label"])["Häufigkeit"], height=220)
        st.dataframe(
            bonus_tbl.sort_values("Häufigkeit", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Häufigste Zahlenpaare (Hauptzahlen)")
    st.dataframe(
        pd.DataFrame(
            [{"Paar": f"{a}-{b}", "Treffer": v} for (a, b), v in feat["mp"].most_common(25)]
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab3:
    st.write(
        "Walk-forward: jede Testziehung nur mit Daten **davor**. "
        "Zum Vergleich dieselben Tipps als reiner Zufall (gleiche Anzahl)."
    )
    st.caption(
        f"Zufallserwartung Haupt-Hits pro Tipp: {exp_hits:.3f}. "
        "Bonus/Superzahl wird getrennt gezählt. "
        "Wenn das Modell den Zufall nicht schlägt, ist das der Befund – keine schärfere Magie."
    )
    train = st.slider("Trainingsfenster", 20, min(150, max(20, len(df) - 5)), 52)
    tests = st.slider("Testziehungen", 5, min(80, max(5, len(df) - train)), 20)
    bt_sims = st.slider("Kandidaten/Modell im Test", 800, 8000, 2500, 200)
    if st.button("🧪 Test starten"):
        records = []
        start = max(train, len(df) - tests)
        main, bonus = cfg["main_cols"], cfg["bonus_cols"]
        progress = st.progress(0.0, text="Backtest …")
        total = max(len(df) - start, 1)
        for step, i in enumerate(range(start, len(df))):
            hist = df.iloc[:i]
            pred = predict(
                hist, cfg, n, bt_sims, int(seed) + i, modes, damp_last=damp_last
            )
            rng_bt = np.random.default_rng(int(seed) + 100000 + i)
            rnd = random_tickets(rng_bt, cfg, n)
            actual_m = set(int(x) for x in df.iloc[i][main].tolist())
            raw_b = df.iloc[i][bonus]
            if isinstance(raw_b, pd.Series):
                actual_b = set(int(x) for x in raw_b.tolist() if pd.notna(x))
            else:
                actual_b = {int(raw_b)} if pd.notna(raw_b) else set()
            if cfg["bonus_count"] == 1:
                actual_b = {x for x in actual_b if 0 <= x <= 9}

            def summarize(tickets):
                mh = [len(actual_m & set(x[1])) for x in tickets]
                bh = [len(actual_b & set(x[2])) for x in tickets] if actual_b else [0] * len(tickets)
                ge3 = sum(1 for h in mh if h >= 3)
                return (
                    float(np.mean(mh)) if mh else 0.0,
                    int(max(mh) if mh else 0),
                    float(np.mean(bh)) if bh else 0.0,
                    int(ge3),
                )

            p_avg, p_best, p_b, p_ge3 = summarize(pred)
            r_avg, r_best, r_b, r_ge3 = summarize(rnd)
            records.append(
                {
                    "Datum": df.iloc[i]["Datum"],
                    "Modell Ø Haupt": round(p_avg, 3),
                    "Zufall Ø Haupt": round(r_avg, 3),
                    "Modell best. Haupt": p_best,
                    "Zufall best. Haupt": r_best,
                    "Modell Ø Bonus": round(p_b, 3),
                    "Zufall Ø Bonus": round(r_b, 3),
                    "Modell ≥3 Haupt": p_ge3,
                    "Zufall ≥3 Haupt": r_ge3,
                }
            )
            progress.progress((step + 1) / total, text=f"Backtest {step + 1}/{total}")

        bt = pd.DataFrame(records)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Modell Ø Haupt", f"{bt['Modell Ø Haupt'].mean():.3f}", help="Pro Tipp")
        m2.metric("Zufall Ø Haupt", f"{bt['Zufall Ø Haupt'].mean():.3f}")
        delta = bt["Modell Ø Haupt"].mean() - bt["Zufall Ø Haupt"].mean()
        m3.metric("Δ vs. Zufall", f"{delta:+.3f}")
        m4.metric("Erwartung (Theorie)", f"{exp_hits:.3f}")
        st.caption(
            f"Beste Haupt-Hits: Modell {int(bt['Modell best. Haupt'].max())} · "
            f"Zufall {int(bt['Zufall best. Haupt'].max())} · "
            f"Tipps mit ≥3 Haupt: Modell {int(bt['Modell ≥3 Haupt'].sum())} / "
            f"Zufall {int(bt['Zufall ≥3 Haupt'].sum())}"
        )
        st.dataframe(bt, use_container_width=True, hide_index=True)

with tab4:
    hf = cfg["history_file"]
    if hf.exists():
        hist = pd.read_csv(hf)
        st.dataframe(hist.tail(200), use_container_width=True, hide_index=True)
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

**Sampling:** gewichtete Zahlen + häufige Paare + kleiner Uniform-Anteil.  
**Ensemble:** gewählte Modelle laufen getrennt, Tipps werden verschränkt und diversifiziert.  
**Bonus:** nur Häufigkeit / Gap / Recency / Decay.  
**Seed:** Standard = Datum der nächsten Ziehung.

**Wichtig:** Scores sind Rankingwerte. Faire Ziehungen sind nicht vorhersagbar. Overdue/Cold sind Spieler-Heuristiken.
"""
    )
