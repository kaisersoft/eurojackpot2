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
import time as time_module
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Jackpot Predictor",
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
        "close_by_weekday": {1: (18, 35), 4: (18, 35)},
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
        "close_by_weekday": {2: (17, 45), 5: (18, 45)},
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


def predict(df, cfg, n, simulations, seed, modes, damp_last=False, progress=None):
    rng = np.random.default_rng(int(seed))
    feat = features(df, cfg)
    if not modes:
        modes = ["balanced"]

    per_mode = max(400, simulations // max(len(modes), 1))
    pool = []
    n_modes = max(len(modes), 1)
    for i, mode in enumerate(modes):
        if progress is not None:
            progress.progress(
                i / (n_modes + 1),
                text=f"Berechne Modell „{mode}“ ({i + 1}/{n_modes}) …",
            )
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

    if progress is not None:
        progress.progress(n_modes / (n_modes + 1), text="Sortiere und filtere Tipps …")
    selected = select_diverse(interleaved, n, cfg["overlap_limit"])
    if progress is not None:
        progress.progress(1.0, text="Fertig")
    return selected


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


def next_submission_deadline(now: datetime, cfg: dict) -> tuple[datetime, date] | None:
    d = now.date()
    for _ in range(16):
        if d.weekday() in cfg["draw_weekdays"]:
            h, m = cfg.get("close_by_weekday", {}).get(d.weekday(), (18, 0))
            deadline = now.replace(year=d.year, month=d.month, day=d.day, hour=h, minute=m, second=0, microsecond=0)
            if deadline > now:
                return deadline, d
        d += timedelta(days=1)
    return None


def render_deadline_timer(cfg: dict, tz_name: str = "Europe/Berlin", height: int = 118) -> None:
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    nxt = next_submission_deadline(now, cfg)
    if not nxt:
        st.caption("Keine Abgabefrist ermittelt.")
        return
    deadline, draw_date = nxt
    wd = cfg["draw_names"].get(draw_date.weekday(), draw_date.strftime("%A"))
    ts_ms = int(deadline.timestamp() * 1000)
    html = f"""
    <div style="font-family:Inter,system-ui,sans-serif;color:#f8fafc;
                background:#0b1220;border:1px solid #1e293b;border-radius:12px;
                padding:0.85rem 1rem;">
      <div style="font-size:0.72rem;color:#94a3b8;font-weight:500;">
        Abgabefrist · {cfg['label']}
      </div>
      <div style="font-size:0.95rem;font-weight:700;margin-top:0.15rem;">
        {wd} {draw_date.strftime('%d.%m.%Y')} · {deadline.strftime('%H:%M')} Uhr
      </div>
      <div id="als-cd" style="font-size:1.35rem;font-weight:800;color:#00e5ff;
           letter-spacing:-0.03em;margin-top:0.2rem;font-variant-numeric:tabular-nums;">
        …
      </div>
      <div style="font-size:0.68rem;color:#64748b;margin-top:0.25rem;">
        Richtwert Online. Bundesland / Annahmestelle kann abweichen.
      </div>
    </div>
    <script>
      const target = {ts_ms};
      const el = document.getElementById("als-cd");
      function pad(n) {{ return String(n).padStart(2, "0"); }}
      function tick() {{
        const ms = target - Date.now();
        if (ms <= 0) {{
          el.textContent = "Abgabe vorbei";
          el.style.color = "#f87171";
          return;
        }}
        const s = Math.floor(ms / 1000);
        const d = Math.floor(s / 86400);
        const h = Math.floor((s % 86400) / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        el.textContent = (d > 0 ? d + "d " : "") + pad(h) + ":" + pad(m) + ":" + pad(sec);
        el.style.color = s < 3600 ? "#f87171" : (s < 6 * 3600 ? "#fbbf24" : "#00e5ff");
      }}
      tick();
      setInterval(tick, 1000);
    </script>
    """
    components.html(html, height=height)


def _ball_html(num: int, kind: str = "main") -> str:
    if kind == "bonus":
        bg = "radial-gradient(circle at 32% 28%, #fff7ed, #f97316 42%, #9a3412)"
        border = "#fb923c"
    else:
        bg = "radial-gradient(circle at 32% 28%, #ecfeff, #22d3ee 38%, #0e7490)"
        border = "#67e8f9"
    return (
        f'<span class="lotto-ball" style="background:{bg};border-color:{border};">'
        f"{int(num):02d}</span>"
    )


def render_top_tip_balls(main, bonus, cfg, score=None, src="") -> None:
    mains = "".join(_ball_html(x, "main") for x in main)
    extras = "".join(_ball_html(x, "bonus") for x in bonus)
    meta = []
    if src:
        meta.append(str(src))
    if score is not None:
        meta.append(f"Score {score:.4f}")
    meta_s = " · ".join(meta)
    st.markdown(
        f"""
        <style>
        .lotto-hero {{
            background: linear-gradient(180deg, #0b1220, #050508);
            border: 1px solid #1e293b;
            border-radius: 18px;
            padding: 1.15rem 1.25rem 1.3rem;
            margin: 0.35rem 0 1.1rem 0;
        }}
        .lotto-hero-kicker {{
            font-size: 0.78rem; color: #94a3b8; font-weight: 600;
            letter-spacing: 0.06em; text-transform: uppercase;
        }}
        .lotto-hero-row {{
            display: flex; flex-wrap: wrap; align-items: center;
            gap: 0.55rem; margin-top: 0.85rem;
        }}
        .lotto-plus {{
            color: #64748b; font-weight: 800; padding: 0 0.2rem; font-size: 1.2rem;
        }}
        .lotto-ball {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 54px; height: 54px; border-radius: 50%;
            color: #082f49; font-weight: 800; font-size: 1.15rem;
            border: 2px solid #67e8f9;
            box-shadow: 0 8px 18px rgba(0,0,0,0.35), inset 0 -6px 10px rgba(0,0,0,0.18);
            font-variant-numeric: tabular-nums;
        }}
        .lotto-hero-meta {{ margin-top: 0.7rem; color: #94a3b8; font-size: 0.82rem; }}
        </style>
        <div class="lotto-hero">
          <div class="lotto-hero-kicker">Tipp · Rang 1</div>
          <div class="lotto-hero-row">
            {mains}
            <span class="lotto-plus">+</span>
            {extras}
          </div>
          <div class="lotto-hero-meta">{cfg['label']} · {cfg['bonus_label']}{(' · ' + meta_s) if meta_s else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Splash (gleicher Screen wie AstroLotto)
# ---------------------------------------------------------------------------
if not st.session_state.get("splash_done"):
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"] {
            display: none !important;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #050508 !important;
        }
        .alsplash {
            position: fixed; inset: 0; z-index: 999999;
            background:
                repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,229,255,0.06) 40px),
                repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,229,255,0.06) 40px),
                #050508;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            font-family: Inter, system-ui, sans-serif;
        }
        .alsplash-k { width: 220px; height: 220px; margin-bottom: 0.4rem;
            filter: drop-shadow(0 0 18px rgba(0,229,255,0.55)); }
        .alsplash-title {
            font-size: 3.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em;
            margin: 0.15rem 0 0 0; line-height: 1.05;
        }
        .alsplash-sub {
            font-size: 1.05rem; font-weight: 700; color: #00e5ff;
            letter-spacing: 0.42em; margin: 0.35rem 0 1.35rem 0;
        }
        .alsplash-badge {
            display: inline-flex; align-items: center; gap: 0.55rem;
            padding: 0.4rem 0.95rem 0.4rem 0.45rem; border-radius: 999px;
            background: #0a0a0a; border: 1px solid rgba(0,229,255,0.4);
            box-shadow: 0 0 14px rgba(0,229,255,0.18);
            color: #e2e8f0; font-size: 0.95rem; font-weight: 500;
        }
        .alsplash-badge strong { color: #00e5ff; }
        .alsplash-bar {
            width: min(420px, 70vw); height: 8px; margin-top: 2.2rem;
            background: #111827; border-radius: 999px; overflow: hidden;
            box-shadow: 0 0 16px rgba(0,229,255,0.18);
        }
        .alsplash-bar > span {
            display: block; height: 100%; width: 0;
            background: linear-gradient(90deg, #00b8d4, #00e5ff);
            border-radius: 999px;
            animation: alsplash-load 2.5s ease-in-out forwards;
        }
        .alsplash-note { margin-top: 1.35rem; color: #64748b; font-size: 0.85rem; }
        @keyframes alsplash-load { from { width: 0; } to { width: 100%; } }
        </style>
        <div class="alsplash">
          <svg class="alsplash-k" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="48" cy="30" r="5" fill="#00E5FF"/>
            <circle cx="48" cy="70" r="4.5" fill="#00E5FF"/>
            <circle cx="48" cy="100" r="6" fill="#00E5FF"/>
            <circle cx="48" cy="130" r="4.5" fill="#00E5FF"/>
            <circle cx="48" cy="170" r="5" fill="#00E5FF"/>
            <circle cx="78" cy="70" r="4" fill="#00E5FF"/>
            <circle cx="95" cy="55" r="3.5" fill="#00E5FF"/>
            <circle cx="112" cy="40" r="4.5" fill="#00E5FF"/>
            <circle cx="135" cy="28" r="5" fill="#00E5FF"/>
            <circle cx="78" cy="130" r="4" fill="#00E5FF"/>
            <circle cx="95" cy="145" r="3.5" fill="#00E5FF"/>
            <circle cx="112" cy="160" r="4.5" fill="#00E5FF"/>
            <circle cx="135" cy="172" r="5" fill="#00E5FF"/>
            <circle cx="70" cy="100" r="3.5" fill="#00E5FF"/>
            <circle cx="100" cy="100" r="5" fill="#00E5FF"/>
            <g stroke="#00E5FF" stroke-width="1.4" stroke-linecap="round" opacity="0.9">
              <line x1="48" y1="30" x2="48" y2="70"/>
              <line x1="48" y1="70" x2="48" y2="100"/>
              <line x1="48" y1="100" x2="48" y2="130"/>
              <line x1="48" y1="130" x2="48" y2="170"/>
              <line x1="48" y1="100" x2="70" y2="100"/>
              <line x1="70" y1="100" x2="100" y2="100"/>
              <line x1="48" y1="70" x2="78" y2="70"/>
              <line x1="78" y1="70" x2="95" y2="55"/>
              <line x1="95" y1="55" x2="112" y2="40"/>
              <line x1="112" y1="40" x2="135" y2="28"/>
              <line x1="48" y1="100" x2="95" y2="55"/>
              <line x1="100" y1="100" x2="112" y2="40"/>
              <line x1="48" y1="130" x2="78" y2="130"/>
              <line x1="78" y1="130" x2="95" y2="145"/>
              <line x1="95" y1="145" x2="112" y2="160"/>
              <line x1="112" y1="160" x2="135" y2="172"/>
              <line x1="48" y1="100" x2="95" y2="145"/>
              <line x1="100" y1="100" x2="112" y2="160"/>
            </g>
          </svg>
          <div class="alsplash-title">Jackpot</div>
          <div class="alsplash-sub">PREDICTOR</div>
          <div class="alsplash-badge">
            Made by <strong>Kaisersoft.ai</strong>
          </div>
          <div class="alsplash-bar"><span></span></div>
          <div class="alsplash-note">Nur zur Unterhaltung · keine Gewinngarantie</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time_module.sleep(2.7)
    st.session_state.splash_done = True
    st.rerun()


def next_upcoming_game(tz_name: str = "Europe/Berlin") -> str:
    """Lotterie mit der nächsten Abgabefrist."""
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    best_key = "eurojackpot"
    best_dt = None
    for key, gcfg in GAMES.items():
        nxt = next_submission_deadline(now, gcfg)
        if nxt and (best_dt is None or nxt[0] < best_dt):
            best_dt = nxt[0]
            best_key = key
    return best_key


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .cf-made-by-sidebar { margin: 0 0 0.15rem 0; }
    .cf-made-link {
        display: inline-flex; align-items: center; gap: 0.55rem;
        text-decoration: none !important;
        padding: 0.35rem 0.75rem 0.35rem 0.4rem; border-radius: 999px;
        background: #0a0a0a; border: 1px solid rgba(0,229,255,0.35);
        box-shadow: 0 0 12px rgba(0,229,255,0.15);
        width: 100%; box-sizing: border-box;
    }
    .cf-made-logo svg { width: 26px; height: 26px;
        filter: drop-shadow(0 0 6px rgba(0,229,255,0.55)); }
    .cf-made-text { font-size: 0.85rem; font-weight: 500; color: #e2e8f0; }
    .cf-made-text strong { color: #00e5ff; font-weight: 700; }
    .cf-logo-title { font-weight: 800; font-size: 1.15rem; color: #f8fafc; letter-spacing: -0.02em; }
    .cf-logo-sub { font-size: 0.7rem; color: #00e5ff; font-weight: 600; letter-spacing: 0.04em; }
    .jp-hero-title { font-size: 1.35rem; font-weight: 800; color: #f8fafc;
        letter-spacing: -0.03em; margin: 0 0 0.15rem 0; }
    .jp-hero-sub { font-size: 0.88rem; color: #94a3b8; margin-bottom: 0.85rem; }

    html, body, .stApp {
        background: #050508 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"] {
        background: transparent !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
        background: #000000 !important;
        border-right: 1px solid #111111 !important;
    }

    .jp-mathsky {
        position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
        background:
            radial-gradient(ellipse 70% 50% at 80% 0%, rgba(0, 80, 90, 0.18), transparent 55%),
            radial-gradient(ellipse 50% 40% at 10% 100%, rgba(20, 40, 80, 0.16), transparent 50%),
            repeating-linear-gradient(0deg, transparent, transparent 47px, rgba(0,229,255,0.04) 48px),
            repeating-linear-gradient(90deg, transparent, transparent 47px, rgba(0,229,255,0.04) 48px),
            #050508;
    }
    .jp-mathsky .glyph {
        position: absolute; color: rgba(0, 229, 255, 0.22);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-weight: 600; white-space: nowrap;
        animation: jp-drift 18s linear infinite;
    }
    .jp-mathsky .col {
        position: absolute; top: -20%;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size: 13px; line-height: 1.35; letter-spacing: 0.08em;
        color: rgba(103, 232, 249, 0.16);
        writing-mode: vertical-rl; text-orientation: mixed;
        animation: jp-fall linear infinite;
    }
    .jp-scan {
        position: absolute; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0,229,255,0.35), transparent);
        animation: jp-scan 9s ease-in-out infinite;
        opacity: 0.5;
    }
    .jp-curve {
        position: absolute; left: 8%; right: 8%; bottom: 8%; height: 22%;
        opacity: 0.22;
    }
    @keyframes jp-drift {
        0% { transform: translate3d(0, 0, 0); opacity: 0.12; }
        50% { opacity: 0.32; }
        100% { transform: translate3d(12px, -28px, 0); opacity: 0.1; }
    }
    @keyframes jp-fall {
        0% { transform: translateY(-10%); opacity: 0; }
        12% { opacity: 0.35; }
        100% { transform: translateY(120vh); opacity: 0; }
    }
    @keyframes jp-scan {
        0% { top: 8%; }
        100% { top: 88%; }
    }
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"] {
        position: relative; z-index: 1;
    }
    </style>
    <div class="jp-hero-title">Jackpot Predictor</div>
    """,
    unsafe_allow_html=True,
)
def _math_background_html() -> str:
    rng = __import__("random").Random(7)
    glyphs = [
        "Σ", "μ", "σ", "P(X)", "C(n,k)", "k²/N", "χ²", "E[X]",
        "∏", "Δ", "H(x)", "p̂", "n=50", "n=49", "log L", "∩",
    ]
    bits = ['<div class="jp-mathsky" aria-hidden="true">']
    bits.append(
        '<svg class="jp-curve" viewBox="0 0 200 40" preserveAspectRatio="none">'
        '<path d="M0 34 C 30 34, 50 8, 80 8 S 130 34, 200 34" '
        'fill="none" stroke="#00e5ff" stroke-width="1.2"/>'
        '<path d="M0 34 C 30 34, 50 8, 80 8 S 130 34, 200 34 L200 40 L0 40 Z" '
        'fill="rgba(0,229,255,0.06)"/>'
        "</svg>"
    )
    bits.append('<div class="jp-scan"></div>')
    for i in range(14):
        left = rng.uniform(2, 96)
        delay = rng.uniform(0, 12)
        dur = rng.uniform(11, 22)
        text = " ".join(str(rng.randint(0, 50)) for _ in range(18))
        bits.append(
            f'<div class="col" style="left:{left:.1f}%;animation-duration:{dur:.1f}s;'
            f'animation-delay:{delay:.1f}s">{text}</div>'
        )
    for i, g in enumerate(glyphs):
        bits.append(
            f'<span class="glyph" style="left:{rng.uniform(4,92):.1f}%;'
            f'top:{rng.uniform(8,78):.1f}%;font-size:{rng.choice([14,16,18,22])}px;'
            f'animation-delay:{i * 0.4:.1f}s">{g}</span>'
        )
    bits.append("</div>")
    return "".join(bits)


st.markdown(_math_background_html(), unsafe_allow_html=True)
st.caption(
    "Statistisches Ranking & Backtesting – keine Gewinnwahrscheinlichkeit. "
    "Höherer Score heißt nur: besser passend zur gewählten Heuristik. "
    "Zufallsziehungen sind nicht vorhersagbar."
)

_game_options = ["eurojackpot", "6aus49"]
if "game_key" not in st.session_state:
    st.session_state.game_key = next_upcoming_game()
game_key = st.radio(
    "Spielmodus",
    options=_game_options,
    format_func=lambda k: GAMES[k]["label"],
    index=_game_options.index(st.session_state.game_key),
    key="game_key",
    horizontal=True,
    help="Vorausgewählt ist die Lotterie mit der nächsten Abgabefrist.",
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
    st.markdown(
        """
        <div class="cf-made-by-sidebar">
          <a href="https://github.com/kaisersoft" target="_blank" rel="noopener noreferrer" class="cf-made-link">
            <span class="cf-made-logo" aria-hidden="true">
              <svg viewBox="0 0 200 200" width="26" height="26" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="48" cy="30" r="5" fill="#00E5FF"/>
                <circle cx="48" cy="70" r="4.5" fill="#00E5FF"/>
                <circle cx="48" cy="100" r="6" fill="#00E5FF"/>
                <circle cx="48" cy="130" r="4.5" fill="#00E5FF"/>
                <circle cx="48" cy="170" r="5" fill="#00E5FF"/>
                <circle cx="78" cy="70" r="4" fill="#00E5FF"/>
                <circle cx="95" cy="55" r="3.5" fill="#00E5FF"/>
                <circle cx="112" cy="40" r="4.5" fill="#00E5FF"/>
                <circle cx="135" cy="28" r="5" fill="#00E5FF"/>
                <circle cx="78" cy="130" r="4" fill="#00E5FF"/>
                <circle cx="95" cy="145" r="3.5" fill="#00E5FF"/>
                <circle cx="112" cy="160" r="4.5" fill="#00E5FF"/>
                <circle cx="135" cy="172" r="5" fill="#00E5FF"/>
                <circle cx="70" cy="100" r="3.5" fill="#00E5FF"/>
                <circle cx="100" cy="100" r="5" fill="#00E5FF"/>
                <g stroke="#00E5FF" stroke-width="1.4" stroke-linecap="round" opacity="0.9">
                  <line x1="48" y1="30" x2="48" y2="70"/>
                  <line x1="48" y1="70" x2="48" y2="100"/>
                  <line x1="48" y1="100" x2="48" y2="130"/>
                  <line x1="48" y1="130" x2="48" y2="170"/>
                  <line x1="48" y1="100" x2="70" y2="100"/>
                  <line x1="70" y1="100" x2="100" y2="100"/>
                  <line x1="48" y1="70" x2="78" y2="70"/>
                  <line x1="78" y1="70" x2="95" y2="55"/>
                  <line x1="95" y1="55" x2="112" y2="40"/>
                  <line x1="112" y1="40" x2="135" y2="28"/>
                  <line x1="48" y1="100" x2="95" y2="55"/>
                  <line x1="100" y1="100" x2="112" y2="40"/>
                  <line x1="48" y1="130" x2="78" y2="130"/>
                  <line x1="78" y1="130" x2="95" y2="145"/>
                  <line x1="95" y1="145" x2="112" y2="160"/>
                  <line x1="112" y1="160" x2="135" y2="172"/>
                  <line x1="48" y1="100" x2="95" y2="145"/>
                  <line x1="100" y1="100" x2="112" y2="160"/>
                </g>
              </svg>
            </span>
            <span class="cf-made-text">Made by <strong>Kaisersoft.ai</strong></span>
          </a>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;margin:0.75rem 0 0.65rem 0;">
            <span style="font-size:1.45rem;">🎯</span>
            <div>
                <div class="cf-logo-title">Jackpot Predictor</div>
                <div class="cf-logo-sub">STATISTIK</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    render_deadline_timer(cfg, height=122)
    if st.button("🚀 Berechnen", type="primary"):
        progress = st.progress(0.0, text="Starte Berechnung …")
        pred = predict(
            df,
            cfg,
            n,
            sims,
            int(seed),
            modes,
            damp_last=damp_last,
            progress=progress,
        )
        progress.empty()
        st.session_state.last_pred = {
            "game": game_key,
            "pred": pred,
            "seed": int(seed),
        }
        save_predictions(pred, cfg, game_key, seed, modes, sims, df["Datum"].max())

    last = st.session_state.get("last_pred")
    if last and last.get("game") == game_key and last.get("pred"):
        pred = last["pred"]
        top = pred[0]
        render_top_tip_balls(top[1], top[2], cfg, score=top[0], src=top[3])
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
