"""Statistics behind the dashboard. Pure pandas / numpy (no Streamlit, no network), so it can be tested offline:

    python tools/decision_stats.py      # runs the self-checks at the bottom

Two sources, two kinds of uncertainty:

- OFFICIAL metric (public leaderboard API): expected purchases per 100 requests, computed by the organisers'
  simulator from each shopper's purchase probability (no luck). It comes in 15-minute pieces. The store drifts
  about ±10 % from one hour to the next, so a version is described RELATIVE TO THE ORGANISERS' REFERENCE in the same
  piece, and its interval comes from how much that ratio varies between pieces (a cluster interval: it includes the
  drift that a binomial interval ignores). Two versions are compared only on the pieces where both ran (paired by
  time): that is the clean A/B.
- REALISED outcomes (RawTree feedback): the clicks and purchases that actually happened (with luck). Wilson
  intervals for proportions. When several groups are compared at once, p-values are Holm-adjusted; otherwise one
  group in twenty looks "significant" by pure chance.
"""
from __future__ import annotations

import html
import math
import re
from itertools import combinations

import numpy as np
import pandas as pd

Z = 1.96
NO_ANSWER = "(sin respuesta)"
MIN_PIECE_REQUESTS = 50          # a 15-minute piece with fewer requests for a version is too noisy to use

# metric key -> (name, unit, weight column). Cold ratios are weighted by cold requests, no-cold by the rest.
METRICS = {
    "conv": ("Conversión", "compras esperadas / 100 peticiones", "n"),
    "cold": ("Cold start", "compras esperadas / 100 peticiones cold", "nc"),
    "warm": ("No-cold (recurrentes)", "compras esperadas / 100 peticiones no-cold", "nw"),
    "eur": ("€ por petición", "€ esperados / petición", "n"),
}

# two-sided 97.5 % quantiles of Student's t (intervals from few pieces must be wider than ±1.96 SE)
_T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18,
         15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00, 120: 1.98}


# ---------------------------------------------------------------- basic statistics

def t975(df: float) -> float:
    if df < 1:
        return math.nan
    if df >= 120:
        return 1.98 if df < 1000 else Z
    keys = sorted(_T975)
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T975[lo]
    return _T975[lo] + (_T975[hi] - _T975[lo]) * (df - lo) / (hi - lo)


def weighted_mean_ci(x, w) -> tuple[float, float, float]:
    """Weighted mean of per-piece values and its 95 % interval, treating each piece as one observation
    (cluster-robust standard error, Student t with pieces - 1 degrees of freedom)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    k = len(x)
    if k == 0:
        return math.nan, math.nan, math.nan
    m = float((w * x).sum() / w.sum())
    if k < 2:
        return m, math.nan, math.nan
    se = math.sqrt(k / (k - 1) * float(((w * (x - m)) ** 2).sum())) / float(w.sum())
    h = t975(k - 1) * se
    return m, m - h, m + h


def wilson(k: float, n: float, z: float = Z) -> tuple[float, float]:
    """95 % Wilson interval for a proportion k/n, returned as fractions (0..1)."""
    if not n or n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def two_prop_pvalue(k1: float, n1: float, k2: float, n2: float) -> float:
    """Two-sided p-value of H0: p1 == p2 (pooled two-proportion z-test)."""
    if not n1 or not n2:
        return math.nan
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the chance of ANY false alarm among the comparisons)."""
    p = [1.0 if (v is None or math.isnan(v)) else float(v) for v in pvalues]
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def add_rate(df: pd.DataFrame, k: str, n: str, name: str) -> pd.DataFrame:
    """Add `name`, `name_lo`, `name_hi` = 100·k/n with its Wilson interval."""
    out = df.copy()
    ks = pd.to_numeric(out[k], errors="coerce").fillna(0)
    ns = pd.to_numeric(out[n], errors="coerce").fillna(0)
    out[name] = (100 * ks / ns.where(ns > 0)).astype(float)
    ci = [wilson(a, b) for a, b in zip(ks, ns)]
    out[f"{name}_lo"] = [100 * lo for lo, _ in ci]
    out[f"{name}_hi"] = [100 * hi for _, hi in ci]
    return out


def fmt_p(p: float) -> str:
    if p is None or math.isnan(p):
        return "p = —"
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def num(x: float, d: int = 2) -> str:
    """Spanish number format (decimal comma)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def signed(x: float, d: int = 2) -> str:
    return ("+" if x >= 0 else "−") + num(abs(x), d)


# ---------------------------------------------------------------- official metric (leaderboard API)

def parse_piece(t: int, data: dict, team: str) -> list[dict]:
    """One 15-minute answer of the leaderboard API -> rows (one per version of `team` and one per other team),
    each with the organisers' reference of the same piece."""
    ref = data.get("reference") or {}
    teams = data.get("teams") or []
    if not ref.get("requests") or not teams:
        return []                                    # no traffic, or hidden (final round)
    us = next((x for x in teams if x.get("team_id") == team), None)
    versions = (us or {}).get("versions") or []
    n = sum(v.get("requests", 0) for v in versions)
    nc = sum(v.get("cold_requests", 0) for v in versions)
    share = nc / n if n else 0.4                     # every team gets the same requests: our cold share is theirs
    ref_warm = (ref["conversion"] - share * ref["cold_conversion"]) / (1 - share)
    base = {"t": t, "ref_conv": ref["conversion"], "ref_cold": ref["cold_conversion"], "ref_warm": ref_warm,
            "ref_eur": ref["revenue_per_request"]}
    rows = []
    for v in versions:
        n, nc = v.get("requests", 0), v.get("cold_requests", 0)
        if not n:
            continue
        warm = (v["conversion"] * n - v["cold_conversion"] * nc) / (n - nc) if n > nc else math.nan
        rows.append({**base, "who": v["model_version"], "kind": "versión", "n": n, "nc": nc, "nw": n - nc,
                     "conv": v["conversion"], "cold": v["cold_conversion"], "warm": warm,
                     "eur": v["revenue_per_request"]})
    for x in teams:
        if x.get("team_id") == team or not x.get("requests"):
            continue
        n = x["requests"]
        rows.append({**base, "who": x["team_id"], "kind": "equipo", "n": n, "nc": share * n, "nw": (1 - share) * n,
                     "conv": x["conversion"], "cold": x["cold_conversion"],
                     "warm": (x["conversion"] - share * x["cold_conversion"]) / (1 - share),
                     "eur": x["revenue_per_request"]})
    return rows


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.n >= MIN_PIECE_REQUESTS]


def leader(df: pd.DataFrame, metric: str = "conv") -> str | None:
    """The other team with the best request-weighted official metric over the window."""
    t = _usable(df[df.kind == "equipo"])
    if t.empty:
        return None
    w = METRICS[metric][2]
    s = t.assign(_x=t[metric] * t[w]).groupby("who").agg(x=("_x", "sum"), w=(w, "sum"), pieces=("t", "nunique"))
    s = s[s.pieces >= max(2, s.pieces.max() // 2)]   # present most of the window, not a short burst
    return None if s.empty else str((s.x / s.w).idxmax())


def version_summary(df: pd.DataFrame, metric: str = "conv", min_pieces: int = 3,
                    extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per version (and per team in `extra`): official value, ratio to the reference with its 95 % interval."""
    w = METRICS[metric][2]
    d = _usable(df[(df.kind == "versión") | df.who.isin(extra)])
    rows = []
    for who, g in d.groupby("who"):
        ratio = g[metric] / g[f"ref_{metric}"]
        m, lo, hi = weighted_mean_ci(ratio, g[w])
        rows.append({"who": who, "kind": g.kind.iloc[0], "pieces": g.t.nunique(), "requests": int(g.n.sum()),
                     "desde": g.t.min(), "hasta": g.t.max() + 900,
                     "valor": float((g[metric] * g[w]).sum() / g[w].sum()) if g[w].sum() else math.nan,
                     "ratio": m, "ratio_lo": lo, "ratio_hi": hi})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["suficiente"] = out.pieces >= min_pieces
    return out.sort_values("ratio", ascending=False).reset_index(drop=True)


def ratio_on_pieces(df: pd.DataFrame, who: str, pieces, metric: str = "conv") -> float:
    """Request-weighted ratio to the reference of `who`, restricted to the given piece starts."""
    w = METRICS[metric][2]
    g = _usable(df[(df.who == who) & df.t.isin(list(pieces))])
    if g.empty or not g[w].sum():
        return math.nan
    return float(((g[metric] / g[f"ref_{metric}"]) * g[w]).sum() / g[w].sum())


def paired(df: pd.DataFrame, a: str, b: str, metric: str = "conv") -> dict:
    """A minus B on the pieces where both ran (same shoppers, same time): estimate and 95 % interval."""
    w = METRICS[metric][2]
    d = _usable(df[df.kind == "versión"])
    x = d[d.who == a].set_index("t")
    y = d[d.who == b].set_index("t")
    both = x.index.intersection(y.index)
    if len(both) == 0:
        return {"a": a, "b": b, "pieces": 0}
    xa, yb = x.loc[both], y.loc[both]
    wt = (xa[w] * yb[w]) / (xa[w] + yb[w])
    diff, lo, hi = weighted_mean_ci(xa[metric] - yb[metric], wt)
    base = float((yb[metric] * yb[w]).sum() / yb[w].sum())
    return {"a": a, "b": b, "pieces": len(both), "diff": diff, "lo": lo, "hi": hi, "base": base,
            "rel": 100 * diff / base if base else math.nan,
            "verdict": ("mejor" if lo > 0 else "peor" if hi < 0 else "sin diferencia clara") if not math.isnan(lo)
            else "pocos datos"}


def ab_pairs(df: pd.DataFrame, metric: str = "conv", min_shared: int = 4) -> list[dict]:
    """Every pair of our versions that ran at the same time in at least `min_shared` pieces."""
    d = _usable(df[df.kind == "versión"])
    pieces = d.groupby("who").t.apply(set)
    out = []
    for a, b in combinations(sorted(pieces.index), 2):
        if len(pieces[a] & pieces[b]) >= min_shared:
            r = paired(df, a, b, metric)
            if r.get("diff", 0) < 0:                 # present every pair as "winner minus the other"
                r = paired(df, b, a, metric)
            out.append(r)
    return sorted(out, key=lambda r: -r["pieces"])


def versions_message(summary: pd.DataFrame, df: pd.DataFrame, metric: str, lead: str | None) -> tuple[str, str]:
    name = METRICS[metric][0].lower()
    ours = summary[(summary.kind == "versión") & summary.suficiente] if not summary.empty else summary
    if ours.empty:
        return "info", ("Ninguna versión tiene aún 3 tramos de 15 minutos con tráfico en esta ventana: "
                        "con menos, el intervalo no dice nada.")
    best = ours.iloc[0]
    pieces = set(_usable(df[df.who == best.who]).t)
    tied = [r.who for r in ours.iloc[1:].itertuples() if r.ratio_hi >= best.ratio_lo]
    head = (f"**{best.who}** tiene el valor más alto en {name}: **{num(best.ratio)}× la referencia** "
            f"(IC 95 %: {num(best.ratio_lo)}–{num(best.ratio_hi)}, {best.pieces} tramos de 15 min)")
    level = "good" if best.ratio_lo > 1 else "warning" if best.ratio_hi < 1 else "info"
    txt = [head + (", por encima del intervalo de todas las demás." if not tied else ".")]
    if lead:
        r = ratio_on_pieces(df, lead, pieces, metric)
        if not math.isnan(r):
            txt.append(f"En esos mismos tramos, el mejor otro equipo ({lead}) estaba en {num(r)}×.")
    if tied:
        settled, open_ = [], []
        for other in tied:
            p = paired(df, best.who, other, metric)
            if p.get("pieces", 0) >= 4 and p["verdict"] in ("mejor", "peor"):
                settled.append(f"supera a **{other}** ({signed(p['diff'])}, IC {signed(p['lo'])} a {signed(p['hi'])})"
                               if p["verdict"] == "mejor" else f"queda por debajo de **{other}** ({signed(p['diff'])})")
            else:
                open_.append(other)
        if settled:
            txt.append(f"En A/B directo (a la vez, mismos compradores) {best.who} " + "; ".join(settled) + ".")
        if open_:
            level = "info"
            txt.append(f"Con {', '.join(open_[:3])} sus intervalos se solapan y no hay A/B que las separe: "
                       "con estos datos no se puede decir cuál es mejor.")
    return level, " ".join(txt)


# ---------------------------------------------------------------- realised outcomes (RawTree feedback)

def version_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per version, realised clicks / purchases / revenue on the requests our model served (served_by = 'team')."""
    cols = ["model_version", "requests", "servidas", "clics", "compras", "revenue", "revenue_sd", "desde", "hasta"]
    df = raw.copy() if raw is not None and not raw.empty else pd.DataFrame(columns=cols)
    if df.empty:
        return df
    for c in ["requests", "servidas", "clics", "compras", "revenue", "revenue_sd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["requests", "servidas", "clics", "compras", "revenue"]] = (
        df[["requests", "servidas", "clics", "compras", "revenue"]].fillna(0))
    df = add_rate(df, "compras", "servidas", "compras_por_100")
    df = add_rate(df, "clics", "servidas", "clics_por_100")
    df["a_tiempo_pct"] = 100 * df.servidas / df.requests.where(df.requests > 0)
    n = df.servidas.where(df.servidas > 0)
    df["eur_por_peticion"] = df.revenue / n
    return df.sort_values("servidas", ascending=False).reset_index(drop=True)


SEGMENT_DIMENSIONS = {
    "Cold start": "cold_label",
    "Tipo de usuario": "usuario",
    "Dispositivo": "device",
    "Región": "region",
    "Página": "page",
}


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1"])


def prepare_segments(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for c in ["servidas", "clics", "compras"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["usuario"] = _truthy(df["is_new"]).map({True: "nuevo", False: "recurrente"})
    df["cold_label"] = _truthy(df["cold"]).map({True: "cold", False: "no-cold"})
    for c in ["device", "region", "page"]:
        df[c] = df[c].fillna("").replace("", "(vacío)")
    return df


def segment_table(prepared: pd.DataFrame, column: str) -> pd.DataFrame:
    g = prepared.groupby(column, as_index=False)[["servidas", "clics", "compras"]].sum()
    g = add_rate(g, "compras", "servidas", "compras_por_100")
    g = add_rate(g, "clics", "servidas", "clics_por_100")
    return g.sort_values("compras_por_100", ascending=False).reset_index(drop=True)


def segment_conclusion(table: pd.DataFrame, column: str, dim_name: str, min_n: int = 200) -> tuple[str, str]:
    """Which groups differ from the rest by more than chance? Each group vs the rest, Holm-adjusted."""
    tot_k, tot_n = table.compras.sum(), table.servidas.sum()
    if tot_n == 0:
        return "info", "Sin datos en esta ventana."
    overall = 100 * tot_k / tot_n
    t = table[table.servidas >= min_n]
    if len(t) < 2:
        return "info", f"Hacen falta al menos dos grupos con ≥ {min_n} peticiones para comparar."
    p = [two_prop_pvalue(r.compras, r.servidas, tot_k - r.compras, tot_n - r.servidas) for r in t.itertuples()]
    t = t.assign(p_adj=holm(p))
    sig = t[t.p_adj < 0.05]
    head = f"Media: **{num(overall)} compras reales / 100**."
    if sig.empty:
        return "good", (f"{head} Ningún grupo por {dim_name.lower()} se separa de la media más de lo que explica el "
                        f"azar ({len(t)} grupos comparados, p corregida por Holm).")
    parts = [f"**{getattr(r, column)}** {num(r.compras_por_100)} ({fmt_p(r.p_adj)})" for r in sig.itertuples()]
    return "warning", (f"{head} Se separan de la media más de lo que explica el azar: " + "; ".join(parts)
                       + f" (p corregida por Holm, {len(t)} grupos).")


def position_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, float, dict]:
    """raw rows: tipo in {'clic','compra','total'}, pos (1..10, 0 = not in carousel), n."""
    df = raw.copy()
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").fillna(0).astype(int)
    total = float(df.loc[df.tipo == "total", "n"].sum())
    out = pd.DataFrame({"posición": range(1, 11)})
    outside = {}
    for kind, col in [("clic", "clics"), ("compra", "compras")]:
        sub = df[df.tipo == kind].groupby("pos")["n"].sum()
        out[col] = out["posición"].map(sub).fillna(0)
        outside[col] = float(sub.get(0, 0))
        out = add_rate(out.assign(_n=total), col, "_n", f"{col}_por_100").drop(columns="_n")
    return out, total, outside


def position_conclusion(pt: pd.DataFrame) -> tuple[str, str]:
    clicks, buys = pt.clics.sum(), pt.compras.sum()
    if clicks == 0:
        return "info", "Aún no hay clics suficientes."
    first = 100 * pt.clics.iloc[0] / clicks
    top3 = 100 * pt.clics.head(3).sum() / clicks
    buy_first = f" y el **{num(100 * pt.compras.iloc[0] / buys, 0)} %** de las compras" if buys else ""
    return "info", (f"La posición 1 se lleva el **{num(first, 0)} %** de los clics{buy_first}; las tres primeras, "
                    f"el {num(top3, 0)} % de los clics. Mezcla dos efectos: la posición en sí y que el modelo pone "
                    "primero lo que cree mejor. Separarlos exigiría barajar el orden al azar.")


def trend_table(raw: pd.DataFrame, cat: pd.DataFrame, level: str = "category", min_count: int = 20) -> pd.DataFrame:
    """Share of the store's purchases per group in the recent window vs the hours before, with Holm-adjusted p."""
    df = raw.copy()
    for c in ["recientes", "base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[level] = df.product_id.map(cat[level]) if level in cat else None
    df[level] = df[level].fillna("(producto nuevo)")
    g = df.groupby(level, as_index=False)[["recientes", "base"]].sum()
    n_rec, n_base = g.recientes.sum(), g.base.sum()
    if n_rec == 0 or n_base == 0:
        return g.assign(cuota_reciente=math.nan, cuota_base=math.nan, cambio_pp=math.nan, p=math.nan, p_adj=math.nan)
    g["cuota_reciente"] = 100 * g.recientes / n_rec
    g["cuota_base"] = 100 * g.base / n_base
    g["cambio_pp"] = g.cuota_reciente - g.cuota_base
    g = g[(g.recientes + g.base) >= min_count].copy()
    g["p"] = [two_prop_pvalue(a, n_rec, b, n_base) for a, b in zip(g.recientes, g.base)]
    g["p_adj"] = holm(g.p.tolist())
    g["dirección"] = g.cambio_pp.map(lambda x: "sube" if x >= 0 else "baja")
    return g.reindex(g.cambio_pp.abs().sort_values(ascending=False).index).reset_index(drop=True)


def trend_conclusion(tt: pd.DataFrame, recent_min: int, level_name: str) -> tuple[str, str]:
    if tt.empty or tt.cambio_pp.isna().all():
        return "info", "Aún no hay compras suficientes en la tienda para medir tendencias."
    sig = tt[tt.p_adj < 0.05]
    if sig.empty:
        return "good", (f"Ninguna {level_name} cambia su cuota de compras en los últimos {recent_min} min más de lo "
                        f"que explica el azar ({len(tt)} comparadas, p corregida por Holm).")
    top = sig.iloc[0]
    return "warning", (f"**{len(sig)}** {level_name}(s) cambian más de lo que explica el azar (p corregida por Holm); "
                       f"la mayor, **{top.iloc[0]}**: {signed(top.cambio_pp, 1)} puntos de cuota. Que la tienda "
                       "cambie no quiere decir que el recomendador deba seguirla: eso solo lo dice un A/B.")


def offline_vs_online(vt: pd.DataFrame, offline: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """offline: columns model_version + one numeric metric (first numeric column used)."""
    off = offline.copy()
    off["model_version"] = off["model_version"].astype(str).str.strip()
    metric = [c for c in off.columns if c != "model_version"][0]
    off = off.rename(columns={metric: "offline"})
    off["offline"] = pd.to_numeric(off["offline"], errors="coerce")
    m = vt[["model_version", "servidas", "compras_por_100"]].merge(off, on="model_version")
    m = m.dropna(subset=["offline", "compras_por_100"])
    rho = m.offline.rank().corr(m.compras_por_100.rank()) if len(m) >= 3 else math.nan
    return m, rho


def offline_conclusion(m: pd.DataFrame, rho: float) -> tuple[str, str]:
    if len(m) < 3 or math.isnan(rho):
        return "info", "Hacen falta al menos 3 versiones con métrica offline y datos en directo para medir si coinciden."
    agree = "ordena las versiones casi igual que" if rho >= 0.7 else "no ordena las versiones como"
    return ("good" if rho >= 0.7 else "warning"), (f"La métrica offline {agree} el directo "
                                                   f"(Spearman ρ = {num(rho)}, {len(m)} versiones).")


def md_to_html(text: str) -> str:
    """Tiny markdown → HTML for the message boxes (bold and inline code only); the text itself is escaped."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text, quote=False))
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


# ---------------------------------------------------------------- self-checks

if __name__ == "__main__":
    lo, hi = wilson(5, 100)
    assert 0.02 < lo < 0.05 < hi < 0.12, (lo, hi)
    assert math.isnan(wilson(0, 0)[0])
    assert two_prop_pvalue(50, 1000, 50, 1000) == 1.0
    assert two_prop_pvalue(80, 1000, 40, 1000) < 0.001
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]), holm([0.01, 0.04, 0.03])
    assert abs(t975(9) - 2.26) < 1e-9 and 2.09 < t975(17) < 2.13
    m, lo, hi = weighted_mean_ci([1.0, 1.2, 1.1, 1.3], [1, 1, 1, 1])
    assert abs(m - 1.15) < 1e-9 and lo < 1.15 < hi

    # two versions in an A/B (pieces 0-3), v_old alone before; v_b converts 10 % less than v_a
    def api(conv_a, conv_b, ref, both=True, old=False):
        vs = [{"model_version": "v_a", "requests": 900, "cold_requests": 360, "conversion": conv_a,
               "cold_conversion": conv_a * 1.4, "revenue_per_request": conv_a / 2}]
        if both:
            vs.append({"model_version": "v_b", "requests": 900, "cold_requests": 360, "conversion": conv_b,
                       "cold_conversion": conv_b * 1.4, "revenue_per_request": conv_b / 2})
        if old:
            vs = [{"model_version": "v_old", "requests": 1800, "cold_requests": 720, "conversion": conv_a,
                   "cold_conversion": conv_a, "revenue_per_request": 1.0}]
        return {"reference": {"requests": 1800, "conversion": ref, "cold_conversion": ref, "revenue_per_request": 1},
                "teams": [{"team_id": "team_14", "requests": 1800, "versions": vs},
                          {"team_id": "team_06", "requests": 1800, "conversion": ref * 1.15,
                           "cold_conversion": ref * 1.5, "revenue_per_request": 1.2}]}
    rows = []
    for i, (a, b, r) in enumerate([(2.2, 2.0, 1.9), (2.4, 2.15, 2.0), (2.1, 1.9, 1.8), (2.3, 2.05, 1.9)]):
        rows += parse_piece(900 * i, api(a, b, r), "team_14")
    for i in range(4, 7):
        rows += parse_piece(900 * i, api(1.9, 0, 1.9, both=False, old=True), "team_14")
    df = pd.DataFrame(rows)
    s = version_summary(df, "conv", extra=("team_06",))
    print(s[["who", "pieces", "valor", "ratio", "ratio_lo", "ratio_hi", "suficiente"]].round(3))
    assert s.iloc[0].who == "v_a" and s.iloc[0].ratio_lo > 1
    pair = ab_pairs(df, "conv")
    print(pair)
    assert pair[0]["a"] == "v_a" and pair[0]["verdict"] == "mejor" and pair[0]["pieces"] == 4
    assert leader(df) == "team_06"
    print(versions_message(s, df, "conv", leader(df)))

    seg = prepare_segments(pd.DataFrame([
        dict(device="mobile", region="ES", page="home", is_new=True, cold=True, servidas=900, clics=100, compras=20),
        dict(device="desktop", region="ES", page="product", is_new=False, cold=False, servidas=3000, clics=600, compras=150),
        dict(device="mobile", region="FR", page="product", is_new="false", cold="false", servidas=1200, clics=250, compras=60),
    ]))
    st_ = segment_table(seg, "page")
    print(segment_conclusion(st_, "page", "Página"))

    pos = pd.DataFrame([dict(tipo="total", pos=0, n=4000)]
                       + [dict(tipo="clic", pos=i, n=int(300 / i)) for i in range(1, 11)]
                       + [dict(tipo="compra", pos=i, n=int(40 / i)) for i in range(1, 11)])
    pt, total, outside = position_table(pos)
    print(position_conclusion(pt))

    cat = pd.DataFrame({"category": ["A", "A", "B", "C"]}, index=["p1", "p2", "p3", "p4"])
    ev = pd.DataFrame(dict(product_id=["p1", "p2", "p3", "p4", "p9"], recientes=[200, 50, 30, 40, 25],
                           base=[600, 300, 400, 500, 10]))
    tt = trend_table(ev, cat)
    print(tt[["category", "cambio_pp", "p", "p_adj"]]); print(trend_conclusion(tt, 30, "categoría"))
    print("OK")
