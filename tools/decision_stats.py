"""Statistics behind the «Decisiones» tab of the dashboard.

Pure pandas / math (no Streamlit, no RawTree), so it can be tested offline:

    python tools/decision_stats.py      # runs the self-checks at the bottom

Conventions
- Rates are "per 100 served requests" (same unit as the challenge: purchases per 100 requests).
- Confidence intervals are 95 % Wilson intervals for proportions (well behaved with small p and n),
  and normal intervals (mean ± 1.96·sd/√n) for revenue per request.
- Differences between two rates use a two-proportion z-test (p-value).
"""
from __future__ import annotations

import math
import re

import pandas as pd

Z = 1.96
REFERENCE = "fallback (referencia)"
NO_ANSWER = "(sin respuesta)"


# ---------------------------------------------------------------- basic statistics

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


# ---------------------------------------------------------------- ① versions

def version_table(raw: pd.DataFrame, fallback: pd.Series | dict | None = None) -> pd.DataFrame:
    """Per-version metrics computed ONLY on requests our model actually served (served_by = 'team').

    raw columns: model_version, requests, servidas, clics, compras, revenue, revenue_sd, desde, hasta
    fallback: totals of requests that got the fallback (best sellers) -> a reference row.
    """
    cols = ["model_version", "requests", "servidas", "clics", "compras", "revenue", "revenue_sd", "desde", "hasta"]
    df = raw.copy() if raw is not None and not raw.empty else pd.DataFrame(columns=cols)
    df["tipo"] = "versión"
    if fallback is not None:
        fb = dict(fallback)
        if float(fb.get("servidas") or 0) > 0:
            row = {"model_version": REFERENCE, "requests": fb["servidas"], "servidas": fb["servidas"],
                   "clics": fb.get("clics", 0), "compras": fb.get("compras", 0),
                   "revenue": fb.get("revenue", 0), "revenue_sd": fb.get("revenue_sd"),
                   "desde": None, "hasta": None, "tipo": "referencia"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    if df.empty:
        return df
    for c in ["requests", "servidas", "clics", "compras", "revenue", "revenue_sd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["requests", "servidas", "clics", "compras", "revenue"]] = (
        df[["requests", "servidas", "clics", "compras", "revenue"]].fillna(0))
    df = add_rate(df, "compras", "servidas", "compras_por_100")
    df = add_rate(df, "clics", "servidas", "clics_por_100")
    df["a_tiempo_pct"] = (100 * df.servidas / df.requests.where(df.requests > 0)).where(df.tipo == "versión")
    n = df.servidas.where(df.servidas > 0)
    df["eur_por_peticion"] = df.revenue / n
    half = Z * df.revenue_sd / n.pow(0.5)
    df["eur_lo"] = (df.eur_por_peticion - half).clip(lower=0)
    df["eur_hi"] = df.eur_por_peticion + half
    return df


def recommend(vt: pd.DataFrame, min_n: int = 300) -> tuple[str, str]:
    """Which version should run in the final? Returns (level, markdown text); level in good|warning|info."""
    if vt is None or vt.empty:
        return "info", "Todavía no hay feedback en esta ventana."
    cand = vt[(vt.tipo == "versión") & (vt.model_version != NO_ANSWER) & (vt.servidas >= min_n)]
    cand = cand.sort_values("compras_por_100", ascending=False)
    if cand.empty:
        return "info", (f"Ninguna versión tiene aún **≥ {min_n} peticiones servidas** en esta ventana. "
                        "Con menos, la diferencia entre versiones es sobre todo ruido: espera antes de decidir.")
    best = cand.iloc[0]
    parts = [f"**{best.model_version}** lidera con **{num(best.compras_por_100)} compras/100** "
             f"(IC 95 %: {num(best.compras_por_100_lo)}–{num(best.compras_por_100_hi)}; "
             f"n = {num(best.servidas, 0)})."]
    level = "good"
    if len(cand) > 1:
        second = cand.iloc[1]
        p = two_prop_pvalue(best.compras, best.servidas, second.compras, second.servidas)
        diff = best.compras_por_100 - second.compras_por_100
        if p < 0.05:
            parts.append(f"Supera a **{second.model_version}** en +{num(diff)} compras/100, "
                         f"diferencia significativa ({fmt_p(p)}).")
        else:
            level = "warning"
            parts.append(f"La ventaja sobre **{second.model_version}** (+{num(diff)}) **no es concluyente** "
                         f"({fmt_p(p)}): puede ser ruido.")
    ref = vt[vt.tipo == "referencia"]
    if not ref.empty and ref.iloc[0].servidas >= 100:
        r = ref.iloc[0]
        p = two_prop_pvalue(best.compras, best.servidas, r.compras, r.servidas)
        parts.append(f"Frente al fallback (best sellers): {'+' if best.compras_por_100 >= r.compras_por_100 else ''}"
                     f"{num(best.compras_por_100 - r.compras_por_100)} compras/100 ({fmt_p(p)}).")
    if not math.isnan(best.a_tiempo_pct) and best.a_tiempo_pct < 95:
        level = "warning"
        parts.append(f"Ojo: solo el {num(best.a_tiempo_pct, 1)} % de sus respuestas llegan a tiempo; "
                     "lo que no llega se convierte en fallback y resta puntos de fiabilidad.")
    if level == "good":
        parts.append(f"→ **Decisión:** llevar **{best.model_version}** a la final.")
    else:
        parts.append("→ **Decisión:** aún no hay un ganador claro. Recoge más datos o compáralas a la vez "
                     "(A/B por `request_id`); en caso de empate, elige la más fiable y sencilla.")
    return level, " ".join(parts)


# ---------------------------------------------------------------- ② segments

SEGMENT_DIMENSIONS = {
    "Tipo de usuario": "usuario",
    "Cold start (puntúa aparte: 10 pts)": "cold_label",
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
    df["cold_label"] = _truthy(df["cold"]).map({True: "cold (nuevo o mira producto nuevo)", False: "no cold"})
    for c in ["device", "region", "page"]:
        df[c] = df[c].fillna("").replace("", "(vacío)")
    return df


def segment_table(prepared: pd.DataFrame, column: str) -> pd.DataFrame:
    g = prepared.groupby(column, as_index=False)[["servidas", "clics", "compras"]].sum()
    g = add_rate(g, "compras", "servidas", "compras_por_100")
    g = add_rate(g, "clics", "servidas", "clics_por_100")
    return g.sort_values("compras_por_100", ascending=False).reset_index(drop=True)


def segment_conclusion(table: pd.DataFrame, column: str, dim_name: str, min_n: int = 200) -> tuple[str, str]:
    tot_k, tot_n = table.compras.sum(), table.servidas.sum()
    if tot_n == 0:
        return "info", "Sin datos en esta ventana."
    overall = 100 * tot_k / tot_n
    t = table[table.servidas >= min_n]
    if len(t) < 2:
        return "info", f"Hace falta más tráfico por segmento (≥ {min_n} peticiones en al menos dos grupos)."
    worst = t.sort_values("compras_por_100").iloc[0]
    rest_k, rest_n = tot_k - worst.compras, tot_n - worst.servidas
    p = two_prop_pvalue(worst.compras, worst.servidas, rest_k, rest_n)
    txt = (f"Media global: **{num(overall)} compras/100**. El segmento más débil por {dim_name.lower()} es "
           f"**{worst[column]}** con {num(worst.compras_por_100)} ({fmt_p(p)} frente al resto, "
           f"{num(100 * worst.servidas / tot_n, 0)} % del tráfico).")
    if p < 0.05:
        return "warning", txt + f" → **Decisión:** priorizar mejoras para **{worst[column]}**."
    return "good", txt + " → Las diferencias entre segmentos no son significativas: no hace falta tratarlos aparte."


# ---------------------------------------------------------------- ③ carousel position

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
    clicks = pt.clics.sum()
    if clicks == 0:
        return "info", "Aún no hay clics suficientes."
    top3 = 100 * pt.clics.head(3).sum() / clicks
    first_last = (pt.clics.iloc[0] / pt.clics.iloc[-1]) if pt.clics.iloc[-1] > 0 else math.inf
    ratio = "∞" if math.isinf(first_last) else f"{num(first_last, 1)}×"
    return "good", (f"Las 3 primeras posiciones reciben el **{num(top3, 0)} % de los clics**; la posición 1 recibe {ratio} "
                    "los clics de la 10. → **Decisión:** ordenar por probabilidad de compra estimada y reservar las "
                    "primeras posiciones para los productos más seguros (exploración, en las últimas).")


# ---------------------------------------------------------------- ④ live trends

def trend_table(raw: pd.DataFrame, cat: pd.DataFrame, level: str = "category", min_count: int = 20) -> pd.DataFrame:
    """Share of purchases per category in the recent window vs the earlier baseline.

    raw: product_id, recientes, base (purchase counts) ; cat: index product_id with `level` column.
    """
    df = raw.copy()
    for c in ["recientes", "base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[level] = df.product_id.map(cat[level]) if level in cat else None
    df[level] = df[level].fillna("(producto nuevo / sin catálogo)")
    g = df.groupby(level, as_index=False)[["recientes", "base"]].sum()
    n_rec, n_base = g.recientes.sum(), g.base.sum()
    if n_rec == 0 or n_base == 0:
        return g.assign(cuota_reciente=math.nan, cuota_base=math.nan, cambio_pp=math.nan, p=math.nan)
    g["cuota_reciente"] = 100 * g.recientes / n_rec
    g["cuota_base"] = 100 * g.base / n_base
    g["cambio_pp"] = g.cuota_reciente - g.cuota_base
    g["p"] = [two_prop_pvalue(a, n_rec, b, n_base) for a, b in zip(g.recientes, g.base)]
    g = g[(g.recientes + g.base) >= min_count]
    g["dirección"] = g.cambio_pp.map(lambda x: "sube" if x >= 0 else "baja")
    return g.reindex(g.cambio_pp.abs().sort_values(ascending=False).index).reset_index(drop=True)


def trend_conclusion(tt: pd.DataFrame, recent_min: int, level_name: str) -> tuple[str, str]:
    if tt.empty or tt.cambio_pp.isna().all():
        return "info", "Aún no hay compras suficientes en la tienda para medir tendencias."
    sig = tt[tt.p < 0.01]
    if sig.empty:
        return "good", (f"Ninguna {level_name} cambia de forma significativa en los últimos {recent_min} min. "
                        "→ **Decisión:** la popularidad histórica basta; una señal de tendencia de ventana corta "
                        "añadiría ruido.")
    top = sig.iloc[0]
    return "warning", (f"**{len(sig)}** {level_name}(s) cambian de forma significativa en los últimos {recent_min} min "
                       f"(p < 0,01); la mayor es **{top.iloc[0]}** ({'+' if top.cambio_pp >= 0 else ''}"
                       f"{num(top.cambio_pp, 1)} pp de cuota). → **Decisión:** incluir en el ranking una señal de "
                       f"popularidad reciente (ventana ≈ {recent_min} min) y refrescarla cada pocos minutos.")


# ---------------------------------------------------------------- ⑤ offline vs online

def offline_vs_online(vt: pd.DataFrame, offline: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """offline: columns model_version + one numeric metric (first numeric column used)."""
    off = offline.copy()
    off["model_version"] = off["model_version"].astype(str).str.strip()
    metric = [c for c in off.columns if c != "model_version"][0]
    off = off.rename(columns={metric: "offline"})
    off["offline"] = pd.to_numeric(off["offline"], errors="coerce")
    m = vt[vt.tipo == "versión"][["model_version", "servidas", "compras_por_100"]].merge(off, on="model_version")
    m = m.dropna(subset=["offline", "compras_por_100"])
    rho = m.offline.rank().corr(m.compras_por_100.rank()) if len(m) >= 3 else math.nan
    return m, rho


def offline_conclusion(m: pd.DataFrame, rho: float) -> tuple[str, str]:
    if len(m) < 3 or math.isnan(rho):
        return "info", ("Con menos de 3 versiones evaluadas offline y online no se puede medir la concordancia. "
                        "Añade filas a `tools/offline_results.csv` a medida que despliegas versiones.")
    if rho >= 0.7:
        return "good", (f"La evaluación offline ordena las versiones casi igual que el stream "
                        f"(Spearman ρ = {num(rho)}). → **Decisión:** se puede usar para descartar ideas sin desplegarlas.")
    return "warning", (f"La evaluación offline y el stream no coinciden (Spearman ρ = {num(rho)}). "
                       "→ **Decisión:** no descartar ideas solo con la métrica offline; revisar cómo se construye "
                       "(p. ej. usar IPS con las propensiones de `history_sessions`).")


def md_to_html(text: str) -> str:
    """Tiny markdown → HTML for the decision boxes (bold and inline code only)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


# ---------------------------------------------------------------- self-checks

if __name__ == "__main__":
    lo, hi = wilson(5, 100)
    assert 0.02 < lo < 0.05 < hi < 0.12, (lo, hi)
    assert math.isnan(wilson(0, 0)[0])
    assert two_prop_pvalue(50, 1000, 50, 1000) == 1.0
    assert two_prop_pvalue(80, 1000, 40, 1000) < 0.001

    raw = pd.DataFrame([
        dict(model_version="v2-cooc", requests=4200, servidas=4100, clics=900, compras=205, revenue=6100.0,
             revenue_sd=40.0, desde="2026-11-14 15:00", hasta="2026-11-14 16:00"),
        dict(model_version="v1-pop", requests=5000, servidas=4950, clics=800, compras=170, revenue=5200.0,
             revenue_sd=38.0, desde="2026-11-14 13:00", hasta="2026-11-14 14:30"),
        dict(model_version="(sin respuesta)", requests=40, servidas=0, clics=0, compras=0, revenue=0.0,
             revenue_sd=None, desde="2026-11-14 13:00", hasta="2026-11-14 16:00"),
    ])
    vt = version_table(raw, {"servidas": 300, "clics": 40, "compras": 6, "revenue": 150.0, "revenue_sd": 30.0})
    assert vt.loc[vt.model_version == REFERENCE, "tipo"].item() == "referencia"
    assert abs(vt.loc[0, "compras_por_100"] - 5.0) < 0.01
    print(recommend(vt, 300))

    seg = prepare_segments(pd.DataFrame([
        dict(device="mobile", region="ES", page="home", is_new=True, cold=True, servidas=900, clics=100, compras=20),
        dict(device="desktop", region="ES", page="product", is_new=False, cold=False, servidas=3000, clics=600, compras=150),
        dict(device="mobile", region="FR", page="product", is_new="false", cold="false", servidas=1200, clics=250, compras=60),
    ]))
    st_ = segment_table(seg, "usuario")
    print(st_[["usuario", "servidas", "compras_por_100"]])
    print(segment_conclusion(st_, "usuario", "Tipo de usuario"))

    pos = pd.DataFrame([dict(tipo="total", pos=0, n=4000)]
                       + [dict(tipo="clic", pos=i, n=int(300 / i)) for i in range(1, 11)]
                       + [dict(tipo="compra", pos=i, n=int(40 / i)) for i in range(1, 11)]
                       + [dict(tipo="compra", pos=0, n=3)])
    pt, total, outside = position_table(pos)
    assert total == 4000 and outside["compras"] == 3
    print(pt.head(3)); print(position_conclusion(pt))

    cat = pd.DataFrame({"category": ["A", "A", "B", "C"]}, index=["p1", "p2", "p3", "p4"])
    ev = pd.DataFrame(dict(product_id=["p1", "p2", "p3", "p4", "p9"], recientes=[200, 50, 30, 40, 25],
                           base=[600, 300, 400, 500, 10]))
    tt = trend_table(ev, cat)
    print(tt); print(trend_conclusion(tt, 30, "categoría"))

    m, rho = offline_vs_online(vt, pd.DataFrame({"model_version": ["v1-pop", "v2-cooc"], "hit@10": [0.21, 0.27]}))
    print(m, rho, offline_conclusion(m, rho))
    print(md_to_html("**hola** y `code`"))
    print("OK")