"""Live dashboard of our RawTree recommender (Datathon FME 2026).

    streamlit run tools/dashboard.py

Three tabs:
- En directo: reliability right now (the organisers' status light), latency, the requests flowing in and what the
  store is buying. Source: RawTree (shop.requests, submissions, feedback, shop.events).
- Versiones: the OFFICIAL metric (expected purchases per 100 requests from the public leaderboard API: no luck) for
  each of our versions, relative to the organisers' reference in the same 15 minutes, with intervals that include
  the drift of the store; plus clean A/B comparisons of versions that ran at the same time.
- Diagnóstico: realised clicks and purchases from RawTree feedback: segments, carousel position, store trends.

Reads TEAM_ID and RAWTREE_API_KEY from .env (project root). Read-only: it never writes to RawTree.
Times are shown in Barcelona time (CET = UTC+1); the data is in UTC. Statistics: tools/decision_stats.py.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

import decision_stats as ds
from decision_stats import num

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
LEADERBOARD = "https://datathon.rawtree.dev/api/state?by=model"
PIECE = 900                                   # the leaderboard API's smallest window: 15 minutes
CET = timedelta(hours=1)
OFFLINE_CSV = Path(__file__).resolve().parent / "offline_results.csv"

# Reference categorical palette (fixed order, validated for colour-blind safety on adjacent series), status colours
# that always travel with an icon + label, and text that never takes a series colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, RED = SERIES[0], SERIES[7]
TEXT, TEXT_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE, BORDER = "#fcfcfb", "rgba(11,11,11,0.10)"
STATUS = {"good": ("#0ca30c", "✅", "Resultado claro"), "warning": ("#fab219", "⚠️", "Atención"),
          "info": (BLUE, "ℹ️", "Lectura")}

st.set_page_config(page_title=f"Recomendador · {TEAM}", page_icon="📊", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.8rem; max-width: 1320px; }}
  h1 {{ font-size: 1.75rem; letter-spacing: -0.01em; margin-bottom: 0; }}
  h2, h3, h4 {{ letter-spacing: -0.01em; }}
  [data-testid="stMetric"] {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px;
                              padding: 12px 16px; }}
  [data-testid="stMetricLabel"] p {{ color: {TEXT_2}; font-size: .76rem; text-transform: uppercase;
                                     letter-spacing: .04em; }}
  .note {{ border: 1px solid {BORDER}; border-left: 4px solid var(--accent); border-radius: 8px; background: #ffffff;
           padding: 10px 14px; margin: 2px 0 14px; color: {TEXT}; line-height: 1.55; }}
  .note .tag {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: {TEXT_2};
                font-weight: 600; }}
  .lead {{ color: {TEXT_2}; margin: -2px 0 12px; font-size: .95rem; line-height: 1.5; }}
  .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
           border: 1px solid {BORDER}; background: #ffffff; font-weight: 600; margin: 4px 0 12px; }}
  .pill .dot {{ width: 11px; height: 11px; border-radius: 50%; background: var(--c); }}
  .pill .sub {{ color: {TEXT_2}; font-weight: 400; }}
</style>
""", unsafe_allow_html=True)


# ================================================================ data access

@st.cache_resource
def session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
    return s


def q(sql: str) -> pd.DataFrame:
    r = session().post(f"{API}/v1/query", json={"sql": sql}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"RawTree {r.status_code}: {r.text[:300]}")
    return pd.DataFrame(r.json()["data"])


def to_cet(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col) + CET


def epoch_to_cet(t) -> pd.Series:
    return pd.to_datetime(t, unit="s") + CET


def sql_str(s: str) -> str:
    return str(s).replace("\\", "").replace("'", "")


@st.cache_data(ttl=600, show_spinner=False)
def catalog() -> pd.DataFrame:
    """Current catalogue: shop.catalog with the latest non-empty value of each field from catalog_updates."""
    df = q("""
        SELECT product_id,
               argMaxIf(category, v, category != '') AS category,
               argMaxIf(subcategory, v, subcategory != '') AS subcategory,
               argMaxIf(brand, v, brand != '') AS brand,
               argMaxIf(price_eur, v, price_eur > 0) AS price_eur
        FROM (
            SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
                   brand::String AS brand, price_eur::Float64 AS price_eur, '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64, toString(ts)
            FROM shop.catalog_updates
        )
        GROUP BY product_id""")
    return df.set_index("product_id")


# ---------------------------------------------------------------- official metric: leaderboard API, 15-min pieces

@st.cache_resource
def piece_store() -> dict:
    """{piece start: (fetched at, rows)}; pieces older than 20 minutes no longer change."""
    return {}


def _fetch_piece(t: int) -> tuple[int, list[dict] | None]:
    try:
        r = requests.get(f"{LEADERBOARD}&from={t}&to={t + PIECE}", timeout=20)
        r.raise_for_status()
        return t, ds.parse_piece(t, r.json(), TEAM)
    except Exception:
        return t, None


def official(hours: float) -> pd.DataFrame:
    store, now = piece_store(), time.time()
    end = int(now) // PIECE * PIECE                       # the last complete piece ends here
    starts = list(range(end - int(hours * 3600), end, PIECE))
    todo = [t for t in starts if t not in store or (t + PIECE > now - 1200 and now - store[t][0] > 60)]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for t, rows in ex.map(_fetch_piece, todo):
                if rows is not None:
                    store[t] = (now, rows)
    return pd.DataFrame([r for t in starts if t in store for r in store[t][1]])


# ---------------------------------------------------------------- RawTree: live tab

def kpis(minutes: int) -> pd.Series:
    df = q(f"""
        SELECT count() AS requests,
               countIf(served_by = 'team') AS on_time,
               countIf(fallback_reason = 'missing') AS missing,
               countIf(fallback_reason = 'late') AS late,
               countIf(fallback_reason = 'invalid') AS invalid,
               countIf(served_by = 'team' AND purchased != '') AS purchases,
               quantileIf(0.5)(latency_ms::Float64, served_by = 'team') AS p50,
               quantileIf(0.95)(latency_ms::Float64, served_by = 'team') AS p95
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND ts > now() - INTERVAL {minutes} MINUTE""")
    return df.iloc[0]


def per_minute(minutes: int) -> pd.DataFrame:
    df = q(f"""
        SELECT toStartOfMinute(ts::DateTime64(3)) AS minute,
               count() AS requests,
               100 * countIf(served_by = 'team') / count() AS on_time_pct,
               quantileIf(0.95)(latency_ms::Float64, served_by = 'team') AS p95_ms
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND ts > now() - INTERVAL {minutes} MINUTE
        GROUP BY minute ORDER BY minute""")
    if not df.empty:
        df["minute"] = to_cet(df["minute"])
    return df


def live_requests(minutes: int, limit: int) -> pd.DataFrame:
    """Each recent request joined with our answer (submissions) and what happened (feedback)."""
    return q(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new,
               r.device AS device, r.page AS page, r.views AS recent_views,
               s.items AS items, s.model_version AS model_version,
               f.request_id != '' AS has_feedback, f.served_by AS served_by, f.fallback_reason AS reason,
               f.latency_ms AS latency_ms, f.shown AS shown,
               f.clicked AS clicked, f.purchased AS purchased, f.revenue_eur AS revenue
        FROM (SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, user_id::String AS user_id,
                     is_new_user::Bool AS is_new_user, context.device::String AS device,
                     context.page::String AS page,
                     arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS views
              FROM shop.requests WHERE ts > now() - INTERVAL {minutes} MINUTE
              ORDER BY ts DESC LIMIT {limit}) AS r
        LEFT JOIN (SELECT request_id::String AS request_id, items::Array(String) AS items,
                          model_version::String AS model_version
                   FROM {TEAM}.submissions
                   WHERE request_id IN (SELECT request_id::String FROM shop.requests WHERE ts > now() - INTERVAL {minutes} MINUTE)
                  ) AS s ON s.request_id = r.request_id
        LEFT JOIN (SELECT request_id::String AS request_id, served_by::String AS served_by,
                          fallback_reason::String AS fallback_reason, latency_ms::Float64 AS latency_ms,
                          shown::Array(String) AS shown, clicked::String AS clicked,
                          purchased::String AS purchased, revenue_eur::Float64 AS revenue_eur
                   FROM {TEAM}.feedback WHERE ts > now() - INTERVAL {minutes + 1} MINUTE
                  ) AS f ON f.request_id = r.request_id
        ORDER BY r.ts DESC""")


def shop_top_products(minutes: int) -> pd.DataFrame:
    return q(f"""
        SELECT product_id::String AS product_id,
               countIf(event_type = 'view') AS vistas,
               countIf(event_type = 'add_to_cart') AS carritos,
               countIf(event_type = 'purchase') AS compras
        FROM shop.events WHERE ts > now() - INTERVAL {minutes} MINUTE
        GROUP BY product_id ORDER BY compras DESC, vistas DESC LIMIT 12""")


def one_request(request_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rid = sql_str(request_id)
    req = q(f"SELECT * FROM shop.requests WHERE request_id = '{rid}'")
    sub = q(f"SELECT * FROM {TEAM}.submissions WHERE request_id = '{rid}'")
    fb = q(f"SELECT * FROM {TEAM}.feedback WHERE request_id = '{rid}'")
    return req, sub, fb


# ---------------------------------------------------------------- RawTree: diagnosis (cached 2 min)

def version_filter(version: str) -> str:
    """SQL clause restricting feedback rows to requests answered by one model_version."""
    if not version or version.startswith("Todas"):
        return ""
    return (f"AND request_id::String IN (SELECT request_id::String FROM {TEAM}.submissions "
            f"WHERE model_version::String = '{sql_str(version)}')")


@st.cache_data(ttl=120, show_spinner=False)
def version_raw(hours: int) -> pd.DataFrame:
    """Per version: realised outcomes ONLY where the shopper saw our answer (served_by = 'team')."""
    return q(f"""
        SELECT if(s.model_version = '', '{ds.NO_ANSWER}', s.model_version) AS model_version,
               count() AS requests,
               countIf(f.served_by = 'team') AS servidas,
               countIf(f.served_by = 'team' AND f.clicked != '') AS clics,
               countIf(f.served_by = 'team' AND f.purchased != '') AS compras,
               sumIf(f.revenue_eur, f.served_by = 'team') AS revenue,
               stddevSampIf(f.revenue_eur, f.served_by = 'team') AS revenue_sd,
               min(f.ts) AS desde, max(f.ts) AS hasta
        FROM (SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, served_by::String AS served_by,
                     clicked::String AS clicked, purchased::String AS purchased, revenue_eur::Float64 AS revenue_eur
              FROM {TEAM}.feedback WHERE request_id != '__init__' AND ts > now() - INTERVAL {hours} HOUR) AS f
        LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS model_version
                   FROM {TEAM}.submissions GROUP BY request_id) AS s ON s.request_id = f.request_id
        GROUP BY model_version ORDER BY servidas DESC""")


@st.cache_data(ttl=300, show_spinner=False)
def new_products() -> list[str]:
    """Products that appeared during the event (in catalog_updates but not in the original catalogue)."""
    df = q("""
        SELECT DISTINCT product_id::String AS product_id FROM shop.catalog_updates
        WHERE product_id::String NOT IN (SELECT product_id::String FROM shop.catalog)""")
    return [] if df.empty else df.product_id.astype(str).tolist()


@st.cache_data(ttl=120, show_spinner=False)
def segments_raw(hours: int, version: str, new_ids: tuple[str, ...]) -> pd.DataFrame:
    ids = ", ".join(f"'{sql_str(i)}'" for i in new_ids)
    cold = f"is_new OR hasAny(views, [{ids}])" if ids else "is_new"
    return q(f"""
        SELECT r.device AS device, r.region AS region, r.page AS page, r.is_new AS is_new, r.cold AS cold,
               count() AS servidas, countIf(f.clicked != '') AS clics, countIf(f.purchased != '') AS compras
        FROM (SELECT request_id::String AS request_id, clicked::String AS clicked, purchased::String AS purchased
              FROM {TEAM}.feedback
              WHERE request_id != '__init__' AND served_by::String = 'team'
                AND ts > now() - INTERVAL {hours} HOUR {version_filter(version)}) AS f
        INNER JOIN (SELECT request_id::String AS request_id, context.device::String AS device,
                           context.region::String AS region, context.page::String AS page,
                           is_new_user::Bool AS is_new,
                           arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS views,
                           {cold} AS cold
                    FROM shop.requests WHERE ts > now() - INTERVAL {hours + 1} HOUR) AS r
            ON r.request_id = f.request_id
        GROUP BY device, region, page, is_new, cold""")


@st.cache_data(ttl=120, show_spinner=False)
def position_raw(hours: int, version: str) -> pd.DataFrame:
    where = (f"request_id != '__init__' AND served_by::String = 'team' "
             f"AND ts > now() - INTERVAL {hours} HOUR {version_filter(version)}")
    return q(f"""
        SELECT 'clic' AS tipo, indexOf(shown, clicked) AS pos, count() AS n
        FROM (SELECT shown::Array(String) AS shown, clicked::String AS clicked
              FROM {TEAM}.feedback WHERE {where} AND clicked::String != '')
        GROUP BY pos
        UNION ALL
        SELECT 'compra' AS tipo, indexOf(shown, purchased) AS pos, count() AS n
        FROM (SELECT shown::Array(String) AS shown, purchased::String AS purchased
              FROM {TEAM}.feedback WHERE {where} AND purchased::String != '')
        GROUP BY pos
        UNION ALL
        SELECT 'total' AS tipo, 0 AS pos, count() AS n FROM {TEAM}.feedback WHERE {where}""")


@st.cache_data(ttl=120, show_spinner=False)
def trends_raw(hours: int, recent_min: int) -> pd.DataFrame:
    return q(f"""
        SELECT product_id::String AS product_id,
               countIf(ts > now() - INTERVAL {recent_min} MINUTE) AS recientes,
               countIf(ts <= now() - INTERVAL {recent_min} MINUTE) AS base
        FROM shop.events
        WHERE event_type = 'purchase' AND ts > now() - INTERVAL {hours} HOUR
        GROUP BY product_id""")


# ================================================================ presentation helpers

def note(level: str, text: str) -> None:
    """Message box: colour + icon + label (the colour never carries the meaning alone)."""
    colour, icon, label = STATUS.get(level, STATUS["info"])
    st.markdown(f'<div class="note" style="--accent:{colour}"><span class="tag">{icon} {label}</span><br>'
                f'{ds.md_to_html(text)}</div>', unsafe_allow_html=True)


def lead(text: str) -> None:
    st.markdown(f'<div class="lead">{ds.md_to_html(text)}</div>', unsafe_allow_html=True)


def styled(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Recessive axes and grid, transparent background, text in text colours."""
    return (chart.configure(background="transparent", font="Source Sans Pro, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=TEXT_2,
                            titleColor=TEXT_2, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(orient="top", labelColor=TEXT_2, titleColor=TEXT_2, labelFontSize=12))


def show_chart(chart) -> None:
    chart = styled(chart)
    try:  # newer Streamlit
        st.altair_chart(chart, width="stretch", theme=None)
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True, theme=None)


def ci_chart(df: pd.DataFrame, label: str, rate: str, x_title: str, n_col: str = "servidas") -> alt.Chart:
    """Dot = estimate, line = 95 % interval, best on top."""
    order = df.sort_values(rate, ascending=False)[label].astype(str).tolist()
    y = alt.Y(f"{label}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    tip = [alt.Tooltip(f"{label}:N", title="grupo"), alt.Tooltip(f"{rate}:Q", format=".2f", title="estimación"),
           alt.Tooltip(f"{rate}_lo:Q", format=".2f", title="IC 95 % inf."),
           alt.Tooltip(f"{rate}_hi:Q", format=".2f", title="IC 95 % sup."),
           alt.Tooltip(f"{n_col}:Q", format=",.0f", title="peticiones")]
    base = alt.Chart(df.dropna(subset=[rate]))
    rule = base.mark_rule(strokeWidth=2.5, color=BLUE).encode(
        y=y, x=alt.X(f"{rate}_lo:Q", title=x_title, scale=alt.Scale(zero=True)), x2=f"{rate}_hi:Q", tooltip=tip)
    dot = base.mark_point(filled=True, size=110, opacity=1, color=BLUE, stroke=SURFACE, strokeWidth=2).encode(
        y=y, x=f"{rate}:Q", tooltip=tip)
    return (rule + dot).properties(height=max(110, 38 * len(order)))


def ratio_chart(summ: pd.DataFrame, lead_team: str | None) -> alt.Chart:
    """Each version (and the leading other team) relative to the reference: dot + 95 % interval, line at 1."""
    d = summ.assign(grupo=summ.kind.map({"versión": "nuestras versiones", "equipo": "mejor otro equipo"}))
    order = d.sort_values("ratio", ascending=False).who.tolist()
    y = alt.Y("who:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    colour = alt.Color("grupo:N", scale=alt.Scale(domain=["nuestras versiones", "mejor otro equipo"],
                                                  range=[BLUE, MUTED]), legend=alt.Legend(title=None))
    tip = [alt.Tooltip("who:N", title="versión / equipo"),
           alt.Tooltip("ratio:Q", format=".2f", title="× referencia"),
           alt.Tooltip("ratio_lo:Q", format=".2f", title="IC 95 % inf."),
           alt.Tooltip("ratio_hi:Q", format=".2f", title="IC 95 % sup."),
           alt.Tooltip("valor:Q", format=".2f", title="valor medio"),
           alt.Tooltip("pieces:Q", title="tramos de 15 min"),
           alt.Tooltip("requests:Q", format=",.0f", title="peticiones")]
    lo = float(min(d.ratio_lo.min(), d.ratio.min(), 1.0))
    hi = float(max(d.ratio_hi.max(), d.ratio.max(), 1.0))
    pad = 0.08 * (hi - lo or 1)
    x = alt.X("ratio_lo:Q", title="× referencia (1 = igual que la referencia)",
              scale=alt.Scale(domain=[lo - pad, hi + pad], zero=False))
    base = alt.Chart(d)
    rule = base.mark_rule(strokeWidth=2.5).encode(y=y, x=x, x2="ratio_hi:Q", color=colour, tooltip=tip)
    dot = base.mark_point(filled=True, size=110, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        y=y, x="ratio:Q", color=colour, tooltip=tip)
    one = alt.Chart(pd.DataFrame({"x": [1.0], "t": ["referencia"]}))
    ref = one.mark_rule(color=TEXT_2, strokeDash=[4, 3]).encode(x="x:Q")
    ref_label = one.mark_text(align="left", dx=4, dy=-6, color=TEXT_2, fontSize=11).encode(
        x="x:Q", y=alt.value(0), text="t:N")
    return (ref + rule + dot + ref_label).properties(height=max(120, 40 * len(order)))


def timeline_chart(df: pd.DataFrame, summ: pd.DataFrame, metric: str, lead_team: str | None) -> alt.Chart:
    """Ratio to the reference piece by piece: how stable each version is, and when each one ran."""
    ours = summ[summ.kind == "versión"].sort_values("requests", ascending=False).head(len(SERIES) - 1)
    names = sorted(ours.who, key=lambda v: float(ours.set_index("who").desde[v]))   # colour by first appearance
    keep = names + ([lead_team] if lead_team else [])
    d = df[df.who.isin(keep) & (df.n >= ds.MIN_PIECE_REQUESTS)].copy()
    d["ratio"] = d[metric] / d[f"ref_{metric}"]
    d["hora"] = epoch_to_cet(d.t + PIECE / 2)
    d = d.sort_values(["who", "t"])
    d["tramo"] = (d.groupby("who").t.diff() != PIECE).cumsum()      # break the line where a version stopped
    domain = names + ([lead_team] if lead_team else [])
    rng = SERIES[:len(names)] + ([MUTED] if lead_team else [])
    colour = alt.Color("who:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None))
    tip = [alt.Tooltip("who:N", title="versión / equipo"), alt.Tooltip("hora:T", format="%H:%M", title="tramo"),
           alt.Tooltip("ratio:Q", format=".2f", title="× referencia"),
           alt.Tooltip(f"{metric}:Q", format=".2f", title="valor"), alt.Tooltip("n:Q", title="peticiones")]
    x = alt.X("hora:T", title=None, axis=alt.Axis(format="%H:%M", labelAngle=0))
    yy = alt.Y("ratio:Q", title="× referencia", scale=alt.Scale(zero=False))
    line = alt.Chart(d).mark_line(strokeWidth=2).encode(x=x, y=yy, color=colour, detail="tramo:N")
    pts = alt.Chart(d).mark_point(filled=True, size=45, opacity=1).encode(x=x, y=yy, color=colour, tooltip=tip)
    ref = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=TEXT_2, strokeDash=[4, 3]).encode(y="y:Q")
    return (ref + line + pts).properties(height=300)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    tip = [alt.Tooltip(f"{x}:O", title="posición"), alt.Tooltip(f"{y}:Q", format=".2f", title=title),
           alt.Tooltip(f"{y}_lo:Q", format=".2f", title="IC 95 % inf."),
           alt.Tooltip(f"{y}_hi:Q", format=".2f", title="IC 95 % sup.")]
    bars = alt.Chart(df).mark_bar(color=BLUE, size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{x}:O", title="posición en el carrusel", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=title), tooltip=tip)
    err = alt.Chart(df).mark_rule(color=TEXT_2, strokeWidth=1.5).encode(x=f"{x}:O", y=f"{y}_lo:Q", y2=f"{y}_hi:Q")
    return (bars + err).properties(height=240)


def threshold_chart(df: pd.DataFrame, col: str, y_title: str, limit: float, limit_label: str) -> alt.Chart:
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(
        x=alt.X("minute:T", title=None, axis=alt.Axis(format="%H:%M", labelAngle=0)),
        y=alt.Y(f"{col}:Q", title=y_title),
        tooltip=[alt.Tooltip("minute:T", format="%H:%M", title="minuto"),
                 alt.Tooltip(f"{col}:Q", format=".1f", title=y_title)])
    lim = alt.Chart(pd.DataFrame({"y": [limit], "t": [limit_label]}))
    rule = lim.mark_rule(color=RED, strokeDash=[4, 3]).encode(y="y:Q")
    text = lim.mark_text(align="left", x=4, dy=-6, color=TEXT_2, fontSize=11).encode(y="y:Q", text="t:N")
    return (rule + line + text).properties(height=210)


def describe(products, cat: pd.DataFrame, clicked: str = "", purchased: str = "") -> pd.DataFrame:
    rows = []
    for i, p in enumerate(products, 1):
        info = cat.loc[p] if p in cat.index else None
        mark = "🛒 compra" if p == purchased else ("👆 clic" if p == clicked else "")
        rows.append({"#": i, "producto": p, "resultado": mark,
                     "categoría": None if info is None else info.category,
                     "subcategoría": None if info is None else info.subcategory,
                     "marca": None if info is None else info.brand,
                     "precio €": None if info is None else info.price_eur})
    return pd.DataFrame(rows)


def guarded(title: str, fn) -> None:
    """Run one block; a failing query only breaks its own block."""
    try:
        fn()
    except Exception as e:
        st.error(f"{title}: no se pudo consultar ({e})")


# ================================================================ page

st.title("Panel del recomendador")
lead(f"{TEAM} · petición → nuestra respuesta → lo que hizo el comprador · hora de Barcelona")

with st.sidebar:
    st.header("Opciones")
    auto = st.toggle("Refresco automático (En directo)", value=True)
    every = st.select_slider("Cada", [5, 10, 15, 30, 60], value=10, format_func=lambda s: f"{s} s")
    kpi_minutes = st.select_slider("Ventana de los indicadores", [5, 15, 30, 60], value=15,
                                   format_func=lambda m: f"{m} min")
    n_rows = st.slider("Peticiones en la tabla", 10, 60, 20)
    st.divider()
    st.markdown("**De dónde sale cada cosa**")
    st.caption("En directo y Diagnóstico: RawTree (lo que pasó de verdad, con suerte). "
               "Versiones: leaderboard oficial (compras esperadas, sin suerte).")
    st.markdown("[Leaderboard ↗](https://datathon.rawtree.dev/leaderboard)")

tab_live, tab_versions, tab_diag = st.tabs(["📡 En directo", "📊 Versiones", "🔎 Diagnóstico"])


# ---------------------------------------------------------------- tab 1: live

def status_cell(row) -> str:
    if not row.has_feedback:
        return "⏳ esperando"
    if row.served_by == "team":
        return "✅ a tiempo"
    return f"❌ {row.reason or 'fallback'}"


def slot_of(product: str, shown) -> str:
    if not product or shown is None:
        return ""
    try:
        return f"#{list(shown).index(product) + 1}"
    except ValueError:
        return "?"


def official_now() -> dict | None:
    """Our official conversion in the last complete 15-minute piece (None while the leaderboard is hidden)."""
    df = official(0.25)
    ours = df[df.kind == "versión"] if not df.empty else df
    if ours.empty:
        return None
    n = ours.n.sum()
    return {"conv": float((ours.conv * ours.n).sum() / n), "ref": float(ours.ref_conv.iloc[0]),
            "desde": epoch_to_cet(int(ours.t.iloc[0])), "n": int(n)}


@st.fragment(run_every=every if auto else None)
def live() -> None:
    try:
        _live()
    except Exception as e:  # keep the page alive if RawTree hiccups
        st.error(f"No se pudo consultar RawTree: {e}")


def _live() -> None:
    # status light: the organisers' rule (> 90 % valid and on time in the last 5 minutes)
    k5 = kpis(5)
    n5 = int(k5.requests or 0)
    ok5 = 100 * k5.on_time / n5 if n5 else 0.0
    green = n5 > 0 and ok5 > 90
    st.markdown(f'<div class="pill" style="--c:{"#0ca30c" if green else "#d03b3b"}"><span class="dot"></span>'
                f'Luz {"verde" if green else "roja"}<span class="sub">· {num(ok5, 1)} % a tiempo y válidas en '
                f'los últimos 5 min ({num(n5, 0)} peticiones)</span></div>', unsafe_allow_html=True)

    k = k5 if kpi_minutes == 5 else kpis(kpi_minutes)
    n = int(k.requests or 0)
    served = int(k.on_time or 0)
    c = st.columns(5)
    c[0].metric("Peticiones por minuto", num(n / kpi_minutes, 0))
    c[1].metric("A tiempo y válidas", f"{num(100 * served / n, 1)} %" if n else "—",
                help="Si baja del 90 % en 5 minutos, la luz se pone roja y se pierden puntos de fiabilidad.")
    c[2].metric("Latencia p50 · p95", f"{num((k.p50 or 0) / 1000)} · {num((k.p95 or 0) / 1000)} s",
                help="Desde que llega la petición hasta que la respuesta está en RawTree. Límite: 3 s.")
    fails = int(k.missing or 0) + int(k.late or 0) + int(k.invalid or 0)
    c[3].metric("Fallbacks", f"{fails:,}".replace(",", "."),
                help=f"Sin respuesta {int(k.missing or 0)} · tarde {int(k.late or 0)} · inválida {int(k.invalid or 0)}. "
                     "En todos ellos el comprador vio los best sellers.")
    off = official_now()
    if off:
        c[4].metric("Conversión oficial", f"{num(off['conv'])} · {num(off['conv'] / off['ref'])}× ref.",
                    help=f"Compras esperadas por 100 peticiones (sin suerte), tramo {off['desde']:%H:%M}–"
                         f"{off['desde'] + timedelta(minutes=15):%H:%M}, y cuántas veces la referencia de los "
                         "organizadores. Fuente: leaderboard.")
    elif served:
        lo, hi = ds.wilson(k.purchases, served)
        c[4].metric("Compras reales / 100", num(100 * k.purchases / served),
                    help=f"Con suerte: IC 95 % {num(100 * lo)}–{num(100 * hi)}. La métrica oficial no está "
                         "disponible (el leaderboard está oculto durante la final).")
    else:
        c[4].metric("Conversión", "—")
    st.caption(f"Indicadores de los últimos {kpi_minutes} min; la luz siempre usa 5 min, como la de los organizadores.")

    pm = per_minute(60)
    if not pm.empty:
        st.markdown("#### Última hora, minuto a minuto")
        a, b = st.columns(2)
        with a:
            show_chart(threshold_chart(pm, "on_time_pct", "% a tiempo y válidas", 90, "90 %: por debajo, luz roja"))
        with b:
            show_chart(threshold_chart(pm, "p95_ms", "latencia p95 (ms)", 3000, "3 s: límite"))

    st.markdown("#### Peticiones en directo")
    df = live_requests(minutes=5, limit=n_rows)
    if df.empty:
        st.info("No hay peticiones en los últimos 5 minutos.")
        return
    df["hora"] = to_cet(df["ts"]).dt.strftime("%H:%M:%S")
    df["estado"] = df.apply(status_cell, axis=1)
    df["vistas"] = df.recent_views.apply(len)
    shown = df.apply(lambda r: r.shown if isinstance(r.shown, list) and len(r.shown) else r["items"], axis=1)
    df["clic"] = [slot_of(p, s) for p, s in zip(df.clicked, shown)]
    df["compra"] = [slot_of(p, s) for p, s in zip(df.purchased, shown)]
    df["latencia s"] = (df.latency_ms / 1000).where(df.has_feedback & (df.served_by == "team"))
    df["€"] = df.revenue.where(df.revenue > 0)
    df["comprador"] = df.is_new.map(lambda x: "nuevo" if str(x).lower() in ("true", "1") else "recurrente")
    st.dataframe(df[["hora", "comprador", "page", "device", "vistas", "model_version", "estado", "latencia s",
                     "clic", "compra", "€", "request_id"]].rename(columns={"page": "página", "device": "dispositivo",
                                                                          "model_version": "versión"}),
                 hide_index=True, width="stretch",
                 column_config={"latencia s": st.column_config.NumberColumn(format="%.2f"),
                                "€": st.column_config.NumberColumn(format="%.2f"),
                                "vistas": st.column_config.NumberColumn(help="Productos que acababa de mirar")})
    st.caption("clic / compra = posición en el carrusel del producto clicado o comprado. "
               "El feedback llega unos segundos después de la petición.")


def shop_and_inspector() -> None:
    st.markdown("#### Lo que más compra la tienda (última hora)")
    lead("Actividad de toda la tienda, no solo de nuestro carrusel.")
    try:
        cat = catalog()
        top = shop_top_products(60)
        if not top.empty:
            top = top.join(cat[["category", "subcategory", "price_eur"]], on="product_id").rename(
                columns={"product_id": "producto", "category": "categoría", "subcategory": "subcategoría",
                         "price_eur": "precio €"})
        st.dataframe(top, hide_index=True, width="stretch")
    except Exception as e:
        st.error(f"No se pudo consultar la tienda: {e}")

    st.divider()
    st.markdown("#### 🔍 Inspeccionar una petición")
    c1, c2 = st.columns([4, 1])
    if c2.button("Última con compra", width="stretch"):
        last = q(f"SELECT request_id FROM {TEAM}.feedback WHERE purchased != '' ORDER BY ts DESC LIMIT 1")
        st.session_state["inspect"] = "" if last.empty else last.request_id[0]
    rid = c1.text_input("request_id", key="inspect", label_visibility="collapsed",
                        placeholder="Pega un request_id de la tabla o pulsa «Última con compra»")
    if not rid:
        return
    try:
        cat = catalog()
        req, sub, fb = one_request(rid.strip())
        if req.empty:
            st.warning("No encuentro esa petición en shop.requests.")
            return
        r = req.iloc[0]
        views = [v["product_id"] for v in (r.get("recent_views") or [])]
        st.markdown(f"**{r.request_id}** · {to_cet(pd.Series([r.ts])).iloc[0]:%H:%M:%S} · comprador `{r.user_id}`"
                    f"{' (nuevo)' if r.is_new_user else ''} · {r.get('context.device')} · "
                    f"{r.get('context.region')} · página *{r.get('context.page')}*")
        f = fb.iloc[0] if not fb.empty else None
        clicked = f.clicked if f is not None else ""
        purchased = f.purchased if f is not None else ""
        a, b = st.columns(2)
        with a:
            st.markdown("**① Lo que acababa de mirar** (el último, abajo)")
            st.dataframe(describe(views, cat), hide_index=True, width="stretch")
        with b:
            shown = list(f.shown) if f is not None else (list(sub.iloc[0]["items"]) if not sub.empty else [])
            who = "" if f is None else (" · nuestra respuesta" if f.served_by == "team"
                                        else f" · FALLBACK ({f.fallback_reason})")
            st.markdown(f"**② Lo que vio**{who}")
            st.dataframe(describe(shown, cat, clicked, purchased), hide_index=True, width="stretch")
        if f is not None:
            st.markdown(f"**③ Resultado:** latencia {f.latency_ms} ms · clic `{clicked or '—'}` · "
                        f"compra `{purchased or '—'}` · {f.revenue_eur} €")
        else:
            st.info("Todavía no hay feedback para esta petición: llega unos segundos después.")
        with st.expander("JSON en bruto de las tres tablas"):
            st.json({"shop.requests": req.to_dict("records"), f"{TEAM}.submissions": sub.to_dict("records"),
                     f"{TEAM}.feedback": fb.to_dict("records")})
    except Exception as e:
        st.error(f"Error: {e}")


# ---------------------------------------------------------------- tab 2: versions (official metric)

def realised_block(hours: int) -> None:
    vt = ds.version_table(version_raw(hours))
    vt = vt[(vt.model_version != ds.NO_ANSWER) & (vt.servidas > 0)]
    if vt.empty:
        st.info("Sin feedback en esta ventana.")
        return
    lead("Compras que ocurrieron de verdad en nuestras respuestas. Llevan suerte: con ~2 compras por cada 100 "
         "peticiones, el intervalo de una versión con 5.000 peticiones es de ±0,4. Sirven para ver la fiabilidad "
         "(«a tiempo»), no para elegir versión.")
    show_chart(ci_chart(vt, "model_version", "compras_por_100", "compras reales / 100 peticiones"))
    t = vt.assign(desde=to_cet(vt.desde).dt.strftime("%H:%M"), hasta=to_cet(vt.hasta).dt.strftime("%H:%M"))
    st.dataframe(t[["model_version", "servidas", "a_tiempo_pct", "compras_por_100", "compras_por_100_lo",
                    "compras_por_100_hi", "eur_por_peticion", "desde", "hasta"]].rename(columns={
                        "model_version": "versión", "a_tiempo_pct": "% a tiempo", "compras_por_100": "compras / 100",
                        "compras_por_100_lo": "IC inf.", "compras_por_100_hi": "IC sup.",
                        "eur_por_peticion": "€ / petición"}),
                 hide_index=True, width="stretch",
                 column_config={c_: st.column_config.NumberColumn(format="%.2f") for c_ in
                                ["% a tiempo", "compras / 100", "IC inf.", "IC sup.", "€ / petición"]})


def versions_tab() -> None:
    lead("La métrica **oficial**: compras esperadas por cada 100 peticiones, calculadas por el simulador con la "
         "probabilidad de compra de cada comprador (sin suerte). La tienda sube y baja ±10 % de una hora a otra, así "
         "que cada versión se mide **frente a la referencia de los organizadores en los mismos 15 minutos**, y su "
         "intervalo sale de cuánto varía tramo a tramo.")
    c = st.columns([1.1, 2.2, 1, 1.2])
    hours = c[0].select_slider("Ventana", [1, 3, 6, 12, 24], value=6, format_func=lambda h: f"{h} h")
    metric = c[1].radio("Métrica", list(ds.METRICS), format_func=lambda m: ds.METRICS[m][0], horizontal=True)
    min_pieces = c[2].select_slider("Mínimo de tramos", [2, 3, 4, 6, 8], value=3,
                                    help="Tramos de 15 minutos que necesita una versión para mostrar su intervalo.")
    show_lead = c[3].toggle("Mejor otro equipo", value=True, help="El equipo con mejor media en la ventana, como "
                                                                   "referencia de lo que se puede conseguir.")
    with st.spinner("Leyendo el leaderboard por tramos de 15 minutos…"):
        df = official(hours)
    if df.empty or (df.kind == "versión").sum() == 0:
        note("info", "El leaderboard no devuelve datos nuestros en esta ventana (durante la final, de 10:00 a 12:00, "
                     "está oculto). Debajo, las compras reales registradas en RawTree.")
        guarded("Compras reales", lambda: realised_block(hours))
        return

    lead_team = ds.leader(df, metric) if show_lead else None
    summ = ds.version_summary(df, metric, min_pieces, extra=(lead_team,) if lead_team else ())
    level, text = ds.versions_message(summ, df, metric, lead_team)
    note(level, text)
    enough = summ[summ.suficiente]
    if not enough.empty:
        a, b = st.columns([1, 1.3])
        with a:
            st.markdown(f"**{ds.METRICS[metric][0]} frente a la referencia**")
            show_chart(ratio_chart(enough, lead_team))
        with b:
            st.markdown("**Tramo a tramo**")
            show_chart(timeline_chart(df, enough, metric, lead_team))
    thin = summ[~summ.suficiente & (summ.kind == "versión")]
    if not thin.empty:
        st.caption(f"Con menos de {min_pieces} tramos (sin intervalo fiable): " + ", ".join(thin.who))

    st.markdown("#### Comparaciones limpias: A/B")
    lead("Versiones que respondieron **a la vez**, repartiéndose las peticiones al azar: mismos compradores, misma "
         "hora. Es la única comparación que no depende de la hora del día.")
    pairs = ds.ab_pairs(df, metric)
    if not pairs:
        st.caption("En esta ventana no hay versiones que hayan corrido a la vez durante al menos 4 tramos.")
    else:
        rows = [{"comparación": f"{p['a']} − {p['b']}", "tramos": p["pieces"],
                 "diferencia": p["diff"], "IC 95 %": f"{ds.signed(p['lo'])} a {ds.signed(p['hi'])}",
                 "relativa": f"{ds.signed(p['rel'], 1)} %", "resultado": {
                     "mejor": f"✅ {p['a']} es mejor", "peor": f"❌ {p['a']} es peor",
                     "sin diferencia clara": "➖ sin diferencia clara", "pocos datos": "pocos datos"}[p["verdict"]]}
                for p in pairs]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"diferencia": st.column_config.NumberColumn(format="%+.3f")})
        st.caption("Si el intervalo cruza el 0, la diferencia puede ser ruido.")

    with st.expander("Tabla de versiones"):
        t = summ.assign(desde=epoch_to_cet(summ.desde).dt.strftime("%a %H:%M"),
                        hasta=epoch_to_cet(summ.hasta).dt.strftime("%a %H:%M"))
        st.dataframe(t[["who", "requests", "pieces", "valor", "ratio", "ratio_lo", "ratio_hi", "desde", "hasta"]]
                     .rename(columns={"who": "versión / equipo", "requests": "peticiones", "pieces": "tramos",
                                      "valor": ds.METRICS[metric][1], "ratio": "× ref.", "ratio_lo": "IC inf.",
                                      "ratio_hi": "IC sup."}),
                     hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.2f") for c_ in
                                    [ds.METRICS[metric][1], "× ref.", "IC inf.", "IC sup."]})
    with st.expander("Compras reales registradas en RawTree (con suerte)"):
        guarded("Compras reales", lambda: realised_block(hours))


# ---------------------------------------------------------------- tab 3: diagnosis (realised outcomes)

def diagnosis_tab() -> None:
    lead("Lo que pasó de verdad (clics y compras registrados en RawTree), solo en peticiones que vio **nuestra** "
         "respuesta. Cuando se comparan muchos grupos a la vez, los p-valores se corrigen (Holm): si no, uno de "
         "cada veinte parecería distinto por puro azar.")
    c = st.columns([1, 1.6, 1])
    hours = c[0].select_slider("Ventana ", [1, 3, 6, 12, 24], value=3, format_func=lambda h: f"{h} h")
    try:
        names = [v for v in version_raw(hours).model_version.astype(str) if v != ds.NO_ANSWER]
    except Exception:
        names = []
    version = c[1].selectbox("Versión", names + ["Todas (mezcla versiones distintas)"],
                             help="Mirar una sola versión evita mezclar modelos distintos.")
    if c[2].button("🔄 Recalcular", width="stretch"):
        for fn in (version_raw, new_products, segments_raw, position_raw, trends_raw):
            fn.clear()

    st.markdown("#### ¿Qué compradores convierten menos?")
    dim_name = st.radio("Segmentar por", list(ds.SEGMENT_DIMENSIONS), horizontal=True, label_visibility="collapsed")

    def segments():
        raw = segments_raw(hours, version, tuple(new_products()))
        if raw.empty:
            st.info("Sin datos en esta ventana.")
            return
        col = ds.SEGMENT_DIMENSIONS[dim_name]
        table = ds.segment_table(ds.prepare_segments(raw), col)
        note(*ds.segment_conclusion(table, col, dim_name))
        show_chart(ci_chart(table, col, "compras_por_100", "compras reales / 100 peticiones"))
        if col == "cold_label":
            st.caption("Cold = comprador nuevo, o que acaba de mirar un producto añadido durante el evento "
                       "(la definición de los organizadores; puntúa aparte, 10 puntos).")

    guarded("Segmentos", segments)

    st.markdown("#### ¿Cuánto pesa cada posición del carrusel?")

    def positions():
        pt, total, outside = ds.position_table(position_raw(hours, version))
        if total == 0:
            st.info("Sin datos en esta ventana.")
            return
        note(*ds.position_conclusion(pt))
        a, b = st.columns(2)
        with a:
            st.markdown("**Clics por 100 peticiones**")
            show_chart(bar_chart(pt, "posición", "clics_por_100", "clics / 100"))
        with b:
            st.markdown("**Compras por 100 peticiones**")
            show_chart(bar_chart(pt, "posición", "compras_por_100", "compras / 100"))
        if outside.get("compras"):
            st.caption(f"{int(outside['compras'])} compras fueron de productos que no estaban en el carrusel.")

    guarded("Posiciones", positions)

    st.markdown("#### ¿Está cambiando lo que compra la tienda?")
    lead("Cuota de compras de toda la tienda en los últimos minutos frente a las horas anteriores.")
    t1, t2 = st.columns(2)
    recent_min = t1.select_slider("Ventana reciente", [10, 30, 60], value=30, format_func=lambda m: f"{m} min")
    level_name = t2.radio("Nivel", ["categoría", "subcategoría"], horizontal=True)

    def trends():
        lvl = "category" if level_name == "categoría" else "subcategory"
        tt = ds.trend_table(trends_raw(6, recent_min), catalog(), lvl)
        note(*ds.trend_conclusion(tt, recent_min, level_name))
        if tt.empty or tt.cambio_pp.isna().all():
            return
        top = tt.head(12).assign(significativo=lambda x: x.p_adj < 0.05)
        order = top.sort_values("cambio_pp", ascending=False)[lvl].tolist()
        tip = [alt.Tooltip(f"{lvl}:N", title=level_name),
               alt.Tooltip("cambio_pp:Q", format="+.2f", title="cambio (puntos de cuota)"),
               alt.Tooltip("cuota_reciente:Q", format=".1f", title="cuota reciente %"),
               alt.Tooltip("cuota_base:Q", format=".1f", title="cuota anterior %"),
               alt.Tooltip("p_adj:Q", format=".4f", title="p corregido")]
        chart = alt.Chart(top).mark_bar(size=16, cornerRadiusEnd=4).encode(
            y=alt.Y(f"{lvl}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("cambio_pp:Q", title="cambio de cuota (puntos porcentuales)"),
            color=alt.Color("dirección:N", scale=alt.Scale(domain=["sube", "baja"], range=[BLUE, RED]),
                            legend=alt.Legend(title=None)),
            opacity=alt.Opacity("significativo:N", scale=alt.Scale(domain=[True, False], range=[1, 0.35]),
                                legend=alt.Legend(title="más que el azar")),
            tooltip=tip).properties(height=max(120, 28 * len(top)))
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TEXT_2).encode(x="x:Q")
        show_chart(chart + zero)

    guarded("Tendencias", trends)

    if OFFLINE_CSV.exists():
        with st.expander("¿Coincide la evaluación offline con el directo?"):
            def offline():
                vt = ds.version_table(version_raw(24))
                m, rho = ds.offline_vs_online(vt, pd.read_csv(OFFLINE_CSV))
                note(*ds.offline_conclusion(m, rho))
                if not m.empty:
                    base = alt.Chart(m)
                    pts = base.mark_point(filled=True, size=110, color=BLUE, stroke=SURFACE, strokeWidth=2,
                                          opacity=1).encode(
                        x=alt.X("offline:Q", title="métrica offline", scale=alt.Scale(zero=False)),
                        y=alt.Y("compras_por_100:Q", title="compras reales / 100", scale=alt.Scale(zero=False)),
                        tooltip=["model_version:N", alt.Tooltip("offline:Q", format=".4f"),
                                 alt.Tooltip("compras_por_100:Q", format=".2f")])
                    labels = base.mark_text(align="left", dx=9, color=TEXT_2, fontSize=12).encode(
                        x="offline:Q", y="compras_por_100:Q", text="model_version:N")
                    show_chart((pts + labels).properties(height=280))
            guarded("Offline", offline)


with tab_live:
    live()
    shop_and_inspector()

with tab_versions:
    versions_tab()

with tab_diag:
    diagnosis_tab()
