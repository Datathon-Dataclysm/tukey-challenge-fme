"""Live dashboard: see what flows between the shop, our recommender and the feedback,
and turn it into decisions.

    streamlit run tools/dashboard.py

Two tabs:
- «En directo»: requests → our answer → what the shopper did (operations / reliability).
- «Decisiones»: one block per decision (which version to ship, where the model fails, how much the
  order matters, whether to follow live trends, whether offline evaluation can be trusted), each
  with a chart, its uncertainty and a one-line conclusion. Statistics live in tools/decision_stats.py.

Reads TEAM_ID and RAWTREE_API_KEY from .env (project root). Read-only: it never writes to RawTree.
Times are shown in Barcelona time (CET = UTC+1); the data itself is in UTC.
Optional: tools/offline_results.csv (model_version,<offline metric>) feeds block ⑤.
Theme: .streamlit/config.toml (colours, font) + the CSS block below (cards, decision boxes).
"""
from __future__ import annotations

import math
import os
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

import decision_stats as ds

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
CET = timedelta(hours=1)
OFFLINE_CSV = Path(__file__).resolve().parent / "offline_results.csv"

# Palette (validated categorical/diverging steps, light mode). Text never uses series colours.
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#9a9893"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#ecebe7", "#fcfcfb"
STATUS = {"good": ("#0ca30c", "✅", "Conclusión"), "warning": ("#fab219", "⚠️", "Atención"),
          "info": (BLUE, "ℹ️", "Pendiente")}

st.set_page_config(page_title=f"RawTree · {TEAM}", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------- style (CSS on top of the theme)
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  [data-testid="stMetric"] {{
      background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 12px 16px;
  }}
  [data-testid="stMetricLabel"] p {{
      color: {TEXT_2}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;
  }}
  .decision {{
      border: 1px solid {GRID}; border-left: 5px solid var(--accent); border-radius: 8px;
      background: #ffffff; padding: 10px 14px; margin: 4px 0 12px; color: {TEXT}; line-height: 1.5;
  }}
  .decision .tag {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: {TEXT_2}; }}
  .question {{ color: {TEXT_2}; margin: -6px 0 8px; font-size: .92rem; }}
  .summary li {{ margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)


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


def sql_str(s: str) -> str:
    return str(s).replace("\\", "").replace("'", "")


@st.cache_data(ttl=600, show_spinner=False)
def catalog() -> pd.DataFrame:
    """Current catalogue: shop.catalog with the latest *non-empty* value of each field from catalog_updates.

    argMaxIf skips partial updates (e.g. a price change without category) so they don't blank fields.
    """
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


# ---------------------------------------------------------------- queries: live tab

def live_requests(minutes: int, limit: int) -> pd.DataFrame:
    """Each recent request joined with our answer (submissions) and what happened (feedback)."""
    return q(f"""
        SELECT r.request_id AS request_id, r.ts AS ts, r.user_id AS user_id, r.is_new_user AS is_new,
               r.device AS device, r.region AS region, r.page AS page, r.views AS recent_views,
               s.items AS items, s.model_version AS model_version,
               f.request_id != '' AS has_feedback, f.served_by AS served_by, f.fallback_reason AS reason,
               f.latency_ms AS latency_ms, f.shown AS shown,
               f.clicked AS clicked, f.purchased AS purchased, f.revenue_eur AS revenue
        FROM (SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, user_id::String AS user_id,
                     is_new_user::Bool AS is_new_user, context.device::String AS device,
                     context.region::String AS region, context.page::String AS page,
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


def kpis(minutes: int) -> pd.Series:
    df = q(f"""
        SELECT count() AS requests,
               countIf(served_by = 'team') AS on_time,
               countIf(clicked != '') AS clicks,
               countIf(purchased != '') AS purchases,
               sum(revenue_eur::Float64) AS revenue,
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
               quantileIf(0.95)(latency_ms::Float64, served_by = 'team') AS p95_ms,
               countIf(fallback_reason = 'missing') AS missing,
               countIf(fallback_reason = 'late') AS late,
               countIf(fallback_reason = 'invalid') AS invalid
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND ts > now() - INTERVAL {minutes} MINUTE
        GROUP BY minute ORDER BY minute""")
    if not df.empty:
        df["minute"] = to_cet(df["minute"])
        df = df.set_index("minute")
    return df


def shop_top_products(minutes: int) -> pd.DataFrame:
    return q(f"""
        SELECT product_id::String AS product_id,
               countIf(event_type = 'view') AS views,
               countIf(event_type = 'add_to_cart') AS carts,
               countIf(event_type = 'purchase') AS purchases
        FROM shop.events WHERE ts > now() - INTERVAL {minutes} MINUTE
        GROUP BY product_id ORDER BY purchases DESC, views DESC LIMIT 15""")


def one_request(request_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rid = sql_str(request_id)
    req = q(f"SELECT * FROM shop.requests WHERE request_id = '{rid}'")
    sub = q(f"SELECT * FROM {TEAM}.submissions WHERE request_id = '{rid}'")
    fb = q(f"SELECT * FROM {TEAM}.feedback WHERE request_id = '{rid}'")
    return req, sub, fb


# ---------------------------------------------------------------- queries: decisions tab (cached 2 min)

def version_filter(version: str) -> str:
    """SQL clause restricting feedback rows to requests answered by one model_version."""
    if not version or version == "Todas":
        return ""
    return (f"AND request_id::String IN (SELECT request_id::String FROM {TEAM}.submissions "
            f"WHERE model_version::String = '{sql_str(version)}')")


@st.cache_data(ttl=120, show_spinner=False)
def version_raw(hours: int) -> pd.DataFrame:
    """Per version, counting outcomes ONLY where the shopper saw our answer (served_by = 'team').

    Late/invalid answers were replaced by the fallback, so their clicks and purchases belong to the
    best sellers, not to our model; they only count against reliability (servidas / requests).
    """
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
        GROUP BY model_version ORDER BY hasta DESC""")


@st.cache_data(ttl=120, show_spinner=False)
def fallback_raw(hours: int) -> pd.Series:
    """Requests that got the fallback (best sellers): a free baseline to compare against."""
    df = q(f"""
        SELECT count() AS servidas, countIf(clicked::String != '') AS clics,
               countIf(purchased::String != '') AS compras,
               sum(revenue_eur::Float64) AS revenue, stddevSamp(revenue_eur::Float64) AS revenue_sd
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND served_by::String != 'team' AND ts > now() - INTERVAL {hours} HOUR""")
    return df.iloc[0] if not df.empty else pd.Series(dtype=float)


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


# ---------------------------------------------------------------- helpers

def status(row) -> str:
    if not row.has_feedback:
        return "⏳ esperando" if row["items"] else "⏳ sin respuesta aún"
    if row.served_by == "team":
        return "✅ a tiempo"
    return f"❌ {row.reason or 'fallback'}"


def position(product: str, shown) -> str:
    if not product:
        return ""
    try:
        return f"#{list(shown).index(product) + 1}"
    except ValueError:
        return "?"


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


def decision(level: str, text: str) -> None:
    """Coloured box with icon + label (status colour never carries meaning alone)."""
    colour, icon, label = STATUS.get(level, STATUS["info"])
    st.markdown(f'<div class="decision" style="--accent:{colour}"><span class="tag">{icon} {label}</span><br>'
                f'{ds.md_to_html(text)}</div>', unsafe_allow_html=True)


def question(text: str) -> None:
    st.markdown(f'<div class="question">{text}</div>', unsafe_allow_html=True)


def styled(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Recessive axes and grid, transparent background, readable text colours."""
    return (chart.configure(background="transparent", font="Source Sans Pro, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domain=False, tickColor=GRID, labelColor=TEXT_2,
                            titleColor=TEXT_2, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(orient="top", labelColor=TEXT_2, titleColor=TEXT_2, labelFontSize=12))


def show_chart(chart) -> None:
    chart = styled(chart)
    try:  # newer Streamlit
        st.altair_chart(chart, width="stretch", theme=None)
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True, theme=None)


def ci_chart(df: pd.DataFrame, label: str, rate: str, x_title: str, n_col: str = "servidas",
             reference: bool = False) -> alt.Chart:
    """Dot = estimate, line = 95 % interval. Sorted best → worst."""
    order = df.sort_values(rate, ascending=False)[label].astype(str).tolist()
    y = alt.Y(f"{label}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    if reference:
        color = alt.Color("tipo:N", scale=alt.Scale(domain=["versión", "referencia"], range=[BLUE, GRAY]),
                          legend=alt.Legend(title=None))
    else:
        color = alt.value(BLUE)
    tip = [alt.Tooltip(f"{label}:N", title="grupo"), alt.Tooltip(f"{rate}:Q", format=".2f", title="estimación"),
           alt.Tooltip(f"{rate}_lo:Q", format=".2f", title="IC 95 % inf."),
           alt.Tooltip(f"{rate}_hi:Q", format=".2f", title="IC 95 % sup."),
           alt.Tooltip(f"{n_col}:Q", format=",.0f", title="peticiones")]
    base = alt.Chart(df.dropna(subset=[rate]))
    rule = base.mark_rule(strokeWidth=2.5).encode(
        y=y, x=alt.X(f"{rate}_lo:Q", title=x_title, scale=alt.Scale(zero=True)), x2=f"{rate}_hi:Q",
        color=color, tooltip=tip)
    dot = base.mark_point(filled=True, size=110, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        y=y, x=f"{rate}:Q", color=color, tooltip=tip)
    return (rule + dot).properties(height=max(110, 38 * len(order)))


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = BLUE) -> alt.Chart:
    tip = [alt.Tooltip(f"{x}:O", title="posición"), alt.Tooltip(f"{y}:Q", format=".2f", title=title),
           alt.Tooltip(f"{y}_lo:Q", format=".2f", title="IC 95 % inf."),
           alt.Tooltip(f"{y}_hi:Q", format=".2f", title="IC 95 % sup.")]
    bars = alt.Chart(df).mark_bar(color=color, size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{x}:O", title="posición en el carrusel", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=title), tooltip=tip)
    err = alt.Chart(df).mark_rule(color=TEXT_2, strokeWidth=1.5).encode(
        x=f"{x}:O", y=f"{y}_lo:Q", y2=f"{y}_hi:Q")
    return (bars + err).properties(height=240)


# ---------------------------------------------------------------- page

st.title(f"🛒 Panel · {TEAM}")
st.caption("Petición (shop.requests) → nuestra respuesta (submissions) → lo que hizo el comprador (feedback). "
           "Horas en horario de Barcelona.")

with st.sidebar:
    st.header("En directo")
    auto = st.toggle("Refresco automático", value=True)
    every = st.slider("Cada cuántos segundos", 3, 60, 5)
    kpi_minutes = st.select_slider("Ventana de los indicadores (min)", [5, 15, 30, 60, 180], value=5)
    n_rows = st.slider("Peticiones en la tabla", 10, 100, 30)
    st.divider()
    st.markdown("[Leaderboard ↗](https://datathon.rawtree.dev/leaderboard)")

tab_live, tab_dec = st.tabs(["📡 En directo", "🧭 Decisiones"])


@st.fragment(run_every=every if auto else None)
def live() -> None:
    try:
        _live()
    except Exception as e:  # keep the page alive if RawTree hiccups (also on fragment reruns)
        st.error(f"Error consultando RawTree: {e}")


def _live() -> None:
    cat = catalog()

    # --- headline numbers -------------------------------------------------
    k = kpis(kpi_minutes)
    n = int(k.requests or 0)
    on_time = 100 * k.on_time / n if n else 0.0
    light = "🟢 VERDE" if on_time > 90 else "🔴 ROJA"
    st.subheader(f"Últimos {kpi_minutes} min · luz {light}")
    c = st.columns(6)
    c[0].metric("Peticiones con feedback", f"{n:,}")
    c[1].metric("A tiempo y válidas", f"{on_time:.1f} %", help="Luz verde si > 90 % en los últimos 5 min")
    c[2].metric("Latencia p50 / p95", f"{(k.p50 or 0) / 1000:.2f} / {(k.p95 or 0) / 1000:.2f} s",
                help="Desde el ts de la petición hasta que la respuesta está en RawTree. Límite: 3 s")
    c[3].metric("Clics por 100", f"{100 * k.clicks / n:.2f}" if n else "—")
    c[4].metric("Compras por 100", f"{100 * k.purchases / n:.2f}" if n else "—",
                help="Incluye las peticiones servidas por el fallback. Con pocas peticiones es muy ruidoso")
    c[5].metric("€ por petición", f"{k.revenue / n:.3f}" if n else "—")

    # --- live flow table ---------------------------------------------------
    st.subheader("Peticiones en directo")
    df = live_requests(minutes=5, limit=n_rows)
    if df.empty:
        st.info("No hay peticiones en los últimos 5 minutos.")
    else:
        df["hora"] = to_cet(df["ts"]).dt.strftime("%H:%M:%S.%f").str[:-3]
        df["estado"] = df.apply(status, axis=1)
        df["vistas"] = df.recent_views.apply(len)
        df["clic"] = df.apply(lambda r: position(r.clicked, r.shown if len(r.shown) else r["items"]), axis=1)
        df["compra"] = df.apply(lambda r: position(r.purchased, r.shown if len(r.shown) else r["items"]), axis=1)
        df["latencia s"] = (df.latency_ms / 1000).where(df.has_feedback & (df.served_by == "team"))
        df["€"] = df.revenue.where(df.revenue > 0)
        df["nuevo"] = df.is_new
        st.dataframe(
            df[["hora", "request_id", "user_id", "nuevo", "device", "region", "page", "vistas",
                "model_version", "estado", "latencia s", "clic", "compra", "€"]],
            hide_index=True, width="stretch",
            column_config={"latencia s": st.column_config.NumberColumn(format="%.2f"),
                           "€": st.column_config.NumberColumn(format="%.2f"),
                           "vistas": st.column_config.NumberColumn(help="Productos en recent_views")},
        )
        st.caption("clic / compra = posición en el carrusel del producto clicado o comprado. "
                   "El feedback llega unos segundos después de la petición.")

    # --- reliability over time --------------------------------------------
    st.subheader("Fiabilidad, última hora (por minuto)")
    pm = per_minute(60)
    if not pm.empty:
        a, b = st.columns(2)
        with a:
            st.markdown("**% a tiempo y válidas**")
            st.line_chart(pm["on_time_pct"], height=220, y_label="%", x_label="", color=BLUE)
        with b:
            st.markdown("**Latencia p95 (ms)** · límite 3000")
            st.line_chart(pm["p95_ms"], height=220, y_label="ms", x_label="", color=BLUE)
        fails = pm[["missing", "late", "invalid"]].sum()
        st.caption(f"Fallbacks en la última hora: missing {int(fails.missing)} · late {int(fails.late)} · "
                   f"invalid {int(fails.invalid)}")

    # --- shop ---------------------------------------------------------------
    st.subheader("Tienda: lo más comprado (60 min)")
    top = shop_top_products(60)
    if not top.empty:
        top = top.join(cat[["category", "subcategory", "price_eur"]], on="product_id")
    st.dataframe(top, hide_index=True, width="stretch")
    st.caption("shop.events: actividad orgánica de toda la tienda, no solo nuestro carrusel. "
               "La comparación de versiones está en la pestaña «Decisiones».")


def inspector() -> None:
    """Inspect one request (outside the auto-refresh so it doesn't jump)."""
    st.divider()
    st.subheader("🔍 Inspeccionar una petición")
    c1, c2 = st.columns([4, 1])
    if c2.button("Última con compra", width="stretch"):
        last = q(f"SELECT request_id FROM {TEAM}.feedback WHERE purchased != '' ORDER BY ts DESC LIMIT 1")
        st.session_state["inspect"] = "" if last.empty else last.request_id[0]
    rid = c1.text_input("request_id (cópialo de la tabla)", key="inspect", label_visibility="collapsed",
                        placeholder="r_... — copia un request_id de la tabla o pulsa «Última con compra»")
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
        st.markdown(
            f"**{r.request_id}** · {to_cet(pd.Series([r.ts])).iloc[0]:%H:%M:%S} · usuario `{r.user_id}`"
            f"{' (nuevo)' if r.is_new_user else ''} · {r.get('context.device')} · "
            f"{r.get('context.region')} · página *{r.get('context.page')}*")
        f = fb.iloc[0] if not fb.empty else None
        clicked = f.clicked if f is not None else ""
        purchased = f.purchased if f is not None else ""
        a, b = st.columns(2)
        with a:
            st.markdown("**① Lo que había mirado** (`recent_views`, el último abajo)")
            st.dataframe(describe(views, cat), hide_index=True, width="stretch")
        with b:
            shown = list(f.shown) if f is not None else (list(sub.iloc[0]["items"]) if not sub.empty else [])
            who = "" if f is None else (" (lo que vio: nuestra respuesta)" if f.served_by == "team"
                                        else f" (lo que vio: FALLBACK por {f.fallback_reason})")
            st.markdown(f"**② Lo que le mostramos**{who}")
            st.dataframe(describe(shown, cat, clicked, purchased), hide_index=True, width="stretch")
        if f is not None:
            st.markdown(f"**③ Resultado:** latencia {f.latency_ms} ms · "
                        f"clic `{clicked or '—'}` · compra `{purchased or '—'}` · {f.revenue_eur} €")
        else:
            st.info("Todavía no hay feedback para esta petición (llega unos segundos después).")
        with st.expander("JSON en bruto de las tres tablas"):
            st.json({"shop.requests": req.to_dict("records"),
                     f"{TEAM}.submissions": sub.to_dict("records"),
                     f"{TEAM}.feedback": fb.to_dict("records")})
    except Exception as e:
        st.error(f"Error: {e}")


def guarded(title: str, fn) -> tuple[str, str] | None:
    """Run one decision block; a failing query only breaks its own block."""
    try:
        return fn()
    except Exception as e:
        st.error(f"{title}: error consultando RawTree — {e}")
        return None


def decisions() -> None:
    st.markdown("Cada bloque responde a **una decisión** del reto con su gráfico, su incertidumbre "
                "(IC 95 %) y una conclusión. Solo se cuentan las peticiones que vio **nuestro** modelo; "
                "las que acabaron en fallback cuentan como fiabilidad, no como acierto del modelo.")
    c = st.columns([1, 1, 1, 1])
    hours = c[0].select_slider("Ventana de análisis (h)", [1, 3, 6, 12, 24], value=3)
    min_n = c[1].select_slider("Mínimo de peticiones para decidir", [100, 300, 500, 1000, 2000], value=300)
    try:
        versions = [v for v in version_raw(hours).model_version.astype(str) if v != ds.NO_ANSWER]
    except Exception:
        versions = []
    version = c[2].selectbox("Versión para ②–③", ["Todas"] + versions,
                             help="Analizar segmentos y posiciones de una sola versión evita mezclar modelos.")
    if c[3].button("🔄 Recalcular", width="stretch"):
        for fn in (version_raw, fallback_raw, new_products, segments_raw, position_raw, trends_raw):
            fn.clear()

    summary = st.container()
    conclusions: list[tuple[str, str, str]] = []

    # ① versions ----------------------------------------------------------------
    st.subheader("① ¿Qué versión llevamos a la final?")
    question("Compras por cada 100 peticiones servidas por nosotros (lo que puntúa), con su intervalo de confianza.")

    def block_versions():
        vt = ds.version_table(version_raw(hours), fallback_raw(hours))
        if vt.empty:
            st.info("Sin feedback en esta ventana.")
            return None
        level, text = ds.recommend(vt, min_n)
        decision(level, text)
        plot = vt[(vt.model_version != ds.NO_ANSWER) & (vt.servidas > 0)]
        a, b = st.columns(2)
        with a:
            st.markdown("**Compras por 100 peticiones**")
            show_chart(ci_chart(plot, "model_version", "compras_por_100", "compras / 100", reference=True))
        with b:
            st.markdown("**€ por petición** (desempate final)")
            eur = plot.rename(columns={"eur_lo": "eur_por_peticion_lo", "eur_hi": "eur_por_peticion_hi"})
            show_chart(ci_chart(eur, "model_version", "eur_por_peticion", "€ / petición", reference=True))
        table = vt.assign(desde=to_cet(vt.desde).dt.strftime("%H:%M"), hasta=to_cet(vt.hasta).dt.strftime("%H:%M"))
        st.dataframe(table[["model_version", "requests", "servidas", "a_tiempo_pct", "clics_por_100",
                            "compras_por_100", "compras_por_100_lo", "compras_por_100_hi",
                            "eur_por_peticion", "desde", "hasta"]],
                     hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.2f") for c_ in
                                    ["a_tiempo_pct", "clics_por_100", "compras_por_100", "compras_por_100_lo",
                                     "compras_por_100_hi", "eur_por_peticion"]})
        st.caption("Cada versión se ha ejecutado en una franja distinta y la tienda cambia con la hora, así que la "
                   "comparación no es perfectamente justa. Para una prueba limpia, servid dos versiones a la vez "
                   "repartiendo por hash de `request_id` (A/B). El fallback sirve de referencia: son los best sellers.")
        return level, text

    res = guarded("①", block_versions)
    if res:
        conclusions.append(("①", *res))

    # ② segments ----------------------------------------------------------------
    st.subheader("② ¿Dónde falla el modelo?")
    question("Conversión por segmento de comprador. El cold start (usuarios nuevos o que miran productos "
             "añadidos durante el evento) puntúa aparte.")
    dim_name = st.radio("Segmentar por", list(ds.SEGMENT_DIMENSIONS), horizontal=True, label_visibility="collapsed")

    def block_segments():
        raw = segments_raw(hours, version, tuple(new_products()))
        if raw.empty:
            st.info("Sin datos en esta ventana.")
            return None
        col = ds.SEGMENT_DIMENSIONS[dim_name]
        table = ds.segment_table(ds.prepare_segments(raw), col)
        level, text = ds.segment_conclusion(table, col, dim_name.split(" (")[0], min_n=max(100, min_n // 3))
        decision(level, text)
        show_chart(ci_chart(table, col, "compras_por_100", "compras / 100"))
        return level, text

    res = guarded("②", block_segments)
    if res:
        conclusions.append(("②", *res))

    # ③ position ----------------------------------------------------------------
    st.subheader("③ ¿Cuánto importa el orden de los 10 productos?")
    question("Clics y compras según la posición en el carrusel (por cada 100 peticiones servidas).")

    def block_position():
        pt, total, outside = ds.position_table(position_raw(hours, version))
        if total == 0:
            st.info("Sin datos en esta ventana.")
            return None
        level, text = ds.position_conclusion(pt)
        decision(level, text)
        a, b = st.columns(2)
        with a:
            st.markdown("**Clics por 100 peticiones**")
            show_chart(bar_chart(pt, "posición", "clics_por_100", "clics / 100"))
        with b:
            st.markdown("**Compras por 100 peticiones**")
            show_chart(bar_chart(pt, "posición", "compras_por_100", "compras / 100"))
        if outside.get("compras"):
            st.caption(f"{int(outside['compras'])} compras fueron de productos fuera del carrusel (no se muestran).")
        return level, text

    res = guarded("③", block_position)
    if res:
        conclusions.append(("③", *res))

    # ④ trends ------------------------------------------------------------------
    st.subheader("④ ¿Hay tendencias en directo que el modelo deba seguir?")
    question("Cuota de compras en toda la tienda: últimos minutos frente a las horas anteriores (cambio en puntos).")
    t1, t2 = st.columns(2)
    recent_min = t1.select_slider("Ventana reciente (min)", [10, 30, 60], value=30)
    level_name = t2.radio("Nivel", ["categoría", "subcategoría"], horizontal=True)

    def block_trends():
        lvl = "category" if level_name == "categoría" else "subcategory"
        tt = ds.trend_table(trends_raw(6, recent_min), catalog(), lvl)
        level, text = ds.trend_conclusion(tt, recent_min, level_name)
        decision(level, text)
        if tt.empty or tt.cambio_pp.isna().all():
            return level, text
        top = tt.head(12)
        order = top.sort_values("cambio_pp", ascending=False)[lvl].tolist()
        tip = [alt.Tooltip(f"{lvl}:N", title=level_name), alt.Tooltip("cambio_pp:Q", format="+.2f", title="cambio (pp)"),
               alt.Tooltip("cuota_reciente:Q", format=".1f", title="cuota reciente %"),
               alt.Tooltip("cuota_base:Q", format=".1f", title="cuota anterior %"),
               alt.Tooltip("p:Q", format=".4f", title="p-valor")]
        chart = alt.Chart(top).mark_bar(size=16, cornerRadiusEnd=4).encode(
            y=alt.Y(f"{lvl}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("cambio_pp:Q", title="cambio de cuota (puntos porcentuales)"),
            color=alt.Color("dirección:N", scale=alt.Scale(domain=["sube", "baja"], range=[BLUE, RED]),
                            legend=alt.Legend(title=None)),
            tooltip=tip).properties(height=max(120, 28 * len(top)))
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TEXT_2).encode(x="x:Q")
        show_chart(chart + zero)
        return level, text

    res = guarded("④", block_trends)
    if res:
        conclusions.append(("④", *res))

    # ⑤ offline vs online -------------------------------------------------------
    st.subheader("⑤ ¿Nos podemos fiar de la evaluación offline?")
    question("Si la métrica offline ordena las versiones igual que el stream, sirve para descartar ideas sin desplegarlas.")

    def block_offline():
        if not OFFLINE_CSV.exists():
            decision("info", f"Crea `tools/{OFFLINE_CSV.name}` con dos columnas: `model_version` y vuestra métrica "
                             "offline (p. ej. `hit_at_10`), una fila por versión desplegada.")
            return None
        vt = ds.version_table(version_raw(24))
        m, rho = ds.offline_vs_online(vt, pd.read_csv(OFFLINE_CSV))
        level, text = ds.offline_conclusion(m, rho)
        decision(level, text)
        if not m.empty:
            base = alt.Chart(m)
            tip = ["model_version:N", alt.Tooltip("offline:Q", format=".4f"),
                   alt.Tooltip("compras_por_100:Q", format=".2f"), alt.Tooltip("servidas:Q", format=",.0f")]
            pts = base.mark_point(filled=True, size=110, color=BLUE, stroke=SURFACE, strokeWidth=2, opacity=1).encode(
                x=alt.X("offline:Q", title="métrica offline", scale=alt.Scale(zero=False)),
                y=alt.Y("compras_por_100:Q", title="compras / 100 (stream)", scale=alt.Scale(zero=False)),
                tooltip=tip)
            labels = base.mark_text(align="left", dx=9, color=TEXT_2, fontSize=12).encode(
                x="offline:Q", y="compras_por_100:Q", text="model_version:N")
            show_chart((pts + labels).properties(height=280))
        return level, text

    res = guarded("⑤", block_offline)
    if res:
        conclusions.append(("⑤", *res))

    # summary on top ------------------------------------------------------------
    with summary:
        if conclusions:
            icons = {k: v[1] for k, v in STATUS.items()}
            items = "".join(f"<li>{icons.get(level, 'ℹ️')} <b>{n}</b> {ds.md_to_html(text.split('→')[-1].strip())}</li>"
                            for n, level, text in conclusions)
            st.markdown(f'<div class="decision summary" style="--accent:{TEXT_2}"><span class="tag">Resumen de '
                        f'decisiones · últimas {hours} h</span><ul style="margin:6px 0 0">{items}</ul></div>',
                        unsafe_allow_html=True)


with tab_live:
    live()
    inspector()

with tab_dec:
    decisions()