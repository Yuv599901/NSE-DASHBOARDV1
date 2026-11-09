"""
NSE Market Dashboard  (Streamlit + nselib)
Modules: Sector | Top Gainers | Top Losers | FII/DII Activity | Bulk Deals | Block Deals

Run:  pip install streamlit nselib pandas plotly
      streamlit run nse_dashboard.py
"""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nselib import capital_market as cm
from nselib import derivatives as dv

import nse_report as rpt
import nse_ta as ta
from plotly.subplots import make_subplots

# fii_dii_trading_activity is not re-exported in capital_market/__init__ (nselib 2.5.x)
try:
    from nselib.capital_market.capital_market_data import fii_dii_trading_activity
except Exception:  # pragma: no cover
    fii_dii_trading_activity = getattr(cm, "fii_dii_trading_activity", None)

st.set_page_config(page_title="NSE Dashboard", page_icon="📈", layout="wide")

# ----------------------------------------------------------------- helpers

def fetch_indices():
    """All-index snapshot. capital_market.market_watch_all_indices is preferred because some
    nselib builds ship a broken nselib.indices (ModuleNotFoundError: config)."""
    try:
        return cm.market_watch_all_indices()
    except Exception as first:
        try:
            from nselib.indices import live_index_performances  # may fail on broken builds
            return live_index_performances()
        except Exception:
            raise first

SECTOR_INDICES_KEY = "SECTORAL INDICES"


def to_num(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Coerce numeric-looking columns (handles '1,234.5', '-', None)."""
    df = df.copy()
    for c in cols or df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            cleaned = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
            conv = pd.to_numeric(cleaned, errors="coerce")
            # only replace if most of the column is numeric
            if conv.notna().mean() > 0.8:
                df[c] = conv
    return df


def coerce(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Force columns to numbers ('1,234.5', ' -56 ', '-' -> float/NaN)."""
    df = df.copy()
    for c in cols:
        if c:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                                  errors="coerce")
    return df


def find_col(df: pd.DataFrame, *candidates):
    """Case/space-insensitive column lookup; returns actual column name or None."""
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    for cand in candidates:  # partial match fallback
        k = cand.lower().replace(" ", "").replace("_", "")
        for nk, orig in norm.items():
            if k in nk:
                return orig
    return None


def colour_pct(v):
    if pd.isna(v):
        return ""
    return "color:#16a34a" if v > 0 else "color:#dc2626" if v < 0 else ""


def safe_call(fn, *args, **kwargs):
    """Run an nselib call and return (df, error)."""
    try:
        df = fn(*args, **kwargs)
        if df is None or len(df) == 0:
            return None, "No data returned for this selection."
        return to_num(df), None
    except Exception as e:  # NSE blocks / holidays / bad dates
        return None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------- cached loaders
@st.cache_data(ttl=300, show_spinner=False)
def load_indices():
    return safe_call(fetch_indices)


@st.cache_data(ttl=300, show_spinner=False)
def load_movers(kind: str):
    return safe_call(cm.top_gainers_or_losers, kind)  # 'gainers' | 'loosers' (sic)


@st.cache_data(ttl=900, show_spinner=False)
def load_fii_dii():
    if fii_dii_trading_activity is None:
        return None, "fii_dii_trading_activity not available in this nselib version."
    return safe_call(fii_dii_trading_activity)


@st.cache_data(ttl=3600, show_spinner=False)
def load_participant_volume(d: str):
    return safe_call(dv.participant_wise_trading_volume, d)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fii_deriv(d: str):
    return safe_call(dv.fii_derivatives_statistics, d)


@st.cache_data(ttl=900, show_spinner=False)
def load_deals(kind: str, period: str | None = None, from_date: str | None = None,
               to_date: str | None = None):
    """nselib takes EITHER period ('1D','1W','1M','6M','1Y') OR from_date/to_date ('dd-mm-YYYY')."""
    fn = cm.bulk_deal_data if kind == "bulk" else cm.block_deals_data
    if from_date and to_date:
        return safe_call(fn, from_date=from_date, to_date=to_date)
    return safe_call(fn, period=period)


@st.cache_data(ttl=1800, show_spinner=False)
def load_delivery(d_str: str):
    """Latest trading day on/before d_str plus previous session. Returns (date, bhav, prev_bhav, err)."""
    d0 = pd.to_datetime(d_str, format="%d-%m-%Y").date()
    day, bhav = rpt.latest_trading_day(d0)
    if day is None:
        return None, None, None, "No bhav copy found in the last 7 days (NSE blocked or not yet published)."
    _, prev = rpt.previous_trading_day(day)
    return day, bhav, prev, None


@st.cache_data(ttl=1800, show_spinner=False)
def load_symbol_delivery(symbol: str, period: str):
    return safe_call(cm.deliverable_position_data, symbol, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlc(symbol: str, years: int):
    """Daily OHLCV from nselib.price_volume_data (nselib chunks >365d requests itself)."""
    try:
        end = date.today()
        start = end - timedelta(days=int(365.25 * years))
        raw = cm.price_volume_data(symbol.strip().upper(), start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y"))
        if raw is None or len(raw) == 0:
            return None, f"No price data for {symbol}."
        df = ta.normalize_ohlc(raw)
        if len(df) < 60:
            return None, f"Only {len(df)} bars for {symbol} - need a longer history."
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📈 NSE Dashboard")
page = st.sidebar.radio(
    "Module",
    ["Sector", "Top Gainers", "Top Losers", "FII / DII Activity", "Delivery Position",
     "Stock Chart (MTF)", "MTF Scanner", "Bulk Deals", "Block Deals"],
)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.subheader("📄 Post-market report")
if st.sidebar.button("Generate PDF"):
    with st.spinner("Collecting data & building PDF (~30-60s)…"):
        try:
            payload = rpt.collect()
            st.session_state["pdf"] = (rpt.build_pdf(payload), payload["date"])
        except Exception as e:
            st.sidebar.error(f"{type(e).__name__}: {e}")
if "pdf" in st.session_state:
    _b, _d = st.session_state["pdf"]
    st.sidebar.download_button(f"⬇️ NSE_PostMarket_{_d:%Y-%m-%d}.pdf", _b,
                               f"NSE_PostMarket_{_d:%Y-%m-%d}.pdf", "application/pdf")
st.sidebar.caption("Source: NSE via `nselib`. Live data is delayed; after market hours shows last close.")


# ======================================================================
# SECTOR
# ======================================================================
def page_sector():
    st.header("Sector Performance")
    df, err = load_indices()
    if err:
        st.error(err)
        return

    key_col = find_col(df, "key")
    name_col = find_col(df, "index", "indexSymbol")
    pct_col = find_col(df, "percentChange")
    last_col = find_col(df, "last")

    view = df
    if key_col and (df[key_col].astype(str).str.upper() == SECTOR_INDICES_KEY).any():
        scope = st.radio("Universe", ["Sectoral", "All indices"], horizontal=True)
        if scope == "Sectoral":
            view = df[df[key_col].astype(str).str.upper() == SECTOR_INDICES_KEY]

    view = view.sort_values(pct_col, ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    best, worst = view.iloc[0], view.iloc[-1]
    c1.metric("Best", best[name_col], f"{best[pct_col]:.2f}%")
    c2.metric("Worst", worst[name_col], f"{worst[pct_col]:.2f}%")
    c3.metric("Advancing sectors", int((view[pct_col] > 0).sum()))
    c4.metric("Declining sectors", int((view[pct_col] < 0).sum()))

    fig = px.bar(
        view, x=pct_col, y=name_col, orientation="h", color=pct_col,
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        labels={pct_col: "% change (1D)", name_col: ""},
    )
    fig.update_layout(height=max(380, 26 * len(view)), yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    # multi-timeframe heat table
    tf_cols = [c for c in [find_col(view, "percentChange"), find_col(view, "perChange30d"),
                           find_col(view, "perChange365d")] if c]
    show = [name_col, last_col] + tf_cols + [c for c in [find_col(view, "pe"), find_col(view, "pb"),
                                                         find_col(view, "dy"),
                                                         find_col(view, "advances"),
                                                         find_col(view, "declines")] if c]
    st.subheader("Multi-timeframe table")
    st.dataframe(
        view[show].style.map(colour_pct, subset=tf_cols).format(precision=2),
        width="stretch", hide_index=True,
    )


# ======================================================================
# GAINERS / LOSERS
# ======================================================================
def page_movers(kind: str):
    title = "Top Gainers" if kind == "gainers" else "Top Losers"
    st.header(title)
    df, err = load_movers(kind)
    if err:
        st.error(err)
        return

    sym = find_col(df, "symbol")
    pct = find_col(df, "perChange", "pChange", "percentChange", "netPrice")
    ltp = find_col(df, "ltp", "lastPrice", "last")
    vol = find_col(df, "tradedQuantity", "tradeVolume", "totalTradedVolume", "volume")
    legend = find_col(df, "legend")

    if legend and df[legend].nunique() > 1:
        sel = st.multiselect("Index / segment", sorted(df[legend].astype(str).unique()),
                             default=sorted(df[legend].astype(str).unique())[:1])
        if sel:
            df = df[df[legend].astype(str).isin(sel)]

    n = st.slider("Rows", 5, min(100, max(len(df), 6)), min(20, len(df)))
    asc = kind != "gainers"
    if pct:
        df = df.sort_values(pct, ascending=asc)
    df = df.head(n)

    if sym and pct:
        fig = px.bar(df, x=sym, y=pct, color=pct, color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, labels={pct: "% change", sym: ""})
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(df.style.map(colour_pct, subset=[pct] if pct else []).format(precision=2),
                 width="stretch", hide_index=True)
    st.download_button("⬇️ CSV", df.to_csv(index=False), f"{kind}.csv", "text/csv")


# ======================================================================
# FII / DII
# ======================================================================
def page_fii():
    st.header("FII / DII Activity")
    tab1, tab2, tab3 = st.tabs(["Cash market (latest)", "Participant-wise volume", "FII derivatives"])

    with tab1:
        df, err = load_fii_dii()
        if err:
            st.error(err)
        else:
            cat = find_col(df, "category")
            buy = find_col(df, "buyValue")
            sell = find_col(df, "sellValue")
            net = find_col(df, "netValue")
            dcol = find_col(df, "date")
            df = coerce(df, [buy, sell, net])
            if dcol:
                st.caption(f"As of {df[dcol].iloc[0]}  •  values in ₹ crore")
            cols = st.columns(len(df))
            for col, (_, r) in zip(cols, df.iterrows()):
                nv = r[net]
                if pd.isna(nv):
                    col.metric(str(r[cat]), "n/a")
                else:
                    col.metric(str(r[cat]), f"₹{nv:,.0f} Cr", "Net buy" if nv > 0 else "Net sell",
                               delta_color="normal" if nv > 0 else "inverse")
            fig = go.Figure()
            fig.add_bar(x=df[cat], y=df[buy], name="Buy", marker_color="#16a34a")
            fig.add_bar(x=df[cat], y=df[sell], name="Sell", marker_color="#dc2626")
            fig.add_bar(x=df[cat], y=df[net], name="Net", marker_color="#2563eb")
            fig.update_layout(barmode="group", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, width="stretch", hide_index=True)

    with tab2:
        d = st.date_input("Trade date", date.today() - timedelta(days=1), key="pv_date")
        df, err = load_participant_volume(d.strftime("%d-%m-%Y"))
        if err:
            st.warning(err + "  (weekend/holiday or data not yet published?)")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    with tab3:
        d = st.date_input("Trade date", date.today() - timedelta(days=1), key="fd_date")
        df, err = load_fii_deriv(d.strftime("%d-%m-%Y"))
        if err:
            st.warning(err + "  (weekend/holiday or data not yet published?)")
        else:
            st.dataframe(df, width="stretch", hide_index=True)


# ======================================================================
# DELIVERY POSITION
# ======================================================================
def page_delivery():
    st.header("Delivery Position")
    asof = st.date_input("As of (falls back to last trading day)", date.today())
    day, bhav, prev, err = load_delivery(asof.strftime("%d-%m-%Y"))
    if err:
        st.error(err)
        return

    c1, c2, c3, c4 = st.columns(4)
    mt = c1.number_input("Min turnover (₹ Cr)", 0.0, 1000.0, 10.0, 5.0)
    md = c2.number_input("Min delivery %", 0.0, 100.0, 55.0, 5.0)
    mv = c3.number_input("Price move % (±)", 0.0, 20.0, 2.0, 0.5)
    sx = c4.number_input("Delivery spike (x prev)", 1.0, 20.0, 2.0, 0.5)
    res = rpt.delivery_analysis(bhav, prev, mt, md, mv, sx, top_n=25)
    s = res["stats"]
    st.caption(f"Session: **{day:%d %b %Y}**  •  EQ series  •  previous session "
               f"{'loaded' if prev is not None else 'unavailable (spike table empty)'}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Advances / Declines", f"{s['advances']:,} / {s['declines']:,}")
    m2.metric("EQ turnover", f"₹{s['turnover_cr']:,.0f} Cr")
    m3.metric("Delivery % (value-wtd)", f"{s['wtd_deliv_pct']:.1f}%")
    m4.metric("Avg delivery % (liquid)", f"{s['avg_deliv_pct']:.1f}%")
    m5.metric("Stocks", f"{s['stocks']:,}")

    labels = {"CLOSE_PRICE": "Close", "TurnoverCr": "Turnover ₹Cr", "DELIV_PER": "Deliv %",
              "DelivValCr": "Deliv ₹Cr", "DelivQtyX": "Deliv Qty ×Prev"}
    tabs = st.tabs(["Highest delivery %", "Accumulation", "Distribution", "Delivery spikes", "Scatter", "Stock history"])
    for tab, key in zip(tabs[:4], ["top_deliv_pct", "accumulation", "distribution", "spike"]):
        with tab:
            df = res[key].rename(columns=labels)
            if df.empty:
                st.info("No stocks matched the filters.")
            else:
                st.dataframe(df.style.map(colour_pct, subset=["Chg%"]).format(precision=2),
                             width="stretch", hide_index=True)
                st.download_button("⬇️ CSV", df.to_csv(index=False), f"delivery_{key}.csv", "text/csv", key=key)

    with tabs[4]:
        eq = res["all"]
        eq = eq[(eq["TurnoverCr"] >= mt)].dropna(subset=["DELIV_PER", "Chg%"])
        fig = px.scatter(eq, x="Chg%", y="DELIV_PER", size="TurnoverCr", hover_name="Symbol",
                         color="Chg%", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                         labels={"DELIV_PER": "Delivery %"}, size_max=28)
        fig.add_hline(y=md, line_dash="dot", line_color="#64748b")
        fig.update_layout(height=520, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Top-right = price up on heavy delivery (accumulation); top-left = price down on heavy delivery.")

    with tabs[5]:
        sc1, sc2 = st.columns([2, 1])
        symbol = sc1.selectbox("Symbol", sorted(res["all"]["Symbol"].unique()),
                               index=None, placeholder="Type to search…")
        period = sc2.radio("Period", ["1M", "6M", "1Y"], horizontal=True)
        if symbol:
            h, e = load_symbol_delivery(symbol, period)
            if e:
                st.warning(e)
            else:
                dc, pc = find_col(h, "DeliverablePercentage", "%DlyQttoTradedQty", "DlyQt"), find_col(h, "ClosePrice", "Close")
                dt = find_col(h, "Date")
                fig = go.Figure()
                if pc:
                    fig.add_scatter(x=h[dt], y=h[pc], name="Close", yaxis="y2", line=dict(color="#2563eb"))
                if dc:
                    fig.add_bar(x=h[dt], y=h[dc], name="Delivery %", marker_color="#94a3b8")
                fig.update_layout(height=420, yaxis=dict(title="Delivery %"),
                                  yaxis2=dict(title="Close", overlaying="y", side="right"),
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(h, width="stretch", hide_index=True)


# ======================================================================
# BULK / BLOCK
# ======================================================================
def page_deals(kind: str):
    st.header("Bulk Deals" if kind == "bulk" else "Block Deals")
    mode = st.radio("Range", ["1D", "1W", "1M", "6M", "1Y", "Custom"], horizontal=True, index=1)
    if mode == "Custom":
        c1, c2 = st.columns(2)
        start = c1.date_input("From", date.today() - timedelta(days=30))
        end = c2.date_input("To", date.today())
        if start > end:
            st.error("'From' must be on or before 'To'.")
            return
        df, err = load_deals(kind, from_date=start.strftime("%d-%m-%Y"), to_date=end.strftime("%d-%m-%Y"))
        period = f"{start:%d%b%y}-{end:%d%b%y}"
    else:
        period = mode
        df, err = load_deals(kind, period)
    if err:
        st.error(err)
        return

    sym = find_col(df, "Symbol")
    client = find_col(df, "ClientName", "Client")
    side = find_col(df, "Buy/Sell", "BuySell")
    qty = find_col(df, "QuantityTraded", "Quantity")
    price = find_col(df, "TradePrice", "Wght.Avg.TradePrice", "Price")

    df = coerce(df, [qty, price])
    f1, f2, f3 = st.columns(3)
    sym_sel = f1.multiselect("Symbol", sorted(df[sym].astype(str).unique())) if sym else []
    side_sel = f2.multiselect("Buy / Sell", sorted(df[side].astype(str).unique())) if side else []
    q = f3.text_input("Client contains")
    if sym_sel:
        df = df[df[sym].astype(str).isin(sym_sel)]
    if side_sel:
        df = df[df[side].astype(str).isin(side_sel)]
    if q and client:
        df = df[df[client].astype(str).str.contains(q, case=False, na=False)]

    if qty and price:
        df = df.assign(ValueCr=(df[qty] * df[price]) / 1e7)

    m1, m2, m3 = st.columns(3)
    m1.metric("Deals", f"{len(df):,}")
    if "ValueCr" in df:
        m2.metric("Total value", f"₹{df['ValueCr'].sum():,.0f} Cr")
    if sym:
        m3.metric("Symbols", df[sym].nunique())

    if sym and side and "ValueCr" in df:
        s = df[side].astype(str).str.upper().str.startswith("B")
        net = (df.assign(Signed=df["ValueCr"].where(s, -df["ValueCr"]))
                 .groupby(sym)["Signed"].sum().sort_values())
        top = pd.concat([net.head(10), net.tail(10)]).drop_duplicates()
        fig = px.bar(top, orientation="h", color=top.values, color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, labels={"value": "Net ₹ Cr", "index": ""})
        fig.update_layout(title="Net buy/sell by symbol (₹ Cr)", coloraxis_showscale=False,
                          showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇️ CSV", df.to_csv(index=False), f"{kind}_deals_{period}.csv", "text/csv")



# ======================================================================
# STOCK CHART - candlestick + Supertrend + RSI, multi-timeframe
# ======================================================================
TF_BARS = {"Daily": 200, "Weekly": 120, "Monthly": 60}


def is_forming(tf: str, last_day: pd.Timestamp) -> bool:
    """Is the latest weekly/monthly bar still incomplete?"""
    if tf == "Weekly":
        return last_day.weekday() < 4
    if tf == "Monthly":
        return (last_day + pd.offsets.BDay(1)).month == last_day.month
    return False


def make_chart(df: pd.DataFrame, tf: str, bars: int, st_label: str, rsi_len: int, show_flips: bool, symbol: str):
    v = df.tail(bars)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=v.index, open=v["open"], high=v["high"], low=v["low"], close=v["close"],
                                 name=symbol, increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
                                 increasing_fillcolor="#16a34a", decreasing_fillcolor="#dc2626"), row=1, col=1)
    up = v["st"].where(v["st_dir"] == 1)
    dn = v["st"].where(v["st_dir"] == -1)
    fig.add_trace(go.Scatter(x=v.index, y=up, mode="lines", name=f"{st_label} up", connectgaps=False,
                             line=dict(color="#16a34a", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=dn, mode="lines", name=f"{st_label} down", connectgaps=False,
                             line=dict(color="#dc2626", width=2)), row=1, col=1)
    if show_flips:
        buys, sells = v[v["flip"] == 1], v[v["flip"] == -1]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["low"] * 0.985, mode="markers", name="ST buy",
                                 marker=dict(symbol="triangle-up", size=11, color="#16a34a")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * 1.015, mode="markers", name="ST sell",
                                 marker=dict(symbol="triangle-down", size=11, color="#dc2626")), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v["rsi"], mode="lines", name=f"RSI {rsi_len}",
                             line=dict(color="#7c3aed", width=1.6)), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="#94a3b8", opacity=0.10, line_width=0, row=2, col=1)
    for lvl, dash in ((70, "dash"), (50, "dot"), (30, "dash")):
        fig.add_hline(y=lvl, line_dash=dash, line_color="#94a3b8", line_width=1, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    if tf == "Daily":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                      dict(values=ta.missing_bdays(v.index))])
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04), margin=dict(l=0, r=0, t=10, b=0))
    return fig


def page_stock_chart():
    st.header("Stock Chart - Supertrend + RSI (Daily / Weekly / Monthly)")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    symbol = c1.text_input("NSE symbol", "RELIANCE").strip().upper()
    stp = c2.number_input("Supertrend ATR period", 1, 50, 4)
    stm = c3.number_input("Supertrend multiplier", 0.1, 10.0, 1.0, 0.1)
    rsl = c4.number_input("RSI length", 2, 50, 14)
    years = c5.number_input("History (years)", 2, 15, 5)
    if not symbol:
        return

    daily, err = load_ohlc(symbol, int(years))
    if err:
        st.error(err)
        return
    frames = ta.build_mtf(daily, int(stp), float(stm), int(rsl))
    summ = {tf: ta.tf_summary(f) for tf, f in frames.items()}
    conf = ta.confluence(summ)
    st_label = f"ST({int(stp)},{stm:g})"

    last = daily.index[-1]
    prev_close = daily["close"].iloc[-2] if len(daily) > 1 else daily["close"].iloc[-1]
    st.caption(f"**{symbol}**  •  last bar {last:%d %b %Y}  •  close ₹{daily['close'].iloc[-1]:,.2f} "
               f"({(daily['close'].iloc[-1] / prev_close - 1) * 100:+.2f}%)  •  {len(daily):,} daily bars")

    banner = f"**MTF confluence:** {conf['label']}  ({conf['bull']} bullish / {conf['bear']} bearish of 3)"
    (st.success if conf["score"] >= 2 else st.error if conf["score"] <= -2 else st.warning)(banner)

    cols = st.columns(3)
    for col, tf in zip(cols, ["Daily", "Weekly", "Monthly"]):
        s = summ[tf]
        forming = is_forming(tf, last)
        if s["dir"] == 0:
            col.metric(tf, "n/a", "not enough bars")
            continue
        col.metric(f"{tf}{' (forming bar)' if forming else ''}", f"{'🟢' if s['dir'] == 1 else '🔴'} {s['trend']}",
                   f"{s['dist_pct']:+.2f}% vs {st_label} {s['st']:,.2f}")
        col.caption(f"RSI {int(rsl)}: **{s['rsi']:.1f}** ({ta.rsi_zone(s['rsi'])})  •  flipped "
                    f"{s['flip_date']:%d %b %Y} ({s['bars_since_flip']} bars ago)")

    show_flips = st.checkbox("Show ST flip markers", True)
    bars = {tf: min(TF_BARS[tf], len(frames[tf])) for tf in TF_BARS}
    with st.expander("Bars shown per timeframe"):
        for tf in TF_BARS:
            mx = len(frames[tf])
            bars[tf] = st.slider(tf, min(20, mx), mx, min(TF_BARS[tf], mx), key=f"bars_{tf}")

    tabs = st.tabs(["Daily", "Weekly", "Monthly", "Data"])
    for tab, tf in zip(tabs[:3], ["Daily", "Weekly", "Monthly"]):
        with tab:
            st.plotly_chart(make_chart(frames[tf], tf, bars[tf], st_label, int(rsl), show_flips, symbol),
                            width="stretch")
    with tabs[3]:
        tf = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], horizontal=True)
        out = frames[tf][["open", "high", "low", "close", "volume", "st", "st_dir", "flip", "rsi"]].tail(250)
        st.dataframe(out.iloc[::-1].style.format(precision=2), width="stretch")
        st.download_button("⬇️ CSV", frames[tf].to_csv(), f"{symbol}_{tf.lower()}_st_rsi.csv", "text/csv")


# ======================================================================
# MTF SCANNER
# ======================================================================
UNIVERSES = {
    "Nifty 50": "nifty50_equity_list",
    "Nifty Next 50": "niftynext50_equity_list",
    "Nifty Midcap 150": "niftymidcap150_equity_list",
    "Nifty Smallcap 250": "niftysmallcap250_equity_list",
}


def scan_row(symbol: str, stp: int, stm: float, rsl: int, years: int):
    daily, err = load_ohlc(symbol, years)
    if err:
        return None, err
    frames = ta.build_mtf(daily, stp, stm, rsl)
    s = {tf: ta.tf_summary(f) for tf, f in frames.items()}
    conf = ta.confluence(s)
    row = {"Symbol": symbol, "Close": daily["close"].iloc[-1], "Score": conf["score"], "Setup": conf["label"]}
    for tf, k in (("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")):
        row[f"{k} trend"] = ("🟢 Bull" if s[tf]["dir"] == 1 else "🔴 Bear") if s[tf]["dir"] else "-"
        row[f"{k} RSI"] = s[tf].get("rsi")
        row[f"{k} flip bars"] = s[tf].get("bars_since_flip")
    return row, None


def page_scanner():
    st.header("MTF Supertrend + RSI Scanner")
    st.caption("Scores each stock +1 per bullish timeframe / -1 per bearish (Daily, Weekly, Monthly): +3 = fully aligned bullish.")
    c1, c2, c3, c4 = st.columns(4)
    stp = c1.number_input("ST ATR period", 1, 50, 4, key="s_stp")
    stm = c2.number_input("ST multiplier", 0.1, 10.0, 1.0, 0.1, key="s_stm")
    rsl = c3.number_input("RSI length", 2, 50, 14, key="s_rsl")
    years = c4.number_input("History (years)", 2, 10, 4, key="s_yrs")

    src = st.radio("Universe", ["Custom list"] + list(UNIVERSES), horizontal=True)
    if src == "Custom list":
        txt = st.text_area("Symbols (comma / space / newline separated)",
                           "RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, LT, ITC")
        symbols = [x.strip().upper() for x in txt.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
    else:
        lst, err = safe_call(getattr(cm, UNIVERSES[src]))
        if err:
            st.error(err)
            return
        symbols = [str(x).strip().upper() for x in lst[find_col(lst, "Symbol")]]
    symbols = list(dict.fromkeys(symbols))
    cap = st.number_input("Max symbols to scan", 1, 500, min(len(symbols), 50))
    symbols = symbols[: int(cap)]
    st.caption(f"{len(symbols)} symbols  •  ~1-2 s each on first run (results are cached for 1 hour)")

    if st.button("▶ Run scan", type="primary"):
        rows, fails = [], []
        bar = st.progress(0.0, text="Scanning…")
        for i, sym in enumerate(symbols, 1):
            row, err = scan_row(sym, int(stp), float(stm), int(rsl), int(years))
            (rows.append(row) if row else fails.append(f"{sym}: {err}"))
            bar.progress(i / len(symbols), text=f"{i}/{len(symbols)}  {sym}")
        bar.empty()
        st.session_state["mtf_scan"] = (pd.DataFrame(rows), fails)

    if "mtf_scan" not in st.session_state:
        return
    res, fails = st.session_state["mtf_scan"]
    if res.empty:
        st.warning("No results.")
    else:
        f1, f2, f3 = st.columns(3)
        min_score = f1.slider("Min score", -3, 3, 2)
        fresh = f2.number_input("Daily ST flipped within N bars (0 = off)", 0, 60, 0)
        rr = f3.slider("Daily RSI range", 0, 100, (0, 100))
        v = res[(res["Score"] >= min_score) & res["D RSI"].between(*rr)]
        if fresh:
            v = v[v["D flip bars"] <= fresh]
        v = v.sort_values(["Score", "D RSI"], ascending=[False, False])
        m1, m2, m3 = st.columns(3)
        m1.metric("Scanned", len(res))
        m2.metric("Matches", len(v))
        m3.metric("All-3 bullish", int((res["Score"] == 3).sum()))
        st.dataframe(v.style.format({"Close": "{:,.2f}", "D RSI": "{:.1f}", "W RSI": "{:.1f}", "M RSI": "{:.1f}"}),
                     width="stretch", hide_index=True)
        st.download_button("⬇️ CSV", v.to_csv(index=False), "mtf_scan.csv", "text/csv")
    if fails:
        with st.expander(f"{len(fails)} symbols failed"):
            st.write("\n".join(fails))


# ------------------------------------------------------------------ router
with st.spinner("Fetching from NSE…"):
    if page == "Sector":
        page_sector()
    elif page == "Top Gainers":
        page_movers("gainers")
    elif page == "Top Losers":
        page_movers("loosers")  # nselib's spelling
    elif page == "FII / DII Activity":
        page_fii()
    elif page == "Delivery Position":
        page_delivery()
    elif page == "Stock Chart (MTF)":
        page_stock_chart()
    elif page == "MTF Scanner":
        page_scanner()
    elif page == "Bulk Deals":
        page_deals("bulk")
    else:
        page_deals("block")
