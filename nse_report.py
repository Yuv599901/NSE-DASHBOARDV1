"""
nse_report.py - data collection, delivery analysis and post-market PDF for the NSE dashboard.

CLI:   python nse_report.py                 # last trading day -> NSE_PostMarket_<date>.pdf
       python nse_report.py 17-10-2026      # specific day (dd-mm-YYYY)
Import: from nse_report import collect, build_pdf, delivery_analysis
"""
from __future__ import annotations

import io
import sys
from datetime import date, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from nselib import capital_market as cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

try:
    from nselib.capital_market.capital_market_data import fii_dii_trading_activity
except Exception:  # pragma: no cover
    fii_dii_trading_activity = getattr(cm, "fii_dii_trading_activity", None)

DD_MM_YYYY = "%d-%m-%Y"


# ------------------------------------------------------------------ helpers

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

def to_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            conv = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                                 errors="coerce")
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


def find_col(df: pd.DataFrame, *cands):
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in cands:
        k = cand.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    for cand in cands:
        k = cand.lower().replace(" ", "").replace("_", "")
        for nk, orig in norm.items():
            if k in nk:
                return orig
    return None


def safe_call(fn, *a, **kw):
    """-> (DataFrame | None, error | None)"""
    try:
        df = fn(*a, **kw)
        if df is None or len(df) == 0:
            return None, "No data returned."
        return to_num(df), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------- trading-day lookup
def get_bhav(d: date):
    """Bhav copy with delivery for a date, or (None, err)."""
    return safe_call(cm.bhav_copy_with_delivery, d.strftime(DD_MM_YYYY))


def latest_trading_day(start: date | None = None, max_back: int = 7):
    """Walk back from `start` until a bhav copy exists. -> (date, bhav_df) or (None, None)."""
    d = start or date.today()
    for _ in range(max_back + 1):
        if d.weekday() < 5:
            df, err = get_bhav(d)
            if df is not None:
                return d, df
        d -= timedelta(days=1)
    return None, None


def previous_trading_day(d: date, max_back: int = 7):
    return latest_trading_day(d - timedelta(days=1), max_back)


# --------------------------------------------------------- delivery analysis
def _prep_eq(bhav: pd.DataFrame) -> pd.DataFrame:
    df = bhav.copy()
    if "SERIES" in df:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
    df = df.rename(columns={"SYMBOL": "Symbol"})
    for c in ["CLOSE_PRICE", "PREV_CLOSE", "TTL_TRD_QNTY", "TURNOVER_LACS", "DELIV_QTY", "DELIV_PER"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Chg%"] = (df["CLOSE_PRICE"] / df["PREV_CLOSE"] - 1) * 100
    df["TurnoverCr"] = df["TURNOVER_LACS"] / 100
    df["DelivValCr"] = df["TurnoverCr"] * df["DELIV_PER"] / 100
    return df


def delivery_analysis(bhav: pd.DataFrame, prev_bhav: pd.DataFrame | None = None,
                      min_turnover_cr: float = 10.0, min_deliv_pct: float = 55.0,
                      move_pct: float = 2.0, spike_x: float = 2.0, top_n: int = 15) -> dict:
    """Return dict of ready-to-render DataFrames + summary stats."""
    eq = _prep_eq(bhav)
    if prev_bhav is not None:
        p = _prep_eq(prev_bhav)[["Symbol", "DELIV_QTY", "DELIV_PER"]]
        p.columns = ["Symbol", "PrevDelivQty", "PrevDelivPer"]
        eq = eq.merge(p, on="Symbol", how="left")
        eq["DelivQtyX"] = eq["DELIV_QTY"] / eq["PrevDelivQty"]
        eq["DelivPerChg"] = eq["DELIV_PER"] - eq["PrevDelivPer"]
    else:
        eq["DelivQtyX"] = float("nan")
        eq["DelivPerChg"] = float("nan")

    liq = eq[eq["TurnoverCr"] >= min_turnover_cr].dropna(subset=["DELIV_PER"])
    cols = ["Symbol", "CLOSE_PRICE", "Chg%", "TurnoverCr", "DELIV_PER", "DelivValCr", "DelivQtyX"]

    out = {
        "all": eq,
        "top_deliv_pct": liq.sort_values("DELIV_PER", ascending=False).head(top_n)[cols],
        "accumulation": liq[(liq["Chg%"] >= move_pct) & (liq["DELIV_PER"] >= min_deliv_pct)]
        .sort_values("DelivValCr", ascending=False).head(top_n)[cols],
        "distribution": liq[(liq["Chg%"] <= -move_pct) & (liq["DELIV_PER"] >= min_deliv_pct)]
        .sort_values("DelivValCr", ascending=False).head(top_n)[cols],
        "spike": liq[liq["DelivQtyX"] >= spike_x].sort_values("DelivQtyX", ascending=False)
        .head(top_n)[cols],
        "stats": {
            "stocks": int(len(eq)),
            "advances": int((eq["Chg%"] > 0).sum()),
            "declines": int((eq["Chg%"] < 0).sum()),
            "unchanged": int((eq["Chg%"] == 0).sum()),
            "turnover_cr": float(eq["TurnoverCr"].sum()),
            "avg_deliv_pct": float(liq["DELIV_PER"].mean()) if len(liq) else float("nan"),
            "wtd_deliv_pct": float(eq["DelivValCr"].sum() / eq["TurnoverCr"].sum() * 100)
            if eq["TurnoverCr"].sum() else float("nan"),
        },
        "params": dict(min_turnover_cr=min_turnover_cr, min_deliv_pct=min_deliv_pct,
                       move_pct=move_pct, spike_x=spike_x),
    }
    return out


# ------------------------------------------------------------- data collect
def collect(trade_date: date | None = None) -> dict:
    """Pull everything the report needs. Each section is (df|None, err|None)."""
    d, bhav = latest_trading_day(trade_date)
    if d is None:
        raise RuntimeError("No bhav copy found in the last 7 days (NSE blocked or data not published).")
    _, prev = previous_trading_day(d)
    data = {
        "date": d,
        "bhav": bhav,
        "delivery": delivery_analysis(bhav, prev),
        "indices": safe_call(fetch_indices),
        "gainers": safe_call(cm.top_gainers_or_losers, "gainers"),
        "losers": safe_call(cm.top_gainers_or_losers, "loosers"),
        "fii_dii": safe_call(fii_dii_trading_activity) if fii_dii_trading_activity else (None, "n/a"),
        "bulk": safe_call(cm.bulk_deal_data, period="1D"),
        "block": safe_call(cm.block_deals_data, period="1D"),
    }
    return data


# ================================================================== PDF
NAVY = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#334155")
GREY = colors.HexColor("#f1f5f9")
GREEN = colors.HexColor("#15803d")
RED = colors.HexColor("#b91c1c")
ACCENT = colors.HexColor("#2563eb")

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=14, textColor=NAVY, spaceBefore=10, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=10.5, textColor=SLATE, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=_ss["Normal"], fontSize=8.5, leading=11)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=7.5, textColor=colors.HexColor("#64748b"))
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=7.5, leading=9)
KPI_V = ParagraphStyle("KpiV", parent=BODY, fontSize=13, leading=15, alignment=TA_CENTER, textColor=NAVY)
KPI_L = ParagraphStyle("KpiL", parent=NOTE, alignment=TA_CENTER)

PAGE_W = A4[0] - 24 * mm


def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if kind != "str" and isinstance(v, str):
        v = pd.to_numeric(v.replace(",", ""), errors="coerce")
        if pd.isna(v):
            return "-"
    if kind == "pct":
        return f"{v:+.2f}%"
    if kind == "pct0":
        return f"{v:.1f}%"
    if kind == "x":
        return f"{v:.1f}x"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "str":
        return str(v)
    return f"{v:,.2f}"


def _table(df: pd.DataFrame, spec: list, col_w=None, signed_cols=()):
    """spec = [(column, header, kind)], signed_cols = headers coloured by sign."""
    rows = [[h for _, h, _ in spec]]
    for _, r in df.iterrows():
        rows.append([_fmt(r.get(c), k) if k != "str" else Paragraph(str(r.get(c, ""))[:34], CELL)
                     for c, _, k in spec])
    t = Table(rows, colWidths=col_w, repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    heads = [h for _, h, _ in spec]
    for j, (_, _, k) in enumerate(spec):
        if k == "str":
            st.append(("ALIGN", (j, 0), (j, -1), "LEFT"))
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        for h in signed_cols:
            if h in heads:
                j = heads.index(h)
                c, _, _ = spec[j]
                v = r.get(c)
                if v is not None and not pd.isna(v) and v != 0:
                    st.append(("TEXTCOLOR", (j, i), (j, i), GREEN if v > 0 else RED))
    t.setStyle(TableStyle(st))
    return t


def _kpis(items):
    cells = [[Paragraph(v, KPI_V) for v, _ in items], [Paragraph(l, KPI_L) for _, l in items]]
    t = Table(cells, colWidths=[PAGE_W / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREY),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def _unavailable(name, err):
    return Paragraph(f"<i>{name}: data unavailable ({err})</i>", NOTE)


def _bar_png(labels, values, title, w=170, h=None):
    h = h or max(60, 5.2 * len(labels) + 14)
    fig, ax = plt.subplots(figsize=(w / 25.4, h / 25.4), dpi=200)
    cols = ["#15803d" if v >= 0 else "#b91c1c" for v in values]
    ax.barh(labels, values, color=cols, height=0.65)
    ax.invert_yaxis()
    ax.axvline(0, color="#64748b", lw=0.6)
    ax.tick_params(labelsize=6)
    lo, hi = min(min(values), 0), max(max(values), 0)
    pad = (hi - lo) * 0.18 or 1
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(title, fontsize=8, loc="left", color="#0f172a")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:+.2f}%", va="center", ha="left" if v >= 0 else "right", fontsize=5.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w * mm, height=h * mm)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(12 * mm, 8 * mm, "Source: NSE via nselib. For research only - not investment advice.")
    canvas.drawRightString(A4[0] - 12 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(data: dict, out=None) -> bytes | None:
    """Render report. `out` = path or file-like; if None returns bytes."""
    d: date = data["date"]
    buf = io.BytesIO() if out is None else out
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=14 * mm,
                            title=f"NSE Post-Market Report {d:%d %b %Y}", author="NSE Dashboard")
    S = []

    # ---- header band
    hdr = Table([[Paragraph(f"<font color='white' size=16><b>NSE Post-Market Report</b></font><br/>"
                            f"<font color='#94a3b8' size=9>{d:%A, %d %B %Y}  |  generated "
                            f"{datetime.now():%d %b %Y %H:%M}</font>", BODY)]], colWidths=[PAGE_W])
    hdr.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    S += [hdr, Spacer(1, 6)]

    # ---- market snapshot
    dl = data["delivery"]["stats"]
    S.append(_kpis([
        (f"{dl['advances']:,} / {dl['declines']:,}", "Advances / Declines (EQ)"),
        (f"{dl['turnover_cr']:,.0f} Cr", "EQ turnover (Rs)"),
        (f"{dl['wtd_deliv_pct']:.1f}%", "Delivery % (value-weighted)"),
        (f"{dl['avg_deliv_pct']:.1f}%", "Avg delivery % (liquid)"),
    ]))

    idx, err = data["indices"]
    if idx is not None:
        ncol, pcol, lcol = find_col(idx, "index"), find_col(idx, "percentChange"), find_col(idx, "last")
        wanted = ["NIFTY 50", "NIFTY BANK", "NIFTY NEXT 50", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250",
                  "NIFTY 500", "INDIA VIX"]
        sub = idx[idx[ncol].astype(str).str.upper().isin(wanted)].copy()
        if len(sub):
            S.append(Paragraph("Key indices", H1))
            sub["o"] = sub[ncol].astype(str).str.upper().map({n: i for i, n in enumerate(wanted)})
            sub = sub.sort_values("o")
            extra = [(find_col(sub, "perChange30d"), "30D %", "pct"), (find_col(sub, "perChange365d"), "1Y %", "pct")]
            spec = [(ncol, "Index", "str"), (lcol, "Last", "num"), (pcol, "1D %", "pct")] + \
                   [(c, h, k) for c, h, k in extra if c]
            S.append(_table(sub, spec, signed_cols=("1D %", "30D %", "1Y %")))

        # ---- sector
        key = find_col(idx, "key")
        sec = idx[idx[key].astype(str).str.upper() == "SECTORAL INDICES"].sort_values(pcol, ascending=False) \
            if key else pd.DataFrame()
        if len(sec):
            S.append(Paragraph("Sector performance", H1))
            S.append(_bar_png([str(x).replace("NIFTY ", "") for x in sec[ncol]], list(sec[pcol]),
                              "Sectoral indices - 1D % change"))
            best, worst = sec.iloc[0], sec.iloc[-1]
            S.append(Paragraph(f"Leader: <b>{best[ncol]}</b> ({best[pcol]:+.2f}%). "
                               f"Laggard: <b>{worst[ncol]}</b> ({worst[pcol]:+.2f}%). "
                               f"{int((sec[pcol] > 0).sum())} of {len(sec)} sectors advanced.", BODY))
    else:
        S += [Paragraph("Key indices & sectors", H1), _unavailable("Indices", err)]

    # ---- gainers / losers
    half = (PAGE_W - 6 * mm) / 2
    side = []
    for name, k, ttl in (("gainers", "gainers", "Top gainers"), ("losers", "losers", "Top losers")):
        g, e = data[k]
        cell = [Paragraph(ttl, H2)]
        if g is None:
            cell.append(_unavailable(ttl, e))
        else:
            sym = find_col(g, "symbol")
            pct = find_col(g, "perChange", "pChange", "percentChange", "netPrice")
            ltp = find_col(g, "ltp", "lastPrice", "last")
            vol = find_col(g, "tradedQuantity", "tradeVolume", "volume")
            g = g.sort_values(pct, ascending=(name == "losers")).drop_duplicates(sym).head(20)
            spec = [(sym, "Symbol", "str")] + ([(ltp, "LTP", "num")] if ltp else []) + \
                   [(pct, "Chg %", "pct")] + ([(vol, "Volume", "int")] if vol else [])
            cell.append(_table(g, spec, col_w=[half * f for f in ([0.3, 0.2, 0.2, 0.3] if len(spec) == 4
                                                                    else [1 / len(spec)] * len(spec))],
                               signed_cols=("Chg %",)))
        side.append(cell)
    wrap = Table([side], colWidths=[half + 3 * mm, half + 3 * mm])
    wrap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (0, 0), 6), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    S.append(KeepTogether([Paragraph("Top gainers & losers", H1), wrap]))

    # ---- FII / DII
    S.append(Paragraph("FII / DII activity (cash, Rs crore)", H1))
    f, e = data["fii_dii"]
    if f is None:
        S.append(_unavailable("FII/DII", e))
    else:
        cat, b, s, n = (find_col(f, "category"), find_col(f, "buyValue"), find_col(f, "sellValue"),
                        find_col(f, "netValue"))
        f = coerce(f, [b, s, n])
        S.append(_table(f, [(cat, "Category", "str"), (b, "Buy", "num"), (s, "Sell", "num"), (n, "Net", "num")],
                        signed_cols=("Net",)))
        dc = find_col(f, "date")
        if dc:
            S.append(Paragraph(f"NSE data as of {f[dc].iloc[0]}.", NOTE))

    # ---- delivery
    S.append(PageBreak())
    S.append(Paragraph("Delivery position", H1))
    p = data["delivery"]["params"]
    S.append(Paragraph(
        f"EQ series only. Liquidity filter: turnover >= Rs {p['min_turnover_cr']:.0f} Cr. "
        f"Accumulation = close >= +{p['move_pct']}% with delivery >= {p['min_deliv_pct']:.0f}%; "
        f"Distribution = close <= -{p['move_pct']}% with the same delivery filter. "
        f"Spike = delivery qty >= {p['spike_x']:.0f}x previous session.", NOTE))
    dspec = [("Symbol", "Symbol", "str"), ("CLOSE_PRICE", "Close", "num"), ("Chg%", "Chg %", "pct"),
             ("TurnoverCr", "Turnover Cr", "num"), ("DELIV_PER", "Deliv %", "pct0"),
             ("DelivValCr", "Deliv Val Cr", "num"), ("DelivQtyX", "Deliv Qty vs Prev", "x")]
    for key, ttl in (("top_deliv_pct", "Highest delivery % (liquid stocks)"),
                     ("accumulation", "Accumulation: price up on high delivery"),
                     ("distribution", "Distribution: price down on high delivery"),
                     ("spike", "Delivery spikes vs previous session")):
        df = data["delivery"][key]
        blk = [Paragraph(ttl, H2)]
        blk.append(_table(df, dspec, signed_cols=("Chg %",)) if len(df)
                   else Paragraph("<i>No stocks matched.</i>", NOTE))
        S.append(KeepTogether(blk))

    # ---- bulk / block
    S.append(PageBreak())
    S.append(Paragraph("Bulk & block deals (latest session)", H1))
    for k, ttl in (("bulk", "Bulk deals"), ("block", "Block deals")):
        g, e = data[k]
        S.append(Paragraph(ttl, H2))
        if g is None:
            S.append(_unavailable(ttl, e))
            continue
        sym, cl, sd = find_col(g, "Symbol"), find_col(g, "ClientName", "Client"), find_col(g, "Buy/Sell", "BuySell")
        q, pr = find_col(g, "QuantityTraded", "Quantity"), find_col(g, "TradePrice", "Wght.Avg.TradePrice", "Price")
        g = coerce(g, [q, pr])
        g["ValueCr"] = g[q] * g[pr] / 1e7
        top = g.sort_values("ValueCr", ascending=False).head(15)
        S.append(_table(top, [(sym, "Symbol", "str"), (cl, "Client", "str"), (sd, "B/S", "str"),
                              (q, "Qty", "int"), (pr, "Price", "num"), ("ValueCr", "Value Cr", "num")],
                        col_w=[24 * mm, 62 * mm, 12 * mm, 26 * mm, 22 * mm, PAGE_W - 146 * mm]))
        S.append(Paragraph(f"{len(g)} deals, total Rs {g['ValueCr'].sum():,.0f} Cr; top 15 by value shown.", NOTE))

    doc.build(S, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue() if out is None else None


# ---------------------------------------------------------------------- CLI
if __name__ == "__main__":
    start = datetime.strptime(sys.argv[1], DD_MM_YYYY).date() if len(sys.argv) > 1 else None
    payload = collect(start)
    path = f"NSE_PostMarket_{payload['date']:%Y-%m-%d}.pdf"
    with open(path, "wb") as fh:
        fh.write(build_pdf(payload))
    print("Saved", path)
