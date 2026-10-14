"""個股多面向評分工具（台股，Streamlit）

安裝：pip install streamlit pandas numpy requests plotly
執行：streamlit run stock_scorer.py
資料：FinMind 開放 API（免費可用；註冊取得 token 可提高呼叫上限）
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "https://api.finmindtrade.com/api/v4/data"
W_DIM = {"基本面": 0.35, "籌碼面": 0.30, "技術面": 0.35}  # 三大面向權重，可自行調整


# ───────────────────────── 資料 ─────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(dataset: str, stock_id: str, days: int, token: str = "") -> pd.DataFrame:
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start}
    if token:
        params["token"] = token
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(j.get("msg", "API error"))
    return pd.DataFrame(j["data"])


# ───────────────────────── 工具 ─────────────────────────
def lin(x, lo, hi):
    """線性映射到 0~100，小於 lo 為 0，大於 hi 為 100"""
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def combine(items):
    """items: [(指標, 數值文字, 分數或None, 權重)] → (加權分數, 明細表)；缺資料的項目自動重新配權"""
    rows = [
        {"指標": n, "數值": v, "分數": "N/A" if s is None else round(s)}
        for n, v, s, _ in items
    ]
    valid = [(s, w) for _, _, s, w in items if s is not None]
    if not valid:
        return None, rows
    tw = sum(w for _, w in valid)
    return sum(s * w for s, w in valid) / tw, rows


# ───────────────────────── 基本面 ─────────────────────────
def score_fundamental(rev, fin, per):
    items, eps_ttm, fair_pe = [], None, None

    # 1) 近3個月營收年增率
    s, v = None, "資料不足"
    if len(rev) >= 15:
        r = rev.sort_values("date")["revenue"].astype(float).values
        base = r[-15:-12].sum()
        if base > 0:
            yoy = r[-3:].sum() / base - 1
            s, v = lin(yoy, -0.10, 0.30), f"{yoy:+.1%}"
    items.append(("近3月營收年增", v, s, 25))

    # 2) 近四季 EPS 成長、3) 毛利率趨勢
    eps_s, eps_v = None, "資料不足"
    gm_s, gm_v = None, "資料不足"
    if not fin.empty:
        f = fin.pivot_table(index="date", columns="type", values="value", aggfunc="first").sort_index()
        if "EPS" in f:
            eps = f["EPS"].dropna()
            if len(eps) >= 4:
                eps_ttm = float(eps.iloc[-4:].sum())
            if len(eps) >= 8:
                prev = float(eps.iloc[-8:-4].sum())
                if prev > 0:
                    g = eps_ttm / prev - 1
                    eps_s, eps_v = lin(g, -0.2, 0.3), f"近四季EPS {eps_ttm:.2f}，年增 {g:+.1%}"
                else:
                    eps_s = 80 if eps_ttm > 0 else 20
                    eps_v = f"近四季EPS {eps_ttm:.2f}（前期虧損或為零）"
        if {"GrossProfit", "Revenue"} <= set(f.columns):
            gm = (f["GrossProfit"] / f["Revenue"]).dropna()
            if len(gm) >= 5:
                d = gm.iloc[-1] - gm.iloc[-5]
                gm_s, gm_v = lin(d, -0.03, 0.03), f"{gm.iloc[-1]:.1%}（較去年同期 {d * 100:+.1f} 個百分點）"
    items.append(("EPS 成長", eps_v, eps_s, 25))
    items.append(("毛利率趨勢", gm_v, gm_s, 15))

    # 4) 本益比位階（越低越便宜）、5) 殖利率
    pe_s, pe_v, dy_s, dy_v = None, "資料不足", None, "資料不足"
    if not per.empty:
        per = per.sort_values("date")
        pe = per["PER"].astype(float)
        cur = pe.iloc[-1]
        hist = pe[pe > 0]
        if cur > 0 and len(hist) >= 60:
            pct = (hist <= cur).mean()
            pe_s, pe_v = (1 - pct) * 100, f"{cur:.1f} 倍（歷史百分位 {pct:.0%}）"
        elif cur <= 0:
            pe_s, pe_v = 20, "目前虧損"
        if len(hist) >= 60:
            fair_pe = float(hist.median())
        dy = float(per["dividend_yield"].iloc[-1])
        dy_s, dy_v = lin(dy, 0, 6), f"{dy:.2f}%"
    items.append(("本益比位階", pe_v, pe_s, 25))
    items.append(("殖利率", dy_v, dy_s, 10))

    score, rows = combine(items)
    return score, rows, eps_ttm, fair_pe


# ───────────────────────── 籌碼面 ─────────────────────────
def score_chip(inst, margin, price):
    items = []
    vol = price.set_index("date")["Trading_Volume"].astype(float)

    def ratio(names):
        if inst.empty:
            return None
        d = inst[inst["name"].isin(names)].copy()
        d["net"] = d["buy"].astype(float) - d["sell"].astype(float)
        d = d.groupby("date")["net"].sum().sort_index().tail(10)
        v = vol.reindex(d.index).sum()
        return d.sum() / v if len(d) >= 5 and v > 0 else None

    for label, names, rng, w in [
        ("外資近10日買賣超/成交量", ["Foreign_Investor", "Foreign_Dealer_Self"], 0.05, 40),
        ("投信近10日買賣超/成交量", ["Investment_Trust"], 0.02, 35),
    ]:
        r = ratio(names)
        items.append((label, "資料不足" if r is None else f"{r:+.2%}", None if r is None else lin(r, -rng, rng), w))

    s, v = None, "資料不足"
    if not margin.empty and "MarginPurchaseTodayBalance" in margin:
        b = margin.sort_values("date")["MarginPurchaseTodayBalance"].astype(float).values
        if len(b) >= 21 and b[-21] > 0:
            chg = b[-1] / b[-21] - 1
            s, v = lin(-chg, -0.15, 0.15), f"{chg:+.1%}"  # 融資減少（浮額清洗）加分
    items.append(("融資餘額20日變化（減少較佳）", v, s, 25))
    return combine(items)


# ───────────────────────── 技術面 ─────────────────────────
def add_indicators(p):
    p = p.copy()
    for col in ["open", "max", "min", "close"]:
        p[col] = p[col].astype(float)
    c = p["close"]
    p["ma20"], p["ma60"] = c.rolling(20).mean(), c.rolling(60).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    p["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    p["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat(
        [p["max"] - p["min"], (p["max"] - c.shift()).abs(), (p["min"] - c.shift()).abs()], axis=1
    ).max(axis=1)
    p["atr"] = tr.rolling(14).mean()
    return p


def score_technical(p):
    x, prev = p.iloc[-1], p.iloc[-2]
    c = x["close"]
    items = []

    n = int(c > x["ma20"]) + int(x["ma20"] > x["ma60"]) + int(x["ma60"] > p["ma60"].iloc[-21])
    items.append(("趨勢（價>MA20>MA60、MA60上彎）", f"{n}/3 條件成立", n / 3 * 100, 35))

    rsi = x["rsi"]
    items.append(("RSI(14)", f"{rsi:.0f}", float(np.interp(rsi, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5])), 20))

    h, hp = x["hist"], prev["hist"]
    ms = 100 if (h > 0 and h > hp) else 60 if h > 0 else 50 if h > hp else 10
    items.append(("MACD 柱狀體", f"{h:+.2f}（{'增強' if h > hp else '減弱'}）", ms, 25))

    hi, lo = p["max"].tail(250).max(), p["min"].tail(250).min()
    pos = (c - lo) / (hi - lo) if hi > lo else 0.5
    items.append(("52週位階", f"{pos:.0%}", float(np.interp(pos, [0, 0.2, 0.5, 0.8, 1], [40, 70, 80, 55, 30])), 20))
    return combine(items)


# ───────────────────────── 買點與目標價 ─────────────────────────
def verdict(total, f_score, x):
    c = x["close"]
    extended = c / x["ma20"] - 1 > 0.08 or x["rsi"] > 70
    pullback = x["ma60"] * 0.98 <= c <= x["ma20"] * 1.02
    if f_score is not None and f_score >= 65 and pullback and total >= 55:
        return "success", "🟢 基本面佳且回測支撐，屬較佳買點"
    if total >= 75 and not extended:
        return "success", "🟢 評分偏多，可考慮分批買進"
    if total >= 75:
        return "warning", "🟡 好股票但短線過熱，建議等拉回"
    if total >= 55:
        return "warning", "🟡 中性，持續觀察"
    return "error", "🔴 評分偏低，不建議此時進場"


def targets(p, eps_ttm, fair_pe):
    x = p.iloc[-1]
    c = x["close"]
    out = {}
    if eps_ttm and eps_ttm > 0 and fair_pe:
        out["本益比法（近四季EPS × 歷史中位本益比）"] = eps_ttm * fair_pe
    hi, lo = p["max"].tail(120).max(), p["min"].tail(120).min()
    out["技術面（前高／1.272 延伸）"] = hi if hi > c * 1.03 else lo + (hi - lo) * 1.272
    atr_stop = c - 2 * x["atr"]
    stop = max(atr_stop, x["ma60"]) if x["ma60"] < c else atr_stop
    return out, stop


# ───────────────────────── 介面 ─────────────────────────
def show_scores(total, scores):
    cols = st.columns(4)
    cols[0].metric("總分", "N/A" if total is None else f"{total:.0f}")
    for col, (k, v) in zip(cols[1:], scores.items()):
        col.metric(k, "N/A" if v is None else f"{v:.0f}")


def run(code, token):
    with st.spinner("抓取資料中…"):
        try:
            price = fetch("TaiwanStockPrice", code, 450, token)
        except Exception as e:
            st.error(f"股價資料取得失敗：{e}")
            return
        if len(price) < 80:
            st.error("找不到這檔股票，或歷史資料不足 80 個交易日。")
            return

        def get(ds, days):
            try:
                return fetch(ds, code, days, token)
            except Exception:
                return pd.DataFrame()

        rev = get("TaiwanStockMonthRevenue", 800)
        fin = get("TaiwanStockFinancialStatements", 1200)
        per = get("TaiwanStockPER", 1100)
        inst = get("TaiwanStockInstitutionalInvestorsBuySell", 60)
        margin = get("TaiwanStockMarginPurchaseShortSale", 90)

    p = add_indicators(price)
    x = p.iloc[-1]
    f_score, f_rows, eps_ttm, fair_pe = score_fundamental(rev, fin, per)
    c_score, c_rows = score_chip(inst, margin, price)
    t_score, t_rows = score_technical(p)
    scores = {"基本面": f_score, "籌碼面": c_score, "技術面": t_score}
    total, _ = combine([(k, "", s, W_DIM[k]) for k, s in scores.items()])
    if total is None:
        st.error("資料不足，無法評分。")
        return

    st.subheader(f"{code}　現價 {x['close']:.2f}")
    show_scores(total, scores)
    level, text = verdict(total, f_score, x)
    getattr(st, level)(text)

    tg, stop = targets(p, eps_ttm, fair_pe)
    mid = float(np.mean(list(tg.values())))
    risk = x["close"] - stop
    c1, c2, c3 = st.columns(3)
    c1.metric("目標價（平均）", f"{mid:.1f}", f"{(mid / x['close'] - 1) * 100:+.1f}%")
    c2.metric("停損參考", f"{stop:.1f}", f"{(stop / x['close'] - 1) * 100:+.1f}%")
    c3.metric("風險報酬比", f"{(mid - x['close']) / risk:.1f}" if mid > x["close"] and risk > 0 else "—")
    for k, v in tg.items():
        st.caption(f"{k}：{v:.1f}")

    tabs = st.tabs(["基本面", "籌碼面", "技術面", "K線"])
    for tab, rows in zip(tabs[:3], [f_rows, c_rows, t_rows]):
        tab.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    q = p.tail(180)
    d = pd.to_datetime(q["date"])
    fig = go.Figure(go.Candlestick(x=d, open=q["open"], high=q["max"], low=q["min"], close=q["close"], name="K線"))
    fig.add_scatter(x=d, y=q["ma20"], name="MA20")
    fig.add_scatter(x=d, y=q["ma60"], name="MA60")
    fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=0, r=0, t=10, b=0))
    tabs[3].plotly_chart(fig, use_container_width=True)

    st.caption("評分與目標價由公開資料以固定規則計算，僅供參考，不構成投資建議；權重與門檻請自行回測調整。")


def main():
    st.set_page_config(page_title="個股評分", page_icon="📈")
    st.title("📈 個股多面向評分")
    token = st.sidebar.text_input("FinMind Token（選填）", type="password")
    with st.form("f"):
        code = st.text_input("股票代碼", "2330").strip()
        go_btn = st.form_submit_button("分析", type="primary")
    if go_btn and code:
        run(code, token)


if __name__ == "__main__":
    main()
