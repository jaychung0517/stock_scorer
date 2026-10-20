"""
個股多面向評分、買點精準決策與停利停損系統（台股實戰量化版 V5，Streamlit）
新增亮點：
1. 排行榜支援「點擊表格列直接切換個股（Interactive Row Selection）」，免除下拉選單二次查找
2. 買入決策五維度檢查表（風報比、乖離率、量價結構、籌碼共振、綜合得分）
3. 三階段停利價位（TP1 保守、TP2 合理、TP3 樂觀）與多重停損防線
4. 單筆部位風控計算機與 K 線視覺化交易計畫

安裝/升級：pip install -U streamlit pandas numpy requests plotly
執行：streamlit run stock_batch_scorer_v5.py
"""
import datetime as dt
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "https://api.finmindtrade.com/api/v4/data"
W_DIM = {"基本面": 0.35, "籌碼面": 0.30, "技術面": 0.35}

STOCK_POOLS = {
    "台股核心權值": ["2330", "2317", "2454", "2308", "2382", "2881", "2882"],
    "高股息與傳產指標": ["2603", "2002", "1101", "2412", "2886", "2891"],
    "AI伺服器與零組件": ["2382", "3231", "6669", "2376", "2356", "3017"],
}


# ───────────────────────── 資料擷取 ─────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_info(token: str = "") -> dict:
    params = {"dataset": "TaiwanStockInfo"}
    if token:
        params["token"] = token
    try:
        r = requests.get(API, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return {item["stock_id"]: item.get("stock_name", "") for item in data}
    except Exception:
        pass
    return {}


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


def get_dataset_safe(ds, code, days, token):
    try:
        return fetch(ds, code, days, token)
    except Exception:
        return pd.DataFrame()


# ───────────────────────── 工具函式 ─────────────────────────
def lin(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def combine(items):
    rows = [
        {"指標": n, "數值": v, "分數": "N/A" if s is None else round(s)}
        for n, v, s, _ in items
    ]
    valid = [(s, w) for _, _, s, w in items if s is not None]
    if not valid:
        return None, rows
    tw = sum(w for _, w in valid)
    return sum(s * w for s, w in valid) / tw, rows


# ───────────────────────── 基本面評分 ─────────────────────────
def score_fundamental(rev, fin, per):
    items, eps_ttm, fair_pe, high_pe = [], None, None, None

    s, v = None, "資料不足"
    if len(rev) >= 15:
        r = rev.sort_values("date")["revenue"].astype(float).values
        base = r[-15:-12].sum()
        if base > 0:
            yoy = r[-3:].sum() / base - 1
            s, v = lin(yoy, -0.10, 0.30), f"{yoy:+.1%}"
    items.append(("近3月營收年增", v, s, 20))

    eps_s, eps_v = None, "資料不足"
    gm_s, gm_v = None, "資料不足"
    roe_s, roe_v = None, "資料不足"

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
                    eps_v = f"近四季EPS {eps_ttm:.2f}（基期負數）"

        if {"GrossProfit", "Revenue"} <= set(f.columns):
            gm = (f["GrossProfit"] / f["Revenue"]).dropna()
            if len(gm) >= 5:
                d = gm.iloc[-1] - gm.iloc[-5]
                gm_s, gm_v = lin(d, -0.03, 0.03), f"{gm.iloc[-1]:.1%}（較去年同期 {d * 100:+.1f} 百分點）"

        profit_col = next((c for c in ["IncomeAfterTaxes", "ConsolidatedNetIncomeAttributableToOwnersOfTheParent"] if c in f), None)
        equity_col = next((c for c in ["TotalEquity", "Equity"] if c in f), None)
        if profit_col and equity_col:
            p_series = f[profit_col].dropna()
            eq_series = f[equity_col].dropna()
            if len(p_series) >= 4 and len(eq_series) >= 1:
                cur_eq = eq_series.iloc[-1]
                if cur_eq > 0:
                    ttm_profit = p_series.iloc[-4:].sum()
                    roe_ttm = (ttm_profit / cur_eq) * 100
                    roe_s = lin(roe_ttm, 5, 20)
                    roe_v = f"{roe_ttm:.2f}%"

    items.append(("EPS 成長", eps_v, eps_s, 20))
    items.append(("毛利率趨勢", gm_v, gm_s, 15))
    items.append(("近四季 ROE", roe_v, roe_s, 20))

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
            high_pe = float(np.percentile(hist, 75))
        dy = float(per["dividend_yield"].iloc[-1])
        dy_s, dy_v = lin(dy, 0, 6), f"{dy:.2f}%"

    items.append(("本益比位階", pe_v, pe_s, 15))
    items.append(("殖利率", dy_v, dy_s, 10))

    score, rows = combine(items)
    return score, rows, eps_ttm, fair_pe, high_pe


# ───────────────────────── 籌碼面評分 ─────────────────────────
def score_chip(inst, margin, shareholding, price):
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
        ("外資近10日買賣超/成交量", ["Foreign_Investor", "Foreign_Dealer_Self"], 0.05, 30),
        ("投信近10日買賣超/成交量", ["Investment_Trust"], 0.02, 25),
    ]:
        r = ratio(names)
        items.append((label, "資料不足" if r is None else f"{r:+.2%}", None if r is None else lin(r, -rng, rng), w))

    m_s, m_v = None, "資料不足"
    if not margin.empty and "MarginPurchaseTodayBalance" in margin:
        b = margin.sort_values("date")["MarginPurchaseTodayBalance"].astype(float).values
        if len(b) >= 21 and b[-21] > 0:
            chg = b[-1] / b[-21] - 1
            m_s, m_v = lin(-chg, -0.15, 0.15), f"{chg:+.1%}"
    items.append(("融資餘額20日變化（減少較佳）", m_v, m_s, 20))

    sh_s, sh_v = None, "資料不足"
    if not shareholding.empty and "holding_percentage" in shareholding:
        sh = shareholding.copy()
        sh["holding_percentage"] = sh["holding_percentage"].astype(float)
        large = sh[sh["HoldingSharesLevel"] >= 12].groupby("date")["holding_percentage"].sum().sort_index()
        if len(large) >= 5:
            diff = large.iloc[-1] - large.iloc[-5]
            sh_s = lin(diff, -1.5, 1.5)
            sh_v = f"{large.iloc[-1]:.2f}%（近4週 {diff:+.2f} 百分點）"
    items.append(("400張大戶持股比率月變化", sh_v, sh_s, 25))

    return combine(items)


# ───────────────────────── 技術面與進階指標 ─────────────────────────
def add_indicators(p):
    p = p.copy()
    for col in ["open", "max", "min", "close", "Trading_Volume"]:
        p[col] = p[col].astype(float)
    c = p["close"]
    p["ma10"] = c.rolling(10).mean()
    p["ma20"] = c.rolling(20).mean()
    p["ma60"] = c.rolling(60).mean()
    p["bb_std"] = c.rolling(20).std()
    p["bb_up"] = p["ma20"] + 2 * p["bb_std"]
    p["bb_lo"] = p["ma20"] - 2 * p["bb_std"]
    p["vol_ma5"] = p["Trading_Volume"].rolling(5).mean()
    p["vol_ma20"] = p["Trading_Volume"].rolling(20).mean()

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    p["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    p["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([p["max"] - p["min"], (p["max"] - c.shift()).abs(), (p["min"] - c.shift()).abs()], axis=1).max(axis=1)
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


# ───────────────────────── 歷史回測模組 ─────────────────────────
def run_backtest(p, buy_threshold=70, hold_days=20):
    df = p.copy()
    c = df["close"]
    c_above_ma20 = (c > df["ma20"]).astype(int)
    ma20_above_ma60 = (df["ma20"] > df["ma60"]).astype(int)
    ma60_turn_up = (df["ma60"] > df["ma60"].shift(20)).astype(int)
    trend_score = (c_above_ma20 + ma20_above_ma60 + ma60_turn_up) / 3 * 100
    rsi_score = df["rsi"].apply(lambda r: float(np.interp(r, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5])))
    macd_score = np.where(df["hist"] > 0, 80, 40)
    rolling_tech = trend_score * 0.45 + rsi_score * 0.25 + macd_score * 0.30

    signals = []
    last_idx = -hold_days
    for i in range(60, len(df) - hold_days):
        if rolling_tech.iloc[i] >= buy_threshold and (i - last_idx) >= hold_days:
            entry_p = c.iloc[i]
            exit_p = c.iloc[i + hold_days]
            ret = (exit_p / entry_p - 1) * 100
            window_min = df["min"].iloc[i: i + hold_days + 1].min()
            mdd = (window_min / entry_p - 1) * 100
            signals.append({
                "進場日期": df["date"].iloc[i],
                "進場價": entry_p,
                "出場價": exit_p,
                f"持有{hold_days}日報酬(%)": round(ret, 2),
                "區間最大回撤(%)": round(mdd, 2),
            })
            last_idx = i
    return pd.DataFrame(signals)


# ───────────────────────── 進階買入評估與三階停利停損系統 ─────────────────────────
def evaluate_trade_plan(p, total, f_score, c_score, t_score, eps_ttm, fair_pe, high_pe):
    x = p.iloc[-1]
    c = float(x["close"])
    atr = float(x["atr"])
    ma10, ma20, ma60 = float(x["ma10"]), float(x["ma20"]), float(x["ma60"])
    bb_up = float(x["bb_up"])

    # 1. 多重停損位計算
    atr_stop = c - 2.0 * atr
    swing_low_20 = float(p["min"].tail(20).min()) * 0.99
    ma60_stop = ma60 * 0.985

    candidate_stops = [s for s in [atr_stop, swing_low_20, ma60_stop] if s < c * 0.99]
    rec_stop = max(candidate_stops) if candidate_stops else atr_stop
    rec_stop = max(rec_stop, c * 0.90)

    stop_details = {
        "波動率停損 (現價 - 2×ATR)": round(atr_stop, 2),
        "型態支撐停損 (近20日低點下緣)": round(swing_low_20, 2),
        "季線防守停損 (60MA下緣1.5%)": round(ma60_stop, 2),
        "移動停利防守線 (10MA短波段線)": round(ma10, 2),
    }

    # 2. 三階段停利目標價 (TP1, TP2, TP3)
    hi_60 = float(p["max"].tail(60).max())
    hi_120 = float(p["max"].tail(120).max())
    lo_120 = float(p["min"].tail(120).min())

    tp1_candidates = [v for v in [hi_60, bb_up] if v > c * 1.02]
    tp1 = min(tp1_candidates) if tp1_candidates else
