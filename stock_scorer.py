"""
個股多面向評分、買點精準決策與停利停損系統（台股實戰穩健版，Streamlit）
排行榜改為「按鈕列」：點代碼或股名即可切換下方個股交易計畫
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
    rows = []
    for n, v, s, _ in items:
        score_str = "N/A" if s is None else round(s)
        rows.append({"指標": n, "數值": v, "分數": score_str})
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
                    eps_s = lin(g, -0.2, 0.3)
                    eps_v = f"近四季EPS {eps_ttm:.2f}，年增 {g:+.1%}"
                else:
                    eps_s = 80 if eps_ttm > 0 else 20
                    eps_v = f"近四季EPS {eps_ttm:.2f}（基期負數）"

        if {"GrossProfit", "Revenue"} <= set(f.columns):
            gm = (f["GrossProfit"] / f["Revenue"]).dropna()
            if len(gm) >= 5:
                d = gm.iloc[-1] - gm.iloc[-5]
                gm_s = lin(d, -0.03, 0.03)
                gm_v = f"{gm.iloc[-1]:.1%}（較去年同期 {d * 100:+.1f} 百分點）"

        profit_cols = ["IncomeAfterTaxes", "ConsolidatedNetIncomeAttributableToOwnersOfTheParent"]
        profit_col = next((c for c in profit_cols if c in f), None)
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
            pe_s = (1 - pct) * 100
            pe_v = f"{cur:.1f} 倍（歷史百分位 {pct:.0%}）"
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
        if len(d) >= 5 and v > 0:
            return d.sum() / v
        return None

    for label, names, rng, w in [
        ("外資近10日買賣超/成交量", ["Foreign_Investor", "Foreign_Dealer_Self"], 0.05, 30),
        ("投信近10日買賣超/成交量", ["Investment_Trust"], 0.02, 25),
    ]:
        r = ratio(names)
        val_str = "資料不足" if r is None else f"{r:+.2%}"
        sc_val = None if r is None else lin(r, -rng, rng)
        items.append((label, val_str, sc_val, w))

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
    tr = pd.concat(
        [p["max"] - p["min"], (p["max"] - c.shift()).abs(), (p["min"] - c.shift()).abs()],
        axis=1,
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
    rsi_sc = float(np.interp(rsi, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5]))
    items.append(("RSI(14)", f"{rsi:.0f}", rsi_sc, 20))

    h, hp = x["hist"], prev["hist"]
    if h > 0 and h > hp:
        ms = 100
    elif h > 0:
        ms = 60
    elif h > hp:
        ms = 50
    else:
        ms = 10
    status_str = "增強" if h > hp else "減弱"
    items.append(("MACD 柱狀體", f"{h:+.2f}（{status_str}）", ms, 25))

    hi, lo = p["max"].tail(250).max(), p["min"].tail(250).min()
    pos = (c - lo) / (hi - lo) if hi > lo else 0.5
    pos_sc = float(np.interp(pos, [0, 0.2, 0.5, 0.8, 1], [40, 70, 80, 55, 30]))
    items.append(("52週位階", f"{pos:.0%}", pos_sc, 20))
    return combine(items)


# ───────────────────────── 歷史回測模組 ─────────────────────────
def run_backtest(p, buy_threshold=70, hold_days=20):
    df = p.copy()
    c = df["close"]
    c_above_ma20 = (c > df["ma20"]).astype(int)
    ma20_above_ma60 = (df["ma20"] > df["ma60"]).astype(int)
    ma60_turn_up = (df["ma60"] > df["ma60"].shift(20)).astype(int)
    trend_score = (c_above_ma20 + ma20_above_ma60 + ma60_turn_up) / 3 * 100
    rsi_score = df["rsi"].apply(
        lambda r: float(np.interp(r, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5]))
    )
    macd_score = np.where(df["hist"] > 0, 80, 40)
    rolling_tech = trend_score * 0.45 + rsi_score * 0.25 + macd_score * 0.30

    signals = []
    last_idx = -hold_days
    for i in range(60, len(df) - hold_days):
        if rolling_tech.iloc[i] >= buy_threshold and (i - last_idx) >= hold_days:
            entry_p = c.iloc[i]
            exit_p = c.iloc[i + hold_days]
            ret = (exit_p / entry_p - 1) * 100
            window_min = df["min"].iloc[i : i + hold_days + 1].min()
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

    atr_stop = c - 2.0 * atr
    swing_low_20 = float(p["min"].tail(20).min()) * 0.99
    ma60_stop = ma60 * 0.985

    candidate_stops = [s for s in [atr_stop, swing_low_20, ma60_stop] if s < c * 0.99]
    if candidate_stops:
        rec_stop = max(candidate_stops)
    else:
        rec_stop = atr_stop
    rec_stop = max(rec_stop, c * 0.90)

    stop_details = {
        "波動率停損 (現價 - 2×ATR)": round(atr_stop, 2),
        "型態支撐停損 (近20日低點下緣)": round(swing_low_20, 2),
        "季線防守停損 (60MA下緣1.5%)": round(ma60_stop, 2),
        "移動停利防守線 (10MA短波段線)": round(ma10, 2),
    }

    hi_60 = float(p["max"].tail(60).max())
    hi_120 = float(p["max"].tail(120).max())
    lo_120 = float(p["min"].tail(120).min())

    tp1_candidates = [v for v in [hi_60, bb_up] if v > c * 1.02]
    if tp1_candidates:
        tp1 = min(tp1_candidates)
    else:
        tp1 = c + 2.0 * atr

    fib_1272 = lo_120 + (hi_120 - lo_120) * 1.272
    if eps_ttm and eps_ttm > 0 and fair_pe:
        pe_target_fair = eps_ttm * fair_pe
    else:
        pe_target_fair = fib_1272
    tp2_raw = float(np.mean([fib_1272, pe_target_fair]))
    tp2 = max(tp2_raw, tp1 * 1.04)

    fib_1618 = lo_120 + (hi_120 - lo_120) * 1.618
    if eps_ttm and eps_ttm > 0 and high_pe:
        pe_target_high = eps_ttm * high_pe
    else:
        pe_target_high = fib_1618
    tp3_raw = float(np.mean([fib_1618, pe_target_high]))
    tp3 = max(tp3_raw, tp2 * 1.06)

    risk = max(c - rec_stop, 0.01)
    reward_tp2 = max(tp2 - c, 0)
    rr_ratio = reward_tp2 / risk

    bias_20 = (c / ma20 - 1) * 100
    buy_zone_low = round(min(ma20, c - 0.8 * atr), 2)
    if bias_20 > 5:
        buy_zone_high = round(min(c, ma10 * 1.01), 2)
    else:
        buy_zone_high = round(c * 1.005, 2)

    last5 = p.tail(5)
    up_vol = last5[last5["close"] >= last5["open"]]["Trading_Volume"].mean()
    dn_vol = last5[last5["close"] < last5["open"]]["Trading_Volume"].mean()
    vol_healthy = (pd.isna(dn_vol)) or (up_vol > dn_vol) or (x["Trading_Volume"] < x["vol_ma20"] and c >= ma20)

    checklist = [
        {
            "檢核項目": "1. 綜合基本與趨勢評分",
            "狀態": "✅ 通過" if (total and total >= 65) else "⚠️ 偏弱",
            "當前數值": f"總分 {total or 0:.0f} 分 (基本面 {f_score or 0:.0f} / 技術面 {t_score or 0:.0f})",
            "判斷標準": "總分 ≥ 65 分代表多面向具備優勢",
            "pass": bool(total and total >= 65),
        },
        {
            "檢核項目": "2. 風險報酬比 (以 TP2 計算)",
            "狀態": "✅ 通過" if rr_ratio >= 1.5 else "⚠️ 不佳",
            "當前數值": f"1 : {rr_ratio:.2f} (潛在獲利 +{reward_tp2/c*100:.1f}% / 停損風險 -{risk/c*100:.1f}%)",
            "判斷標準": "風報比 ≥ 1.5 倍才具備出手價值",
            "pass": bool(rr_ratio >= 1.5),
        },
        {
            "檢核項目": "3. 均線乖離與追高風險",
            "狀態": "✅ 通過" if (-2 <= bias_20 <= 7 and x["rsi"] <= 72) else "⚠️ 過熱或破線",
            "當前數值": f"月線乖離 {bias_20:+.1f}%，RSI 為 {x['rsi']:.0f}",
            "判斷標準": "月線乖離介於 -2% ~ +7% 且 RSI ≤ 72 不追高",
            "pass": bool(-2 <= bias_20 <= 7 and x["rsi"] <= 72),
        },
        {
            "檢核項目": "4. 籌碼面主力動向",
            "狀態": "✅ 通過" if (c_score and c_score >= 55) else "⚠️ 籌碼渙散",
            "當前數值": f"籌碼得分 {c_score or 0:.0f} 分",
            "判斷標準": "籌碼分 ≥ 55 代表法人或大戶偏多操作",
            "pass": bool(c_score and c_score >= 55),
        },
        {
            "檢核項目": "5. 短期量價結構",
            "狀態": "✅ 通過" if vol_healthy else "⚠️ 出貨疑慮",
            "當前數值": "價漲量增 / 量縮守支撐" if vol_healthy else "下跌出量或高檔爆量收黑",
            "判斷標準": "攻擊時有量、拉回時量縮",
            "pass": bool(vol_healthy),
        },
    ]

    pass_count = sum(1 for item in checklist if item["pass"])

    if pass_count == 5:
        action = "🟢 強烈建議買進（五維度全數過關，風報比極佳）"
        action_level = "success"
    elif pass_count == 4 and rr_ratio >= 1.3:
        action = "🟢 適合分批買進（多數條件符合，可先建基本倉）"
        action_level = "success"
    elif (total and total >= 70) and (bias_20 > 7 or rr_ratio < 1.3):
        action = f"🟡 標的優質但現價過高（建議掛單等待拉回 {buy_zone_low}～{buy_zone_high} 元再買）"
        action_level = "warning"
    elif pass_count >= 3:
        action = "🟡 條件好壞參半，建議小部位試單或觀望"
        action_level = "warning"
    else:
        action = "🔴 不建議此時買入（風報比不佳或趨勢偏空，宜避開）"
        action_level = "error"

    return {
        "action": action,
        "action_level": action_level,
        "pass_count": pass_count,
        "checklist": checklist,
        "rec_stop": round(rec_stop, 2),
        "stop_pct": round((rec_stop / c - 1) * 100, 2),
        "stop_details": stop_details,
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "rr_ratio": round(rr_ratio, 2),
        "buy_zone": (buy_zone_low, buy_zone_high),
    }


# ───────────────────────── 單股分析計算打包 ─────────────────────────
def calculate_stock_profile(code, stock_name, token):
    price = get_dataset_safe("TaiwanStockPrice", code, 550, token)
    if len(price) < 80:
        return None

    rev = get_dataset_safe("TaiwanStockMonthRevenue", code, 850, token)
    fin = get_dataset_safe("TaiwanStockFinancialStatements", code, 1400, token)
    per = get_dataset_safe("TaiwanStockPER", code, 1100, token)
    inst = get_dataset_safe("TaiwanStockInstitutionalInvestorsBuySell", code, 60, token)
    margin = get_dataset_safe("TaiwanStockMarginPurchaseShortSale", code, 90, token)
    shareholding = get_dataset_safe("TaiwanStockShareholding", code, 180, token)

    p = add_indicators(price)
    x = p.iloc[-1]
    f_score, f_rows, eps_ttm, fair_pe, high_pe = score_fundamental(rev, fin, per)
    c_score, c_rows = score_chip(inst, margin, shareholding, price)
    t_score, t_rows = score_technical(p)
    scores = {"基本面": f_score, "籌碼面": c_score, "技術面": t_score}
    total, _ = combine([(k, "", s, W_DIM[k]) for k, s in scores.items()])

    plan = evaluate_trade_plan(p, total, f_score, c_score, t_score, eps_ttm, fair_pe, high_pe)

    return {
        "code": code,
        "name": stock_name,
        "display_name": f"{code} {stock_name}".strip(),
        "close": float(x["close"]),
        "total": round(total) if total else 0,
        "f_score": round(f_score) if f_score else 0,
        "c_score": round(c_score) if c_score else 0,
        "t_score": round(t_score) if t_score else 0,
        "plan": plan,
        "p_df": p,
        "f_rows": f_rows,
        "c_rows": c_rows,
        "t_rows": t_rows,
    }


# ───────────────────────── 畫面呈現 ─────────────────────────
def main():
    st.set_page_config(page_title="台股量化買點決策與部位系統", page_icon="📈", layout="wide")
    st.title("📈 台股多面向批次評分、買點檢核與停利停損系統")

    token = st.sidebar.text_input("FinMind Token（選填，填入提高限額）", type="password")
    info_map = fetch_stock_info(token)

    pool_choice = st.sidebar.selectbox("快速選取股票池範本", ["自訂輸入"] + list(STOCK_POOLS.keys()))
    if pool_choice != "自訂輸入":
        default_codes = ", ".join(STOCK_POOLS[pool_choice])
    else:
        default_codes = "2330, 2454, 2317, 2382, 2603"
    raw_input = st.sidebar.text_area("股票代碼清單（逗號、空白或換行分隔）", default_codes, height=100)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 資金與風控參數設定")
    capital = st.sidebar.number_input("預計投入總本金（萬元）", min_value=5.0, max_value=10000.0, value=100.0, step=10.0)
    risk_pct = st.sidebar.slider("單筆交易最大可承受總本金虧損 (%)", 0.5, 5.0, 2.0, 0.5)

    batch_btn = st.sidebar.button("🚀 開始批次評分與買點掃描", type="primary")

    codes = [c.strip() for c in raw_input.replace("\n", ",").replace(" ", ",").split(",") if c.strip()]
    codes = list(dict.fromkeys(codes))

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = {}

    if batch_btn and codes:
        st.session_state["batch_results"] = {}
        st.session_state.pop("selected_code", None)
        progress_bar = st.progress(0)
        status_box = st.empty()

        for idx, code in enumerate(codes):
            s_name = info_map.get(code, "")
            status_box.text(f"正在掃描分析：{code} {s_name} ({idx+1}/{len(codes)}) ...")
            try:
                res = calculate_stock_profile(code, s_name, token)
                if res:
                    st.session_state["batch_results"][code] = res
            except Exception as e:
                st.warning(f"標的 {code} 擷取失敗：{e}")
            progress_bar.progress((idx + 1) / len(codes))
            if not token:
                time.sleep(0.3)

        status_box.success("批次掃描完成！")
        progress_bar.empty()

    results = st.session_state["batch_results"]

    if results:
        summary_rows = []
        for c, data in results.items():
            pl = data["plan"]
            upside_tp2 = (pl["tp2"] / data["close"] - 1) * 100
            summary_rows.append({
                "代碼": data["code"],
                "股名": data["name"] if data["name"] else "—",
                "現價": data["close"],
                "總分": data["total"],
                "基本面": data["f_score"],
                "籌碼面": data["c_score"],
                "技術面": data["t_score"],
                "檢核通過": f"{pl['pass_count']} / 5 項",
                "風報比": pl["rr_ratio"],
                "停損價": f"{pl['rec_stop']} ({pl['stop_pct']:+.1f}%)",
                "合理停利(TP2)": f"{pl['tp2']} ({upside_tp2:+.1f}%)",
                "操作建議": pl["action"].split("（")[0],
            })

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values(["總分", "風報比"], ascending=[False, False])
            .reset_index(drop=True)
        )

        # ── 可點選排行榜：每列一個按鈕，點代碼或股名即切換下方個股 ──
        if st.session_state.get("selected_code") not in results:
            st.session_state["selected_code"] = summary_df.iloc[0]["代碼"]

        st.subheader("📋 批次評分與風報比排行榜")
        st.caption("點選個股按鈕（代碼＋股名），下方會直接切換成該股的完整交易計畫；停損、停利、三面向分數請看下方詳情。")

        def short_action(text):
            for key, label in [
                ("強烈建議買進", "🟢 強烈買進"),
                ("適合分批買進", "🟢 分批買進"),
                ("標的優質但現價過高", "🟡 等拉回"),
                ("條件好壞參半", "🟡 觀望/試單"),
                ("不建議此時買入", "🔴 避開"),
            ]:
                if key in text:
                    return label
            return text.split("（")[0]

        cols_def = [("個股", 2.6), ("現價", 1.3), ("總分", 0.8), ("檢核", 0.8), ("風報比", 1.0), ("建議", 2.0)]
        widths = [w for _, w in cols_def]

        def make_cols():
            try:
                return st.columns(widths, vertical_alignment="center")
            except TypeError:  # 舊版 Streamlit 不支援垂直置中
                return st.columns(widths)

        def cell(col, text, bold=False):
            t = f"<b>{text}</b>" if bold else text
            col.markdown(f"<div style='white-space:nowrap'>{t}</div>", unsafe_allow_html=True)

        hdr = make_cols()
        for h, (name, _) in zip(hdr, cols_def):
            cell(h, name, bold=True)

        for _, r in summary_df.iterrows():
            code = r["代碼"]
            pl_r = results[code]["plan"]
            label = f"{code} {r['股名']}" if r["股名"] != "—" else str(code)
            is_sel = code == st.session_state["selected_code"]
            rc = make_cols()

            if rc[0].button(label, key=f"pick_{code}", type="primary" if is_sel else "secondary"):
                st.session_state["selected_code"] = code
                st.rerun()

            cell(rc[1], f"{r['現價']:,.2f}")
            cell(rc[2], f"{r['總分']}")
            cell(rc[3], f"{pl_r['pass_count']}/5")
            cell(rc[4], f"{r['風報比']:.2f}")
            cell(rc[5], short_action(pl_r["action"]))

        selected_code = st.session_state["selected_code"]
        st.write("---")

        if selected_code and selected_code in results:
            d = results[selected_code]
            pl = d["plan"]
            p = d["p_df"]
            cur_p = d["close"]

            st.markdown(f"## 🎯 個股交易計畫：{d['display_name']}　（最新收盤價：**{cur_p:.2f} 元**）")
            getattr(st, pl["action_level"])(f"**買入評估結論：{pl['action']}**")

            m0, m1, m2, m3, m4 = st.columns(5)
            m0.metric("綜合總分", f"{d['total']} 分")
            m1.metric("基本面 (35%)", f"{d['f_score']} 分")
            m2.metric("籌碼面 (30%)", f"{d['c_score']} 分")
            m3.metric("技術面 (35%)", f"{d['t_score']} 分")
            m4.metric("檢核表通過數", f"{pl['pass_count']} / 5 項")

            st.markdown("#### 📌 實戰價位規劃（建議買入區、三段式停利、防守停損）")
            p1, p2, p3, p4, p5 = st.columns(5)
            bz_lo, bz_hi = pl["buy_zone"]
            p1.metric("理想買入區間", f"{bz_lo} ~ {bz_hi}", "拉回分批承接")
            p2.metric("🛑 建議停損價", f"{pl['rec_stop']} 元", f"{pl['stop_pct']:+.1f}%", delta_color="inverse")
            p3.metric("🎯 第一停利 (TP1)", f"{pl['tp1']} 元", f"{(pl['tp1']/cur_p - 1)*100:+.1f}% (先出1/3)")
            p4.metric("🎯 第二停利 (TP2)", f"{pl['tp2']} 元", f"{(pl['tp2']/cur_p - 1)*100:+.1f}% (再出1/3)")
            p5.metric("🚀 樂觀停利 (TP3)", f"{pl['tp3']} 元", f"{(pl['tp3']/cur_p - 1)*100:+.1f}% (移動停利)")

            max_loss_ntd = capital * 10000 * (risk_pct / 100.0)
            loss_per_share = max(cur_p - pl["rec_stop"], 0.1)
            suggested_shares = int(max_loss_ntd // loss_per_share)
            max_affordable_shares = int((capital * 10000) // cur_p)
            final_shares = min(suggested_shares, max_affordable_shares)
            position_value = final_shares * cur_p
            position_weight = (position_value / (capital * 10000)) * 100

            with st.expander("⚖️ 單筆風險部位試算與停損防線細節（點擊展開）", expanded=True):
                rc1, rc2 = st.columns([1.2, 1])
                with rc1:
                    st.markdown("**💡 資金控管建議（固定風險比例法）**")
                    st.write(
                        f"- 您的總本金為 **{capital:.0f} 萬元**，設定單筆最大容忍虧損 **{risk_pct:.1f}%**（即最多賠 **{max_loss_ntd:,.0f} 元**）。\n"
                        f"- 若於現價 **{cur_p} 元** 買進，跌破 **{pl['rec_stop']} 元** 停損，每股風險為 **{loss_per_share:.2f} 元**。\n"
                        f"- **建議買進部位**：約 **{final_shares:,} 股**（約 **{final_shares/1000:.2f} 張**），"
                        f"投入金額約 **{position_value/10000:.1f} 萬元**（佔總資金 **{position_weight:.1f}%**）。"
                    )
                with rc2:
                    st.markdown("**🛡️ 各類停損防線參考價位**")
                    for k_stop, v_stop in pl["stop_details"].items():
                        diff_s = (v_stop / cur_p - 1) * 100
                        st.caption(f"• {k_stop}：**{v_stop} 元** ({diff_s:+.1f}%)")

            t0, t1, t2, t3, t4, t5 = st.tabs([
                "✅ 買入五維度檢核", "📈 K線與停利停損圖", "基本面明細", "籌碼面明細", "技術面明細", "🎯 歷史勝率回測"
            ])

            with t0:
                chk_df = pd.DataFrame(pl["checklist"]).drop(columns=["pass"])
                st.dataframe(chk_df, hide_index=True, use_container_width=True)

            with t1:
                q = p.tail(150)
                dates = pd.to_datetime(q["date"])
                fig = go.Figure(
                    go.Candlestick(
                        x=dates, open=q["open"], high=q["max"], low=q["min"], close=q["close"], name="K線"
                    )
                )
                fig.add_scatter(x=dates, y=q["ma20"], name="20MA (月線)", line=dict(color="orange", width=1.5))
                fig.add_scatter(x=dates, y=q["ma60"], name="60MA (季線)", line=dict(color="royalblue", width=1.5))

                fig.add_hline(
                    y=pl["rec_stop"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"停損價 {pl['rec_stop']}",
                    annotation_position="bottom right",
                )
                fig.add_hline(
                    y=pl["tp1"],
                    line_dash="dot",
                    line_color="seagreen",
                    annotation_text=f"TP1 保守停利 {pl['tp1']}",
                    annotation_position="top right",
                )
                fig.add_hline(
                    y=pl["tp2"],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"TP2 合理停利 {pl['tp2']}",
                    annotation_position="top right",
                )
                fig.add_hrect(
                    y0=bz_lo,
                    y1=bz_hi,
                    fillcolor="lightgreen",
                    opacity=0.15,
                    line_width=0,
                    annotation_text="建議承接區",
                    annotation_position="bottom left",
                )

                fig.update_layout(xaxis_rangeslider_visible=False, height=480, margin=dict(l=0, r=0, t=15, b=0))
                st.plotly_chart(fig, use_container_width=True)

            t2.dataframe(pd.DataFrame(d["f_rows"]), hide_index=True, use_container_width=True)
            t3.dataframe(pd.DataFrame(d["c_rows"]), hide_index=True, use_container_width=True)
            t4.dataframe(pd.DataFrame(d["t_rows"]), hide_index=True, use_container_width=True)

            with t5:
                st.write(f"##### {d['display_name']} 歷史進場訊號前向報酬檢驗")
                col_b1, col_b2 = st.columns(2)
                th = col_b1.slider("回測進場分數門檻", 60, 85, 70, 5, key=f"th_{selected_code}")
                h_days = col_b2.selectbox("進場後持有交易日數", [10, 20, 60], index=1, key=f"hd_{selected_code}")

                bt_res = run_backtest(p, buy_threshold=th, hold_days=h_days)
                if bt_res.empty:
                    st.info("該股票在歷史區間內未觸發符合門檻的交易訊號。")
                else:
                    col_name = f"持有{h_days}日報酬(%)"
                    win_rate = (bt_res[col_name] > 0).mean() * 100
                    avg_ret = bt_res[col_name].mean()
                    avg_mdd = bt_res["區間最大回撤(%)"].mean()

                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric("觸發次數", f"{len(bt_res)} 次")
                    w2.metric("持有勝率", f"{win_rate:.1f}%")
                    w3.metric("平均報酬率", f"{avg_ret:+.2f}%")
                    w4.metric("平均持倉最大回撤", f"{avg_mdd:.2f}%")
                    st.dataframe(bt_res, hide_index=True, use_container_width=True)
    else:
        st.info("👈 請在左側設定本金與風險參數，並點擊『🚀 開始批次評分與買點掃描』。")


if __name__ == "__main__":
    main()
