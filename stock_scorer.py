"""
個股多面向評分與批次快篩工具（台股進階版，Streamlit）
功能亮點：
1. 支援批次輸入代碼 / 常用預設群組（如台灣50精選、高股息等）
2. 批次計算總分、三大面向分數與即時買點訊號，生成排行清單
3. 點選清單中個股，即時載入完整多面向明細、目標價、K線與歷史勝率回測

安裝：pip install streamlit pandas numpy requests plotly
執行：streamlit run stock_batch_scorer.py
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

# 預設自選池
STOCK_POOLS = {
    "台股核心權值": ["2330", "2317", "2454", "2308", "2382", "2881", "2882"],
    "高股息與傳產指標": ["2603", "2002", "1101", "2412", "2886", "2891"],
    "AI伺服器與零組件": ["2382", "3231", "6669", "2376", "2356", "3017"],
}


# ───────────────────────── 資料擷取 ─────────────────────────
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
    items, eps_ttm, fair_pe = [], None, None

    # 1. 營收
    s, v = None, "資料不足"
    if len(rev) >= 15:
        r = rev.sort_values("date")["revenue"].astype(float).values
        base = r[-15:-12].sum()
        if base > 0:
            yoy = r[-3:].sum() / base - 1
            s, v = lin(yoy, -0.10, 0.30), f"{yoy:+.1%}"
    items.append(("近3月營收年增", v, s, 20))

    # 2. EPS、毛利、ROE
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

    # 3. 本益比與殖利率
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

    items.append(("本益比位階", pe_v, pe_s, 15))
    items.append(("殖利率", dy_v, dy_s, 10))

    score, rows = combine(items)
    return score, rows, eps_ttm, fair_pe


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


# ───────────────────────── 技術面評分 ─────────────────────────
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


# ───────────────────────── 買點研判與目標價 ─────────────────────────
def verdict(total, f_score, x):
    c = x["close"]
    extended = c / x["ma20"] - 1 > 0.08 or x["rsi"] > 70
    pullback = x["ma60"] * 0.98 <= c <= x["ma20"] * 1.02
    if f_score is not None and f_score >= 65 and pullback and total >= 55:
        return "success", "🟢 回測支撐買點"
    if total >= 70 and not extended:
        return "success", "🟢 分批佈局區"
    if total >= 70:
        return "warning", "🟡 短線過熱等拉回"
    if total >= 50:
        return "warning", "🟡 中性觀望"
    return "error", "🔴 偏弱觀望"


def targets(p, eps_ttm, fair_pe):
    x = p.iloc[-1]
    c = x["close"]
    out = {}
    if eps_ttm and eps_ttm > 0 and fair_pe:
        out["本益比估值"] = eps_ttm * fair_pe
    hi, lo = p["max"].tail(120).max(), p["min"].tail(120).min()
    out["形態目標價"] = hi if hi > c * 1.03 else lo + (hi - lo) * 1.272
    atr_stop = c - 2 * x["atr"]
    stop = max(atr_stop, x["ma60"]) if x["ma60"] < c else atr_stop
    return out, stop


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


# ───────────────────────── 單股分析計算打包 ─────────────────────────
def calculate_stock_profile(code, token):
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
    f_score, f_rows, eps_ttm, fair_pe = score_fundamental(rev, fin, per)
    c_score, c_rows = score_chip(inst, margin, shareholding, price)
    t_score, t_rows = score_technical(p)
    scores = {"基本面": f_score, "籌碼面": c_score, "技術面": t_score}
    total, _ = combine([(k, "", s, W_DIM[k]) for k, s in scores.items()])

    _, v_text = verdict(total if total else 0, f_score, x)
    tg, stop = targets(p, eps_ttm, fair_pe)
    mid_target = float(np.mean(list(tg.values()))) if tg else np.nan

    return {
        "code": code,
        "close": x["close"],
        "total": round(total) if total else 0,
        "f_score": round(f_score) if f_score else 0,
        "c_score": round(c_score) if c_score else 0,
        "t_score": round(t_score) if t_score else 0,
        "verdict": v_text,
        "target": mid_target,
        "stop": stop,
        "p_df": p,
        "f_rows": f_rows,
        "c_rows": c_rows,
        "t_rows": t_rows,
        "eps_ttm": eps_ttm,
        "fair_pe": fair_pe,
        "targets_dict": tg,
    }


# ───────────────────────── 畫面呈現 ─────────────────────────
def main():
    st.set_page_config(page_title="台股多面向批次評分與快篩", page_icon="📈", layout="wide")
    st.title("📈 台股多面向批次評分與快篩系統")

    # 側邊欄設定
    token = st.sidebar.text_input("FinMind Token（選填，填入提高限額）", type="password")
    pool_choice = st.sidebar.selectbox("快速選取股票池範本", ["自訂輸入"] + list(STOCK_POOLS.keys()))

    if pool_choice != "自訂輸入":
        default_codes = ", ".join(STOCK_POOLS[pool_choice])
    else:
        default_codes = "2330, 2454, 2317, 2382, 2603"

    raw_input = st.sidebar.text_area("股票代碼清單（逗號、空白或換行分隔）", default_codes, height=120)
    batch_btn = st.sidebar.button("🚀 開始批次評分掃描", type="primary")

    # 解析輸入代碼
    codes = [c.strip() for c in raw_input.replace("\n", ",").replace(" ", ",").split(",") if c.strip()]
    codes = list(dict.fromkeys(codes))  # 去除重複

    # 初始化 session state 暫存批次分析結果
    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = {}

    if batch_btn and codes:
        st.session_state["batch_results"] = {}
        progress_bar = st.progress(0)
        status_box = st.empty()

        for idx, code in enumerate(codes):
            status_box.text(f"正在掃描分析：{code} ({idx+1}/{len(codes)}) ...")
            try:
                res = calculate_stock_profile(code, token)
                if res:
                    st.session_state["batch_results"][code] = res
            except Exception as e:
                st.warning(f"標的 {code} 擷取失敗：{e}")
            progress_bar.progress((idx + 1) / len(codes))
            # 避免無 token 時觸發每分鐘 rate limit
            if not token:
                time.sleep(0.3)

        status_box.success("批次掃描完成！")
        progress_bar.empty()

    results = st.session_state["batch_results"]

    if results:
        # 整理排行表格
        summary_rows = []
        for c, data in results.items():
            upside = (data["target"] / data["close"] - 1) * 100 if pd.notna(data["target"]) else None
            summary_rows.append({
                "代碼": c,
                "現價": data["close"],
                "總分": data["total"],
                "基本面": data["f_score"],
                "籌碼面": data["c_score"],
                "技術面": data["t_score"],
                "目標價": round(data["target"], 1) if pd.notna(data["target"]) else "N/A",
                "潛在上漲空間": f"{upside:+.1f}%" if upside is not None else "N/A",
                "狀態訊號": data["verdict"],
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("總分", ascending=False).reset_index(drop=True)

        st.subheader("📋 批次評分排行榜（依總分排序）")
        st.caption("提示：可檢視評分較高或標註『🟢 回測支撐買點』之個股，並在下方深入檢視完整面向與歷史勝率。")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("🔍 個股深入多面向檢驗")

        selected_code = st.selectbox(
            "選擇要深入檢視的個股：",
            options=summary_df["代碼"].tolist(),
            format_func=lambda x: f"{x} - 總分 {results[x]['total']} 分（{results[x]['verdict']}）"
        )

        if selected_code and selected_code in results:
            d = results[selected_code]
            p = d["p_df"]
            x = p.iloc[-1]

            # 頂部儀表板
            m0, m1, m2, m3 = st.columns(4)
            m0.metric("目前總分", f"{d['total']} 分")
            m1.metric("基本面得分 (35%)", f"{d['f_score']} 分")
            m2.metric("籌碼面得分 (30%)", f"{d['c_score']} 分")
            m3.metric("技術面得分 (35%)", f"{d['t_score']} 分")

            # 買點與目標價
            c1, c2, c3 = st.columns(3)
            c1.metric("收盤價", f"{x['close']:.2f} 元")
            upside_val = (d["target"] / x["close"] - 1) * 100 if pd.notna(d["target"]) else None
            c2.metric("綜合目標價", f"{d['target']:.1f} 元" if pd.notna(d["target"]) else "N/A", f"{upside_val:+.1f}%" if upside_val else None)
            c3.metric("動態停損參考", f"{d['stop']:.1f} 元", f"{(d['stop'] / x['close'] - 1) * 100:+.1f}%")

            # 面向明細分頁
            t1, t2, t3, t4, t5 = st.tabs(["基本面明細", "籌碼面明細", "技術面明細", "K線型態", "🎯 該股歷史勝率回測"])
            t1.dataframe(pd.DataFrame(d["f_rows"]), hide_index=True, use_container_width=True)
            t2.dataframe(pd.DataFrame(d["c_rows"]), hide_index=True, use_container_width=True)
            t3.dataframe(pd.DataFrame(d["t_rows"]), hide_index=True, use_container_width=True)

            # 繪製 K 線
            q = p.tail(180)
            dates = pd.to_datetime(q["date"])
            fig = go.Figure(go.Candlestick(x=dates, open=q["open"], high=q["max"], low=q["min"], close=q["close"], name="K線"))
            fig.add_scatter(x=dates, y=q["ma20"], name="20MA (月線)", line=dict(color="orange", width=1.5))
            fig.add_scatter(x=dates, y=q["ma60"], name="60MA (季線)", line=dict(color="blue", width=1.5))
            fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=0, r=0, t=10, b=0))
            t4.plotly_chart(fig, use_container_width=True)

            # 回測面板
            with t5:
                st.write(f"##### {selected_code} 過去買點訊號前向持有勝率檢驗")
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
        st.info("👈 請在左側輸入股票代碼清單，並點擊『🚀 開始批次評分掃描』。")


if __name__ == "__main__":
    main()
