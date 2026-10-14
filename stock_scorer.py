"""
個股多面向評分工具（台股進階版，Streamlit）
功能擴充：
1. 基本面加入 ROE（杜邦核心指標）
2. 籌碼面加入集保大戶持股比率變化（400張以上大戶持股集中度）
3. 新增「歷史回測驗證模組」，檢視歷史買進訊號的勝率與平均報酬

安裝套件：pip install streamlit pandas numpy requests plotly
執行方式：streamlit run stock_scorer_v2.py
"""
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "https://api.finmindtrade.com/api/v4/data"
W_DIM = {"基本面": 0.35, "籌碼面": 0.30, "技術面": 0.35}


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


# ───────────────────────── 工具函式 ─────────────────────────
def lin(x, lo, hi):
    """線性映射到 0~100 分"""
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def combine(items):
    """缺資料項目自動重分配權重"""
    rows = [
        {"指標": n, "數值": v, "分數": "N/A" if s is None else round(s)}
        for n, v, s, _ in items
    ]
    valid = [(s, w) for _, _, s, w in items if s is not None]
    if not valid:
        return None, rows
    tw = sum(w for _, w in valid)
    return sum(s * w for s, w in valid) / tw, rows


# ───────────────────────── 基本面評分（含 ROE） ─────────────────────────
def score_fundamental(rev, fin, per):
    items, eps_ttm, fair_pe = [], None, None

    # 1. 近3個月營收年增率 (權重 20)
    s, v = None, "資料不足"
    if len(rev) >= 15:
        r = rev.sort_values("date")["revenue"].astype(float).values
        base = r[-15:-12].sum()
        if base > 0:
            yoy = r[-3:].sum() / base - 1
            s, v = lin(yoy, -0.10, 0.30), f"{yoy:+.1%}"
    items.append(("近3月營收年增", v, s, 20))

    # 2. EPS 成長率 (權重 20)、3. 毛利率趨勢 (權重 15)、4. 近四季 ROE (權重 20)
    eps_s, eps_v = None, "資料不足"
    gm_s, gm_v = None, "資料不足"
    roe_s, roe_v = None, "資料不足"

    if not fin.empty:
        f = fin.pivot_table(index="date", columns="type", values="value", aggfunc="first").sort_index()
        
        # EPS 計算
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
                    eps_v = f"近四季EPS {eps_ttm:.2f}（基期為負）"

        # 毛利率變動
        if {"GrossProfit", "Revenue"} <= set(f.columns):
            gm = (f["GrossProfit"] / f["Revenue"]).dropna()
            if len(gm) >= 5:
                d = gm.iloc[-1] - gm.iloc[-5]
                gm_s, gm_v = lin(d, -0.03, 0.03), f"{gm.iloc[-1]:.1%}（較去年同期 {d * 100:+.1f} 百分點）"

        # ROE 計算（綜合損益歸屬母公司業主淨利 / 權益總額）
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
                    # ROE 8% 基準，>20% 給滿分
                    roe_s = lin(roe_ttm, 5, 20)
                    roe_v = f"{roe_ttm:.2f}%"

    items.append(("EPS 成長", eps_v, eps_s, 20))
    items.append(("毛利率趨勢", gm_v, gm_s, 15))
    items.append(("近四季 ROE", roe_v, roe_s, 20))

    # 5. 本益比位階 (權重 15)、6. 殖利率 (權重 10)
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


# ───────────────────────── 籌碼面評分（含大戶持股） ─────────────────────────
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

    # 1. 外資、2. 投信買賣超佔比
    for label, names, rng, w in [
        ("外資近10日買賣超/成交量", ["Foreign_Investor", "Foreign_Dealer_Self"], 0.05, 30),
        ("投信近10日買賣超/成交量", ["Investment_Trust"], 0.02, 25),
    ]:
        r = ratio(names)
        items.append((label, "資料不足" if r is None else f"{r:+.2%}", None if r is None else lin(r, -rng, rng), w))

    # 3. 融資籌碼變化
    m_s, m_v = None, "資料不足"
    if not margin.empty and "MarginPurchaseTodayBalance" in margin:
        b = margin.sort_values("date")["MarginPurchaseTodayBalance"].astype(float).values
        if len(b) >= 21 and b[-21] > 0:
            chg = b[-1] / b[-21] - 1
            m_s, m_v = lin(-chg, -0.15, 0.15), f"{chg:+.1%}"
    items.append(("融資餘額20日變化（減少較佳）", m_v, m_s, 20))

    # 4. 集保 400 張以上大戶持股比率月變化 (權重 25)
    sh_s, sh_v = None, "資料不足"
    if not shareholding.empty and "holding_percentage" in shareholding:
        sh = shareholding.copy()
        sh["holding_percentage"] = sh["holding_percentage"].astype(float)
        # FinMind level >= 12 通常為 400張以上大戶（15 為千張）
        large = sh[sh["HoldingSharesLevel"] >= 12].groupby("date")["holding_percentage"].sum().sort_index()
        if len(large) >= 5:
            # 觀察最新一週相對於 4 週前（約1個月）的變化幅度
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


# ───────────────────────── 買點研判與目標價 ─────────────────────────
def verdict(total, f_score, x):
    c = x["close"]
    extended = c / x["ma20"] - 1 > 0.08 or x["rsi"] > 70
    pullback = x["ma60"] * 0.98 <= c <= x["ma20"] * 1.02
    if f_score is not None and f_score >= 65 and pullback and total >= 55:
        return "success", "🟢 基本面扎實且回測均線支撐，為高性價比買點"
    if total >= 70 and not extended:
        return "success", "🟢 綜合評分亮眼且無過熱現象，適合分批佈局"
    if total >= 70:
        return "warning", "🟡 綜合得分高但短期指標過熱（乖離/RSI偏高），建議等待拉回"
    if total >= 50:
        return "warning", "🟡 訊號中性或多空拉鋸，暫時觀望"
    return "error", "🔴 綜合評分低迷，避開為宜"


def targets(p, eps_ttm, fair_pe):
    x = p.iloc[-1]
    c = x["close"]
    out = {}
    if eps_ttm and eps_ttm > 0 and fair_pe:
        out["本益比估值（近四季EPS × 歷史中位本益比）"] = eps_ttm * fair_pe
    hi, lo = p["max"].tail(120).max(), p["min"].tail(120).min()
    out["技術形態（前高／1.272 費波那契延伸）"] = hi if hi > c * 1.03 else lo + (hi - lo) * 1.272
    atr_stop = c - 2 * x["atr"]
    stop = max(atr_stop, x["ma60"]) if x["ma60"] < c else atr_stop
    return out, stop


# ───────────────────────── 歷史回測模組 ─────────────────────────
def run_backtest(p, buy_threshold=70, hold_days=20):
    """
    歷史訊號回測：
    使用過去的技術面動能與均線狀態模擬滾動分數，計算觸發買進門檻後持有 N 日的前向報酬率與勝率
    """
    df = p.copy()
    c = df["close"]
    
    # 建立滾動技術代理分數（歷史技術面即時評估）
    c_above_ma20 = (c > df["ma20"]).astype(int)
    ma20_above_ma60 = (df["ma20"] > df["ma60"]).astype(int)
    ma60_turn_up = (df["ma60"] > df["ma60"].shift(20)).astype(int)
    trend_score = (c_above_ma20 + ma20_above_ma60 + ma60_turn_up) / 3 * 100
    
    rsi_score = df["rsi"].apply(lambda r: float(np.interp(r, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5])))
    macd_score = np.where(df["hist"] > 0, 80, 40)
    rolling_tech = trend_score * 0.45 + rsi_score * 0.25 + macd_score * 0.30
    
    # 觸發條件（分數跨過門檻且距上次訊號大於持有期，避免重複取樣）
    signals = []
    last_idx = -hold_days
    for i in range(60, len(df) - hold_days):
        if rolling_tech.iloc[i] >= buy_threshold and (i - last_idx) >= hold_days:
            entry_p = c.iloc[i]
            exit_p = c.iloc[i + hold_days]
            ret = (exit_p / entry_p - 1) * 100
            
            # 最大潛在跌幅 (MDD during holding)
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
            
    res_df = pd.DataFrame(signals)
    return res_df


# ───────────────────────── 主流程 ─────────────────────────
def run(code, token):
    with st.spinner("抓取各面向資料與籌碼集中度中…"):
        try:
            price = fetch("TaiwanStockPrice", code, 550, token)
        except Exception as e:
            st.error(f"股價資料取得失敗：{e}")
            return
        if len(price) < 80:
            st.error("查無資料或交易歷史不足 80 天。")
            return

        def get(ds, days):
            try:
                return fetch(ds, code, days, token)
            except Exception:
                return pd.DataFrame()

        rev = get("TaiwanStockMonthRevenue", 850)
        fin = get("TaiwanStockFinancialStatements", 1400)
        per = get("TaiwanStockPER", 1100)
        inst = get("TaiwanStockInstitutionalInvestorsBuySell", 60)
        margin = get("TaiwanStockMarginPurchaseShortSale", 90)
        shareholding = get("TaiwanStockShareholding", 180)  # 大戶持股集中度

    p = add_indicators(price)
    x = p.iloc[-1]
    f_score, f_rows, eps_ttm, fair_pe = score_fundamental(rev, fin, per)
    c_score, c_rows = score_chip(inst, margin, shareholding, price)
    t_score, t_rows = score_technical(p)
    scores = {"基本面": f_score, "籌碼面": c_score, "技術面": t_score}
    total, _ = combine([(k, "", s, W_DIM[k]) for k, s in scores.items()])

    if total is None:
        st.error("有效資料不足，無法完成評分。")
        return

    # 頂部總分展示
    st.subheader(f"📊 {code} 現價：{x['close']:.2f} 元")
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("總評分", f"{total:.0f}" if total else "N/A")
    c1.metric("基本面 (35%)", f"{f_score:.0f}" if f_score else "N/A")
    c2.metric("籌碼面 (30%)", f"{c_score:.0f}" if c_score else "N/A")
    c3.metric("技術面 (35%)", f"{t_score:.0f}" if t_score else "N/A")

    # 買點判斷
    level, verdict_text = verdict(total, f_score, x)
    getattr(st, level)(verdict_text)

    # 目標價與風險回報估算
    tg, stop = targets(p, eps_ttm, fair_pe)
    mid_target = float(np.mean(list(tg.values())))
    risk = x["close"] - stop
    
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("綜合目標價", f"{mid_target:.1f} 元", f"{(mid_target / x['close'] - 1) * 100:+.1f}%")
    col_t2.metric("動態停損參考", f"{stop:.1f} 元", f"{(stop / x['close'] - 1) * 100:+.1f}%")
    rr_ratio = (mid_target - x["close"]) / risk if (mid_target > x["close"] and risk > 0) else None
    col_t3.metric("風報比 (Reward/Risk)", f"{rr_ratio:.2f}" if rr_ratio else "無顯著優勢")

    for k, v in tg.items():
        st.caption(f"📌 {k}：約 {v:.1f} 元")

    # 分頁展示明細與回測
    t1, t2, t3, t4, t5 = st.tabs(["基本面明細", "籌碼面明細", "技術面明細", "K線走勢", "🎯 評分歷史回測"])
    t1.dataframe(pd.DataFrame(f_rows), hide_index=True, use_container_width=True)
    t2.dataframe(pd.DataFrame(c_rows), hide_index=True, use_container_width=True)
    t3.dataframe(pd.DataFrame(t_rows), hide_index=True, use_container_width=True)

    # 繪製 K 線圖
    q = p.tail(180)
    d = pd.to_datetime(q["date"])
    fig = go.Figure(go.Candlestick(x=d, open=q["open"], high=q["max"], low=q["min"], close=q["close"], name="K線"))
    fig.add_scatter(x=d, y=q["ma20"], name="20MA (月線)", line=dict(color="orange", width=1.5))
    fig.add_scatter(x=d, y=q["ma60"], name="60MA (季線)", line=dict(color="blue", width=1.5))
    fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=10, b=0))
    t4.plotly_chart(fig, use_container_width=True)

    # 歷史回測驗證面板
    with t5:
        st.write("##### 歷史進場訊號前向報酬檢驗")
        st.caption("模擬在過去若於技術與動能評分高於買進門檻時進場，持有特定交易日數後的預測勝率與損益。")
        col_b1, col_b2 = st.columns(2)
        th = col_b1.slider("回測進場分數門檻", 60, 85, 70, 5)
        h_days = col_b2.selectbox("進場後持有天數 (交易日)", [10, 20, 60], index=1)
        
        bt_res = run_backtest(p, buy_threshold=th, hold_days=h_days)
        if bt_res.empty:
            st.info("該段歷史區間未觸發符合門檻的交易訊號。")
        else:
            col_name = f"持有{h_days}日報酬(%)"
            win_rate = (bt_res[col_name] > 0).mean() * 100
            avg_ret = bt_res[col_name].mean()
            avg_mdd = bt_res["區間最大回撤(%)"].mean()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("觸發訊號次數", f"{len(bt_res)} 次")
            m2.metric("勝率", f"{win_rate:.1f}%")
            m3.metric("平均報酬率", f"{avg_ret:+.2f}%")
            m4.metric("平均持倉最大回撤", f"{avg_mdd:.2f}%")
            
            st.dataframe(bt_res, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="台股多面向評分與量化回測", page_icon="📈", layout="wide")
    st.title("📈 台股多面向自動評分與回測系統")
    token = st.sidebar.text_input("FinMind Token（填入可提升 API 請求頻率限制）", type="password")
    with st.form("input_form"):
        code = st.text_input("輸入台股代碼（如 2330, 2454, 2317）", "2330").strip()
        submit = st.form_submit_button("執行多面向分析", type="primary")
    if submit and code:
        run(code, token)


if __name__ == "__main__":
    main()
