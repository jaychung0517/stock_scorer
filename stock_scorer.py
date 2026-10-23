"""
個股多面向評分、買點精準決策與停利停損系統（台股實戰穩健版，Streamlit）
功能：
- 基本面、籌碼面、技術面評分
- 量價型態初篩、資料完整度與風報比排名
- 可點選排行榜切換個股交易計畫
- 三階停利停損、風險部位試算
- 含交易成本的非重疊交易回測、樣本外結果及 0050 基準比較

注意：
1. 財報科目口徑可能因資料集而異；本程式不會自動把所有財報數字都當成單季或累計數字。
2. 回測採簡化的下一交易日開盤進場模型，未完整模擬漲跌停、滑價、除權息及流動性。
3. 0050 僅作為可取得資料時的市場代理基準，不等同加權股價指數。
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
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") == 200:
            return {
                str(item.get("stock_id", "")).strip(): item.get("stock_name", "")
                for item in payload.get("data", [])
                if item.get("stock_id")
            }
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(dataset: str, stock_id: str, days: int, token: str = "") -> pd.DataFrame:
    start = (dt.date.today() - dt.timedelta(days=int(days))).isoformat()
    params = {
        "dataset": dataset,
        "data_id": str(stock_id),
        "start_date": start,
    }
    if token:
        params["token"] = token
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != 200:
        raise RuntimeError(payload.get("msg", "FinMind API error"))
    return pd.DataFrame(payload.get("data", []))


def get_dataset_safe(ds, code, days, token):
    try:
        df = fetch(ds, str(code), days, token)
        if df.empty:
            return df
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ───────────────────────── 工具函式 ─────────────────────────
def lin(x, lo, hi):
    if x is None or not np.isfinite(x) or hi == lo:
        return None
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def safe_float(value, default=np.nan):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def combine(items):
    rows = []
    for name, value, score, weight in items:
        rows.append({
            "指標": name,
            "數值": value,
            "分數": "N/A" if score is None or not np.isfinite(score) else round(float(score), 1),
        })
    valid = [
        (float(score), float(weight))
        for _, _, score, weight in items
        if score is not None and np.isfinite(score) and weight > 0
    ]
    if not valid:
        return None, rows
    total_weight = sum(weight for _, weight in valid)
    return sum(score * weight for score, weight in valid) / total_weight, rows


def normalize_financial_data(fin):
    """清理財報資料；不在未確認口徑前自動做累計轉單季。"""
    if fin is None or fin.empty:
        return pd.DataFrame(columns=["date", "type", "value"])
    df = fin.copy()
    if not {"date", "type", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=["date", "type", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["type"] = df["type"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "type", "value"])
    return df.sort_values(["type", "date"]).drop_duplicates(["date", "type"], keep="last")


def quarterly_aligned_growth(series):
    """取相隔約四季的可比值，避免直接拿相鄰季度當年增。"""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(s) < 5:
        return None
    latest_date = pd.Timestamp(s.index[-1])
    target = latest_date - pd.DateOffset(months=12)
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    candidates = np.where(np.abs((idx - target).days) <= 60)[0]
    if len(candidates) == 0:
        return None
    old = safe_float(s.iloc[int(candidates[-1])])
    cur = safe_float(s.iloc[-1])
    if not np.isfinite(old) or not np.isfinite(cur) or old == 0:
        return None
    return cur / old - 1


def data_completeness(scores):
    keys = ["基本面", "籌碼面", "技術面"]
    return sum(scores.get(k) is not None for k in keys) / len(keys) * 100


# ───────────────────────── 基本面評分 ─────────────────────────
def score_fundamental(rev, fin, per):
    items, eps_ttm, fair_pe, high_pe = [], None, None, None

    # 營收年增：比較最近三個月與去年同期三個月。
    s, v = None, "資料不足"
    if rev is not None and not rev.empty and {"date", "revenue"}.issubset(rev.columns):
        r = rev.copy()
        r["date"] = pd.to_datetime(r["date"], errors="coerce")
        r["revenue"] = pd.to_numeric(r["revenue"], errors="coerce")
        r = r.dropna(subset=["date", "revenue"]).sort_values("date")
        if len(r) >= 15:
            vals = r["revenue"].to_numpy(dtype=float)
            base = np.nansum(vals[-15:-12])
            recent = np.nansum(vals[-3:])
            if base > 0:
                yoy = recent / base - 1
                s, v = lin(yoy, -0.10, 0.30), f"{yoy:+.1%}"
    items.append(("近3月營收年增", v, s, 20))

    eps_s, eps_v = None, "資料不足"
    gm_s, gm_v = None, "資料不足"
    roe_s, roe_v = None, "資料不足"

    f = normalize_financial_data(fin)
    if not f.empty:
        pivot = f.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
        # EPS：只在最近資料至少覆蓋四個不同季度時估算。資料集若提供累計 EPS，
        # 請先確認口徑後再轉單季；此處不盲目加總可能為累計的數字。
        if "EPS" in pivot.columns:
            eps = pivot["EPS"].dropna().sort_index()
            quarter_keys = eps.index.to_period("Q")
            eps = eps[~quarter_keys.duplicated(keep="last")]
            if len(eps) >= 4:
                eps_ttm = float(eps.iloc[-4:].sum())
                if len(eps) >= 8:
                    prev_ttm = float(eps.iloc[-8:-4].sum())
                    if prev_ttm > 0:
                        growth = eps_ttm / prev_ttm - 1
                        eps_s = lin(growth, -0.2, 0.3)
                        eps_v = f"近四筆 EPS 合計 {eps_ttm:.2f}，對前四筆 {growth:+.1%}"
                    else:
                        eps_s = 80 if eps_ttm > 0 else 20
                        eps_v = f"近四筆 EPS 合計 {eps_ttm:.2f}（前期基期非正）"
                else:
                    eps_s = 70 if eps_ttm > 0 else 25
                    eps_v = f"近四筆 EPS 合計 {eps_ttm:.2f}（缺少完整同比資料）"

        # 毛利率：用同期間 GrossProfit / Revenue，與約四季前比較。
        if {"GrossProfit", "Revenue"}.issubset(pivot.columns):
            denominator = pivot["Revenue"].replace(0, np.nan)
            gm = (pivot["GrossProfit"] / denominator).replace([np.inf, -np.inf], np.nan).dropna()
            gm = gm[~gm.index.to_period("Q").duplicated(keep="last")]
            if len(gm) >= 5:
                growth = quarterly_aligned_growth(gm)
                if growth is not None:
                    diff = float(gm.iloc[-1] - gm.iloc[-2]) if len(gm) >= 2 else 0.0
                    # 評分採毛利率同比變化；顯示最新毛利率及同比差。
                    latest_date = gm.index[-1]
                    target = latest_date - pd.DateOffset(months=12)
                    candidates = np.where(np.abs((pd.DatetimeIndex(gm.index) - target).days) <= 60)[0]
                    if len(candidates):
                        delta = float(gm.iloc[-1] - gm.iloc[int(candidates[-1])])
                        gm_s = lin(delta, -0.03, 0.03)
                        gm_v = f"{gm.iloc[-1]:.1%}（較約去年同期 {delta * 100:+.1f} 百分點）"
                    else:
                        gm_s = lin(growth, -0.1, 0.1)
                        gm_v = f"{gm.iloc[-1]:.1%}（同比資料近似）"

        # ROE：只有資料看起來至少涵蓋四季才計算。注意財報流量科目口徑需由來源確認。
        profit_cols = [
            "IncomeAfterTaxes",
            "ConsolidatedNetIncomeAttributableToOwnersOfTheParent",
        ]
        profit_col = next((col for col in profit_cols if col in pivot.columns), None)
        equity_col = next((col for col in ["TotalEquity", "Equity"] if col in pivot.columns), None)
        if profit_col and equity_col:
            profit = pivot[profit_col].dropna().sort_index()
            equity = pivot[equity_col].dropna().sort_index()
            profit = profit[~profit.index.to_period("Q").duplicated(keep="last")]
            equity = equity[~equity.index.to_period("Q").duplicated(keep="last")]
            if len(profit) >= 4 and len(equity) >= 2:
                # 若來源提供的是年初至今累計損益，這裡必須先依來源規格轉成單季。
                # 為避免無法判斷口徑時錯算，若季度月份序列明顯不完整則不評分。
                last4 = profit.iloc[-4:]
                q_months = [pd.Timestamp(d).month for d in last4.index]
                if len(set(pd.Timestamp(d).to_period("Q") for d in last4.index)) == 4:
                    ttm_profit = float(last4.sum())
                    avg_equity = float(equity.iloc[-2:].mean())
                    if avg_equity > 0:
                        roe_ttm = ttm_profit / avg_equity * 100
                        # 避免資料單位或累計口徑異常導致離譜分數。
                        if -100 <= roe_ttm <= 100:
                            roe_s = lin(roe_ttm, 5, 20)
                            roe_v = f"{roe_ttm:.2f}%（依近四筆損益估算）"

    items.extend([
        ("EPS 成長", eps_v, eps_s, 20),
        ("毛利率趨勢", gm_v, gm_s, 15),
        ("近四季 ROE", roe_v, roe_s, 20),
    ])

    pe_s, pe_v, dy_s, dy_v = None, "資料不足", None, "資料不足"
    if per is not None and not per.empty and "date" in per.columns:
        per_df = per.copy().sort_values("date")
        if "PER" in per_df:
            pe = pd.to_numeric(per_df["PER"], errors="coerce").dropna()
            if len(pe):
                cur = float(pe.iloc[-1])
                hist = pe[pe > 0]
                if cur > 0 and len(hist) >= 60:
                    pct = float((hist <= cur).mean())
                    pe_s = (1 - pct) * 100
                    pe_v = f"{cur:.1f} 倍（歷史百分位 {pct:.0%}）"
                elif cur <= 0:
                    pe_s, pe_v = 20, "目前本益比非正"
                if len(hist) >= 60:
                    fair_pe = float(hist.median())
                    high_pe = float(np.percentile(hist, 75))
        if "dividend_yield" in per_df:
            dy_series = pd.to_numeric(per_df["dividend_yield"], errors="coerce").dropna()
            if len(dy_series):
                dy = float(dy_series.iloc[-1])
                # FinMind dividend_yield 常以百分比呈現，若值介於 0～1 需依來源規格確認。
                dy_s, dy_v = lin(dy, 0, 6), f"{dy:.2f}%"

    items.extend([
        ("本益比位階", pe_v, pe_s, 15),
        ("殖利率", dy_v, dy_s, 10),
    ])
    score, rows = combine(items)
    return score, rows, eps_ttm, fair_pe, high_pe


# ───────────────────────── 籌碼面評分 ─────────────────────────
def score_chip(inst, margin, shareholding, price):
    items = []
    if price is not None and not price.empty and {"date", "Trading_Volume"}.issubset(price.columns):
        vol = price.set_index("date")["Trading_Volume"].pipe(pd.to_numeric, errors="coerce")
    else:
        vol = pd.Series(dtype=float)

    def ratio(names):
        if inst is None or inst.empty or not {"name", "date", "buy", "sell"}.issubset(inst.columns):
            return None
        d = inst[inst["name"].isin(names)].copy()
        if d.empty:
            return None
        d["buy"] = pd.to_numeric(d["buy"], errors="coerce")
        d["sell"] = pd.to_numeric(d["sell"], errors="coerce")
        d["net"] = d["buy"] - d["sell"]
        d = d.groupby("date")["net"].sum().sort_index().tail(10)
        common = vol.reindex(d.index).dropna()
        d = d.reindex(common.index)
        denominator = common.sum()
        if len(d) >= 5 and denominator > 0:
            return float(d.sum() / denominator)
        return None

    for label, names, rng, weight in [
        ("外資近10日買賣超/成交量", ["Foreign_Investor", "Foreign_Dealer_Self"], 0.05, 30),
        ("投信近10日買賣超/成交量", ["Investment_Trust"], 0.02, 25),
    ]:
        value = ratio(names)
        items.append((
            label,
            "資料不足" if value is None else f"{value:+.2%}",
            None if value is None else lin(value, -rng, rng),
            weight,
        ))

    margin_score, margin_value = None, "資料不足"
    if margin is not None and not margin.empty and "MarginPurchaseTodayBalance" in margin:
        m = margin.sort_values("date").copy()
        m["MarginPurchaseTodayBalance"] = pd.to_numeric(m["MarginPurchaseTodayBalance"], errors="coerce")
        b = m["MarginPurchaseTodayBalance"].dropna().to_numpy()
        if len(b) >= 21 and b[-21] > 0:
            change = b[-1] / b[-21] - 1
            margin_score = lin(-change, -0.15, 0.15)
            margin_value = f"{change:+.1%}"
    items.append(("融資餘額20日變化（減少較佳）", margin_value, margin_score, 20))

    sh_score, sh_value = None, "資料不足"
    if shareholding is not None and not shareholding.empty:
        required = {"date", "holding_percentage", "HoldingSharesLevel"}
        if required.issubset(shareholding.columns):
            sh = shareholding.copy()
            sh["holding_percentage"] = pd.to_numeric(sh["holding_percentage"], errors="coerce")
            sh["HoldingSharesLevel"] = pd.to_numeric(sh["HoldingSharesLevel"], errors="coerce")
            large = sh[sh["HoldingSharesLevel"] >= 12].groupby("date")["holding_percentage"].sum().sort_index()
            if len(large) >= 5:
                change = float(large.iloc[-1] - large.iloc[-5])
                sh_score = lin(change, -1.5, 1.5)
                sh_value = f"{large.iloc[-1]:.2f}%（近4週 {change:+.2f} 百分點）"
    items.append(("400張大戶持股比率月變化", sh_value, sh_score, 25))
    return combine(items)


# ───────────────────────── 技術面與進階指標 ─────────────────────────
def add_indicators(price):
    p = price.copy()
    rename = {"high": "max", "low": "min"}
    p = p.rename(columns={k: v for k, v in rename.items() if k in p.columns and v not in p.columns})
    required = ["date", "open", "max", "min", "close", "Trading_Volume"]
    if not set(required).issubset(p.columns):
        raise ValueError(f"股價資料缺少必要欄位：{sorted(set(required) - set(p.columns))}")
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    for col in ["open", "max", "min", "close", "Trading_Volume"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.dropna(subset=["date", "open", "max", "min", "close"]).sort_values("date")
    p = p.drop_duplicates("date", keep="last").reset_index(drop=True)

    c = p["close"]
    p["ma10"] = c.rolling(10, min_periods=10).mean()
    p["ma20"] = c.rolling(20, min_periods=20).mean()
    p["ma60"] = c.rolling(60, min_periods=60).mean()
    p["bb_std"] = c.rolling(20, min_periods=20).std()
    p["bb_up"] = p["ma20"] + 2 * p["bb_std"]
    p["bb_lo"] = p["ma20"] - 2 * p["bb_std"]
    p["vol_ma5"] = p["Trading_Volume"].rolling(5, min_periods=5).mean()
    p["vol_ma20"] = p["Trading_Volume"].rolling(20, min_periods=20).mean()

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = up / down.replace(0, np.nan)
    p["rsi"] = 100 - 100 / (1 + rs)
    p.loc[(down == 0) & (up > 0), "rsi"] = 100
    p.loc[(down == 0) & (up == 0), "rsi"] = 50

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    p["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        p["max"] - p["min"],
        (p["max"] - c.shift()).abs(),
        (p["min"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    p["atr"] = tr.rolling(14, min_periods=14).mean()
    return p


def score_technical(p):
    if p is None or len(p) < 61:
        return None, [{"指標": "技術面", "數值": "資料不足（至少需要61筆）", "分數": "N/A"}]
    x = p.iloc[-1]
    items = []

    vals = [x["close"], x["ma20"], x["ma60"], p["ma60"].iloc[-21]]
    if all(pd.notna(v) for v in vals):
        n = int(x["close"] > x["ma20"]) + int(x["ma20"] > x["ma60"]) + int(x["ma60"] > p["ma60"].iloc[-21])
        trend_score = n / 3 * 100
        items.append(("趨勢（價>MA20、MA20>MA60、MA60上彎）", f"{n}/3 條件成立", trend_score, 35))
    else:
        items.append(("趨勢（價>MA20、MA20>MA60、MA60上彎）", "資料不足", None, 35))

    rsi = safe_float(x["rsi"])
    rsi_score = float(np.interp(rsi, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5])) if np.isfinite(rsi) else None
    items.append(("RSI(14)", "資料不足" if rsi_score is None else f"{rsi:.0f}", rsi_score, 20))

    h, hp = safe_float(x["hist"]), safe_float(p.iloc[-2]["hist"])
    if np.isfinite(h) and np.isfinite(hp):
        if h > 0 and h > hp:
            macd_score = 100
        elif h > 0:
            macd_score = 60
        elif h > hp:
            macd_score = 50
        else:
            macd_score = 10
        items.append(("MACD 柱狀體", f"{h:+.2f}（{'增強' if h > hp else '減弱'}）", macd_score, 25))
    else:
        items.append(("MACD 柱狀體", "資料不足", None, 25))

    hi, lo = p["max"].tail(250).max(), p["min"].tail(250).min()
    c = float(x["close"])
    pos = (c - lo) / (hi - lo) if hi > lo else np.nan
    pos_score = float(np.interp(pos, [0, 0.2, 0.5, 0.8, 1], [40, 70, 80, 55, 30])) if np.isfinite(pos) else None
    items.append(("近250交易日位階", "資料不足" if pos_score is None else f"{pos:.0%}", pos_score, 20))
    return combine(items)


def volume_pattern_prefilter(p):
    """放量上漲、20日突破、低檔反彈、量縮守月線等量價快篩。"""
    if p is None or len(p) < 60:
        return {"pass": False, "pattern": "資料不足", "volume_ratio": np.nan, "low_position": np.nan}

    df = p if "ma20" in p.columns else add_indicators(p)
    x, prev = df.iloc[-1], df.iloc[-2]
    close = safe_float(x["close"])
    volume = safe_float(x["Trading_Volume"])
    vol_ma5 = safe_float(x["vol_ma5"])
    vol_ma20 = safe_float(x["vol_ma20"])
    high20 = safe_float(df["max"].iloc[-21:-1].max())
    low250 = safe_float(df["min"].tail(250).min())
    high250 = safe_float(df["max"].tail(250).max())
    volume_ratio = volume / vol_ma5 if np.isfinite(volume) and np.isfinite(vol_ma5) and vol_ma5 > 0 else np.nan
    low_position = (close - low250) / (high250 - low250) if high250 > low250 else np.nan

    patterns = []
    if np.isfinite(volume_ratio) and volume_ratio >= 1.5 and close > safe_float(prev["close"]):
        patterns.append("放量上漲")
    if np.isfinite(high20) and close > high20:
        patterns.append("20日突破")
    if (
        np.isfinite(low_position) and low_position <= 0.35
        and close > safe_float(prev["close"])
        and np.isfinite(volume_ratio) and volume_ratio >= 1.2
    ):
        patterns.append("低檔反彈")
    ma20 = safe_float(x["ma20"])
    if np.isfinite(ma20) and close >= ma20 and np.isfinite(vol_ma20) and volume < vol_ma20:
        patterns.append("量縮守月線")

    return {
        "pass": bool(patterns),
        "pattern": "、".join(patterns) if patterns else "未符合",
        "volume_ratio": round(volume_ratio, 2) if np.isfinite(volume_ratio) else np.nan,
        "low_position": round(low_position * 100, 1) if np.isfinite(low_position) else np.nan,
    }


# ───────────────────────── 歷史回測：成本、非重疊交易、樣本外 ─────────────────────────
def run_backtest(
    p,
    buy_threshold=70,
    hold_days=20,
    commission_rate=0.001425,
    commission_discount=0.6,
    sell_tax_rate=0.003,
    train_ratio=0.7,
    test_only=False,
):
    """
    訊號日收盤後判斷，下一交易日開盤進場，持有指定交易日後收盤出場。
    以固定本金、單筆全額投入的簡化交易報酬計算；不代表多股票組合回測。
    """
    if p is None or p.empty or len(p) < max(100, hold_days + 65):
        return pd.DataFrame()

    df = p.copy().sort_values("date").reset_index(drop=True)
    c = pd.to_numeric(df["close"], errors="coerce")
    o = pd.to_numeric(df["open"], errors="coerce")

    trend = (
        (c > df["ma20"]).astype(int)
        + (df["ma20"] > df["ma60"]).astype(int)
        + (df["ma60"] > df["ma60"].shift(20)).astype(int)
    ) / 3 * 100
    rsi_score = df["rsi"].apply(
        lambda value: float(np.interp(value, [0, 30, 45, 60, 70, 85, 100], [60, 75, 85, 70, 40, 15, 5]))
        if pd.notna(value) else np.nan
    )
    macd_score = pd.Series(np.where(df["hist"] > 0, 80.0, 40.0), index=df.index)
    tech_score = trend * 0.45 + rsi_score * 0.25 + macd_score * 0.30

    split_idx = int(len(df) * train_ratio)
    start_idx = max(60, split_idx) if test_only else 60
    end_idx = len(df) - hold_days - 1
    if start_idx >= end_idx:
        return pd.DataFrame()

    buy_fee = commission_rate * commission_discount
    sell_fee_and_tax = commission_rate * commission_discount + sell_tax_rate
    rows = []
    next_available_signal_idx = start_idx

    for i in range(start_idx, end_idx):
        if i < next_available_signal_idx:
            continue
        if pd.isna(tech_score.iloc[i]) or tech_score.iloc[i] < buy_threshold:
            continue

        entry_idx = i + 1
        exit_idx = entry_idx + hold_days - 1
        if exit_idx >= len(df):
            break

        entry_price = safe_float(o.iloc[entry_idx])
        exit_price = safe_float(c.iloc[exit_idx])
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
            continue

        gross = exit_price / entry_price - 1
        net = (exit_price * (1 - sell_fee_and_tax)) / (entry_price * (1 + buy_fee)) - 1
        window_low = pd.to_numeric(df.loc[entry_idx:exit_idx, "min"], errors="coerce").min()
        mdd = window_low / entry_price - 1 if pd.notna(window_low) else np.nan
        rows.append({
            "訊號日期": df.loc[i, "date"],
            "進場日期": df.loc[entry_idx, "date"],
            "出場日期": df.loc[exit_idx, "date"],
            "進場價": round(entry_price, 2),
            "出場價": round(exit_price, 2),
            "毛報酬(%)": round(gross * 100, 2),
            "扣成本報酬(%)": round(net * 100, 2),
            "持倉期間最大回撤(%)": round(mdd * 100, 2) if pd.notna(mdd) else np.nan,
            "樣本區段": "樣本外" if i >= split_idx else "樣本內",
        })
        next_available_signal_idx = exit_idx + 1

    return pd.DataFrame(rows)


def summarize_backtest(trades):
    if trades is None or trades.empty:
        return {
            "交易次數": 0, "勝率(%)": np.nan, "平均報酬(%)": np.nan,
            "累積報酬(%)": np.nan, "平均最大回撤(%)": np.nan,
        }
    returns = pd.to_numeric(trades["扣成本報酬(%)"], errors="coerce").dropna() / 100
    if returns.empty:
        return {
            "交易次數": len(trades), "勝率(%)": np.nan, "平均報酬(%)": np.nan,
            "累積報酬(%)": np.nan, "平均最大回撤(%)": np.nan,
        }
    return {
        "交易次數": int(len(returns)),
        "勝率(%)": round((returns > 0).mean() * 100, 2),
        "平均報酬(%)": round(returns.mean() * 100, 2),
        "累積報酬(%)": round(((1 + returns).prod() - 1) * 100, 2),
        "平均最大回撤(%)": round(pd.to_numeric(trades["持倉期間最大回撤(%)"], errors="coerce").mean(), 2),
    }


def benchmark_return(token, start_date, end_date, start_value=100000):
    """
    使用 0050 作為市場代理基準。若資料無法取得，回傳 None。
    比較區間以第一筆收盤至最後一筆收盤計算，未含股息。
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = max((end - start).days + 30, 90)
        b = get_dataset_safe("TaiwanStockPrice", "0050", days, token)
        if b.empty or not {"date", "close"}.issubset(b.columns):
            return None
        b["date"] = pd.to_datetime(b["date"], errors="coerce")
        b["close"] = pd.to_numeric(b["close"], errors="coerce")
        b = b.dropna(subset=["date", "close"]).sort_values("date")
        b = b[(b["date"] >= start) & (b["date"] <= end)]
        if len(b) < 2 or b["close"].iloc[0] <= 0:
            return None
        return (b["close"].iloc[-1] / b["close"].iloc[0] - 1) * 100
    except Exception:
        return None


# ───────────────────────── 進階買入評估與三階停利停損系統 ─────────────────────────
def evaluate_trade_plan(p, total, f_score, c_score, t_score, eps_ttm, fair_pe, high_pe):
    x = p.iloc[-1]
    c = safe_float(x["close"])
    atr = safe_float(x["atr"])
    ma10, ma20, ma60 = safe_float(x["ma10"]), safe_float(x["ma20"]), safe_float(x["ma60"])
    bb_up = safe_float(x["bb_up"])

    if not np.isfinite(c) or c <= 0:
        raise ValueError("最新收盤價無效，無法計算交易計畫。")
    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.03, 0.01)
    if not np.isfinite(ma10): ma10 = c
    if not np.isfinite(ma20): ma20 = c
    if not np.isfinite(ma60): ma60 = c
    if not np.isfinite(bb_up): bb_up = c + 2 * atr

    atr_stop = c - 2.0 * atr
    swing_low_20 = safe_float(p["min"].tail(20).min()) * 0.99
    ma60_stop = ma60 * 0.985
    candidate_stops = [v for v in [atr_stop, swing_low_20, ma60_stop] if np.isfinite(v) and v < c * 0.99]
    rec_stop = max(candidate_stops) if candidate_stops else atr_stop
    rec_stop = min(max(rec_stop, c * 0.90), c * 0.99)

    stop_details = {
        "波動率停損（現價 - 2×ATR）": round(atr_stop, 2),
        "型態支撐停損（近20日低點下緣）": round(swing_low_20, 2),
        "季線防守停損（60MA下緣1.5%）": round(ma60_stop, 2),
        "移動停利防守線（10MA短波段線）": round(ma10, 2),
    }

    hi_60 = safe_float(p["max"].tail(60).max())
    hi_120 = safe_float(p["max"].tail(120).max())
    lo_120 = safe_float(p["min"].tail(120).min())
    tp1_candidates = [v for v in [hi_60, bb_up] if np.isfinite(v) and v > c * 1.02]
    tp1 = min(tp1_candidates) if tp1_candidates else c + 2 * atr

    if not np.isfinite(hi_120): hi_120 = c
    if not np.isfinite(lo_120): lo_120 = c * 0.9
    fib_1272 = lo_120 + (hi_120 - lo_120) * 1.272
    pe_target_fair = eps_ttm * fair_pe if eps_ttm is not None and fair_pe is not None and eps_ttm > 0 and fair_pe > 0 else fib_1272
    tp2 = max(float(np.mean([fib_1272, pe_target_fair])), tp1 * 1.04)

    fib_1618 = lo_120 + (hi_120 - lo_120) * 1.618
    pe_target_high = eps_ttm * high_pe if eps_ttm is not None and high_pe is not None and eps_ttm > 0 and high_pe > 0 else fib_1618
    tp3 = max(float(np.mean([fib_1618, pe_target_high])), tp2 * 1.06)

    risk = max(c - rec_stop, 0.01)
    reward_tp2 = max(tp2 - c, 0)
    rr_ratio = reward_tp2 / risk
    bias_20 = (c / ma20 - 1) * 100 if ma20 > 0 else 0
    buy_zone_low = round(max(0.01, min(ma20, c - 0.8 * atr)), 2)
    buy_zone_high = round(min(c, ma10 * 1.01) if bias_20 > 5 else c * 1.005, 2)
    if buy_zone_high < buy_zone_low:
        buy_zone_low, buy_zone_high = buy_zone_high, buy_zone_low

    last5 = p.tail(5)
    up_vol = last5.loc[last5["close"] >= last5["open"], "Trading_Volume"].mean()
    dn_vol = last5.loc[last5["close"] < last5["open"], "Trading_Volume"].mean()
    vol_healthy = bool(pd.isna(dn_vol) or (pd.notna(up_vol) and up_vol > dn_vol) or (x["Trading_Volume"] < x["vol_ma20"] and c >= ma20))

    total_val = safe_float(total, 0)
    f_val, c_val, t_val = safe_float(f_score, 0), safe_float(c_score, 0), safe_float(t_score, 0)
    checklist = [
        {
            "檢核項目": "1. 綜合基本與趨勢評分",
            "狀態": "✅ 通過" if total is not None and total_val >= 65 else "⚠️ 偏弱",
            "當前數值": f"總分 {total_val:.0f} 分（基本面 {f_val:.0f} / 技術面 {t_val:.0f}）",
            "判斷標準": "總分 ≥ 65 分代表多面向具備優勢",
            "pass": bool(total is not None and total_val >= 65),
        },
        {
            "檢核項目": "2. 風險報酬比（以 TP2 計算）",
            "狀態": "✅ 通過" if rr_ratio >= 1.5 else "⚠️ 不佳",
            "當前數值": f"1 : {rr_ratio:.2f}（潛在獲利 +{reward_tp2/c*100:.1f}% / 停損風險 -{risk/c*100:.1f}%）",
            "判斷標準": "風報比 ≥ 1.5 倍才具備出手價值",
            "pass": bool(rr_ratio >= 1.5),
        },
        {
            "檢核項目": "3. 均線乖離與追高風險",
            "狀態": "✅ 通過" if (-2 <= bias_20 <= 7 and safe_float(x["rsi"], 100) <= 72) else "⚠️ 過熱或破線",
            "當前數值": f"月線乖離 {bias_20:+.1f}%，RSI 為 {safe_float(x['rsi'], 0):.0f}",
            "判斷標準": "月線乖離介於 -2% ~ +7% 且 RSI ≤ 72 不追高",
            "pass": bool(-2 <= bias_20 <= 7 and safe_float(x["rsi"], 100) <= 72),
        },
        {
            "檢核項目": "4. 籌碼面主力動向",
            "狀態": "✅ 通過" if c_score is not None and c_val >= 55 else "⚠️ 籌碼渙散/資料不足",
            "當前數值": f"籌碼得分 {c_val:.0f} 分",
            "判斷標準": "籌碼分 ≥ 55 代表法人或大戶偏多操作",
            "pass": bool(c_score is not None and c_val >= 55),
        },
        {
            "檢核項目": "5. 短期量價結構",
            "狀態": "✅ 通過" if vol_healthy else "⚠️ 出貨疑慮",
            "當前數值": "價漲量增 / 量縮守支撐" if vol_healthy else "下跌出量或高檔爆量收黑",
            "判斷標準": "攻擊時有量、拉回時量縮",
            "pass": bool(vol_healthy),
        },
    ]
    pass_count = sum(bool(item["pass"]) for item in checklist)

    if pass_count == 5:
        action = "🟢 強烈建議買進（五維度全數過關，仍須遵守風控）"
        action_level = "success"
    elif pass_count == 4 and rr_ratio >= 1.3:
        action = "🟢 適合分批買進（多數條件符合，可考慮分批）"
        action_level = "success"
    elif total is not None and total_val >= 70 and (bias_20 > 7 or rr_ratio < 1.3):
        action = f"🟡 標的優質但現價過高（可等待拉回 {buy_zone_low}～{buy_zone_high} 元）"
        action_level = "warning"
    elif pass_count >= 3:
        action = "🟡 條件好壞參半，建議小部位試單或觀望"
        action_level = "warning"
    else:
        action = "🔴 不建議此時買入（風報比不佳或趨勢偏弱）"
        action_level = "error"

    return {
        "action": action, "action_level": action_level, "pass_count": pass_count,
        "checklist": checklist, "rec_stop": round(rec_stop, 2),
        "stop_pct": round((rec_stop / c - 1) * 100, 2), "stop_details": stop_details,
        "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
        "rr_ratio": round(rr_ratio, 2), "buy_zone": (buy_zone_low, buy_zone_high),
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
    if len(p) < 61:
        return None
    x = p.iloc[-1]

    f_score, f_rows, eps_ttm, fair_pe, high_pe = score_fundamental(rev, fin, per)
    c_score, c_rows = score_chip(inst, margin, shareholding, price)
    t_score, t_rows = score_technical(p)
    scores = {"基本面": f_score, "籌碼面": c_score, "技術面": t_score}
    total, _ = combine([(key, "", value, W_DIM[key]) for key, value in scores.items()])
    plan = evaluate_trade_plan(p, total, f_score, c_score, t_score, eps_ttm, fair_pe, high_pe)
    prefilter = volume_pattern_prefilter(p)
    complete = data_completeness(scores)

    return {
        "code": str(code), "name": stock_name,
        "display_name": f"{code} {stock_name}".strip(),
        "close": float(x["close"]),
        "total": round(total, 1) if total is not None else 0,
        "f_score": round(f_score, 1) if f_score is not None else None,
        "c_score": round(c_score, 1) if c_score is not None else None,
        "t_score": round(t_score, 1) if t_score is not None else None,
        "data_completeness": round(complete, 1),
        "prefilter": prefilter, "plan": plan, "p_df": p,
        "f_rows": f_rows, "c_rows": c_rows, "t_rows": t_rows,
    }


# ───────────────────────── 排名 ─────────────────────────
def make_summary(results):
    rows = []
    for code, data in results.items():
        plan = data["plan"]
        pre = data.get("prefilter", {})
        close = data["close"]
        upside = (plan["tp2"] / close - 1) * 100 if close > 0 else np.nan
        rows.append({
            "代碼": code,
            "股名": data["name"] or "—",
            "現價": close,
            "總分": data["total"],
            "基本面": data["f_score"] if data["f_score"] is not None else "N/A",
            "籌碼面": data["c_score"] if data["c_score"] is not None else "N/A",
            "技術面": data["t_score"] if data["t_score"] is not None else "N/A",
            "資料完整度(%)": data.get("data_completeness", 0),
            "量價型態": pre.get("pattern", "未分析"),
            "量比": pre.get("volume_ratio", np.nan),
            "快篩通過": "是" if pre.get("pass") else "否",
            "檢核通過": f"{plan['pass_count']} / 5 項",
            "風報比": plan["rr_ratio"],
            "停損價": f"{plan['rec_stop']} ({plan['stop_pct']:+.1f}%)",
            "合理停利(TP2)": f"{plan['tp2']} ({upside:+.1f}%)",
            "操作建議": plan["action"].split("（")[0],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_快篩"] = df["快篩通過"].map({"是": 1, "否": 0})
    df = df.sort_values(
        ["_快篩", "總分", "資料完整度(%)", "風報比"],
        ascending=[False, False, False, False],
    ).drop(columns=["_快篩"])
    return df.reset_index(drop=True)


# ───────────────────────── 畫面呈現 ─────────────────────────
def main():
    st.set_page_config(page_title="台股量化買點決策與部位系統", page_icon="📈", layout="wide")
    st.title("📈 台股多面向批次評分、買點檢核與停利停損系統")
    st.caption("本工具為量化研究輔助，不構成投資建議。評分與回測均有資料及模型限制。")

    token = st.sidebar.text_input("FinMind Token（選填，填入提高限額）", type="password")
    info_map = fetch_stock_info(token)

    pool_choice = st.sidebar.selectbox("快速選取股票池範本", ["自訂輸入"] + list(STOCK_POOLS.keys()))
    default_codes = ", ".join(STOCK_POOLS[pool_choice]) if pool_choice != "自訂輸入" else "2330, 2454, 2317, 2382, 2603"
    raw_input = st.sidebar.text_area("股票代碼清單（逗號、空白或換行分隔）", default_codes, height=100)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 資金與風控參數設定")
    capital = st.sidebar.number_input("預計投入總本金（萬元）", min_value=5.0, max_value=10000.0, value=100.0, step=10.0)
    risk_pct = st.sidebar.slider("單筆交易最大可承受總本金虧損 (%)", 0.5, 5.0, 2.0, 0.5)
    st.sidebar.caption("回測成本可在回測頁籤調整；部位試算未包含滑價與跳空風險。")

    batch_btn = st.sidebar.button("🚀 開始批次評分與買點掃描", type="primary")
    codes = [c.strip() for c in raw_input.replace("\n", ",").replace(" ", ",").split(",") if c.strip()]
    codes = list(dict.fromkeys(codes))

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = {}

    if batch_btn and codes:
        st.session_state["batch_results"] = {}
        st.session_state.pop("selected_code", None)
        progress = st.progress(0)
        status = st.empty()
        for idx, code in enumerate(codes):
            name = info_map.get(code, "")
            status.text(f"正在掃描分析：{code} {name} ({idx + 1}/{len(codes)}) ...")
            try:
                result = calculate_stock_profile(code, name, token)
                if result:
                    st.session_state["batch_results"][code] = result
            except Exception as exc:
                st.warning(f"標的 {code} 分析失敗：{exc}")
            progress.progress((idx + 1) / len(codes))
            if not token:
                time.sleep(0.3)
        status.success("批次掃描完成！")
        progress.empty()

    results = st.session_state["batch_results"]
    if not results:
        st.info("👈 請在左側設定股票代碼，並點擊『🚀 開始批次評分與買點掃描』。")
        return

    summary_df = make_summary(results)
    if summary_df.empty:
        st.info("目前沒有可顯示的股票資料。")
        return

    if st.session_state.get("selected_code") not in results:
        st.session_state["selected_code"] = str(summary_df.iloc[0]["代碼"])

    st.subheader("📋 批次評分、量價快篩與風報比排行榜")
    st.caption("優先顯示通過量價快篩的標的，再依總分、資料完整度及風報比排序。點選個股按鈕可切換下方交易計畫。")
    show_prefilter_only = st.checkbox("只顯示通過量價快篩的個股", value=False)
    display_df = summary_df
    if show_prefilter_only:
        display_df = summary_df[summary_df["快篩通過"] == "是"]

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

    cols_def = [("個股", 2.5), ("現價", 1.0), ("總分", 0.7), ("完整度", 0.9), ("檢核", 0.7), ("風報比", 0.8), ("量價型態", 1.7), ("建議", 1.5)]
    widths = [width for _, width in cols_def]

    def make_cols():
        try:
            return st.columns(widths, vertical_alignment="center")
        except TypeError:
            return st.columns(widths)

    def cell(col, text, bold=False):
        value = f"<b>{text}</b>" if bold else str(text)
        col.markdown(f"<div style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{value}</div>", unsafe_allow_html=True)

    header = make_cols()
    for col, (name, _) in zip(header, cols_def):
        cell(col, name, True)

    for _, row in display_df.iterrows():
        code = str(row["代碼"])
        plan = results[code]["plan"]
        label = f"{code} {row['股名']}" if row["股名"] != "—" else code
        selected = code == st.session_state["selected_code"]
        cols = make_cols()
        if cols[0].button(label, key=f"pick_{code}", type="primary" if selected else "secondary"):
            st.session_state["selected_code"] = code
            st.rerun()
        cell(cols[1], f"{row['現價']:,.2f}")
        cell(cols[2], f"{row['總分']:.1f}")
        cell(cols[3], f"{row['資料完整度(%)']:.0f}%")
        cell(cols[4], f"{plan['pass_count']}/5")
        cell(cols[5], f"{row['風報比']:.2f}")
        cell(cols[6], row["量價型態"])
        cell(cols[7], short_action(plan["action"]))

    selected_code = st.session_state["selected_code"]
    if selected_code not in results:
        selected_code = str(summary_df.iloc[0]["代碼"])
        st.session_state["selected_code"] = selected_code
    st.write("---")

    d = results[selected_code]
    plan, p, cur_price = d["plan"], d["p_df"], d["close"]
    st.markdown(f"## 🎯 個股交易計畫：{d['display_name']}　（最新收盤價：**{cur_price:.2f} 元**）")
    getattr(st, plan["action_level"])(f"**買入評估結論：{plan['action']}**")

    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric("綜合總分", f"{d['total']:.1f} 分")
    m1.metric("基本面 (35%)", "N/A" if d["f_score"] is None else f"{d['f_score']:.1f} 分")
    m2.metric("籌碼面 (30%)", "N/A" if d["c_score"] is None else f"{d['c_score']:.1f} 分")
    m3.metric("技術面 (35%)", "N/A" if d["t_score"] is None else f"{d['t_score']:.1f} 分")
    m4.metric("檢核表通過數", f"{plan['pass_count']} / 5 項")

    st.markdown("#### 📌 實戰價位規劃（建議買入區、三段式停利、防守停損）")
    p1, p2, p3, p4, p5 = st.columns(5)
    buy_low, buy_high = plan["buy_zone"]
    p1.metric("理想買入區間", f"{buy_low} ~ {buy_high}", "拉回分批承接")
    p2.metric("🛑 建議停損價", f"{plan['rec_stop']} 元", f"{plan['stop_pct']:+.1f}%", delta_color="inverse")
    p3.metric("🎯 第一停利 (TP1)", f"{plan['tp1']} 元", f"{(plan['tp1']/cur_price-1)*100:+.1f}% (先出1/3)")
    p4.metric("🎯 第二停利 (TP2)", f"{plan['tp2']} 元", f"{(plan['tp2']/cur_price-1)*100:+.1f}% (再出1/3)")
    p5.metric("🚀 樂觀停利 (TP3)", f"{plan['tp3']} 元", f"{(plan['tp3']/cur_price-1)*100:+.1f}% (移動停利)")

    max_loss_ntd = capital * 10000 * risk_pct / 100
    loss_per_share = max(cur_price - plan["rec_stop"], 0.01)
    suggested_shares = int(max_loss_ntd // loss_per_share)
    max_affordable_shares = int((capital * 10000) // cur_price)
    final_shares = min(suggested_shares, max_affordable_shares)
    # 台股整股可用股數；若希望只買整張，改為 (final_shares // 1000) * 1000。
    position_value = final_shares * cur_price
    position_weight = position_value / (capital * 10000) * 100 if capital > 0 else 0

    with st.expander("⚖️ 單筆風險部位試算與停損防線細節", expanded=True):
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("**💡 資金控管建議（固定風險比例法）**")
            st.write(
                f"- 總本金 **{capital:.0f} 萬元**，單筆最大容忍虧損 **{risk_pct:.1f}%**（約 **{max_loss_ntd:,.0f} 元**）。\n"
                f"- 若於現價 **{cur_price:.2f} 元** 買進，跌至 **{plan['rec_stop']:.2f} 元** 停損，每股風險約 **{loss_per_share:.2f} 元**。\n"
                f"- 建議股數約 **{final_shares:,} 股**（**{final_shares/1000:.2f} 張**），投入約 **{position_value/10000:.1f} 萬元**（佔總資金 **{position_weight:.1f}%**）。"
            )
        with right:
            st.markdown("**🛡️ 各類停損防線參考價位**")
            for key, value in plan["stop_details"].items():
                diff = (value / cur_price - 1) * 100
                st.caption(f"• {key}：**{value:.2f} 元**（{diff:+.1f}%）")

    tab_check, tab_chart, tab_fund, tab_chip, tab_tech, tab_bt = st.tabs([
        "✅ 買入五維度檢核", "📈 K線與停利停損圖", "基本面明細", "籌碼面明細", "技術面明細", "🎯 歷史勝率回測"
    ])

    with tab_check:
        check_df = pd.DataFrame(plan["checklist"]).drop(columns=["pass"])
        st.dataframe(check_df, hide_index=True, use_container_width=True)

    with tab_chart:
        q = p.tail(150)
        dates = pd.to_datetime(q["date"])
        fig = go.Figure(go.Candlestick(
            x=dates, open=q["open"], high=q["max"], low=q["min"], close=q["close"], name="K線"
        ))
        fig.add_scatter(x=dates, y=q["ma20"], name="20MA（月線）", line=dict(color="orange", width=1.5))
        fig.add_scatter(x=dates, y=q["ma60"], name="60MA（季線）", line=dict(color="royalblue", width=1.5))
        fig.add_hline(y=plan["rec_stop"], line_dash="dash", line_color="red",
                      annotation_text=f"停損價 {plan['rec_stop']}", annotation_position="bottom right")
        fig.add_hline(y=plan["tp1"], line_dash="dot", line_color="seagreen",
                      annotation_text=f"TP1 {plan['tp1']}", annotation_position="top right")
        fig.add_hline(y=plan["tp2"], line_dash="dash", line_color="green",
                      annotation_text=f"TP2 {plan['tp2']}", annotation_position="top right")
        fig.add_hrect(y0=buy_low, y1=buy_high, fillcolor="lightgreen", opacity=0.15, line_width=0,
                      annotation_text="建議承接區", annotation_position="bottom left")
        fig.update_layout(xaxis_rangeslider_visible=False, height=480, margin=dict(l=0, r=0, t=15, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab_fund:
        st.dataframe(pd.DataFrame(d["f_rows"]), hide_index=True, use_container_width=True)
    with tab_chip:
        st.dataframe(pd.DataFrame(d["c_rows"]), hide_index=True, use_container_width=True)
    with tab_tech:
        st.dataframe(pd.DataFrame(d["t_rows"]), hide_index=True, use_container_width=True)

    with tab_bt:
        st.write(f"##### {d['display_name']} 歷史進場訊號前向報酬檢驗")
        col1, col2 = st.columns(2)
        threshold = col1.slider("回測進場分數門檻", 60, 85, 70, 5, key=f"th_{selected_code}")
        hold_days = col2.selectbox("進場後持有交易日數", [10, 20, 60], index=1, key=f"hd_{selected_code}")
        st.markdown("**交易成本參數**")
        c1, c2, c3 = st.columns(3)
        commission = c1.number_input("單邊手續費率", min_value=0.0, max_value=0.01, value=0.001425, step=0.0001, format="%.5f")
        discount = c2.number_input("手續費折扣", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        tax = c3.number_input("賣出證交稅率", min_value=0.0, max_value=0.02, value=0.003, step=0.001, format="%.3f")
        out_sample_only = st.checkbox("只顯示樣本外回測（前70%資料作訓練區段，後30%作測試區段）", value=True, key=f"oos_{selected_code}")
        bt = run_backtest(
            p, buy_threshold=threshold, hold_days=hold_days,
            commission_rate=commission, commission_discount=discount,
            sell_tax_rate=tax, train_ratio=0.7, test_only=out_sample_only,
        )
        if bt.empty:
            st.info("資料長度不足或該區間沒有符合門檻的交易訊號。可嘗試降低門檻或調整持有天數。")
        else:
            stats = summarize_backtest(bt)
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("交易次數", f"{stats['交易次數']} 次")
            w2.metric("持有勝率", "N/A" if pd.isna(stats["勝率(%)"]) else f"{stats['勝率(%)']:.1f}%")
            w3.metric("平均單筆淨報酬", "N/A" if pd.isna(stats["平均報酬(%)"]) else f"{stats['平均報酬(%)']:+.2f}%")
            w4.metric("交易複利累積報酬", "N/A" if pd.isna(stats["累積報酬(%)"]) else f"{stats['累積報酬(%)']:+.2f}%")
            st.caption(f"平均持倉期間最大回撤：{stats['平均最大回撤(%)']:.2f}%")
            if out_sample_only:
                st.caption("此表只列樣本外訊號；樣本切分依歷史資料筆數，不代表模型已經過最佳化驗證。")
            st.dataframe(bt, hide_index=True, use_container_width=True)

            benchmark = benchmark_return(token, bt["進場日期"].min(), bt["出場日期"].max())
            if benchmark is None:
                st.info("無法取得相同期間的 0050 資料，因此暫時無法比較基準報酬。")
            else:
                st.metric("0050 基準累積報酬（未含股息）", f"{benchmark:+.2f}%")
                st.caption("0050 是市場代理基準，期間以首末收盤價估算，未含股息；與策略的資金曝險及交易時點不同，僅供參考。")


if __name__ == "__main__":
    main()
