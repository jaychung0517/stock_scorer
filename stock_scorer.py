"""
台股多面向批次評分、快篩與「評分有效性回測」（Streamlit）— v4 修正版

相對 v3 的主要修正：
1. ROE：改抓資產負債表 TaiwanStockBalanceSheet（權益在這裡），用近五季平均權益計算
2. 大戶持股：FinMind 的 TaiwanStockHoldingSharesPer 需付費會員，免費 token 抓不到。
   改用集保結算所(TDCC)免費開放資料（只有最新一週），每次掃描自動存入本機
   holders_history.csv，累積滿 4 週後才產生趨勢分數；在此之前該指標顯示 N/A 並自動重新配權
3. 缺資料不再被當成 0 分：保留 N/A、顯示「有效指標 x/y」，並新增資料診斷面板
4. 回測：改成驗證「真正的總分」(基本面+籌碼面+技術面)，使用「當時已公布」的資料避免未來函數，
   多檔合併並與基準比較（高/低分組報酬、Spearman IC）
5. 其他：股名快取失敗不再被鎖 24 小時、支援全形逗號、偵測 FinMind 限流、
   目標價分開顯示並補回風險報酬比、相容新舊版 Streamlit 的寬度參數

v6 修正（依審查意見逐條查證後採納的部分）：
8. ROE：FinMind 損益表的 EquityAttributableToOwnersOfParent 其實是「綜合損益總額歸屬母公司」，不是淨利，
   改用 IncomeAfterTaxes ÷ 權益總額 Equity，並依季末日期對齊；兩者季度對不上就顯示 N/A，不硬算
9. 新高改為「收盤價突破前期最高收盤」（不含當日），避免窗口少算一天
10. 大戶持股歷史以 (日期, 代碼) 為唯一鍵合併；回測會帶入『當時已累積』的大戶歷史，與即時評分口徑一致
11. 資料完整度 <60% 標示低可信度；目標價改以區間呈現，風險報酬比用保守目標價（區間低端）計算
12. 回測新增「不重疊取樣」與逐年檢視；限流偵測加入 HTTP 429

v5 新增：
6. 量價型態快篩：爆量（今日量 ≥ 昨量／5日均量／60日均量的 N 倍）＋近期出現破底翻／創20日新高／創歷史新高，
   只用股價資料（每檔 1 次 API），可把通過的股票一鍵放進左側清單再做完整評分
7. 最近 5 次輸入的清單記錄（只存在本次連線；重新整理網頁會清空）

安裝：pip install streamlit pandas numpy requests plotly
執行：streamlit run stock_batch_scorer_v6.py
"""
import datetime as dt
import io
import os
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "https://api.finmindtrade.com/api/v4/data"
TDCC_URL = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
HOLDER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holders_history.csv")
W_DIM = {"基本面": 0.35, "籌碼面": 0.30, "技術面": 0.35}

# 各資料集抓取的歷史長度（天）。掃描與回測共用同一份快取，回測不需額外呼叫 API
DATASETS = {
    "price": "TaiwanStockPrice",
    "rev": "TaiwanStockMonthRevenue",
    "fin": "TaiwanStockFinancialStatements",
    "bs": "TaiwanStockBalanceSheet",
    "per": "TaiwanStockPER",
    "inst": "TaiwanStockInstitutionalInvestorsBuySell",
    "margin": "TaiwanStockMarginPurchaseShortSale",
}
HIST_DAYS = {"price": 1900, "rev": 2500, "fin": 2700, "bs": 2700, "per": 2500, "inst": 1900, "margin": 1900}

# 回測用的「資料公布延遲」假設（天）。月營收約次月 10 日前公布；季報 45 天、年報 90 天
LAG_REV = 40
LAG_Q, LAG_Y = 46, 91

# 損益表／資產負債表的欄位名稱候選（依序嘗試；若都對不上，到「資料診斷」查看實際欄位）
# 注意：損益表的 EquityAttributableToOwnersOfParent 是「綜合損益總額歸屬母公司」（含其他綜合損益），不是淨利。
# 淨利用 IncomeAfterTaxes（本期淨利，含非控制權益），搭配資產負債表的權益總額 Equity，兩者口徑一致。
NI_CANDS = ["IncomeAfterTaxes"]
EQ_CANDS = ["Equity", "TotalEquity"]

STOCK_POOLS = {
    "台股核心權值": ["2330", "2317", "2454", "2308", "2382", "2881", "2882"],
    "高股息與傳產指標": ["2603", "2002", "1101", "2412", "2886", "2891"],
    "AI伺服器與零組件": ["2382", "3231", "6669", "2376", "2356", "3017"],
}


def _ver():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


# 新版 Streamlit 用 width="stretch"，舊版用 use_container_width
STRETCH = {"width": "stretch"} if _ver() >= (1, 50) else {"use_container_width": True}


# ───────────────────────── 資料擷取 ─────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_stock_info(token: str) -> dict:
    params = {"dataset": "TaiwanStockInfo"}
    if token:
        params["token"] = token
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(j.get("msg", "API error"))
    out = {}
    for item in j.get("data", []):
        out.setdefault(item["stock_id"], item.get("stock_name", ""))
    if not out:
        raise RuntimeError("empty")
    return out


def fetch_stock_info(token: str) -> dict:
    """失敗時丟例外不會被快取，所以這裡捕捉後回傳空字典，下次重新整理仍會重試"""
    try:
        return _fetch_stock_info(token)
    except Exception:
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
        raise RuntimeError(f"{j.get('status')} {j.get('msg', 'API error')}")
    return pd.DataFrame(j["data"])


def safe_get(key, code, token):
    """回傳 (資料, 狀態文字)。失敗不再靜默：狀態會顯示在資料診斷面板"""
    try:
        df = fetch(DATASETS[key], code, HIST_DAYS[key], token)
        return df, (f"OK {len(df)} 筆" if len(df) else "無資料")
    except Exception as e:
        return pd.DataFrame(), f"失敗：{str(e)[:90]}"


def is_rate_limited(status: dict) -> bool:
    return any(("402" in s or "429" in s or "limit" in s.lower()) for s in status.values() if s.startswith("失敗"))


# ── 大戶持股（TDCC 免費開放資料，僅最新一週，自行累積歷史）──
def parse_tdcc_csv(text: str) -> pd.DataFrame:
    """集保戶股權分散表：持股分級 12~15 = 400 張以上（400,001 股以上）"""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.shape[1] < 6:
        raise ValueError("TDCC 欄位數不符")
    df = df.iloc[:, :6]
    df.columns = ["date", "stock_id", "level", "holders", "shares", "pct"]
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    big = df[df["level"].between(12, 15)]
    out = big.groupby(["date", "stock_id"], as_index=False)["pct"].sum()
    out.columns = ["date", "stock_id", "large_pct"]
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    out["stock_id"] = out["stock_id"].str.strip()
    return out.dropna()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_tdcc_large() -> pd.DataFrame:
    hdr = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(TDCC_URL, headers=hdr, timeout=60)
    if r.status_code != 200 or not r.content:
        r = requests.post(TDCC_URL, headers=hdr, timeout=60)
    r.raise_for_status()
    return parse_tdcc_csv(r.content.decode("utf-8-sig"))


def load_holder_history() -> pd.DataFrame:
    if os.path.exists(HOLDER_FILE):
        try:
            return pd.read_csv(HOLDER_FILE, dtype={"stock_id": str})
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "stock_id", "large_pct"])


def merge_holder_history(hist: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """以 (date, stock_id) 為唯一鍵合併：同一天不同股票的資料各自保留"""
    if snap is None or snap.empty:
        return hist
    if hist.empty:
        return snap.drop_duplicates(["date", "stock_id"]).reset_index(drop=True)
    comb = pd.concat([hist, snap], ignore_index=True).drop_duplicates(["date", "stock_id"], keep="last")
    return comb.sort_values(["date", "stock_id"]).reset_index(drop=True)


def update_holder_history(snap: pd.DataFrame) -> pd.DataFrame:
    hist = load_holder_history()
    comb = merge_holder_history(hist, snap)
    if len(comb) > len(hist):
        try:
            comb.to_csv(HOLDER_FILE, index=False)
        except OSError:
            pass  # 雲端環境檔案系統可能唯讀／重啟即清空
    return comb


# ───────────────────────── 工具函式 ─────────────────────────
def lin(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def combine(items):
    """items: [(指標, 數值文字, 分數或None, 權重)]；缺資料的項目自動重新配權"""
    rows = [{"指標": n, "數值": v, "分數": "N/A" if s is None else round(s)} for n, v, s, _ in items]
    valid = [(s, w) for _, _, s, w in items if s is not None]
    if not valid:
        return None, rows
    tw = sum(w for _, w in valid)
    return sum(s * w for s, w in valid) / tw, rows


def fmt(v, nd=0):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def pivot_statement(df):
    """財報長表 → 寬表（index=季末日，columns=科目），並加上『可取得日』供回測避免未來函數"""
    if df.empty or not {"date", "type", "value"} <= set(df.columns):
        return pd.DataFrame()
    f = df.pivot_table(index="date", columns="type", values="value", aggfunc="first").sort_index()
    dts = pd.to_datetime(f.index)
    lag = np.where(dts.month == 12, LAG_Y, LAG_Q)
    f["__avail__"] = (dts + pd.to_timedelta(lag, unit="D")).strftime("%Y-%m-%d")
    return f


def prepare(raw):
    d = dict(raw)
    rev = raw["rev"]
    if not rev.empty and "date" in rev:
        rev = rev.sort_values("date").copy()
        rev["avail"] = (pd.to_datetime(rev["date"]) + pd.Timedelta(days=LAG_REV)).dt.strftime("%Y-%m-%d")
    d["rev"] = rev
    d["fin"] = pivot_statement(raw["fin"])
    d["bs"] = pivot_statement(raw["bs"])
    for k in ("per", "inst", "margin"):
        df = raw[k]
        if not df.empty and "date" in df:
            df = df.sort_values("date").copy()
            df["avail"] = df["date"]
        d[k] = df
    return d


def cut(df, t):
    """只保留 t 當天『已公布』的資料"""
    if df is None or df.empty:
        return df
    key = "__avail__" if "__avail__" in df.columns else "avail"
    if key not in df.columns:
        return df
    return df[df[key] <= t]


# ───────────────────────── 基本面評分 ─────────────────────────
def calc_roe(fin, bs):
    """ROE(%) = 近四季本期淨利 ÷ 近五個季末權益平均。淨利與權益依『季末日期』對齊；對不上就回傳 None"""
    if fin is None or fin.empty or bs is None or bs.empty:
        return None, "資料不足"
    ni_col = next((c for c in NI_CANDS if c in fin and fin[c].notna().sum() >= 4), None)
    eq_col = next((c for c in EQ_CANDS if c in bs and bs[c].notna().sum() >= 5), None)
    if not ni_col or not eq_col:
        return None, "資料不足（找不到淨利或權益欄位，見資料診斷）"
    ni, eq = fin[ni_col].dropna(), bs[eq_col].dropna()
    last4 = list(ni.index[-4:])
    if (pd.to_datetime(last4[-1]) - pd.to_datetime(last4[0])).days > 300:
        return None, "資料不足（近四季財報季度不連續）"
    if last4[-1] not in eq.index:
        return None, "資料不足（最新一季財報與資產負債表季度未對齊）"
    eq5 = eq[eq.index <= last4[-1]].iloc[-5:]
    if len(eq5) < 5 or (eq5 <= 0).any():
        return None, "資料不足（權益資料不足五季或為負）"
    roe = float(ni.loc[last4].sum() / eq5.mean() * 100)
    if abs(roe) > 150:
        return None, f"異常（{roe:.0f}%），請檢查欄位與單位"
    return roe, f"{roe:.2f}%（{ni_col} ÷ {eq_col}）"

def score_fundamental(rev, fin, bs, per):
    items, eps_ttm, fair_pe = [], None, None

    s, v = None, "資料不足"
    if rev is not None and len(rev) >= 15 and "revenue" in rev:
        r = rev["revenue"].astype(float).values
        base = r[-15:-12].sum()
        if base > 0:
            yoy = r[-3:].sum() / base - 1
            s, v = lin(yoy, -0.10, 0.30), f"{yoy:+.1%}"
    items.append(("近3月營收年增", v, s, 20))

    eps_s, eps_v = None, "資料不足"
    gm_s, gm_v = None, "資料不足"
    roe_s, roe_v = None, "資料不足"
    if fin is not None and not fin.empty:
        if "EPS" in fin:
            eps = fin["EPS"].dropna()
            if len(eps) >= 4:
                eps_ttm = float(eps.iloc[-4:].sum())
            if len(eps) >= 8:
                prev = float(eps.iloc[-8:-4].sum())
                if prev > 0:
                    g = eps_ttm / prev - 1
                    eps_s, eps_v = lin(g, -0.2, 0.3), f"近四季EPS {eps_ttm:.2f}，年增 {g:+.1%}"
                else:
                    eps_s = 80 if eps_ttm > 0 else 20
                    eps_v = f"近四季EPS {eps_ttm:.2f}（基期為負或零）"
        if {"GrossProfit", "Revenue"} <= set(fin.columns):
            gm = (fin["GrossProfit"] / fin["Revenue"]).dropna()
            if len(gm) >= 5:
                d = gm.iloc[-1] - gm.iloc[-5]
                gm_s, gm_v = lin(d, -0.03, 0.03), f"{gm.iloc[-1]:.1%}（較去年同期 {d * 100:+.1f} 百分點）"

        # ROE：淨利與權益依季末日期對齊（見 calc_roe）
        roe, roe_v = calc_roe(fin, bs)
        if roe is not None:
            roe_s = lin(roe, 5, 20)
    items.append(("EPS 成長", eps_v, eps_s, 20))
    items.append(("毛利率趨勢", gm_v, gm_s, 15))
    items.append(("近四季 ROE", roe_v, roe_s, 20))

    pe_s, pe_v, dy_s, dy_v = None, "資料不足", None, "資料不足"
    if per is not None and not per.empty and {"PER", "dividend_yield"} <= set(per.columns):
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
def score_chip(inst, margin, holders, price):
    items = []
    vol = price.set_index("date")["Trading_Volume"].astype(float)

    def ratio(names):
        if inst is None or inst.empty or "name" not in inst:
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
    if margin is not None and not margin.empty and "MarginPurchaseTodayBalance" in margin:
        b = margin["MarginPurchaseTodayBalance"].astype(float).values
        if len(b) >= 21 and b[-21] > 0:
            chg = b[-1] / b[-21] - 1
            m_s, m_v = lin(-chg, -0.15, 0.15), f"{chg:+.1%}"
    items.append(("融資餘額20日變化（減少較佳）", m_v, m_s, 20))

    # 大戶(400張以上)持股比：需本機累積 ≥4 週歷史才有趨勢
    h_s, h_v = None, "無資料（TDCC 尚未下載或不含此股）"
    if holders is not None and not holders.empty:
        hs = holders.sort_values("date").copy()
        hs["dt"] = pd.to_datetime(hs["date"])
        last = hs.iloc[-1]
        old = hs[hs["dt"] <= last["dt"] - pd.Timedelta(days=25)]
        if not old.empty:
            diff = float(last["large_pct"] - old.iloc[-1]["large_pct"])
            h_s, h_v = lin(diff, -1.5, 1.5), f"{last['large_pct']:.2f}%（近4週 {diff:+.2f} 個百分點）"
        else:
            span = (last["dt"] - hs.iloc[0]["dt"]).days
            h_v = f"{last['large_pct']:.2f}%（已累積 {span} 天，滿 25 天後才計趨勢分）"
    items.append(("400張以上大戶持股比變化", h_v, h_s, 25))
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
    if len(p) < 80:
        return None, []
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
    if total is None:
        return "⚪ 資料不足"
    c = x["close"]
    extended = c / x["ma20"] - 1 > 0.08 or x["rsi"] > 70
    pullback = x["ma60"] * 0.98 <= c <= x["ma20"] * 1.02
    if f_score is not None and f_score >= 65 and pullback and total >= 55:
        return "🟢 回測支撐買點"
    if total >= 70 and not extended:
        return "🟢 分批佈局區"
    if total >= 70:
        return "🟡 短線過熱等拉回"
    if total >= 50:
        return "🟡 中性觀望"
    return "🔴 偏弱觀望"


def targets(p, eps_ttm, fair_pe):
    x = p.iloc[-1]
    c = x["close"]
    out = {}
    if eps_ttm and eps_ttm > 0 and fair_pe:
        out["本益比估值（近四季EPS×歷史中位本益比）"] = eps_ttm * fair_pe
    hi, lo = p["max"].tail(120).max(), p["min"].tail(120).min()
    out["形態目標價（前高／1.272延伸）"] = hi if hi > c * 1.03 else lo + (hi - lo) * 1.272
    atr_stop = c - 2 * x["atr"]
    stop = max(atr_stop, x["ma60"]) if x["ma60"] < c else atr_stop
    return out, stop


# ───────────────────────── 單日評分快照（即時與回測共用）─────────────────────────
def snapshot(d, p, i, live=False, holders=None):
    """以 p 的第 i 列為評分日。live=True 用 API 現有全部資料；否則只用當日『已公布』資料"""
    t = "9999-12-31" if live else p["date"].iloc[i]
    pi = p.iloc[: i + 1]
    f, f_rows, eps_ttm, fair_pe = score_fundamental(cut(d["rev"], t), cut(d["fin"], t), cut(d["bs"], t), cut(d["per"], t))
    c, c_rows = score_chip(cut(d["inst"], t), cut(d["margin"], t), holders, pi)
    tt, t_rows = score_technical(pi)
    scores = {"基本面": f, "籌碼面": c, "技術面": tt}
    total, _ = combine([(k, "", s, W_DIM[k]) for k, s in scores.items()])
    return {"scores": scores, "total": total, "rows": (f_rows, c_rows, t_rows), "eps_ttm": eps_ttm, "fair_pe": fair_pe}


def calculate_stock_profile(code, name, token, holders_hist):
    raw, status = {}, {}
    raw["price"], status["price"] = safe_get("price", code, token)
    if len(raw["price"]) < 80:
        return {"code": code, "error": "股價資料不足（代碼錯誤、新上市，或被限流）", "status": status,
                "rate_limited": is_rate_limited(status)}
    for k in ["rev", "fin", "bs", "per", "inst", "margin"]:
        raw[k], status[k] = safe_get(k, code, token)

    d = prepare(raw)
    p = add_indicators(raw["price"])
    h = pd.DataFrame()
    if holders_hist is not None and not holders_hist.empty:
        h = holders_hist[holders_hist["stock_id"] == code][["date", "large_pct"]]
    s = snapshot(d, p, len(p) - 1, live=True, holders=h)
    x = p.iloc[-1]
    tg, stop = targets(p, s["eps_ttm"], s["fair_pe"])
    vals = list(tg.values())
    mid = float(np.mean(vals))  # 兩種估值假設不同，平均僅供參考，不用來算風險報酬
    low_t = float(min(vals))    # 保守目標價（區間低端）
    risk = x["close"] - stop
    rr = (low_t - x["close"]) / risk if low_t > x["close"] and risk > 0 else None

    all_rows = [r for rows in s["rows"] for r in rows]
    valid = sum(1 for r in all_rows if r["分數"] != "N/A")
    return {
        "code": code, "name": name, "display_name": f"{code} {name}".strip(),
        "close": x["close"], "total": s["total"],
        "f_score": s["scores"]["基本面"], "c_score": s["scores"]["籌碼面"], "t_score": s["scores"]["技術面"],
        "verdict": verdict(s["total"], s["scores"]["基本面"], x),
        "target": mid, "low_target": low_t, "stop": stop, "rr": rr, "targets_dict": tg,
        "valid": valid, "total_items": len(all_rows),
        "p_df": p, "rows": s["rows"], "data": d, "status": status,
        "rate_limited": is_rate_limited(status),
    }


# ───────────────────────── 評分有效性回測 ─────────────────────────
def backtest_stock(res, step=5, horizons=(20, 60), start_i=130, holders_hist=None):
    """每隔 step 個交易日計算一次當時的評分，記錄『隔日收盤進場』後 20/60 日報酬"""
    d, p = res["data"], res["p_df"]
    n = len(p)
    rows = []
    hh_all = pd.DataFrame()
    if holders_hist is not None and not holders_hist.empty:
        hh_all = holders_hist[holders_hist["stock_id"] == res["code"]][["date", "large_pct"]]
    for i in range(start_i, n - 1 - min(horizons), step):
        x = p.iloc[i]
        h = None
        if not hh_all.empty:  # 只用當時已累積、且至少 7 天前的大戶資料，絕不回填未來資料
            cutoff = (pd.Timestamp(x["date"]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            h = hh_all[hh_all["date"] <= cutoff]
        s = snapshot(d, p, i, holders=h)
        if s["total"] is None:
            continue
        entry = p["close"].iloc[i + 1]
        row = {
            "代碼": res["code"], "日期": x["date"], "總分": s["total"],
            "基本面": s["scores"]["基本面"], "籌碼面": s["scores"]["籌碼面"], "技術面": s["scores"]["技術面"],
            "判斷": verdict(s["total"], s["scores"]["基本面"], x),
        }
        for h in horizons:
            j = i + 1 + h
            row[f"ret{h}"] = p["close"].iloc[j] / entry - 1 if j < n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def group_table(sub, key, col, base):
    g = sub.groupby(key, observed=True)[col]
    out = pd.DataFrame({"樣本數": g.size(), "平均報酬": g.mean(), "中位數": g.median(), "勝率": g.apply(lambda s: (s > 0).mean())})
    out["超額報酬"] = out["平均報酬"] - base
    return out


def show_table(df):
    show = df.copy()
    for c in ["平均報酬", "中位數", "超額報酬"]:
        show[c] = show[c].map(lambda v: f"{v:+.2%}")
    show["勝率"] = show["勝率"].map(lambda v: f"{v:.1%}")
    st.dataframe(show.reset_index(), hide_index=True, **STRETCH)


# ───────────────────────── 量價型態快篩（只用股價資料，每檔 1 次 API）─────────────────────────
VOL_BASES = ["昨量", "5日均量", "60日均量"]
PATTERNS = ["破底翻", "創20日新高", "創歷史新高"]


def parse_codes(raw: str) -> list:
    return list(dict.fromkeys(c for c in re.split(r"[,\s，、;；]+", raw or "") if c))


def remember_codes(codes):
    """記錄最近 5 次輸入的清單（只存在這次連線的 session，重新整理網頁就會清空）"""
    if not codes:
        return
    s = ", ".join(codes)
    lst = [x for x in st.session_state.get("recent_lists", []) if x != s]
    st.session_state["recent_lists"] = ([s] + lst)[:5]


def remember_current():  # 當按鈕的 on_click：在畫面重繪前先更新，下拉選單才會立刻出現新紀錄
    remember_codes(parse_codes(st.session_state.get("codes_text", "")))


def apply_pool():
    p = st.session_state.get("pool_choice")
    if p in STOCK_POOLS:
        st.session_state["codes_text"] = ", ".join(STOCK_POOLS[p])


def apply_recent():
    r = st.session_state.get("recent_choice")
    if r and r != "—":
        st.session_state["codes_text"] = r
    st.session_state["recent_choice"] = "—"


def set_codes(codes):
    st.session_state["codes_text"] = ", ".join(codes)


def volume_ratio_table(p):
    """每日成交量 ÷ 基準；基準皆『不含當日』：前一日量、前5日均量、前60日均量"""
    v = p["Trading_Volume"].astype(float).reset_index(drop=True)
    prev = v.shift(1)
    return pd.DataFrame({
        "昨量": v / prev.replace(0, np.nan),
        "5日均量": v / prev.rolling(5).mean().replace(0, np.nan),
        "60日均量": v / prev.rolling(60).mean().replace(0, np.nan),
    })


def detect_spring(p, lookback=5, window=60, k=5):
    """破底翻：收盤跌破『前 window 日最低價』後，k 日內收盤站回該前低之上，
    站回日在近 lookback 日內，且最新收盤仍在前低之上"""
    close = p["close"].astype(float).values
    low = p["min"].astype(float).values
    n = len(close)
    support = pd.Series(low).shift(1).rolling(window).min().values
    for b in range(n - 2, max(window, n - 1 - lookback - k) - 1, -1):
        if not close[b] < support[b]:  # support 為 NaN 時比較為 False
            continue
        ref = support[b]
        for d in range(b + 1, min(b + k, n - 1) + 1):
            if close[d] > ref:
                if d >= n - lookback and close[n - 1] > ref:
                    return {"ref": float(ref), "b": p["date"].iloc[b], "d": p["date"].iloc[d]}
                break
    return None


def detect_patterns(p, selected, lookback=5, window=60):
    c = p["close"].astype(float).reset_index(drop=True)
    dates = p["date"].reset_index(drop=True)
    n = len(c)
    found = {}

    def last_true(mask):
        idx = [i for i in range(max(0, n - lookback), n) if bool(mask.iloc[i])]
        return dates.iloc[idx[-1]] if idx else None

    if "創20日新高" in selected:
        d = last_true(c > c.shift(1).rolling(20).max())  # 收盤突破前 20 日最高收盤（不含當日）
        if d:
            found["創20日新高"] = f"創20日新高（{d[5:]}）"
    if "創歷史新高" in selected and n >= 250:  # 以抓到的約 5 年資料為準，且為未還原權值價格
        d = last_true(c > c.shift(1).cummax())  # 收盤突破資料期間內所有前期最高收盤
        if d:
            found["創歷史新高"] = f"創資料期間新高（{d[5:]}）"
    if "破底翻" in selected and n > window + 10:
        s = detect_spring(p, lookback, window)
        if s:
            found["破底翻"] = f"破底翻（前{window}日低 {s['ref']:.1f}，{s['b'][5:]}跌破→{s['d'][5:]}站回）"
    return found


def screen_stock(p, bases, mult, patterns, lookback, window, relax_volume):
    """條件：(任一量能基準 ≥ mult 倍) 且 (任一選定型態在近 lookback 日內出現)"""
    if len(p) < 70:
        return None
    ratios = volume_ratio_table(p)
    win = ratios.tail(lookback if relax_volume else 1)
    hits = {}
    for b in bases:
        col = win[b].dropna()
        if not col.empty and col.max() >= mult:
            hits[b] = float(col.max())
    pats = detect_patterns(p, patterns, lookback, window)
    last = ratios.iloc[-1]
    c = p["close"].astype(float)
    return {
        "close": float(c.iloc[-1]), "chg": float(c.iloc[-1] / c.iloc[-2] - 1),
        "r_prev": last["昨量"], "r_5": last["5日均量"], "r_60": last["60日均量"],
        "vol_hits": hits, "patterns": pats, "passed": bool(hits) and bool(pats),
    }


# ───────────────────────── 畫面 ─────────────────────────
def render_detail(d):
    p = d["p_df"]
    x = p.iloc[-1]
    st.markdown(f"### 🏢 {d['display_name']}　現價：**{x['close']:.2f} 元**　{d['verdict']}")
    m = st.columns(4)
    m[0].metric("總分", fmt(d["total"]))
    m[1].metric("基本面 (35%)", fmt(d["f_score"]))
    m[2].metric("籌碼面 (30%)", fmt(d["c_score"]))
    m[3].metric("技術面 (35%)", fmt(d["t_score"]))
    st.caption(f"有效指標 {d['valid']}/{d['total_items']}（N/A 的指標會自動排除並重新配權）")
    if d["valid"] / d["total_items"] < 0.6:
        st.warning("⚠️ 有效指標不足 60%，總分可信度低，請勿與資料完整的股票直接比較。")

    tg = d["targets_dict"]
    vals = list(tg.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目標價區間", f"{min(vals):.1f}–{max(vals):.1f}")
    c2.metric("保守目標價（低端）", f"{d['low_target']:.1f}", f"{(d['low_target'] / x['close'] - 1) * 100:+.1f}%")
    c3.metric("停損參考", f"{d['stop']:.1f}", f"{(d['stop'] / x['close'] - 1) * 100:+.1f}%")
    c4.metric("風險報酬比（保守目標）", "—" if d["rr"] is None else f"{d['rr']:.1f}")
    for k, v in tg.items():
        st.caption(f"{k}：{v:.1f}")

    t1, t2, t3, t4, t5 = st.tabs(["基本面明細", "籌碼面明細", "技術面明細", "K線型態", "資料診斷"])
    for tab, rows in zip([t1, t2, t3], d["rows"]):
        tab.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    q = p.tail(180)
    dates = pd.to_datetime(q["date"])
    fig = go.Figure(go.Candlestick(x=dates, open=q["open"], high=q["max"], low=q["min"], close=q["close"], name="K線"))
    fig.add_scatter(x=dates, y=q["ma20"], name="MA20", line=dict(color="orange", width=1.5))
    fig.add_scatter(x=dates, y=q["ma60"], name="MA60", line=dict(color="blue", width=1.5))
    fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=0, r=0, t=10, b=0))
    t4.plotly_chart(fig, **STRETCH)
    with t5:
        st.dataframe(pd.DataFrame([{"資料集": DATASETS[k], "狀態": v} for k, v in d["status"].items()]), hide_index=True, **STRETCH)
        for k, label in [("fin", "損益表欄位"), ("bs", "資產負債表欄位")]:
            df = d["data"][k]
            cols = [c for c in df.columns if c != "__avail__"] if not df.empty else []
            st.caption(f"{label}：" + (", ".join(map(str, cols)) if cols else "（無）"))
        st.caption("若 ROE 顯示找不到欄位，請從上面的欄位名稱挑出『淨利』與『權益』，加到檔案開頭的 NI_CANDS / EQ_CANDS。")


def render_backtest(ok_results):
    st.subheader("📊 評分有效性回測")
    st.caption(
        "每 5 個交易日計算一次『當時的總分』（只用當時已公布的資料），隔日收盤進場，看之後 20/60 日報酬。"
        "若高分組沒有明顯優於低分組、IC 接近 0，代表目前的評分規則沒有預測力，應調整權重與門檻。"
    )
    if not ok_results:
        st.info("請先在左側執行批次掃描。")
        return
    c1, c2 = st.columns(2)
    h = c1.selectbox("持有交易日數", [20, 60], index=0)
    codes = list(ok_results.keys())
    scope = c2.selectbox("統計範圍", ["全部掃描標的"] + codes, format_func=lambda c: c if c == "全部掃描標的" else ok_results[c]["display_name"])

    if st.button("執行回測"):
        frames, bar = [], st.progress(0)
        for k, code in enumerate(codes):
            frames.append(backtest_stock(ok_results[code], holders_hist=st.session_state.get("holder_hist")))
            bar.progress((k + 1) / len(codes))
        bar.empty()
        st.session_state["bt_df"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    bt = st.session_state.get("bt_df")
    if bt is None:
        return
    if bt.empty:
        st.warning("沒有足夠的歷史資料可回測。")
        return
    if scope != "全部掃描標的":
        bt = bt[bt["代碼"] == scope]
    col = f"ret{h}"
    sub = bt.dropna(subset=[col]).copy()
    if st.checkbox("不重疊取樣（每檔每隔持有期取一次：樣本較少，但彼此較獨立）", value=False):
        k = max(1, h // 5)
        sub = sub.sort_values(["代碼", "日期"])
        sub = sub[sub.groupby("代碼").cumcount() % k == 0]
    if len(sub) < 30:
        st.warning(f"可用樣本只有 {len(sub)} 筆，統計意義有限。")
        if sub.empty:
            return
    base = sub[col].mean()
    ic = sub["總分"].rank().corr(sub[col].rank())  # 等同 Spearman，且不需要 scipy
    sub["分組"] = pd.qcut(sub["總分"].rank(method="first"), 3, labels=["低分組", "中分組", "高分組"])
    t_score = group_table(sub, "分組", col, base)
    spread = t_score.loc["高分組", "平均報酬"] - t_score.loc["低分組", "平均報酬"]

    m = st.columns(4)
    m[0].metric("樣本數", f"{len(sub)}")
    m[1].metric("Spearman IC", f"{ic:+.3f}")
    m[2].metric("基準（全樣本平均）", f"{base:+.2%}")
    m[3].metric("高分組 − 低分組", f"{spread:+.2%}")

    st.markdown("**依總分三等分**")
    show_table(t_score)
    fig = go.Figure(go.Bar(x=list(t_score.index), y=(t_score["平均報酬"] * 100).tolist()))
    fig.add_hline(y=base * 100, line_dash="dash", annotation_text="基準")
    fig.update_layout(height=280, yaxis_title=f"平均{h}日報酬(%)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, **STRETCH)

    st.markdown("**逐年檢視**（看是否只在特定年份有效）")
    sub["年"] = sub["日期"].str[:4]
    yr = []
    for y, g in sub.groupby("年"):
        hi, lo = g.loc[g["分組"] == "高分組", col].mean(), g.loc[g["分組"] == "低分組", col].mean()
        yr.append({
            "年": y, "樣本數": len(g),
            "IC": g["總分"].rank().corr(g[col].rank()) if len(g) >= 10 else np.nan,
            "高分組−低分組": hi - lo if pd.notna(hi) and pd.notna(lo) else np.nan,
        })
    yd = pd.DataFrame(yr)
    yd["IC"] = yd["IC"].map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}")
    yd["高分組−低分組"] = yd["高分組−低分組"].map(lambda v: "—" if pd.isna(v) else f"{v:+.2%}")
    st.dataframe(yd, hide_index=True, **STRETCH)

    st.markdown("**依買點判斷分組**")
    show_table(group_table(sub, "判斷", col, base))
    st.caption(
        "解讀提醒：①取樣間隔 5 天而持有 20/60 天，樣本彼此重疊，實際有效樣本比顯示的少；"
        "②標的只限你掃描的股票，且是現存股票（有存活者偏差）；③未計手續費與稅；"
        "④大戶持股只在『已累積歷史』的期間才計入，更早的期間籌碼面僅含法人與融資；⑤權重與門檻若依這份結果反覆調整，會變成過度配適，"
        "建議保留最近一段期間不參與調參，另做驗證。"
    )


def render_screener(codes, token, info_map):
    st.subheader("🔎 量價型態快篩")
    st.caption(
        "只用股價資料（每檔 1 次 API），從左側清單中篩出「爆量 ＋ 近期出現指定型態」的股票。"
        "量能基準皆不含當日；型態以收盤價判斷。新高＝收盤價突破前期最高收盤（不含當日）；『歷史新高』以抓到的約 5 年資料為準，價格未還原除權息。"
    )
    c1, c2 = st.columns(2)
    bases = c1.multiselect("量能基準（任一成立）", VOL_BASES, default=VOL_BASES)
    mult = c2.number_input("爆量倍數（今日量 ÷ 基準 ≥）", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    pats = st.multiselect("型態（任一成立）", PATTERNS, default=PATTERNS)
    c3, c4, c5 = st.columns(3)
    lookback = c3.slider("型態需出現在近 N 個交易日內", 1, 20, 5)
    window = c4.selectbox("破底翻：跌破的前低視窗（日）", [20, 60], index=1)
    relax = c5.checkbox("量能也放寬到近 N 日內任一天", value=False)

    if st.button("開始快篩", type="primary", on_click=remember_current):
        if not codes or not bases or not pats:
            st.warning("請先在左側輸入股票代碼，並至少選一個量能基準與一個型態。")
        else:
            rows, bar, box = [], st.progress(0), st.empty()
            for i, code in enumerate(codes):
                box.text(f"快篩中：{code} {info_map.get(code, '')} ({i + 1}/{len(codes)})")
                price, status = safe_get("price", code, token)
                if is_rate_limited({"price": status}):
                    st.error("偵測到 FinMind 呼叫上限，後續標的未掃描。請稍後再試，或減少標的數量。")
                    break
                if len(price) >= 70:
                    try:
                        r = screen_stock(price, bases, mult, pats, lookback, window, relax)
                    except Exception as e:
                        r = None
                        st.warning(f"{code} 計算失敗：{str(e)[:60]}")
                    if r:
                        r.update({"code": code, "name": info_map.get(code, "")})
                        rows.append(r)
                bar.progress((i + 1) / len(codes))
                time.sleep(0.2)
            box.empty()
            bar.empty()
            st.session_state["screen_rows"] = rows

    rows = st.session_state.get("screen_rows")
    if rows is None:
        return
    if not rows:
        st.info("沒有可用的資料（資料不足 70 個交易日或被限流）。")
        return
    df = pd.DataFrame([{
        "通過": "✅" if r["passed"] else "",
        "代碼": r["code"], "股名": r["name"] or "—", "收盤": r["close"], "漲跌幅": f"{r['chg']:+.1%}",
        "量比(昨量)": r["r_prev"], "量比(5日均)": r["r_5"], "量比(60日均)": r["r_60"],
        "達標量能": "、".join(r["vol_hits"]) or "—",
        "型態": "；".join(r["patterns"].values()) or "—",
        "_s": r["r_5"] if pd.notna(r["r_5"]) else 0,
    } for r in rows])
    df = df.sort_values(["通過", "_s"], ascending=[False, False]).drop(columns="_s")
    passed = df[df["通過"] == "✅"]["代碼"].tolist()
    st.markdown(f"**通過 {len(passed)} 檔 ／ 共掃描 {len(df)} 檔**")
    only = st.checkbox("只顯示通過者", value=True)
    show = df[df["通過"] == "✅"] if only else df
    st.dataframe(
        show, hide_index=True, **STRETCH,
        column_config={
            "收盤": st.column_config.NumberColumn(format="%.2f"),
            "量比(昨量)": st.column_config.NumberColumn(format="%.1f"),
            "量比(5日均)": st.column_config.NumberColumn(format="%.1f"),
            "量比(60日均)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    if passed:
        st.button("📥 把通過的股票放進左側清單（再做完整評分）", on_click=set_codes, args=(passed,))


def main():
    st.set_page_config(page_title="台股多面向批次評分與回測", page_icon="📈", layout="wide")
    st.title("📈 台股多面向批次評分、快篩與回測")

    token = st.sidebar.text_input("FinMind Token（選填，填入提高限額）", type="password")
    info_map = fetch_stock_info(token)
    st.session_state.setdefault("codes_text", "2330, 2454, 2317, 2382, 2603")
    st.sidebar.selectbox("快速選取股票池範本", ["自訂輸入"] + list(STOCK_POOLS.keys()), key="pool_choice", on_change=apply_pool)
    recents = st.session_state.get("recent_lists", [])
    if recents:
        st.sidebar.selectbox(
            "最近輸入（本次連線最近 5 次）", ["—"] + recents, key="recent_choice", on_change=apply_recent,
            format_func=lambda s: s if s == "—" else f"{len(parse_codes(s))} 檔：{s[:22]}{'…' if len(s) > 22 else ''}",
        )
    raw_input = st.sidebar.text_area("股票代碼清單（逗號、空白、換行或全形逗號分隔）", key="codes_text", height=120)
    batch_btn = st.sidebar.button("🚀 開始批次評分掃描", type="primary", on_click=remember_current)
    if "holder_weeks" in st.session_state:
        st.sidebar.caption(f"大戶持股：本機已累積 {st.session_state['holder_weeks']} 週資料（需跨 4 週才有趨勢分）")

    codes = parse_codes(raw_input)

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = {}

    if batch_btn and codes:
        st.session_state["batch_results"] = {}
        st.session_state.pop("bt_df", None)
        try:
            hist = update_holder_history(fetch_tdcc_large())
        except Exception as e:
            hist = load_holder_history()
            st.sidebar.warning(f"TDCC 大戶資料下載失敗：{str(e)[:60]}")
        st.session_state["holder_weeks"] = hist["date"].nunique()
        st.session_state["holder_hist"] = hist

        bar, box = st.progress(0), st.empty()
        for idx, code in enumerate(codes):
            name = info_map.get(code, "")
            box.text(f"正在掃描：{code} {name} ({idx + 1}/{len(codes)}) ...")
            try:
                res = calculate_stock_profile(code, name, token, hist)
            except Exception as e:
                res = {"code": code, "error": f"計算失敗：{e}", "rate_limited": False}
            st.session_state["batch_results"][code] = res
            bar.progress((idx + 1) / len(codes))
            if res.get("rate_limited"):
                st.error("偵測到 FinMind 呼叫上限，後續標的未掃描。請稍後再試，或減少標的數量。")
                break
            time.sleep(0.2)
        box.empty()
        bar.empty()

    results = st.session_state["batch_results"]
    ok = {c: r for c, r in results.items() if not r.get("error")}
    failed = {c: r["error"] for c, r in results.items() if r.get("error")}

    tab_main, tab_scr, tab_bt = st.tabs(["📋 批次排行與個股診斷", "🔎 量價型態快篩", "📊 評分有效性回測"])

    with tab_main:
        if failed:
            st.warning("以下標的未納入：" + "；".join(f"{c}（{e}）" for c, e in failed.items()))
        if not ok:
            st.info("👈 請在左側輸入股票代碼清單，並點擊『🚀 開始批次評分掃描』。")
        else:
            rows = []
            for c, r in ok.items():
                up = (r["low_target"] / r["close"] - 1) * 100
                cov = r["valid"] / r["total_items"]
                rows.append({
                    "代碼": r["code"], "股名": r["name"] or "—", "現價": r["close"],
                    "總分": r["total"], "基本面": r["f_score"], "籌碼面": r["c_score"], "技術面": r["t_score"],
                    "有效指標": f"{r['valid']}/{r['total_items']}", "可信度": "高" if cov >= 0.8 else "中" if cov >= 0.6 else "⚠低",
                    "保守目標價": r["low_target"],
                    "保守空間": f"{up:+.1f}%", "風險報酬": r["rr"], "買點訊號": r["verdict"],
                })
            sdf = pd.DataFrame(rows).sort_values("總分", ascending=False, na_position="last").reset_index(drop=True)
            st.subheader("📋 批次評分排行榜（依總分排序）")
            st.caption("N/A 代表該面向完全沒有資料；『有效指標』偏低的個股，分數可信度較差。")
            st.dataframe(
                sdf, hide_index=True, **STRETCH,
                column_config={
                    "現價": st.column_config.NumberColumn(format="%.2f"),
                    "總分": st.column_config.NumberColumn(format="%.0f"),
                    "基本面": st.column_config.NumberColumn(format="%.0f"),
                    "籌碼面": st.column_config.NumberColumn(format="%.0f"),
                    "技術面": st.column_config.NumberColumn(format="%.0f"),
                    "保守目標價": st.column_config.NumberColumn(format="%.1f"),
                    "風險報酬": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.write("---")
            st.subheader("🔍 個股深入檢驗")
            sel = st.selectbox(
                "選擇個股：", sdf["代碼"].tolist(),
                format_func=lambda c: f"{ok[c]['display_name']}　總分 {fmt(ok[c]['total'])}（{ok[c]['verdict']}）",
            )
            render_detail(ok[sel])

    with tab_scr:
        render_screener(codes, token, info_map)

    with tab_bt:
        render_backtest(ok)

    st.caption("評分與目標價由公開資料以固定規則計算，僅供參考，不構成投資建議。")


if __name__ == "__main__":
    main()
