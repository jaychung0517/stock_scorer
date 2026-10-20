"""
個股多面向評分、買點精準決策與停利停損系統（台股實戰量化版 V5，Streamlit）
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
