"""每週抓集保(TDCC)全市場「400張以上大戶持股比」，併入 holders_history.csv
給 GitHub Actions 用，只需要 pandas 與 requests（不依賴 streamlit）。
"""
import datetime as dt
import io
import os
import sys

import pandas as pd
import requests

URL = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holders_history.csv")
KEEP_DAYS = 800  # 只保留約兩年多，避免檔案越來越大


def parse_tdcc_csv(text: str) -> pd.DataFrame:
    """持股分級 12~15 = 400 張以上；16 為差異數調整、17 為合計，不計入"""
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
    out = out.dropna()
    return out[out["stock_id"].str.fullmatch(r"\d{4,6}")]  # 只留股票／ETF 代碼


def main():
    hdr = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(URL, headers=hdr, timeout=60)
    if r.status_code != 200 or not r.content:
        r = requests.post(URL, headers=hdr, timeout=60)
    r.raise_for_status()
    snap = parse_tdcc_csv(r.content.decode("utf-8-sig"))
    if snap.empty:
        sys.exit("TDCC 回傳空資料")

    if os.path.exists(FILE):
        hist = pd.read_csv(FILE, dtype={"stock_id": str})
    else:
        hist = pd.DataFrame(columns=["date", "stock_id", "large_pct"])
    new = snap[~snap["date"].isin(hist["date"])]
    if new.empty:
        print("這一週的資料已存在，略過。")
        return
    hist = new if hist.empty else pd.concat([hist, new], ignore_index=True)
    hist = hist.drop_duplicates(["date", "stock_id"])
    cutoff = (dt.date.today() - dt.timedelta(days=KEEP_DAYS)).isoformat()
    hist = hist[hist["date"] >= cutoff].sort_values(["date", "stock_id"])
    hist.to_csv(FILE, index=False)
    print(f"新增 {len(new)} 筆（資料日期 {new['date'].iloc[0]}），歷史共 {hist['date'].nunique()} 週。")


if __name__ == "__main__":
    main()
