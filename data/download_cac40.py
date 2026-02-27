"""Download CAC40 stock data. Run: python data/download_cac40.py"""
import os
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance: pip install yfinance")

TICKERS = {"AC.PA": "Accor", "BNP.PA": "BNP Paribas", "CAP.PA": "Capgemini", "AI.PA": "Air Liquide"}
START, END = "2014-01-01", "2020-01-01"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_CAC40.csv")

frames = []
for ticker, name in TICKERS.items():
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START, end=END, progress=False)
    if df.empty:
        print(f"  WARNING: No data for {ticker}")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df[["Close"]].rename(columns={"Close": "Closing_Price"})
    s["Name"], s["Date"] = name, s.index
    frames.append(s.reset_index(drop=True))
    print(f"  {len(s)} days")

merged = pd.concat(frames, ignore_index=True)[["Date", "Name", "Closing_Price"]]
merged.sort_values(["Name", "Date"]).to_csv(OUT, index=True)
print(f"\nSaved {len(merged)} rows -> {OUT}")
