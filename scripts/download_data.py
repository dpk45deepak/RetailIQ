"""Download the UCI Online Retail II dataset and convert it to one CSV (much faster to load than the xlsx).

    python scripts/download_data.py

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii  (CC BY 4.0)
If your network blocks the download, fetch the zip in a browser, unzip it and place
``online_retail_II.xlsx`` in ``data/raw/``; the loader reads it directly.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print("downloading", URL)
    try:
        payload = urllib.request.urlopen(URL, timeout=120).read()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed ({exc}).\nDownload manually from the dataset page and put the file in {RAW}")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extractall(RAW)
    xlsx = next(RAW.glob("online_retail_II.xlsx"))
    print("reading both sheets (about a minute) ...")
    df = pd.concat(pd.read_excel(xlsx, sheet_name=None).values(), ignore_index=True)
    out = RAW / "online_retail_II.csv"
    df.to_csv(out, index=False)
    print(f"saved {out} with {len(df):,} rows (expected 1,067,371)")


if __name__ == "__main__":
    main()
