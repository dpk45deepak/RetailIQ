"""Synthetic stand-in for the UCI "Online Retail II" dataset.

WHY THIS EXISTS
---------------
The real dataset has to be downloaded from the UCI website. When it is not
available (offline machine, CI, first run) we generate a dataset with the SAME
columns and the SAME kinds of data-quality problems (cancellations, guest
checkouts, duplicates, zero prices, non-product codes, data-entry outliers).

It is NOT real data. Every artefact built from it is labelled
"synthetic" in the dashboard. Drop the real file in ``data/raw/`` and rerun
``python scripts/run_pipeline.py`` to rebuild everything on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START = pd.Timestamp("2009-12-01")
END = pd.Timestamp("2011-12-09")

# theme -> (word pool, base price, price spread)
THEMES: dict[str, tuple[list[str], float, float]] = {
    "KITCHEN": (["MUG", "TEAPOT", "CAKE TIN", "JAR", "SPOON", "BOWL", "APRON", "LUNCH BAG"], 3.2, 0.5),
    "GARDEN": (["PLANTER", "SEED TRAY", "WATERING CAN", "BIRD FEEDER", "LANTERN", "TROWEL"], 4.5, 0.55),
    "CHRISTMAS": (["BAUBLE", "STOCKING", "ADVENT CALENDAR", "WREATH", "SANTA", "SNOWFLAKE"], 2.1, 0.5),
    "STATIONERY": (["NOTEBOOK", "PENCIL SET", "CARD", "WRAP", "STICKER", "DIARY"], 1.4, 0.45),
    "BATHROOM": (["SOAP DISH", "TOWEL", "BATH SALTS", "SPONGE", "MIRROR", "TOOTHBRUSH MUG"], 3.8, 0.5),
    "JEWELLERY": (["NECKLACE", "BRACELET", "EARRINGS", "RING", "BROOCH", "HAIR CLIP"], 2.9, 0.6),
    "TOYS": (["PUZZLE", "DOLL", "SKITTLES", "SPINNING TOP", "TEDDY", "PLAY TENT"], 4.1, 0.55),
    "HOME DECOR": (["CUSHION", "PHOTO FRAME", "CANDLE HOLDER", "CLOCK", "DOORSTOP", "SIGN"], 5.6, 0.6),
    "PARTY": (["BUNTING", "BALLOON", "PAPER CUP", "NAPKINS", "GARLAND", "CONFETTI"], 1.9, 0.45),
    "STORAGE": (["BASKET", "CRATE", "TIN", "DRAWER KNOBS", "HOOK", "BOX SET"], 6.3, 0.6),
}
ADJECTIVES = ["VINTAGE", "RETRO", "HEART", "FLORAL", "STRIPEY", "SPOTTY", "RED", "BLUE", "IVORY",
              "PINK", "GREEN", "WOODEN", "METAL", "GLASS", "ENAMEL", "FELT", "LACE", "BOTANICAL"]
COUNTRIES = ["United Kingdom"] * 0 + ["Germany", "France", "EIRE", "Spain", "Netherlands",
                                      "Belgium", "Switzerland", "Portugal", "Australia", "Norway"]
NON_PRODUCT = [("POST", "POSTAGE", 18.0), ("D", "Discount", -6.0), ("M", "Manual", 25.0),
               ("BANK CHARGES", "Bank Charges", 15.0)]

# archetype: share, orders/yr, lines/order, qty scale, active window
ARCHETYPES = {
    "champion": dict(share=0.08, rate=26, lines=28, qty=1.8, window=(0.0, 1.0)),
    "loyal": dict(share=0.20, rate=11, lines=16, qty=1.2, window=(0.0, 1.0)),
    "occasional": dict(share=0.30, rate=4, lines=11, qty=1.0, window=(0.0, 1.0)),
    "new": dict(share=0.12, rate=5, lines=10, qty=1.0, window=(0.93, 1.0)),
    "onetime": dict(share=0.17, rate=2, lines=8, qty=0.9, window=(0.0, 1.0)),
    "lapsed": dict(share=0.13, rate=8, lines=13, qty=1.1, window=(0.0, 0.5)),
}


def _build_catalog(rng: np.random.Generator, n_products: int) -> pd.DataFrame:
    rows = []
    theme_names = list(THEMES)
    for i in range(n_products):
        theme = theme_names[i % len(theme_names)]
        words, base, spread = THEMES[theme]
        noun = words[rng.integers(len(words))]
        adj = rng.choice(ADJECTIVES, size=rng.integers(1, 3), replace=False)
        desc = " ".join([*adj, noun] + ([theme] if rng.random() < 0.35 else []))
        code = f"{rng.integers(10000, 99999)}" + (rng.choice(list("ABCDE")) if rng.random() < 0.3 else "")
        price = float(np.round(np.exp(rng.normal(np.log(base), spread)), 2).clip(0.29, 60))
        rows.append((code, desc, theme, price))
    cat = pd.DataFrame(rows, columns=["StockCode", "Description", "Theme", "Price"])
    cat = cat.drop_duplicates("StockCode").reset_index(drop=True)
    # Zipf-like popularity inside each theme (a few products sell far more)
    cat["pop"] = 1.0 / (cat.groupby("Theme").cumcount() + 1) ** 0.8
    return cat


def generate_synthetic_retail(n_customers: int = 2000, n_products: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Return a raw-looking dataframe with the Online Retail II column names."""
    rng = np.random.default_rng(seed)
    cat = _build_catalog(rng, n_products)
    themes = list(THEMES)
    by_theme = {t: cat[cat.Theme == t].reset_index(drop=True) for t in themes}
    theme_probs = {t: (d["pop"] / d["pop"].sum()).to_numpy() for t, d in by_theme.items()}

    days = pd.date_range(START, END, freq="D")
    dow_w = np.array([0.0 if d.dayofweek == 5 else (0.55 if d.dayofweek == 6 else 1.0) for d in days])
    month_w = np.array([{9: 1.3, 10: 1.5, 11: 1.9, 12: 1.2, 1: 0.7, 2: 0.8}.get(d.month, 1.0) for d in days])
    day_w = dow_w * month_w
    n_days = len(days)

    names, shares = zip(*[(k, v["share"]) for k, v in ARCHETYPES.items()])
    kinds = rng.choice(names, size=n_customers, p=np.array(shares) / sum(shares))
    customer_ids = rng.choice(np.arange(12346, 18288), size=n_customers, replace=False)
    countries = np.where(rng.random(n_customers) < 0.9, "United Kingdom", rng.choice(COUNTRIES, n_customers))

    frames: list[pd.DataFrame] = []
    invoice_no = 489434

    theme_arrays = {t: (d["StockCode"].to_numpy(), d["Description"].to_numpy(), d["Price"].to_numpy()) for t, d in by_theme.items()}
    theme_idx = {t: i for i, t in enumerate(themes)}

    def make_order(date: pd.Timestamp, pref: np.ndarray, spec: dict, qty_scale: float):
        n_lines = max(1, int(rng.negative_binomial(3, 3 / (3 + spec["lines"]))) + 1)
        counts = rng.multinomial(n_lines, pref)
        codes, descs, prices = [], [], []
        for t, k in zip(themes, counts):
            if k == 0:
                continue
            c, d, p = theme_arrays[t]
            pick = np.unique(rng.choice(len(c), size=k, p=theme_probs[t]))   # unique -> no duplicate lines
            codes.extend(c[pick]); descs.extend(d[pick]); prices.extend(p[pick])
        m = len(codes)
        qty = np.clip(np.round(np.exp(rng.normal(np.log(3 * qty_scale), 0.8, m))), 1, 400).astype(int)
        qty = np.where(rng.random(m) < 0.35, (qty // 6 + 1) * 6, qty)  # pack sizes of 6
        secs = int(rng.integers(8 * 3600, 20 * 3600))
        return pd.DataFrame({"StockCode": codes, "Description": descs, "Price": prices, "Quantity": qty,
                             "InvoiceDate": date.normalize() + pd.Timedelta(seconds=secs)})

    for cid, kind, country in zip(customer_ids, kinds, countries):
        spec = ARCHETYPES[kind]
        lo, hi = int(spec["window"][0] * (n_days - 1)), int(spec["window"][1] * (n_days - 1))
        if kind == "onetime":  # two orders at most, early in life
            lo = int(rng.integers(0, n_days - 60)); hi = min(n_days - 1, lo + 40)
        elif kind == "occasional":
            lo = int(rng.integers(0, n_days // 3))
        span = max(hi - lo, 1)
        n_orders = max(1, int(rng.poisson(spec["rate"] * span / 365 * rng.gamma(4, 0.25))))
        if kind == "onetime":
            n_orders = int(rng.integers(1, 3))
        idx = rng.choice(np.arange(lo, hi + 1), size=n_orders, p=day_w[lo:hi + 1] / day_w[lo:hi + 1].sum())
        idx.sort()
        pref = rng.dirichlet(np.full(len(themes), 0.25))   # each customer has favourite themes
        pref = 0.85 * pref + 0.15 / len(themes)
        qty_scale = spec["qty"] * rng.lognormal(0, 0.35)
        for d in idx:
            date = days[d]
            order = make_order(date, pref, spec, qty_scale)
            # seasonal Christmas push in autumn
            order["Invoice"] = str(invoice_no); invoice_no += 1
            order["Customer ID"] = float(cid); order["Country"] = country
            frames.append(order)
            if rng.random() < 0.035:  # cancellation: some lines come back within 30 days
                back = order.sample(frac=rng.uniform(0.3, 1.0), random_state=int(rng.integers(1e9))).copy()
                back["Invoice"] = "C" + str(invoice_no); invoice_no += 1
                back["Quantity"] = -back["Quantity"]
                back["InvoiceDate"] = back["InvoiceDate"] + pd.Timedelta(days=int(rng.integers(1, 30)), hours=1)
                frames.append(back)
            if rng.random() < 0.25:  # postage / manual lines
                code, desc, price = NON_PRODUCT[rng.integers(len(NON_PRODUCT))]
                frames.append(pd.DataFrame({"StockCode": [code], "Description": [desc], "Price": [abs(price)],
                                            "Quantity": [1 if price > 0 else -1], "InvoiceDate": [order.InvoiceDate.iloc[0]],
                                            "Invoice": [order.Invoice.iloc[0]], "Customer ID": [float(cid)], "Country": [country]}))

    # Guest checkouts: valid sales with no Customer ID (~18% of lines in the real data)
    sales = pd.concat(frames, ignore_index=True)
    n_guest = int(len(sales) * 0.2)
    pref = np.full(len(themes), 1 / len(themes))
    g_frames = []
    for _ in range(max(1, n_guest // 14)):
        d = days[rng.choice(n_days, p=day_w / day_w.sum())]
        o = make_order(d, pref, ARCHETYPES["occasional"], 1.0)
        o["Invoice"] = str(invoice_no); invoice_no += 1
        o["Customer ID"] = np.nan; o["Country"] = "United Kingdom"
        g_frames.append(o)
    df = pd.concat([sales, *g_frames], ignore_index=True)

    # ---- Inject realistic data-quality problems -------------------------------------
    zero = df.sample(frac=0.003, random_state=seed).index          # free / adjustment lines
    df.loc[zero, "Price"] = 0.0
    df.loc[df.sample(frac=0.002, random_state=seed + 1).index, "Description"] = np.nan
    adj = pd.DataFrame({"StockCode": ["21870"] * 8, "Description": ["damaged"] * 4 + ["?"] * 4,
                        "Quantity": rng.integers(-60, -1, 8), "InvoiceDate": START + pd.to_timedelta(rng.integers(0, 600, 8), "D"),
                        "Price": 0.0, "Invoice": [str(invoice_no + i) for i in range(8)],
                        "Customer ID": np.nan, "Country": "United Kingdom"})
    invoice_no += 8
    huge = df[df["Customer ID"].notna() & (df.Quantity > 0)].iloc[[1000]].copy()   # entry error + reversal
    huge["Quantity"] = 74215; huge["Invoice"] = str(invoice_no)
    rev = huge.copy(); rev["Quantity"] = -74215; rev["Invoice"] = "C" + str(invoice_no + 1)
    rev["InvoiceDate"] = rev["InvoiceDate"] + pd.Timedelta(hours=2)
    df = pd.concat([df, adj, huge, rev], ignore_index=True)
    dups = df.sample(frac=0.005, random_state=seed + 2)
    df = pd.concat([df, dups], ignore_index=True)

    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    df = df[df.InvoiceDate <= END + pd.Timedelta(days=1)]
    df["Description"] = df["Description"].astype("object")
    return df[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]]
