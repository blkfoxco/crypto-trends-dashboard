#!/usr/bin/env python3
"""
Crypto Google Trends + BTC Price Dashboard
Generates a stacked area chart with BTC price overlay (log scale)
Last 5 years of data
"""

import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
from pytrends.request import TrendReq

# ── Config ────────────────────────────────────────────────────────────────────

TERMS = [
    "Cryptocurrency", "Bitcoin", "Doge", "Ethereum", "XRP",
    "Solana", "Cardano", "Chainlink", "Litecoin", "Hyperliquid",
    "Binance", "Coinbase", "Bybit", "OKX", "CoinMarketCap",
]

ANCHOR = "Bitcoin"
TIMEFRAME = "today 5-y"
OUTPUT = "crypto_trends_dashboard.png"

COLORS = [
    '#e6194b', '#f58231', '#ffe119', '#3cb44b', '#42d4f4',
    '#4363d8', '#911eb4', '#f032e6', '#a9a9a9', '#9A6324',
    '#800000', '#469990', '#000075', '#aaffc3', '#dcbeff',
]

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_trends():
    """Fetch trends for all terms, normalised via Bitcoin anchor."""
    pytrends = TrendReq(hl='en-US', tz=420, timeout=(10, 25))

    other_terms = [t for t in TERMS if t != ANCHOR]
    batches = [other_terms[i:i+4] for i in range(0, len(other_terms), 4)]

    all_data = {}
    bitcoin_baseline = None

    for i, batch in enumerate(batches):
        kw_list = [ANCHOR] + batch
        print(f"  Batch {i+1}/{len(batches)}: {kw_list}")

        for attempt in range(3):
            try:
                pytrends.build_payload(kw_list, timeframe=TIMEFRAME)
                df = pytrends.interest_over_time()
                break
            except Exception as e:
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  Skipping batch {i+1} after 3 failed attempts")
            continue

        if df.empty:
            print(f"  Warning: No data for batch {i+1}")
            continue

        if 'isPartial' in df.columns:
            df = df.drop('isPartial', axis=1)

        if bitcoin_baseline is None:
            bitcoin_baseline = df[ANCHOR].copy()
            all_data[ANCHOR] = bitcoin_baseline
        
        # Scale factor: normalise this batch's Bitcoin to the baseline
        scale = bitcoin_baseline / df[ANCHOR].replace(0, np.nan)

        for term in batch:
            if term in df.columns:
                all_data[term] = (df[term] * scale).clip(0, 150)

        if i < len(batches) - 1:
            time.sleep(4)

    result = pd.DataFrame(all_data).dropna(how='all')
    print(f"  Got {len(result)} weeks of data")
    return result


def fetch_btc_price():
    """Fetch 5-year BTC weekly price history from Binance (no auth needed)."""
    print("  Fetching BTC price from Binance...")
    # 260 weekly candles ≈ 5 years
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1w", "limit": 260},
        timeout=30
    )
    r.raise_for_status()
    candles = r.json()
    # Binance kline: [open_time, open, high, low, close, ...]
    df = pd.DataFrame(candles, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    df['date']  = pd.to_datetime(df['open_time'], unit='ms')
    df['price'] = df['close'].astype(float)
    return df.set_index('date')['price']


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(trends_df, btc_price):
    """Render the stacked area + BTC price chart."""

    BG     = '#0d1117'
    GRID   = '#1e242c'
    TEXT   = '#c9d1d9'
    SUBTEXT = '#8b949e'

    fig, ax1 = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)

    # Order terms so the largest are on top
    ordered = [t for t in TERMS if t in trends_df.columns]
    ordered.sort(key=lambda t: trends_df[t].sum())

    data   = [trends_df[t].fillna(0).values for t in ordered]
    colors = COLORS[:len(ordered)]

    ax1.stackplot(
        trends_df.index, data,
        labels=ordered,
        colors=colors,
        alpha=0.88
    )

    ax1.set_ylabel('Google Trends Interest (normalised)', color=SUBTEXT, fontsize=10)
    ax1.tick_params(colors=SUBTEXT, labelsize=9)
    ax1.set_ylim(0)

    for spine in ax1.spines.values():
        spine.set_color(GRID)
    ax1.spines['top'].set_visible(False)
    ax1.grid(axis='y', color=GRID, linewidth=0.6, linestyle='--')
    ax1.set_axisbelow(True)

    # ── BTC price (right axis, log scale) ────────────────────────────────────
    ax2 = ax1.twinx()

    btc_weekly  = btc_price.resample('W').last()
    btc_aligned = btc_weekly.reindex(trends_df.index, method='nearest',
                                     tolerance=pd.Timedelta('14D'))

    ax2.plot(trends_df.index, btc_aligned.values,
             color='white', linewidth=1.8, label='BTC Price', zorder=5, alpha=0.95)

    ax2.set_yscale('log')
    ax2.set_ylabel('BTC Price (USD, log scale)', color=SUBTEXT, fontsize=10)
    ax2.tick_params(colors=SUBTEXT, labelsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x:,.0f}'
    ))
    ax2.spines['right'].set_color(GRID)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # ── X axis ───────────────────────────────────────────────────────────────
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', color=SUBTEXT, fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels = ax1.get_legend_handles_labels()
    btc_h, btc_l   = ax2.get_legend_handles_labels()

    ax1.legend(
        handles[::-1] + btc_h,
        labels[::-1]  + btc_l,
        loc='upper left', ncol=4, fontsize=8,
        facecolor='#161b22', edgecolor='#30363d',
        labelcolor=TEXT, framealpha=0.92,
        borderpad=0.8, handlelength=1.5
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax1.set_title(
        f'Crypto Google Trends + BTC Price  ·  Last 5 Years  ·  {datetime.now().strftime("%Y-%m-%d")}',
        color=TEXT, fontsize=13, pad=14, loc='left', x=0.01
    )

    plt.tight_layout(pad=1.5)
    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n✅  Chart saved → {OUTPUT}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📊 Fetching Google Trends (this takes ~1 min due to rate limiting)...")
    trends = fetch_trends()

    print("\n💰 Fetching BTC price...")
    btc = fetch_btc_price()

    print("\n🎨 Rendering chart...")
    plot(trends, btc)
