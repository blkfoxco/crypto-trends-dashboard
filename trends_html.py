#!/usr/bin/env python3
"""
Crypto Google Trends + BTC Price — Interactive HTML Dashboard
Hover, zoom, toggle terms via legend. Dark theme.
"""

import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pytrends.request import TrendReq

# ── Config ────────────────────────────────────────────────────────────────────

TERMS = [
    "Cryptocurrency", "Bitcoin", "Doge", "Ethereum", "XRP",
    "Solana", "Cardano", "Chainlink", "Litecoin", "Hyperliquid",
    "Binance", "Coinbase", "Bybit", "OKX", "CoinMarketCap",
]

ANCHOR    = "Bitcoin"
TIMEFRAME = "today 5-y"
OUTPUT    = "crypto_trends_dashboard.html"

COLORS = [
    '#e6194b', '#f58231', '#ffe119', '#3cb44b', '#42d4f4',
    '#4363d8', '#f032e6', '#a9a9a9', '#9A6324', '#800000',
    '#469990', '#000075', '#aaffc3', '#dcbeff', '#911eb4',
]

# Key events to annotate
EVENTS = [
    ("2021-04-14", "BTC ATH ~$65k"),
    ("2021-11-10", "BTC ATH ~$69k"),
    ("2022-05-09", "LUNA Collapse"),
    ("2022-11-08", "FTX Collapse"),
    ("2024-04-20", "BTC Halving"),
    ("2024-03-14", "BTC ATH ~$73k"),
    ("2025-01-20", "BTC ATH ~$109k"),
    ("2025-05-22", "BTC ATH ~$111k"),
    ("2025-10-06", "BTC ATH ~$126k"),
]

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_trends():
    pytrends    = TrendReq(hl='en-US', tz=420, timeout=(10, 25))
    other_terms = [t for t in TERMS if t != ANCHOR]
    batches     = [other_terms[i:i+4] for i in range(0, len(other_terms), 4)]

    all_data         = {}
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
    print("  Fetching BTC price from Binance...")
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1w", "limit": 260},
            timeout=30
        )
        r.raise_for_status()
        candles = r.json()
        df = pd.DataFrame(candles, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','qav','trades','tbbav','tbqav','ignore'
        ])
        df['date']  = pd.to_datetime(df['open_time'], unit='ms')
        df['price'] = df['close'].astype(float)
        return df.set_index('date')['price']
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 'unknown'
        print(f"  Binance failed ({status_code}). Falling back to CoinGecko...")
    except requests.RequestException as e:
        print(f"  Binance request failed ({e}). Falling back to CoinGecko...")

    end_date = int(time.time())
    start_date = end_date - (260 * 7 * 24 * 60 * 60)

    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
        params={
            "vs_currency": "usd",
            "from": start_date,
            "to": end_date,
        },
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    prices = data.get('prices', [])
    if not prices:
        raise ValueError("CoinGecko returned no BTC price data")

    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    weekly = df.set_index('date')['price'].resample('W').last().dropna()
    return weekly.tail(260)


# ── Plotting ──────────────────────────────────────────────────────────────────

def build_chart(trends_df, btc_price):

    # Order: smallest total first so big names stack on top
    ordered = [t for t in TERMS if t in trends_df.columns]
    ordered.sort(key=lambda t: trends_df[t].sum())

    dates = trends_df.index.tolist()
    fig   = go.Figure()

    def hex_to_rgba(hex_color, alpha=0.75):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    # ── Stacked area traces ───────────────────────────────────────────────────
    for i, term in enumerate(ordered):
        vals      = trends_df[term].fillna(0).tolist()
        color_hex = COLORS[i % len(COLORS)]

        fig.add_trace(go.Scatter(
            x          = dates,
            y          = vals,
            name       = term,
            mode       = 'lines',
            stackgroup = 'one',
            line       = dict(width=0.5, color=color_hex),
            fillcolor  = hex_to_rgba(color_hex, 0.80),
            hovertemplate = f'<b>{term}</b><br>%{{x|%b %Y}}<br>Interest: %{{y:.1f}}<extra></extra>',
        ))

    # ── BTC price line ────────────────────────────────────────────────────────
    btc_weekly  = btc_price.resample('W').last()
    btc_aligned = btc_weekly.reindex(trends_df.index, method='nearest',
                                     tolerance=pd.Timedelta('14D'))

    fig.add_trace(go.Scatter(
        x          = dates,
        y          = btc_aligned.values,
        name       = 'BTC Price',
        yaxis      = 'y2',
        mode       = 'lines',
        line       = dict(color='white', width=2),
        hovertemplate = '<b>BTC Price</b><br>%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>',
    ))

    # ── Event annotations ─────────────────────────────────────────────────────
    annotations = []
    shapes      = []

    for date_str, label in EVENTS:
        dt = pd.Timestamp(date_str)
        # Only annotate if within our date range
        if trends_df.index[0] <= dt <= trends_df.index[-1]:
            shapes.append(dict(
                type      = 'line',
                x0        = dt, x1 = dt,
                y0        = 0,  y1 = 1,
                yref      = 'paper',
                line      = dict(color='rgba(255,255,255,0.25)', width=1, dash='dot'),
            ))
            annotations.append(dict(
                x          = dt,
                y          = 1.01,
                yref       = 'paper',
                text       = label,
                showarrow  = False,
                font       = dict(size=9, color='#8b949e'),
                xanchor    = 'center',
                textangle  = -35,
            ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title = dict(
            text      = f'Crypto Google Trends + BTC Price  ·  Last 5 Years  ·  {datetime.now().strftime("%Y-%m-%d")}',
            font      = dict(size=16, color='#c9d1d9'),
            x         = 0.01,
            xanchor   = 'left',
        ),

        paper_bgcolor = '#0d1117',
        plot_bgcolor  = '#0d1117',

        xaxis = dict(
            showgrid      = True,
            gridcolor     = '#1e242c',
            tickfont      = dict(color='#8b949e'),
            tickformat    = '%b %Y',
            dtick         = 'M6',
            linecolor     = '#30363d',
        ),

        yaxis = dict(
            title      = dict(text='Google Trends Interest (normalised)', font=dict(color='#8b949e')),
            tickfont   = dict(color='#8b949e'),
            showgrid   = True,
            gridcolor  = '#1e242c',
            linecolor  = '#30363d',
            fixedrange = False,
        ),

        yaxis2 = dict(
            title       = dict(text='BTC Price USD', font=dict(color='#8b949e')),
            tickfont    = dict(color='#8b949e'),
            overlaying  = 'y',
            side        = 'right',
            type        = 'linear',
            showgrid    = False,
            linecolor   = '#30363d',
            tickprefix  = '$',
            tickformat  = ',.0f',
        ),

        legend = dict(
            bgcolor     = '#161b22',
            bordercolor = '#30363d',
            borderwidth = 1,
            font        = dict(color='#c9d1d9', size=11),
            orientation = 'v',
            x           = 0.01,
            y           = 0.99,
            xanchor     = 'left',
            yanchor     = 'top',
            itemclick       = 'toggle',
            itemdoubleclick = 'toggleothers',
        ),

        hovermode   = 'x unified',
        hoverlabel  = dict(
            bgcolor   = '#161b22',
            bordercolor = '#30363d',
            font      = dict(color='#c9d1d9', size=11),
        ),

        margin  = dict(t=60, r=80, b=60, l=60),
        height  = 700,

        shapes      = shapes,
        annotations = annotations,
    )

    return fig


def main():
    print("📊 Fetching Google Trends (batched, ~1 min)...")
    trends = fetch_trends()

    print("\n💰 Fetching BTC price...")
    btc = fetch_btc_price()

    print("\n🎨 Building interactive chart...")
    fig = build_chart(trends, btc)

    fig.write_html(
        OUTPUT,
        include_plotlyjs = 'cdn',
        full_html        = True,
        config           = {
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'width':  1800,
                'height': 900,
                'scale':  2,
            }
        }
    )

    # Post-process HTML: dark page + muted active button colours + methodology
    dark_style = (
        '<style>\n'
        '  @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap");\n'
        '\n'
        '  *, *::before, *::after { box-sizing: border-box; }\n'
        '\n'
        '  html, body {\n'
        '    background-color: #0d1117;\n'
        '    color: #c9d1d9;\n'
        '    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;\n'
        '    margin: 0;\n'
        '    padding: 0;\n'
        '    font-size: 15px;\n'
        '    line-height: 1.6;\n'
        '  }\n'
        '\n'
        '  .page-header {\n'
        '    max-width: 1400px;\n'
        '    margin: 0 auto;\n'
        '    padding: 32px 32px 0;\n'
        '  }\n'
        '\n'
        '  .page-header h1 {\n'
        '    font-size: 22px;\n'
        '    font-weight: 600;\n'
        '    color: #e6edf3;\n'
        '    margin: 0 0 6px;\n'
        '    letter-spacing: -0.3px;\n'
        '  }\n'
        '\n'
        '  .page-header p {\n'
        '    font-size: 14px;\n'
        '    color: #8b949e;\n'
        '    margin: 0;\n'
        '  }\n'
        '\n'
        '  .page-header a {\n'
        '    color: #58a6ff;\n'
        '    text-decoration: none;\n'
        '  }\n'
        '\n'
        '  .chart-wrapper {\n'
        '    max-width: 1400px;\n'
        '    margin: 0 auto;\n'
        '    padding: 16px 32px 0;\n'
        '  }\n'
        '\n'
        '  /* Make the Plotly div respect our container */\n'
        '  .plotly-graph-div {\n'
        '    width: 100% !important;\n'
        '  }\n'
        '\n'
        '  .methodology {\n'
        '    max-width: 1400px;\n'
        '    margin: 0 auto;\n'
        '    padding: 40px 32px 64px;\n'
        '  }\n'
        '\n'
        '  .methodology h2 {\n'
        '    font-size: 16px;\n'
        '    font-weight: 600;\n'
        '    color: #e6edf3;\n'
        '    margin: 0 0 20px;\n'
        '    text-transform: uppercase;\n'
        '    letter-spacing: 0.8px;\n'
        '    border-bottom: 1px solid #21262d;\n'
        '    padding-bottom: 12px;\n'
        '  }\n'
        '\n'
        '  .method-grid {\n'
        '    display: grid;\n'
        '    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));\n'
        '    gap: 16px;\n'
        '    margin-bottom: 24px;\n'
        '  }\n'
        '\n'
        '  .method-card {\n'
        '    background: #161b22;\n'
        '    border: 1px solid #21262d;\n'
        '    border-radius: 8px;\n'
        '    padding: 20px;\n'
        '  }\n'
        '\n'
        '  .method-card h3 {\n'
        '    font-size: 13px;\n'
        '    font-weight: 600;\n'
        '    color: #79c0ff;\n'
        '    margin: 0 0 10px;\n'
        '    text-transform: uppercase;\n'
        '    letter-spacing: 0.6px;\n'
        '  }\n'
        '\n'
        '  .method-card p, .method-card ul {\n'
        '    font-size: 14px;\n'
        '    color: #8b949e;\n'
        '    margin: 0;\n'
        '  }\n'
        '\n'
        '  .method-card ul {\n'
        '    padding-left: 18px;\n'
        '  }\n'
        '\n'
        '  .method-card li {\n'
        '    margin-bottom: 6px;\n'
        '  }\n'
        '\n'
        '  .method-card strong {\n'
        '    color: #c9d1d9;\n'
        '  }\n'
        '\n'
        '  .terms-list {\n'
        '    display: flex;\n'
        '    flex-wrap: wrap;\n'
        '    gap: 6px;\n'
        '    margin-top: 4px;\n'
        '  }\n'
        '\n'
        '  .term-pill {\n'
        '    background: #21262d;\n'
        '    border: 1px solid #30363d;\n'
        '    border-radius: 20px;\n'
        '    padding: 2px 10px;\n'
        '    font-size: 12px;\n'
        '    color: #c9d1d9;\n'
        '  }\n'
        '\n'
        '  .caveat {\n'
        '    font-size: 13px;\n'
        '    color: #6e7681;\n'
        '    background: #161b22;\n'
        '    border: 1px solid #21262d;\n'
        '    border-left: 3px solid #30363d;\n'
        '    border-radius: 4px;\n'
        '    padding: 12px 16px;\n'
        '    margin: 0;\n'
        '  }\n'
        '\n'
        '  .page-footer {\n'
        '    border-top: 1px solid #21262d;\n'
        '    padding: 16px 32px;\n'
        '    max-width: 1400px;\n'
        '    margin: 0 auto;\n'
        '    font-size: 12px;\n'
        '    color: #6e7681;\n'
        '    display: flex;\n'
        '    justify-content: space-between;\n'
        '    flex-wrap: wrap;\n'
        '    gap: 8px;\n'
        '  }\n'
        '</style>\n'
    )

    # Intercept Plotly's dynamic bright active-button fill and tone it down.
    # Plotly sets the fill attribute directly on SVG rects, so we use a
    # MutationObserver to catch every change and override bright fills.
    active_button_script = (
        '<script>\n'
        '(function () {\n'
        '  var ACTIVE_FILL   = "#2d3b4e";  /* muted blue-grey */\n'
        '  var ACTIVE_TEXT   = "#c9d1d9";\n'
        '\n'
        '  /* Returns true if a hex colour is "bright" (lightness > 60 %) */\n'
        '  function isBright(hex) {\n'
        '    hex = hex.replace(/^#/, "");\n'
        '    if (hex.length === 3) hex = hex.split("").map(function(c){return c+c;}).join("");\n'
        '    var r = parseInt(hex.slice(0,2),16)/255,\n'
        '        g = parseInt(hex.slice(2,4),16)/255,\n'
        '        b = parseInt(hex.slice(4,6),16)/255;\n'
        '    var max = Math.max(r,g,b), min = Math.min(r,g,b);\n'
        '    return (max + min) / 2 > 0.60;\n'
        '  }\n'
        '\n'
        '  function patchRect(el) {\n'
        '    if (!el.closest) return;\n'
        '    if (!el.closest(".updatemenu-button")) return;\n'
        '    var fill = el.getAttribute("fill") || "";\n'
        '    if (fill.startsWith("#") && isBright(fill)) {\n'
        '      el.setAttribute("fill", ACTIVE_FILL);\n'
        '    }\n'
        '  }\n'
        '\n'
        '  function attachObserver(gd) {\n'
        '    /* Initial pass */\n'
        '    gd.querySelectorAll(".updatemenu-button rect").forEach(patchRect);\n'
        '    /* Watch for future changes */\n'
        '    new MutationObserver(function (mutations) {\n'
        '      mutations.forEach(function (m) {\n'
        '        if (m.attributeName === "fill") patchRect(m.target);\n'
        '      });\n'
        '    }).observe(gd, { subtree: true, attributes: true, attributeFilter: ["fill"] });\n'
        '  }\n'
        '\n'
        '  document.addEventListener("DOMContentLoaded", function () {\n'
        '    var gd = document.querySelector(".plotly-graph-div");\n'
        '    if (!gd) return;\n'
        '    gd.addEventListener("plotly_afterplot", function () { attachObserver(gd); }, { once: true });\n'
        '  });\n'
        '})();\n'
        '</script>\n'
    )

    # Build metadata tags for head
    meta_tags = (
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="Interactive chart showing crypto Google Trends interest alongside Bitcoin price over the last 5 years. Updated {datetime.now().strftime("%Y-%m-%d")}.">\n'
        f'<title>Crypto Google Trends + BTC Price · Last 5 Years</title>\n'
    )

    # Terms list as pills
    terms_pills = '\n'.join(
        f'<span class="term-pill">{t}</span>' for t in TERMS
    )

    # Page header (injected before Plotly div)
    page_header = (
        '<div class="page-header">\n'
        '  <h1>Crypto Google Trends &amp; Bitcoin Price</h1>\n'
        f'  <p>5-year weekly view · Last updated {datetime.now().strftime("%d %b %Y")}</p>\n'
        '</div>\n'
        '<div class="chart-wrapper">\n'
    )

    # Methodology section (injected after Plotly div, before footer)
    methodology_section = (
        '</div><!-- end chart-wrapper -->\n'
        '\n'
        '<div class="methodology">\n'
        '  <h2>How it works</h2>\n'
        '  <div class="method-grid">\n'
        '\n'
        '    <div class="method-card">\n'
        '      <h3>What you\'re seeing</h3>\n'
        '      <p>Stacked area chart of relative Google search interest for 15 crypto terms over 5 years.\n'
        '      Bitcoin\'s USD price is overlaid as a white line on a logarithmic right-hand axis.\n'
        '      Click legend entries to show/hide individual terms.</p>\n'
        '    </div>\n'
        '\n'
        '    <div class="method-card">\n'
        '      <h3>Data sources</h3>\n'
        '      <ul>\n'
        '        <li><strong>Google Trends</strong> — weekly search interest via the unofficial <code>pytrends</code> API</li>\n'
        '        <li><strong>BTC Price</strong> — weekly close from Binance BTCUSDT spot (no auth required)</li>\n'
        '      </ul>\n'
        '    </div>\n'
        '\n'
        '    <div class="method-card">\n'
        '      <h3>Cross-batch normalisation</h3>\n'
        '      <p>Google Trends returns scores of 0–100 <em>relative within a single query</em>.\n'
        '      Since max 5 terms can be queried at once, we batch in groups of 4 with Bitcoin as a\n'
        '      shared anchor. Each batch\'s Bitcoin series is compared to the first batch\'s Bitcoin\n'
        '      baseline to derive a scale factor, letting all 15 terms share a common y-axis.</p>\n'
        '    </div>\n'
        '\n'
        '    <div class="method-card">\n'
        '      <h3>Terms tracked</h3>\n'
        '      <div class="terms-list">\n'
        f'        {terms_pills}\n'
        '      </div>\n'
        '    </div>\n'
        '\n'
        '    <div class="method-card">\n'
        '      <h3>Key events annotated</h3>\n'
        '      <ul>\n'
        '        <li><strong>Apr 2021</strong> — BTC ATH ~$65k</li>\n'
        '        <li><strong>Nov 2021</strong> — BTC ATH ~$69k</li>\n'
        '        <li><strong>May 2022</strong> — LUNA collapse</li>\n'
        '        <li><strong>Nov 2022</strong> — FTX collapse</li>\n'
        '        <li><strong>Mar 2024</strong> — BTC ATH ~$73k</li>\n'
        '        <li><strong>Apr 2024</strong> — Bitcoin halving</li>\n'
        '        <li><strong>Jan 2025</strong> — BTC ATH ~$109k</li>\n'
        '        <li><strong>May 2025</strong> — BTC ATH ~$111k</li>\n'
        '        <li><strong>Oct 2025</strong> — BTC ATH ~$126k</li>\n'
        '      </ul>\n'
        '    </div>\n'
        '\n'
        '    <div class="method-card">\n'
        '      <h3>BTC price axis</h3>\n'
        '      <p>BTC price is plotted on a linear scale (right axis, USD),\n'
        '      making absolute dollar moves easy to read at a glance.</p>\n'
        '    </div>\n'
        '\n'
        '  </div>\n'
        '\n'
        '  <p class="caveat">⚠️ Google Trends represents <em>relative</em> search popularity, not absolute volume.\n'
        '  The anchor normalisation approximates cross-batch comparability but may introduce minor distortions\n'
        '  where Bitcoin\'s own trend line differs between batches. Use as a directional signal, not a precise metric.</p>\n'
        '</div>\n'
        '\n'
        '<footer class="page-footer">\n'
        f'  <span>Data: Google Trends (pytrends) + Binance BTCUSDT · Generated {datetime.now().strftime("%Y-%m-%d %H:%M UTC+7")}</span>\n'
        '  <span>BTC price linear scale (USD) · Google Trends normalised to Bitcoin anchor · <a href="https://github.com/blkfoxco/crypto-trends-dashboard" target="_blank" rel="noopener" style="color:#58a6ff;text-decoration:none;">GitHub</a></span>\n'
        '</footer>\n'
    )

    with open(OUTPUT, 'r', encoding='utf-8') as f:
        html = f.read()

    # Inject meta + styles into <head>
    html = html.replace('<head>', '<head>\n' + meta_tags + dark_style, 1)

    # Wrap body content: inject page-header before the plotly div, close chart-wrapper after it
    html = html.replace('<body>', '<body>\n' + page_header, 1)

    # Close chart-wrapper + inject methodology before </body>
    html = html.replace('</body>', methodology_section + active_button_script + '</body>', 1)

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅  Dashboard saved → {OUTPUT}")
    print(f"    Open in browser: file://{__import__('os').path.abspath(OUTPUT)}")


if __name__ == "__main__":
    main()
