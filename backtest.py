"""Backtest the bot's moving average crossover rule, and measure current signal strength.

Rules, matching trading_bot.py:
  - BUY when the short SMA crosses above the long SMA between two completed days.
  - SELL when it crosses below, only if holding (long-only, no shorting).
  - Signals use the close; the trade happens at the NEXT day's open, as the
    scheduled bot does at 9:30am ET.
  - A trade still open at the end is valued at the last close.

Usage:
  python3 backtest.py                         # symbol and periods from config.json
  python3 backtest.py FDX META NOW --years 10
  python3 backtest.py AAPL --short 10 --long 30

Results describe the past only; they are not a prediction or investment advice.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def download(tickers, years):
    now = datetime.now(ET)
    start = now.date() - timedelta(days=int(years * 365.25) + 120)  # extra for the SMA warm-up
    data = yf.download(tickers, start=start.isoformat(), interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker")
    out = {}
    for t in tickers:
        try:
            df = (data[t] if len(tickers) > 1 else data.droplevel(0, axis=1)
                  if isinstance(data.columns, pd.MultiIndex) else data)
        except KeyError:
            continue
        df = df.dropna(subset=["Open", "Close"])
        if now.hour < 16:  # drop today's unfinished bar
            df = df[[d.date() < now.date() for d in df.index]]
        if not df.empty:
            out[t] = df
    return out


def backtest(df, short_w, long_w, years):
    close, open_ = df["Close"], df["Open"]
    s, l = close.rolling(short_w).mean(), close.rolling(long_w).mean()
    above = (s > l).where(s.notna() & l.notna())
    prev = above.shift(1)
    buy = (prev == False) & (above == True)   # noqa: E712  (NaN-safe comparisons)
    sell = (prev == True) & (above == False)  # noqa: E712

    test_start = pd.Timestamp(df.index[-1]) - pd.DateOffset(days=int(years * 365.25))
    idx = df.index
    trades, entry = [], None
    for i in range(len(idx) - 1):
        if idx[i] < test_start:
            continue
        if entry is None and buy.iloc[i]:
            entry = (idx[i + 1], open_.iloc[i + 1])
        elif entry is not None and sell.iloc[i]:
            trades.append((entry[0], idx[i + 1], entry[1], open_.iloc[i + 1], False))
            entry = None
    if entry is not None:
        trades.append((entry[0], idx[-1], entry[1], close.iloc[-1], True))

    rets = [exit_ / entry_ - 1 for _, _, entry_, exit_, _ in trades]
    period = close[close.index >= test_start]
    equity, peak, max_dd = 1.0, 1.0, 0.0
    streak = worst_streak = 0
    for r in rets:
        equity *= 1 + r
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
        streak = streak + 1 if r <= 0 else 0
        worst_streak = max(worst_streak, streak)
    days_in = sum((x - e).days for e, x, *_ in trades)
    span = max((period.index[-1] - period.index[0]).days, 1)
    return {
        "trades": len(trades),
        "win_rate": (sum(r > 0 for r in rets) / len(rets) * 100) if rets else None,
        "avg_trade": (sum(rets) / len(rets) * 100) if rets else None,
        "best": max(rets) * 100 if rets else None,
        "worst": min(rets) * 100 if rets else None,
        "strategy": (equity - 1) * 100,
        "buy_hold": (period.iloc[-1] / period.iloc[0] - 1) * 100,
        "max_dd": max_dd * 100,
        "loss_streak": worst_streak,
        "time_in": days_in / span * 100,
        "open_trade": bool(trades and trades[-1][4]),
    }


def strength(df, short_w, long_w):
    """Common measures of how firm the latest crossover is."""
    close, vol = df["Close"], df["Volume"]
    s, l = close.rolling(short_w).mean(), close.rolling(long_w).mean()
    above = (s > l).where(s.notna() & l.notna())
    flips = above.ne(above.shift(1)) & above.notna() & above.shift(1).notna()
    last_flip = flips[flips].index[-1] if flips.any() else None
    if last_flip is None:
        return None
    i = df.index.get_loc(last_flip)
    avg_vol = vol.iloc[max(0, i - 20):i].mean()
    return {
        "signal": "BUY" if above.iloc[-1] else "SELL",
        "crossed": last_flip.date(),
        "days_ago": len(df) - 1 - i,
        "gap_now": (s.iloc[-1] / l.iloc[-1] - 1) * 100,
        "gap_trend": (s.iloc[-1] / l.iloc[-1]) - (s.iloc[-2] / l.iloc[-2]),
        "vol_ratio": vol.iloc[i] / avg_vol if avg_vol else float("nan"),
        "price_vs_short": (close.iloc[-1] / s.iloc[-1] - 1) * 100,
    }


def fmt(v, spec="{:+.1f}%"):
    return "-" if v is None else spec.format(v)


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    p = argparse.ArgumentParser(description="Backtest the SMA crossover rule")
    p.add_argument("tickers", nargs="*", default=[cfg.get("symbol", "AAPL")])
    p.add_argument("--years", type=float, default=5)
    p.add_argument("--short", type=int, default=cfg["short_window"])
    p.add_argument("--long", type=int, default=cfg["long_window"])
    a = p.parse_args()
    tickers = [t.upper() for t in a.tickers]

    data = download(tickers, a.years)
    print(f"SMA{a.short}/SMA{a.long} crossover, long-only, last {a.years:g} years, "
          f"trades at next day's open\n")
    print(f"{'Ticker':<7}{'Trades':>7}{'Win%':>7}{'AvgTrd':>8}{'Best':>8}{'Worst':>8}"
          f"{'Strategy':>10}{'Buy&Hold':>10}{'MaxDD':>8}{'LossRun':>8}{'InMkt':>7}")
    rows = []
    for t in tickers:
        if t not in data or len(data[t]) < a.long + 2:
            print(f"{t:<7} not enough data")
            continue
        r = backtest(data[t], a.short, a.long, a.years)
        rows.append((t, r))
        print(f"{t:<7}{r['trades']:>7}{fmt(r['win_rate'], '{:.0f}%'):>7}{fmt(r['avg_trade']):>8}"
              f"{fmt(r['best']):>8}{fmt(r['worst']):>8}{fmt(r['strategy'], '{:+.0f}%'):>10}"
              f"{fmt(r['buy_hold'], '{:+.0f}%'):>10}{fmt(r['max_dd'], '{:.0f}%'):>8}"
              f"{r['loss_streak']:>8}{fmt(r['time_in'], '{:.0f}%'):>7}"
              f"{'  (trade open)' if r['open_trade'] else ''}")

    print("\nLatest crossover strength")
    print(f"{'Ticker':<7}{'Signal':<7}{'Crossed':<12}{'DaysAgo':>8}{'GapNow':>8}{'Gap':>10}"
          f"{'Vol x avg':>10}{'Px vs SMA' + str(a.short):>12}")
    for t, _ in rows:
        st = strength(data[t], a.short, a.long)
        if st is None:
            continue
        trend = "widening" if (st["gap_trend"] > 0) == (st["signal"] == "BUY") else "narrowing"
        print(f"{t:<7}{st['signal']:<7}{st['crossed']!s:<12}{st['days_ago']:>8}"
              f"{st['gap_now']:>+7.2f}%{trend:>10}{st['vol_ratio']:>9.1f}x{st['price_vs_short']:>+11.1f}%")

    print("\nColumns: Win% = share of trades that made money; AvgTrd = average trade return;"
          "\n  Strategy vs Buy&Hold = total return following the rule vs just holding;"
          "\n  MaxDD = worst peak-to-trough drop, measured trade to trade (drops during a trade"
          "\n  can be deeper); LossRun = most losing trades in a row; (trade open) = still holding"
          "\n  or a SELL that executes at the next open;"
          "\n  InMkt = share of time holding shares; Gap = whether the averages are moving apart;"
          "\n  Vol x avg = volume on the crossover day vs the 20 days before.")
    print("Past results only - not a prediction or investment advice.")


if __name__ == "__main__":
    main()
