"""Auto-trader: screen chip and tech stocks, debate the best candidates, and paper-trade
the judge's top BUY. Alpaca PAPER account only.

Each run (once per trading day when scheduled):
  1. EXITS - for positions this auto-trader opened (never the AAPL bot's, never ones
     you opened by hand), sell if the loss reaches stop_loss_pct or the SMA20 has
     crossed below the SMA50.
  2. SCREEN - check every stock in config.json "universe" against the checklist:
        uptrend    SMA20 above SMA50
        firming    the gap is widening and the SMA50 is rising
        momentum   up over the last month
        backtest   the crossover rule returned at least min_backtest_pct over 5 years
     Passing stocks are ranked by 1-month performance against the S&P 500.
  3. DEBATE - the top max_debates_per_day candidates go to bull-agent vs bear-agent
     (with web search if enabled), and judge-agent gives a verdict.
  4. BUY - the highest-confidence BUY verdict at or above min_confidence is bought
     for about trade_dollars, as long as fewer than max_positions are open.
     At most one new position per day. No BUY verdict means no trade.

Safety limits:
  - paper=True is hard-coded; there is no live-trading option.
  - Long only, market orders, one buy per day, position count and size capped.
  - Skips a stock with an open order, an existing position, or one in exclude_symbols
    (AAPL by default, since the separate crossover-bot project trades it).

Usage:
  python3 auto_trader.py --screen-only   # steps 1-2 only: free, no Claude calls, no orders
  python3 auto_trader.py --dry-run       # full run with debates (costs Claude credits), no orders
  python3 auto_trader.py                 # full run, places paper orders
  python3 auto_trader.py --scheduled     # for launchd: once per trading day, while the market is open
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

import backtest as bt
import debate as dbt

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "auto_state.json"
ET = ZoneInfo("America/New_York")

(HERE / "logs").mkdir(exist_ok=True)
log = logging.getLogger("auto_trader")
log.setLevel(logging.INFO)
log.propagate = False
for h in (logging.StreamHandler(), logging.FileHandler(HERE / "logs" / "auto_trader.log")):
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)


# ---------------------------------------------------------------------- state

def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {"positions": {}}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def market_is_open(client, attempts=3, wait_seconds=10):
    for attempt in range(attempts):
        try:
            return client.get_clock().is_open
        except Exception:
            if attempt < attempts - 1:
                time.sleep(wait_seconds)
    return None


def open_order_symbols(client):
    return {o.symbol for o in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))}


# ---------------------------------------------------------------------- exits

def manage_exits(client, state, data_by_sym, short_w, long_w, stop_pct, dry_run):
    positions = {p.symbol: p for p in client.get_all_positions()}
    pending = open_order_symbols(client)
    for sym in list(state["positions"]):
        pos = positions.get(sym)
        if pos is None:
            if sym in pending:
                log.info("EXIT %s: buy order not filled yet.", sym)
            else:
                log.info("EXIT %s: no longer held (sold elsewhere?). Removing from tracking.", sym)
                state["positions"].pop(sym)
            continue
        if sym in pending:
            log.info("EXIT %s: an order is already open, skipping.", sym)
            continue
        loss_pct = float(pos.unrealized_plpc) * 100
        reason = None
        if loss_pct <= -stop_pct:
            reason = f"stop loss: {loss_pct:.1f}% (limit -{stop_pct}%)"
        else:
            df = data_by_sym.get(sym)
            if df is not None:
                c = df["Close"]
                s, l = c.rolling(short_w).mean().iloc[-1], c.rolling(long_w).mean().iloc[-1]
                if s < l:
                    reason = f"SMA{short_w} {s:.2f} is below SMA{long_w} {l:.2f}"
        if reason is None:
            log.info("EXIT %s: keep (P/L %+.1f%%).", sym, loss_pct)
            continue
        qty = float(pos.qty)
        if dry_run:
            log.info("EXIT %s: [dry run] would sell %g shares - %s.", sym, qty, reason)
            continue
        order = client.submit_order(MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
        log.info("EXIT %s: SELL %g shares, order %s - %s.", sym, qty, order.id, reason)
        state["positions"].pop(sym)


# --------------------------------------------------------------------- screen

def screen(data_by_sym, short_w, long_w, exclude, min_backtest_pct):
    spy = data_by_sym.get("SPY")
    spy_1m = (spy["Close"].iloc[-1] / spy["Close"].iloc[-22] - 1) * 100 if spy is not None else 0
    passed, rows = [], []
    for sym, df in data_by_sym.items():
        if sym == "SPY" or len(df) < 300:
            continue
        c = df["Close"]
        s, l = c.rolling(short_w).mean(), c.rolling(long_w).mean()
        st = bt.strength(df, short_w, long_w)
        r = bt.backtest(df, short_w, long_w, years=5)
        m1 = (c.iloc[-1] / c.iloc[-22] - 1) * 100
        checks = {
            "uptrend": bool(s.iloc[-1] > l.iloc[-1]),
            "firming": bool(st and st["signal"] == "BUY" and st["gap_trend"] > 0
                            and l.iloc[-1] > l.iloc[-6]),
            "momentum": bool(m1 > 0),
            "backtest": bool(r["strategy"] >= min_backtest_pct),
        }
        row = {"symbol": sym, "checks": checks, "m1": m1, "vs_spy_1m": m1 - spy_1m,
               "backtest": r["strategy"], "excluded": sym in exclude}
        rows.append(row)
        if all(checks.values()) and not row["excluded"]:
            passed.append(row)
    passed.sort(key=lambda x: x["vs_spy_1m"], reverse=True)
    return passed, rows


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Screen, debate and paper-trade (Alpaca paper only)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--screen-only", action="store_true", help="exits check + screen; no Claude calls, no orders")
    mode.add_argument("--dry-run", action="store_true", help="full run with debates, no orders")
    p.add_argument("--scheduled", action="store_true", help="once per trading day, while the market is open")
    a = p.parse_args()

    dbt.load_env(HERE / ".env")
    cfg = json.loads((HERE / "config.json").read_text())
    at = cfg["auto_trader"]
    short_w, long_w = cfg["short_window"], cfg["long_window"]
    no_orders = a.dry_run or a.screen_only

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env.")
    alpaca = TradingClient(key, secret, paper=True)  # paper only, by design

    state = load_state()
    today = datetime.now(ET).date().isoformat()
    if a.scheduled:
        if state.get("last_run_date") == today:
            return
        is_open = market_is_open(alpaca)
        if is_open is None:
            log.warning("Couldn't reach Alpaca to check the market clock. Will retry next hourly wake.")
            return
        if not is_open:
            return

    log.info("===== Auto-trader run (%s ET%s) =====", datetime.now(ET).strftime("%a %Y-%m-%d %H:%M"),
             ", screen only" if a.screen_only else ", dry run" if a.dry_run else "")
    universe = [s.upper() for s in at["universe"]]
    data_by_sym = bt.download(universe + ["SPY"], years=5)
    missing = [s for s in universe if s not in data_by_sym]
    if missing:
        log.info("No price data for: %s", " ".join(missing))

    # 1. exits
    manage_exits(alpaca, state, data_by_sym, short_w, long_w, at["stop_loss_pct"], no_orders)

    # 2. screen
    held = {pos.symbol for pos in alpaca.get_all_positions()}
    exclude = held | open_order_symbols(alpaca) | {s.upper() for s in at.get("exclude_symbols", [])}
    passed, rows = screen(data_by_sym, short_w, long_w, exclude, at.get("min_backtest_pct", 10))
    uptrend = sum(r["checks"]["uptrend"] for r in rows)
    log.info("SCREEN: %d stocks, %d in uptrend, %d passed all checks%s.", len(rows), uptrend, len(passed),
             (": " + ", ".join(f"{r['symbol']} ({r['vs_spy_1m']:+.1f}% vs SPY)" for r in passed)) if passed else "")
    near = sorted((r for r in rows if sum(r["checks"].values()) == 3 and not r["excluded"]),
                  key=lambda r: r["vs_spy_1m"], reverse=True)
    for r in near[:5]:
        failed = [k for k, v in r["checks"].items() if not v][0]
        log.info("SCREEN: near miss %s (failed %s)", r["symbol"], failed)

    if a.screen_only:
        log.info("Screen only: stopping before debates.")
        log.info("===== Run finished =====")
        return

    # 3. debate
    open_count = len(state["positions"])
    if open_count >= at["max_positions"]:
        log.info("BUY: already at max_positions (%d). No new trades.", at["max_positions"])
        candidates = []
    else:
        candidates = passed[: at["max_debates_per_day"]]
    if not candidates and open_count < at["max_positions"]:
        log.info("BUY: no stock passed the checklist. No trade today.")

    if candidates and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Add ANTHROPIC_API_KEY to .env to run debates.")
    claude = dbt.anthropic.Anthropic() if candidates else None
    verdicts, total_cost = [], 0.0
    for r in candidates:
        sym = r["symbol"]
        packet = dbt.market_packet(sym, data_by_sym[sym], short_w, long_w)
        try:
            result, usage = dbt.debate(claude, packet, rebuttal=at.get("rebuttal", True),
                                       web=at.get("web_search", True))
        except RuntimeError as e:
            log.warning("DEBATE %s failed: %s", sym, e)
            continue
        path = dbt.write_report(packet, result, usage)
        v = result["verdict"]
        total_cost += usage.cost
        log.info("DEBATE %s: %s %d%% (stronger: %s, ~$%.2f) - %s", sym, v["verdict"], v["confidence"],
                 v["stronger_case"], usage.cost, path.name)
        verdicts.append((sym, v, packet["price"]["close"], path))

    # 4. buy
    buys = [x for x in verdicts if x[1]["verdict"] == "BUY" and x[1]["confidence"] >= at["min_confidence"]]
    buys.sort(key=lambda x: x[1]["confidence"], reverse=True)
    if verdicts and not buys:
        log.info("BUY: no BUY verdict at %d%%+ confidence. No trade today.", at["min_confidence"])
    if buys:
        sym, v, price, path = buys[0]
        qty = math.floor(at["trade_dollars"] / price)
        cash = float(alpaca.get_account().cash)
        if qty < 1:
            log.info("BUY %s: price $%.2f is above trade_dollars. Skipping.", sym, price)
        elif qty * price > cash:
            log.info("BUY %s: not enough cash ($%.2f). Skipping.", sym, cash)
        elif no_orders:
            log.info("BUY %s: [dry run] would buy %d shares (~$%.0f) - judge %s %d%%.",
                     sym, qty, qty * price, v["verdict"], v["confidence"])
        else:
            order = alpaca.submit_order(MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            state["positions"][sym] = {"opened": today, "qty": qty, "est_price": price,
                                       "order_id": str(order.id), "confidence": v["confidence"],
                                       "report": str(path)}
            log.info("BUY %s: %d shares (~$%.0f), order %s - judge BUY %d%%. Report: %s",
                     sym, qty, qty * price, order.id, v["confidence"], path.name)

    if total_cost:
        log.info("Claude cost this run: ~$%.2f", total_cost)
    if not no_orders:
        state["last_run_date"] = today
        save_state(state)
    elif a.scheduled and a.dry_run:  # don't re-debate (and re-spend) every hour
        dry = load_state()
        dry["last_run_date"] = today
        save_state(dry)
    log.info("===== Run finished =====")


if __name__ == "__main__":
    main()
