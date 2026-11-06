# auto-trader

An AI-assisted stock picker for an **Alpaca paper trading** account (simulated money).
It screens a list of chip and tech stocks, has three Claude agents debate the best
candidates, and buys the judge's top BUY.

## What it does each trading day

1. **Exits** - for positions it opened itself, sells if the loss reaches `stop_loss_pct`
   or the 20-day moving average falls below the 50-day.
2. **Screen** - every stock in `universe` must pass four checks:
   - **uptrend:** SMA20 above SMA50
   - **firming:** the gap is widening and SMA50 is rising
   - **momentum:** up over the last month
   - **backtest:** the crossover rule returned at least `min_backtest_pct` over 5 years

   Passing stocks are ranked by 1-month performance against the S&P 500.
3. **Debate** - the top `max_debates_per_day` go to:
   - **bull-agent** - the strongest honest case for buying
   - **bear-agent** - the strongest honest case against
   - **judge-agent** - weighs both and returns BUY / SELL / HOLD with a 0-100 confidence

   The agents see prices, moving averages, the backtest, earnings and dividend dates,
   analyst estimates and revisions, price targets, recent headlines and performance vs
   SPY and the sector ETF. With `web_search` on, bull and bear also search the web for
   news, upcoming events and the macro calendar, and cite sources.
4. **Buy** - the highest-confidence BUY at or above `min_confidence`, for about
   `trade_dollars`. At most one new position per day and `max_positions` in total.
   No qualifying verdict means no trade.

Paper trading is hard-coded (`paper=True`). Long only, market orders.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # paste your Alpaca PAPER keys and an Anthropic API key
```

Get an Anthropic API key at console.anthropic.com. Debates use Claude Opus 5.5.

## Usage

```bash
python3 auto_trader.py --screen-only   # free: exits check + screen, no Claude calls, no orders
python3 auto_trader.py --dry-run       # full run with debates (costs credits), no orders
python3 auto_trader.py                 # full run, places paper orders

python3 debate.py FDX                  # one-off debate, report only
python3 debate.py FDX SPG --web        # with web search
python3 debate.py FDX --dry-run        # show the agents' data, no API calls
python3 backtest.py NVDA AMD --years 5 # crossover backtest
python3 check_account.py               # balances and positions
```

## Cost

Debates only run on days when a stock passes the screen. A debate costs about
$0.17 without web search and $0.55-0.75 with it, so a day with 3 debates is roughly
$2.25. Each report shows its actual cost.

## Running it every trading day (macOS)

`launchd/com.tradingbot.autotrader.plist` wakes the bot at :30 past every hour; in
`--scheduled` mode it runs once per trading day while the market is open. Edit the paths
in the plist if your folder or Python is elsewhere.

```bash
cp launchd/com.tradingbot.autotrader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tradingbot.autotrader.plist
```

Output: `logs/auto_trader.log` (actions), `reports/<date>/` (debates),
`auto_state.json` (positions it manages).

## Disclaimer

For learning with simulated money. Verdicts are AI-generated opinions based on past data
and public information. Not financial advice.
