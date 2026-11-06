"""Check the connection to an Alpaca paper trading account.

Reads APCA_API_KEY_ID and APCA_API_SECRET_KEY from the environment or from a
.env file next to this script, then prints buying power, cash and positions.
Read-only: it never places orders.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://paper-api.alpaca.markets/v2"


def load_env():
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get(path, key, secret):
    req = urllib.request.Request(
        BASE_URL + path,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def main():
    load_env()
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY (set them in .env).")

    try:
        account = get("/account", key, secret)
        positions = get("/positions", key, secret)
    except urllib.error.HTTPError as e:
        sys.exit(f"Connection failed: HTTP {e.code} {e.read().decode()[:200]}")

    print("Connected to Alpaca paper account", account["account_number"])
    print(f"  Status:        {account['status']}")
    print(f"  Cash:          ${float(account['cash']):,.2f}")
    print(f"  Buying power:  ${float(account['buying_power']):,.2f}")
    print(f"  Equity:        ${float(account['equity']):,.2f}")
    print()
    if not positions:
        print("No open positions.")
    else:
        print(f"{'Symbol':<8}{'Qty':>10}{'Avg cost':>12}{'Price':>12}{'Mkt value':>14}{'P/L':>12}")
        for p in positions:
            print(
                f"{p['symbol']:<8}{float(p['qty']):>10g}"
                f"{float(p['avg_entry_price']):>12,.2f}{float(p['current_price']):>12,.2f}"
                f"{float(p['market_value']):>14,.2f}{float(p['unrealized_pl']):>12,.2f}"
            )


if __name__ == "__main__":
    main()
