"""Download 5m OHLCV history for the most liquid USDT pairs into data/*.csv.

Run this on YOUR machine (it needs exchange network access):

    pip install ccxt pandas
    python scripts/download_data.py                       # 15 months, top 12 USDT pairs
    python scripts/download_data.py --months 18 --top 15
    python scripts/download_data.py --symbols BTC/USDT ETH/USDT SOL/USDT

Resumable: re-running continues from the last saved candle of each file.
The newest (still forming) candle is always dropped. A coverage line per symbol
shows missing bars; files below --min-coverage are flagged.

NOTE (survivorship bias): symbols are ranked by CURRENT 24h volume, so the list is
biased towards coins that are big today. Mention this when you read the report.
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

STABLES = {"USDC", "TUSD", "BUSD", "FDUSD", "USDP", "DAI", "USTC", "RLUSD", "EUR", "AEUR",
           "EURI", "USD1", "UST", "PAXG", "XAUT", "WBTC", "WBETH", "BFUSD", "USDE", "USDS"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR|[235]L|[235]S)$")
TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def rank_symbols(exchange, top, exclude):
    tickers = exchange.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or ":" in sym:          # spot USDT pairs only
            continue
        base = sym.split("/")[0]
        if base in STABLES or LEVERAGED.search(base) or sym in exclude:
            continue
        vol = t.get("quoteVolume") or 0
        if vol:
            rows.append((sym, float(vol)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [s for s, _ in rows[:top]]


def csv_path(out_dir, symbol, timeframe):
    return os.path.join(out_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")


def fetch_with_retry(exchange, symbol, timeframe, since, limit, retries=6):
    import ccxt
    delay = 1.0
    for attempt in range(retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable) as e:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries - 1} after error: {type(e).__name__}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def download_symbol(exchange, symbol, timeframe, months, out_dir, limit=1000):
    path = csv_path(out_dir, symbol, timeframe)
    step = TF_MS[timeframe]
    now_ms = exchange.milliseconds()
    start_ms = int((pd.Timestamp(now_ms, unit="ms") - pd.DateOffset(months=months)).value // 10**6)
    since = start_ms
    existing = None
    if os.path.exists(path):
        existing = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if len(existing):
            since = max(since, int(existing.index[-1].value // 10**6) + step)
    rows = []
    while since < now_ms:
        batch = fetch_with_retry(exchange, symbol, timeframe, since, limit)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= since:
            break
        since = nxt
        if len(rows) % (limit * 20) < limit:
            print(f"    {symbol}: {len(rows):,} new candles...", flush=True)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
    if existing is not None and len(existing):
        df = pd.concat([existing, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df):
        df = df.iloc[:-1]                                    # drop the still-forming candle
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path)
    return df


def coverage(df, timeframe):
    if len(df) < 2:
        return 0, 0.0, 0
    step = pd.Timedelta(milliseconds=TF_MS[timeframe])
    expected = int((df.index[-1] - df.index[0]) / step) + 1
    missing = expected - len(df)
    return expected, len(df) / expected, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--timeframe", default="5m", choices=sorted(TF_MS))
    ap.add_argument("--months", type=int, default=15, help="history to fetch (default 15)")
    ap.add_argument("--top", type=int, default=12, help="how many volume-ranked USDT pairs (default 12)")
    ap.add_argument("--symbols", nargs="*", help="explicit symbols instead of ranking")
    ap.add_argument("--exclude", nargs="*", default=["PEPE/USDT"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-coverage", type=float, default=0.98)
    args = ap.parse_args()

    try:
        import ccxt
    except ImportError:
        sys.exit("ccxt is not installed: pip install ccxt")
    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})

    symbols = args.symbols or rank_symbols(exchange, args.top, set(args.exclude))
    if not symbols:
        sys.exit("No symbols found (is the exchange reachable?)")
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
    problems = []
    for sym in symbols:
        print(f"== {sym}", flush=True)
        try:
            df = download_symbol(exchange, sym, args.timeframe, args.months, args.out)
        except Exception as e:                                # keep going; report at the end
            print(f"   FAILED: {type(e).__name__}: {e}")
            problems.append((sym, f"download failed: {e}"))
            continue
        exp, cov, miss = coverage(df, args.timeframe)
        span = f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}" if len(df) else "empty"
        flag = "" if cov >= args.min_coverage else "   <-- LOW COVERAGE"
        print(f"   {len(df):,} candles, {span}, coverage {cov:.2%}, missing {miss:,}{flag}")
        if cov < args.min_coverage:
            problems.append((sym, f"coverage {cov:.2%}"))
        if len(df) and (df.index[-1] - df.index[0]).days < args.months * 28:
            problems.append((sym, f"history only {(df.index[-1] - df.index[0]).days} days "
                                  f"(listed later than requested start)"))
    print()
    if problems:
        print("Check these before trusting results:")
        for sym, why in problems:
            print(f"  - {sym}: {why}")
    else:
        print("All symbols complete.")
    print(f"Files written to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
