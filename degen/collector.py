"""Record every launch and scheduled token outcome from an EventSource."""
from __future__ import annotations

from datetime import timedelta

from degen.source import EventSource


SNAPSHOTS = ((60, "1m"), (300, "5m"), (900, "15m"), (3600, "1h"),
             (21600, "6h"), (86400, "24h"))


def _text(value):
    return str(value or "")[:200]


class Collector:
    def __init__(self, source: EventSource, store):
        self.source, self.store = source, store

    def collect(self):
        launches, trades = {}, {}
        pending = {}
        for event in self.source.events():
            if event.kind == "launch":
                launch_price = None
                launches[event.mint] = {"ts": event.ts, "price": launch_price}
                trades[event.mint] = []
                pending[event.mint] = set(seconds for seconds, _ in SNAPSHOTS)
                with self.store.writer() as db:
                    db.execute("INSERT OR IGNORE INTO launches(mint,ts,creator,name,symbol,description,url,launch_price) VALUES(?,?,?,?,?,?,?,?)",
                               (event.mint, event.ts.isoformat(), _text(event.creator), _text(event.name),
                                _text(event.symbol), _text(event.description), _text(event.url), launch_price))
            elif event.kind == "trade":
                if event.mint not in launches:
                    continue
                amount = float(event.sol_amount)
                tokens = float(event.token_amount)
                price = amount / tokens if tokens > 0 else 0.0
                trades[event.mint].append((event, price))
                if launches[event.mint]["price"] is None and price > 0:
                    launches[event.mint]["price"] = price
                    with self.store.writer() as db:
                        db.execute("UPDATE launches SET launch_price=? WHERE mint=? AND launch_price IS NULL",
                                   (price, event.mint))
                with self.store.writer() as db:
                    db.execute("INSERT INTO trades(mint,ts,side,sol_amount,token_amount,wallet,price) VALUES(?,?,?,?,?,?,?)",
                               (event.mint, event.ts.isoformat(), event.side.lower(), amount, tokens,
                                _text(event.wallet), price))
            else:
                raise ValueError(f"Unsupported event kind: {event.kind}")
            # A chronological source advances scheduled observations. At EOF,
            # flush below so every launch receives every horizon even if quiet.
            for mint, record in tuple(launches.items()):
                due = [s for s in pending[mint] if event.ts >= record["ts"] + timedelta(seconds=s)]
                for seconds in due:
                    self._snapshot(mint, seconds, record, trades[mint])
                    pending[mint].remove(seconds)
        for mint, record in launches.items():
            for seconds in sorted(pending[mint]):
                self._snapshot(mint, seconds, record, trades[mint])
        return len(launches)

    def _snapshot(self, mint, seconds, launch, token_trades):
        at = launch["ts"] + timedelta(seconds=seconds)
        visible = [(event, price) for event, price in token_trades if event.ts <= at]
        buys = sum(event.side.lower() == "buy" for event, _ in visible)
        sells = sum(event.side.lower() == "sell" for event, _ in visible)
        buyers = {event.wallet for event, _ in visible if event.side.lower() == "buy" and event.wallet}
        last_trade = visible[-1][0].ts if visible else launch["ts"]
        alive = int(at - last_trade < timedelta(minutes=30))
        initial = launch["price"]
        price_multiple = (visible[-1][1] / initial) if initial and visible and visible[-1][1] > 0 else None
        holder_share = self.source.top_holder_share(mint, at)
        if holder_share is not None:
            holder_share = max(0.0, min(1.0, float(holder_share)))
        with self.store.writer() as db:
            db.execute("INSERT OR IGNORE INTO snapshots(mint,offset_seconds,ts,trade_count,buy_sell_ratio,unique_buyers,top10_holder_share,price_vs_launch,sol_volume,alive) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (mint, seconds, at.isoformat(), len(visible),
                        buys / sells if sells else None, len(buyers),
                        holder_share, price_multiple,
                        sum(float(event.sol_amount) for event, _ in visible), alive))
