import re

path = r"C:\Users\webdeveloper\Downloads\rea-main\unified-crypto-scalper\main.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

close_pos = """    def _close_position(self, symbol, pos, price, reason):
        entry = float(pos.get('entryPrice', 0.0))
        contracts = float(pos.get('contracts', 0.0))
        side = pos.get('side', 'long')
        if side == 'long':
            pnl = (price - entry) * contracts
        else:
            pnl = (entry - price) * contracts
        logger.info(f"[{symbol}] Closed {side.upper()} position: {reason} | PnL: ${pnl:.4f}")
        if self.config.mode == "live":
            opp_side = 'sell' if side == 'long' else 'buy'
            self.exchange.create_market_order(symbol, opp_side, contracts)
            try:
                self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass
        else:
            self.exchange.close_position(symbol, price)
        self.risk_manager.record_trade(pnl, symbol)
        self.risk_manager.release_position(symbol)
        self.position_entry_cycle.pop(symbol, None)
        self.position_targets.pop(symbol, None)

"""

new_check = """    def check_exits(self, symbol: str):
        try:
            pos = self.exchange.get_position(symbol)
            if not pos:
                return

            df = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, 5)
            if df is None or len(df) == 0:
                return

            price = df.iloc[-1]['close']
            entry = float(pos.get('entryPrice', 0.0))
            side = pos.get('side', 'long')
            sc = self.config.scalping
            targets = self.position_targets.get(symbol, {})
            tp_price = targets.get('tp')
            sl_price = targets.get('sl')

            self.risk_manager.update_position_price(symbol, price)

            # 1. Hard SL (never skipped)
            if side == 'long':
                hit_sl = (price <= sl_price) if (sl_price and sl_price < entry) else (price <= entry * (1 - sc.stop_loss_pct))
            else:
                hit_sl = (price >= sl_price) if (sl_price and sl_price > entry) else (price >= entry * (1 + sc.stop_loss_pct))
            if hit_sl:
                self._close_position(symbol, pos, price, "STOP LOSS")
                return

            # 2. TP
            if side == 'long':
                hit_tp = (price >= tp_price) if (tp_price and tp_price > entry) else (price >= entry * (1 + sc.take_profit_pct))
            else:
                hit_tp = (price <= tp_price) if (tp_price and tp_price < entry) else (price <= entry * (1 - sc.take_profit_pct))
            if hit_tp:
                self._close_position(symbol, pos, price, "TAKE PROFIT")
                return

            # 3. Trailing / breakeven (isolated so failure can't skip SL above)
            try:
                trail_hit, trail_reason = self.risk_manager.check_trailing_stop(symbol, price)
                if trail_hit:
                    self._close_position(symbol, pos, price, f"TRAILING STOP: {trail_reason}")
                    return
            except Exception as e:
                logger.error(f"Trailing check error for {symbol}: {e}")

            # 4. Time exit (using cycle count for now; 1.6 migrates to wall clock)
            entry_cycle = self.position_entry_cycle.get(symbol, self.cycle_count)
            max_hold = getattr(sc, 'max_hold_cycles', 60)
            if self.cycle_count - entry_cycle > max_hold:
                self._close_position(symbol, pos, price, "TIME EXIT")
                return

        except Exception as e:
            logger.error(f"Error checking exits for {symbol}: {e}")

"""

pattern = re.compile(r"    def check_exits\(self, symbol: str\):.*?    def print_status\(self\):", re.DOTALL)
new_src, n = pattern.subn(close_pos + new_check + "    def print_status(self):", src, count=1)
assert n == 1, f"replacements: {n}"
with open(path, "w", encoding="utf-8") as f:
    f.write(new_src)
print("done")
