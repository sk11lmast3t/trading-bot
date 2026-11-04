# Backtest Report - ema_rsi_atr, grid_scalper, combined

Period: 2026-09-29 to 2026-10-29 (30 days)  
Symbols: BTC/USDT, ETH/USDT, SOL/USDT (5m)  
Initial balance: 1000.0, REAL_FEES: commission 0.001, slippage 0.0005

## Per-symbol results (REAL_FEES)

| Strategy | Symbol | Trades | Win Rate | Net PnL | Sharpe | Max DD |
|---------|--------|--------|----------|---------|--------|--------|
| grid_scalper | BTC/USDT | 325 | 33.23% | -50.34 | -115.20 | -8.15% |
| grid_scalper | ETH/USDT | 323 | 39.63% | -50.18 | -102.80 | -8.10% |
| grid_scalper | SOL/USDT | 316 | 45.57% | -50.03 | -81.88 | -8.03% |
| ema_rsi_atr | BTC/USDT | 341 | 21.11% | -50.17 | -112.01 | -8.27% |
| ema_rsi_atr | ETH/USDT | 356 | 24.16% | -49.26 | -78.55 | -8.36% |
| ema_rsi_atr | SOL/USDT | 315 | 24.44% | -49.81 | -68.98 | -8.02% |
| combined | BTC/USDT | 306 | 29.08% | -49.07 | -104.81 | -7.90% |
| combined | ETH/USDT | 256 | 29.30% | -50.96 | -103.46 | -7.51% |
| combined | SOL/USDT | 342 | 43.27% | -50.08 | -64.90 | -8.24% |

## Aggregate (REAL_FEES, summed across symbols)

| Strategy | Total Trades | Total Net PnL |
|---------|--------------|---------------|
| grid_scalper | 964 | -150.55 |
| ema_rsi_atr | 1012 | -149.24 |
| combined | 904 | -150.11 |

## Expectancy

Net expectancy = (avg win * win rate) - (avg loss * loss rate). With negative total PnL and similar magnitude losses across symbols/strategies, **no strategy shows positive net expectancy under real fees**.

All strategies are loss-making with real fees; breakeven only appears in isolated zero-fee cases (not representative). Max drawdowns ~7.5-8.4%, win rates generally low except grid_scalper on SOL (45.57%) which is still unprofitable after fees.
