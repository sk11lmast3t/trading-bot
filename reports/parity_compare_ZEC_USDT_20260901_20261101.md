# Paper/backtester comparison: ZEC/USDT

Window: 2026-09-01 00:00:00 through 2026-11-01 00:00:00 (end exclusive); strategy `mtf_trend_pullback`; scenario `market`.

Replay clock advances to each five-minute candle close. Only completed CSV candles are used. A flat forming row repeats the latest completed close as the sentinel that `process_symbol` drops before analysis and as the close mark `check_exits` reads. It contains no future OHLC information.

## Nearest same-side entry offset histogram

| Bot entry time minus Backtester entry time | Pairs |
|---|---:|
| +5 min | 118 |

Expected offset is exactly +5 minutes. Matching uses `(side, bot_entry_time - 5 minutes)` against `(side, backtester_entry_time)`.

### Nearest pairs not exactly +5 minutes

None.

All difference columns use bot minus Backtester. Exit-time residual subtracts the known +1 bar entry-stamp convention.

## Trade totals

| Path | Trades | All trades net bps/trade | Matched-only net bps/trade | Total net PnL |
|---|---:|---:|---:|---:|
| Bot | 118 | +14.74076 | +14.74076 | +1.72256441 |
| Backtester | 118 | +14.74076 | +14.74076 | +1.72256441 |

## Counts by side

| Path | Long | Short | Total |
|---|---:|---:|---:|
| Bot | 67 | 51 | 118 |
| Backtester | 67 | 51 | 118 |
| Matched | 67 | 51 | 118 |
| Bot unmatched | 0 | 0 | 0 |
| Backtester unmatched | 0 | 0 | 0 |

## Matched trade difference summary

Mean and median retain the sign. P95 and max are absolute difference magnitudes. `Nonzero` counts values above a small numeric tolerance.

| Metric | Unit | Count | Mean | Median | P95 abs | Max abs | Nonzero |
|---|---|---:|---:|---:|---:|---:|---:|
| Entry price diff (bot - backtester) | bps | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| SL diff (bot - backtester) | bps | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| TP diff (bot - backtester) | bps | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Position size diff (bot - backtester) | % | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Residual exit time diff (bot - backtester; after -1 bar) | bars | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Exit price diff (bot - backtester) | bps | 118 | -0.12711864 | +0.00000000 | +3.00000000 | +3.00000000 | 35 |
| Entry fee diff (bot - backtester) | quote | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Exit fee diff (bot - backtester) | quote | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Entry slippage diff (bot - backtester) | quote | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Exit slippage diff (bot - backtester) | quote | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Slippage diff (bot - backtester) | quote | 118 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Net PnL diff (bot - backtester) | bps of BT notional | 118 | +0.00000000 | -0.00000000 | +0.00000000 | +0.00000000 | 0 |

## Exit reason agreement matrix

Rows are bot exit reason; columns are Backtester exit reason.

| Bot \ Backtester | SL | TIME | TP |
|---|---:|---:|---:|
| SL | 24 | 0 | 0 |
| TIME | 0 | 83 | 0 |
| TP | 0 | 0 | 11 |

## Unmatched trade explanations

None.

## Mismatch causes

| Cause | Count |
|---|---:|
| bot close vs backtester high/low | 35 |
| entry timing offset | 0 |
| other exit-price mismatch | 0 |
| other exit-reason mismatch | 0 |
| other pnl mismatch | 0 |
| other: no normalized backtester trade | 0 |
| other: no normalized bot trade | 0 |
| sizing | 0 |
| slippage/fee formula | 35 |
| time-exit timing | 0 |

Mismatch detail CSV: `parity_compare_ZEC_USDT_20260901_20261101_mismatches.csv`.

For each unmatched entry the report records the signal bar, its OHLC values, and the other path's captured decision or explains that the decision was not captured. Exit mismatch causes use the backtest exit bar's high/low and separate time exits, size, and cost differences.

## Before/after touch-exit summary

Exit price mean retains its sign; exit price p95 and max are absolute magnitudes. Residual exit time is bot minus Backtester after subtracting the known +1 bar entry-stamp offset. Before values are the prior close-only replay summary with this same offset removed from the exit-time distribution.

| Measure | Before touch exits | After touch exits |
|---|---:|---:|
| Exit-reason disagreements | 6 | 0 |
| Exit price diff mean (bps; bot - Backtester) | +3.37709 | -0.12712 |
| Exit price diff p95 abs (bps) | 83.24070 | 3.00000 |
| Exit price diff max abs (bps) | 295.48419 | 3.00000 |
| Residual exit-time diff mean (bars; bot - Backtester) | +1.98291 | 0.00000 |
| Residual exit-time diff p95 abs (bars) | 8.40000 | 0.00000 |
| Residual exit-time diff max abs (bars) | 31.00000 | 0.00000 |
