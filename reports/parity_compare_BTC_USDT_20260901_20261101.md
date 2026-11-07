# Paper/backtester comparison: BTC/USDT

Window: 2026-09-01 00:00:00 through 2026-11-01 00:00:00 (end exclusive); strategy `mtf_trend_pullback`; scenario `market`.

Replay clock advances to each five-minute candle close. Only completed CSV candles are used. A flat forming row repeats the latest completed close as the sentinel that `process_symbol` drops before analysis and as the close mark `check_exits` reads. It contains no future OHLC information.

## Nearest same-side entry offset histogram

| Bot entry time minus Backtester entry time | Pairs |
|---|---:|
| +20 min | 1 |
| +5 min | 87 |

Expected offset is exactly +5 minutes. Matching uses `(side, bot_entry_time - 5 minutes)` against `(side, backtester_entry_time)`.

### Nearest pairs not exactly +5 minutes

- short bot=2026-10-24 07:25:00, backtester=2026-10-24 07:05:00, offset=+20 minutes; not normalized into a match.

## Trade totals

| Path | Trades | Net bps per trade (mean) | Total net PnL |
|---|---:|---:|---:|
| Bot | 88 | -18.98263 | -1.65806174 |
| Backtester | 90 | -16.13439 | -1.44302147 |

## Counts by side

| Path | Long | Short | Total |
|---|---:|---:|---:|
| Bot | 40 | 48 | 88 |
| Backtester | 41 | 49 | 90 |
| Matched | 40 | 47 | 87 |
| Bot unmatched | 0 | 1 | 1 |
| Backtester unmatched | 1 | 2 | 3 |

## Matched trade difference summary

Mean and median retain the sign. P95 and max are absolute difference magnitudes. `Nonzero` counts values above a small numeric tolerance.

| Metric | Unit | Count | Mean | Median | P95 abs | Max abs | Nonzero |
|---|---|---:|---:|---:|---:|---:|---:|
| Entry price diff | bps | 87 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| SL diff | bps | 87 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| TP diff | bps | 87 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Position size diff | % | 87 | -0.09797020 | -0.12175782 | +0.20195615 | +0.20988082 | 86 |
| Exit time diff | bars | 87 | +2.83908046 | +2.00000000 | +5.40000000 | +32.00000000 | 87 |
| Exit price diff | bps | 87 | +0.03817480 | +0.02019321 | +23.33927987 | +42.48736806 | 87 |
| Entry fee diff | quote | 87 | -0.00000484 | -0.00000602 | +0.00001000 | +0.00001039 | 86 |
| Exit fee diff | quote | 87 | -0.00000482 | -0.00000468 | +0.00001548 | +0.00002446 | 87 |
| Entry slippage diff | quote | 87 | -0.00000291 | -0.00000361 | +0.00000600 | +0.00000623 | 86 |
| Exit slippage diff | quote | 87 | -0.00000289 | -0.00000281 | +0.00000929 | +0.00001467 | 87 |
| Slippage diff | quote | 87 | -0.00000580 | -0.00000360 | +0.00001511 | +0.00001846 | 87 |
| Net PnL diff | bps of BT notional | 87 | -0.82143145 | -0.61809590 | +23.51319545 | +42.52171168 | 87 |

## Exit reason agreement matrix

Rows are bot exit reason; columns are Backtester exit reason.

| Bot \ Backtester | SL | TIME | TP |
|---|---:|---:|---:|
| SL | 17 | 0 | 0 |
| TIME | 1 | 64 | 0 |
| TP | 0 | 0 | 5 |

## Unmatched trade explanations

- short 2026-10-24 07:20:00: signal bar 2026-10-24 07:20:00; close/high/low=O=80922,H=80922,L=80793.2,C=80793.2; strategy not evaluated; backtester position open from 2026-10-24 07:05:00 through 2026-10-24 11:05:00
- long 2026-10-02 01:45:00: signal bar 2026-10-02 01:45:00; close/high/low=O=84578,H=84671.4,L=84577.1,C=84666.7; signal=LONG, status=skipped:position_open, entry=84666.72, SL=84007.99589416727, TP=85984.16821166547
- short 2026-09-10 04:10:00: signal bar 2026-09-10 04:10:00; close/high/low=O=78373.1,H=78377.8,L=78344.9,C=78344.9; signal=SHORT, status=skipped:position_open, entry=78344.86, SL=78991.75540254396, TP=77051.06919491208
- short 2026-10-24 07:05:00: signal bar 2026-10-24 07:05:00; close/high/low=O=80922,H=80922,L=80793.2,C=80793.2; signal=SHORT, status=skipped:position_open, entry=80793.22, SL=81309.9862184404, TP=79759.6875631192

## Mismatch causes

| Cause | Count |
|---|---:|
| bot close vs backtester high/low | 23 |
| entry timing offset | 1 |
| other exit-price mismatch | 0 |
| other exit-reason mismatch | 0 |
| other pnl mismatch | 0 |
| other: no normalized backtester trade | 1 |
| other: no normalized bot trade | 3 |
| sizing | 86 |
| slippage/fee formula | 0 |
| time-exit timing | 65 |

Mismatch detail CSV: `parity_compare_BTC_USDT_20260901_20261101_mismatches.csv`.

For each unmatched entry the report records the signal bar, its OHLC values, and the other path's captured decision or explains that the decision was not captured. Exit mismatch causes use the backtest exit bar's high/low and separate time exits, size, and cost differences.
