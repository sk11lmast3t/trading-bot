# Paper/backtester comparison: ZEC/USDT

Window: 2026-09-01 00:00:00 through 2026-11-01 00:00:00 (end exclusive); strategy `mtf_trend_pullback`; scenario `market`.

Replay clock advances to each five-minute candle close. Only completed CSV candles are used. A flat forming row repeats the latest completed close as the sentinel that `process_symbol` drops before analysis and as the close mark `check_exits` reads. It contains no future OHLC information.

## Nearest same-side entry offset histogram

| Bot entry time minus Backtester entry time | Pairs |
|---|---:|
| +5 min | 117 |

Expected offset is exactly +5 minutes. Matching uses `(side, bot_entry_time - 5 minutes)` against `(side, backtester_entry_time)`.

### Nearest pairs not exactly +5 minutes

None.

## Trade totals

| Path | Trades | Net bps per trade (mean) | Total net PnL |
|---|---:|---:|---:|
| Bot | 117 | +12.68211 | +1.46373243 |
| Backtester | 118 | +14.74076 | +1.72256441 |

## Counts by side

| Path | Long | Short | Total |
|---|---:|---:|---:|
| Bot | 66 | 51 | 117 |
| Backtester | 67 | 51 | 118 |
| Matched | 66 | 51 | 117 |
| Bot unmatched | 0 | 0 | 0 |
| Backtester unmatched | 1 | 0 | 1 |

## Matched trade difference summary

Mean and median retain the sign. P95 and max are absolute difference magnitudes. `Nonzero` counts values above a small numeric tolerance.

| Metric | Unit | Count | Mean | Median | P95 abs | Max abs | Nonzero |
|---|---|---:|---:|---:|---:|---:|---:|
| Entry price diff | bps | 117 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| SL diff | bps | 117 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| TP diff | bps | 117 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | 0 |
| Position size diff | % | 117 | -0.06138498 | -0.04815768 | +0.33820800 | +0.37439312 | 116 |
| Exit time diff | bars | 117 | +2.98290598 | +2.00000000 | +9.40000000 | +32.00000000 | 117 |
| Exit price diff | bps | 117 | +3.37708883 | +0.86067273 | +83.24069916 | +295.48418919 | 117 |
| Entry fee diff | quote | 117 | -0.00000307 | -0.00000248 | +0.00001701 | +0.00001891 | 116 |
| Exit fee diff | quote | 117 | -0.00000145 | -0.00000404 | +0.00004295 | +0.00017061 | 117 |
| Entry slippage diff | quote | 117 | -0.00000184 | -0.00000149 | +0.00001021 | +0.00001134 | 116 |
| Exit slippage diff | quote | 117 | -0.00000087 | -0.00000242 | +0.00002577 | +0.00010236 | 117 |
| Slippage diff | quote | 117 | -0.00000271 | -0.00000324 | +0.00002974 | +0.00011122 | 117 |
| Net PnL diff | bps of BT notional | 117 | +1.91356389 | -4.28887011 | +81.07001579 | +304.84436342 | 117 |

## Exit reason agreement matrix

Rows are bot exit reason; columns are Backtester exit reason.

| Bot \ Backtester | SL | TIME | TP |
|---|---:|---:|---:|
| SL | 21 | 0 | 0 |
| TIME | 3 | 82 | 3 |
| TP | 0 | 0 | 8 |

## Unmatched trade explanations

- long 2026-09-06 15:10:00: signal bar 2026-09-06 15:10:00; close/high/low=O=1155.22,H=1160.91,L=1155.16,C=1160.91; signal=LONG, status=skipped:position_open, entry=1160.91, SL=1115.1683584822374, TP=1252.3932830355252

## Mismatch causes

| Cause | Count |
|---|---:|
| bot close vs backtester high/low | 35 |
| entry timing offset | 0 |
| other exit-price mismatch | 0 |
| other exit-reason mismatch | 0 |
| other pnl mismatch | 0 |
| other: no normalized backtester trade | 0 |
| other: no normalized bot trade | 1 |
| sizing | 116 |
| slippage/fee formula | 0 |
| time-exit timing | 88 |

Mismatch detail CSV: `parity_compare_ZEC_USDT_20260901_20261101_mismatches.csv`.

For each unmatched entry the report records the signal bar, its OHLC values, and the other path's captured decision or explains that the decision was not captured. Exit mismatch causes use the backtest exit bar's high/low and separate time exits, size, and cost differences.
