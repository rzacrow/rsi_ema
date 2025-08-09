# rsi_divergence_backtest_long_only.py
import os
import sys
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import math

# -------------------------
# ======== PARAMS =========
# -------------------------
PARAMS = {
    "rsi_len": 9,
    "lbL": 1,
    "lbR": 3,
    # default take profit RSI level for LONG (use 70 to match your Pine default)
    "takeProfitRSILevel_long": 70,
    "rangeUpper": 60,
    "rangeLower": 5,
    # Stop options: "NONE", "PERC", "ATR"
    "sl_type": "NONE",
    "stopLossPerc": 5.0,       # percent if PERC, or multiplier used with ATR logic as in prior versions
    "atr_length": 14,
    # initial balance in USD
    "initial_balance": 100.0,
    # whether to allow pyramiding (we enforce only single position -> False)
    "allow_pyramiding": False,
}


@dataclass
class Trade:
    side: str  # always "long"
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_index: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    signal: Optional[str] = None  # "regular_bull" / "hidden_bull"
    pnl: Optional[float] = None
    balance_after: Optional[float] = None
    position_notional: Optional[float] = None
    exit_reason: Optional[str] = None


# -------------------------
# Utility indicators
# -------------------------
def compute_rsi(series: pd.Series, length: int) -> pd.Series:
    """Wilder RSI (EWMA) - compatible with TradingView rsi()"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:length] = np.nan
    return rsi


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr


# -------------------------
# Pivot detection (center-based)
# -------------------------
def detect_pivots(series: pd.Series, left: int, right: int) -> Dict[str, List[int]]:
    """
    return indices of pivot centers (same definition as Pine's pivotlow/pivothigh)
    """
    n = len(series)
    piv_lows = []
    piv_highs = []
    for i in range(left, n - right):
        val = series.iat[i]
        left_slice = series.iloc[i - left:i]
        right_slice = series.iloc[i + 1:i + 1 + right]
        if left_slice.size == 0 or right_slice.size == 0:
            continue
        if (val < left_slice.min()) and (val < right_slice.min()):
            piv_lows.append(i)
        if (val > left_slice.max()) and (val > right_slice.max()):
            piv_highs.append(i)
    return {"lows": piv_lows, "highs": piv_highs}


# -------------------------
# Backtest (LONG only, marker-aligned)
# -------------------------
def backtest_long_only(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    # params
    rsi_len = params["rsi_len"]
    lbL = params["lbL"]
    lbR = params["lbR"]
    tp_long = params["takeProfitRSILevel_long"]
    rangeUpper = params["rangeUpper"]
    rangeLower = params["rangeLower"]
    sl_type = params["sl_type"]
    stopLossPerc = params["stopLossPerc"]
    atr_length = params["atr_length"]
    balance = params["initial_balance"]

    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], rsi_len)
    df["atr"] = compute_atr(df, atr_length)
    n = len(df)

    # detect pivot centers on RSI
    pivots = detect_pivots(df["rsi"].fillna(method="ffill").fillna(0), lbL, lbR)
    piv_lows = pivots["lows"]   # centers
    piv_highs = pivots["highs"]

    # convert centers -> marker indices (center - lbR)
    marker_lows = [p - lbR for p in piv_lows if (p - lbR) >= 0]
    marker_highs = [p - lbR for p in piv_highs if (p - lbR) >= 0]

    def prev_marker(marker_list: List[int], cur_marker: int) -> Optional[int]:
        prevs = [m for m in marker_list if m < cur_marker]
        return prevs[-1] if prevs else None

    trades: List[Trade] = []
    position: Optional[Trade] = None
    trailing_long = None

    # iterate over bars (use marker-aligned indices)
    for i in range(n):
        # ENTRY LONG: if current index is a marker low and there is no open position
        if i in marker_lows and position is None:
            prev_m = prev_marker(marker_lows, i)
            if prev_m is not None:
                bars_between = i - prev_m
                if rangeLower <= bars_between <= rangeUpper:
                    osc_cur = float(df["rsi"].iat[i])
                    osc_prev = float(df["rsi"].iat[prev_m])
                    price_cur_low = float(df["low"].iat[i])
                    price_prev_low = float(df["low"].iat[prev_m])
                    bullCond = (osc_cur > osc_prev) and (price_cur_low < price_prev_low)
                    hiddenBullCond = (osc_cur < osc_prev) and (price_cur_low > price_prev_low)
                    if bullCond or hiddenBullCond:
                        entry_price = float(df["close"].iat[i])  # enter at close of marker bar
                        notional = balance  # full balance per your requirement
                        position = Trade(
                            side="long",
                            entry_index=i,
                            entry_time=df.index[i],
                            entry_price=entry_price,
                            signal=("regular_bull" if bullCond else "hidden_bull"),
                            position_notional=notional
                        )
                        # init trailing
                        if sl_type == "NONE":
                            trailing_long = None
                        elif sl_type == "PERC":
                            sl_val = entry_price * stopLossPerc / 100.0
                            trailing_long = float(df["low"].iat[i]) - sl_val
                        else:  # ATR
                            sl_val = stopLossPerc * float(df["atr"].iat[i])
                            trailing_long = float(df["low"].iat[i]) - sl_val
                        continue

        # If in position, evaluate exit conditions each bar
        if position is not None:
            prev_rsi = df["rsi"].iat[i - 1] if i - 1 >= 0 else np.nan
            rsi_val = df["rsi"].iat[i]

            # TP: RSI crossover above tp_long
            rsi_cross_over = (not np.isnan(prev_rsi)) and (prev_rsi < tp_long) and (rsi_val >= tp_long)

            # Bearish divergence detection at marker_highs aligned to current i
            bear_trigger = False
            if i in marker_highs:
                prev_mh = prev_marker(marker_highs, i)
                if prev_mh is not None:
                    osc_cur_h = float(df["rsi"].iat[i])
                    osc_prev_h = float(df["rsi"].iat[prev_mh])
                    price_cur_h = float(df["high"].iat[i])
                    price_prev_h = float(df["high"].iat[prev_mh])
                    bearCond = (osc_cur_h < osc_prev_h) and (price_cur_h > price_prev_h)
                    hiddenBearCond = (osc_cur_h > osc_prev_h) and (price_cur_h < price_prev_h)
                    if bearCond or hiddenBearCond:
                        bear_trigger = True

            # trailing stop update and check
            hit_trailing = False
            if sl_type in ("PERC", "ATR") and trailing_long is not None:
                if sl_type == "PERC":
                    sl_val = df["close"].iat[i] * stopLossPerc / 100.0
                    candidate = df["low"].iat[i] - sl_val
                else:
                    sl_val = stopLossPerc * df["atr"].iat[i]
                    candidate = df["low"].iat[i] - sl_val
                trailing_long = max(trailing_long, candidate)
                if df["close"].iat[i] < trailing_long:
                    hit_trailing = True

            # decide close
            if rsi_cross_over or bear_trigger or hit_trailing:
                exit_price = float(df["close"].iat[i])
                position.exit_index = i
                position.exit_time = df.index[i]
                position.exit_price = exit_price
                # PnL for long
                pnl = (exit_price - position.entry_price) / position.entry_price * position.position_notional
                position.pnl = pnl
                balance = round(balance + pnl, 8)
                position.balance_after = balance
                # reason
                if rsi_cross_over:
                    position.exit_reason = "tp_rsi"
                elif bear_trigger:
                    position.exit_reason = "bear_div"
                else:
                    position.exit_reason = "trailing_stop"
                trades.append(position)
                position = None
                trailing_long = None
                continue

    # close any open position at end of data (EOD close)
    if position is not None:
        exit_price = float(df["close"].iat[-1])
        position.exit_index = n - 1
        position.exit_time = df.index[-1]
        position.exit_price = exit_price
        pnl = (exit_price - position.entry_price) / position.entry_price * position.position_notional
        position.pnl = pnl
        balance = round(balance + pnl, 8)
        position.balance_after = balance
        position.exit_reason = "eod_close"
        trades.append(position)
        position = None

    # build DataFrame in the exact format you requested
    recs = []
    for t in trades:
        recs.append({
            "side": "long",
            "signal": t.signal,
            "entry_time": t.entry_time,
            "entry_index": t.entry_index,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_index": t.exit_index,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl": t.pnl,
            "position_notional": t.position_notional,
            "balance_after": t.balance_after
        })
    return pd.DataFrame(recs)


# -------------------------
# CLI / Runner
# -------------------------
def main():
    if len(sys.argv) >= 2:
        SYMBOL = sys.argv[1]
    else:
        SYMBOL = input("Enter SYMBOL (filename without .csv): ").strip()

    file_path = f"data/{SYMBOL}.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        df = df.drop_duplicates(subset=["datetime"], keep='first')
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Required columns missing in {file_path}")
            sys.exit(1)

        print(f"✅ {SYMBOL} data loaded successfully | Records: {len(df)}")

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

    params = PARAMS.copy()
    # if you want a different TP (e.g. 80) set: params["takeProfitRSILevel_long"] = 80
    trades_df = backtest_long_only(df, params)

    out_file = f"signals_{SYMBOL}.csv"
    if not trades_df.empty:
        # ensure datetime columns are serialized nicely
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        trades_df.to_csv(out_file, index=False)
        print(f"Signals saved to {out_file} | Trades: {len(trades_df)}")
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        total_pnl = trades_df["pnl"].sum()
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        gross_profit = wins["pnl"].sum() if not wins.empty else 0
        gross_loss = -losses["pnl"].sum() if not losses.empty else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else math.inf
        print("=== BACKTEST SUMMARY ===")
        print(f"Trades: {len(trades_df)}, Wins: {len(wins)}, Losses: {len(losses)}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.6f} USD")
        print(f"Profit factor: {profit_factor:.3f}")
    else:
        print("No trades generated.")


if __name__ == "__main__":
    main()
