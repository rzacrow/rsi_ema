# rsi_divergence_pine_singlepos_compound.py
import os, sys
from typing import List, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import math

# -----------------------------
# ======= PARAMETERS ==========
# -----------------------------
PARAMS = {
    # Pine defaults from your script
    "rsi_len": 9,
    "lbL": 1,
    "lbR": 3,
    "takeProfitLong": 70,    # Take Profit RSI level for LONG
    "takeProfitShort": 30,   # Take Profit RSI level for SHORT
    "rangeUpper": 60,
    "rangeLower": 5,
    # Stop settings
    "sl_type": "NONE",       # "ATR", "PERC", "NONE"
    "stopLoss": 5.0,         # percent for PERC or multiplier for ATR
    "fixedStopLoss": 100.0,  # used when sl_type == "NONE"
    "atr_length": 14,
    # backtest sizing
    "initial_balance": 100.0,
    # IMPORTANT: single position only -> pyramiding effectively 1
    "pyramiding": 1,
}

# -----------------------------
# ======= Trade dataclass =====
# -----------------------------
@dataclass
class Trade:
    side: str                  # "long" or "short"
    signal: str                # regular_bull / hidden_bull / regular_bear / hidden_bear
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    position_notional: float
    exit_index: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    balance_after: Optional[float] = None
    exit_reason: Optional[str] = None

# -----------------------------
# ======= Indicators ==========
# -----------------------------
def compute_rsi(series: pd.Series, length: int) -> pd.Series:
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
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

# -----------------------------
# ======= Pivot detection =====
# center-based (like Pine pivotlow/pivothigh)
# -----------------------------
def detect_pivots(series: pd.Series, left: int, right: int) -> Dict[str, List[int]]:
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

# -----------------------------
# ======= Backtest core =======
# -----------------------------
def backtest_singlepos_compound(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    # params
    rsi_len = params["rsi_len"]
    lbL = params["lbL"]
    lbR = params["lbR"]
    tp_long = params["takeProfitLong"]
    tp_short = params["takeProfitShort"]
    rangeUpper = params["rangeUpper"]
    rangeLower = params["rangeLower"]
    sl_type = params["sl_type"]
    stopLoss = params["stopLoss"]
    fixedStopLoss = params["fixedStopLoss"]
    atr_length = params["atr_length"]
    balance = params["initial_balance"]
    pyramiding = 1  # enforced single position

    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], rsi_len)
    df["atr"] = compute_atr(df, atr_length)
    n = len(df)

    # detect centers and convert to marker indices (center - lbR)
    piv = detect_pivots(df["rsi"].fillna(method="ffill").fillna(0), lbL, lbR)
    piv_lows = piv["lows"]
    piv_highs = piv["highs"]
    marker_lows = [p - lbR for p in piv_lows if (p - lbR) >= 0]
    marker_highs = [p - lbR for p in piv_highs if (p - lbR) >= 0]

    def prev_marker(marker_list, cur):
        prevs = [m for m in marker_list if m < cur]
        return prevs[-1] if prevs else None

    # open position holder (single pos)
    open_pos: Optional[Trade] = None
    closed_trades: List[Trade] = []

    # persistent trailing stops
    trailing_sl_long = None
    trailing_sl_short = None
    prev_trailing_sl_long = None
    prev_trailing_sl_short = None

    # iterate bars
    for i in range(n):
        prev_close = df["close"].iat[i - 1] if i - 1 >= 0 else np.nan
        prev_rsi = df["rsi"].iat[i - 1] if i - 1 >= 0 else np.nan
        curr_close = df["close"].iat[i]
        curr_rsi = df["rsi"].iat[i]

        # ---------- ENTRIES ----------
        # Only open new entry if there is NO open position at all
        no_open = open_pos is None

        # LONG entry at marker lows
        if no_open and (i in marker_lows):
            prev_m = prev_marker(marker_lows, i)
            if prev_m is not None:
                bars_between = i - prev_m
                if rangeLower <= bars_between <= rangeUpper:
                    osc_cur = float(df["rsi"].iat[i]); osc_prev = float(df["rsi"].iat[prev_m])
                    price_cur_low = float(df["low"].iat[i]); price_prev_low = float(df["low"].iat[prev_m])
                    bullCond = (osc_cur > osc_prev) and (price_cur_low < price_prev_low)
                    hiddenBullCond = (osc_cur < osc_prev) and (price_cur_low > price_prev_low)
                    if (bullCond or hiddenBullCond):
                        # Open single long; per your requirement use FULL balance
                        entry_price = float(df["close"].iat[i])
                        t = Trade(side="long", signal=("regular_bull" if bullCond else "hidden_bull"),
                                  entry_index=i, entry_time=df.index[i],
                                  entry_price=entry_price, position_notional=balance)
                        open_pos = t
                        # init trailing
                        if sl_type == "NONE":
                            trailing_sl_long = None
                        elif sl_type == "PERC":
                            sl_val = entry_price * stopLoss / 100.0
                            trailing_sl_long = float(df["low"].iat[i]) - sl_val
                        else:  # ATR
                            sl_val = stopLoss * float(df["atr"].iat[i])
                            trailing_sl_long = float(df["low"].iat[i]) - sl_val
                        # after entry, no further entries allowed until closed
                        no_open = False

        # SHORT entry at marker highs
        if no_open and (i in marker_highs):
            prev_m = prev_marker(marker_highs, i)
            if prev_m is not None:
                bars_between = i - prev_m
                if rangeLower <= bars_between <= rangeUpper:
                    osc_cur = float(df["rsi"].iat[i]); osc_prev = float(df["rsi"].iat[prev_m])
                    price_cur_high = float(df["high"].iat[i]); price_prev_high = float(df["high"].iat[prev_m])
                    bearCond = (osc_cur < osc_prev) and (price_cur_high > price_prev_high)
                    hiddenBearCond = (osc_cur > osc_prev) and (price_cur_high < price_prev_high)
                    if (bearCond or hiddenBearCond):
                        # Open single short; full balance notional
                        entry_price = float(df["close"].iat[i])
                        t = Trade(side="short", signal=("regular_bear" if bearCond else "hidden_bear"),
                                  entry_index=i, entry_time=df.index[i],
                                  entry_price=entry_price, position_notional=balance)
                        open_pos = t
                        # init trailing short
                        if sl_type == "NONE":
                            trailing_sl_short = None
                        elif sl_type == "PERC":
                            sl_val = entry_price * stopLoss / 100.0
                            trailing_sl_short = float(df["high"].iat[i]) + sl_val
                        else:  # ATR
                            sl_val = stopLoss * float(df["atr"].iat[i])
                            trailing_sl_short = float(df["high"].iat[i]) + sl_val
                        no_open = False

        # ---------- TRAILING STOP CALCULATION ----------
        # compute sl_val per Pine semantics
        atr_val = float(df["atr"].iat[i]) if not np.isnan(df["atr"].iat[i]) else 0.0
        if sl_type == "ATR":
            sl_val = stopLoss * atr_val
        elif sl_type == "PERC":
            sl_val = curr_close * stopLoss / 100.0
        else:
            sl_val = fixedStopLoss

        # update trailing if long open
        if open_pos is not None and open_pos.side == "long":
            prev_trailing_sl_long = trailing_sl_long
            if sl_type == "NONE":
                trailing_sl_long = curr_close - sl_val
            else:
                baseline = float(df["low"].iat[i])
                candidate = baseline - sl_val
                trailing_sl_long = candidate if prev_trailing_sl_long is None else max(prev_trailing_sl_long, candidate)
        else:
            prev_trailing_sl_long = trailing_sl_long
            # keep trailing_sl_long as None when not long

        # update trailing if short open
        if open_pos is not None and open_pos.side == "short":
            prev_trailing_sl_short = trailing_sl_short
            if sl_type == "NONE":
                trailing_sl_short = curr_close + sl_val
            else:
                baseline = float(df["high"].iat[i])
                candidate = baseline + sl_val
                trailing_sl_short = candidate if prev_trailing_sl_short is None else min(prev_trailing_sl_short, candidate)
        else:
            prev_trailing_sl_short = trailing_sl_short

        # ---------- CLOSE CONDITIONS ----------
        longCloseRSI = (not np.isnan(prev_rsi)) and (prev_rsi < tp_long) and (curr_rsi >= tp_long)
        shortCloseRSI = (not np.isnan(prev_rsi)) and (prev_rsi > tp_short) and (curr_rsi <= tp_short)

        # If long open, evaluate close conditions (TP RSI, bearish divergence, or trailing crossover)
        if open_pos is not None and open_pos.side == "long":
            close_long = False
            reason = None
            # sl_type NONE: TP by RSI
            if sl_type == "NONE" and longCloseRSI:
                close_long = True; reason = "tp_rsi"
            # sl_type != NONE: use crossover(trailing_sl_long, close) semantics
            elif sl_type != "NONE":
                if prev_trailing_sl_long is not None and (not np.isnan(prev_close)) and trailing_sl_long is not None:
                    if (prev_trailing_sl_long < prev_close) and (trailing_sl_long >= curr_close):
                        close_long = True; reason = "trailing_cross"
            # bearish divergence on marker_highs
            if not close_long and (i in marker_highs):
                prev_mh = prev_marker(marker_highs, i)
                if prev_mh is not None:
                    osc_cur_h = float(df["rsi"].iat[i]); osc_prev_h = float(df["rsi"].iat[prev_mh])
                    price_cur_h = float(df["high"].iat[i]); price_prev_h = float(df["high"].iat[prev_mh])
                    bearCond = (osc_cur_h < osc_prev_h) and (price_cur_h > price_prev_h)
                    hiddenBearCond = (osc_cur_h > osc_prev_h) and (price_cur_h < price_prev_h)
                    if (bearCond or hiddenBearCond):
                        close_long = True; reason = "bear_div"
            if close_long:
                # close at close price of this bar
                exit_price = curr_close
                open_pos.exit_index = i
                open_pos.exit_time = df.index[i]
                open_pos.exit_price = exit_price
                open_pos.exit_reason = reason
                open_pos.pnl = (open_pos.exit_price - open_pos.entry_price) / open_pos.entry_price * open_pos.position_notional
                balance = round(balance + open_pos.pnl, 8)
                open_pos.balance_after = balance
                closed_trades.append(open_pos)
                open_pos = None
                trailing_sl_long = None
                trailing_sl_short = None

        # If short open, evaluate close conditions (TP RSI, bullish divergence, or trailing crossover)
        if open_pos is not None and open_pos.side == "short":
            close_short = False
            reason = None
            if sl_type == "NONE" and shortCloseRSI:
                close_short = True; reason = "tp_rsi"
            elif sl_type != "NONE":
                if prev_trailing_sl_short is not None and (not np.isnan(prev_close)) and trailing_sl_short is not None:
                    if (prev_close < prev_trailing_sl_short) and (curr_close >= trailing_sl_short):
                        close_short = True; reason = "trailing_cross"
            # bullish divergence on marker_lows
            if not close_short and (i in marker_lows):
                prev_ml = prev_marker(marker_lows, i)
                if prev_ml is not None:
                    osc_cur = float(df["rsi"].iat[i]); osc_prev = float(df["rsi"].iat[prev_ml])
                    price_cur = float(df["low"].iat[i]); price_prev = float(df["low"].iat[prev_ml])
                    bullCond = (osc_cur > osc_prev) and (price_cur < price_prev)
                    hiddenBullCond = (osc_cur < osc_prev) and (price_cur > price_prev)
                    if (bullCond or hiddenBullCond):
                        close_short = True; reason = "bull_div"
            if close_short:
                exit_price = curr_close
                open_pos.exit_index = i
                open_pos.exit_time = df.index[i]
                open_pos.exit_price = exit_price
                open_pos.exit_reason = reason
                open_pos.pnl = (open_pos.entry_price - open_pos.exit_price) / open_pos.entry_price * open_pos.position_notional
                balance = round(balance + open_pos.pnl, 8)
                open_pos.balance_after = balance
                closed_trades.append(open_pos)
                open_pos = None
                trailing_sl_long = None
                trailing_sl_short = None

    # EOD close: if still open, close at last bar
    if open_pos is not None:
        i = n - 1
        exit_price = float(df["close"].iat[i])
        open_pos.exit_index = i
        open_pos.exit_time = df.index[i]
        open_pos.exit_price = exit_price
        open_pos.exit_reason = "eod_close"
        if open_pos.side == "long":
            open_pos.pnl = (open_pos.exit_price - open_pos.entry_price) / open_pos.entry_price * open_pos.position_notional
        else:
            open_pos.pnl = (open_pos.entry_price - open_pos.exit_price) / open_pos.entry_price * open_pos.position_notional
        balance = round(balance + open_pos.pnl, 8)
        open_pos.balance_after = balance
        closed_trades.append(open_pos)
        open_pos = None

    # build DataFrame
    recs = []
    for t in closed_trades:
        recs.append({
            "side": t.side,
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
    trades_df = pd.DataFrame(recs)
    return trades_df

# -----------------------------
# ======= Runner / I/O ========
# -----------------------------
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
        for col in ['open','high','low','close','volume']:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")
    except Exception as e:
        print("Error loading data:", e)
        sys.exit(1)

    params = PARAMS.copy()
    trades_df = backtest_singlepos_compound(df, params)

    out_file = f"signals_{SYMBOL}.csv"
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"])
        trades_df.to_csv(out_file, index=False)
        print(f"Signals saved to {out_file} | Trades: {len(trades_df)}")
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        total_pnl = trades_df["pnl"].sum()
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df)>0 else 0
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
