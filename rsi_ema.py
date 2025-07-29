import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ===== دریافت تنظیمات از کاربر =====
SYMBOL = input("Enter cryptocurrency symbol (e.g. XRPUSDT): ").strip().upper()
STOP_LOSS_PCT = float(input("Enter stop loss percentage (e.g. 0.3 for 0.3%): ")) / 100
TAKE_PROFIT_PCT = float(input("Enter take profit percentage (e.g. 0.5 for 0.5%): ")) / 100
RISK_PER_TRADE = float(input("Enter risk per trade percentage (e.g. 1 for 1%): ")) / 100

# ===== تنظیمات اثر مرکب =====
compounding = input("Enable compounding? (y/n): ").lower().strip() == 'y'
print(f"Compounding {'ENABLED' if compounding else 'DISABLED'}")

# ===== تنظیمات تیک پروفیت =====
print("\nSelect TP type:")
print("1. Fixed TP")
print("2. Fixed TP + Spread Compensation")
print("3. Fixed TP + Spread Compensation + SL Compensation")
print("4. Fixed TP + SL Compensation")
tp_type = int(input("Enter TP type (1-4): "))

# تنظیمات پیش‌فرض برای اسپرد
SPREAD_DEFAULT = 0.0008  # 0.08% اسپرد پیش‌فرض

spread_comp = SPREAD_DEFAULT
sl_comp = 0.0

if tp_type in [2, 3]:
    spread_input = input(f"Enter spread compensation percentage (default={SPREAD_DEFAULT*100:.2f}%): ")
    spread_comp = float(spread_input)/100 if spread_input else SPREAD_DEFAULT

# ===== تنظیمات ریسک‌فری =====
risk_free_enabled = input("Enable risk-free mode? (y/n): ").lower().strip() == 'y'
risk_free_pct = 0.0

if risk_free_enabled:
    risk_free_input = input("Enter risk-free percentage (e.g. 0.3 for 0.3%): ")
    risk_free_pct = float(risk_free_input)/100 if risk_free_input else 0.003

# ===== پارامترهای ثابت استراتژی =====
RSI_LENGTH = 14
EMA_LENGTH = 12
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
INITIAL_BALANCE = 100.0
MAX_POST_BOX_CANDLES = -1  # حداکثر 200 کندل بعد از آخرین کندل باکس

# ===== توابع کمکی =====
def compute_rsi(series: pd.Series, length: int) -> pd.Series:
    """محاسبه RSI با مدیریت خطاها و تقسیم بر صفر"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===== خواندن داده‌ها =====
file_path = f"data/{SYMBOL}.csv"
if not os.path.exists(file_path):
    print(f"File {file_path} not found!")
    exit()

try:
    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df = df.drop_duplicates(subset=["datetime"], keep='first')
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    
    # بررسی وجود داده‌های ضروری
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Required columns missing in {file_path}")
        exit()
        
    print(f"✅ {SYMBOL} data loaded successfully | Records: {len(df)}")

except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# ===== محاسبه RSI 3 دقیقه‌ای =====
# ایجاد داده‌های 3 دقیقه‌ای
df_3m = df.resample('3T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# محاسبه RSI برای داده‌های 3 دقیقه‌ای
df_3m["rsi_3m"] = compute_rsi(df_3m["close"], RSI_LENGTH)

# پر کردن مقادیر خالی
df_3m["rsi_3m"].ffill(inplace=True)

# مپ کردن RSI 3 دقیقه‌ای به داده‌های 1 دقیقه‌ای
df['rsi_3m'] = df_3m['rsi_3m'].reindex(df.index, method='ffill')

# ===== محاسبات اندیکاتورها =====
df["ema"] = df["close"].ewm(span=EMA_LENGTH, adjust=False).mean()
df["prev_ema"] = df["ema"].shift(1)

# ===== تشخیص محدوده باکس‌ها بر اساس RSI 3 دقیقه‌ای =====
boxes = []
state = "neutral"
start_time = None
highs, lows = [], []
current_box = None

for timestamp, row in df.iterrows():
    rsi_val = row["rsi_3m"]
    if state == "neutral":
        if rsi_val < RSI_OVERSOLD:
            state = "oversold"
            start_time = timestamp
            highs = [row["high"]]
            lows = [row["low"]]
            current_box = {
                "start": start_time,
                "end": None,
                "top": row["high"],
                "bottom": row["low"],
                "type": "buy",
                "id": len(boxes) + 1,
                "ema_entered": False,
                "ema_has_entered": False,
                "tp_hit": False
            }
        elif rsi_val > RSI_OVERBOUGHT:
            state = "overbought"
            start_time = timestamp
            highs = [row["high"]]
            lows = [row["low"]]
            current_box = {
                "start": start_time,
                "end": None,
                "top": row["high"],
                "bottom": row["low"],
                "type": "sell",
                "id": len(boxes) + 1,
                "ema_entered": False,
                "ema_has_entered": False,
                "tp_hit": False
            }
    elif state == "oversold":
        if row["high"] > current_box["top"]:
            current_box["top"] = row["high"]
        if row["low"] < current_box["bottom"]:
            current_box["bottom"] = row["low"]
        current_box["end"] = timestamp
        if current_box["bottom"] <= row["ema"] <= current_box["top"]:
            current_box["ema_entered"] = True
            current_box["ema_has_entered"] = True
        if rsi_val > RSI_OVERSOLD:
            boxes.append(current_box)
            state = "neutral"
            current_box = None
    elif state == "overbought":
        if row["high"] > current_box["top"]:
            current_box["top"] = row["high"]
        if row["low"] < current_box["bottom"]:
            current_box["bottom"] = row["low"]
        current_box["end"] = timestamp
        if current_box["bottom"] <= row["ema"] <= current_box["top"]:
            current_box["ema_entered"] = True
            current_box["ema_has_entered"] = True
        if rsi_val < RSI_OVERBOUGHT:
            boxes.append(current_box)
            state = "neutral"
            current_box = None
if current_box is not None:
    boxes.append(current_box)

print(f"Boxes detected: {len(boxes)} (Buy: {sum(1 for b in boxes if b['type']=='buy')} | Sell: {sum(1 for b in boxes if b['type']=='sell')}")

# ===== شبیه‌سازی معاملات =====
trades = []
balance = INITIAL_BALANCE
open_positions = []
cumulative_sl_loss = 0.0

active_box = None
active_box_index = -1
last_trade_result = None

for i in range(1, len(df)):
    current_time = df.index[i]
    current_candle = df.iloc[i]
    prev_candle = df.iloc[i-1]

    # ===== Box activation: only switch to a new box when a new one is formed =====
    if active_box is None or (active_box_index + 1 < len(boxes) and current_time >= boxes[active_box_index + 1]["start"]):
        # Move to the next box if time has reached its start
        for idx, box in enumerate(boxes):
            if current_time >= box["start"]:
                active_box = box
                active_box_index = idx
                last_trade_result = None
        # After loop, active_box is the latest box whose start <= current_time

    # ===== EMA entry gate: set ema_has_entered only once per box =====
    if active_box and not active_box["ema_has_entered"]:
        if active_box["bottom"] <= current_candle["ema"] <= active_box["top"]:
            active_box["ema_has_entered"] = True

    # ===== مدیریت معاملات باز =====
    positions_to_remove = []
    for position in open_positions:
        exit_price = None
        exit_time = None
        exit_reason = None
        if position["type"] == "long":
            current_price = current_candle["close"]
            entry_price = position["entry_price"]
            stop_loss = position["stop_loss"]
            take_profit = position["take_profit"]
            if not position.get("max_price"):
                position["max_price"] = entry_price
            position["max_price"] = max(position["max_price"], current_candle["high"])
        else:
            current_price = current_candle["close"]
            entry_price = position["entry_price"]
            stop_loss = position["stop_loss"]
            take_profit = position["take_profit"]
            if not position.get("min_price"):
                position["min_price"] = entry_price
            position["min_price"] = min(position["min_price"], current_candle["low"])
        # Risk-free logic
        if risk_free_enabled and not position.get("risk_free_active"):
            if position["type"] == "long":
                risk_free_level = entry_price * (1 + risk_free_pct)
                if position["max_price"] >= risk_free_level:
                    position["risk_free_active"] = True
                    position["risk_free_level"] = risk_free_level
                    print(f"🔒 Risk-free mode ACTIVATED for LONG at {current_time} | Level: {risk_free_level:.6f}")
            else:
                risk_free_level = entry_price * (1 - risk_free_pct)
                if position["min_price"] <= risk_free_level:
                    position["risk_free_active"] = True
                    position["risk_free_level"] = risk_free_level
                    print(f"🔒 Risk-free mode ACTIVATED for SHORT at {current_time} | Level: {risk_free_level:.6f}")
        risk_free_exit = False
        if risk_free_enabled and position.get("risk_free_active"):
            if position["type"] == "long":
                if current_candle["close"] < position["risk_free_level"] and current_candle["close"] < current_candle["open"]:
                    exit_price = current_candle["close"]
                    exit_time = current_time
                    exit_reason = "Risk-Free"
                    risk_free_exit = True
            else:
                if current_candle["close"] > position["risk_free_level"] and current_candle["close"] > current_candle["open"]:
                    exit_price = current_candle["close"]
                    exit_time = current_time
                    exit_reason = "Risk-Free"
                    risk_free_exit = True
        # EMA-based SL
        box = next((b for b in boxes if b["id"] == position["box_id"]), None)
        if box:
            if position["type"] == "long":
                if current_candle["ema"] <= box["top"]:
                    exit_price = current_candle["close"]
                    exit_time = current_time
                    exit_reason = "EMA_SL"
            else:
                if current_candle["ema"] >= box["bottom"]:
                    exit_price = current_candle["close"]
                    exit_time = current_time
                    exit_reason = "EMA_SL"
        # Normal exit
        if not risk_free_exit and exit_reason is None:
            if position["type"] == "long":
                if current_candle["low"] <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_time = current_time
                    exit_reason = "SL"
                    loss_pct = max(0, abs(entry_price - exit_price) / entry_price)
                    if exit_price < entry_price:
                        cumulative_sl_loss += loss_pct
                        print(f"📉 SL hit! Added {loss_pct*100:.4f}% to cumulative loss | Total: {cumulative_sl_loss*100:.4f}%")
                elif current_candle["high"] >= take_profit:
                    exit_price = take_profit
                    exit_time = current_time
                    exit_reason = "TP"
            elif position["type"] == "short":
                if current_candle["high"] >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_time = current_time
                    exit_reason = "SL"
                    loss_pct = max(0, abs(entry_price - exit_price) / entry_price)
                    if exit_price > entry_price:
                        cumulative_sl_loss += loss_pct
                        print(f"📉 SL hit! Added {loss_pct*100:.4f}% to cumulative loss | Total: {cumulative_sl_loss*100:.4f}%")
                elif current_candle["low"] <= take_profit:
                    exit_price = take_profit
                    exit_time = current_time
                    exit_reason = "TP"
        # Register closed trade
        if exit_price and exit_reason:
            position_size = position["position_size"]
            if position["type"] == "long":
                pnl_usd = (exit_price - entry_price) * position_size
            else:
                pnl_usd = (entry_price - exit_price) * position_size
            prev_balance = balance
            balance += pnl_usd
            risk_amount = position["risk_amount"]
            pnl_pct_equity = pnl_usd / risk_amount if risk_amount != 0 else 0
            # Add to cumulative_sl_loss for both SL and EMA_SL
            if exit_reason in ["SL", "EMA_SL"]:
                loss_pct = max(0, abs(entry_price - exit_price) / entry_price)
                cumulative_sl_loss += loss_pct
                print(f"📉 {exit_reason} hit! Added {loss_pct*100:.4f}% to cumulative loss | Total: {cumulative_sl_loss*100:.4f}%")
            trades.append({
                "type": position["type"],
                "entry_time": position["entry_time"],
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct_equity,
                "pnl_usd": pnl_usd,
                "balance": balance,
                "exit_reason": exit_reason,
                "box_id": position["box_id"],
                "risk_free_activated": position.get("risk_free_active", False),
                "compensation_added": position.get("compensation_added", 0.0)
            })
            print(f"⛔ Exit {position['type']} trade | "
                  f"Reason: {exit_reason} | "
                  f"PnL: {pnl_pct_equity*100:.4f}% | "
                  f"Balance: ${balance:.2f}")
            positions_to_remove.append(position)
            # Mark TP hit for the box if TP was hit
            if exit_reason == "TP":
                for b in boxes:
                    if b["id"] == position["box_id"]:
                        b["tp_hit"] = True
            if exit_reason == "TP":
                last_trade_result = "TP"
            elif exit_reason == "SL":
                last_trade_result = "SL"
            elif exit_reason == "Risk-Free":
                last_trade_result = "RF"
            elif exit_reason == "EMA_SL":
                last_trade_result = "EMA_SL"
    for position in positions_to_remove:
        open_positions.remove(position)

    # After trade exits, handle SL compensation reset/reduction only for TP exits with compensation
    if trades and trades[-1]["exit_reason"] == "TP" and tp_type in [3, 4] and trades[-1].get("compensation_added", 0) > 0:
        comp = trades[-1]["compensation_added"]
        if comp == cumulative_sl_loss:
            print(f"🔧 Full SL compensation applied: {comp*100:.4f}%")
            cumulative_sl_loss = 0
        elif comp == cumulative_sl_loss * 0.5:
            print(f"🔧 Half SL compensation applied: {comp*100:.4f}% | Remaining: {cumulative_sl_loss*100/2:.4f}%")
            cumulative_sl_loss = cumulative_sl_loss * 0.5
        elif comp == cumulative_sl_loss * (1/3):
            print(f"🔧 One-third SL compensation applied: {comp*100:.4f}% | Remaining: {cumulative_sl_loss*100*2/3:.4f}%")
            cumulative_sl_loss = cumulative_sl_loss * (2/3)
        else:
            print(f"🔧 No SL compensation (above 3%): {cumulative_sl_loss*100:.4f}%")

    # ===== ورود به معاملات جدید فقط برای باکس فعال =====
    if active_box and active_box["ema_has_entered"] and not active_box["tp_hit"]:
        box = active_box
        risk_balance = balance if compounding else INITIAL_BALANCE
        base_tp = TAKE_PROFIT_PCT
        take_profit_pct = base_tp
        compensation_added = 0.0
        # Always apply compensation logic for every new trade (including re-entries)
        if tp_type in [3, 4] and cumulative_sl_loss > 0:
            if cumulative_sl_loss <= 0.01:
                compensation_added = cumulative_sl_loss
            elif cumulative_sl_loss <= 0.02:
                compensation_added = cumulative_sl_loss * 0.5
            elif cumulative_sl_loss <= 0.03:
                compensation_added = cumulative_sl_loss * (1/3)
            else:
                compensation_added = 0
            take_profit_pct += compensation_added
        if tp_type in [2, 3]:
            take_profit_pct += spread_comp
        entry_signal = False
        trade_type = None
        signal_details = ""
        box_open_position = any(pos["box_id"] == box["id"] for pos in open_positions)
        if not box_open_position and box["type"] == "sell":
            if current_candle["ema"] < box["bottom"] and current_candle["close"] < current_candle["ema"]:
                entry_signal = True
                trade_type = "short"
                signal_details = f"EMA below BOTTOM and close below EMA ({box['bottom']:.6f})"
        if not box_open_position and box["type"] == "buy":
            if current_candle["ema"] > box["top"] and current_candle["close"] > current_candle["ema"]:
                entry_signal = True
                trade_type = "long"
                signal_details = f"EMA above TOP and close above EMA ({box['top']:.6f})"
        if box_open_position == False and last_trade_result in ["SL", "RF", "EMA_SL"]:
            if box["type"] == "sell":
                if current_candle["ema"] < box["bottom"] and current_candle["close"] < current_candle["ema"]:
                    entry_signal = True
                    trade_type = "short"
                    signal_details = f"(Re-entry) EMA below BOTTOM and close below EMA ({box['bottom']:.6f})"
            elif box["type"] == "buy":
                if current_candle["ema"] > box["top"] and current_candle["close"] > current_candle["ema"]:
                    entry_signal = True
                    trade_type = "long"
                    signal_details = f"(Re-entry) EMA above TOP and close above EMA ({box['top']:.6f})"
        if entry_signal:
            risk_amount = risk_balance * RISK_PER_TRADE
            position_size = risk_amount / current_candle["close"]
            if trade_type == "long":
                stop_loss = current_candle["close"] * (1 - STOP_LOSS_PCT)
                take_profit = current_candle["close"] * (1 + take_profit_pct)
            else:
                stop_loss = current_candle["close"] * (1 + STOP_LOSS_PCT)
                take_profit = current_candle["close"] * (1 - take_profit_pct)
            new_position = {
                "type": trade_type,
                "entry_time": current_time,
                "entry_price": current_candle["close"],
                "position_size": position_size,
                "risk_amount": risk_amount,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "box_id": box["id"],
                "risk_free_activated": False,
                "compensation_added": compensation_added
            }
            open_positions.append(new_position)
            print(f"⚡ {trade_type.upper()} entry at {current_time} | "
                  f"Price: {current_candle['close']:.6f} | "
                  f"Risk: ${risk_amount:.4f}")
            print(f"   Box: {box['type']} ({box['id']}) | "
                  f"TP: {take_profit_pct*100:.4f}% | "
                  f"SL comp: {compensation_added*100:.4f}%")
            print(f"   Signal: {signal_details}")

# ===== بستن معاملات باز در پایان بکتست =====
for position in open_positions:
    exit_price = df.iloc[-1]["close"]
    exit_time = df.index[-1]
    exit_reason = "End of Backtest"

    position_size = position["position_size"]
    entry_price = position["entry_price"]
    
    if position["type"] == "long":
        pnl_usd = (exit_price - entry_price) * position_size
    else:
        pnl_usd = (entry_price - exit_price) * position_size
    
    # به‌روزرسانی بالانس
    balance += pnl_usd
    
    # محاسبه درصد سود/زیان
    risk_amount = position["risk_amount"]
    pnl_pct_equity = pnl_usd / risk_amount if risk_amount != 0 else 0

    trades.append({
        "type": position["type"],
        "entry_time": position["entry_time"],
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct_equity,
        "pnl_usd": pnl_usd,
        "balance": balance,
        "exit_reason": exit_reason,
        "box_id": position["box_id"]
    })
    print(f"⛔ Closing open {position['type']} trade | "
          f"PnL: {pnl_pct_equity*100:.4f}% | "
          f"Balance: ${balance:.2f}")

# ===== ذخیره نتایج =====
if trades:
    output_columns = ["type", "entry_time", "entry_price", "exit_time", "exit_price", "pnl_pct", "pnl_usd", "balance", "exit_reason", "box_id"]
    trades_df = pd.DataFrame(trades)[output_columns]
    
    # تبدیل تاریخ‌ها به رشته
    trades_df["entry_time"] = trades_df["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    trades_df["exit_time"] = trades_df["exit_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    output_file = f"trades_{SYMBOL}.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\nTrades saved to {output_file}")
else:
    print("\nNo trades executed")

# ===== گزارش نهایی =====
print("\n" + "="*50)
print(f"Backtest Results for {SYMBOL}")
print("="*50)
print(f"Strategy Settings:")
print(f"- Stop Loss: {STOP_LOSS_PCT*100:.4f}%")
print(f"- Base Take Profit: {TAKE_PROFIT_PCT*100:.4f}%")
print(f"- TP Type: {tp_type} | Spread comp: {spread_comp*100:.4f}% | Cumulative SL Loss: {cumulative_sl_loss*100:.4f}%")
print(f"- Compounding: {'Enabled' if compounding else 'Disabled'}")
print(f"- Risk-Free: {'Enabled' if risk_free_enabled else 'Disabled'} ({risk_free_pct*100:.4f}%)")
print(f"- Initial Balance: ${INITIAL_BALANCE:.2f}")

if trades:
    win_trades = [t for t in trades if t['pnl_usd'] > 0]
    loss_trades = [t for t in trades if t['pnl_usd'] <= 0]
    win_rate = len(win_trades)/len(trades)*100
    
    # محاسبه کل جبران SL اعمال شده
    total_compensation = sum(t.get('compensation_added', 0) for t in trades if 'compensation_added' in t) * 100
    
    print("\n" + "="*50)
    print(f"Performance Metrics:")
    print("="*50)
    print(f"Total Trades: {len(trades)}")
    print(f"Winning Trades: {len(win_trades)} ({win_rate:.2f}%)")
    print(f"Losing Trades: {len(loss_trades)}")
    print(f"Total SL Compensation Applied: {total_compensation:.4f}%")
    
    if win_trades:
        avg_win = sum(t['pnl_pct'] for t in win_trades)/len(win_trades)*100
        max_win = max(t['pnl_pct'] for t in trades)*100
    else:
        avg_win = max_win = 0.0
    
    if loss_trades:
        avg_loss = sum(t['pnl_pct'] for t in loss_trades)/len(loss_trades)*100
        max_loss = min(t['pnl_pct'] for t in trades)*100
    else:
        avg_loss = max_loss = 0.0
    
    profit_factor = abs(sum(t['pnl_usd'] for t in win_trades) / sum(t['pnl_usd'] for t in loss_trades)) if loss_trades else float('inf')
    
    print(f"\nAverage Win: {avg_win:.4f}%")
    print(f"Average Loss: {avg_loss:.4f}%")
    print(f"Max Win: {max_win:.4f}%")
    print(f"Max Loss: {max_loss:.4f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    print("\n" + "="*50)
    print(f"Balance Analysis:")
    print("="*50)
    print(f"Final Balance: ${balance:.2f}")
    print(f"Net Profit: ${balance - INITIAL_BALANCE:.2f}")
    print(f"Return: {((balance/INITIAL_BALANCE)-1)*100:.2f}%")
    
    # محاسبه حداکثر افت سرمایه
    equity_curve = [INITIAL_BALANCE]
    for t in trades:
        equity_curve.append(t['balance'])
    
    peak = INITIAL_BALANCE
    max_drawdown = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd
    
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # تحلیل عملکرد باکس‌ها
    box_performance = {}
    for box in boxes:
        box_trades = [t for t in trades if t.get("box_id") == box.get("id", "")]
        if box_trades:
            box_profit = sum(t['pnl_usd'] for t in box_trades)
            box_performance[box.get("id", "unknown")] = {
                "type": box["type"],
                "trades": len(box_trades),
                "profit": box_profit,
                "ema_entered": box.get("ema_entered", False)
            }
    
    print("\n" + "="*50)
    print(f"Box Performance Analysis:")
    print("="*50)
    for box_id, perf in box_performance.items():
        print(f"Box {box_id} ({perf['type']}): {perf['trades']} trades | Profit: ${perf['profit']:.2f} | EMA Entered: {'Yes' if perf['ema_entered'] else 'No'}")
    
else:
    print("\nNo trades to analyze")

print("\n" + "="*50)
print("Backtest completed successfully!")
print("="*50)