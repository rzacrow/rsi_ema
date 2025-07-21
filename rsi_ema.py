import pandas as pd
import numpy as np
import os
from datetime import datetime

# ===== دریافت تنظیمات از کاربر =====
SYMBOL = input("Enter cryptocurrency symbol (e.g. XRPUSDT): ").strip().upper()
STOP_LOSS_PCT = float(input("Enter stop loss percentage (e.g. 0.3 for 0.3%): ")) / 100
TAKE_PROFIT_PCT = float(input("Enter take profit percentage (e.g. 0.5 for 0.5%): ")) / 100
RISK_PER_TRADE = 0.01  # ریسک 1% از بالانس در هر معامله

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

if tp_type in [3, 4]:
    sl_input = input("Enter SL compensation percentage: ")
    sl_comp = float(sl_input)/100 if sl_input else 0.0

# ===== تنظیمات ریسک‌فری =====
risk_free_enabled = input("Enable risk-free mode? (y/n): ").lower().strip() == 'y'
risk_free_pct = 0.0

if risk_free_enabled:
    risk_free_input = input("Enter risk-free percentage (e.g. 0.3 for 0.3%): ")
    risk_free_pct = float(risk_free_input)/100 if risk_free_input else 0.003

# ===== پارامترهای ثابت استراتژی =====
RSI_LENGTH = 14
EMA_LENGTH = 12
BOX_LENGTH_BARS = 50
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
INITIAL_BALANCE = 100.0

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

# ===== محاسبات اندیکاتورها =====
df["rsi"] = compute_rsi(df["close"], RSI_LENGTH)
df["ema"] = df["close"].ewm(span=EMA_LENGTH, adjust=False).mean()

# ===== تشخیص محدوده باکس‌های RSI =====
boxes = []
state = "neutral"
start_time = None
highs, lows = [], []

for timestamp, row in df.iterrows():
    rsi_val = row["rsi"]
    
    if state == "neutral":
        if rsi_val < RSI_OVERSOLD:
            state = "oversold"
            start_time = timestamp
            highs = [row["high"]]
            lows = [row["low"]]
        elif rsi_val > RSI_OVERBOUGHT:
            state = "overbought"
            start_time = timestamp
            highs = [row["high"]]
            lows = [row["low"]]
    
    elif state == "oversold":
        highs.append(row["high"])
        lows.append(row["low"])
        
        # استفاده از ساعت برای محاسبه مدت زمان باکس
        duration_hours = (timestamp - start_time).total_seconds() / 3600
        
        if rsi_val > RSI_OVERSOLD or duration_hours > BOX_LENGTH_BARS:
            boxes.append({
                "start": start_time,
                "end": timestamp,
                "top": max(highs),
                "bottom": min(lows),
                "type": "buy"
            })
            state = "neutral"
    
    elif state == "overbought":
        highs.append(row["high"])
        lows.append(row["low"])
        
        duration_hours = (timestamp - start_time).total_seconds() / 3600
        
        if rsi_val < RSI_OVERBOUGHT or duration_hours > BOX_LENGTH_BARS:
            boxes.append({
                "start": start_time,
                "end": timestamp,
                "top": max(highs),
                "bottom": min(lows),
                "type": "sell"
            })
            state = "neutral"

print(f"Boxes detected: {len(boxes)} (Buy: {sum(1 for b in boxes if b['type']=='buy')} | Sell: {sum(1 for b in boxes if b['type']=='sell')})")

# ===== شبیه‌سازی معاملات =====
trades = []
balance = INITIAL_BALANCE
position = None
next_box_idx = 0

for i in range(1, len(df)):
    current_time = df.index[i]
    prev_candle = df.iloc[i-1]
    current_candle = df.iloc[i]
    
    # ===== مدیریت معاملات باز =====
    if position:
        exit_price = None
        exit_time = None
        
        # محاسبه سود فعلی
        if position["type"] == "long":
            current_profit_pct = (current_candle["close"] - position["entry_price"]) / position["entry_price"]
        else:
            current_profit_pct = (position["entry_price"] - current_candle["close"]) / position["entry_price"]
        
        # ===== سیستم ریسک‌فری =====
        if risk_free_enabled and not position.get("risk_free_activated"):
            if current_profit_pct >= risk_free_pct:
                # فعال کردن ریسک‌فری و انتقال استاپ به نقطه ورود
                if position["type"] == "long":
                    position["stop_loss"] = position["entry_price"]
                else:
                    position["stop_loss"] = position["entry_price"]
                
                position["risk_free_activated"] = True
                print(f"🔒 Risk-free activated for trade at {current_time}")
        
        # ===== شرایط خروج =====
        if position["type"] == "long":
            # حد ضرر
            if current_candle["low"] <= position["stop_loss"]:
                exit_price = position["stop_loss"]
                exit_time = current_time
            # حد سود
            elif current_candle["high"] >= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_time = current_time
            # بازگشت به باکس
            elif current_candle["close"] < position["box_top"]:
                exit_price = current_candle["close"]
                exit_time = current_time
                
        elif position["type"] == "short":
            if current_candle["high"] >= position["stop_loss"]:
                exit_price = position["stop_loss"]
                exit_time = current_time
            elif current_candle["low"] <= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_time = current_time
            elif current_candle["close"] > position["box_bottom"]:
                exit_price = current_candle["close"]
                exit_time = current_time
        
        # ===== ثبت معامله بسته شده =====
        if exit_price and exit_time:
            # محاسبه سود/زیان
            if position["type"] == "long":
                pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
            else:
                pnl_pct = (position["entry_price"] - exit_price) / position["entry_price"]
            
            pnl_usd = pnl_pct * position["position_value"]
            balance += pnl_usd
            
            # ذخیره معامله با فرمت دقیق
            trades.append({
                "type": position["type"],
                "entry_time": position["entry_time"],
                "entry_price": position["entry_price"],
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "balance": balance
            })
            
            position = None
    
    # ===== ورود به معاملات جدید =====
    if not position and next_box_idx < len(boxes):
        current_box = boxes[next_box_idx]
        
        # بررسی اینکه آیا زمان فعلی بعد از پایان باکس است
        if current_time > current_box["end"]:
            # تعیین حجم معامله بر اساس اثر مرکب
            risk_balance = balance if compounding else INITIAL_BALANCE
            
            # سیگنال خرید
            if (current_box["type"] == "buy" and 
                prev_candle["close"] > prev_candle["ema"] and
                current_candle["close"] > current_box["top"]):
                
                # محاسبه تیک پروفیت بر اساس نوع انتخاب شده
                base_tp = TAKE_PROFIT_PCT
                if tp_type == 1:
                    take_profit_pct = base_tp
                elif tp_type == 2:
                    take_profit_pct = base_tp + spread_comp
                elif tp_type == 3:
                    take_profit_pct = base_tp + spread_comp + sl_comp
                elif tp_type == 4:
                    take_profit_pct = base_tp + sl_comp
                else:
                    take_profit_pct = base_tp  # حالت پیش فرض
                
                # مدیریت ریسک و حجم معامله
                position_size = (risk_balance * RISK_PER_TRADE) / (current_candle["close"] * STOP_LOSS_PCT)
                position_value = position_size * current_candle["close"]
                
                position = {
                    "type": "long",
                    "entry_time": current_time,
                    "entry_price": current_candle["close"],
                    "position_size": position_size,
                    "position_value": position_value,
                    "stop_loss": current_candle["close"] * (1 - STOP_LOSS_PCT),
                    "take_profit": current_candle["close"] * (1 + take_profit_pct),
                    "box_top": current_box["top"],
                    "box_bottom": current_box["bottom"],
                    "risk_free_activated": False
                }
                next_box_idx += 1
                print(f"⚡ LONG entry at {current_time} | Price: {current_candle['close']:.6f}")
            
            # سیگنال فروش
            elif (current_box["type"] == "sell" and 
                  prev_candle["close"] < prev_candle["ema"] and
                  current_candle["close"] < current_box["bottom"]):
                
                # محاسبه تیک پروفیت بر اساس نوع انتخاب شده
                base_tp = TAKE_PROFIT_PCT
                if tp_type == 1:
                    take_profit_pct = base_tp
                elif tp_type == 2:
                    take_profit_pct = base_tp + spread_comp
                elif tp_type == 3:
                    take_profit_pct = base_tp + spread_comp + sl_comp
                elif tp_type == 4:
                    take_profit_pct = base_tp + sl_comp
                else:
                    take_profit_pct = base_tp  # حالت پیش فرض
                
                position_size = (risk_balance * RISK_PER_TRADE) / (current_candle["close"] * STOP_LOSS_PCT)
                position_value = position_size * current_candle["close"]
                
                position = {
                    "type": "short",
                    "entry_time": current_time,
                    "entry_price": current_candle["close"],
                    "position_size": position_size,
                    "position_value": position_value,
                    "stop_loss": current_candle["close"] * (1 + STOP_LOSS_PCT),
                    "take_profit": current_candle["close"] * (1 - take_profit_pct),
                    "box_top": current_box["top"],
                    "box_bottom": current_box["bottom"],
                    "risk_free_activated": False
                }
                next_box_idx += 1
                print(f"⚡ SHORT entry at {current_time} | Price: {current_candle['close']:.6f}")

# ===== ذخیره نتایج =====
if trades:
    # ایجاد دیتافریم با فرمت دقیق
    output_columns = ["type", "entry_time", "entry_price", "exit_time", "exit_price", 
                     "pnl_pct", "pnl_usd", "balance"]
    
    trades_df = pd.DataFrame(trades)[output_columns]
    
    # تبدیل تاریخ‌ها به رشته با فرمت دقیق
    trades_df["entry_time"] = trades_df["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    trades_df["exit_time"] = trades_df["exit_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # ذخیره با فرمت درخواستی
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
print(f"- Stop Loss: {STOP_LOSS_PCT*100:.2f}%")
print(f"- Base Take Profit: {TAKE_PROFIT_PCT*100:.2f}%")
print(f"- TP Type: {tp_type} with spread: {spread_comp*100:.2f}% | SL comp: {sl_comp*100:.2f}%")
print(f"- Compounding: {'Enabled' if compounding else 'Disabled'}")
print(f"- Risk-Free: {'Enabled' if risk_free_enabled else 'Disabled'} ({risk_free_pct*100:.2f}%)")
print(f"- Initial Balance: ${INITIAL_BALANCE:.2f}")

if trades:
    win_trades = [t for t in trades if t['pnl_usd'] > 0]
    loss_trades = [t for t in trades if t['pnl_usd'] <= 0]
    win_rate = len(win_trades)/len(trades)*100
    
    print("\n" + "="*50)
    print(f"Performance Metrics:")
    print("="*50)
    print(f"Total Trades: {len(trades)}")
    print(f"Winning Trades: {len(win_trades)} ({win_rate:.2f}%)")
    print(f"Losing Trades: {len(loss_trades)}")
    
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
else:
    print("\nNo trades to analyze")

print("\n" + "="*50)
print("Backtest completed successfully!")
print("="*50)