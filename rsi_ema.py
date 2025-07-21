import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

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

# ===== تنظیمات ریسک‌فری =====
risk_free_enabled = input("Enable risk-free mode? (y/n): ").lower().strip() == 'y'
risk_free_pct = 0.0

if risk_free_enabled:
    risk_free_input = input("Enter risk-free percentage (e.g. 0.3 for 0.3%): ")
    risk_free_pct = float(risk_free_input)/100 if risk_free_input else 0.003

# ===== پارامترهای ثابت استراتژی =====
RSI_LENGTH = 14
EMA_LENGTH = 12
BOX_LENGTH_HOURS = 50  # حداکثر طول باکس بر اساس ساعت
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
INITIAL_BALANCE = 100.0
MIN_BREAKOUT_INTERVAL = timedelta(minutes=5)  # حداقل فاصله بین شکست‌های متوالی
BREAKOUT_CONFIRMATION_BARS = 2  # تعداد کندل‌های تأییدیه شکست

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
box_id_counter = 0

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
        
        if rsi_val > RSI_OVERSOLD or duration_hours > BOX_LENGTH_HOURS:
            box_id_counter += 1
            boxes.append({
                "id": box_id_counter,
                "start": start_time,
                "end": timestamp,
                "top": max(highs),
                "bottom": min(lows),
                "type": "buy",
                "last_breakout": None  # زمان آخرین شکست
            })
            state = "neutral"
    
    elif state == "overbought":
        highs.append(row["high"])
        lows.append(row["low"])
        
        duration_hours = (timestamp - start_time).total_seconds() / 3600
        
        if rsi_val < RSI_OVERBOUGHT or duration_hours > BOX_LENGTH_HOURS:
            box_id_counter += 1
            boxes.append({
                "id": box_id_counter,
                "start": start_time,
                "end": timestamp,
                "top": max(highs),
                "bottom": min(lows),
                "type": "sell",
                "last_breakout": None  # زمان آخرین شکست
            })
            state = "neutral"

print(f"Boxes detected: {len(boxes)} (Buy: {sum(1 for b in boxes if b['type']=='buy')} | Sell: {sum(1 for b in boxes if b['type']=='sell')})")

# ===== شبیه‌سازی معاملات =====
trades = []
balance = INITIAL_BALANCE
position = None
cumulative_sl_loss = 0.0  # جمع ضررهای SL برای جبران
pending_breakouts = {}  # شکست‌های در انتظار تأیید

for i in range(1, len(df)):
    current_time = df.index[i]
    prev_candle = df.iloc[i-1]
    current_candle = df.iloc[i]
    
    # ===== مدیریت معاملات باز =====
    if position:
        exit_price = None
        exit_time = None
        exit_reason = None
        
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
                exit_reason = "SL"
                cumulative_sl_loss += abs(position["entry_price"] - exit_price) / position["entry_price"]
            # حد سود
            elif current_candle["high"] >= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_time = current_time
                exit_reason = "TP"
                # کاهش ضرر انباشته در صورت سود
                cumulative_sl_loss = max(0, cumulative_sl_loss - abs(exit_price - position["entry_price"]) / position["entry_price"])
            # بازگشت به باکس
            elif current_candle["close"] < position["box_top"]:
                exit_price = current_candle["close"]
                exit_time = current_time
                exit_reason = "Box"
                # اگر سود داشتیم، بخشی از ضررهای قبلی را جبران می‌کنیم
                if exit_price > position["entry_price"]:
                    profit_pct = (exit_price - position["entry_price"]) / position["entry_price"]
                    cumulative_sl_loss = max(0, cumulative_sl_loss - profit_pct)
        
        elif position["type"] == "short":
            if current_candle["high"] >= position["stop_loss"]:
                exit_price = position["stop_loss"]
                exit_time = current_time
                exit_reason = "SL"
                cumulative_sl_loss += abs(position["entry_price"] - exit_price) / position["entry_price"]
            elif current_candle["low"] <= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_time = current_time
                exit_reason = "TP"
                cumulative_sl_loss = max(0, cumulative_sl_loss - abs(exit_price - position["entry_price"]) / position["entry_price"])
            elif current_candle["close"] > position["box_bottom"]:
                exit_price = current_candle["close"]
                exit_time = current_time
                exit_reason = "Box"
                if exit_price < position["entry_price"]:
                    profit_pct = (position["entry_price"] - exit_price) / position["entry_price"]
                    cumulative_sl_loss = max(0, cumulative_sl_loss - profit_pct)
        
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
                "balance": balance,
                "exit_reason": exit_reason,
                "box_id": position["box_id"]
            })
            
            print(f"⛔ Exit {position['type']} trade | Reason: {exit_reason} | PnL: {pnl_pct*100:.2f}%")
            position = None
    
    # ===== مدیریت شکست‌های در انتظار تأیید =====
    # حذف شکست‌های قدیمی
    pending_breakouts = {box_id: breakout for box_id, breakout in pending_breakouts.items() 
                         if breakout["start_time"] + timedelta(hours=24) > current_time}
    
    # ===== ورود به معاملات جدید =====
    # تعیین حجم معامله بر اساس اثر مرکب
    risk_balance = balance if compounding else INITIAL_BALANCE
    
    # بررسی شکست‌های در انتظار تأیید
    for box_id, breakout in list(pending_breakouts.items()):
        # بررسی اینکه آیا زمان تأیید فرا رسیده است
        if breakout["confirmation_end"] <= current_time:
            # بررسی وضعیت قیمت برای تأیید نهایی
            if breakout["type"] == "buy":
                confirmed = all(df.loc[breakout["start_time"]:current_time]["close"] > breakout["level"])
            else:
                confirmed = all(df.loc[breakout["start_time"]:current_time]["close"] < breakout["level"])
            
            # اگر تأیید شد، ورود به معامله
            if confirmed:
                box = next((b for b in boxes if b["id"] == box_id), None)
                if box:
                    # محاسبه تیک پروفیت با در نظر گرفتن جبران SL
                    base_tp = TAKE_PROFIT_PCT
                    if tp_type == 1:
                        take_profit_pct = base_tp
                    elif tp_type == 2:
                        take_profit_pct = base_tp + spread_comp
                    elif tp_type == 3:
                        take_profit_pct = base_tp + spread_comp + cumulative_sl_loss
                    elif tp_type == 4:
                        take_profit_pct = base_tp + cumulative_sl_loss
                    else:
                        take_profit_pct = base_tp
                    
                    # مدیریت ریسک و حجم معامله
                    position_size = (risk_balance * RISK_PER_TRADE) / (current_candle["close"] * STOP_LOSS_PCT)
                    position_value = position_size * current_candle["close"]
                    
                    position = {
                        "type": "long" if box["type"] == "buy" else "short",
                        "entry_time": current_time,
                        "entry_price": current_candle["close"],
                        "position_size": position_size,
                        "position_value": position_value,
                        "stop_loss": current_candle["close"] * (1 - STOP_LOSS_PCT) if box["type"] == "buy" else current_candle["close"] * (1 + STOP_LOSS_PCT),
                        "take_profit": current_candle["close"] * (1 + take_profit_pct) if box["type"] == "buy" else current_candle["close"] * (1 - take_profit_pct),
                        "box_top": box["top"],
                        "box_bottom": box["bottom"],
                        "box_id": box["id"],
                        "risk_free_activated": False
                    }
                    
                    # به‌روزرسانی زمان آخرین شکست
                    box["last_breakout"] = current_time
                    print(f"⚡ {'LONG' if box['type']=='buy' else 'SHORT'} entry from box {box['id']} at {current_time}")
                    print(f"   Price: {current_candle['close']:.6f} | TP: {take_profit_pct*100:.2f}% | Cum SL: {cumulative_sl_loss*100:.2f}%")
                    
                    # حذف این شکست از لیست انتظار
                    del pending_breakouts[box_id]
                    break
    
    # جستجو برای شکست‌های جدید
    if not position:
        for box in boxes:
            # بررسی اینکه آیا زمان فعلی بعد از پایان باکس است
            if current_time <= box["end"]:
                continue
                
            # بررسی فاصله زمانی از آخرین شکست
            if box["last_breakout"] and (current_time - box["last_breakout"]) < MIN_BREAKOUT_INTERVAL:
                continue
                
            # بررسی شرایط شکست
            if box["type"] == "buy":
                # شکست مقاومت
                breakout_condition = prev_candle["close"] < box["top"] and current_candle["close"] > box["top"]
                ema_condition = prev_candle["close"] > prev_candle["ema"]
            else:
                # شکست حمایت
                breakout_condition = prev_candle["close"] > box["bottom"] and current_candle["close"] < box["bottom"]
                ema_condition = prev_candle["close"] < prev_candle["ema"]
            
            # اگر شرایط شکست برقرار بود
            if breakout_condition and ema_condition:
                # ثبت شکست جدید در لیست انتظار
                pending_breakouts[box["id"]] = {
                    "type": box["type"],
                    "level": box["top"] if box["type"] == "buy" else box["bottom"],
                    "start_time": current_time,
                    "confirmation_end": current_time + timedelta(minutes=15 * BREAKOUT_CONFIRMATION_BARS)
                }
                print(f"⚠️ Potential breakout detected from box {box['id']} at {current_time}")
                break  # فقط یک شکست در هر کندل

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
print(f"- TP Type: {tp_type} | Spread comp: {spread_comp*100:.2f}% | Cumulative SL Loss: {cumulative_sl_loss*100:.2f}%")
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
    
    # تحلیل عملکرد باکس‌ها
    box_performance = {}
    for box in boxes:
        box_trades = [t for t in trades if "box_id" in t and t["box_id"] == box["id"]]
        if box_trades:
            box_profit = sum(t['pnl_usd'] for t in box_trades)
            box_performance[box["id"]] = {
                "type": box["type"],
                "trades": len(box_trades),
                "profit": box_profit
            }
    
    print("\n" + "="*50)
    print(f"Box Performance Analysis:")
    print("="*50)
    for box_id, perf in box_performance.items():
        print(f"Box {box_id} ({perf['type']}): {perf['trades']} trades | Profit: ${perf['profit']:.2f}")
    
else:
    print("\nNo trades to analyze")

print("\n" + "="*50)
print("Backtest completed successfully!")
print("="*50)