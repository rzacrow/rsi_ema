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
MAX_POST_BOX_CANDLES = 200  # حداکثر 200 کندل بعد از آخرین کندل باکس

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
                "post_trade_taken": False
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
                "post_trade_taken": False
            }
    
    elif state == "oversold":
        # به‌روزرسانی سقف و کف باکس
        if row["high"] > current_box["top"]:
            current_box["top"] = row["high"]
        if row["low"] < current_box["bottom"]:
            current_box["bottom"] = row["low"]
        
        current_box["end"] = timestamp
        
        if rsi_val > RSI_OVERSOLD:
            # بستن باکس و ذخیره آن
            boxes.append(current_box)
            state = "neutral"
            current_box = None
    
    elif state == "overbought":
        # به‌روزرسانی سقف و کف باکس
        if row["high"] > current_box["top"]:
            current_box["top"] = row["high"]
        if row["low"] < current_box["bottom"]:
            current_box["bottom"] = row["low"]
        
        current_box["end"] = timestamp
        
        if rsi_val < RSI_OVERBOUGHT:
            # بستن باکس و ذخیره آن
            boxes.append(current_box)
            state = "neutral"
            current_box = None

# اضافه کردن باکس فعال اگر وجود داشته باشد
if current_box is not None:
    boxes.append(current_box)

print(f"Boxes detected: {len(boxes)} (Buy: {sum(1 for b in boxes if b['type']=='buy')} | Sell: {sum(1 for b in boxes if b['type']=='sell')})")

# ===== شبیه‌سازی معاملات =====
trades = []
balance = INITIAL_BALANCE
open_positions = []  # لیست معاملات باز
cumulative_sl_loss = 0.0  # ضرر انباشته از استاپ‌لاس‌ها

# لیست برای پیگیری باکس‌های فعال
active_boxes = []
current_active_box = None

for i in range(1, len(df)):
    current_time = df.index[i]
    current_candle = df.iloc[i]
    prev_candle = df.iloc[i-1]
    
    # ===== مدیریت معاملات باز =====
    positions_to_remove = []
    for position in open_positions:
        exit_price = None
        exit_time = None
        exit_reason = None
        
        # محاسبه سود فعلی
        if position["type"] == "long":
            current_price = current_candle["close"]
            entry_price = position["entry_price"]
            stop_loss = position["stop_loss"]
            take_profit = position["take_profit"]
        else:
            current_price = current_candle["close"]
            entry_price = position["entry_price"]
            stop_loss = position["stop_loss"]
            take_profit = position["take_profit"]
        
        # ===== سیستم ریسک‌فری اصلاح شده =====
        if risk_free_enabled and not position.get("risk_free_activated"):
            # محاسبه سطح ریسک فری
            if position["type"] == "long":
                risk_free_level = entry_price * (1 - risk_free_pct)
                # فعال‌سازی فقط در صورت حرکت قیمت در جهت مخالف
                if current_candle["low"] <= risk_free_level:
                    position["stop_loss"] = risk_free_level
                    position["risk_free_activated"] = True
                    print(f"🔒 Risk-free activated for LONG at {current_time}")
            else:  # short
                risk_free_level = entry_price * (1 + risk_free_pct)
                # فعال‌سازی فقط در صورت حرکت قیمت در جهت مخالف
                if current_candle["high"] >= risk_free_level:
                    position["stop_loss"] = risk_free_level
                    position["risk_free_activated"] = True
                    print(f"🔒 Risk-free activated for SHORT at {current_time}")
        
        # ===== شرایط خروج =====
        if position["type"] == "long":
            # حد ضرر
            if current_candle["low"] <= stop_loss:
                exit_price = stop_loss
                exit_time = current_time
                exit_reason = "SL"
                # ثبت ضرر انباشته (فقط درصد)
                loss_pct = abs(entry_price - exit_price) / entry_price
                cumulative_sl_loss += loss_pct
            # حد سود
            elif current_candle["high"] >= take_profit:
                exit_price = take_profit
                exit_time = current_time
                exit_reason = "TP"
                
        elif position["type"] == "short":
            if current_candle["high"] >= stop_loss:
                exit_price = stop_loss
                exit_time = current_time
                exit_reason = "SL"
                loss_pct = abs(entry_price - exit_price) / entry_price
                cumulative_sl_loss += loss_pct
            elif current_candle["low"] <= take_profit:
                exit_price = take_profit
                exit_time = current_time
                exit_reason = "TP"
        
        # ===== ثبت معامله بسته شده =====
        if exit_price and exit_reason:
            # محاسبه سود/زیان بر اساس اندازه پوزیشن
            position_size = position["position_size"]
            
            if position["type"] == "long":
                pnl_usd = (exit_price - entry_price) * position_size
            else:
                pnl_usd = (entry_price - exit_price) * position_size
            
            # به‌روزرسانی بالانس
            prev_balance = balance
            balance += pnl_usd
            
            # محاسبه درصد سود/زیان نسبت به سرمایه اختصاص یافته
            risk_amount = position["risk_amount"]
            pnl_pct_equity = pnl_usd / risk_amount if risk_amount != 0 else 0
            
            # ذخیره معامله
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
            
            print(f"⛔ Exit {position['type']} trade | "
                  f"Reason: {exit_reason} | "
                  f"PnL: {pnl_pct_equity*100:.4f}% | "
                  f"Balance: ${balance:.2f}")
            
            positions_to_remove.append(position)
    
    # حذف معاملات بسته شده از لیست معاملات باز
    for position in positions_to_remove:
        open_positions.remove(position)
    
    # ===== به‌روزرسانی باکس‌های فعال =====
    # حذف باکس‌های منقضی شده و اضافه کردن باکس‌های جدید
    active_boxes = []
    for box in boxes:
        # محاسبه زمان پایان باکس + 200 کندل بعد
        end_time = box["end"] + timedelta(minutes=MAX_POST_BOX_CANDLES)
        
        # اگر زمان فعلی در محدوده باکس باشد
        if box["start"] <= current_time <= end_time:
            active_boxes.append(box)
    
    # ===== ورود به معاملات جدید =====
    for box in active_boxes:
        # تعیین حجم معامله بر اساس اثر مرکب
        risk_balance = balance if compounding else INITIAL_BALANCE
        
        # ===== محاسبه تیک پروفیت با جبران =====
        base_tp = TAKE_PROFIT_PCT
        take_profit_pct = base_tp
        compensation_added = 0.0
        
        # ===== سیستم جبران ضرر پلکانی =====
        if tp_type in [3, 4] and cumulative_sl_loss > 0:
            if cumulative_sl_loss <= 0.01:  # تا 1%
                compensation_added = cumulative_sl_loss
                cumulative_sl_loss = 0  # کل ضرر انباشته کسر می‌شود
                print(f"🔧 Full SL compensation: {compensation_added*100:.4f}%")
            elif cumulative_sl_loss <= 0.02:  # 1-2%
                compensation_added = cumulative_sl_loss * 0.5
                cumulative_sl_loss -= compensation_added  # نصف ضرر کسر می‌شود
                print(f"🔧 Half SL compensation: {compensation_added*100:.4f}% | Remaining: {cumulative_sl_loss*100:.4f}%")
            elif cumulative_sl_loss <= 0.03:  # 2-3%
                compensation_added = cumulative_sl_loss * (1/3)
                cumulative_sl_loss -= compensation_added  # یک سوم ضرر کسر می‌شود
                print(f"🔧 One-third SL compensation: {compensation_added*100:.4f}% | Remaining: {cumulative_sl_loss*100:.4f}%")
            else:  # بالای 3%
                compensation_added = 0
                print(f"🔧 No SL compensation (above 3%): {cumulative_sl_loss*100:.4f}%")
            
            # اعمال جبران به TP فعلی
            if compensation_added > 0:
                take_profit_pct += compensation_added
        
        # اعمال جبران اسپرد برای نوع 2 و 3
        if tp_type in [2, 3]:
            take_profit_pct += spread_comp
        
        # ===== شرایط ورود =====
        entry_signal = False
        trade_type = None
        signal_details = ""
        
        # در طول مدت باکس (از اولین تا آخرین کندل)
        if current_time <= box["end"]:
            if box["type"] == "buy":  # باکس اشباع فروش (خرید)
                if prev_candle["ema"] < box["top"] and current_candle["ema"] > box["top"]:
                    entry_signal = True
                    trade_type = "long"
                    signal_details = f"EMA exited from TOP of box ({box['top']:.6f})"
            
            elif box["type"] == "sell":  # باکس اشباع خرید (فروش)
                if prev_candle["ema"] > box["bottom"] and current_candle["ema"] < box["bottom"]:
                    entry_signal = True
                    trade_type = "short"
                    signal_details = f"EMA exited from BOTTOM of box ({box['bottom']:.6f})"
        
        # بعد از پایان باکس (تا 200 کندل بعد)
        elif not box["post_trade_taken"]:
            # شرط ورود برای اشباع فروش (خرید)
            if box["type"] == "buy":
                if current_candle["ema"] > box["top"]:
                    entry_signal = True
                    trade_type = "long"
                    box["post_trade_taken"] = True
                    signal_details = f"After Box: EMA above TOP of box ({box['top']:.6f})"
            
            # شرط ورود برای اشباع خرید (فروش)
            elif box["type"] == "sell":
                if current_candle["ema"] < box["bottom"]:
                    entry_signal = True
                    trade_type = "short"
                    box["post_trade_taken"] = True
                    signal_details = f"After Box: EMA below BOTTOM of box ({box['bottom']:.6f})"
        
        # ===== ایجاد معامله اگر سیگنال وجود داشت =====
        if entry_signal:
            # محاسبه حجم معامله بر اساس ریسک تعیین شده
            risk_amount = risk_balance * RISK_PER_TRADE
            position_size = risk_amount / current_candle["close"]

            # تعیین حد ضرر و حد سود
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
                "risk_amount": risk_amount,  # مقدار سرمایه اختصاص یافته
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "box_id": box["id"],
                "risk_free_activated": False,
                "compensation_added": compensation_added  # ذخیره میزان جبران
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
        box_trades = [t for t in trades if t.get("box_id") == box.get("id", "")]
        if box_trades:
            box_profit = sum(t['pnl_usd'] for t in box_trades)
            box_performance[box.get("id", "unknown")] = {
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