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
RISK_FREE_THRESHOLD = 0.005  # 0.5% default threshold for risk-free activation

if risk_free_enabled:
    print(f"Risk-free mode ENABLED with {RISK_FREE_THRESHOLD*100:.1f}% threshold")
else:
    print("Risk-free mode DISABLED")

# ===== پارامترهای ثابت استراتژی =====
EMA_LENGTH = 12
INITIAL_BALANCE = 100.0

# ===== پارامترهای Ichimoku =====
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
DISPLACEMENT = 26

# ===== پارامترهای Cloud Switch Detection =====
CLOUD_SWITCH_LEAD = 4  # 4 candles before 26
CLOUD_SWITCH_LAG = 4   # 4 candles after 26
CLOUD_SWITCH_CHECK_START = DISPLACEMENT - CLOUD_SWITCH_LEAD  # 22 candles
CLOUD_SWITCH_CHECK_END = DISPLACEMENT + CLOUD_SWITCH_LAG     # 30 candles

# ===== توابع کمکی =====
def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """محاسبه اندیکاتورهای Ichimoku"""
    # Tenkan-sen (Conversion Line)
    high_9 = df['high'].rolling(window=TENKAN_PERIOD).max()
    low_9 = df['low'].rolling(window=TENKAN_PERIOD).min()
    df['tenkan'] = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line)
    high_26 = df['high'].rolling(window=KIJUN_PERIOD).max()
    low_26 = df['low'].rolling(window=KIJUN_PERIOD).min()
    df['kijun'] = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(DISPLACEMENT)
    
    # Senkou Span B (Leading Span B)
    high_52 = df['high'].rolling(window=SENKOU_SPAN_B_PERIOD).max()
    low_52 = df['low'].rolling(window=SENKOU_SPAN_B_PERIOD).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(DISPLACEMENT)
    
    # Chikou Span (Lagging Span)
    df['chikou_span'] = df['close'].shift(-DISPLACEMENT)
    
    return df

def detect_tenkan_kijun_cross(df: pd.DataFrame) -> list:
    """تشخیص تقاطع Tenkan و Kijun با ذخیره high/low کندل"""
    crosses = []
    
    for i in range(1, len(df)):
        prev_tenkan = df.iloc[i-1]['tenkan']
        prev_kijun = df.iloc[i-1]['kijun']
        curr_tenkan = df.iloc[i]['tenkan']
        curr_kijun = df.iloc[i]['kijun']
        
        # Bullish cross (Tenkan crosses above Kijun)
        if prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun:
            crosses.append({
                'index': i,
                'timestamp': df.index[i],
                'type': 'bullish',
                'price': df.iloc[i]['close'],
                'candle_high': df.iloc[i]['high'],
                'candle_low': df.iloc[i]['low']
            })
        
        # Bearish cross (Tenkan crosses below Kijun)
        elif prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun:
            crosses.append({
                'index': i,
                'timestamp': df.index[i],
                'type': 'bearish',
                'price': df.iloc[i]['close'],
                'candle_high': df.iloc[i]['high'],
                'candle_low': df.iloc[i]['low']
            })
    
    return crosses

def detect_cloud_switch(df: pd.DataFrame, start_idx: int, end_idx: int) -> dict:
    """تشخیص تغییر وضعیت ابر در بازه مشخص با اعتبارسنجی دقیق"""
    if start_idx >= len(df) or end_idx >= len(df):
        return None
    
    # بررسی تغییر فاز در بازه مشخص
    for i in range(start_idx, min(end_idx + 1, len(df))):
        if i < DISPLACEMENT:  # ابر قبل از جابجایی 26 کندلی وجود ندارد
            continue
            
        # وضعیت فعلی ابر
        curr_span_a = df.iloc[i]['senkou_span_a']
        curr_span_b = df.iloc[i]['senkou_span_b']
        
        if pd.isna(curr_span_a) or pd.isna(curr_span_b):
            continue
            
        # وضعیت قبلی ابر (کندل قبل)
        prev_span_a = df.iloc[i-1]['senkou_span_a']
        prev_span_b = df.iloc[i-1]['senkou_span_b']
        
        if pd.isna(prev_span_a) or pd.isna(prev_span_b):
            continue
            
        # تشخیص تغییر فاز
        prev_bullish = prev_span_a > prev_span_b
        curr_bullish = curr_span_a > curr_span_b
        
        if prev_bullish != curr_bullish:
            return {
                'index': i,
                'timestamp': df.index[i],
                'old_state': 'bullish' if prev_bullish else 'bearish',
                'new_state': 'bullish' if curr_bullish else 'bearish',
                'span_a': curr_span_a,
                'span_b': curr_span_b
            }
    
    return None
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
df = compute_ichimoku(df)
df["ema"] = df["close"].ewm(span=EMA_LENGTH, adjust=False).mean()
df["prev_ema"] = df["ema"].shift(1)

# ===== تشخیص تقاطع‌های Tenkan و Kijun =====
crosses = detect_tenkan_kijun_cross(df)
print(f"Tenkan-Kijun crosses detected: {len(crosses)}")
# ===== تشخیص باکس‌ها بر اساس Ichimoku =====
boxes = []
valid_crosses = []

for i, cross in enumerate(crosses):
    cross_idx = cross['index']
    cross_type = cross['type']
    cross_timestamp = cross['timestamp']
    
    # بررسی وجود کراس‌های جدید در بازه فعلی (از کندل بعد از کراس تا 30 کندل بعد)
    has_new_cross = False
    for j in range(i+1, len(crosses)):
        future_cross = crosses[j]
        # اگر کراس جدید در بازه 1 تا 30 کندل بعدی قرار دارد
        if future_cross['index'] <= cross_idx + CLOUD_SWITCH_CHECK_END:
            print(f"⚠️ کراس جدید در {future_cross['timestamp']} شناسایی شد (در بازه 30 کندلی) - نادیده گرفتن کراس فعلی در {cross_timestamp}")
            has_new_cross = True
            break
    
    if has_new_cross:
        continue

    # محدوده بررسی تغییر فاز ابر (از 22 تا 30 کندل بعد از کراس)
    start_check_idx = cross_idx + CLOUD_SWITCH_CHECK_START
    end_check_idx = cross_idx + CLOUD_SWITCH_CHECK_END
    
    # تشخیص تغییر فاز ابر در بازه مشخص
    cloud_switch = None
    for idx in range(start_check_idx, min(end_check_idx + 1, len(df))):
        if idx < DISPLACEMENT:  # ابر قبل از جابجایی 26 کندلی وجود ندارد
            continue
            
        # وضعیت فعلی ابر
        curr_span_a = df.iloc[idx]['senkou_span_a']
        curr_span_b = df.iloc[idx]['senkou_span_b']
        
        if pd.isna(curr_span_a) or pd.isna(curr_span_b):
            continue
            
        # وضعیت قبلی ابر (کندل قبل)
        prev_span_a = df.iloc[idx-1]['senkou_span_a']
        prev_span_b = df.iloc[idx-1]['senkou_span_b']
        
        if pd.isna(prev_span_a) or pd.isna(prev_span_b):
            continue
            
        # تشخیص وضعیت ابر
        prev_bullish = prev_span_a > prev_span_b
        curr_bullish = curr_span_a > curr_span_b
        
        # بررسی تغییر فاز ابر
        if cross_type == 'bearish' and prev_bullish and not curr_bullish:
            # تغییر از مثبت به منفی برای کراس نزولی
            cloud_switch = {
                'index': idx,
                'timestamp': df.index[idx],
                'type': 'bearish',
                'span_a': curr_span_a,
                'span_b': curr_span_b
            }
            break
            
        elif cross_type == 'bullish' and not prev_bullish and curr_bullish:
            # تغییر از منفی به مثبت برای کراس صعودی
            cloud_switch = {
                'index': idx,
                'timestamp': df.index[idx],
                'type': 'bullish',
                'span_a': curr_span_a,
                'span_b': curr_span_b
            }
            break
    
    if cloud_switch is None:
        print(f"❌ تغییر فاز ابر مناسب برای کراس {cross_type} در {cross_timestamp} یافت نشد")
        continue
    
    print(f"✅ تغییر فاز ابر شناسایی شد: {cloud_switch['timestamp']} (از {'مثبت' if cross_type == 'bearish' else 'منفی'} به {'منفی' if cross_type == 'bearish' else 'مثبت'})")
    
    # ایجاد باکس با مرزهای صحیح
    if cross_type == 'bullish':
        box = {
            "start": cross_timestamp,
            "end": cloud_switch['timestamp'],
            "top": cross['candle_high'],  # High کندل کراس
            "bottom": cross['candle_low'],  # Low کندل کراس
            "type": "buy",
            "id": len(boxes) + 1,
            "ema_has_entered": False,
            "tp_hit": False,
            "cross_idx": cross_idx,
            "cloud_switch_idx": cloud_switch['index']
        }
    else:  # کراس نزولی
        box = {
            "start": cross_timestamp,
            "end": cloud_switch['timestamp'],
            "top": cross['candle_high'],  # High کندل کراس
            "bottom": cross['candle_low'],  # Low کندل کراس
            "type": "sell",
            "id": len(boxes) + 1,
            "ema_has_entered": False,
            "tp_hit": False,
            "cross_idx": cross_idx,
            "cloud_switch_idx": cloud_switch['index']
        }
    
    boxes.append(box)
    valid_crosses.append(cross)
    print(f"✅ باکس معتبر {box['type']} ایجاد شد | "
          f"شروع: {box['start']} | "
          f"پایان: {box['end']} | "
          f"سقف: {box['top']:.6f} | "
          f"کف: {box['bottom']:.6f}")

print(f"Number of valid boxes: {len(boxes)} (Buy: {sum(1 for b in boxes if b['type']=='buy')} | Sell: {sum(1 for b in boxes if b['type']=='sell')})")

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

    # ===== Box activation =====
    if active_box is None or (active_box_index + 1 < len(boxes) and current_time >= boxes[active_box_index + 1]["start"]):
        # Move to the next box if time has reached its start
        for idx, box in enumerate(boxes):
            if current_time >= box["start"]:
                active_box = box
                active_box_index = idx
                last_trade_result = None
                print(f"📦 Activated box {box['id']} ({box['type']}) at {current_time}")

    # ===== EMA entry gate =====
    if active_box and not active_box["ema_has_entered"]:
        if active_box["type"] == "buy":
            # For long: EMA must be above the box high
            if current_candle["ema"] > active_box["top"]:
                active_box["ema_has_entered"] = True
                print(f"📈 EMA entered BUY box {active_box['id']} at {current_time}")
        else:
            # For short: EMA must be below the box low
            if current_candle["ema"] < active_box["bottom"]:
                active_box["ema_has_entered"] = True
                print(f"📉 EMA entered SELL box {active_box['id']} at {current_time}")

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
                risk_free_level = entry_price * (1 + RISK_FREE_THRESHOLD)
                if position["max_price"] >= risk_free_level:
                    position["risk_free_active"] = True
                    position["stop_loss"] = entry_price  # Set SL to entry price (breakeven)
                    print(f"🔒 Risk-free mode ACTIVATED for LONG at {current_time} | SL moved to entry: {entry_price:.6f}")
            else:
                risk_free_level = entry_price * (1 - RISK_FREE_THRESHOLD)
                if position["min_price"] <= risk_free_level:
                    position["risk_free_active"] = True
                    position["stop_loss"] = entry_price  # Set SL to entry price (breakeven)
                    print(f"🔒 Risk-free mode ACTIVATED for SHORT at {current_time} | SL moved to entry: {entry_price:.6f}")
        
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
        if exit_reason is None:
            if position["type"] == "long":
                if current_candle["low"] <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_time = current_time
                    exit_reason = "SL"
                elif current_candle["high"] >= take_profit:
                    exit_price = take_profit
                    exit_time = current_time
                    exit_reason = "TP"
            elif position["type"] == "short":
                if current_candle["high"] >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_time = current_time
                    exit_reason = "SL"
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
            elif exit_reason == "EMA_SL":
                last_trade_result = "EMA_SL"
    
    for position in positions_to_remove:
        open_positions.remove(position)

    # After trade exits, handle SL compensation reset/reduction
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

    # ===== ورود به معاملات جدید =====
    if active_box and active_box["ema_has_entered"] and not active_box["tp_hit"]:
        box = active_box
        risk_balance = balance if compounding else INITIAL_BALANCE
        base_tp = TAKE_PROFIT_PCT
        take_profit_pct = base_tp
        compensation_added = 0.0
        
        # Always apply compensation logic for every new trade
        if tp_type in [3, 4] and cumulative_sl_loss > 0:
            if cumulative_sl_loss <= 0.03:
                compensation_added = cumulative_sl_loss
            else:
                compensation_added = 0
            take_profit_pct += compensation_added
        
        if tp_type in [2, 3]:
            take_profit_pct += spread_comp
        
        entry_signal = False
        trade_type = None
        signal_details = ""
        box_open_position = any(pos["box_id"] == box["id"] for pos in open_positions)
        
        # Entry conditions based on Ichimoku strategy
        if not box_open_position and box["type"] == "sell":
            # Short entry: EMA below box low
            if current_candle["ema"] < box["bottom"]:
                entry_signal = True
                trade_type = "short"
                signal_details = f"EMA below box bottom ({box['bottom']:.6f})"
        
        if not box_open_position and box["type"] == "buy":
            # Long entry: EMA above box high
            if current_candle["ema"] > box["top"]:
                entry_signal = True
                trade_type = "long"
                signal_details = f"EMA above box top ({box['top']:.6f})"
        
        # Re-entry conditions
        if not box_open_position and last_trade_result in ["SL", "EMA_SL"]:
            if box["type"] == "sell":
                if current_candle["ema"] < box["bottom"]:
                    entry_signal = True
                    trade_type = "short"
                    signal_details = f"(Re-entry) EMA below box bottom ({box['bottom']:.6f})"
            elif box["type"] == "buy":
                if current_candle["ema"] > box["top"]:
                    entry_signal = True
                    trade_type = "long"
                    signal_details = f"(Re-entry) EMA above box top ({box['top']:.6f})"
        
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
print(f"- Risk-Free: {'Enabled' if risk_free_enabled else 'Disabled'} (Threshold: {RISK_FREE_THRESHOLD*100:.1f}%)")
print(f"- Initial Balance: ${INITIAL_BALANCE:.2f}")
print(f"- Ichimoku Settings: Tenkan={TENKAN_PERIOD}, Kijun={KIJUN_PERIOD}, Senkou={SENKOU_SPAN_B_PERIOD}")
print(f"- Cloud Switch: Lead={CLOUD_SWITCH_LEAD}, Lag={CLOUD_SWITCH_LAG}")

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
                "ema_entered": box.get("ema_has_entered", False)
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