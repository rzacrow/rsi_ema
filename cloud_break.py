import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === تنظیمات ورودی کاربر ===
SYMBOL = input("Enter cryptocurrency symbol (e.g. XRPUSDT): ").strip().upper()
STOP_LOSS_PCT = float(input("Enter stop loss percentage (e.g. 0.3 for 0.3%): ")) / 100
TAKE_PROFIT_PCT = float(input("Enter take profit percentage (e.g. 0.5 for 0.5%): ")) / 100

# === پارامترهای ایچیموکو (ضربدر 5) ===
CONVERSION_LINE = 9 * 5  # 45
BASE_LINE = 26 * 5       # 130
LEADING_B_PERIOD = 52 * 5 # 260
LAGGING_SPAN = 26        # بدون تغییر
MAX_REVERSE_CANDLES = 1

# ===== توابع ایچیموکو =====
def ichimoku(df, conversion_line=CONVERSION_LINE, base_line=BASE_LINE, 
             lagging_span=LAGGING_SPAN, leading_b_period=LEADING_B_PERIOD):
    """
    محاسبه اندیکاتور ایچیموکو با پارامترهای سفارشی
    """
    def kijun_sen(df, period):
        high = df['high']
        low = df['low']
        return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2

    tenkan_sen = kijun_sen(df, conversion_line)
    kijun_sen_line = kijun_sen(df, base_line)
    leading_span_a = (tenkan_sen + kijun_sen_line) / 2
    leading_span_b = kijun_sen(df, leading_b_period)
    lagging = df['close'].shift(-lagging_span)
    
    upper_kumo = leading_span_a.combine(leading_span_b, max)
    lower_kumo = leading_span_a.combine(leading_span_b, min)

    cloud_color = [
        "green" if a > b else "red"
        for a, b in zip(leading_span_a.fillna(0), leading_span_b.fillna(0))
    ]

    return {
        'conversion_line': tenkan_sen.tolist(),
        'baseline': kijun_sen_line.tolist(),
        'leading_span_a': leading_span_a.tolist(),
        'leading_span_b': leading_span_b.tolist(),
        'lagging_span': lagging.tolist(),
        'upper_kumo': upper_kumo.tolist(),
        'lower_kumo': lower_kumo.tolist(),
        'cloud': cloud_color
    }

def ichi_signals(df, conversion_line=CONVERSION_LINE, base_line=BASE_LINE, 
                 lagging_span=LAGGING_SPAN, leading_b_period=LEADING_B_PERIOD, 
                 max_reverse_candles=MAX_REVERSE_CANDLES):
    """
    تولید سیگنال‌های معاملاتی بر اساس استراتژی اصلاح شده
    """
    ichi = ichimoku(df, conversion_line, base_line, lagging_span, leading_b_period)
    
    upper = ichi['upper_kumo']
    lower = ichi['lower_kumo']
    cloud = ichi['cloud']
    
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    closes = df['close'].tolist()
    opens = df['open'].tolist()
    times = df.index.tolist()
    
    # شناسایی تغییرات رنگ ابر
    changes = []
    for i in range(1, len(cloud)):
        if (cloud[i] == 'red' and cloud[i-1] == 'green') or (cloud[i] == 'green' and cloud[i-1] == 'red'):
            changes.append(i + lagging_span)
    
    signals = []
    # ذخیره نقاط ورود به ابر برای جلوگیری از سیگنال‌های تکراری
    cloud_entries = set()
    
    for i in range(30, len(highs)):
        # جلوگیری از خطای اندیس
        if i < lagging_span or i-1 < lagging_span:
            continue
            
        # شناسایی ورود به ابر
        entry_detected = False
        entry_type = None
        
        # تشخیص ورود از بالا (برای فروش)
        if (highs[i-1] > upper[i-1-lagging_span] and 
            highs[i] <= upper[i-lagging_span] and 
            i not in cloud_entries):
            entry_detected = True
            entry_type = 'sell'
            cloud_entries.add(i)
        
        # تشخیص ورود از پایین (برای خرید)
        elif (lows[i-1] < lower[i-1-lagging_span] and 
              lows[i] >= lower[i-lagging_span] and 
              i not in cloud_entries):
            entry_detected = True
            entry_type = 'buy'
            cloud_entries.add(i)
        
        if not entry_detected:
            continue
        
        # جستجوی اولین شکست ابر پس از ورود
        breakout_index = None
        reverse_count = 0
        
        # برای سیگنال فروش: جستجوی خروج از پایین ابر
        if entry_type == 'sell':
            for j in range(i, min(i+30, len(lows))):
                if j < lagging_span or j-1 < lagging_span:
                    continue
                
                # شمارش کندل‌های صعودی (مخالف روند)
                if opens[j] < closes[j]:
                    reverse_count += 1
                
                # شناسایی شکست به پایین
                if (lows[j] <= lower[j-lagging_span] and 
                    lows[j-1] > lower[j-1-lagging_span]):
                    breakout_index = j
                    break
        
        # برای سیگنال خرید: جستجوی خروج از بالا ابر
        elif entry_type == 'buy':
            for j in range(i, min(i+30, len(highs))):
                if j < lagging_span or j-1 < lagging_span:
                    continue
                
                # شمارش کندل‌های نزولی (مخالف روند)
                if opens[j] > closes[j]:
                    reverse_count += 1
                
                # شناسایی شکست به بالا
                if (highs[j] >= upper[j-lagging_span] and 
                    highs[j-1] < upper[j-1-lagging_span]):
                    breakout_index = j
                    break
        
        # بررسی شرایط ورود
        if breakout_index is not None and reverse_count <= max_reverse_candles:
            # تعیین قیمت ورود
            if entry_type == 'sell':
                entry_price = lower[breakout_index-lagging_span]  # ورود روی اسپن پایین
            else:
                entry_price = upper[breakout_index-lagging_span]  # ورود روی اسپن بالا
            
            signals.append({
                'type': entry_type,
                'confirmation_index': i,          # نقطه ورود به ابر
                'entry_index': breakout_index,    # نقطه شکست ابر
                'entry_time': times[breakout_index],
                'entry_price': entry_price        # قیمت اسپن مربوطه
            })
    
    # فیلتر سیگنال‌ها بر اساس تغییرات ابر
    verified = []
    for change_idx in changes:
        for j in range(len(signals)):
            signal = signals[j]
            
            if (signal['confirmation_index'] >= change_idx and 
                (j == 0 or signals[j-1]['confirmation_index'] < change_idx)):
                
                if signal not in verified:
                    verified.append(signal)
    
    verified.sort(key=lambda x: x['entry_index'])
    
    return verified
def simulate_trade(signal, df, stop_loss_pct, take_profit_pct):
    entry_idx = signal['entry_index']    # نقطه شکست ابر
    entry_time = signal['entry_time']
    entry_price = signal['entry_price']
    
    # تعیین حد ضرر و حد سود
    if signal['type'] == 'buy':
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
    else:
        stop_loss = entry_price * (1 + stop_loss_pct)
        take_profit = entry_price * (1 - take_profit_pct)
    
    # شبیه‌سازی معامله از نقطه ورود تا پایان
    exit_reason = "End of Data"
    exit_price = df['close'].iloc[-1]
    exit_time = df.index[-1]
    
    # بررسی خروج با SL یا TP (شروع از کندل بعد از نقطه ورود)
    for i in range(entry_idx + 1, len(df)):
        candle = df.iloc[i]
        
        if signal['type'] == 'buy':
            if candle['low'] <= stop_loss:
                exit_reason = "SL"
                exit_price = stop_loss
                exit_time = df.index[i]
                break
            elif candle['high'] >= take_profit:
                exit_reason = "TP"
                exit_price = take_profit
                exit_time = df.index[i]
                break
        else:
            if candle['high'] >= stop_loss:
                exit_reason = "SL"
                exit_price = stop_loss
                exit_time = df.index[i]
                break
            elif candle['low'] <= take_profit:
                exit_reason = "TP"
                exit_price = take_profit
                exit_time = df.index[i]
                break
    
    # محاسبه سود/زیان
    if signal['type'] == 'buy':
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100
    
    return {
        'type': signal['type'],
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'pnl_pct': pnl,
        'exit_reason': exit_reason
    }
# ===== خواندن داده‌ها =====
file_path = f"data/{SYMBOL}.csv"
if not os.path.exists(file_path):
    print(f"File {file_path} not found!")
    exit()

try:
    # خواندن داده‌ها با فرمت مشخص شده
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

# ===== تولید سیگنال‌ها =====
print("Generating Ichimoku signals...")
signals = ichi_signals(df)

if not signals:
    print("No signals generated")
    exit()

# ===== شبیه‌سازی معاملات =====
print("Simulating trades...")
trades = []
for signal in signals:
    trade = simulate_trade(signal, df, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
    trades.append(trade)

# ===== ذخیره نتایج =====
output_columns = ['type', 'entry_time', 'entry_price', 'exit_time', 
                 'exit_price', 'pnl_pct', 'exit_reason']
trades_df = pd.DataFrame(trades)[output_columns]

# ذخیره در فایل CSV
output_file = f"trades_{SYMBOL}.csv"
trades_df.to_csv(output_file, index=False)
print(f"\nTrades saved to {output_file}")

# ===== گزارش نهایی =====
print("\n" + "="*50)
print(f"Backtest Results for {SYMBOL}")
print("="*50)
print(f"Total Signals: {len(trades)}")

if trades:
    win_trades = [t for t in trades if t['pnl_pct'] > 0]
    loss_trades = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(win_trades)/len(trades)*100
    
    print(f"Winning Trades: {len(win_trades)} ({win_rate:.2f}%)")
    print(f"Losing Trades: {len(loss_trades)}")
    
    # آمار خروج
    tp_exits = [t for t in trades if t['exit_reason'] == 'TP']
    sl_exits = [t for t in trades if t['exit_reason'] == 'SL']
    print(f"TP Exits: {len(tp_exits)} | SL Exits: {len(sl_exits)}")
    
    if win_trades:
        avg_win = sum(t['pnl_pct'] for t in win_trades)/len(win_trades)
        max_win = max(t['pnl_pct'] for t in trades)
    else:
        avg_win = max_win = 0.0
    
    if loss_trades:
        avg_loss = sum(t['pnl_pct'] for t in loss_trades)/len(loss_trades)
        max_loss = min(t['pnl_pct'] for t in trades)
    else:
        avg_loss = max_loss = 0.0
    
    print(f"\nAverage Win: {avg_win:.4f}%")
    print(f"Average Loss: {avg_loss:.4f}%")
    print(f"Max Win: {max_win:.4f}%")
    print(f"Max Loss: {max_loss:.4f}%")

print("\n" + "="*50)
print("Backtest completed successfully!")
print("="*50)