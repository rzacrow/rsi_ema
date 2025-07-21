import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os

# === تنظیمات اولیه ورودی کاربر ===
symbol_input = input("Enter symbol (e.g. BTCUSDT): ").strip().upper()
# تبدیل برای درخواست به بایننس
symbol = f"{symbol_input[:3]}/{symbol_input[3:]}"

timeframe = "1m"
chunk_days = 10  # دریافت داده در بازه‌های 10 روزه برای جلوگیری از محدودیت

# ساختار پوشه و فایل
os.makedirs("data", exist_ok=True)
csv_file = f"data/{symbol_input}.csv"

# تابع خواندن آخرین timestamp موجود برای resume
def get_last_timestamp(filepath):
    try:
        with open(filepath, 'rb') as f:
            f.seek(-2048, os.SEEK_END)
            lines = f.readlines()
        last_line = lines[-1].decode().strip()
        ts_str = last_line.split(",")[1]
        ts = int(float(ts_str))
        # برگرداندن شیء datetime یک کندل بعدی
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc) + timedelta(minutes=1)
    except Exception as e:
        print(f"⚠️ خطا در خواندن آخرین timestamp: {e}")
        return None

# اتصال به صرافی
exchange = ccxt.binance({"enableRateLimit": True})

# بررسی وجود فایل و تعیین بازه‌‌ی بارگیری
days_back = None
if os.path.exists(csv_file):
    print(f"📁 فایل موجود است: {csv_file}")
    start_time = get_last_timestamp(csv_file)
    header = False
    mode = "a"
    # اگر newline انتها نیست، اضافه کن
    with open(csv_file, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b'\n':
            f.write(b'\n')
else:
    # در صورت عدم وجود فایل، ورودی تعداد روزها را بگیر
    days_back = int(input("Enter number of days to fetch (e.g. 30): ").strip())
    print(f"📁 فایل یافت نشد. ساخت فایل جدید و بارگیری {days_back} روز گذشته...")
    start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
    header = True
    mode = "w"

end_time = datetime.now(timezone.utc)
print(f"📦 دریافت داده از {start_time} تا {end_time} برای {symbol}")

current_start = start_time

# دریافت داده در بازه‌های chunk_days
while current_start < end_time:
    current_end = min(current_start + timedelta(days=chunk_days), end_time)
    since = exchange.parse8601(current_start.strftime('%Y-%m-%dT%H:%M:%SZ'))
    buffer = []
    print(f"⬇ دریافت بازه {current_start} تا {current_end}")

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"⚠️ خطا در دریافت: {e} -- تلاش دوباره بعد از 5 ثانیه...")
            time.sleep(5)
            continue

        if not bars:
            break

        buffer.extend(bars)
        last_ts = bars[-1][0]
        since = last_ts + 60_000  # یک دقیقه جلوتر

        if since >= exchange.parse8601(current_end.strftime('%Y-%m-%dT%H:%M:%SZ')):
            break

        time.sleep(0.9)

    # ذخیره به CSV
    if buffer:
        df_chunk = pd.DataFrame(buffer, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_chunk["datetime"] = pd.to_datetime(df_chunk["timestamp"], unit="ms")
        df_chunk = df_chunk[["datetime", "timestamp", "open", "high", "low", "close", "volume"]]
        df_chunk.to_csv(csv_file, mode=mode, header=header, index=False)
        print(f"✅ ذخیره {len(df_chunk)} ردیف به {csv_file}")
        header = False
        mode = "a"

    current_start = current_end
    time.sleep(1)

print(f"🎉 اتمام فرآیند بارگیری برای {symbol_input}!")
