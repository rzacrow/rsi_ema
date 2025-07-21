import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os

exchange = ccxt.binance({"enableRateLimit": True})
symbol = "BTC/USDT"
timeframe = "1m"
chunk_days = 10

os.makedirs("data", exist_ok=True)
output_csv = "data/BTCUSDT.csv"

def get_last_timestamp(filepath):
    try:
        with open(filepath, 'rb') as f:
            f.seek(-2048, 2)
            lines = f.readlines()
        last_line = lines[-1].decode().strip()
        ts_str = last_line.split(",")[1]
        ts = int(float(ts_str))  # ← اینجا اصلاح شده
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc) + timedelta(minutes=1)
    except Exception as e:
        print("⚠️ خطا در خواندن آخرین timestamp:", e)
        return None

if os.path.exists(output_csv):
    print(f"📁 فایل موجود است: {output_csv}")
    start_time = get_last_timestamp(output_csv)
    header = False
    mode = "a"

    # ✅ بررسی اینکه آیا انتهای فایل newline داره یا نه
    with open(output_csv, 'rb+') as f:
        f.seek(-1, 2)  # رفتن به آخرین بایت
        last_char = f.read(1)
        if last_char != b'\n':
            f.write(b'\n')  # اگه newline نبود، اضافه کن
else:
    print("📁 فایل وجود ندارد، دریافت از ۶ ماه پیش شروع می‌شود")
    start_time = datetime.now(timezone.utc) - timedelta(days=180)
    header = True
    mode = "w"

end_time = datetime.now(timezone.utc)

print(f"📦 دریافت داده از {start_time} تا {end_time}")
current_start = start_time

while current_start < end_time:
    current_end = min(current_start + timedelta(days=chunk_days), end_time)
    since = exchange.parse8601(current_start.strftime('%Y-%m-%dT%H:%M:%SZ'))
    chunk = []
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

        chunk += bars
        last_ts = bars[-1][0]
        since = last_ts + 60_000

        if since >= exchange.parse8601(current_end.strftime('%Y-%m-%dT%H:%M:%SZ')):
            break

        time.sleep(0.9)

    if chunk:
        df = pd.DataFrame(chunk, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["datetime", "timestamp", "open", "high", "low", "close", "volume"]]
        df.to_csv(output_csv, mode=mode, header=header, index=False)
        print(f"✅ {len(df)} ردیف ذخیره شد.")

        # بعد از بار اول، header نزن دیگه
        header = False
        mode = "a"

    current_start = current_end
    time.sleep(1)
