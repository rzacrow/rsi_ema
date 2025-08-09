import time, hashlib, hmac, json, requests, uuid, os
from dotenv import load_dotenv
from collections import OrderedDict

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("SECRET_KEY")
BASE_URL = "https://lbkperp.lbank.com"
SYMBOL = "BTC_USDT"

def get_server_timestamp():
    url = f"{BASE_URL}/cfd/openApi/v1/pub/getTime"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return str(r.json()["data"])

def make_sign(params: dict) -> str:
    sorted_params = OrderedDict(sorted(params.items()))
    param_str = "&".join(f"{k}={v}" for k, v in sorted_params.items())
    md5_str = hashlib.md5(param_str.encode()).hexdigest().upper()
    return hmac.new(
        API_SECRET.encode(), 
        md5_str.encode(), 
        hashlib.sha256
    ).hexdigest()

def place_futures_order(volume: float, direction: str, offset: str, product_group: str = "SwapU"):
    ts      = get_server_timestamp()
    echostr = uuid.uuid4().hex
    sigmeth = "HmacSHA256"

    # ۱) آماده‌سازی پارامترها
    params = {
        "api_key": API_KEY,
        "productGroup": product_group,
        "symbol": SYMBOL,
        "positionSide":"long",  # "LONG" / "SHORT"
        "type": "open",             # "open" / "close"
        "orderType": "market",
        "volume":           str(volume),
        "timestamp":        ts,
        "echostr":          echostr,
        "signature_method": sigmeth
    }
    # ۲) اضافه‌کردن امضا
    params["sign"] = make_sign(params)

    url = f"{BASE_URL}/cfd/openApi/v1/prv/order"
    headers = {
        "Content-Type":      "application/json",
        "timestamp":         ts,
        "echostr":           echostr,
        "signature_method":  sigmeth
    }

    # ۳) ارسال POST با data= (فرم)
    res = requests.post(url, headers=headers, json=params, timeout=5)


    if res.status_code != 200:
        print("→ Request URL:",    res.request.url)
        print("→ Request Headers:",res.request.headers)
        print("→ Request Body:",   res.request.body)
        raise Exception(f"----- Error {res.status_code}: {res.text}")

    return res.json()

def set_leverage(symbol: str, leverage: int):
    ts = get_server_timestamp()
    echostr = uuid.uuid4().hex

    params = {
        "api_key": API_KEY,
        "symbol": symbol,
        "leverage": str(leverage),
        "timestamp": ts,
        "echostr": echostr,
        "signature_method": "HmacSHA256"
    }
    
    params["sign"] = make_sign(params)
    
    url = f"{BASE_URL}/cfd/openApi/v1/position/leverage"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    res = requests.post(url, data=params, headers=headers)
    
    if res.status_code != 200:
        raise Exception(f"Leverage Error {res.status_code}: {res.text}")
    return res.json()



def initialize(symbol: str, leverage: int):
    resp = set_leverage(symbol, leverage)
    if resp.get("result") == "success":
        print(f"✅ لوریج {leverage}× برای {symbol} تنظیم شد")
    else:
        print("⚠️ خطا در تنظیم لوریج:", resp)

# توابع معاملاتی (اصلاح شده)
def open_long(volume: float, leverage: int = 20):
    #initialize(SYMBOL, leverage)
    return place_futures_order(volume=volume, direction="long", offset="open")

def open_short(volume: float, leverage: int = 20):
    initialize(SYMBOL, leverage)
    return place_futures_order(volume, "short", "open")

def close_long(volume: float):
    return place_futures_order(volume, "long", "close")

def close_short(volume: float):
    return place_futures_order(volume, "short", "close")

res = open_long(volume=0.0001, leverage=80)
print(res)