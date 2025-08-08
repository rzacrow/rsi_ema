import os, time, hmac, hashlib, json, requests
from dotenv import load_dotenv


load_dotenv()
API_KEY    = os.getenv("ACCESS_ID")
API_SECRET = os.getenv("SECRET_KEY")
BASE_URL   = "https://api.coinex.com"

if not API_KEY or not API_SECRET:
    raise RuntimeError("متغیرهای محیطی COINEX_API_KEY و COINEX_API_SECRET تنظیم نشده‌اند.")

def _get_timestamp_ms() -> str:
    return str(int(time.time() * 1000))

def _sign_request(method: str, path: str, body: dict | None, timestamp: str) -> str:
    # 1) stringify body
    body_json = json.dumps(body, separators=(",", ":")) if body else ""
    # 2) build the string to sign
    to_sign = method.upper() + path + body_json + timestamp

    # DEBUG: print the exact string we sign
    print("\n=== DEBUG SIGNATURE ===")
    print("to_sign:")
    print(to_sign)
    print("-----")

    # 3) compute HMAC-SHA256 using latin-1 encoding
    signature = hmac.new(
        API_SECRET.encode('latin-1'),
        to_sign.encode('latin-1'),
        hashlib.sha256
    ).hexdigest().lower()

    # DEBUG: print computed signature
    print("computed signature:", signature)
    print("=======================\n")

    return signature

def _build_headers(method: str, path: str, body: dict | None = None) -> dict:
    ts = _get_timestamp_ms()
    sign = _sign_request(method, path, body, ts)
    return {
        "Content-Type": "application/json",
        "X-COINEX-KEY": API_KEY,
        "X-COINEX-TIMESTAMP": ts,
        "X-COINEX-SIGN": sign
    }

def _post(path: str, body: dict | None = None) -> dict:
    url = BASE_URL + path
    headers = _build_headers("POST", path, body)
    print("Request URL:", url)
    print("Request Headers:", headers)
    print("Request Body:", body or {})
    resp = requests.post(url, headers=headers, data=json.dumps(body or {}))
    j = resp.json()
    if j.get("code") != 0:
        raise RuntimeError(f"خطا {j['code']}: {j['message']} — جزئیات: {j.get('data')}")
    return j["data"]

def set_futures_leverage(
    market: str,
    leverage: int,
    margin_mode: str = "isolated"
) -> dict:
    path = "/v2/futures/adjust-position-leverage"
    body = {
        "market": market,
        "market_type": "FUTURES",
        "leverage": leverage,
        "margin_mode": margin_mode
    }
    return _post(path, body)

if __name__ == "__main__":
    # فراخوانی برای تست و دیدن لاگ‌ها
    print("Running leverage adjustment...")
    res = set_futures_leverage("BTCUSDT", leverage=20, margin_mode="isolated")
    print("Result:", res)
