import requests
import hashlib
import hmac

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
url = "https://api.kcex.com/futures/v1/orders"

params = {
    "symbol": "BTC_USDT",
    "side": "BUY",
    "type": "LIMIT",
    "price": "50000",
    "quantity": "0.001",
    "timestamp": int(time.time() * 1000)
}

# تولید امضا
signature = hmac.new(
    api_secret.encode(),
    msg=urlencode(params).encode(),
    digestmod=hashlib.sha256
).hexdigest()

params["signature"] = signature
headers = {"X-MBX-APIKEY": api_key}
response = requests.post(url, headers=headers, params=params)
print(response.json())