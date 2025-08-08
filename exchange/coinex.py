# -*- coding: utf-8 -*-
import os
import time
import hmac
import hashlib
import json
import requests
from typing import Optional, Dict, Literal, Any
from dotenv import load_dotenv
from decimal import Decimal, InvalidOperation

load_dotenv()
# -------------------------------------------------------------------
# بارگذاری امن کلیدها از متغیرهای محیطی
# -------------------------------------------------------------------
API_KEY    = os.getenv("COINEX_API_KEY")
API_SECRET = os.getenv("COINEX_API_SECRET")
BASE_URL   = "https://api.coinex.com"

if not API_KEY or not API_SECRET:
    raise RuntimeError("لطفاً متغیرهای محیطی COINEX_API_KEY و COINEX_API_SECRET را تنظیم کنید.")

# -------------------------------------------------------------------
# توابع کمکی
# -------------------------------------------------------------------
def _get_timestamp_ms() -> str:
    """زمان فعلی به میلی‌ثانیه."""
    return str(int(time.time() * 1000))

def _sign(method: str, path: str, body: Optional[Dict[str, Any]], timestamp: str) -> str:
    """
    آماده‌سازی و امضای داده‌ها طبق مستندات CoinEx:
      signature_str = METHOD + path + compact_json_body + timestamp
    """
    # 1) بدنه را به JSON کامپکت تبدیل می‌کنیم
    body_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
    # 2) رشته‌ی ادغام‌شده برای امضا
    signature_str = f"{method.upper()}{path}{body_str}{timestamp}"

    # 3) محاسبه‌ی HMAC-SHA256 با کلید مخفی و encoding latin-1
    signature = hmac.new(
        API_SECRET.encode("latin-1"),
        signature_str.encode("latin-1"),
        hashlib.sha256
    ).hexdigest().lower()

    return signature

def _build_headers(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    ساخت هدرهای مورد نیاز:
      - Content-Type: application/json; charset=utf-8
      - Accept: application/json
      - X-COINEX-KEY
      - X-COINEX-SIGN
      - X-COINEX-TIMESTAMP
    """
    ts = _get_timestamp_ms()
    sig = _sign(method, path, body, ts)
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "X-COINEX-KEY": API_KEY,
        "X-COINEX-SIGN": sig,
        "X-COINEX-TIMESTAMP": ts
    }

def _post(path: str, body: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Any:
    """
    ارسال درخواست POST و بررسی پاسخ
    """
    url     = BASE_URL + path
    headers = _build_headers("POST", path, body)
    body_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
    
    if dry_run:
        return {"url": url, "headers": headers, "data": body_str}

    resp = requests.post(url, headers=headers, data=body_str)
        # بررسی status
    if not resp.ok:
        # متن کامل برای دیباگ
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Response is not JSON: {resp.text}")

    if data.get("code") != 0:
        raise RuntimeError(f"API error {data.get('code')}: {data.get('message')} — {data.get('data')}")

    return data["data"]


# -------------------------------------------------------------------
# فانکشن‌های فیوچرز
# -------------------------------------------------------------------

_VALID_SIDES = {"buy", "sell"}
_VALID_ORDERTYPES = {"limit", "market", "maker_only", "ioc", "fok"}
_VALID_TRIGGER_TYPES = {"latest_price", "mark_price", "index_price"}
_VALID_STP = {"ct", "cm", "both"}

def place_futures_order(
    market: str,
    side: Literal["buy","sell"],
    order_type: Literal["limit","market","maker_only","ioc","fok"],
    amount: float,
    price: Optional[float] = None,
    client_id: Optional[str] = None,
    trigger_price: Optional[float] = None,
    trigger_price_type: Optional[Literal["latest_price","mark_price","index_price"]] = None,
    is_hide: bool = False,
    stp_mode: Optional[Literal["ct","cm","both"]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    
    if side not in _VALID_SIDES:
        raise ValueError("side must be 'buy' or 'sell'")
    if order_type not in _VALID_ORDERTYPES:
        raise ValueError(f"order_type must be one of {_VALID_ORDERTYPES}")
    if order_type == "limit" and price is None:
        raise ValueError("برای order_type='limit' پارامتر price ضروری است.")
    if stp_mode and stp_mode not in _VALID_STP:
        raise ValueError(f"stp_mode must be one of {_VALID_STP}")
    if trigger_price is not None and trigger_price_type not in _VALID_TRIGGER_TYPES:
        raise ValueError(f"trigger_price_type must be one of {_VALID_TRIGGER_TYPES}")


    try:
        amt_str = str(Decimal(amount))
    except (InvalidOperation, TypeError):
        amt_str = str(amount)

    path = "/v2/futures/order"
    body: Dict[str, Any] = {
        "market": market,
        "market_type": "FUTURES",
        "side": side,
        "type": order_type,
        "amount": str(amount)
    }

    if price is not None:
        try:
            body["price"] = str(Decimal(price))
        except (InvalidOperation, TypeError):
            body["price"] = str(price)

    if client_id:                     body["client_id"] = client_id
    if is_hide:                       body["is_hide"] = True
    if stp_mode:                      body["stp_mode"] = stp_mode
    if trigger_price is not None and trigger_price_type:
        body["trigger_price"]      = str(trigger_price)
        body["trigger_price_type"] = trigger_price_type

    return _post(path, body, dry_run=dry_run)

def close_futures_position(
    market: str,
    close_type: Literal["limit","market"],
    amount: Optional[float] = None,
    price: Optional[float] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    if close_type not in {"limit", "market"}:
        raise ValueError("close_type must be 'limit' or 'market'")
    if close_type == "limit" and price is None:
        raise ValueError("برای بستن با لیمیت، پارامتر price ضروری است.")

    body = {
        "market": market,
        "market_type": "FUTURES",
        "type": close_type
    }
    if amount is not None:
        body["amount"] = str(Decimal(amount))
    if close_type == "limit":
        body["price"] = str(Decimal(price))

    path = "/v2/futures/close-position"
    return _post(path, body, dry_run=dry_run)

def set_futures_leverage(
    market: str,
    leverage: int,
    margin_mode: Literal["isolated","cross"] = "isolated"
) -> Dict[str, Any]:
    path = "/v2/futures/adjust-position-leverage"
    body = {
        "market": market,
        "market_type": "FUTURES",
        "leverage": leverage,
        "margin_mode": margin_mode
    }
    return _post(path, body)

# -------------------------------------------------------------------
# مثال استفاده
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1) تنظیم لوریج ۲۰x (isolated) در BTCUSDT
    print("تنظیم لوریج:", set_futures_leverage("BTCUSDT", leverage=20, margin_mode="isolated"))

    # 2) باز کردن پوزیشن مارکت لانگ با حجم 0.01 BTC
    print("باز کردن مارکت لانگ:", place_futures_order(
        market="BTCUSDT",
        side="buy",
        order_type="market",
        amount=0.01
    ))

    # 3) بستن نصف پوزیشن با سفارش لیمیت در قیمت 35000
    print("بستن پوزیشن:", close_futures_position(
        market="BTCUSDT",
        close_type="limit",
        amount=0.005,
        price=35000
    ))
