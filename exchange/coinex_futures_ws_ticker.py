import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("coinex-futures-ws")


async def _keepalive_ping(ws: websockets.WebSocketClientProtocol, interval: float = 30.0):
    """هر `interval` ثانیه یک server.ping می‌فرستیم (RPC style)"""
    ping_id = 100000
    try:
        while True:
            await asyncio.sleep(interval)
            ping_msg = {"method": "server.ping", "params": [], "id": ping_id}
            try:
                await ws.send(json.dumps(ping_msg, separators=(",", ":")))
            except Exception:
                LOGGER.exception("failed to send ping")
                return
            ping_id += 1
    except asyncio.CancelledError:
        return


async def subscribe_futures_ticker(
    market: str,
    on_price: Callable[[float], None],
    ws_url: str = "wss://socket.coinex.com/v2/futures",
    ping_interval: float = 30.0,
    reconnect_backoff_base: float = 1.0,
):
    """
    مشترک شدن روی ticker فیوچرز یک مارکت و فراخوانی on_price با قیمت جدید.
    on_price: تابع sync یا async که یک float می‌پذیره.
    """
    backoff = reconnect_backoff_base
    while True:
        try:
            LOGGER.info("Connecting to %s", ws_url)
            # compression='deflate' طبق مستندات توصیه شده
            async with websockets.connect(ws_url, compression="deflate", max_size=None) as ws:
                # subscribe (برای فیوچرز می‌فرستیم market_list)
                subscribe_msg = {"method": "state.subscribe", "params": {"market_list": [market]}, "id": 1}
                await ws.send(json.dumps(subscribe_msg, separators=(",", ":")))
                LOGGER.info("Sent subscribe for %s", market)

                # لانچ keepalive ping
                ping_task = asyncio.create_task(_keepalive_ping(ws, interval=ping_interval))

                # خواندن پیام‌ها
                async for raw in ws:
                    # بعضی پیام‌ها ممکنه فشرده/باینری باشند ولی کتابخانه websockets handle می‌کنه
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        # اگر JSON نبود نادیده بگیر
                        continue

                    # پاسخ subscribe (id==1) - میتوان لاگ کرد ولی نیازی به پردازش خاص نیست
                    if msg.get("id") == 1 and msg.get("code") is not None:
                        if msg.get("code") == 0:
                            LOGGER.debug("Subscribe acknowledged")
                        else:
                            LOGGER.warning("Subscribe returned error: %s", msg)
                        continue

                    # پیام push برای وضعیت بازار (مطابق docs: method == "state.update", data -> state_list)
                    if msg.get("method") == "state.update":
                        data = msg.get("data") or msg.get("params") or {}
                        state_list = data.get("state_list") if isinstance(data, dict) else None
                        if state_list and isinstance(state_list, list):
                            for item in state_list:
                                # هر item شامل فیلد 'market' و 'last' و ... هست (مطابق docs)
                                try:
                                    if item.get("market") == market:
                                        last = item.get("last")
                                        if last is not None:
                                            price = float(last)
                                            # اگر on_price async است، await کن
                                            if asyncio.iscoroutinefunction(on_price):
                                                await on_price(price)
                                            else:
                                                on_price(price)
                                except Exception:
                                    LOGGER.exception("Failed processing state_list item: %s", item)

                # اگر حلقه async for خارج شد (قطع اتصال)، cancel ping
                ping_task.cancel()

        except Exception:
            LOGGER.exception("WebSocket error, reconnecting in %.1f seconds", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)  # exponential backoff


# مثال استفاده ساده
def print_price(p: float):
    print("FUTURES PRICE:", p)


if __name__ == "__main__":
    market_name = "BTCUSDT"  # یا هر مارکت فیوچرز که CoinEx پشتیبانی می‌کنه
    asyncio.run(subscribe_futures_ticker(market_name, print_price))
