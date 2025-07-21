import pandas as pd
from datetime import timedelta


def read_csv_recent_days(file_path, days):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    last_date = df['datetime'].max()
    start_date = last_date - timedelta(days=days)
    df = df[df['datetime'] >= start_date]
    return df


def run_ichimoku_strategy(
    df,
    ichimoku_tf=5,
    shift=26,
    max_opposite_candles=2,
    min_cloud_depth_candles=2,
    sl1_percent=0.1,
    sl_static_percent=0.5
):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # compute ichimoku
    ich = df.resample(f'{ichimoku_tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    ich['tenkan'] = (ich['high'].rolling(9).max() + ich['low'].rolling(9).min()) / 2
    ich['kijun'] = (ich['high'].rolling(26).max() + ich['low'].rolling(26).min()) / 2
    ich['senkou_a'] = ((ich['tenkan'] + ich['kijun']) / 2).shift(shift)
    ich['senkou_b'] = ((ich['high'].rolling(52).max() + ich['low'].rolling(52).min()) / 2).shift(shift)
    ich.dropna(inplace=True)

    trades = []
    used_clouds = set()

    for time, row in ich.iterrows():
        a, b = row['senkou_a'], row['senkou_b']
        lo, hi = min(a, b), max(a, b)
        trend = 'long' if a < b else 'short'

        cloud_id = f"{time}_{lo:.2f}_{hi:.2f}"
        if cloud_id in used_clouds:
            continue

        window_end = time + timedelta(minutes=ichimoku_tf * shift)
        segment = df.loc[time:window_end]
        if segment.empty:
            used_clouds.add(cloud_id)
            continue

        # enforce minimum cloud depth
        depth_candles = segment[(segment['close'] > lo) & (segment['close'] < hi)]
        if len(depth_candles) < min_cloud_depth_candles:
            used_clouds.add(cloud_id)
            continue

        # detect separation from span A toward B
        sep_time = None
        for i in range(1, len(segment)):
            prev = segment.iloc[i - 1]
            curr = segment.iloc[i]
            if trend == 'long' and prev.close < a and lo < curr.close < hi:
                sep_time = segment.index[i]
                break
            elif trend == 'short' and prev.close > a and lo < curr.close < hi:
                sep_time = segment.index[i]
                break

        if sep_time is None:
            used_clouds.add(cloud_id)
            continue

        # validate conditions before entry
        opp_count = 0
        entry_time = None
        for t, candle in segment.loc[sep_time:].iterrows():
            if trend == 'long' and candle.low <= hi:
                opposite = candle.close < candle.open
            elif trend == 'short' and candle.high >= lo:
                opposite = candle.close > candle.open
            else:
                continue

            if opposite:
                opp_count += 1
                if opp_count > max_opposite_candles:
                    sep_time = None
                    break

            # entry when span B is touched (not close)
            if (trend == 'long' and candle.high >= hi) or (trend == 'short' and candle.low <= lo):
                entry_time = t
                break

        if sep_time is None or entry_time is None:
            used_clouds.add(cloud_id)
            continue

        # avoid duplicate trades per cloud
        used_clouds.add(cloud_id)

        entry_price = float(df.at[entry_time, 'close'])

        # monitor exit
        future = df.loc[entry_time + timedelta(minutes=ichimoku_tf):]
        exit_time = None
        exit_price = None
        for t, candle in future.iterrows():
            price = candle.close
            if (trend == 'long' and price <= entry_price * (1 - sl_static_percent / 100)) or \
               (trend == 'short' and price >= entry_price * (1 + sl_static_percent / 100)):
                exit_time, exit_price = t, price
                break

            pnl = (price - entry_price) / entry_price * (1 if trend == 'long' else -1)
            if pnl <= -sl1_percent / 100:
                exit_time, exit_price = t, price
                break

            if (trend == 'long' and candle.low <= lo) or (trend == 'short' and candle.high >= hi):
                exit_time, exit_price = t, price
                break

        if exit_time is None:
            continue

        pnl = (exit_price - entry_price) / entry_price * (1 if trend == 'long' else -1)
        trades.append({
            'type': trend,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'pnl_usd': pnl * 100
        })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['balance'] = 100 * (1 + trades_df['pnl_usd'] / 100).cumprod()
    return trades_df


if __name__ == "__main__":
    file_path      = "data/BTCUSDT.csv"
    days           = 30
    base_tf        = 1
    ichimoku_tf    = 5
    tenkan         = 9
    kijun          = 26
    senkou_b       = 52
    displacement   = 26
    max_opp_candle = 2
    min_candle_break = 2
    sl1_percent    = 0.1
    sl2_fixed      = 0.5

    df = read_csv_recent_days(file_path, days)
    trades_df = run_ichimoku_strategy(
        df,
        ichimoku_tf,
        displacement,
        max_opp_candle,
        min_candle_break,
        sl1_percent,
        sl2_fixed
    )
    trades_df.to_csv("trades.csv", index=False)
    print(f"✅ Generated {len(trades_df)} trades → trades.csv")
