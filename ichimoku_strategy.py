import pandas as pd
from datetime import timedelta

def read_csv_recent_days(file_path, days):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    last_date = df['datetime'].max()
    start_date = last_date - timedelta(days=days)
    return df[df['datetime'] >= start_date]

def extract_clouds(ich_df):
    clouds = []
    prev_trend = None
    curr_start = None
    for t, row in ich_df.iterrows():
        a, b = row['senkou_a'], row['senkou_b']
        trend = 'bull' if a < b else 'bear'
        if prev_trend is None:
            prev_trend = trend
            curr_start = t
        elif trend != prev_trend:
            low, high = sorted([ich_df.loc[curr_start, 'senkou_a'], ich_df.loc[curr_start, 'senkou_b']])
            clouds.append((curr_start, t, low, high))
            curr_start = t
            prev_trend = trend
    low, high = sorted([ich_df.loc[curr_start, 'senkou_a'], ich_df.loc[curr_start, 'senkou_b']])
    clouds.append((curr_start, ich_df.index[-1], low, high))
    return clouds

def run_ichimoku_strategy(df, ichimoku_tf=5, shift=26, max_opp=2, sl1=0.1, sl_static=0.5):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Resample and compute Ichimoku
    ich = df.resample(f'{ichimoku_tf}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    ich['senkou_a'] = ((ich['high'].rolling(9).max() + ich['low'].rolling(9).min()) / 2 + \
                       (ich['high'].rolling(26).max() + ich['low'].rolling(26).min()) / 2) / 2
    ich['senkou_b'] = (ich['high'].rolling(52).max() + ich['low'].rolling(52).min()) / 2
    ich['senkou_a'] = ich['senkou_a'].shift(shift)
    ich['senkou_b'] = ich['senkou_b'].shift(shift)
    ich.dropna(inplace=True)

    # Extract clouds
    clouds = extract_clouds(ich)
    trades = []
    used_clouds = set()

    for start, end, low, high in clouds:
        segment = df.loc[start:end]
        if segment.empty or (start, end) in used_clouds:
            continue

        # Find entry and exit
        entry_idx = None
        direction = None
        for i in range(1, len(segment)):
            prev = segment.iloc[i - 1]
            curr = segment.iloc[i]
            # Enter from top, exit bottom (Short)
            if prev['close'] > high and curr['low'] <= low:
                entry_idx = i
                direction = 'short'
                break
            # Enter from bottom, exit top (Long)
            if prev['close'] < low and curr['high'] >= high:
                entry_idx = i
                direction = 'long'
                break
        if entry_idx is None:
            continue

        # Find last candle on initial span
        last_span_idx = None
        for j in range(entry_idx - 1, -1, -1):
            val = segment.iloc[j]
            if direction == 'short' and val['high'] >= high:
                last_span_idx = j
                break
            elif direction == 'long' and val['low'] <= low:
                last_span_idx = j
                break
        if last_span_idx is None:
            last_span_idx = 0

        # Count opposing candles to exit point
        opp_count = 0
        entry_time = segment.iloc[entry_idx].name
        entry_price = low if direction == 'short' else high
        for k in range(last_span_idx + 1, entry_idx):
            c = segment.iloc[k]
            if direction == 'short' and c['close'] > c['open']:
                opp_count += 1
            if direction == 'long' and c['close'] < c['open']:
                opp_count += 1
        if opp_count > max_opp:
            used_clouds.add((start, end))
            continue

        # Exit logic
        future = df.loc[entry_time + timedelta(minutes=ichimoku_tf):end]
        exit_time = None
        exit_price = None
        for _, fc in future.iterrows():
            price = fc['close']
            if (direction == 'long' and price <= entry_price * (1 - sl_static / 100)) or \
               (direction == 'short' and price >= entry_price * (1 + sl_static / 100)):
                exit_price = price
                exit_time = fc.name
                break
            pnl = (price - entry_price) / entry_price * (1 if direction == 'long' else -1)
            if pnl <= -sl1 / 100:
                exit_price = price
                exit_time = fc.name
                break
        if exit_time is None:
            used_clouds.add((start, end))
            continue

        pnl = (exit_price - entry_price) / entry_price * (1 if direction == 'long' else -1)
        trades.append({
            'type': direction,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'pnl_usd': pnl * 100
        })

        # Mark cloud as used
        used_clouds.add((start, end))

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['balance'] = 100 * (1 + trades_df['pnl_usd'] / 100).cumprod()
    return trades_df

if __name__ == '__main__':
    df = read_csv_recent_days('data/BTCUSDT.csv', 30)
    trades = run_ichimoku_strategy(df)
    trades.to_csv('trades.csv', index=False)
    print(f"✅ Generated {len(trades)} trades → trades.csv")