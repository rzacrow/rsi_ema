def ichimoku(df, conversion_line=9, base_line=26, lagging_span=26, leading_b_period=52):

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

def ichi_signals(df, conversion_line=9, base_line=26, lagging_span=26, leading_b_period=52, max_reverse_candles=1):
    ichi = ichimoku(df, conversion_line, base_line, lagging_span, leading_b_period)
    
    upper = ichi['upper_kumo']
    lower = ichi['lower_kumo']
    cloud = ichi['cloud']
    
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    closes = df['close'].tolist()
    opens = df['open'].tolist()
    
    changes = []
    for i in range(1, len(cloud)):
        if (cloud[i] == 'red' and cloud[i-1] == 'green') or (cloud[i] == 'green' and cloud[i-1] == 'red'):
            changes.append(i + lagging_span)
    
    signals = []
    for i in range(30, len(highs)):
        if highs[i] >= upper[i-lagging_span] and highs[i-1] < upper[i-lagging_span-1]:
            verify = True
            entry_index = None
            
            for j in range(1, min(30, i)):
                if (i-j >= 0 and 
                    highs[i-j] >= lower[i-j-lagging_span] and 
                    highs[i-j-1] < lower[i-j-1-lagging_span]):
                    
                    entry_index = i - j
                    
                    revers_candles = 0
                    for k in range(entry_index, i):
                        if opens[k] > closes[k]:
                            revers_candles += 1
                        if revers_candles > max_reverse_candles:
                            verify = False
                            break
                    
                    break

            if verify and entry_index is not None:
                signals.append({
                    'type': 'buy',
                    'entry_index': entry_index,
                    'exit_index': i
                })

        elif lows[i] <= lower[i-lagging_span] and lows[i-1] > lower[i-lagging_span-1]:
            verify = True
            entry_index = None
            
            for j in range(1, min(30, i)):
                if (i-j >= 0 and 
                    lows[i-j] <= upper[i-j-lagging_span] and 
                    lows[i-j-1] > upper[i-j-1-lagging_span]):
                    
                    entry_index = i - j
                    
                    revers_candles = 0
                    for k in range(entry_index, i):
                        if opens[k] < closes[k]:
                            revers_candles += 1
                        if revers_candles > max_reverse_candles:
                            verify = False
                            break
                    
                    break

            if verify and entry_index is not None:
                signals.append({
                    'type': 'sell',
                    'entry_index': entry_index,
                    'exit_index': i
                })
    
    verified = []
    for change_idx in changes:
        for j in range(len(signals)):
            signal = signals[j]
            
            if (signal['entry_index'] >= change_idx and 
                (j == 0 or signals[j-1]['entry_index'] < change_idx)):
                
                if signal not in verified:
                    verified.append(signal)
    
    verified.sort(key=lambda x: x['entry_index'])
    
    return verified
