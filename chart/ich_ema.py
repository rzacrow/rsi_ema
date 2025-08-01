import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from ta.trend import EMAIndicator

# === تنظیمات ورودی کاربر ===
symbol_input = input("Enter symbol (e.g. XRPUSDT): ").strip().upper()

# مسیر فایل‌ها
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_file = os.path.join(base_dir, "data", f"{symbol_input}.csv")
trades_file = os.path.join(base_dir, f"trades_{symbol_input}.csv")

# بررسی وجود فایل‌ها
if not os.path.isfile(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")
if not os.path.isfile(trades_file):
    raise FileNotFoundError(f"Trades file not found: {trades_file}")

print(f"📂 Loading data from: {data_file}")
print(f"📂 Loading trades from: {trades_file}")

# === بارگیری داده‌ها ===
print("📥 Reading data CSV...")
df = pd.read_csv(data_file, parse_dates=["datetime"]).set_index("datetime")
if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
    raise ValueError("Required columns missing in data file")
print(f"📊 Loaded data: {len(df)} rows, from {df.index.min()} to {df.index.max()}")

print("📥 Reading trades CSV...")
trades = pd.read_csv(trades_file, parse_dates=["entry_time", "exit_time"])
required_cols = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'type', 'pnl_pct', 'pnl_usd', 'balance', 'exit_reason']
if not all(col in trades.columns for col in required_cols):
    raise ValueError(f"Required columns missing in trades file: {required_cols}")
print(f"📊 Loaded trades: {len(trades)} rows, entries from {trades['entry_time'].min()} to {trades['entry_time'].max()}")

# فیلتر کردن تریدهای نامعتبر
trades = trades[trades['type'].isin(['long', 'short'])].dropna(subset=required_cols)
print(f"📊 Filtered trades: {len(trades)} rows")

# === پارامترهای Ichimoku ===
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
DISPLACEMENT = 26

# === پارامترهای Cloud Switch Detection ===
CLOUD_SWITCH_LEAD = 4  # 4 candles before 26
CLOUD_SWITCH_LAG = 4   # 4 candles after 26
CLOUD_SWITCH_CHECK_START = DISPLACEMENT - CLOUD_SWITCH_LEAD  # 22 candles
CLOUD_SWITCH_CHECK_END = DISPLACEMENT + CLOUD_SWITCH_LAG     # 30 candles

# === توابع کمکی ===
def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """محاسبه اندیکاتورهای Ichimoku"""
    # Tenkan-sen (Conversion Line)
    high_9 = df['high'].rolling(window=TENKAN_PERIOD).max()
    low_9 = df['low'].rolling(window=TENKAN_PERIOD).min()
    df['tenkan'] = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line)
    high_26 = df['high'].rolling(window=KIJUN_PERIOD).max()
    low_26 = df['low'].rolling(window=KIJUN_PERIOD).min()
    df['kijun'] = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(DISPLACEMENT)
    
    # Senkou Span B (Leading Span B)
    high_52 = df['high'].rolling(window=SENKOU_SPAN_B_PERIOD).max()
    low_52 = df['low'].rolling(window=SENKOU_SPAN_B_PERIOD).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(DISPLACEMENT)
    
    # Chikou Span (Lagging Span)
    df['chikou_span'] = df['close'].shift(-DISPLACEMENT)
    
    return df

def detect_tenkan_kijun_cross(df: pd.DataFrame) -> list:
    """تشخیص تقاطع Tenkan و Kijun"""
    crosses = []
    
    for i in range(1, len(df)):
        prev_tenkan = df.iloc[i-1]['tenkan']
        prev_kijun = df.iloc[i-1]['kijun']
        curr_tenkan = df.iloc[i]['tenkan']
        curr_kijun = df.iloc[i]['kijun']
        
        # Bullish cross (Tenkan crosses above Kijun)
        if prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun:
            crosses.append({
                'index': i,
                'timestamp': df.index[i],
                'type': 'bullish',
                'price': df.iloc[i]['close']
            })
        
        # Bearish cross (Tenkan crosses below Kijun)
        elif prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun:
            crosses.append({
                'index': i,
                'timestamp': df.index[i],
                'type': 'bearish',
                'price': df.iloc[i]['close']
            })
    
    return crosses

def detect_cloud_switch(df: pd.DataFrame, start_idx: int, end_idx: int) -> dict:
    """تشخیص تغییر وضعیت ابر در بازه مشخص"""
    if start_idx >= len(df) or end_idx >= len(df):
        return None
    
    # Check if cloud switches from bullish to bearish or vice versa
    for i in range(start_idx, min(end_idx + 1, len(df))):
        if i < DISPLACEMENT:
            continue
            
        # Get current and previous cloud status
        curr_span_a = df.iloc[i]['senkou_span_a']
        curr_span_b = df.iloc[i]['senkou_span_b']
        
        if i > 0:
            prev_span_a = df.iloc[i-1]['senkou_span_a']
            prev_span_b = df.iloc[i-1]['senkou_span_b']
        else:
            continue
        
        # Check for cloud switch
        curr_bullish = curr_span_a > curr_span_b
        prev_bullish = prev_span_a > prev_span_b
        
        if curr_bullish != prev_bullish:
            return {
                'index': i,
                'timestamp': df.index[i],
                'type': 'bullish' if curr_bullish else 'bearish',
                'span_a': curr_span_a,
                'span_b': curr_span_b
            }
    
    return None

# === محاسبه اندیکاتورها ===
print("⚙️ Computing Ichimoku and EMA indicators...")

# محاسبه Ichimoku
df = compute_ichimoku(df)

# محاسبه Ichimoku 3 دقیقه‌ای (مطابق استراتژی اصلی)
df_3m = df.resample('3T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
})

# محاسبه Ichimoku برای داده‌های 3 دقیقه‌ای
df_3m = compute_ichimoku(df_3m)

# مپ کردن Ichimoku 3 دقیقه‌ای به داده‌های 1 دقیقه‌ای
for col in ['tenkan', 'kijun', 'senkou_span_a', 'senkou_span_b', 'chikou_span']:
    df[f'{col}_3m'] = df_3m[col].reindex(df.index, method='ffill')

# EMA
df['ema'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
print(f"✅ Ichimoku and EMA calculated. Sample Tenkan: {df['tenkan_3m'].iloc[-1]:.2f}, Kijun: {df['kijun_3m'].iloc[-1]:.2f}")

# === تشخیص باکس‌های Ichimoku (مطابق استراتژی اصلی) ===
print("⚙️ Detecting Ichimoku boxes...")

# تشخیص تقاطع‌های Tenkan و Kijun
crosses = detect_tenkan_kijun_cross(df)
print(f"Tenkan-Kijun crosses detected: {len(crosses)}")

# تشخیص باکس‌ها بر اساس Ichimoku
ichimoku_boxes = []
valid_crosses = []

for cross in crosses:
    cross_idx = cross['index']
    cross_type = cross['type']
    
    # Check for cloud switch in the specified interval
    start_check_idx = cross_idx + CLOUD_SWITCH_CHECK_START
    end_check_idx = cross_idx + CLOUD_SWITCH_CHECK_END
    
    cloud_switch = detect_cloud_switch(df, start_check_idx, end_check_idx)
    
    if cloud_switch is None:
        print(f"❌ No cloud switch detected for {cross_type} cross at {cross['timestamp']}")
        continue
    
    # Validate cloud switch matches cross type
    if cross_type == 'bullish' and cloud_switch['type'] == 'bearish':
        print(f"❌ Bullish cross with bearish cloud switch - invalid at {cross['timestamp']}")
        continue
    elif cross_type == 'bearish' and cloud_switch['type'] == 'bullish':
        print(f"❌ Bearish cross with bullish cloud switch - invalid at {cross['timestamp']}")
        continue
    
    # Create box
    cross_candle = df.iloc[cross_idx]
    cloud_switch_candle = df.iloc[cloud_switch['index']]
    
    if cross_type == 'bullish':
        # Long setup
        box = {
            "start": cross['timestamp'],
            "end": cloud_switch['timestamp'],
            "top": cross_candle['high'],  # High of cross candle
            "bottom": cross_candle['low'],  # Low of cross candle
            "type": "buy",
            "cross_idx": cross_idx,
            "cloud_switch_idx": cloud_switch['index']
        }
    else:
        # Short setup
        box = {
            "start": cross['timestamp'],
            "end": cloud_switch['timestamp'],
            "top": cross_candle['high'],  # High of cross candle
            "bottom": cross_candle['low'],  # Low of cross candle
            "type": "sell",
            "cross_idx": cross_idx,
            "cloud_switch_idx": cloud_switch['index']
        }
    
    ichimoku_boxes.append(box)
    valid_crosses.append(cross)
    print(f"✅ Valid {cross_type} setup detected at {cross['timestamp']} with cloud switch at {cloud_switch['timestamp']}")

print(f"✅ Detected {len(ichimoku_boxes)} Ichimoku boxes")

# === محاسبه معیارهای عملکرد ===
print("⚙️ Computing performance metrics...")
initial_balance = 100.0
balance = trades['balance'].iloc[-1] if not trades.empty else initial_balance
total_trades = len(trades)

if total_trades > 0:
    winning_trades = len(trades[trades['pnl_pct'] > 0])
    losing_trades = len(trades[trades['pnl_pct'] <= 0])
    win_rate = (winning_trades / total_trades) * 100
    avg_gain = trades[trades['pnl_pct'] > 0]['pnl_pct'].mean() * 100
    avg_loss = trades[trades['pnl_pct'] <= 0]['pnl_pct'].mean() * 100
    gross_profit = trades[trades['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(trades[trades['pnl_usd'] <= 0]['pnl_usd'].sum())
    net_profit = gross_profit - gross_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    roi = ((balance - initial_balance) / initial_balance) * 100
    balance_series = pd.Series([initial_balance] + trades['balance'].tolist())
    max_drawdown = ((balance_series.cummax() - balance_series) / balance_series.cummax()).max() * 100
    max_drawdown_usd = (balance_series.cummax() - balance_series).max()
    buy_hold_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
else:
    winning_trades = losing_trades = win_rate = avg_gain = avg_loss = 0
    gross_profit = gross_loss = net_profit = max_drawdown_usd = 0
    profit_factor = float('inf')
    roi = max_drawdown = buy_hold_return = 0

print("✅ Performance metrics calculated.")

# === رسم نمودار ===
print("📊 Starting to plot chart...")
# ایجاد نمودار تک‌صفحه‌ای بدون ساب‌پلات
fig = go.Figure()

# کندل‌ها
print("📈 Adding candlestick...")
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name=symbol_input
))

# Ichimoku Cloud (Senkou Span A & B) - Proper continuous cloud display
print("📈 Adding Ichimoku cloud...")

# Create cloud segments for proper display
def create_cloud_segments(df):
    """Create proper cloud segments without overlapping"""
    segments = []
    current_segment = {'start': 0, 'type': None}
    
    for i in range(len(df)):
        if i < DISPLACEMENT:  # Skip first 26 periods where cloud doesn't exist
            continue
            
        is_bullish = df['senkou_span_a_3m'].iloc[i] > df['senkou_span_b_3m'].iloc[i]
        
        if current_segment['type'] is None:
            current_segment['type'] = 'bullish' if is_bullish else 'bearish'
        elif current_segment['type'] != ('bullish' if is_bullish else 'bearish'):
            # Cloud type changed, save current segment and start new one
            current_segment['end'] = i - 1
            segments.append(current_segment)
            current_segment = {'start': i, 'type': 'bullish' if is_bullish else 'bearish'}
    
    # Add final segment
    if current_segment['type'] is not None:
        current_segment['end'] = len(df) - 1
        segments.append(current_segment)
    
    return segments

# Create cloud segments
cloud_segments = create_cloud_segments(df)

# Add cloud segments
for segment in cloud_segments:
    start_idx = segment['start']
    end_idx = segment['end']
    cloud_type = segment['type']
    
    if start_idx >= len(df) or end_idx >= len(df):
        continue
        
    # Add Span A for this segment
    fig.add_trace(go.Scatter(
        x=df.index[start_idx:end_idx+1], 
        y=df['senkou_span_a_3m'].iloc[start_idx:end_idx+1],
        mode='lines',
        name=f'Senkou Span A - {cloud_type.title()}',
        line=dict(color='green', width=1),
        fill=None,
        showlegend=False
    ))
    
    # Add Span B with fill for this segment
    fillcolor = 'rgba(0,255,0,0.2)' if cloud_type == 'bullish' else 'rgba(255,0,0,0.2)'
    fig.add_trace(go.Scatter(
        x=df.index[start_idx:end_idx+1], 
        y=df['senkou_span_b_3m'].iloc[start_idx:end_idx+1],
        mode='lines',
        name=f'Senkou Span B - {cloud_type.title()}',
        line=dict(color='red', width=1),
        fill='tonexty',
        fillcolor=fillcolor,
        showlegend=False
    ))

# Add main Span A and B lines for reference
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['senkou_span_a_3m'],
    mode='lines',
    name='Senkou Span A (3m)',
    line=dict(color='green', width=1),
    fill=None
))

fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['senkou_span_b_3m'],
    mode='lines',
    name='Senkou Span B (3m)',
    line=dict(color='red', width=1),
    fill=None
))

# Add legend entries for cloud types
fig.add_trace(go.Scatter(
    x=[], y=[],
    mode='lines',
    name='Bullish Cloud',
    line=dict(color='green', width=1),
    fill='tonexty',
    fillcolor='rgba(0,255,0,0.2)'
))

fig.add_trace(go.Scatter(
    x=[], y=[],
    mode='lines',
    name='Bearish Cloud',
    line=dict(color='red', width=1),
    fill='tonexty',
    fillcolor='rgba(255,0,0,0.2)'
))

# Tenkan and Kijun lines
print("📈 Adding Tenkan and Kijun...")
fig.add_trace(go.Scatter(
    x=df.index, y=df['tenkan_3m'],
    mode='lines',
    name='Tenkan (3m)',
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['kijun_3m'],
    mode='lines',
    name='Kijun (3m)',
    line=dict(color='orange', width=2)
))

# EMA
print("📈 Adding EMA...")
fig.add_trace(go.Scatter(
    x=df.index, y=df['ema'], 
    mode='lines', 
    name='EMA (12)',
    line=dict(color='purple', width=2)
))

# باکس‌های Ichimoku
print("📈 Adding Ichimoku boxes...")
for box in ichimoku_boxes:
    fig.add_shape(
        type="rect",
        x0=box['start'],
        y0=box['bottom'],
        x1=box['end'],
        y1=box['top'],
        fillcolor="rgba(0,255,0,0.2)" if box['type'] == 'buy' else "rgba(255,0,0,0.2)",
        line=dict(color="green" if box['type'] == 'buy' else "red", width=1),
        name=f"{box['type']} box"
    )

# سیگنال‌های ورود
print("📈 Adding entry signals...")
for trade_type, color, symbol in [('long', 'green', 'triangle-up'), ('short', 'red', 'triangle-down')]:
    df_tr = trades[trades['type'] == trade_type]
    if not df_tr.empty:
        fig.add_trace(go.Scatter(
            x=df_tr['entry_time'], 
            y=df_tr['entry_price'],
            mode='markers', 
            name=f"{trade_type.title()} Entry",
            marker=dict(color=color, symbol=symbol, size=10, line=dict(width=1, color='black')),
            customdata=np.stack((
                df_tr['pnl_pct'] * 100, 
                df_tr['exit_reason']
            ), axis=-1),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Entry Price: %{y:.6f}<br>' +
                'PnL: %{customdata[0]:.2f}%<br>' +
                'Exit Reason: %{customdata[1]}<extra></extra>'
            ),
            text=[f"{trade_type.title()} Entry" for _ in range(len(df_tr))]
        ))

# سیگنال‌های خروج با نشانگرهای مختلف
print("📈 Adding exit signals...")
exit_reasons = {
    'TP': ('circle', 'green', 'Take Profit'),
    'SL': ('x', 'red', 'Stop Loss'),
    'EMA_SL': ('x', 'red', 'EMA SL'),
    'Risk-Free': ('square', 'yellow', 'Risk-Free'),
    'Box': ('square', 'blue', 'Box Exit')
}

for reason, (symbol, color, name) in exit_reasons.items():
    df_exit = trades[trades['exit_reason'] == reason]
    if not df_exit.empty:
        fig.add_trace(go.Scatter(
            x=df_exit['exit_time'], 
            y=df_exit['exit_price'],
            mode='markers', 
            name=name,
            marker=dict(
                color=color, 
                symbol=symbol, 
                size=10, 
                line=dict(width=1, color='black')
            ),
            customdata=np.stack((
                df_exit['pnl_pct'] * 100, 
                df_exit['type']
            ), axis=-1),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Exit Price: %{y:.6f}<br>' +
                'PnL: %{customdata[0]:.2f}%<br>' +
                'Type: %{customdata[1]}<extra></extra>'
            ),
            text=[f"{name}" for _ in range(len(df_exit))]
        ))

# خطوط اتصال ورود به خروج
print("📈 Adding trade connections...")
for _, trade in trades.iterrows():
    line_color = 'green' if trade['pnl_pct'] > 0 else 'red'
    fig.add_trace(go.Scatter(
        x=[trade['entry_time'], trade['exit_time']],
        y=[trade['entry_price'], trade['exit_price']],
        mode='lines',
        line=dict(color=line_color, width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

print("📊 Updating layout...")
fig.update_layout(
    title=f"{symbol_input} Ichimoku-EMA Strategy Analysis",
    height=900,  # Increased height for better visibility
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=False))
)

# === ایجاد بخش HTML برای معیارها ===
metrics_html = f"""
<div style="font-family:Arial; padding:20px; background-color:#f8f9fa; border-radius:10px; margin-top:20px;">
  <h2 style="color:#2c3e50;">Performance Metrics</h2>
  <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:15px;">
    <div style="background-color:#ffffff; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
      <h3 style="color:#3498db;">Trades Summary</h3>
      <p><b>Total Trades:</b> {total_trades}</p>
      <p><b>Winning Trades:</b> {winning_trades} ({win_rate:.2f}%)</p>
      <p><b>Losing Trades:</b> {losing_trades}</p>
      <p><b>Avg Gain:</b> {avg_gain:.2f}%</p>
      <p><b>Avg Loss:</b> {avg_loss:.2f}%</p>
    </div>
    
    <div style="background-color:#ffffff; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
      <h3 style="color:#3498db;">Profit & Loss</h3>
      <p><b>Gross Profit:</b> ${gross_profit:.2f}</p>
      <p><b>Gross Loss:</b> ${gross_loss:.2f}</p>
      <p><b>Net Profit:</b> <span style="color:{'green' if net_profit >= 0 else 'red'};">${net_profit:.2f}</span></p>
      <p><b>Profit Factor:</b> {profit_factor:.2f}</p>
      <p><b>ROI:</b> <span style="color:{'green' if roi >= 0 else 'red'};">{roi:.2f}%</span></p>
    </div>
    
    <div style="background-color:#ffffff; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
      <h3 style="color:#3498db;">Risk Metrics</h3>
      <p><b>Initial Balance:</b> ${initial_balance:.2f}</p>
      <p><b>Final Balance:</b> ${balance:.2f}</p>
      <p><b>Max Drawdown:</b> ${max_drawdown_usd:.2f} ({max_drawdown:.2f}%)</p>
      <p><b>Buy & Hold Return:</b> {buy_hold_return:.2f}%</p>
    </div>
  </div>
  
  <div style="background-color:#ffffff; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin-top:15px;">
    <h3 style="color:#3498db;">Ichimoku Strategy Settings</h3>
    <p><b>Tenkan Period:</b> {TENKAN_PERIOD}</p>
    <p><b>Kijun Period:</b> {KIJUN_PERIOD}</p>
    <p><b>Senkou Span B Period:</b> {SENKOU_SPAN_B_PERIOD}</p>
    <p><b>Cloud Switch Lead:</b> {CLOUD_SWITCH_LEAD} candles</p>
    <p><b>Cloud Switch Lag:</b> {CLOUD_SWITCH_LAG} candles</p>
    <p><b>Valid Ichimoku Boxes:</b> {len(ichimoku_boxes)}</p>
  </div>
</div>
"""

# ذخیره چارت و معیارها
print("💾 Saving chart to HTML...")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_html = os.path.join(output_dir, f"ichimoku_strategy_analysis_{symbol_input}.html")
html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
html_content = html_content.replace('</body>', metrics_html + '</body>')
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"✅ Chart saved to {output_html}")