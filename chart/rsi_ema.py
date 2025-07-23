import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from ta.momentum import RSIIndicator
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

# === محاسبه اندیکاتورها ===
print("⚙️ Computing RSI and EMA indicators...")

# محاسبه RSI 3 دقیقه‌ای (مطابق استراتژی اصلی)
df_3m = df.resample('3T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
})
df_3m['rsi_3m'] = RSIIndicator(close=df_3m['close'], window=14).rsi()
df['rsi_3m'] = df_3m['rsi_3m'].reindex(df.index, method='ffill')

# EMA
df['ema'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
print(f"✅ RSI and EMA calculated. Sample RSI: {df['rsi_3m'].iloc[-1]:.2f}, EMA: {df['ema'].iloc[-1]:.2f}")

# === تشخیص باکس‌های RSI (مطابق استراتژی اصلی) ===
print("⚙️ Detecting RSI boxes...")
def detect_rsi_boxes(df, rsi_oversold=30, rsi_overbought=70):
    boxes = []
    state = "neutral"
    start_time = None
    highs, lows = [], []
    
    for timestamp, row in df.iterrows():
        rsi_val = row["rsi_3m"]  # استفاده از RSI 3 دقیقه‌ای
        
        if state == "neutral":
            if rsi_val < rsi_oversold:
                state = "oversold"
                start_time = timestamp
                highs = [row["high"]]
                lows = [row["low"]]
            elif rsi_val > rsi_overbought:
                state = "overbought"
                start_time = timestamp
                highs = [row["high"]]
                lows = [row["low"]]
        
        elif state == "oversold":
            highs.append(row["high"])
            lows.append(row["low"])
            
            if rsi_val > rsi_oversold:
                boxes.append({
                    "start": start_time,
                    "end": timestamp,
                    "top": max(highs),
                    "bottom": min(lows),
                    "type": "buy"
                })
                state = "neutral"
        
        elif state == "overbought":
            highs.append(row["high"])
            lows.append(row["low"])
            
            if rsi_val < rsi_overbought:
                boxes.append({
                    "start": start_time,
                    "end": timestamp,
                    "top": max(highs),
                    "bottom": min(lows),
                    "type": "sell"
                })
                state = "neutral"
    
    return boxes

rsi_boxes = detect_rsi_boxes(df)
print(f"✅ Detected {len(rsi_boxes)} RSI boxes")

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
# ایجاد ساب‌پلات با دو ردیف
fig = make_subplots(
    rows=2, 
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3]
)

# کندل‌ها
print("📈 Adding candlestick...")
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name=symbol_input
), row=1, col=1)

# EMA
print("📈 Adding EMA...")
fig.add_trace(go.Scatter(
    x=df.index, y=df['ema'], 
    mode='lines', 
    name='EMA (12)',
    line=dict(color='purple', width=2)
), row=1, col=1)

# باکس‌های RSI
print("📈 Adding RSI boxes...")
for box in rsi_boxes:
    fig.add_shape(
        type="rect",
        x0=box['start'],
        y0=box['bottom'],
        x1=box['end'],
        y1=box['top'],
        fillcolor="rgba(0,255,0,0.2)" if box['type'] == 'buy' else "rgba(255,0,0,0.2)",
        line=dict(color="green" if box['type'] == 'buy' else "red", width=1),
        name=f"{box['type']} box",
        row=1, col=1
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
        ), row=1, col=1)

# سیگنال‌های خروج با نشانگرهای مختلف
print("📈 Adding exit signals...")
exit_reasons = {
    'TP': ('circle', 'green', 'Take Profit'),
    'SL': ('x', 'red', 'Stop Loss'),
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
        ), row=1, col=1)

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
    ), row=1, col=1)

# افزودن RSI به عنوان زیرنمودار
print("📈 Adding RSI subplot...")
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['rsi_3m'],
    mode='lines',
    name='RSI (14)',
    line=dict(color='blue', width=1.5)
), row=2, col=1)

# خطوط افقی RSI
fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=2, col=1)
fig.add_annotation(
    xref='x domain', yref='y2',
    x=0.01, y=70,
    text="Overbought (70)",
    showarrow=False,
    font=dict(size=10),
    row=2, col=1
)
fig.add_annotation(
    xref='x domain', yref='y2',
    x=0.01, y=30,
    text="Oversold (30)",
    showarrow=False,
    font=dict(size=10),
    row=2, col=1
)

print("📊 Updating layout...")
fig.update_layout(
    title=f"{symbol_input} RSI-EMA Strategy Analysis",
    height=900,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified'
)

# تنظیمات محورها
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

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
</div>
"""

# ذخیره چارت و معیارها
print("💾 Saving chart to HTML...")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_html = os.path.join(output_dir, f"strategy_analysis_{symbol_input}.html")
html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
html_content = html_content.replace('</body>', metrics_html + '</body>')
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"✅ Chart saved to {output_html}")