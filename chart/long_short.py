# plot_signals_simple.py
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============ تنظیمات ورودی ============
symbol_input = input("Enter symbol (e.g. XRPUSDT): ").strip().upper()

# مسیر فایل‌ها (فرض: ساختار پروژه قبلی)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_file = os.path.join(base_dir, "data", f"{symbol_input}.csv")
trades_file = os.path.join(base_dir, f"signals_{symbol_input}.csv")

# بررسی وجود فایل‌ها
if not os.path.isfile(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")
if not os.path.isfile(trades_file):
    raise FileNotFoundError(f"Trades file not found: {trades_file}")

print(f"📂 Loading data from: {data_file}")
print(f"📂 Loading trades from: {trades_file}")

# ============ خواندن داده‌ها ============
print("📥 Reading data CSV...")
df = pd.read_csv(data_file, parse_dates=["datetime"]).set_index("datetime")
required_price_cols = ['open', 'high', 'low', 'close']
if not all(col in df.columns for col in required_price_cols):
    raise ValueError(f"Data file must contain columns: {required_price_cols}")
print(f"📊 Loaded price data: {len(df)} rows, from {df.index.min()} to {df.index.max()}")

print("📥 Reading trades CSV...")
trades = pd.read_csv(trades_file, parse_dates=["entry_time", "exit_time"])
# expected columns in your sample: side,signal,entry_time,entry_index,entry_price,exit_time,exit_index,exit_price,exit_reason,pnl,position_notional,balance_after
expected = ['side', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'exit_reason', 'pnl', 'position_notional', 'balance_after']
if not all(col in trades.columns for col in expected):
    missing = [c for c in expected if c not in trades.columns]
    raise ValueError(f"Trades file missing required columns: {missing}")
print(f"📊 Loaded trades: {len(trades)} rows, entries from {trades['entry_time'].min()} to {trades['entry_time'].max()}")

# استانداردسازی نام‌ها (دوباره نام‌گذاری در صورت نیاز)
trades = trades.rename(columns={"side": "type", "pnl": "pnl_usd", "balance_after": "balance"})

# محاسبه درصد PnL اگر موقعیت ناتشال وجود داشته باشد
trades['pnl_pct'] = np.where(
    (trades['position_notional'] != 0) & (~trades['position_notional'].isna()),
    trades['pnl_usd'] / trades['position_notional'] * 100.0,
    np.nan
)

# مرتب‌سازی و پاک‌سازی
trades = trades.sort_values("entry_time").reset_index(drop=True)

# ============ محاسبه EMA ساده ============
print("⚙️ Computing EMA(12)...")
df['ema'] = df['close'].ewm(span=12, adjust=False).mean()

# ============ محاسبه متریک‌ها ============
print("⚙️ Computing performance metrics...")
initial_balance = trades['position_notional'].iloc[0] if len(trades) > 0 else 100.0
final_balance = trades['balance'].iloc[-1] if len(trades) > 0 else initial_balance
total_trades = len(trades)
if total_trades > 0:
    winning_trades = trades[trades['pnl_usd'] > 0].shape[0]
    losing_trades = trades[trades['pnl_usd'] <= 0].shape[0]
    win_rate = winning_trades / total_trades * 100
    avg_gain = trades[trades['pnl_pct'] > 0]['pnl_pct'].mean() if not trades[trades['pnl_pct'] > 0].empty else 0
    avg_loss = trades[trades['pnl_pct'] <= 0]['pnl_pct'].mean() if not trades[trades['pnl_pct'] <= 0].empty else 0
    gross_profit = trades[trades['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = -trades[trades['pnl_usd'] <= 0]['pnl_usd'].sum()
    net_profit = gross_profit - gross_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    roi = (final_balance - initial_balance) / initial_balance * 100 if initial_balance != 0 else 0
    balance_series = pd.Series([initial_balance] + trades['balance'].tolist())
    max_drawdown = ((balance_series.cummax() - balance_series) / balance_series.cummax()).max() * 100
    max_drawdown_usd = (balance_series.cummax() - balance_series).max()
    buy_hold_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
else:
    winning_trades = losing_trades = win_rate = avg_gain = avg_loss = 0
    gross_profit = gross_loss = net_profit = max_drawdown = max_drawdown_usd = buy_hold_return = 0
    profit_factor = float('inf')
    roi = 0

print("✅ Performance metrics calculated.")

# ============ رسم نمودار (candles + EMA + entries/exits) ============
print("📊 Starting to plot chart...")
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

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

# سیگنال‌های ورود (long / short)
print("📈 Adding entry signals...")
for side, color, marker in [('long', 'green', 'triangle-up'), ('short', 'red', 'triangle-down')]:
    df_tr = trades[trades['type'].str.lower() == side]
    if not df_tr.empty:
        hover = []
        for _, r in df_tr.iterrows():
            pct = r['pnl_pct'] if not np.isnan(r['pnl_pct']) else None
            hover.append(f"Type: {r.get('signal', '')}<br>PnL: {r['pnl_usd']:.6f} USD<br>PnL%: {pct:.4f}%<br>Exit: {r.get('exit_reason','')}")
        fig.add_trace(go.Scatter(
            x=df_tr['entry_time'],
            y=df_tr['entry_price'],
            mode='markers',
            name=f"{side.title()} Entry",
            marker=dict(color=color, symbol=marker, size=10, line=dict(width=1, color='black')),
            text=hover,
            hoverinfo='text'
        ), row=1, col=1)

# سیگنال‌های خروج — استفاده از exit_reason برای تعیین نماد/رنگ
print("📈 Adding exit signals...")
# map common reasons to symbols/colors (expandable)
exit_map = {
    'tp_rsi': ('circle', 'green', 'TP (RSI)'),
    'bear_div': ('x', 'red', 'Bear Div'),
    'hidden_bear': ('x', 'red', 'Hidden Bear'),
    'trailing_stop': ('x', 'orange', 'Trailing Stop'),
    'eod_close': ('diamond', 'blue', 'EOD Close'),
    # generic fallback
    'default': ('square', 'blue', 'Exit')
}

# iterate and plot exits grouped by reason
for reason_key in trades['exit_reason'].unique():
    if pd.isna(reason_key):
        continue
    key = reason_key if reason_key in exit_map else 'default'
    symbol, color, label = exit_map.get(key, exit_map['default'])
    df_exit = trades[trades['exit_reason'] == reason_key]
    hover = []
    for _, r in df_exit.iterrows():
        pct = r['pnl_pct'] if not np.isnan(r['pnl_pct']) else None
        hover.append(f"Type: {r.get('signal','')}<br>PnL: {r['pnl_usd']:.6f} USD<br>PnL%: {pct:.4f}%<br>Reason: {r.get('exit_reason','')}")
    fig.add_trace(go.Scatter(
        x=df_exit['exit_time'],
        y=df_exit['exit_price'],
        mode='markers',
        name=f"Exit: {label}",
        marker=dict(color=color, symbol=symbol, size=9, line=dict(width=1, color='black')),
        text=hover,
        hoverinfo='text'
    ), row=1, col=1)

# خطوط اتصال ورود به خروج
print("📈 Adding trade connections...")
for _, trade in trades.iterrows():
    if pd.isna(trade['entry_time']) or pd.isna(trade['exit_time']):
        continue
    line_color = 'green' if trade['pnl_usd'] > 0 else 'red'
    fig.add_trace(go.Scatter(
        x=[trade['entry_time'], trade['exit_time']],
        y=[trade['entry_price'], trade['exit_price']],
        mode='lines',
        line=dict(color=line_color, width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

# لِی‌آوت
print("📊 Updating layout...")
fig.update_layout(
    title=f"{symbol_input} Strategy Signals",
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified'
)
fig.update_xaxes(rangeslider_visible=False)
fig.update_yaxes(title_text="Price")

# HTML metrics
metrics_html = f"""
<div style="font-family:Arial; padding:20px; background-color:#f8f9fa; border-radius:10px; margin-top:20px;">
  <h2 style="color:#2c3e50;">Performance Metrics</h2>
  <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:15px;">
    <div style="background-color:#ffffff; padding:15px; border-radius:8px;">
      <h3>Trades Summary</h3>
      <p><b>Total Trades:</b> {total_trades}</p>
      <p><b>Winning Trades:</b> {winning_trades} ({win_rate:.2f}%)</p>
      <p><b>Losing Trades:</b> {losing_trades}</p>
      <p><b>Avg Gain:</b> {avg_gain:.2f}%</p>
      <p><b>Avg Loss:</b> {avg_loss:.2f}%</p>
    </div>
    <div style="background-color:#ffffff; padding:15px; border-radius:8px;">
      <h3>Profit & Loss</h3>
      <p><b>Gross Profit:</b> ${gross_profit:.6f}</p>
      <p><b>Gross Loss:</b> ${gross_loss:.6f}</p>
      <p><b>Net Profit:</b> ${net_profit:.6f}</p>
      <p><b>Profit Factor:</b> {profit_factor:.3f}</p>
      <p><b>ROI:</b> {roi:.2f}%</p>
    </div>
    <div style="background-color:#ffffff; padding:15px; border-radius:8px;">
      <h3>Risk Metrics</h3>
      <p><b>Initial Balance:</b> ${initial_balance:.2f}</p>
      <p><b>Final Balance:</b> ${final_balance:.2f}</p>
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
