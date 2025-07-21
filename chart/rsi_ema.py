import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# === تنظیمات ورودی کاربر ===
symbol_input = input("Enter symbol (e.g. XRPUSDT): ").strip().upper()

# مسیر فایل‌ها
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_file = os.path.join(base_dir, "data", f"{symbol_input}.csv")
trades_file = os.path.join(base_dir, "trades.csv")

# بررسی وجود فایل‌ها
if not os.path.isfile(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")
if not os.path.isfile(trades_file):
    raise FileNotFoundError(f"Trades file not found: {trades_file}")

print(f"Loading data from: {data_file}")
print(f"Loading trades from: {trades_file}")

# === بارگیری داده‌ها ===
print("📥 Reading data CSV...")
df = pd.read_csv(data_file, parse_dates=["datetime"]).set_index("datetime")
if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
    raise ValueError("Required columns missing in data file")
print(f"📊 Loaded data: {len(df)} rows, from {df.index.min()} to {df.index.max()}")

print("📥 Reading trades CSV...")
trades = pd.read_csv(trades_file, parse_dates=["entry_time", "exit_time"])
required_cols = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'type']
if not all(col in trades.columns for col in required_cols):
    raise ValueError(f"Required columns missing in trades file: {required_cols}")
print(f"📊 Loaded trades: {len(trades)} rows, entries from {trades['entry_time'].min()} to {trades['entry_time'].max()}")
print("Trade types:", trades['type'].unique())
print("Any NaN in trades:", trades[required_cols].isna().any())

# فیلتر کردن تریدهای نامعتبر
trades = trades[trades['type'].isin(['long', 'short'])].dropna(subset=required_cols)
print(f"📊 Filtered trades: {len(trades)} rows")

# === محاسبه اندیکاتورها ===
print("⚙️ Computing RSI and EMA indicators...")
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
df['ema'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
print(f"✅ RSI and EMA calculated. Sample RSI: {df['rsi'].iloc[-1]:.2f}, EMA: {df['ema'].iloc[-1]:.2f}")

# === محاسبه معیارهای ترید ===
print("⚙️ Computing trade metrics...")
initial_balance = 100.0  # سرمایه اولیه
balance = initial_balance
balances = []
pnl_usds = []

# محاسبه pnl_pct و به‌روزرسانی balance برای هر ترید
for index, row in trades.iterrows():
    if row['type'] == 'long':
        pnl_pct = (row['exit_price'] - row['entry_price']) / row['entry_price']
    else:  # short
        pnl_pct = (row['entry_price'] - row['exit_price']) / row['entry_price']
    pnl_usd = balance * pnl_pct
    balance = balance * (1 + pnl_pct)
    pnl_usds.append(pnl_usd)
    balances.append(balance)

# اضافه کردن ستون‌ها به دیتافریم
trades['pnl_pct'] = trades.apply(
    lambda row: (row['exit_price'] - row['entry_price']) / row['entry_price'] if row['type'] == 'long'
    else (row['entry_price'] - row['exit_price']) / row['entry_price'], axis=1)
trades['pnl_usd'] = pnl_usds
trades['balance'] = balances
print("✅ Trade metrics calculated.")

# === محاسبه معیارهای عملکرد ===
print("⚙️ Computing performance metrics...")
total_trades = len(trades)
winning_trades = len(trades[trades['pnl_pct'] > 0])
losing_trades = len(trades[trades['pnl_pct'] <= 0])
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
avg_gain = trades[trades['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if winning_trades > 0 else 0
avg_loss = trades[trades['pnl_pct'] <= 0]['pnl_pct'].mean() * 100 if losing_trades > 0 else 0
gross_profit = trades[trades['pnl_usd'] > 0]['pnl_usd'].sum()
gross_loss = abs(trades[trades['pnl_usd'] <= 0]['pnl_usd'].sum())
net_profit = gross_profit - gross_loss
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
roi = ((balance - initial_balance) / initial_balance) * 100
balance_series = pd.Series([initial_balance] + balances)
max_drawdown = ((balance_series.cummax() - balance_series) / balance_series.cummax()).max() * 100
max_drawdown_usd = (balance_series.cummax() - balance_series).max()
buy_hold_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
print("✅ Performance metrics calculated.")

# === رسم نمودار ===
print("📊 Starting to plot chart...")
fig = go.Figure()

# کندل‌ها
print("📈 Adding candlestick...")
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name=symbol_input
))

# EMA
print("📈 Adding EMA...")
fig.add_trace(go.Scatter(
    x=df.index, y=df['ema'], mode='lines', name='EMA'
))

# سیگنال‌های ورود/خروج
print("📈 Adding trade signals...")
type_map = {
    'long': {'marker': {'color': 'green', 'symbol': 'triangle-up', 'size': 10}},
    'short': {'marker': {'color': 'red', 'symbol': 'triangle-down', 'size': 10}}
}
for ttype in ['long', 'short']:
    df_tr = trades[trades['type'] == ttype]
    print(f"📈 Adding {ttype} entries: {len(df_tr)} points")
    fig.add_trace(go.Scatter(
        x=df_tr['entry_time'], y=df_tr['entry_price'],
        mode='markers', name=f"{ttype.title()} Entry",
        marker=type_map[ttype]['marker']
    ))
print("📈 Adding exit points...")
fig.add_trace(go.Scatter(
    x=trades['exit_time'], y=trades['exit_price'],
    mode='markers', name='Exit', marker={'color': 'blue', 'symbol': 'x', 'size': 8}
))

print("📊 Updating layout...")
fig.update_layout(
    title=f"{symbol_input} RSI-EMA Strategy Chart with Performance Metrics",
    xaxis_rangeslider_visible=False,
    height=700
)

# === ایجاد بخش HTML برای معیارها ===
metrics_html = f"""
<div style="font-family:Arial; padding:10px; max-width:800px;">
  <h3>Performance Metrics</h3>
  <p><b>Total Trades:</b> {total_trades}</p>
  <p><b>Winning Trades:</b> {winning_trades}</p>
  <p><b>Losing Trades:</b> {losing_trades}</p>
  <p><b>Win Rate:</b> {win_rate:.2f}%</p>
  <p><b>Avg Gain:</b> {avg_gain:.2f}%</p>
  <p><b>Avg Loss:</b> {avg_loss:.2f}%</p>
  <p><b>Gross Profit:</b> ${gross_profit:.2f}</p>
  <p><b>Gross Loss:</b> ${gross_loss:.2f}</p>
  <p><b>Net Profit:</b> ${net_profit:.2f}</p>
  <p><b>Profit Factor:</b> {profit_factor:.2f}</p>
  <p><b>ROI:</b> {roi:.2f}%</p>
  <p><b>Max Drawdown:</b> ${max_drawdown_usd:.2f} ({max_drawdown:.2f}%)</p>
  <p><b>Buy & Hold Return:</b> {buy_hold_return:.2f}%</p>
</div>
"""

# ذخیره چارت و معیارها
print("💾 Saving chart to HTML...")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_html = os.path.join(output_dir, "rsi_ema.html")
html_content = fig.to_html(include_plotlyjs=True, full_html=True)
html_content = html_content.replace('</body>', metrics_html + '</body>')
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"✅ Chart saved to {output_html}")