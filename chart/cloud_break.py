import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import EMAIndicator
import plotly.io as pio

# === تنظیمات ورودی ===
symbol = input("Enter symbol (e.g. XRPUSDT): ").strip().upper()

# === مسیرهای فایل ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", f"{symbol}.csv")
signals_path = os.path.join(base_dir, f"trades_{symbol}.csv")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_html = os.path.join(output_dir, f"ichimoku_strategy_{symbol}.html")

# === بارگیری داده‌ها ===
print(f"📥 Loading data from {data_path}...")
df = pd.read_csv(data_path, parse_dates=["datetime"]).set_index("datetime")

# === بارگیری سیگنال‌ها ===
print(f"📥 Loading signals from {signals_path}...")
signals = pd.read_csv(signals_path, parse_dates=["entry_time", "exit_time"])

# === محاسبه اندیکاتورها ===
print("⚙️ Calculating indicators...")

# پارامترهای Ichimoku (ضربدر 5)
TENKAN_PERIOD = 9 * 5  # 45
KIJUN_PERIOD = 26 * 5  # 130
SENKOU_B_PERIOD = 52 * 5  # 260
DISPLACEMENT = 26  # بدون تغییر

# محاسبه Ichimoku روی داده‌های 1 دقیقه‌ای
def calculate_ichimoku(df):
    # Tenkan-sen
    tenkan_high = df['high'].rolling(TENKAN_PERIOD).max()
    tenkan_low = df['low'].rolling(TENKAN_PERIOD).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen
    kijun_high = df['high'].rolling(KIJUN_PERIOD).max()
    kijun_low = df['low'].rolling(KIJUN_PERIOD).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(DISPLACEMENT)
    
    # Senkou Span B
    senkou_b_high = df['high'].rolling(SENKOU_B_PERIOD).max()
    senkou_b_low = df['low'].rolling(SENKOU_B_PERIOD).min()
    senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(DISPLACEMENT)
    
    # ابر کومو
    upper_kumo = np.maximum(senkou_a, senkou_b)
    lower_kumo = np.minimum(senkou_a, senkou_b)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'upper_kumo': upper_kumo,
        'lower_kumo': lower_kumo
    }

# محاسبه ایچیموکو
ichi = calculate_ichimoku(df)

# محاسبه EMA
df['ema'] = EMAIndicator(close=df['close'], window=12).ema_indicator()

# === ایجاد نمودار ===
print("📊 Creating chart...")
fig = go.Figure()

# کندل‌ها
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name=symbol,
    increasing_line_color='#2ECC71',  # سبز
    decreasing_line_color='#E74C3C'   # قرمز
))

# ابر ایچیموکو
def create_cloud_segments(ichi):
    segments = []
    current = {'start': 0, 'type': None}
    senkou_a = ichi['senkou_a']
    senkou_b = ichi['senkou_b']
    
    for i in range(len(senkou_a)):
        if i < DISPLACEMENT or pd.isna(senkou_a[i]) or pd.isna(senkou_b[i]):
            continue
            
        bullish = senkou_a[i] > senkou_b[i]
        
        if current['type'] is None:
            current['type'] = 'bullish' if bullish else 'bearish'
        elif current['type'] != ('bullish' if bullish else 'bearish'):
            current['end'] = i - 1
            segments.append(current)
            current = {'start': i, 'type': 'bullish' if bullish else 'bearish'}
    
    if current['type']:
        current['end'] = len(senkou_a) - 1
        segments.append(current)
    
    return segments, senkou_a, senkou_b

# ایجاد و افزودن ابر به نمودار
cloud_segments, senkou_a, senkou_b = create_cloud_segments(ichi)
for segment in cloud_segments:
    start, end = segment['start'], segment['end']
    color = 'rgba(46, 204, 113, 0.2)' if segment['type'] == 'bullish' else 'rgba(231, 76, 60, 0.2)'
    
    fig.add_trace(go.Scatter(
        x=df.index[start:end+1],
        y=senkou_a[start:end+1],
        line=dict(color='green', width=1),
        fill=None,
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[start:end+1],
        y=senkou_b[start:end+1],
        line=dict(color='red', width=1),
        fill='tonexty',
        fillcolor=color,
        showlegend=False
    ))

# افزودن خطوط ایچیموکو برای مرجع
fig.add_trace(go.Scatter(
    x=df.index, y=senkou_a,
    mode='lines',
    name='Senkou Span A',
    line=dict(color='green', width=1),
    fill=None
))

fig.add_trace(go.Scatter(
    x=df.index, y=senkou_b,
    mode='lines',
    name='Senkou Span B',
    line=dict(color='red', width=1),
    fill=None
))

# EMA
fig.add_trace(go.Scatter(
    x=df.index, y=df['ema'],
    line=dict(color='purple', width=2),
    name='EMA 12'
))

# === افزودن سیگنال‌ها به نمودار ===
if not signals.empty:
    # سیگنال‌های خرید
    buy_signals = signals[signals['type'] == 'buy']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['entry_time'],
            y=buy_signals['entry_price'],
            mode='markers',
            marker=dict(
                color='green', 
                symbol='triangle-up', 
                size=12,
                line=dict(width=2, color='black')
            ),
            name='Buy Entry',
            customdata=buy_signals['exit_price'],
            hovertemplate=(
                '<b>Buy Signal</b><br>' +
                'Entry Time: %{x}<br>' +
                'Entry Price: %{y:.6f}<br>' +
                'Target Price: %{customdata:.6f}<extra></extra>'
            )
        ))
    
    # سیگنال‌های فروش
    sell_signals = signals[signals['type'] == 'sell']
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['entry_time'],
            y=sell_signals['entry_price'],
            mode='markers',
            marker=dict(
                color='red', 
                symbol='triangle-down', 
                size=12,
                line=dict(width=2, color='black')
            ),
            name='Sell Entry',
            customdata=sell_signals['exit_price'],
            hovertemplate=(
                '<b>Sell Signal</b><br>' +
                'Entry Time: %{x}<br>' +
                'Entry Price: %{y:.6f}<br>' +
                'Target Price: %{customdata:.6f}<extra></extra>'
            )
        ))
    
    # نقاط خروج - TP با دایره سبز
    tp_exits = signals[signals['exit_reason'] == 'TP']
    if not tp_exits.empty:
        fig.add_trace(go.Scatter(
            x=tp_exits['exit_time'],
            y=tp_exits['exit_price'],
            mode='markers',
            marker=dict(
                color='green', 
                symbol='circle', 
                size=10,
                line=dict(width=1, color='white')
            ),
            name='TP Exit',
            hovertemplate=(
                '<b>TP Exit</b><br>' +
                'Exit Time: %{x}<br>' +
                'Exit Price: %{y:.6f}<extra></extra>'
            )
        ))
    
    # نقاط خروج - SL با ضربدر قرمز
    sl_exits = signals[signals['exit_reason'] == 'SL']
    if not sl_exits.empty:
        fig.add_trace(go.Scatter(
            x=sl_exits['exit_time'],
            y=sl_exits['exit_price'],
            mode='markers',
            marker=dict(
                color='red', 
                symbol='x', 
                size=10,
                line=dict(width=1, color='white')
            ),
            name='SL Exit',
            hovertemplate=(
                '<b>SL Exit</b><br>' +
                'Exit Time: %{x}<br>' +
                'Exit Price: %{y:.6f}<extra></extra>'
            )
        ))
    
    # خطوط اتصال ورود به خروج
    for _, signal in signals.iterrows():
        fig.add_trace(go.Scatter(
            x=[signal['entry_time'], signal['exit_time']],
            y=[signal['entry_price'], signal['exit_price']],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

# === محاسبه آمار عملکرد ===
def calculate_performance_metrics(signals):
    if signals.empty:
        return {}
    
    # محاسبه سود/زیان
    signals['pnl'] = signals.apply(
        lambda row: row['exit_price'] - row['entry_price'] if row['type'] == 'buy' 
        else row['entry_price'] - row['exit_price'], axis=1
    )
    
    # محاسبه درصد سود/زیان
    signals['pnl_pct'] = signals['pnl'] / signals['entry_price'] * 100
    
    # معاملات برنده و بازنده
    win_trades = signals[signals['pnl'] > 0]
    loss_trades = signals[signals['pnl'] <= 0]
    
    # آمار کلی
    total_trades = len(signals)
    win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0
    total_profit = win_trades['pnl_pct'].sum()
    total_loss = abs(loss_trades['pnl_pct'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # حداکثر سود و زیان
    max_win = win_trades['pnl_pct'].max() if not win_trades.empty else 0
    max_loss = loss_trades['pnl_pct'].min() if not loss_trades.empty else 0
    
    # حداکثر افت سرمایه
    equity_curve = [100]
    for pnl in signals['pnl_pct']:
        equity_curve.append(equity_curve[-1] * (1 + pnl/100))
    
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # تعداد خروج‌ها
    tp_count = len(signals[signals['exit_reason'] == 'TP'])
    sl_count = len(signals[signals['exit_reason'] == 'SL'])
    
    return {
        'total_trades': total_trades,
        'win_trades': len(win_trades),
        'loss_trades': len(loss_trades),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'profit_factor': profit_factor,
        'max_win': max_win,
        'max_loss': max_loss,
        'max_drawdown': max_drawdown,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'final_equity': equity_curve[-1] - 100
    }

# محاسبه آمار
metrics = calculate_performance_metrics(signals)

# === تنظیمات نهایی نمودار ===
fig.update_layout(
    title=f'{symbol} Ichimoku Strategy Signals',
    height=900,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=False)),
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    margin=dict(b=200)  # حاشیه پایین بیشتر برای بخش آماری
)

# === ایجاد بخش آماری به صورت HTML ===
def create_stats_html(metrics):
    if not metrics:
        return "<div></div>"
    
    return f"""
<div style="font-family: Arial; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
    <h2 style="color: #2c3e50;">Performance Metrics</h2>
    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #3498db;">Trades Summary</h3>
            <p><b>Total Trades:</b> {metrics['total_trades']}</p>
            <p><b>Winning Trades:</b> {metrics['win_trades']} ({metrics['win_rate']:.2f}%)</p>
            <p><b>Losing Trades:</b> {metrics['loss_trades']}</p>
            <p><b>TP Exits:</b> {metrics['tp_count']}</p>
            <p><b>SL Exits:</b> {metrics['sl_count']}</p>
        </div>
        
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #3498db;">Profit & Loss</h3>
            <p><b>Total Profit:</b> {metrics['total_profit']:.2f}%</p>
            <p><b>Total Loss:</b> {metrics['total_loss']:.2f}%</p>
            <p><b>Profit Factor:</b> {metrics['profit_factor']:.2f}</p>
            <p><b>Max Win:</b> {metrics['max_win']:.2f}%</p>
            <p><b>Max Loss:</b> {metrics['max_loss']:.2f}%</p>
        </div>
        
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #3498db;">Risk Metrics</h3>
            <p><b>Max Drawdown:</b> {metrics['max_drawdown']:.2f}%</p>
            <p><b>Final Equity:</b> {metrics['final_equity']:.2f}%</p>
        </div>
    </div>
</div>
"""

# ذخیره نمودار به صورت HTML
print(f"💾 Saving chart to {output_html}...")
html_content = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)

# افزودن بخش آماری به HTML
stats_html = create_stats_html(metrics)
html_content = html_content.replace('</body>', stats_html + '</body>')

# ذخیره فایل نهایی
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ Done!")
print(f"Chart saved to: {os.path.abspath(output_html)}")