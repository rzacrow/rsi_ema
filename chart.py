import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

# ——————————————————————————————————————————————————————
# 1. Load trades & compute metrics
# ——————————————————————————————————————————————————————
trades = pd.read_csv("trades.csv", parse_dates=['entry_time','exit_time'])
total_trades    = len(trades)
winning_trades  = len(trades[trades["pnl_usd"] > 0])
losing_trades   = len(trades[trades["pnl_usd"] <= 0])
win_rate        = winning_trades/total_trades*100 if total_trades else 0
avg_gain        = trades[trades["pnl_usd"]>0]["pnl_pct"].mean()*100 if winning_trades else 0
avg_loss        = trades[trades["pnl_usd"]<=0]["pnl_pct"].mean()*100 if losing_trades else 0
gross_profit    = trades[trades["pnl_usd"]>0]["pnl_usd"].sum()
gross_loss      = trades[trades["pnl_usd"]<=0]["pnl_usd"].sum()
net_profit      = gross_profit + gross_loss
profit_factor   = gross_profit/abs(gross_loss) if gross_loss!=0 else float("inf")
initial_balance = 100
final_balance   = trades["balance"].iloc[-1] if "balance" in trades else initial_balance
roi             = (final_balance-initial_balance)/initial_balance*100
trades["equity_curve"] = trades["balance"]
trades["peak"]         = trades["equity_curve"].cummax()
trades["drawdown"]     = trades["equity_curve"] - trades["peak"]
max_drawdown   = trades["drawdown"].min()
buy_hold_return = (trades["exit_price"].iloc[-1] - trades["entry_price"].iloc[0]) \
                  / trades["entry_price"].iloc[0] * 100

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
  <p><b>Max Drawdown:</b> ${max_drawdown:.2f}</p>
  <p><b>Buy & Hold Return:</b> {buy_hold_return:.2f}%</p>
</div>
"""

# ——————————————————————————————————————————————————————
# 2. Load price & compute Ichimoku cloud
# ——————————————————————————————————————————————————————
start = trades["entry_time"].min() - timedelta(minutes=60)
end   = trades["exit_time"].max()   + timedelta(minutes=60)

price = pd.read_csv("data/BTCUSDT.csv", parse_dates=['datetime'])
price.set_index("datetime", inplace=True)
price = price.loc[start:end]

ich_tf      = 5
displacement= 26
# resample to 5m
ich = price.resample(f"{ich_tf}T").agg({
    'open':'first','high':'max','low':'min','close':'last'
}).dropna()
# compute lines
tenkan = (ich['high'].rolling(9).max()+ich['low'].rolling(9).min())/2
kijun  = (ich['high'].rolling(26).max()+ich['low'].rolling(26).min())/2
senkou_a = ((tenkan+kijun)/2).shift(displacement)
senkou_b = ((ich['high'].rolling(52).max()+ich['low'].rolling(52).min())/2).shift(displacement)
# align back to 1m
cloud = pd.DataFrame({'senkou_a':senkou_a,'senkou_b':senkou_b})
cloud = cloud.reindex(price.index, method='ffill')

# ——————————————————————————————————————————————————————
# 3. Draw chart
# ——————————————————————————————————————————————————————
fig = go.Figure()
# candles
fig.add_trace(go.Candlestick(
    x=price.index, open=price['open'],
    high=price['high'], low=price['low'], close=price['close'],
    name='BTC/USDT'
))
# cloud area
for i in range(1,len(cloud)):
    a1,b1 = cloud['senkou_a'].iloc[i-1], cloud['senkou_b'].iloc[i-1]
    a2,b2 = cloud['senkou_a'].iloc[i],   cloud['senkou_b'].iloc[i]
    t1,t2 = cloud.index[i-1], cloud.index[i]
    if a1< b1 and a2< b2:
        # bullish cloud
        fig.add_trace(go.Scatter(
            x=[t1,t2,t2,t1], y=[a1,a2,b2,b1],
            fill='toself', fillcolor='rgba(0,255,0,0.2)',
            line=dict(width=0), hoverinfo='skip', showlegend=False
        ))
    elif a1> b1 and a2> b2:
        # bearish cloud
        fig.add_trace(go.Scatter(
            x=[t1,t2,t2,t1], y=[b1,b2,a2,a1],
            fill='toself', fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0), hoverinfo='skip', showlegend=False
        ))
# plot Senkou lines
fig.add_trace(go.Scatter(x=cloud.index,y=cloud['senkou_a'],
                         mode='lines', line=dict(color='green'), name='Senkou A'))
fig.add_trace(go.Scatter(x=cloud.index,y=cloud['senkou_b'],
                         mode='lines', line=dict(color='red'),   name='Senkou B'))

# entry markers
longs  = trades[trades['type']=='long']
shorts = trades[trades['type']=='short']
fig.add_trace(go.Scatter(
    x=longs['entry_time'], y=longs['entry_price'],
    mode='markers', marker=dict(color='green',symbol='triangle-up',size=8),
    name='Long Entry'
))
fig.add_trace(go.Scatter(
    x=shorts['entry_time'], y=shorts['entry_price'],
    mode='markers', marker=dict(color='red',symbol='triangle-down',size=8),
    name='Short Entry'
))

fig.update_layout(
    title="Ichimoku Strategy with Metrics",
    xaxis_rangeslider_visible=False,
    height=700,
)

# ——————————————————————————————————————————————————————
# 4. Export combined HTML
# ——————————————————————————————————————————————————————
chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
full_html = f"""
<html><head><meta charset="utf-8"><title>Report</title></head>
<body style="font-family:Arial; margin:0; padding:0;">
  <div style="width:100%; height:75vh;">{chart_html}</div>
  {metrics_html}
</body></html>
"""
with open("ichimoku_report.html","w",encoding="utf-8") as f:
    f.write(full_html)

print("✅ Report saved → ichimoku_report.html")
