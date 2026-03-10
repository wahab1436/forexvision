import plotly.graph_objects as go
import pandas as pd

def create_candlestick_chart(df, title="Market Overview"):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )])
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')
    return fig

def create_equity_curve(equity_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity_data, mode='lines', name='Equity'))
    fig.update_layout(title='Equity Curve', yaxis_title='Balance')
    return fig
