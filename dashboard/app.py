import streamlit as st
import pandas as pd
import os
from dashboard.plotly_charts import create_candlestick_chart, create_equity_curve
from utils.data_utils import DataManager
from loguru import logger

st.set_page_config(page_title="ForexVision Dashboard", layout="wide")

st.title("ForexVision Trading Dashboard")

@st.cache_data
def load_data(ticker):
    dm = DataManager()
    return dm.get_data(ticker, mode='historical')

ticker = st.selectbox("Select Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])

if st.button("Refresh Data"):
    st.rerun()

df = load_data(ticker)

if df is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_candlestick_chart(df.tail(100), title=f"{ticker} Price Action"), use_container_width=True)
    with col2:
        st.metric("Last Close", df['Close'].iloc[-1])
        st.metric("Volatility (ATR)", df['atr_14'].iloc[-1] if 'atr_14' in df else "N/A")
        
    st.subheader("System Status")
    st.write("Data Source: Yahoo Finance (Primary)")
    st.write("Model Status: Loaded")
else:
    st.error("No data available")
