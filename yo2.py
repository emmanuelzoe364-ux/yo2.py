import os
import time
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging 
from PIL import Image
import yfinance as yf

# ==========================================
# ⚙️ CONFIGURATION & SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Aduse Agile Dashboard", layout="wide")

# ==========================================
# 🏛️ BRANDING
# ==========================================
try:
    logo_path = "logo.png" 
    image = Image.open(logo_path)
    st.image(image, use_container_width=True) 
    st.caption("⚡ AGILE DASHBOARD - Fisher & Delta Core")
except Exception:
    st.title("ADUSE AGILE DASHBOARD")
    st.markdown("---")

# ==========================================
# 🛠️ MOMENTUM HELPERS
# ==========================================
def calculate_fisher_transform(series, window=10):
    min_val = series.rolling(window=window).min()
    max_val = series.rolling(window=window).max()
    diff = (max_val - min_val).replace(0, 0.00001)
    norm = 2 * ((series - min_val) / diff) - 1
    norm = norm.fillna(0).clip(-0.999, 0.999)
    return 0.5 * np.log((1 + norm) / (1 - norm))

def calculate_delta_hybrid(roc, z_ratio):
    return roc - (z_ratio * 0.0002)

# ==========================================
# 🎛️ SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("Agile Settings")
TICKERS_YF = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}

# Validate period for yfinance
period_days = st.sidebar.number_input("Fetch period (days)", min_value=1, max_value=59, value=7) 
interval = st.sidebar.selectbox("Interval", options=["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"], index=3) 

st.sidebar.markdown("---")
index_ema_span = st.sidebar.number_input("Index EMA Span", min_value=1, max_value=10, value=5)
ema_long = 30
ema_very_long = 72

st.sidebar.markdown("---")
z_lookback_base = st.sidebar.number_input("Base Z-Lookback", min_value=10, max_value=500, value=168)
z_threshold = st.sidebar.slider("Signal Z-Threshold", 1.0, 3.0, 2.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Fisher/Delta Settings")
fisher_window = st.sidebar.number_input("Fisher Window", value=10)

# ==========================================
# 📥 DATA FETCHING ENGINE (YFINANCE)
# ==========================================
@st.cache_data(ttl=timedelta(minutes=5))
def fetch_and_process_data(tickers_dict, period_days, interval_str, index_ema_span, z_lookback_base, fisher_window):
    try:
        # Constraints: yfinance 1m data only allows 7 days max
        if interval_str == "1m" and period_days > 7:
            period_days = 7
            
        yf_period = f"{period_days}d"
        symbols = list(tickers_dict.values())
        data = yf.download(tickers=symbols, period=yf_period, interval=interval_str, group_by='ticker', auto_adjust=True)
        
        if data.empty:
            return pd.DataFrame()

        df_list = []
        for raw_sym, yf_sym in tickers_dict.items():
            col_name = raw_sym.replace('USDT', '')
            temp = data[yf_sym][['Close']].copy()
            temp.columns = [col_name]
            df_list.append(temp)
        
        df = pd.concat(df_list, axis=1)
        df = df.ffill(limit=3).dropna()
        
        # Calculations
        df['btc_cum'] = df['BTC'] / df['BTC'].rolling(window=z_lookback_base).mean()
        df['eth_cum'] = df['ETH'] / df['ETH'].rolling(window=z_lookback_base).mean()
        df['index_cum'] = df[['btc_cum', 'eth_cum']].mean(axis=1) 
        df['index_cum_smooth'] = df['index_cum'].ewm(span=index_ema_span, adjust=False).mean()
        df['EMA_long'] = df['index_cum_smooth'].ewm(span=ema_long, adjust=False).mean() 
        df['EMA_very_long'] = df['index_cum_smooth'].ewm(span=ema_very_long, adjust=False).mean() 

        # Z-Ratio
        df['LR'] = np.log(df['ETH'] / df['BTC']) 
        df['LR_Smoothed'] = df['LR'].ewm(span=30, adjust=False).mean()
        
        lr_vals = df['LR_Smoothed'].values
        z_ratios = np.full(len(df), np.nan)
        for i in range(z_lookback_base, len(df)):
            subset = lr_vals[max(0, i-z_lookback_base):i]
            if len(subset) > 1 and np.std(subset) > 0:
                z_ratios[i] = (lr_vals[i] - np.mean(subset)) / np.std(subset)
        df['Z_Ratio'] = z_ratios

        # Spread
        index_anchor = df['EMA_very_long'] 
        df['BTC_Dev'] = df['btc_cum'] - index_anchor
        df['ETH_Dev'] = df['eth_cum'] - index_anchor
        df['Z_Score_BTC_INDEX'] = df['BTC_Dev'] / df['BTC_Dev'].rolling(window=z_lookback_base).std()
        df['Z_Score_ETH_INDEX'] = df['ETH_Dev'] / df['ETH_Dev'].rolling(window=z_lookback_base).std()
        df['Z_Spread_Diff_Raw'] = df['Z_Score_ETH_INDEX'] - df['Z_Score_BTC_INDEX']
        df['Z_Spread_14_MA'] = df['Z_Spread_Diff_Raw'].rolling(window=14).mean()
        df['Z_Spread_30_MA'] = df['Z_Spread_Diff_Raw'].rolling(window=30).mean()
        
        df['Raw_Angle'] = np.degrees(np.arctan2(df['Z_Spread_Diff_Raw'], df['Z_Ratio']))
        df['Unified_Oscillator_Deg'] = df['Raw_Angle'].ewm(span=30, adjust=False).mean()

        # Momentum
        df['LR_ROC'] = df['LR_Smoothed'].diff()
        df['Fisher'] = calculate_fisher_transform(df['LR_ROC'], window=fisher_window)
        df['Delta_Hybrid'] = calculate_delta_hybrid(df['LR_ROC'], df['Z_Ratio'])

        return df.dropna(subset=['index_cum_smooth'])
    except Exception as e:
        st.error(f"Data Fetching Error: {e}")
        return pd.DataFrame()

# ==========================================
# 🚦 SIGNAL LOGIC
# ==========================================
def apply_signals(df, z_thr):
    df['Signal'] = "Neutral"
    long_setup = (df['Z_Ratio'] <= -z_thr) & (df['Z_Spread_Diff_Raw'] >= 0) & (df['Unified_Oscillator_Deg'] > 90)
    long_trigger = (df['Fisher'] > df['Fisher'].shift(1)) | (df['Delta_Hybrid'] > df['Delta_Hybrid'].shift(1))
    short_setup = (df['Z_Ratio'] >= z_thr) & (df['Z_Spread_Diff_Raw'] <= 0) & (df['Unified_Oscillator_Deg'] < 0)
    short_trigger = (df['Fisher'] < df['Fisher'].shift(1)) | (df['Delta_Hybrid'] < df['Delta_Hybrid'].shift(1))
    
    df.loc[long_setup & long_trigger, 'Signal'] = 'LONG'
    df.loc[short_setup & short_trigger, 'Signal'] = 'SHORT'
    return df

# Run engine
df = fetch_and_process_data(TICKERS_YF, period_days, interval, index_ema_span, z_lookback_base, fisher_window)

if df.empty:
    st.warning("⚠️ No data returned. Please try a shorter period or a larger interval.")
    st.stop()

df = apply_signals(df, z_threshold)

# ==========================================
# 📊 VISUALIZATION
# ==========================================
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                    row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
                    vertical_spacing=0.03)

fig.add_trace(go.Scatter(x=df.index, y=df['index_cum_smooth'], name='Index', line=dict(color='#0077c9', width=2), connectgaps=True), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_long'], name='EMA 30', line=dict(color='red', width=1), connectgaps=True), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_very_long'], name='EMA 72', line=dict(color='purple', dash='dot'), connectgaps=True), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['Z_Ratio'], name='Z-Ratio', line=dict(color='orange'), connectgaps=True), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Unified_Oscillator_Deg'], name='Angle', line=dict(color='yellow'), connectgaps=True), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Fisher'], name='Fisher', line=dict(color='cyan'), connectgaps=True), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Delta_Hybrid'], name='Delta Hybrid', line=dict(color='lime'), connectgaps=True), row=5, col=1)

fig.update_layout(height=1000, template="plotly_dark", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 📑 DATA FEED (ROBUST VERSION)
# ==========================================
st.markdown("### 📋 Market Data & Signals")

# Prepare display dataframe
cols = ['BTC', 'ETH', 'Z_Ratio', 'Z_Spread_14_MA', 'Z_Spread_30_MA', 'Unified_Oscillator_Deg', 'Fisher', 'Delta_Hybrid', 'Signal']
df_display = df[cols].tail(100).reset_index()

# Robust Color Function
def highlight_signals(row):
    color = ''
    if row.Signal == 'LONG':
        color = 'background-color: rgba(0, 255, 0, 0.2); color: #00FF00; font-weight: bold'
    elif row.Signal == 'SHORT':
        color = 'background-color: rgba(255, 0, 0, 0.2); color: #FF0000; font-weight: bold'
    return [color if i == len(row)-1 else '' for i in range(len(row))]

# Render using Streamlit's built-in column config (very stable)
st.dataframe(
    df_display.style.apply(highlight_signals, axis=1),
    use_container_width=True,
    height=450,
    column_config={
        "Date": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
        "BTC": st.column_config.NumberColumn("BTC Price", format="$%.2f"),
        "ETH": st.column_config.NumberColumn("ETH Price", format="$%.2f"),
        "Z_Ratio": st.column_config.NumberColumn("Z-Ratio", format="%.3f"),
        "Unified_Oscillator_Deg": st.column_config.NumberColumn("Angle", format="%.1f°"),
        "Fisher": st.column_config.NumberColumn("Fisher", format="%.4f"),
        "Delta_Hybrid": st.column_config.NumberColumn("Delta", format="%.5f"),
        "Signal": st.column_config.TextColumn("Signal")
    }
)

if st.button("🔄 Refresh Data"):
    fetch_and_process_data.clear()
    st.rerun()
