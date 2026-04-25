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

from binance.client import Client
from binance.exceptions import BinanceAPIException

# ==========================================
# ⚙️ CONFIGURATION & SETUP
# ==========================================
BINANCE_CLIENT = Client("", "", requests_params={'timeout': 30}) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Aduse Agile (Fisher & Delta Hybrid)", layout="wide")

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
    """Normalized Fisher Transform of the input series."""
    min_val = series.rolling(window=window).min()
    max_val = series.rolling(window=window).max()
    # Prevent division by zero
    diff = (max_val - min_val).replace(0, 0.00001)
    
    # Normalize to -1 to 1
    norm = 2 * ((series - min_val) / diff) - 1
    norm = norm.fillna(0).clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + norm) / (1 - norm))
    return fisher

def calculate_delta_hybrid(roc, z_ratio):
    """Delta Hybrid calculation balancing ROC and Z-Ratio."""
    return roc - (z_ratio * 0.0002)

# ==========================================
# 🎛️ SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("Agile Settings")
TICKERS = ["BTCUSDT", "ETHUSDT"]

period_days = st.sidebar.number_input("Fetch period (days)", min_value=1, max_value=365, value=2) 
interval = st.sidebar.selectbox("Interval", options=["1m", "5m", "15m","30m","1h","1d"], index=0) 

st.sidebar.markdown("---")
index_ema_span = st.sidebar.number_input("Cumulative Index EMA Span", min_value=1, max_value=10, value=5)
ema_long = 30
ema_very_long = 72

st.sidebar.markdown("---")
z_lookback_base = st.sidebar.number_input("Base Z-Lookback", min_value=10, max_value=500, value=168)
z_threshold = st.sidebar.slider("Signal Z-Threshold", 1.0, 3.0, 2.0)
corr_threshold = st.sidebar.slider("Correlation Threshold", 0.0, 1.0, 0.70)

st.sidebar.markdown("---")
st.sidebar.subheader("Fisher/Delta Settings")
fisher_window = st.sidebar.number_input("Fisher Window", value=10)
volume_length = 14

# ==========================================
# 📥 DATA FETCHING ENGINE
# ==========================================
def map_interval(interval_str):
    mapping = {
        "1m": Client.KLINE_INTERVAL_1MINUTE, 
        "5m": Client.KLINE_INTERVAL_5MINUTE, 
        "15m": Client.KLINE_INTERVAL_15MINUTE, 
        "30m": Client.KLINE_INTERVAL_30MINUTE, 
        "1h": Client.KLINE_INTERVAL_1HOUR, 
        "1d": Client.KLINE_INTERVAL_1DAY
    }
    return mapping.get(interval_str, Client.KLINE_INTERVAL_1HOUR)

def fetch_binance_data(symbol, interval_str, start_date):
    try:
        binance_interval = map_interval(interval_str)
        start_date_str = start_date.strftime("%d %b, %Y %H:%M:%S")
        klines = BINANCE_CLIENT.get_historical_klines(symbol=symbol, interval=binance_interval, start_str=start_date_str)
        
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'QAV', 'Tr', 'TBB', 'TBQ', 'Ign'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df.set_index('Open time', inplace=True)
        return df[['Close', 'Volume']].rename(columns={'Close': 'price', 'Volume': 'volume'})
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=timedelta(seconds=60))
def fetch_and_process_data(tickers, period_days, interval, index_ema_span, z_lookback_base, fisher_window):
    # Determine lookback buffer
    CALC_DAYS = period_days + 15
    start_date = datetime.now() - timedelta(days=CALC_DAYS) 

    data_dict = {}
    for ticker in tickers:
        df_asset = fetch_binance_data(ticker, interval, start_date)
        if df_asset.empty: return pd.DataFrame()
        data_dict[ticker] = df_asset
        
    close_prices = pd.DataFrame({k.replace('USDT', ''): v['price'] for k, v in data_dict.items()})
    volume_series = pd.DataFrame({k.replace('USDT', ''): v['volume'] for k, v in data_dict.items()}).sum(axis=1)
    df = close_prices.dropna().copy()
    df = df.iloc[:-1] 

    # 1. Rolling Correlation & Adaptive Lookback
    interval_map = {"1m": 43200, "5m": 8640, "15m": 2880, "30m": 1440, "1h": 720, "1d": 30}
    corr_window = min(len(df)//2, interval_map.get(interval, 720))
    df['Rolling_Corr'] = df['BTC'].rolling(window=corr_window).corr(df['ETH'])

    returns = df['BTC'].pct_change()
    volatility = returns.rolling(window=z_lookback_base).std()
    mean_vol = volatility.rolling(window=z_lookback_base*2).mean()
    vol_ratio = (volatility / mean_vol).fillna(1.0)
    df['Adaptive_Lookback'] = (z_lookback_base / vol_ratio).clip(z_lookback_base*0.5, z_lookback_base*2.0).astype(int)

    # 2. Normalized Prices & Index
    df['btc_cum'] = df['BTC'] / df['BTC'].rolling(window=z_lookback_base).mean()
    df['eth_cum'] = df['ETH'] / df['ETH'].rolling(window=z_lookback_base).mean()
    df['index_cum'] = df[['btc_cum', 'eth_cum']].mean(axis=1) 
    df['index_cum_smooth'] = df['index_cum'].ewm(span=index_ema_span, adjust=False).mean()
    df['EMA_long'] = df['index_cum_smooth'].ewm(span=ema_long, adjust=False).mean() 
    df['EMA_very_long'] = df['index_cum_smooth'].ewm(span=ema_very_long, adjust=False).mean() 

    # 3. Z-Ratio & Spread
    df['LR'] = np.log(df['ETH'] / df['BTC']) 
    df['LR_Smoothed'] = df['LR'].ewm(span=30, adjust=False).mean()
    
    lr_vals = df['LR_Smoothed'].values
    lookbacks = df['Adaptive_Lookback'].values
    z_ratios = np.full(len(df), np.nan)
    for i in range(z_lookback_base, len(df)):
        w = lookbacks[i]
        subset = lr_vals[max(0, i-w):i]
        if len(subset) > 0 and np.std(subset) > 0:
            z_ratios[i] = (lr_vals[i] - np.mean(subset)) / np.std(subset)
    df['Z_Ratio'] = z_ratios

    index_anchor = df['EMA_very_long'] 
    df['BTC_Dev'] = df['btc_cum'] - index_anchor
    df['ETH_Dev'] = df['eth_cum'] - index_anchor
    df['Z_Score_BTC_INDEX'] = df['BTC_Dev'] / df['BTC_Dev'].rolling(window=z_lookback_base).std()
    df['Z_Score_ETH_INDEX'] = df['ETH_Dev'] / df['ETH_Dev'].rolling(window=z_lookback_base).std()
    
    # --- Z-SPREAD RAW AND MOVING AVERAGES ---
    df['Z_Spread_Diff_Raw'] = df['Z_Score_ETH_INDEX'] - df['Z_Score_BTC_INDEX']
    df['Z_Spread_14_MA'] = df['Z_Spread_Diff_Raw'].rolling(window=14).mean()
    df['Z_Spread_30_MA'] = df['Z_Spread_Diff_Raw'].rolling(window=30).mean()
    
    df['Raw_Angle'] = np.degrees(np.arctan2(df['Z_Spread_Diff_Raw'], df['Z_Ratio']))
    df['Unified_Oscillator_Deg'] = df['Raw_Angle'].ewm(span=30, adjust=False).mean()

    # 4. Fisher & Delta Hybrid Calculation
    df['LR_ROC'] = df['LR_Smoothed'].diff()
    df['Fisher'] = calculate_fisher_transform(df['LR_ROC'], window=fisher_window)
    df['Delta_Hybrid'] = calculate_delta_hybrid(df['LR_ROC'], df['Z_Ratio'])

    # Apply Correlation Mask
    mask = df['Rolling_Corr'] < corr_threshold
    cols_to_mask = ['Z_Ratio', 'Unified_Oscillator_Deg', 'Raw_Angle', 'Fisher', 'Delta_Hybrid']
    df.loc[mask, cols_to_mask] = np.nan

    df['volume'] = volume_series.reindex(df.index, method='nearest')
    df['Volume_MA'] = df['volume'].rolling(volume_length, min_periods=1).mean()

    cutoff_date = df.index[-1] - timedelta(days=period_days)
    return df[df.index > cutoff_date].copy()

# ==========================================
# 🚦 SIGNAL LOGIC
# ==========================================
def apply_signals(df, z_thr):
    df['Signal'] = "Neutral"
    
    # LONG Condition: Z-Ratio oversold, Spread improving, Q2 Angle, Momentum shift Up
    long_setup = (df['Z_Ratio'] <= -z_thr) & (df['Z_Spread_Diff_Raw'] >= 0) & (df['Unified_Oscillator_Deg'] > 90)
    long_trigger = (df['Fisher'] > df['Fisher'].shift(1)) | (df['Delta_Hybrid'] > df['Delta_Hybrid'].shift(1))
    
    # SHORT Condition: Z-Ratio overbought, Spread declining, Q4 Angle, Momentum shift Down
    short_setup = (df['Z_Ratio'] >= z_thr) & (df['Z_Spread_Diff_Raw'] <= 0) & (df['Unified_Oscillator_Deg'] < 0)
    short_trigger = (df['Fisher'] < df['Fisher'].shift(1)) | (df['Delta_Hybrid'] < df['Delta_Hybrid'].shift(1))
    
    df.loc[long_setup & long_trigger, 'Signal'] = 'LONG'
    df.loc[short_setup & short_trigger, 'Signal'] = 'SHORT'
    return df

# Execute Processing
df = fetch_and_process_data(TICKERS, period_days, interval, index_ema_span, z_lookback_base, fisher_window)

if df.empty:
    st.error("No data returned.")
    st.stop()

df = apply_signals(df, z_threshold)

# ==========================================
# 📊 VISUALIZATION
# ==========================================
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                    row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
                    vertical_spacing=0.03)

# Row 1: Index and Assets
fig.add_trace(go.Scatter(x=df.index, y=df['index_cum_smooth'], name='Index', line=dict(color='#0077c9', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_long'], name='EMA 30', line=dict(color='red', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_very_long'], name='EMA 72', line=dict(color='purple', dash='dot')), row=1, col=1) 

# Row 2: Z-Ratio
fig.add_trace(go.Scatter(x=df.index, y=df['Z_Ratio'], name='Z-Ratio', line=dict(color='orange')), row=2, col=1)
fig.add_hline(y=z_threshold, line=dict(color='red', dash='dot'), row=2, col=1)
fig.add_hline(y=-z_threshold, line=dict(color='green', dash='dot'), row=2, col=1)
fig.add_hline(y=0, line=dict(color='gray', width=0.5), row=2, col=1)

# Row 3: Angle
fig.add_trace(go.Scatter(x=df.index, y=df['Unified_Oscillator_Deg'], name='Angle', line=dict(color='yellow')), row=3, col=1)
fig.add_hline(y=0, row=3, col=1)

# Row 4: Fisher Transform
fig.add_trace(go.Scatter(x=df.index, y=df['Fisher'], name='Fisher', line=dict(color='cyan')), row=4, col=1)
fig.add_hline(y=0, row=4, col=1)

# Row 5: Delta Hybrid
fig.add_trace(go.Scatter(x=df.index, y=df['Delta_Hybrid'], name='Delta Hybrid', line=dict(color='lime')), row=5, col=1)
fig.add_hline(y=0, row=5, col=1)

# Annotate Signals as Vertical Lines
for idx, row in df[df['Signal'] == 'LONG'].iterrows():
    fig.add_vline(x=idx, line=dict(color='green', width=1, dash='dot'), opacity=0.3)
for idx, row in df[df['Signal'] == 'SHORT'].iterrows():
    fig.add_vline(x=idx, line=dict(color='red', width=1, dash='dot'), opacity=0.3)

fig.update_layout(height=1100, template="plotly_dark", hovermode="x unified", 
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  margin=dict(l=50, r=50, t=50, b=50))

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 📑 DATA FEED
# ==========================================
st.markdown("### 📋 Market Data & Signals")

data_cols = [
    'index_cum_smooth', 
    'Z_Ratio', 
    'Z_Spread_Diff_Raw', 
    'Z_Spread_14_MA',   # New Column
    'Z_Spread_30_MA',   # New Column
    'Unified_Oscillator_Deg', 
    'Fisher', 
    'Delta_Hybrid',
    'Signal'
]

df_view = df[data_cols].tail(10000).copy()

def color_signal(val):
    if val == 'LONG': return 'color: #00FF00; font-weight: bold'
    if val == 'SHORT': return 'color: #FF0000; font-weight: bold'
    return ''

st.dataframe(
    df_view.style.applymap(color_signal, subset=['Signal']).format({c: "{:.4f}" for c in data_cols if c != 'Signal'}),
    use_container_width=True,
    height=600,  
    column_config={
        "_index": st.column_config.DatetimeColumn("Open Time", format="D MMM YYYY, HH:mm:ss"),
        "Unified_Oscillator_Deg": st.column_config.NumberColumn(format="%.1f°"),
        "Z_Spread_14_MA": st.column_config.NumberColumn("Spread 14 MA", format="%.4f"),
        "Z_Spread_30_MA": st.column_config.NumberColumn("Spread 30 MA", format="%.4f"),
    }
)

if st.button(f"🔄 Refresh Data"):
    fetch_and_process_data.clear()
    st.rerun()