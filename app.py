import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Premium Responsive Dashboard Configuration
st.set_page_config(
    page_title="XLE Energy Stocks RRG Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark UI Accent & Styling
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1 { font-weight: 800; color: #0F172A; letter-spacing: -1px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛢️ US Energy Stocks Rotation (XLE RRG Dashboard)")
st.caption("🛡️ Snapshot Engine Enabled | Lookback & Historical Offset Mode | Auto-Refresh: 30 Min")

# 💡 30-MINUTE SMART CACHE WITH PARAMETERS
@st.cache_data(ttl=1800, show_spinner="Loading Historical Energy Matrix...")
def calculate_rrg_cached(tickers_dict, benchmark, interval, window=14, tail_length=5, history_offset=0):
    now = datetime.now()
    
    # Extra buffer days mangte hain taaki offset shifting ke liye kaafi data bacha rahe
    buffer_days = history_offset * 7 if interval in ['1wk', '1mo'] else history_offset + 10
    
    if interval == '60m': 
        start_date = (now - timedelta(days=32 + buffer_days)).strftime('%Y-%m-%d')
    elif interval == '1d': 
        start_date = (now - timedelta(days=365 + buffer_days)).strftime('%Y-%m-%d')
    elif interval == '1wk': 
        start_date = (now - timedelta(days=730 + buffer_days)).strftime('%Y-%m-%d')
    else: 
        start_date = (now - timedelta(days=1500 + buffer_days)).strftime('%Y-%m-%d')
        
    end_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')

    all_tickers = list(tickers_dict.keys()) + [benchmark]
    batch_size = 15
    combined_df = pd.DataFrame()
    
    try:
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            batch_data = yf.download(
                batch, start=start_date, end=end_date, 
                interval=interval, auto_adjust=True, progress=False
            )
            
            if not batch_data.empty:
                if isinstance(batch_data.columns, pd.MultiIndex):
                    if 'Close' in batch_data.columns.levels[0]:
                        batch_close = batch_data['Close']
                    else:
                        continue
                else:
                    if 'Close' in batch_data.columns:
                        batch_close = batch_data[['Close']]
                    else:
                        batch_close = batch_data
                        
                combined_df = pd.concat([combined_df, batch_close], axis=1)
                
        if combined_df.empty:
            return pd.DataFrame(), pd.DataFrame(), "No Data"
            
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        combined_df = combined_df.ffill().bfill()
        
        valid_tickers = [t for t in tickers_dict.keys() if t in combined_df.columns and not combined_df[t].isna().all()]
        
        if not valid_tickers or benchmark not in combined_df.columns:
            return pd.DataFrame(), pd.DataFrame(), "No Data"
        
        # RRG Mathematical Modeling
        rs_ratios = pd.DataFrame()
        for t in valid_tickers:
            rs_ratios[t] = (combined_df[t] / combined_df[benchmark]) * 100
            
        if rs_ratios.shape[0] < (window * 2):
            return pd.DataFrame(), pd.DataFrame(), "No Data"
            
        rs_ratio_smoothed = rs_ratios.ewm(span=window, adjust=False).mean()
        mean_rs = rs_ratio_smoothed.rolling(window=window).mean()
        std_rs = rs_ratio_smoothed.rolling(window=window).std()
        jdk_rs_ratio = 100 + ((rs_ratio_smoothed - mean_rs) / std_rs) * 10
        
        rs_momentum = rs_ratio_smoothed.pct_change(periods=window) * 100
        rs_mom_smoothed = rs_momentum.ewm(span=window, adjust=False).mean()
        mean_mom = rs_mom_smoothed.rolling(window=window).mean()
        std_mom = rs_mom_smoothed.rolling(window=window).std()
        jdk_rs_momentum = 100 + ((rs_mom_smoothed - mean_mom) / std_mom) * 10
        
        # Dropna processing before slicing
        jdk_rs_ratio = jdk_rs_ratio.dropna()
        jdk_rs_momentum = jdk_rs_momentum.dropna()
        
        # 📊 🆕 HISTORICAL SNAPSHOT SLICING ENGINE
        if history_offset > 0:
            if history_offset >= len(jdk_rs_ratio):
                history_offset = len(jdk_rs_ratio) - tail_length - 1
            
            # Pure matrix pointer shift backwards
            jdk_rs_ratio = jdk_rs_ratio.iloc[:-history_offset]
            jdk_rs_momentum = jdk_rs_momentum.iloc[:-history_offset]
            
        snapshot_date = jdk_rs_ratio.index[-1].strftime('%B %d, %Y')
        
        return jdk_rs_ratio.tail(tail_length), jdk_rs_momentum.tail(tail_length), snapshot_date
        
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), "Error"

def plot_rrg_labeled(jdk_rs_ratio, jdk_rs_momentum, tickers, title_date):
    if jdk_rs_ratio.empty or jdk_rs_momentum.empty or len(jdk_rs_ratio.columns) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Data Unavailable for this specific offset window.<br>Please reduce history offset.", 
            showarrow=False, font=dict(size=16, color="#64748B")
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white', height=400)
        return fig
        
    fig = go.Figure()
    
    all_x = jdk_rs_ratio.values.flatten()
    all_y = jdk_rs_momentum.values.flatten()
    
    if len(all_x) == 0 or len(all_y) == 0:
        max_pad = 5.0
    else:
        max_pad = max(abs(all_x.max() - 100), abs(100 - all_x.min()), abs(all_y.max() - 100), abs(100 - all_y.min())) + 1.5
    
    x_range = [100 - max_pad, 100 + max_pad]
    y_range = [100 - max_pad, 100 + max_pad]
    
    # Quadrant Shading
    fig.add_shape(type="rect", x0=100, y0=100, x1=100+max_pad, y1=100+max_pad, fillcolor="rgba(34,197,94,0.02)", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=100-max_pad, x1=100+max_pad, y1=100, fillcolor="rgba(234,179,8,0.02)", line_width=0)
    fig.add_shape(type="rect", x0=100-max_pad, y0=100-max_pad, x1=100, y1=100, fillcolor="rgba(239,68,68,0.02)", line_width=0)
    fig.add_shape(type="rect", x0=100-max_pad, y0=100, x1=100, y1=100+max_pad, fillcolor="rgba(59,130,246,0.02)", line_width=0)
    
    fig.add_shape(type="line", x0=100, y0=100-max_pad, x1=100, y1=100+max_pad, line=dict(color="rgba(148,163,184,0.4)", width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=100-max_pad, y0=100, x1=100+max_pad, y1=100, line=dict(color="rgba(148,163,184,0.4)", width=1.5, dash="dash"))
    
    for col in jdk_rs_ratio.columns:
        x_vals = jdk_rs_ratio[col].values
        y_vals = jdk_rs_momentum[col].values
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers', name=tickers.get(col, col), 
            line=dict(width=2.5),
            marker=dict(
                size=[4]*(len(x_vals)-1) + [12],
                symbol=['circle']*(len(x_vals)-1) + ['triangle-up'],
                line=dict(width=1, color="white")
            ),
            hovertemplate=f"<b>{tickers.get(col,col)}</b><br>RS Ratio: %{{x:.2f}}<br>RS Momentum: %{{y:.2f}}<extra></extra>"
        ))
        
        fig.add_annotation(
            x=x_vals[-1], y=y_vals[-1], text=f"<b>{col}</b>", 
            showarrow=False, xshift=14, yshift=6,
            font=dict(size=11, color="#0F172A", family="Arial Black"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(148,163,184,0.3)", borderpad=1, borderwidth=1, align="center"
        )
            
    fig.add_annotation(x=100+max_pad*0.75, y=100+max_pad*0.88, text="🟩 LEADING", font=dict(color="#16a34a", size=14, weight="bold"), showarrow=False)
    fig.add_annotation(x=100+max_pad*0.75, y=100-max_pad*0.88, text="🟨 WEAKENING", font=dict(color="#ca8a04", size=14, weight="bold"), showarrow=False)
    fig.add_annotation(x=100-max_pad*0.75, y=100-max_pad*0.88, text="🟥 LAGGING", font=dict(color="#dc2626", size=14, weight="bold"), showarrow=False)
    fig.add_annotation(x=100-max_pad*0.75, y=100+max_pad*0.88, text="🟦 IMPROVING", font=dict(color="#2563eb", size=14, weight="bold"), showarrow=False)
    
    fig.update_layout(
        title=dict(text=f"📅 Snapshot View Target Date: <b>{title_date}</b>", font=dict(size=15, color="#475569")),
        xaxis_title="👉 Trend Strength (RS Ratio)", yaxis_title="🚀 Stock Velocity (RS Momentum)",
        xaxis=dict(range=x_range, gridcolor="rgba(241,245,249,1)", zeroline=False),
        yaxis=dict(range=y_range, gridcolor="rgba(241,245,249,1)", zeroline=False),
        height=820, margin=dict(l=20, r=30, t=50, b=20), plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Universe
xle_full_universe = {
    'XOM': 'Exxon Mobil Corp.', 'CVX': 'Chevron Corp.', 'COP': 'ConocoPhillips', 
    'EOG': 'EOG Resources Inc.', 'OXY': 'Occidental Petroleum', 'DVN': 'Devon Energy Corp.', 
    'FANG': 'Diamondback Energy', 'HES': 'Hess Corp.', 'MRO': 'Marathon Oil', 
    'APA': 'APA Corporation', 'EQT': 'EQT Corp.', 'CTRA': 'Cerra Energy', 
    'MPC': 'Marathon Petroleum', 'VLO': 'Valero Energy Corp.', 'PSX': 'Phillips 66', 
    'SLB': 'Schlumberger Ltd.', 'HAL': 'Halliburton Co.', 'BKR': 'Baker Hughes Co.', 
    'WMB': 'Williams Companies', 'OKE': 'ONEOK Inc.', 'KMI': 'Kinder Morgan Inc.', 
    'TRGP': 'Targa Resources'
}
xle_benchmark = 'XLE'

# Sidebar Controls
st.sidebar.header("⚙️ Configuration Controls")

selected_stocks = st.sidebar.multiselect(
    "Select Energy Stocks to Plot",
    options=list(xle_full_universe.keys()), default=list(xle_full_universe.keys())[:12] # Top 12 selected by default
)

tail = st.sidebar.slider("Tail Length (History)", min_value=3, max_value=15, value=5)

# 📅 NEW: DYNAMIC LOOKBACK OFFSET SLIDER 
offset = st.sidebar.slider(
    "⏳ Historical Offset (Shift Backwards)", 
    min_value=0, max_value=7, value=0,
    help="Is slider ko aage badha kar aap pichle 1 se lekar 7 periods (Days/Weeks) pehle ka RRG structure track kar sakte hain."
)

if st.sidebar.button("🔄 Force Clear Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

active_tickers = {k: xle_full_universe[k] for k in selected_stocks} if selected_stocks else xle_full_universe

# Tabs Setup
t1, t2, t3, t4 = st.tabs(["📊 Hourly View", "📈 Daily Matrix", "📆 Weekly Rotation", "⏳ Monthly Macro"])

with t1:
    r, m, s_date = calculate_rrg_cached(active_tickers, xle_benchmark, '60m', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)

with t2:
    r, m, s_date = calculate_rrg_cached(active_tickers, xle_benchmark, '1d', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)

with t3:
    r, m, s_date = calculate_rrg_cached(active_tickers, xle_benchmark, '1wk', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)

with t4:
    r, m, s_date = calculate_rrg_cached(active_tickers, xle_benchmark, '1mo', tail_length=tail, history_offset=offset)
    st.plotly_chart(plot_rrg_labeled(r, m, active_tickers, s_date), use_container_width=True)
