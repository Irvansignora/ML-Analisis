"""
Sales ML Analytics Dashboard - Full Rebuild
============================================
CEO-level dashboard: KPI, Sales, Profitability, Customer, Regional, Forecast, Anomaly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import base64

from preprocessing import DataPreprocessor
from ml_model import SalesForecaster, ProductSegmenter, AnomalyDetector, ModelComparator
from utils import SalesAnalyzer, ReportGenerator, format_currency, format_number, create_sample_data

st.set_page_config(
    page_title="Sales ML Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── OCEAN BLUE GLASSMORPHISM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 0% 0%,   rgba(6,182,212,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 0%,  rgba(14,165,233,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 100%, rgba(6,182,212,0.12) 0%, transparent 60%),
        linear-gradient(160deg, #020b18 0%, #03142e 40%, #041d3d 70%, #020b18 100%);
    min-height: 100vh;
}

#MainMenu { visibility: hidden !important; }
header[data-testid="stHeader"] { height: 0px !important; min-height: 0px !important; overflow: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
.main .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(2,11,24,0.97) 0%, rgba(3,20,46,0.98) 100%) !important;
    border-right: 1px solid rgba(6,182,212,0.2) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #e0f2fe !important; }
[data-testid="stSidebar"] small { color: #64748b !important; }
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h4 { color: #bae6fd !important; font-weight: 600 !important; }

.hero-header {
    background: linear-gradient(135deg, rgba(6,182,212,0.9) 0%, rgba(14,165,233,0.85) 50%, rgba(56,189,248,0.8) 100%);
    border-radius: 20px; padding: 32px 44px; margin-bottom: 24px;
    position: relative; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(6,182,212,0.25), inset 0 1px 0 rgba(255,255,255,0.2);
}
.hero-header::before {
    content: ''; position: absolute; top: -60%; right: -5%;
    width: 380px; height: 380px; background: rgba(255,255,255,0.06); border-radius: 50%;
}
.hero-title { font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0; position: relative; z-index: 1; }
.hero-sub { font-size: 1rem; color: rgba(255,255,255,0.92); margin: 6px 0 0 0; position: relative; z-index: 1; }

.kpi-card {
    background: rgba(6,182,212,0.07); backdrop-filter: blur(16px);
    border: 1px solid rgba(6,182,212,0.25); border-radius: 16px;
    padding: 20px 18px; position: relative; overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(6,182,212,0.08), inset 0 1px 0 rgba(255,255,255,0.07);
}
.kpi-card:hover { border-color: rgba(6,182,212,0.5); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(6,182,212,0.18); }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 16px 16px 0 0; }
.kpi-card.c1::before { background: linear-gradient(90deg,#06b6d4,#0ea5e9); }
.kpi-card.c2::before { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.c3::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.c4::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-card.c5::before { background: linear-gradient(90deg,#ec4899,#f472b6); }
.kpi-card.c6::before { background: linear-gradient(90deg,#38bdf8,#7dd3fc); }
.kpi-card.c7::before { background: linear-gradient(90deg,#10b981,#06b6d4); }
.kpi-card.c8::before { background: linear-gradient(90deg,#f59e0b,#ec4899); }
.kpi-icon  { font-size: 1.6rem; margin-bottom: 8px; display: block; }
.kpi-value { font-size: 1.35rem; font-weight: 700; color: #fff; display: block; line-height: 1.2; }
.kpi-label { font-size: 0.72rem; color: #bae6fd; font-weight: 500; text-transform: uppercase; letter-spacing: 0.7px; margin-top: 4px; display: block; }
.kpi-delta { font-size: 0.78rem; margin-top: 6px; display: inline-block; padding: 2px 8px; border-radius: 20px; font-weight: 600; }
.kpi-delta.up   { background: rgba(16,185,129,0.15); color: #34d399; }
.kpi-delta.down { background: rgba(239,68,68,0.15);  color: #f87171; }
.kpi-delta.neu  { background: rgba(100,116,139,0.2); color: #94a3b8; }

.glass-card {
    background: rgba(255,255,255,0.04); backdrop-filter: blur(20px);
    border: 1px solid rgba(6,182,212,0.18); border-radius: 16px;
    padding: 22px; margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
}
.card-title {
    font-size: 1rem; font-weight: 700; color: #e0f2fe;
    margin: 0 0 16px 0; padding-bottom: 10px;
    border-bottom: 1px solid rgba(6,182,212,0.15);
    display: flex; align-items: center; gap: 8px;
}
.insight-card {
    background: rgba(6,182,212,0.07); backdrop-filter: blur(12px);
    border: 1px solid rgba(6,182,212,0.22); border-radius: 10px;
    padding: 12px 15px; margin-bottom: 9px;
    color: #e0f2fe; font-size: 0.88rem;
    display: flex; align-items: flex-start; gap: 9px;
}
.insight-card::before { content: '💡'; font-size: 0.95rem; flex-shrink: 0; }

.section-divider {
    border: none; border-top: 1px solid rgba(6,182,212,0.12);
    margin: 24px 0;
}
.section-header {
    font-size: 1.15rem; font-weight: 700; color: #38bdf8;
    margin: 28px 0 16px 0; display: flex; align-items: center; gap: 10px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.empty-state { text-align: center; padding: 60px 30px; color: #7dd3fc; }
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; display: block; }
.empty-state h3 { color: #bae6fd; font-size: 1.1rem; margin-bottom: 6px; }

.sidebar-logo { text-align: center; padding: 18px 0 8px; }
.sidebar-logo .logo-text {
    font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #06b6d4, #0ea5e9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: block; filter: drop-shadow(0 0 8px rgba(6,182,212,0.4));
}
.sidebar-logo .logo-sub { font-size: 0.68rem; color: #475569 !important; display: block; margin-top: 2px; letter-spacing: 1.5px; text-transform: uppercase; }

.stat-badge {
    display: flex; align-items: center; gap: 7px;
    background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.22);
    border-radius: 8px; padding: 7px 11px;
    color: #7dd3fc; font-size: 0.8rem; font-weight: 600;
    margin: 3px 0; width: 100%;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(2,11,24,0.7); backdrop-filter: blur(12px);
    border-radius: 12px; padding: 4px; gap: 3px;
    border: 1px solid rgba(6,182,212,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #7dd3fc !important;
    font-weight: 500 !important; padding: 7px 16px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important; box-shadow: 0 2px 12px rgba(6,182,212,0.35) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(6,182,212,0.06) !important;
    border: 1.5px dashed rgba(6,182,212,0.35) !important; border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span { color: #7dd3fc !important; font-weight: 500 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div small { color: #64748b !important; }

[data-testid="metric-container"] {
    background: rgba(6,182,212,0.07) !important; backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(6,182,212,0.2) !important; border-radius: 12px !important; padding: 14px !important;
}
[data-testid="metric-container"] label { color: #7dd3fc !important; font-size: 0.75rem !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #fff !important; font-weight: 700 !important; }

.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    border: 1px solid rgba(6,182,212,0.4) !important; border-radius: 10px !important;
    color: white !important; font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(6,182,212,0.25) !important; transition: all 0.25s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(6,182,212,0.4) !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(6,182,212,0.07) !important; border: 1px solid rgba(6,182,212,0.25) !important;
    border-radius: 10px !important; color: #e0f2fe !important;
}
[data-testid="stAlert"] {
    background: rgba(6,182,212,0.08) !important; border: 1px solid rgba(6,182,212,0.25) !important;
    border-radius: 10px !important; color: #e0f2fe !important;
}
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(2,11,24,0.8); }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#06b6d4,#0ea5e9); border-radius: 3px; }
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── CHART THEME ───────────────────────────────────────────────────────────────
CL = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#7dd3fc', size=11),
    xaxis=dict(gridcolor='rgba(6,182,212,0.1)', linecolor='rgba(6,182,212,0.2)', tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='rgba(6,182,212,0.1)', linecolor='rgba(6,182,212,0.2)', tickfont=dict(color='#64748b')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7dd3fc', size=11)),
    margin=dict(l=10, r=10, t=35, b=10), hovermode='x unified',
)
COLORS = ['#06b6d4','#0ea5e9','#10b981','#f59e0b','#8b5cf6','#ec4899','#38bdf8','#ef4444','#a78bfa','#34d399']

def ct(fig, title=''):
    fig.update_layout(**CL)
    if title: fig.update_layout(title=dict(text=title, font=dict(color='#e0f2fe', size=13)))
    return fig

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k in ['df','preprocessor','analyzer','forecaster','segmenter','detector',
          'forecast_df','segments_df','anomaly_df','model_type','forecast_periods',
          'forecast_metrics','comparison_df','date_min','date_max',
          'filter_cat','filter_region','df_filtered']:
    if k not in st.session_state: st.session_state[k] = None

# ── HELPERS ───────────────────────────────────────────────────────────────────
def dl(df, filename, text):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color:#06b6d4;font-weight:600;text-decoration:none;">⬇️ {text}</a>'

def card(title):
    st.markdown(f'<div class="glass-card"><p class="card-title">{title}</p>', unsafe_allow_html=True)

def endcard():
    st.markdown('</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)

def empty(icon, title, sub=''):
    st.markdown(f'<div class="empty-state"><span class="icon">{icon}</span><h3>{title}</h3><p>{sub}</p></div>', unsafe_allow_html=True)

def kpi(color, icon, value, label, delta=None, delta_up=True):
    d = ''
    if delta is not None:
        cls = 'up' if delta_up else ('down' if not delta_up else 'neu')
        arrow = '▲' if delta_up else '▼'
        d = f'<span class="kpi-delta {cls}">{arrow} {delta}</span>'
    st.markdown(f'''
    <div class="kpi-card {color}">
        <span class="kpi-icon">{icon}</span>
        <span class="kpi-value">{value}</span>
        <span class="kpi-label">{label}</span>
        {d}
    </div>''', unsafe_allow_html=True)

def get_df():
    return st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df

def process_uploaded_files(files):
    all_data = []
    for f in files:
        try:
            if f.name.endswith('.csv'):          df = pd.read_csv(f)
            elif f.name.endswith(('.xlsx','.xls')): df = pd.read_excel(f)
            elif f.name.endswith('.json'):       df = pd.read_json(f)
            else: st.error(f"Format tidak didukung: {f.name}"); continue
            all_data.append(df)
            st.success(f"✅ {f.name} ({len(df):,} baris)")
        except Exception as e: st.error(f"❌ {f.name}: {e}")
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        pp = DataPreprocessor()
        processed = pp.preprocess(combined)
        st.session_state.df          = processed
        st.session_state.preprocessor= pp
        st.session_state.analyzer    = SalesAnalyzer(processed)
        st.session_state.df_filtered = None
        missing = [c for c in ['date','revenue'] if c not in processed.columns]
        if missing: st.warning(f"⚠️ Kolom tidak terdeteksi: {missing}")
        else: st.success(f"🎉 {len(processed):,} records siap dianalisis!")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo"><span class="logo-text">⚡ SalesML</span><span class="logo-sub">Analytics Pro</span></div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("##### 📁 Upload Data")
        files = st.file_uploader("CSV / Excel / JSON", type=['csv','xlsx','xls','json'],
                                  accept_multiple_files=True, label_visibility='collapsed')
        if files:
            if st.button("🚀 Proses Data", type="primary", use_container_width=True):
                with st.spinner("Memproses..."): process_uploaded_files(files)

        st.markdown("")
        if st.button("📥 Load Sample Data", use_container_width=True):
            with st.spinner("Generate sample..."):
                df = create_sample_data(n_records=1000)
                st.session_state.df       = df
                st.session_state.analyzer = SalesAnalyzer(df)
                st.session_state.df_filtered = None
                st.success("Sample data loaded!")
                st.rerun()

        st.markdown("---")

        df = st.session_state.df
        if df is not None:
            st.markdown(f'<div class="stat-badge">📊 {len(df):,} records</div>', unsafe_allow_html=True)
            if 'date' in df.columns:
                dmin, dmax = df['date'].min().date(), df['date'].max().date()
                st.markdown(f'<div class="stat-badge">📅 {dmin} → {dmax}</div>', unsafe_allow_html=True)
            st.markdown("")

            # ── FILTERS ──
            st.markdown("##### 🔍 Filter Data")

            if 'date' in df.columns:
                d1, d2 = st.date_input("Rentang Tanggal",
                    value=[df['date'].min().date(), df['date'].max().date()],
                    min_value=df['date'].min().date(), max_value=df['date'].max().date())
            else: d1, d2 = None, None

            cats = sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else []
            sel_cat = st.multiselect("Kategori", cats, placeholder="Semua kategori")

            regions = sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else []
            sel_reg = st.multiselect("Wilayah", regions, placeholder="Semua wilayah")

            channels = sorted(df['channel'].dropna().unique().tolist()) if 'channel' in df.columns else []
            sel_ch = st.multiselect("Channel", channels, placeholder="Semua channel")

            if st.button("🔎 Terapkan Filter", use_container_width=True):
                filtered = df.copy()
                if d1 and d2 and 'date' in df.columns:
                    filtered = filtered[(filtered['date'].dt.date >= d1) & (filtered['date'].dt.date <= d2)]
                if sel_cat and 'category' in df.columns:
                    filtered = filtered[filtered['category'].isin(sel_cat)]
                if sel_reg and 'region' in df.columns:
                    filtered = filtered[filtered['region'].isin(sel_reg)]
                if sel_ch and 'channel' in df.columns:
                    filtered = filtered[filtered['channel'].isin(sel_ch)]
                st.session_state.df_filtered = filtered
                st.session_state.analyzer    = SalesAnalyzer(filtered)
                st.success(f"Filter diterapkan: {len(filtered):,} records")

            if st.button("❌ Reset Filter", use_container_width=True):
                st.session_state.df_filtered = None
                st.session_state.analyzer    = SalesAnalyzer(df)
                st.rerun()

        st.markdown("---")
        st.markdown("##### ⚙️ Model Settings")
        st.session_state.model_type      = st.selectbox("Forecasting Model",
            ['gradient_boosting','random_forest','extra_trees','ensemble','xgboost','lightgbm','ridge','linear'])
        st.session_state.forecast_periods = st.slider("Forecast (hari)", 7, 180, 30)

        st.markdown("---")
        st.markdown('<div style="color:#334155;font-size:0.72rem;text-align:center;padding:6px 0">SalesML Analytics Pro v3.0<br><span style="color:#0ea5e9">Powered by Irvan_signora</span></div>', unsafe_allow_html=True)

# ── TAB 1: KPI OVERVIEW ───────────────────────────────────────────────────────
def tab_kpi():
    st.markdown('<div class="hero-header"><p class="hero-title">📊 CEO Dashboard</p><p class="hero-sub">Ringkasan performa bisnis secara real-time</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("📂","Belum ada data","Upload file atau load sample data")

    rev = df['revenue'].sum() if 'revenue' in df.columns else 0
    qty = df['quantity'].sum() if 'quantity' in df.columns else 0
    txn = len(df)
    aov = rev / txn if txn else 0

    # Estimate profit (pakai margin 30% jika tidak ada cost)
    has_cost = 'cost' in df.columns or 'hpp' in df.columns
    cost_col = 'cost' if 'cost' in df.columns else ('hpp' if 'hpp' in df.columns else None)
    if cost_col:
        gross_profit = rev - df[cost_col].sum()
    else:
        gross_profit = rev * 0.30  # estimasi 30% margin
    margin_pct = (gross_profit / rev * 100) if rev else 0

    # MoM Growth
    mom, yoy = None, None
    if 'date' in df.columns and 'revenue' in df.columns:
        monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
        if len(monthly) >= 2:
            mom = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100) if monthly.iloc[-2] else 0
        if len(monthly) >= 13:
            yoy = ((monthly.iloc[-1] - monthly.iloc[-13]) / monthly.iloc[-13] * 100) if monthly.iloc[-13] else 0

    # ── ROW 1: 4 KPI ──
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi('c1','💰', format_currency(rev), 'Total Revenue',
                 f"{mom:.1f}% vs bln lalu" if mom is not None else None, mom >= 0 if mom else True)
    with c2: kpi('c2','📦', format_number(qty), 'Total Qty Terjual')
    with c3: kpi('c3','🧾', format_number(txn), 'Total Transaksi')
    with c4: kpi('c4','🛒', format_currency(aov), 'Avg Order Value (AOV)')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 2: 4 KPI ──
    c5,c6,c7,c8 = st.columns(4)
    with c5: kpi('c5','💎', format_currency(gross_profit), 'Gross Profit',
                 "Estimasi 30%" if not cost_col else None, True)
    with c6: kpi('c6','📈', f"{margin_pct:.1f}%", 'Profit Margin',
                 f"{margin_pct:.1f}%" if margin_pct else None, margin_pct >= 20)
    with c7: kpi('c7','📅', f"{mom:+.1f}%" if mom is not None else "N/A", 'Growth MoM',
                 None, mom >= 0 if mom is not None else True)
    with c8: kpi('c8','🗓️', f"{yoy:+.1f}%" if yoy is not None else "N/A", 'Growth YoY',
                 None, yoy >= 0 if yoy is not None else True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue Trend ──
    section("📈 Tren Penjualan")
    if 'date' in df.columns:
        freq_opt = st.radio("Granularitas", ["Harian","Mingguan","Bulanan"], horizontal=True, label_visibility='collapsed')
        freq_map = {"Harian":"D","Mingguan":"W","Bulanan":"M"}
        freq = freq_map[freq_opt]

        daily = df.groupby(df['date'].dt.to_period(freq))['revenue'].sum().reset_index()
        daily['date'] = daily['date'].dt.to_timestamp()
        daily['ma7']  = daily['revenue'].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily['date'], y=daily['revenue'], name='Revenue',
                             marker_color='rgba(6,182,212,0.6)'))
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['ma7'], name='Moving Avg',
                                 line=dict(color='#f59e0b', width=2), mode='lines'))
        ct(fig); st.plotly_chart(fig, width='stretch')

    # ── Key Insights ──
    section("💡 Key Insights")
    insights = SalesAnalyzer(df).generate_insights()
    cols = st.columns(2)
    for i, ins in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

# ── TAB 2: SALES PERFORMANCE ──────────────────────────────────────────────────
def tab_sales():
    st.markdown('<div class="hero-header"><p class="hero-title">📊 Sales Performance</p><p class="hero-sub">Top & bottom produk, salesperson, channel</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("📊","Belum ada data")
    if 'revenue' not in df.columns: return empty("⚠️","Kolom revenue tidak ditemukan")

    # ── Top & Bottom Products ──
    section("🔝 Top & Bottom Produk")
    if 'product' in df.columns:
        prod = df.groupby('product').agg(
            revenue=('revenue','sum'),
            qty=('quantity','sum') if 'quantity' in df.columns else ('revenue','count'),
            txn=('revenue','count')
        ).reset_index().sort_values('revenue', ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><p class="card-title">🏆 Top 10 by Revenue</p>', unsafe_allow_html=True)
            top10 = prod.head(10)
            fig = go.Figure(go.Bar(x=top10['revenue'], y=top10['product'], orientation='h',
                                   marker=dict(color=top10['revenue'],
                                               colorscale=[[0,'#0c4a6e'],[1,'#38bdf8']])))
            fig.update_layout(yaxis=dict(autorange='reversed'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><p class="card-title">🔻 Bottom 10 (Perlu Perhatian)</p>', unsafe_allow_html=True)
            bot10 = prod.tail(10).sort_values('revenue')
            fig = go.Figure(go.Bar(x=bot10['revenue'], y=bot10['product'], orientation='h',
                                   marker=dict(color=bot10['revenue'],
                                               colorscale=[[0,'#ef4444'],[1,'#f97316']])))
            fig.update_layout(yaxis=dict(autorange='reversed'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        # Top by Qty
        if 'quantity' in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="glass-card"><p class="card-title">📦 Top 10 by Quantity</p>', unsafe_allow_html=True)
                top_qty = prod.sort_values('qty', ascending=False).head(10)
                fig = go.Figure(go.Bar(x=top_qty['qty'], y=top_qty['product'], orientation='h',
                                       marker_color='rgba(16,185,129,0.7)'))
                fig.update_layout(yaxis=dict(autorange='reversed'))
                ct(fig); st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="glass-card"><p class="card-title">🔄 Revenue vs Quantity Scatter</p>', unsafe_allow_html=True)
                fig = px.scatter(prod.head(30), x='qty', y='revenue', text='product',
                                 color='revenue', color_continuous_scale='Blues',
                                 size='revenue', size_max=30)
                fig.update_traces(textposition='top center', textfont=dict(color='#7dd3fc', size=9))
                ct(fig); st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Top Salesperson ──
    section("👤 Top Salesperson")
    sales_col = next((c for c in ['salesperson','sales_person','sales','nama_sales',
                                   'sales_name','agen','agent','pic'] if c in df.columns), None)
    if sales_col:
        sp = df.groupby(sales_col).agg(revenue=('revenue','sum'), txn=('revenue','count')).reset_index()
        sp = sp.sort_values('revenue', ascending=False).head(15)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><p class="card-title">🏅 Top Salesperson by Revenue</p>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=sp['revenue'], y=sp[sales_col], orientation='h',
                                   marker=dict(color=sp['revenue'],
                                               colorscale=[[0,'#0c4a6e'],[1,'#06b6d4']])))
            fig.update_layout(yaxis=dict(autorange='reversed'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card"><p class="card-title">📋 Detail Salesperson</p>', unsafe_allow_html=True)
            sp['revenue_fmt'] = sp['revenue'].apply(format_currency)
            st.dataframe(sp[[sales_col,'revenue_fmt','txn']].rename(columns={sales_col:'Salesperson','revenue_fmt':'Revenue','txn':'Transaksi'}),
                         use_container_width=True, height=280)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Gunakan channel/region sebagai proxy
        proxy = next((c for c in ['channel','store','courier'] if c in df.columns), None)
        if proxy:
            section(f"📡 Performa per {proxy.title()}")
            sp = df.groupby(proxy)['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="glass-card"><p class="card-title">📊 Revenue per Channel/Toko</p>', unsafe_allow_html=True)
                fig = go.Figure(go.Bar(x=sp['revenue'], y=sp[proxy], orientation='h',
                                       marker=dict(color=COLORS[:len(sp)])))
                fig.update_layout(yaxis=dict(autorange='reversed'))
                ct(fig); st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="glass-card"><p class="card-title">🥧 Distribusi Revenue</p>', unsafe_allow_html=True)
                fig = go.Figure(go.Pie(labels=sp[proxy], values=sp['revenue'], hole=0.5,
                                       marker=dict(colors=COLORS)))
                ct(fig); st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("ℹ️ Kolom salesperson/channel tidak ditemukan di dataset")

# ── TAB 3: PROFITABILITY ──────────────────────────────────────────────────────
def tab_profit():
    st.markdown('<div class="hero-header"><p class="hero-title">💰 Profitability Analysis</p><p class="hero-sub">Margin, profit per produk, revenue vs profit</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("💰","Belum ada data")
    if 'revenue' not in df.columns: return empty("⚠️","Kolom revenue tidak ditemukan")

    # Estimate profit
    cost_col = next((c for c in ['cost','hpp','harga_pokok','cogs'] if c in df.columns), None)
    df2 = df.copy()
    if cost_col:
        df2['profit'] = df2['revenue'] - df2[cost_col]
        df2['margin'] = df2['profit'] / df2['revenue'] * 100
        st.info(f"✅ Menggunakan kolom **{cost_col}** untuk kalkulasi profit")
    else:
        df2['profit'] = df2['revenue'] * 0.30
        df2['margin'] = 30.0
        st.warning("⚠️ Kolom HPP/Cost tidak ditemukan. Profit diestimasi 30% dari revenue.")

    # ── Summary ──
    total_rev  = df2['revenue'].sum()
    total_prof = df2['profit'].sum()
    avg_margin = df2['margin'].mean() if cost_col else 30.0

    c1,c2,c3 = st.columns(3)
    with c1: kpi('c1','💰', format_currency(total_rev), 'Total Revenue')
    with c2: kpi('c2','💎', format_currency(total_prof), 'Gross Profit')
    with c3: kpi('c3','📈', f"{avg_margin:.1f}%", 'Avg Margin')

    st.markdown("<br>", unsafe_allow_html=True)

    if 'product' in df2.columns:
        prod_p = df2.groupby('product').agg(
            revenue=('revenue','sum'),
            profit=('profit','sum'),
            txn=('revenue','count')
        ).reset_index()
        prod_p['margin_pct'] = prod_p['profit'] / prod_p['revenue'] * 100

        section("📊 Profit per Produk")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><p class="card-title">🏆 Top 10 Produk by Margin %</p>', unsafe_allow_html=True)
            top_m = prod_p.sort_values('margin_pct', ascending=False).head(10)
            fig = go.Figure(go.Bar(x=top_m['margin_pct'], y=top_m['product'], orientation='h',
                                   marker=dict(color=top_m['margin_pct'],
                                               colorscale=[[0,'#064e3b'],[1,'#10b981']])))
            fig.update_layout(yaxis=dict(autorange='reversed'),
                              xaxis=dict(ticksuffix='%'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><p class="card-title">💸 Produk Margin Terendah (Rugi?)</p>', unsafe_allow_html=True)
            bot_m = prod_p.sort_values('margin_pct').head(10)
            colors_bar = ['#ef4444' if x < 0 else '#f97316' if x < 10 else '#f59e0b' for x in bot_m['margin_pct']]
            fig = go.Figure(go.Bar(x=bot_m['margin_pct'], y=bot_m['product'], orientation='h',
                                   marker_color=colors_bar))
            fig.update_layout(yaxis=dict(autorange='reversed'), xaxis=dict(ticksuffix='%'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        # Revenue vs Profit scatter
        section("🔍 Revenue vs Profit Analysis")
        st.markdown('<div class="glass-card"><p class="card-title">⚡ Scatter: Revenue vs Profit (ukuran = transaksi)</p>', unsafe_allow_html=True)
        fig = px.scatter(prod_p, x='revenue', y='profit', size='txn', text='product',
                         color='margin_pct', color_continuous_scale='RdYlGn',
                         hover_data={'revenue': ':,.0f','profit': ':,.0f','margin_pct': ':.1f'})
        fig.update_traces(textposition='top center', textfont=dict(size=9, color='#7dd3fc'))
        fig.add_hline(y=0, line_color='rgba(239,68,68,0.5)', line_dash='dash',
                      annotation_text="Break Even", annotation_font_color='#ef4444')
        ct(fig); st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        # Table
        st.markdown('<div class="glass-card"><p class="card-title">📋 Detail Profitabilitas per Produk</p>', unsafe_allow_html=True)
        tbl = prod_p.copy()
        tbl['Revenue']    = tbl['revenue'].apply(format_currency)
        tbl['Profit']     = tbl['profit'].apply(format_currency)
        tbl['Margin %']   = tbl['margin_pct'].apply(lambda x: f"{x:.1f}%")
        tbl['Transaksi']  = tbl['txn']
        st.dataframe(tbl[['product','Revenue','Profit','Margin %','Transaksi']]
                     .rename(columns={'product':'Produk'})
                     .sort_values('Margin %', ascending=False),
                     use_container_width=True, height=300)
        st.markdown(dl(prod_p, 'profitability.csv', 'Download CSV'), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: CUSTOMER ANALYSIS ──────────────────────────────────────────────────
def tab_customer():
    st.markdown('<div class="hero-header"><p class="hero-title">👥 Customer Analysis</p><p class="hero-sub">RFM, CLV, top customer, segmentasi</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("👥","Belum ada data")

    cust_col = next((c for c in ['customer','pelanggan','customer_name','nama_pelanggan','buyer','pembeli'] if c in df.columns), None)
    if not cust_col:
        return empty("👥","Kolom customer tidak ditemukan","Pastikan ada kolom: customer, pelanggan, buyer, dll")

    # ── RFM Analysis ──
    section("🎯 RFM Analysis")
    if 'date' in df.columns and 'revenue' in df.columns:
        today = df['date'].max()
        rfm = df.groupby(cust_col).agg(
            recency  =('date', lambda x: (today - x.max()).days),
            frequency=('revenue', 'count'),
            monetary =('revenue', 'sum')
        ).reset_index()

        # Score 1-5
        for col, asc in [('recency', True), ('frequency', False), ('monetary', False)]:
            try:
                rfm[f'{col}_score'] = pd.qcut(rfm[col], q=5, labels=[5,4,3,2,1] if asc else [1,2,3,4,5], duplicates='drop')
            except: rfm[f'{col}_score'] = 3

        rfm['rfm_score'] = (rfm['recency_score'].astype(int) +
                            rfm['frequency_score'].astype(int) +
                            rfm['monetary_score'].astype(int))

        def segment(s):
            if s >= 13: return '⭐ Champions'
            elif s >= 10: return '🔄 Loyal'
            elif s >= 7:  return '🌱 Potential'
            elif s >= 4:  return '😴 At Risk'
            else:          return '💀 Lost'

        rfm['segment'] = rfm['rfm_score'].apply(segment)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><p class="card-title">🎯 Distribusi Segmen RFM</p>', unsafe_allow_html=True)
            seg_count = rfm['segment'].value_counts()
            fig = go.Figure(go.Pie(labels=seg_count.index, values=seg_count.values,
                                   hole=0.5, marker=dict(colors=COLORS)))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><p class="card-title">💰 Monetary per Segmen</p>', unsafe_allow_html=True)
            seg_mon = rfm.groupby('segment')['monetary'].sum().reset_index().sort_values('monetary', ascending=False)
            fig = go.Figure(go.Bar(x=seg_mon['segment'], y=seg_mon['monetary'],
                                   marker=dict(color=COLORS[:len(seg_mon)])))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        section("🏆 Top Customer")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><p class="card-title">💎 Top 15 Customer by Revenue (CLV)</p>', unsafe_allow_html=True)
            top_c = rfm.sort_values('monetary', ascending=False).head(15)
            fig = go.Figure(go.Bar(x=top_c['monetary'], y=top_c[cust_col], orientation='h',
                                   marker=dict(color=top_c['monetary'],
                                               colorscale=[[0,'#0c4a6e'],[1,'#06b6d4']])))
            fig.update_layout(yaxis=dict(autorange='reversed'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><p class="card-title">🔁 Top 15 Repeat Customer (Frequency)</p>', unsafe_allow_html=True)
            top_f = rfm.sort_values('frequency', ascending=False).head(15)
            fig = go.Figure(go.Bar(x=top_f['frequency'], y=top_f[cust_col], orientation='h',
                                   marker_color='rgba(16,185,129,0.7)'))
            fig.update_layout(yaxis=dict(autorange='reversed'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><p class="card-title">📋 Detail RFM</p>', unsafe_allow_html=True)
        rfm_show = rfm.copy()
        rfm_show['monetary'] = rfm_show['monetary'].apply(format_currency)
        st.dataframe(rfm_show[[cust_col,'segment','recency','frequency','monetary','rfm_score']]
                     .rename(columns={cust_col:'Customer','recency':'Recency (hari)',
                                      'frequency':'Frequency','monetary':'Monetary','rfm_score':'Score'})
                     .sort_values('Score', ascending=False),
                     use_container_width=True, height=300)
        st.markdown(dl(rfm, 'rfm_analysis.csv', 'Download RFM CSV'), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 5: REGIONAL ───────────────────────────────────────────────────────────
def tab_regional():
    st.markdown('<div class="hero-header"><p class="hero-title">📍 Regional Analysis</p><p class="hero-sub">Performa per wilayah, kota, dan channel</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("📍","Belum ada data")

    region_col = next((c for c in ['region','lokasi','kota','city','wilayah','area','cabang'] if c in df.columns), None)
    if not region_col:
        return empty("📍","Kolom wilayah tidak ditemukan","Pastikan ada kolom: region, lokasi, kota, wilayah, dll")

    reg = df.groupby(region_col).agg(
        revenue=('revenue','sum'),
        txn=('revenue','count'),
        qty=('quantity','sum') if 'quantity' in df.columns else ('revenue','count')
    ).reset_index().sort_values('revenue', ascending=False)

    # KPI
    c1,c2,c3 = st.columns(3)
    with c1: kpi('c1','🗺️', str(reg[region_col].nunique()), 'Total Wilayah')
    with c2: kpi('c2','🏆', str(reg.iloc[0][region_col]), 'Wilayah Terbaik')
    with c3: kpi('c3','📉', str(reg.iloc[-1][region_col]), 'Wilayah Terendah')

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><p class="card-title">🏆 Revenue per Wilayah</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=reg['revenue'], y=reg[region_col], orientation='h',
                               marker=dict(color=reg['revenue'],
                                           colorscale=[[0,'#0c4a6e'],[1,'#38bdf8']])))
        fig.update_layout(yaxis=dict(autorange='reversed'))
        ct(fig); st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card"><p class="card-title">🥧 Kontribusi Wilayah</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(labels=reg[region_col], values=reg['revenue'],
                               hole=0.5, marker=dict(colors=COLORS)))
        ct(fig); st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # Growth per wilayah (jika ada date)
    if 'date' in df.columns:
        section("📈 Growth Tren per Wilayah")
        st.markdown('<div class="glass-card"><p class="card-title">📈 Tren Revenue per Wilayah (Bulanan)</p>', unsafe_allow_html=True)
        monthly_reg = df.groupby([df['date'].dt.to_period('M'), region_col])['revenue'].sum().reset_index()
        monthly_reg['date'] = monthly_reg['date'].dt.to_timestamp()
        top_regions = reg[region_col].head(6).tolist()
        fig = go.Figure()
        for i, r in enumerate(top_regions):
            d = monthly_reg[monthly_reg[region_col] == r]
            fig.add_trace(go.Scatter(x=d['date'], y=d['revenue'], name=r,
                                     line=dict(color=COLORS[i], width=2), mode='lines+markers'))
        ct(fig); st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # Table
    st.markdown('<div class="glass-card"><p class="card-title">📋 Detail per Wilayah</p>', unsafe_allow_html=True)
    tbl = reg.copy()
    tbl['Revenue'] = tbl['revenue'].apply(format_currency)
    tbl['Share %'] = (tbl['revenue'] / tbl['revenue'].sum() * 100).apply(lambda x: f"{x:.1f}%")
    st.dataframe(tbl[[region_col,'Revenue','txn','Share %']].rename(
        columns={region_col:'Wilayah','txn':'Transaksi'}), use_container_width=True, height=280)
    st.markdown(dl(reg, 'regional.csv', 'Download CSV'), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 6: CATEGORY & PARETO ─────────────────────────────────────────────────
def tab_category():
    st.markdown('<div class="hero-header"><p class="hero-title">🎯 Category & Product Mix</p><p class="hero-sub">Pareto 80/20, kontribusi kategori, slow-moving</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("🎯","Belum ada data")
    if 'revenue' not in df.columns: return empty("⚠️","Kolom revenue tidak ditemukan")

    # ── Category Distribution ──
    if 'category' in df.columns:
        cat = df.groupby('category').agg(revenue=('revenue','sum'), txn=('revenue','count')).reset_index()
        cat['share'] = cat['revenue'] / cat['revenue'].sum() * 100
        cat = cat.sort_values('revenue', ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><p class="card-title">🗂️ Revenue per Kategori</p>', unsafe_allow_html=True)
            fig = go.Figure(go.Pie(labels=cat['category'], values=cat['revenue'], hole=0.5,
                                   marker=dict(colors=COLORS), textfont=dict(color='white')))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><p class="card-title">📊 Margin per Kategori</p>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=cat['category'], y=cat['share'],
                                   marker=dict(color=COLORS[:len(cat)])))
            fig.update_layout(yaxis=dict(ticksuffix='%'))
            ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Pareto 80/20 ──
    section("📐 Pareto Analysis (80/20 Rule)")
    if 'product' in df.columns:
        st.markdown('<div class="glass-card"><p class="card-title">📐 Produk mana yang hasilkan 80% Revenue?</p>', unsafe_allow_html=True)
        prod = df.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
        prod['cumsum']  = prod['revenue'].cumsum()
        prod['cumpct']  = prod['cumsum'] / prod['revenue'].sum() * 100
        prod['rank']    = range(1, len(prod)+1)

        # 80% cutoff
        cut80 = prod[prod['cumpct'] <= 80]
        pct_products = len(cut80) / len(prod) * 100

        st.markdown(f'<div class="insight-card">📌 <strong>{len(cut80)} produk ({pct_products:.0f}%)</strong> menghasilkan 80% dari total revenue — prinsip Pareto terbukti!</div>', unsafe_allow_html=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=prod['product'].head(30), y=prod['revenue'].head(30),
                             name='Revenue', marker_color='rgba(6,182,212,0.7)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=prod['product'].head(30), y=prod['cumpct'].head(30),
                                 name='Kumulatif %', line=dict(color='#f59e0b', width=2)), secondary_y=True)
        fig.add_hline(y=80, line_color='rgba(239,68,68,0.6)', line_dash='dash',
                      annotation_text="80%", secondary_y=True)
        fig.update_layout(yaxis2=dict(ticksuffix='%', range=[0,105]))
        ct(fig); st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        # Slow-moving
        section("🐌 Produk Slow-Moving")
        st.markdown('<div class="glass-card"><p class="card-title">⚠️ Produk dengan Revenue Terendah (Perlu Evaluasi)</p>', unsafe_allow_html=True)
        slow = prod.tail(15).sort_values('revenue')
        slow['share'] = slow['revenue'] / prod['revenue'].sum() * 100
        slow['Revenue'] = slow['revenue'].apply(format_currency)
        slow['Share %'] = slow['share'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(slow[['product','Revenue','Share %']].rename(columns={'product':'Produk'}),
                     use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 7: ANOMALY ────────────────────────────────────────────────────────────
def tab_anomaly():
    st.markdown('<div class="hero-header"><p class="hero-title">🚨 Anomaly Detection</p><p class="hero-sub">Deteksi transaksi & penurunan tidak wajar</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("🚨","Belum ada data")

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div class="glass-card"><p class="card-title">⚙️ Settings</p>', unsafe_allow_html=True)
        method = st.selectbox("Metode", ['isolation_forest','zscore'])
        contamination = st.slider("Expected Outlier %", 1, 20, 5) / 100
        if st.button("🔍 Deteksi Anomali", type="primary", use_container_width=True):
            with st.spinner("Mendeteksi..."):
                try:
                    det = AnomalyDetector(method=method, contamination=contamination)
                    anom_df = det.fit(df)
                    st.session_state.detector   = det
                    st.session_state.anomaly_df = anom_df
                    n = int(anom_df['anomaly'].sum())
                    st.success(f"✅ {n} anomali ({n/len(anom_df)*100:.1f}%)")
                except Exception as e: st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if st.session_state.anomaly_df is not None:
            adf = st.session_state.anomaly_df
            normal, anoms = adf[adf['anomaly']==0], adf[adf['anomaly']==1]

            c_a, c_b, c_c = st.columns(3)
            with c_a: st.metric("Total Data", f"{len(adf):,}")
            with c_b: st.metric("Normal", f"{len(normal):,}")
            with c_c: st.metric("🚨 Anomali", f"{len(anoms):,}")

            st.markdown('<div class="glass-card"><p class="card-title">📈 Visualisasi Anomali</p>', unsafe_allow_html=True)
            if 'date' in adf.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=normal['date'], y=normal['revenue'], mode='markers',
                                         name='Normal', marker=dict(color='#0ea5e9', size=4, opacity=0.5)))
                fig.add_trace(go.Scatter(x=anoms['date'], y=anoms['revenue'], mode='markers',
                                         name='🚨 Anomali', marker=dict(color='#ef4444', size=9, symbol='x')))
                ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

            if not anoms.empty:
                st.markdown('<div class="glass-card"><p class="card-title">📋 Daftar Anomali</p>', unsafe_allow_html=True)
                cols = [c for c in ['date','product','revenue','quantity','anomaly_score'] if c in anoms.columns]
                st.dataframe(anoms[cols].sort_values('anomaly_score', ascending=False),
                             use_container_width=True, height=260)
                st.markdown(dl(anoms, 'anomalies.csv', 'Download CSV'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty("🚨","Belum ada hasil","Klik Deteksi Anomali")

# ── TAB 8: FORECAST ───────────────────────────────────────────────────────────
def tab_forecast():
    st.markdown('<div class="hero-header"><p class="hero-title">🔮 Sales Forecast</p><p class="hero-sub">Prediksi revenue 1–6 bulan ke depan dengan ML</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("🔮","Belum ada data")

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div class="glass-card"><p class="card-title">⚙️ Settings</p>', unsafe_allow_html=True)
        model_type = st.selectbox("Model", ['gradient_boosting','random_forest','extra_trees','ensemble','xgboost','lightgbm','ridge','linear'])
        periods    = st.slider("Periode Forecast (hari)", 7, 180, 30)
        do_tuning  = st.checkbox("Hyperparameter Tuning", False)

        if st.button("🚀 Train & Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_type}..."):
                try:
                    fc = SalesForecaster(model_type=model_type)
                    m  = fc.fit(df, do_tuning=do_tuning)
                    st.session_state.forecaster      = fc
                    st.session_state.forecast_metrics= m
                    st.session_state.forecast_df     = fc.forecast_future(df, periods=periods)
                    st.success("✅ Selesai!")
                except Exception as e: st.error(f"❌ {e}")

        if st.session_state.forecast_metrics:
            m = st.session_state.forecast_metrics
            st.markdown("---")
            st.metric("R² Score", f"{m.get('test_r2',0):.4f}")
            st.metric("RMSE",     f"{format_currency(m.get('test_rmse',0))}")
            st.metric("MAPE",     f"{m.get('test_mape',0):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if st.session_state.forecast_df is not None:
            fc_df = st.session_state.forecast_df
            st.markdown('<div class="glass-card"><p class="card-title">📈 Forecast vs Historical</p>', unsafe_allow_html=True)
            if 'date' in df.columns:
                daily = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily['date'], y=daily['revenue'],
                                         fill='tozeroy', mode='lines', name='Historis',
                                         line=dict(color='#06b6d4', width=2),
                                         fillcolor='rgba(6,182,212,0.1)'))
                if 'forecast' in fc_df.columns:
                    fig.add_trace(go.Scatter(x=fc_df['date'], y=fc_df['forecast'],
                                             mode='lines', name='Forecast',
                                             line=dict(color='#f59e0b', width=2, dash='dash')))
                ct(fig); st.plotly_chart(fig, width='stretch')
            st.markdown(dl(fc_df, 'forecast.csv', 'Download Forecast CSV'), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature importance
            if st.session_state.forecaster:
                imp = st.session_state.forecaster.get_feature_importance()
                if not imp.empty and imp['importance'].sum() > 0:
                    st.markdown('<div class="glass-card"><p class="card-title">🔍 Feature Importance</p>', unsafe_allow_html=True)
                    top = imp.head(12)
                    fig = go.Figure(go.Bar(x=top['importance'], y=top['feature'], orientation='h',
                                          marker=dict(color=top['importance'],
                                                      colorscale=[[0,'#0c4a6e'],[1,'#38bdf8']])))
                    fig.update_layout(yaxis=dict(autorange='reversed'))
                    ct(fig); st.plotly_chart(fig, width='stretch')
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty("🔮","Belum ada forecast","Klik Train & Forecast")

# ── TAB 9: MODEL COMPARISON ───────────────────────────────────────────────────
def tab_models():
    st.markdown('<div class="hero-header"><p class="hero-title">⚖️ Model Comparison</p><p class="hero-sub">Bandingkan semua ML model, temukan yang terbaik</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("⚖️","Belum ada data")

    if st.button("🚀 Bandingkan Semua Model", type="primary"):
        with st.spinner("Training semua model... (3–5 menit)"):
            try:
                comp = ModelComparator()
                comp_df = comp.compare_models(df)
                st.session_state.comparison_df = comp_df
                if comp.best_model_type:
                    st.success(f"🏆 Model terbaik: **{comp.best_model_type}**")
            except Exception as e: st.error(f"❌ {e}")

    if st.session_state.comparison_df is not None:
        cdf = st.session_state.comparison_df
        st.markdown('<div class="glass-card"><p class="card-title">📊 Hasil Perbandingan</p>', unsafe_allow_html=True)
        st.dataframe(cdf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if 'test_rmse' in cdf.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="glass-card"><p class="card-title">📉 RMSE (lebih kecil = lebih baik)</p>', unsafe_allow_html=True)
                d = cdf['test_rmse'].dropna().astype(float).sort_values()
                fig = go.Figure(go.Bar(x=d.index, y=d.values,
                                       marker=dict(color=d.values, colorscale=[[0,'#06b6d4'],[1,'#ef4444']])))
                ct(fig); st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                if 'test_r2' in cdf.columns:
                    st.markdown('<div class="glass-card"><p class="card-title">📈 R² Score (lebih besar = lebih baik)</p>', unsafe_allow_html=True)
                    d2 = cdf['test_r2'].dropna().astype(float).sort_values(ascending=False)
                    fig = go.Figure(go.Bar(x=d2.index, y=d2.values,
                                          marker=dict(color=d2.values, colorscale=[[0,'#ef4444'],[1,'#10b981']])))
                    ct(fig); st.plotly_chart(fig, width='stretch')
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        empty("⚖️","Belum ada perbandingan","Klik tombol di atas")

# ── TAB 10: REPORTS ───────────────────────────────────────────────────────────
def tab_reports():
    st.markdown('<div class="hero-header"><p class="hero-title">📑 Reports & Export</p><p class="hero-sub">Download semua hasil analisis</p></div>', unsafe_allow_html=True)
    df = get_df()
    if df is None: return empty("📑","Belum ada data")
    analyzer = SalesAnalyzer(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="glass-card"><p class="card-title">📄 PDF Report</p>', unsafe_allow_html=True)
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Generating..."):
                try:
                    Path('reports').mkdir(exist_ok=True)
                    r = ReportGenerator(analyzer)
                    r.generate_pdf_report('reports/sales_report.pdf',
                        st.session_state.forecast_df, st.session_state.segments_df, st.session_state.anomaly_df)
                    with open('reports/sales_report.pdf','rb') as f:
                        st.download_button("⬇️ Download PDF", f.read(), 'sales_report.pdf', 'application/pdf', use_container_width=True)
                except Exception as e: st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card"><p class="card-title">📊 Excel Report</p>', unsafe_allow_html=True)
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Generating..."):
                try:
                    Path('reports').mkdir(exist_ok=True)
                    r = ReportGenerator(analyzer)
                    r.export_to_excel('reports/sales_analysis.xlsx',
                        st.session_state.forecast_df, st.session_state.segments_df, st.session_state.anomaly_df)
                    with open('reports/sales_analysis.xlsx','rb') as f:
                        st.download_button("⬇️ Download Excel", f.read(), 'sales_analysis.xlsx',
                            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
                except Exception as e: st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="glass-card"><p class="card-title">📁 CSV Export</p>', unsafe_allow_html=True)
        if st.button("Export CSV ZIP", use_container_width=True):
            with st.spinner("Exporting..."):
                try:
                    import zipfile
                    Path('reports').mkdir(exist_ok=True)
                    r = ReportGenerator(analyzer)
                    r.export_to_csv('reports')
                    zp = 'reports/csv_export.zip'
                    with zipfile.ZipFile(zp,'w') as zf:
                        for csv in Path('reports').glob('*.csv'): zf.write(csv, csv.name)
                    with open(zp,'rb') as f:
                        st.download_button("⬇️ Download ZIP", f.read(), 'csv_export.zip', 'application/zip', use_container_width=True)
                except Exception as e: st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Raw data preview
    section("🔍 Raw Data Preview")
    st.markdown('<div class="glass-card"><p class="card-title">📋 Data (filtered)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(500), use_container_width=True, height=350)
    st.markdown(dl(df, 'data_export.csv', 'Download Full Data CSV'), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    tabs = st.tabs([
        "📊 KPI Overview",
        "📈 Sales Performance",
        "💰 Profitability",
        "👥 Customer & RFM",
        "📍 Regional",
        "🎯 Category & Pareto",
        "🚨 Anomaly",
        "🔮 Forecast",
        "⚖️ Model Comparison",
        "📑 Reports"
    ])
    fns = [tab_kpi, tab_sales, tab_profit, tab_customer,
           tab_regional, tab_category, tab_anomaly, tab_forecast,
           tab_models, tab_reports]
    for tab, fn in zip(tabs, fns):
        with tab: fn()

if __name__ == "__main__":
    main()
