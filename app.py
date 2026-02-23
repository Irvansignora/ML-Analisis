"""
Streamlit Dashboard for Sales ML System
=======================================
Dashboard interaktif untuk analisis data penjualan - Modern UI
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
from utils import SalesAnalyzer, ReportGenerator, Visualizer, format_currency, format_number, create_sample_data

st.set_page_config(
    page_title="Sales ML Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MODERN CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background: #0f1117; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #0f1117 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(99,102,241,0.3);
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -30%; left: 60%;
    width: 250px; height: 250px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -1px;
    position: relative; z-index: 1;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.95);
    margin: 8px 0 0 0;
    font-weight: 400;
    position: relative; z-index: 1;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.purple::before { background: linear-gradient(90deg,#6366f1,#8b5cf6); }
.kpi-card.cyan::before   { background: linear-gradient(90deg,#06b6d4,#0ea5e9); }
.kpi-card.green::before  { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.orange::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.pink::before   { background: linear-gradient(90deg,#ec4899,#f472b6); }
.kpi-icon {
    font-size: 1.8rem;
    margin-bottom: 10px;
    display: block;
}
.kpi-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    display: block;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.78rem;
    color: #cbd5e1;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
    display: block;
}
.kpi-delta {
    font-size: 0.8rem;
    margin-top: 8px;
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
}
.kpi-delta.up   { background:rgba(16,185,129,0.15); color:#34d399; }
.kpi-delta.down { background:rgba(239,68,68,0.15);  color:#f87171; }

/* ── Section Cards ── */
.section-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 20px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Insight Cards ── */
.insight-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #e2e8f0;
    font-size: 0.92rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.insight-card::before { content: '💡'; font-size: 1rem; flex-shrink: 0; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: #94a3b8;
}
.empty-state .icon { font-size: 4rem; margin-bottom: 16px; display: block; }
.empty-state h3 { color: #cbd5e1; font-size: 1.3rem; margin-bottom: 8px; }

/* ── Sidebar Enhancements ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 10px;
}
.sidebar-logo .logo-text {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg,#6366f1,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}
.sidebar-logo .logo-sub {
    font-size: 0.75rem;
    color: #64748b !important;
    display: block;
    margin-top: 2px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s;
    color: #94a3b8 !important;
    font-size: 0.92rem;
    font-weight: 500;
    margin-bottom: 4px;
    border: 1px solid transparent;
}
.nav-item.active {
    background: rgba(99,102,241,0.15);
    border-color: rgba(99,102,241,0.3);
    color: #a5b4fc !important;
}
.nav-item:hover { background: rgba(99,102,241,0.1); }
.stat-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 8px;
    padding: 8px 12px;
    color: #c7d2fe;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 4px 0;
    width: 100%;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30,33,48,0.8);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,102,241,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
}

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e2130,#252840);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stSlider > div,
.stFileUploader > div {
    background: rgba(30,33,48,0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Plotly charts dark bg ── */
.js-plotly-plot { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }

/* ── Success/Error/Info ── */
.stSuccess { background:rgba(16,185,129,0.1)!important; border-left:3px solid #10b981!important; }
.stError   { background:rgba(239,68,68,0.1)!important;  border-left:3px solid #ef4444!important; }
.stInfo    { background:rgba(99,102,241,0.1)!important; border-left:3px solid #6366f1!important; }
.stWarning { background:rgba(245,158,11,0.1)!important; border-left:3px solid #f59e0b!important; }
</style>
""", unsafe_allow_html=True)

# ── CHART THEME ──────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='rgba(99,102,241,0.1)', linecolor='rgba(99,102,241,0.2)',
               tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='rgba(99,102,241,0.1)', linecolor='rgba(99,102,241,0.2)',
               tickfont=dict(color='#64748b')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
    margin=dict(l=10, r=10, t=30, b=10),
)
COLORS = ['#6366f1','#06b6d4','#10b981','#f59e0b','#ec4899','#8b5cf6','#3b82f6','#ef4444']

# ── SESSION STATE ────────────────────────────────────────────────────────────
for key in ['df','preprocessor','analyzer','forecaster','segmenter',
            'detector','forecast_df','segments_df','anomaly_df',
            'model_type','forecast_periods','forecast_metrics','comparison_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color:#6366f1;text-decoration:none;font-weight:600;">⬇️ {text}</a>'

def apply_chart_theme(fig, title=''):
    fig.update_layout(**CHART_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color='#f1f5f9', size=14, family='Inter')))
    return fig

def empty_state(icon, title, subtitle=''):
    st.markdown(f"""
    <div class="empty-state">
        <span class="icon">{icon}</span>
        <h3>{title}</h3>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    all_data = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.csv'):
                df = pd.read_csv(f)
            elif f.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(f)
            elif f.name.endswith('.json'):
                df = pd.read_json(f)
            else:
                st.error(f"Format tidak didukung: {f.name}"); continue
            all_data.append(df)
            st.success(f"✅ {f.name} ({len(df):,} rows)")
        except Exception as e:
            st.error(f"❌ {f.name}: {e}")
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        pp = DataPreprocessor()
        processed = pp.preprocess(combined)
        st.session_state.df = processed
        st.session_state.preprocessor = pp
        st.session_state.analyzer = SalesAnalyzer(processed)
        cols_detected = {k: v for k, v in pp.mapped_columns.items()}
        st.success(f"🎉 Berhasil! Total: {len(processed):,} records")
        if cols_detected:
            st.info(f"✅ Kolom terdeteksi: {cols_detected}")
        missing_critical = [c for c in ['date','revenue'] if c not in processed.columns]
        if missing_critical:
            st.warning(f"⚠️ Kolom penting tidak terdeteksi: {missing_critical}. Cek nama kolom di file Anda.")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <span class="logo-text">⚡ SalesML</span>
            <span class="logo-sub">Analytics Dashboard</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 📁 Upload Data")
        uploaded = st.file_uploader("CSV / Excel / JSON", type=['csv','xlsx','xls','json'],
                                     accept_multiple_files=True, label_visibility='collapsed')
        if uploaded:
            if st.button("🚀 Proses Data", type="primary", use_container_width=True):
                with st.spinner("Memproses..."):
                    process_uploaded_files(uploaded)

        st.markdown("")
        if st.button("📥 Load Sample Data", use_container_width=True):
            with st.spinner("Generate sample..."):
                df = create_sample_data(n_records=1000)
                st.session_state.df = df
                st.session_state.analyzer = SalesAnalyzer(df)
                st.success("Sample data loaded!")
                st.rerun()

        st.markdown("---")

        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown(f"""
            <div class="stat-badge">📊 {len(df):,} records</div>
            <div class="stat-badge">📅 {df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')}</div>
            """, unsafe_allow_html=True)
            st.markdown("")

        st.markdown("##### ⚙️ Model Settings")
        model_type = st.selectbox("Model Forecasting",
            ['gradient_boosting','random_forest','extra_trees','ensemble',
             'stacking','xgboost','lightgbm','ridge','linear'],
            help="gradient_boosting & ensemble biasanya terbaik")
        forecast_periods = st.slider("Forecast (hari)", 7, 90, 30)
        st.session_state.model_type = model_type
        st.session_state.forecast_periods = forecast_periods

        st.markdown("---")
        st.markdown("""
        <div style="color:#475569;font-size:0.75rem;text-align:center;padding:10px 0">
            SalesML v2.0 · Built with Streamlit<br>
            <span style="color:#6366f1">Powered by ML & AI</span>
        </div>""", unsafe_allow_html=True)

# ── OVERVIEW TAB ─────────────────────────────────────────────────────────────
def render_overview():
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">📊 Sales Analytics</p>
        <p class="hero-sub">Monitor, analisis, dan prediksi performa penjualan dengan Machine Learning</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("📂", "Belum ada data", "Upload file CSV/Excel atau klik Load Sample Data di sidebar")
        return

    df = st.session_state.df
    analyzer = st.session_state.analyzer
    kpis = analyzer.calculate_kpis()

    # KPI Cards
    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1, 'purple', '💰', format_currency(kpis.get('total_revenue',0)), 'Total Revenue'),
        (c2, 'cyan',   '🧾', format_number(kpis.get('total_transactions',0)), 'Transaksi'),
        (c3, 'green',  '📦', format_currency(kpis.get('avg_order_value',0)), 'Avg Order'),
        (c4, 'orange', '🏷️', format_number(kpis.get('unique_products',0)), 'Produk Unik'),
        (c5, 'pink',   '📈', format_currency(kpis.get('avg_daily_revenue',0)), 'Avg/Hari'),
    ]
    for col, color, icon, value, label in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <span class="kpi-icon">{icon}</span>
                <span class="kpi-value">{value}</span>
                <span class="kpi-label">{label}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Revenue Trend + Top Products
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📈 Tren Revenue Harian</p>', unsafe_allow_html=True)
        daily = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily['revenue'],
            fill='tozeroy', mode='lines',
            line=dict(color='#6366f1', width=2),
            fillcolor='rgba(99,102,241,0.1)',
            name='Revenue'
        ))
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🏆 Top 10 Produk</p>', unsafe_allow_html=True)
        top = analyzer.get_top_products(n=10)
        if not top.empty:
            fig = go.Figure(go.Bar(
                x=top['revenue'], y=top['product'],
                orientation='h',
                marker=dict(
                    color=top['revenue'],
                    colorscale=[[0,'#4338ca'],[0.5,'#6366f1'],[1,'#06b6d4']]
                )
            ))
            fig.update_layout(yaxis=dict(autorange='reversed'))
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Category Pie + Monthly Growth
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🗂️ Revenue per Kategori</p>', unsafe_allow_html=True)
        cats = analyzer.get_category_analysis()
        if not cats.empty:
            fig = go.Figure(go.Pie(
                labels=cats['category'], values=cats['total_revenue'],
                hole=0.55,
                marker=dict(colors=COLORS),
                textfont=dict(color='white')
            ))
            fig.update_layout(
                showlegend=True,
                legend=dict(font=dict(color='#94a3b8'))
            )
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📅 Pertumbuhan Bulanan (MoM)</p>', unsafe_allow_html=True)
        monthly = analyzer.calculate_growth_metrics()
        if not monthly.empty and 'revenue_mom' in monthly.columns:
            mom = monthly.dropna(subset=['revenue_mom'])
            colors_bar = ['#10b981' if x >= 0 else '#ef4444' for x in mom['revenue_mom']]
            fig = go.Figure(go.Bar(
                x=mom['date'], y=mom['revenue_mom'],
                marker_color=colors_bar,
                name='MoM %'
            ))
            fig.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_dash='dash')
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Insights
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">💡 Key Insights</p>', unsafe_allow_html=True)
    insights = analyzer.generate_insights()
    cols = st.columns(2)
    for i, insight in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── FORECASTING TAB ──────────────────────────────────────────────────────────
def render_forecasting():
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">🔮 Sales Forecasting</p>
        <p class="hero-sub">Prediksi revenue masa depan dengan model ML terbaik</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("🔮", "Belum ada data", "Load data terlebih dahulu")
        return

    df = st.session_state.df
    c1, c2 = st.columns([1, 3])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚙️ Training Settings</p>', unsafe_allow_html=True)
        model_type = st.selectbox("Model", ['gradient_boosting','random_forest','extra_trees',
                                             'ensemble','stacking','xgboost','lightgbm','ridge','linear'])
        do_tuning = st.checkbox("Hyperparameter Tuning", value=False,
                                 help="Lebih akurat tapi lebih lama")
        periods = st.slider("Periode Forecast (hari)", 7, 90, 30)

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_type}..."):
                try:
                    fc = SalesForecaster(model_type=model_type)
                    metrics = fc.fit(df, do_tuning=do_tuning)
                    st.session_state.forecaster = fc
                    st.session_state.forecast_metrics = metrics
                    st.session_state.forecast_df = fc.forecast_future(df, periods=periods)
                    st.success("✅ Selesai!")
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.session_state.forecast_metrics:
            m = st.session_state.forecast_metrics
            st.markdown("---")
            st.markdown("**📊 Performa Model**")
            st.metric("R² Score", f"{m.get('test_r2',0):.4f}")
            st.metric("RMSE", f"{m.get('test_rmse',0):,.0f}")
            st.metric("MAPE", f"{m.get('test_mape',0):.1f}%")
            st.metric("MAE", f"{m.get('test_mae',0):,.0f}")

        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if st.session_state.forecast_df is not None:
            fc_df = st.session_state.forecast_df
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">📈 Hasil Forecast</p>', unsafe_allow_html=True)

            daily = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily['date'], y=daily['revenue'],
                fill='tozeroy', mode='lines', name='Historis',
                line=dict(color='#6366f1', width=2),
                fillcolor='rgba(99,102,241,0.1)'
            ))
            if 'forecast' in fc_df.columns:
                fig.add_trace(go.Scatter(
                    x=fc_df['date'], y=fc_df['forecast'],
                    mode='lines', name='Forecast',
                    line=dict(color='#06b6d4', width=2, dash='dash')
                ))
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_download_link(fc_df, 'forecast.csv', 'Download Forecast CSV'),
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature Importance
            if st.session_state.forecaster:
                imp = st.session_state.forecaster.get_feature_importance()
                if not imp.empty and 'importance' in imp.columns and imp['importance'].sum() > 0:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<p class="section-title">🔍 Feature Importance</p>', unsafe_allow_html=True)
                    top_imp = imp.head(12)
                    fig = go.Figure(go.Bar(
                        x=top_imp['importance'], y=top_imp['feature'],
                        orientation='h',
                        marker=dict(color=top_imp['importance'],
                                    colorscale=[[0,'#4338ca'],[1,'#06b6d4']])
                    ))
                    fig.update_layout(yaxis=dict(autorange='reversed'))
                    apply_chart_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("🔮", "Belum ada forecast", "Train model untuk melihat hasil prediksi")


# ── SEGMENTATION TAB ─────────────────────────────────────────────────────────
def render_segmentation():
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">🎯 Product Segmentation</p>
        <p class="hero-sub">Kelompokkan produk berdasarkan performa dengan K-Means clustering</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("🎯", "Belum ada data", "Load data terlebih dahulu")
        return

    df = st.session_state.df
    c1, c2 = st.columns([1, 3])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚙️ Clustering Settings</p>', unsafe_allow_html=True)
        n_clusters = st.slider("Jumlah Cluster", 2, 8, 4)

        if st.button("🚀 Jalankan Clustering", type="primary", use_container_width=True):
            with st.spinner("Clustering..."):
                try:
                    seg = ProductSegmenter(n_clusters=n_clusters)
                    seg_df = seg.fit(df)
                    st.session_state.segmenter = seg
                    st.session_state.segments_df = seg_df
                    st.success(f"✅ {n_clusters} segment ditemukan!")
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if st.session_state.segments_df is not None:
            seg_df = st.session_state.segments_df

            c_v1, c_v2 = st.columns(2)
            with c_v1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">📊 Distribusi Segment</p>', unsafe_allow_html=True)
                counts = seg_df['segment'].value_counts()
                fig = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values,
                    hole=0.55, marker=dict(colors=COLORS),
                    textfont=dict(color='white')
                ))
                apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c_v2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">💰 Revenue per Segment</p>', unsafe_allow_html=True)
                seg_rev = seg_df.groupby('segment')['total_revenue'].sum().reset_index()
                fig = go.Figure(go.Bar(
                    x=seg_rev['segment'], y=seg_rev['total_revenue'],
                    marker=dict(color=COLORS[:len(seg_rev)])
                ))
                apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">📋 Detail Segmentasi</p>', unsafe_allow_html=True)
            st.dataframe(seg_df.sort_values('total_revenue', ascending=False),
                         use_container_width=True, height=300)
            st.markdown(get_download_link(seg_df, 'segments.csv', 'Download CSV'),
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("🎯", "Belum ada segmentasi", "Jalankan clustering untuk melihat hasil")


# ── ANOMALY TAB ───────────────────────────────────────────────────────────────
def render_anomaly():
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">🚨 Anomaly Detection</p>
        <p class="hero-sub">Deteksi transaksi mencurigakan dengan Isolation Forest & Z-Score</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("🚨", "Belum ada data", "Load data terlebih dahulu")
        return

    df = st.session_state.df
    c1, c2 = st.columns([1, 3])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚙️ Detection Settings</p>', unsafe_allow_html=True)
        method = st.selectbox("Metode", ['isolation_forest', 'zscore'])
        contamination = st.slider("Expected Outlier (%)", 1, 20, 5) / 100

        if st.button("🔍 Deteksi Anomali", type="primary", use_container_width=True):
            with st.spinner("Mendeteksi..."):
                try:
                    det = AnomalyDetector(method=method, contamination=contamination)
                    anom_df = det.fit(df)
                    st.session_state.detector = det
                    st.session_state.anomaly_df = anom_df
                    n = int(anom_df['anomaly'].sum())
                    pct = n / len(anom_df) * 100
                    st.success(f"✅ Ditemukan {n} anomali ({pct:.1f}%)")
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if st.session_state.anomaly_df is not None:
            anom_df = st.session_state.anomaly_df
            normal = anom_df[anom_df['anomaly'] == 0]
            anomalies = anom_df[anom_df['anomaly'] == 1]

            # Summary metrics
            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Total Data", f"{len(anom_df):,}")
            with m2: st.metric("Normal", f"{len(normal):,}")
            with m3: st.metric("🚨 Anomali", f"{len(anomalies):,}")

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">📈 Visualisasi Anomali</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=normal['date'], y=normal['revenue'], mode='markers',
                name='Normal', marker=dict(color='#6366f1', size=5, opacity=0.5)
            ))
            fig.add_trace(go.Scatter(
                x=anomalies['date'], y=anomalies['revenue'], mode='markers',
                name='🚨 Anomali', marker=dict(color='#ef4444', size=10, symbol='x', opacity=0.9)
            ))
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if not anomalies.empty:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">📋 Daftar Anomali</p>', unsafe_allow_html=True)
                cols_show = [c for c in ['date','product','revenue','quantity','anomaly_score'] if c in anomalies.columns]
                st.dataframe(anomalies[cols_show].sort_values('anomaly_score', ascending=False),
                             use_container_width=True, height=280)
                st.markdown(get_download_link(anomalies, 'anomalies.csv', 'Download Anomalies CSV'),
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("🚨", "Belum ada deteksi", "Jalankan deteksi untuk melihat hasil")


# ── MODEL COMPARISON TAB ──────────────────────────────────────────────────────
def render_comparison():
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">⚖️ Model Comparison</p>
        <p class="hero-sub">Bandingkan semua model ML dan temukan yang terbaik secara otomatis</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("⚖️", "Belum ada data", "Load data terlebih dahulu")
        return

    df = st.session_state.df

    if st.button("🚀 Bandingkan Semua Model", type="primary"):
        with st.spinner("Training & membandingkan semua model... (ini butuh beberapa menit)"):
            try:
                comp = ModelComparator()
                comp_df = comp.compare_models(df)
                st.session_state.comparison_df = comp_df
                if comp.best_model_type:
                    st.success(f"🏆 Model terbaik: **{comp.best_model_type}**")
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state.comparison_df is not None:
        comp_df = st.session_state.comparison_df

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Hasil Perbandingan</p>', unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if 'test_rmse' in comp_df.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">📉 RMSE Comparison (lebih kecil = lebih baik)</p>', unsafe_allow_html=True)
                rmse_data = comp_df['test_rmse'].dropna().astype(float).sort_values()
                fig = go.Figure(go.Bar(
                    x=rmse_data.index, y=rmse_data.values,
                    marker=dict(color=rmse_data.values,
                                colorscale=[[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']])
                ))
                apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                if 'test_r2' in comp_df.columns:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<p class="section-title">📈 R² Score (lebih besar = lebih baik)</p>', unsafe_allow_html=True)
                    r2_data = comp_df['test_r2'].dropna().astype(float).sort_values(ascending=False)
                    fig = go.Figure(go.Bar(
                        x=r2_data.index, y=r2_data.values,
                        marker=dict(color=r2_data.values,
                                    colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#10b981']])
                    ))
                    apply_chart_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        empty_state("⚖️", "Belum ada perbandingan", "Klik tombol di atas untuk mulai membandingkan")


# ── REPORTS TAB ───────────────────────────────────────────────────────────────
def render_reports():
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">📑 Reports & Export</p>
        <p class="hero-sub">Export hasil analisis ke berbagai format</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("📑", "Belum ada data", "Load data terlebih dahulu")
        return

    analyzer = st.session_state.analyzer
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📄 PDF Report</p>', unsafe_allow_html=True)
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Generating..."):
                try:
                    Path('reports').mkdir(exist_ok=True)
                    reporter = ReportGenerator(analyzer)
                    reporter.generate_pdf_report(
                        'reports/sales_report.pdf',
                        st.session_state.forecast_df,
                        st.session_state.segments_df,
                        st.session_state.anomaly_df
                    )
                    with open('reports/sales_report.pdf','rb') as f:
                        st.download_button("⬇️ Download PDF", f.read(),
                                           'sales_report.pdf', 'application/pdf',
                                           use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Excel Report</p>', unsafe_allow_html=True)
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Generating..."):
                try:
                    Path('reports').mkdir(exist_ok=True)
                    reporter = ReportGenerator(analyzer)
                    reporter.export_to_excel(
                        'reports/sales_analysis.xlsx',
                        st.session_state.forecast_df,
                        st.session_state.segments_df,
                        st.session_state.anomaly_df
                    )
                    with open('reports/sales_analysis.xlsx','rb') as f:
                        st.download_button("⬇️ Download Excel", f.read(),
                                           'sales_analysis.xlsx',
                                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                           use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📁 CSV Export</p>', unsafe_allow_html=True)
        if st.button("Export CSV", use_container_width=True):
            with st.spinner("Exporting..."):
                try:
                    import zipfile
                    Path('reports').mkdir(exist_ok=True)
                    reporter = ReportGenerator(analyzer)
                    reporter.export_to_csv('reports')
                    zip_path = 'reports/csv_export.zip'
                    with zipfile.ZipFile(zip_path,'w') as zf:
                        for csv_file in Path('reports').glob('*.csv'):
                            zf.write(csv_file, csv_file.name)
                    with open(zip_path,'rb') as f:
                        st.download_button("⬇️ Download ZIP", f.read(),
                                           'csv_export.zip', 'application/zip',
                                           use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    tabs = st.tabs(["📊 Overview","🔮 Forecasting","🎯 Segmentation",
                    "🚨 Anomaly Detection","⚖️ Model Comparison","📑 Reports"])
    with tabs[0]: render_overview()
    with tabs[1]: render_forecasting()
    with tabs[2]: render_segmentation()
    with tabs[3]: render_anomaly()
    with tabs[4]: render_comparison()
    with tabs[5]: render_reports()

if __name__ == "__main__":
    main()
