"""
Streamlit Dashboard for Sales ML System
=======================================
Dashboard interaktif untuk analisis data penjualan
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import base64
from io import BytesIO

# Import custom modules
from preprocessing import DataPreprocessor, preprocess_file
from ml_model import SalesForecaster, ProductSegmenter, AnomalyDetector, ModelComparator
from utils import SalesAnalyzer, ReportGenerator, Visualizer, format_currency, create_sample_data

# Page configuration
st.set_page_config(
    page_title="Sales ML Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2ca02c;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
if 'segmenter' not in st.session_state:
    st.session_state.segmenter = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'segments_df' not in st.session_state:
    st.session_state.segments_df = None
if 'anomaly_df' not in st.session_state:
    st.session_state.anomaly_df = None


def get_download_link(df, filename, link_text):
    """Generate download link untuk DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def render_sidebar():
    """Render sidebar dengan upload dan settings"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=80)
        st.title("📊 Sales ML Analytics")
        
        st.markdown("---")
        
        # File upload section
        st.subheader("📁 Upload Data")
        
        uploaded_files = st.file_uploader(
            "Upload file (CSV, Excel, JSON)",
            type=['csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    process_uploaded_files(uploaded_files)
        
        st.markdown("---")
        
        # Sample data button
        if st.button("📥 Load Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                df = create_sample_data(n_records=1000)
                st.session_state.df = df
                st.session_state.preprocessor = DataPreprocessor()
                st.session_state.analyzer = SalesAnalyzer(df)
                st.success("Sample data loaded!")
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        if st.session_state.df is not None:
            st.info(f"📊 Loaded: {len(st.session_state.df):,} records")
            st.info(f"📅 Date range: {st.session_state.df['date'].min().strftime('%Y-%m-%d')} to {st.session_state.df['date'].max().strftime('%Y-%m-%d')}")
        
        # Model settings
        st.markdown("**Model Settings**")
        model_type = st.selectbox(
            "Forecasting Model",
            ['gradient_boosting', 'random_forest', 'extra_trees', 'ensemble',
             'stacking', 'xgboost', 'lightgbm', 'ridge', 'lasso', 'linear'],
            help="Pilih model untuk forecasting. gradient_boosting & ensemble biasanya terbaik."
        )
        
        forecast_periods = st.slider(
            "Forecast Periods (days)",
            min_value=7,
            max_value=90,
            value=30
        )
        
        st.session_state.model_type = model_type
        st.session_state.forecast_periods = forecast_periods
        
        st.markdown("---")
        
        # About
        st.markdown("**ℹ️ About**")
        st.markdown("""
        Sales ML Analytics adalah sistem analisis data penjualan 
        dengan Machine Learning untuk forecasting, segmentasi, 
        dan anomaly detection.
        """)


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    all_data = []
    
    for uploaded_file in uploaded_files:
        try:
            # Determine file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue
            
            all_data.append(df)
            st.success(f"✅ Loaded: {uploaded_file.name} ({len(df)} rows)")
            
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess(combined_df)
        
        # Store in session state
        st.session_state.df = processed_df
        st.session_state.preprocessor = preprocessor
        st.session_state.analyzer = SalesAnalyzer(processed_df)
        
        st.success(f"🎉 Data processed successfully! Total: {len(processed_df):,} records")


def render_overview():
    """Render overview dashboard"""
    st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.info("👈 Please upload data or load sample data from the sidebar")
        return
    
    df = st.session_state.df
    analyzer = st.session_state.analyzer
    
    # KPI Cards
    kpis = analyzer.calculate_kpis()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Revenue",
            format_currency(kpis.get('total_revenue', 0)),
            help="Total revenue dari semua transaksi"
        )
    
    with col2:
        st.metric(
            "Total Transactions",
            f"{kpis.get('total_transactions', 0):,}",
            help="Jumlah total transaksi"
        )
    
    with col3:
        st.metric(
            "Avg Order Value",
            format_currency(kpis.get('avg_order_value', 0)),
            help="Rata-rata nilai per transaksi"
        )
    
    with col4:
        st.metric(
            "Unique Products",
            f"{kpis.get('unique_products', 0)}",
            help="Jumlah produk unik"
        )
    
    with col5:
        st.metric(
            "Avg Daily Revenue",
            format_currency(kpis.get('avg_daily_revenue', 0)),
            help="Rata-rata revenue per hari"
        )
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        viz = Visualizer(df)
        fig = viz.create_revenue_trend_chart()
        if hasattr(fig, 'update_layout'):
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)
    
    with col2:
        st.subheader("🏆 Top 10 Products")
        top_products = analyzer.get_top_products(n=10)
        if not top_products.empty:
            fig = px.bar(
                top_products,
                x='revenue',
                y='product',
                orientation='h',
                color='revenue',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Revenue by Category")
        categories = analyzer.get_category_analysis()
        if not categories.empty:
            fig = px.pie(
                categories,
                values='total_revenue',
                names='category',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Monthly Growth")
        monthly = analyzer.calculate_growth_metrics()
        if not monthly.empty and 'revenue_mom' in monthly.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly['date'],
                y=monthly['revenue_mom'],
                name='MoM Growth (%)',
                marker_color=['green' if x > 0 else 'red' for x in monthly['revenue_mom']]
            ))
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Growth (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Insights section
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    insights = analyzer.generate_insights()
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


def render_forecasting():
    """Render forecasting page"""
    st.markdown('<h1 class="main-header">🔮 Sales Forecasting</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please load data first")
        return
    
    df = st.session_state.df
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Training Settings")
        
        model_type = st.selectbox(
            "Model Type",
            ['gradient_boosting', 'random_forest', 'extra_trees', 'ensemble',
             'stacking', 'xgboost', 'lightgbm', 'ridge', 'lasso', 'linear', 'prophet'],
            index=0
        )
        
        do_tuning = st.checkbox("Hyperparameter Tuning", value=False)
        
        forecast_periods = st.slider(
            "Forecast Periods",
            min_value=7,
            max_value=90,
            value=30
        )
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    forecaster = SalesForecaster(model_type=model_type)
                    metrics = forecaster.fit(df, do_tuning=do_tuning)
                    
                    st.session_state.forecaster = forecaster
                    st.session_state.forecast_metrics = metrics
                    
                    # Generate forecast
                    forecast_df = forecaster.forecast_future(df, periods=forecast_periods)
                    st.session_state.forecast_df = forecast_df
                    
                    st.success("✅ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display metrics
        if 'forecast_metrics' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Model Performance")
            metrics = st.session_state.forecast_metrics
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("MAE", f"{metrics.get('test_mae', 0):,.0f}")
            with col_m2:
                st.metric("RMSE", f"{metrics.get('test_rmse', 0):,.0f}")
            
            st.metric("R² Score", f"{metrics.get('test_r2', 0):.3f}")
        
        # Save model
        if st.session_state.forecaster is not None:
            st.markdown("---")
            if st.button("💾 Save Model", use_container_width=True):
                st.session_state.forecaster.save_model('models/forecaster.pkl')
                st.success("Model saved!")
    
    with col2:
        st.subheader("📈 Forecast Visualization")
        
        if st.session_state.forecast_df is not None:
            viz = Visualizer(df)
            fig = viz.create_forecast_chart(st.session_state.forecast_df)
            
            if hasattr(fig, 'update_layout'):
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)
            
            # Forecast table
            st.subheader("📋 Forecast Data")
            st.dataframe(st.session_state.forecast_df, use_container_width=True)
            
            # Download forecast
            st.markdown(
                get_download_link(st.session_state.forecast_df, 'forecast.csv', '📥 Download Forecast CSV'),
                unsafe_allow_html=True
            )
        else:
            st.info("Train a model to see forecast")
        
        # Feature importance
        if st.session_state.forecaster is not None:
            st.markdown("---")
            st.subheader("🔍 Feature Importance")
            
            importance_df = st.session_state.forecaster.get_feature_importance()
            if not importance_df.empty and 'importance' in importance_df.columns:
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h'
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)


def render_segmentation():
    """Render segmentation page"""
    st.markdown('<h1 class="main-header">🎯 Product Segmentation</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please load data first")
        return
    
    df = st.session_state.df
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Clustering Settings")
        
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=8,
            value=4
        )
        
        if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
            with st.spinner("Running K-Means clustering..."):
                try:
                    segmenter = ProductSegmenter(n_clusters=n_clusters)
                    segments_df = segmenter.fit(df)
                    
                    st.session_state.segmenter = segmenter
                    st.session_state.segments_df = segments_df
                    
                    st.success("✅ Clustering completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Save model
        if st.session_state.segmenter is not None:
            st.markdown("---")
            if st.button("💾 Save Model", use_container_width=True):
                import pickle
                with open('models/segmenter.pkl', 'wb') as f:
                    pickle.dump(st.session_state.segmenter, f)
                st.success("Model saved!")
    
    with col2:
        if st.session_state.segments_df is not None:
            segments_df = st.session_state.segments_df
            
            # Segment distribution
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.subheader("📊 Segment Distribution")
                segment_counts = segments_df['segment'].value_counts()
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_v2:
                st.subheader("💰 Revenue by Segment")
                segment_revenue = segments_df.groupby('segment')['total_revenue'].sum().reset_index()
                fig = px.bar(
                    segment_revenue,
                    x='segment',
                    y='total_revenue',
                    color='segment'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment details
            st.subheader("📋 Segment Details")
            st.dataframe(segments_df.sort_values('total_revenue', ascending=False), use_container_width=True)
            
            # Download
            st.markdown(
                get_download_link(segments_df, 'segments.csv', '📥 Download Segments CSV'),
                unsafe_allow_html=True
            )
        else:
            st.info("Run clustering to see segments")


def render_anomaly_detection():
    """Render anomaly detection page"""
    st.markdown('<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please load data first")
        return
    
    df = st.session_state.df
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Detection Settings")
        
        method = st.selectbox(
            "Detection Method",
            ['isolation_forest', 'zscore']
        )
        
        contamination = st.slider(
            "Expected Contamination (%)",
            min_value=1,
            max_value=20,
            value=5
        ) / 100
        
        if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
            with st.spinner("Detecting anomalies..."):
                try:
                    detector = AnomalyDetector(method=method, contamination=contamination)
                    anomaly_df = detector.fit(df)
                    
                    st.session_state.detector = detector
                    st.session_state.anomaly_df = anomaly_df
                    
                    n_anomalies = anomaly_df['anomaly'].sum()
                    st.success(f"✅ Found {n_anomalies} anomalies ({n_anomalies/len(anomaly_df)*100:.2f}%)")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Save model
        if st.session_state.detector is not None:
            st.markdown("---")
            if st.button("💾 Save Model", use_container_width=True):
                import pickle
                with open('models/detector.pkl', 'wb') as f:
                    pickle.dump(st.session_state.detector, f)
                st.success("Model saved!")
    
    with col2:
        if st.session_state.anomaly_df is not None:
            anomaly_df = st.session_state.anomaly_df
            
            # Anomaly visualization
            st.subheader("📈 Anomaly Visualization")
            
            normal = anomaly_df[anomaly_df['anomaly'] == 0]
            anomalies = anomaly_df[anomaly_df['anomaly'] == 1]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=normal['date'],
                y=normal['revenue'],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=6, opacity=0.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['revenue'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Revenue',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly list
            st.subheader("📋 Anomaly List")
            anomaly_list = anomalies[['date', 'product', 'revenue', 'quantity', 'anomaly_score']].sort_values('anomaly_score', ascending=False)
            st.dataframe(anomaly_list, use_container_width=True)
            
            # Download
            if not anomalies.empty:
                st.markdown(
                    get_download_link(anomalies, 'anomalies.csv', '📥 Download Anomalies CSV'),
                    unsafe_allow_html=True
                )
        else:
            st.info("Run anomaly detection to see results")


def render_model_comparison():
    """Render model comparison page"""
    st.markdown('<h1 class="main-header">⚖️ Model Comparison</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please load data first")
        return
    
    df = st.session_state.df
    
    if st.button("🚀 Compare All Models", type="primary"):
        with st.spinner("Training and comparing models..."):
            try:
                comparator = ModelComparator()
                comparison_df = comparator.compare_models(df)
                
                st.session_state.comparison_df = comparison_df
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if 'comparison_df' in st.session_state:
        comparison_df = st.session_state.comparison_df
        
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        if 'test_rmse' in comparison_df.columns:
            fig = go.Figure()
            
            metrics_to_plot = ['test_mae', 'test_rmse']
            available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
            
            for metric in available_metrics:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=comparison_df.index,
                    y=comparison_df[metric]
                ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title='Model',
                yaxis_title='Error',
                title='Model Error Comparison (Lower is Better)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # R2 comparison
        if 'test_r2' in comparison_df.columns:
            fig = px.bar(
                x=comparison_df.index,
                y=comparison_df['test_r2'],
                color=comparison_df['test_r2'],
                color_continuous_scale='RdYlGn',
                labels={'x': 'Model', 'y': 'R² Score'}
            )
            fig.update_layout(title='R² Score Comparison (Higher is Better)')
            st.plotly_chart(fig, use_container_width=True)


def render_reports():
    """Render reports page"""
    st.markdown('<h1 class="main-header">📑 Reports & Export</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please load data first")
        return
    
    df = st.session_state.df
    analyzer = st.session_state.analyzer
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📄 PDF Report")
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    reporter = ReportGenerator(analyzer)
                    reporter.generate_pdf_report(
                        output_path='reports/sales_report.pdf',
                        forecast_df=st.session_state.forecast_df,
                        segments_df=st.session_state.segments_df,
                        anomaly_df=st.session_state.anomaly_df
                    )
                    
                    # Read and provide download
                    with open('reports/sales_report.pdf', 'rb') as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name='sales_report.pdf',
                        mime='application/pdf',
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("📊 Excel Report")
        if st.button("Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel..."):
                try:
                    reporter = ReportGenerator(analyzer)
                    reporter.export_to_excel(
                        output_path='reports/sales_analysis.xlsx',
                        forecast_df=st.session_state.forecast_df,
                        segments_df=st.session_state.segments_df,
                        anomaly_df=st.session_state.anomaly_df
                    )
                    
                    # Read and provide download
                    with open('reports/sales_analysis.xlsx', 'rb') as f:
                        excel_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_bytes,
                        file_name='sales_analysis.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col3:
        st.subheader("📁 CSV Export")
        if st.button("Export to CSV", use_container_width=True):
            with st.spinner("Exporting CSV..."):
                try:
                    reporter = ReportGenerator(analyzer)
                    reporter.export_to_csv(output_dir='reports')
                    
                    # Create zip file
                    import zipfile
                    zip_path = 'reports/csv_export.zip'
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for csv_file in Path('reports').glob('*.csv'):
                            zipf.write(csv_file, csv_file.name)
                    
                    # Read and provide download
                    with open(zip_path, 'rb') as f:
                        zip_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download CSV ZIP",
                        data=zip_bytes,
                        file_name='csv_export.zip',
                        mime='application/zip',
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def main():
    """Main function"""
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tabs = st.tabs([
        "📊 Overview",
        "🔮 Forecasting",
        "🎯 Segmentation",
        "🚨 Anomaly Detection",
        "⚖️ Model Comparison",
        "📑 Reports"
    ])
    
    with tabs[0]:
        render_overview()
    
    with tabs[1]:
        render_forecasting()
    
    with tabs[2]:
        render_segmentation()
    
    with tabs[3]:
        render_anomaly_detection()
    
    with tabs[4]:
        render_model_comparison()
    
    with tabs[5]:
        render_reports()


if __name__ == "__main__":
    main()
