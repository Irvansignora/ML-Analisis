"""
Utilities Module for Sales ML System
====================================
Helper functions untuk analisis, reporting, dan visualisasi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Try importing plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly tidak terinstall. Beberapa visualisasi tidak tersedia.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style untuk matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SalesAnalyzer:
    """
    Kelas untuk analisis deskriptif data penjualan
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang sudah dipreprocess
        """
        self.df = df.copy()
        self.summary = {}
        logger.info("SalesAnalyzer initialized")
    
    def calculate_kpis(self) -> Dict[str, float]:
        """
        Kalkulasi KPI utama
        
        Returns:
        --------
        dict
            Dictionary berisi KPI
        """
        kpis = {}
        
        # Basic KPIs
        if 'revenue' in self.df.columns:
            kpis['total_revenue'] = self.df['revenue'].sum()
            kpis['avg_revenue'] = self.df['revenue'].mean()
            kpis['median_revenue'] = self.df['revenue'].median()
        
        if 'quantity' in self.df.columns:
            kpis['total_quantity'] = self.df['quantity'].sum()
            kpis['avg_quantity'] = self.df['quantity'].mean()
        
        kpis['total_transactions'] = len(self.df)
        
        if 'revenue' in self.df.columns and 'quantity' in self.df.columns:
            kpis['avg_order_value'] = kpis['total_revenue'] / kpis['total_transactions']
        
        # Product metrics
        if 'product' in self.df.columns:
            kpis['unique_products'] = self.df['product'].nunique()
        
        if 'customer' in self.df.columns:
            kpis['unique_customers'] = self.df['customer'].nunique()
        
        # Date range
        if 'date' in self.df.columns:
            kpis['date_range_days'] = (self.df['date'].max() - self.df['date'].min()).days
            kpis['avg_daily_revenue'] = kpis.get('total_revenue', 0) / max(kpis['date_range_days'], 1)
        
        self.summary['kpis'] = kpis
        return kpis
    
    def calculate_growth_metrics(self) -> pd.DataFrame:
        """
        Kalkulasi growth metrics (MoM, YoY)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan growth metrics
        """
        if 'date' not in self.df.columns or 'revenue' not in self.df.columns:
            logger.warning("Date atau revenue column tidak ditemukan")
            return pd.DataFrame()
        
        # BUG FIX: pastikan semua kolom yang akan di-agg benar-benar ada dan numerik
        agg_dict = {}
        if 'revenue' in self.df.columns:
            self.df['revenue'] = pd.to_numeric(self.df['revenue'], errors='coerce').fillna(0)
            agg_dict['revenue'] = 'sum'
        if 'quantity' in self.df.columns:
            self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors='coerce').fillna(0)
            agg_dict['quantity'] = 'sum'
        
        if not agg_dict:
            logger.warning("Tidak ada kolom numerik untuk diagregasi")
            return pd.DataFrame()
        
        # Monthly aggregation
        monthly = self.df.groupby(self.df['date'].dt.to_period('M')).agg(agg_dict).reset_index()
        monthly['date'] = monthly['date'].dt.to_timestamp()
        
        # Calculate MoM growth
        monthly['revenue_mom'] = monthly['revenue'].pct_change() * 100
        if 'quantity' in monthly.columns:
            monthly['quantity_mom'] = monthly['quantity'].pct_change() * 100
        
        # Calculate YoY growth (if data spans > 1 year)
        if (monthly['date'].max() - monthly['date'].min()).days > 365:
            monthly['revenue_yoy'] = monthly['revenue'].pct_change(12) * 100
        
        self.summary['monthly_growth'] = monthly
        return monthly
    
    def get_top_products(self, n: int = 10, by: str = 'revenue') -> pd.DataFrame:
        """
        Get top N products
        
        Parameters:
        -----------
        n : int
            Jumlah produk
        by : str
            Metrik untuk ranking ('revenue', 'quantity', 'transactions')
            
        Returns:
        --------
        pd.DataFrame
            Top products
        """
        if 'product' not in self.df.columns:
            logger.warning("Product column tidak ditemukan")
            return pd.DataFrame()
        
        if by == 'revenue' and 'revenue' in self.df.columns:
            top = self.df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(n)
        elif by == 'quantity' and 'quantity' in self.df.columns:
            top = self.df.groupby('product')['quantity'].sum().sort_values(ascending=False).head(n)
        elif by == 'transactions':
            top = self.df.groupby('product').size().sort_values(ascending=False).head(n)
        else:
            return pd.DataFrame()
        
        top_df = top.reset_index()
        top_df.columns = ['product', by]
        top_df['rank'] = range(1, len(top_df) + 1)
        
        return top_df
    
    def get_declining_products(self, periods: int = 3) -> pd.DataFrame:
        """
        Identifikasi produk dengan penurunan tertinggi
        
        Parameters:
        -----------
        periods : int
            Jumlah periode untuk cek tren penurunan
            
        Returns:
        --------
        pd.DataFrame
            Produk dengan penurunan tertinggi
        """
        if 'date' not in self.df.columns or 'product' not in self.df.columns or 'revenue' not in self.df.columns:
            return pd.DataFrame()
        
        # Monthly revenue per product
        monthly_product = self.df.groupby([
            self.df['date'].dt.to_period('M'), 'product'
        ])['revenue'].sum().reset_index()
        monthly_product['date'] = monthly_product['date'].dt.to_timestamp()
        
        # Calculate trend untuk setiap produk
        declining = []
        for product in monthly_product['product'].unique():
            product_data = monthly_product[monthly_product['product'] == product].sort_values('date')
            if len(product_data) >= periods:
                recent = product_data['revenue'].tail(periods)
                if recent.is_monotonic_decreasing:
                    decline_pct = ((recent.iloc[0] - recent.iloc[-1]) / recent.iloc[0]) * 100
                    declining.append({
                        'product': product,
                        'decline_pct': decline_pct,
                        'latest_revenue': recent.iloc[-1]
                    })
        
        if declining:
            declining_df = pd.DataFrame(declining).sort_values('decline_pct', ascending=False)
            return declining_df
        
        return pd.DataFrame()
    
    def get_category_analysis(self) -> pd.DataFrame:
        """
        Analisis per kategori
        
        Returns:
        --------
        pd.DataFrame
            Analisis kategori
        """
        if 'category' not in self.df.columns:
            logger.warning("Category column tidak ditemukan")
            return pd.DataFrame()
        
        agg_dict = {'revenue': ['sum', 'mean', 'count']}
        if 'quantity' in self.df.columns:
            agg_dict['quantity'] = ['sum', 'mean']
        
        category_stats = self.df.groupby('category').agg(agg_dict).reset_index()
        
        if 'quantity' in self.df.columns:
            category_stats.columns = ['category', 'total_revenue', 'avg_revenue', 'transactions',
                                     'total_quantity', 'avg_quantity']
        else:
            category_stats.columns = ['category', 'total_revenue', 'avg_revenue', 'transactions']
            category_stats['total_quantity'] = 0
            category_stats['avg_quantity'] = 0
        
        # Calculate percentage
        category_stats['revenue_pct'] = (category_stats['total_revenue'] / 
                                        category_stats['total_revenue'].sum()) * 100
        
        return category_stats.sort_values('total_revenue', ascending=False)
    
    def get_weekly_pattern(self) -> pd.DataFrame:
        """
        Analisis pola mingguan
        
        Returns:
        --------
        pd.DataFrame
            Pola penjualan per hari dalam seminggu
        """
        if 'date' not in self.df.columns or 'revenue' not in self.df.columns:
            return pd.DataFrame()
        
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        
        agg_dict2 = {'revenue': ['sum', 'mean']}
        if 'quantity' in self.df.columns:
            agg_dict2['quantity'] = ['sum', 'mean']
        weekly = self.df.groupby('day_of_week').agg(agg_dict2).reset_index()
        
        # Order by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly['day_of_week'] = pd.Categorical(weekly['day_of_week'], categories=day_order, ordered=True)
        weekly = weekly.sort_values('day_of_week')
        
        return weekly
    
    def generate_insights(self) -> List[str]:
        """
        Generate insight otomatis dalam bentuk teks
        
        Returns:
        --------
        list
            List of insight strings
        """
        insights = []
        
        # KPI insights
        kpis = self.calculate_kpis()
        insights.append(f"Total revenue: Rp {kpis.get('total_revenue', 0):,.0f}")
        insights.append(f"Total transactions: {kpis.get('total_transactions', 0):,}")
        insights.append(f"Average order value: Rp {kpis.get('avg_order_value', 0):,.0f}")
        
        # Growth insights
        growth = self.calculate_growth_metrics()
        if not growth.empty and 'revenue_mom' in growth.columns:
            latest_mom = growth['revenue_mom'].iloc[-1]
            if not pd.isna(latest_mom):
                trend = "naik" if latest_mom > 0 else "turun"
                insights.append(f"Revenue bulan lalu {trend} {abs(latest_mom):.1f}% (MoM)")
        
        # Top product insights
        top_products = self.get_top_products(n=3)
        if not top_products.empty:
            top_product = top_products.iloc[0]['product']
            top_revenue = top_products.iloc[0]['revenue']
            insights.append(f"Produk terlaris: {top_product} (Rp {top_revenue:,.0f})")
        
        # Declining products
        declining = self.get_declining_products()
        if not declining.empty:
            worst = declining.iloc[0]
            insights.append(f"Peringatan: {worst['product']} turun {worst['decline_pct']:.1f}% dalam 3 bulan terakhir")
        
        # Category insights
        categories = self.get_category_analysis()
        if not categories.empty:
            top_category = categories.iloc[0]['category']
            cat_pct = categories.iloc[0]['revenue_pct']
            insights.append(f"Kategori terbesar: {top_category} ({cat_pct:.1f}% dari total revenue)")
        
        # Weekend vs weekday
        if 'date' in self.df.columns and 'revenue' in self.df.columns:
            weekend_data = self.df[self.df['date'].dt.dayofweek >= 5]['revenue']
            weekday_data = self.df[self.df['date'].dt.dayofweek < 5]['revenue']
            # BUG FIX: guard against empty slice yang menyebabkan RuntimeWarning: Mean of empty slice
            if not weekend_data.empty and not weekday_data.empty:
                weekend_revenue = weekend_data.mean()
                weekday_revenue = weekday_data.mean()
                if pd.notna(weekend_revenue) and pd.notna(weekday_revenue) and weekday_revenue > 0:
                    if weekend_revenue > weekday_revenue:
                        insights.append(f"Weekend revenue lebih tinggi {((weekend_revenue/weekday_revenue-1)*100):.1f}% dari weekday")
        
        return insights


class ReportGenerator:
    """
    Kelas untuk generate report dalam berbagai format
    """
    
    def __init__(self, analyzer: SalesAnalyzer):
        """
        Initialize report generator
        
        Parameters:
        -----------
        analyzer : SalesAnalyzer
            Instance SalesAnalyzer
        """
        self.analyzer = analyzer
        logger.info("ReportGenerator initialized")
    
    def generate_pdf_report(self, output_path: str = 'reports/sales_report.pdf',
                           forecast_df: Optional[pd.DataFrame] = None,
                           segments_df: Optional[pd.DataFrame] = None,
                           anomaly_df: Optional[pd.DataFrame] = None):
        """
        Generate PDF report lengkap
        
        Parameters:
        -----------
        output_path : str
            Path untuk menyimpan PDF
        forecast_df : pd.DataFrame, optional
            DataFrame forecast
        segments_df : pd.DataFrame, optional
            DataFrame segmentasi
        anomaly_df : pd.DataFrame, optional
            DataFrame anomaly detection
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(output_path) as pdf:
            # Page 1: Executive Summary
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            
            # Title
            fig.text(0.5, 0.95, 'Sales Analysis Report', 
                    ha='center', fontsize=24, weight='bold')
            fig.text(0.5, 0.90, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    ha='center', fontsize=12)
            
            # KPIs
            kpis = self.analyzer.calculate_kpis()
            y_pos = 0.80
            fig.text(0.1, y_pos, 'Key Performance Indicators', 
                    fontsize=16, weight='bold')
            
            kpi_texts = [
                f"Total Revenue: Rp {kpis.get('total_revenue', 0):,.0f}",
                f"Total Transactions: {kpis.get('total_transactions', 0):,}",
                f"Average Order Value: Rp {kpis.get('avg_order_value', 0):,.0f}",
                f"Unique Products: {kpis.get('unique_products', 0)}",
                f"Unique Customers: {kpis.get('unique_customers', 0)}"
            ]
            
            for i, text in enumerate(kpi_texts):
                fig.text(0.15, y_pos - 0.05 - (i * 0.05), text, fontsize=12)
            
            # Insights
            fig.text(0.1, 0.45, 'Key Insights', fontsize=16, weight='bold')
            insights = self.analyzer.generate_insights()
            for i, insight in enumerate(insights[:5]):
                fig.text(0.15, 0.40 - (i * 0.04), f"• {insight}", fontsize=10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Revenue Trend
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Daily revenue trend
            daily = self.analyzer.df.groupby(self.analyzer.df['date'].dt.date)['revenue'].sum()
            axes[0, 0].plot(daily.index, daily.values, linewidth=2)
            axes[0, 0].set_title('Daily Revenue Trend', fontsize=14, weight='bold')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Revenue')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Monthly revenue
            monthly = self.analyzer.calculate_growth_metrics()
            if not monthly.empty:
                axes[0, 1].bar(monthly['date'], monthly['revenue'])
                axes[0, 1].set_title('Monthly Revenue', fontsize=14, weight='bold')
                axes[0, 1].set_xlabel('Month')
                axes[0, 1].set_ylabel('Revenue')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Top products
            top_products = self.analyzer.get_top_products(n=10)
            if not top_products.empty:
                axes[1, 0].barh(top_products['product'], top_products['revenue'])
                axes[1, 0].set_title('Top 10 Products by Revenue', fontsize=14, weight='bold')
                axes[1, 0].set_xlabel('Revenue')
            
            # Category distribution
            categories = self.analyzer.get_category_analysis()
            if not categories.empty:
                axes[1, 1].pie(categories['total_revenue'], labels=categories['category'],
                              autopct='%1.1f%%')
                axes[1, 1].set_title('Revenue by Category', fontsize=14, weight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Forecast (if available)
            if forecast_df is not None and not forecast_df.empty:
                fig, ax = plt.subplots(figsize=(16, 8))
                
                # Historical
                daily = self.analyzer.df.groupby(self.analyzer.df['date'].dt.date)['revenue'].sum()
                ax.plot(daily.index, daily.values, label='Historical', linewidth=2)
                
                # Forecast
                if 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
                    ax.plot(forecast_df['date'], forecast_df['forecast'], 
                           label='Forecast', linewidth=2, linestyle='--', color='red')
                
                ax.set_title('Sales Forecast', fontsize=16, weight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Revenue')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 4: Segmentation (if available)
            if segments_df is not None and not segments_df.empty:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Segment distribution
                segment_counts = segments_df['segment'].value_counts()
                axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
                axes[0].set_title('Product Segment Distribution', fontsize=14, weight='bold')
                
                # Segment performance
                segment_perf = segments_df.groupby('segment')['total_revenue'].sum()
                axes[1].bar(segment_perf.index, segment_perf.values)
                axes[1].set_title('Revenue by Segment', fontsize=14, weight='bold')
                axes[1].set_ylabel('Revenue')
                axes[1].tick_params(axis='x', rotation=45)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 5: Anomalies (if available)
            if anomaly_df is not None and not anomaly_df.empty:
                fig, ax = plt.subplots(figsize=(16, 8))
                
                # Plot all data
                normal = anomaly_df[anomaly_df['anomaly'] == 0]
                anomalies = anomaly_df[anomaly_df['anomaly'] == 1]
                
                ax.scatter(normal['date'], normal['revenue'], c='blue', alpha=0.5, label='Normal')
                ax.scatter(anomalies['date'], anomalies['revenue'], c='red', s=100, label='Anomaly')
                
                ax.set_title('Anomaly Detection', fontsize=16, weight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Revenue')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        logger.info(f"PDF report saved to {output_path}")
    
    def export_to_excel(self, output_path: str = 'reports/sales_analysis.xlsx',
                       forecast_df: Optional[pd.DataFrame] = None,
                       segments_df: Optional[pd.DataFrame] = None,
                       anomaly_df: Optional[pd.DataFrame] = None):
        """
        Export hasil analisis ke Excel
        
        Parameters:
        -----------
        output_path : str
            Path untuk menyimpan Excel
        forecast_df : pd.DataFrame, optional
            DataFrame forecast
        segments_df : pd.DataFrame, optional
            DataFrame segmentasi
        anomaly_df : pd.DataFrame, optional
            DataFrame anomaly detection
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Raw data
            self.analyzer.df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # KPIs
            kpis = self.analyzer.calculate_kpis()
            kpi_df = pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value'])
            kpi_df.to_excel(writer, sheet_name='KPIs', index=False)
            
            # Monthly growth
            monthly = self.analyzer.calculate_growth_metrics()
            if not monthly.empty:
                monthly.to_excel(writer, sheet_name='Monthly Growth', index=False)
            
            # Top products
            top_products = self.analyzer.get_top_products(n=20)
            if not top_products.empty:
                top_products.to_excel(writer, sheet_name='Top Products', index=False)
            
            # Category analysis
            categories = self.analyzer.get_category_analysis()
            if not categories.empty:
                categories.to_excel(writer, sheet_name='Category Analysis', index=False)
            
            # Declining products
            declining = self.analyzer.get_declining_products()
            if not declining.empty:
                declining.to_excel(writer, sheet_name='Declining Products', index=False)
            
            # Forecast
            if forecast_df is not None and not forecast_df.empty:
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
            
            # Segments
            if segments_df is not None and not segments_df.empty:
                segments_df.to_excel(writer, sheet_name='Product Segments', index=False)
            
            # Anomalies
            if anomaly_df is not None and not anomaly_df.empty:
                anomalies = anomaly_df[anomaly_df['anomaly'] == 1]
                if not anomalies.empty:
                    anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
        
        logger.info(f"Excel report saved to {output_path}")
    
    def export_to_csv(self, output_dir: str = 'reports'):
        """
        Export hasil analisis ke multiple CSV files
        
        Parameters:
        -----------
        output_dir : str
            Directory untuk menyimpan CSV
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # KPIs
        kpis = self.analyzer.calculate_kpis()
        pd.DataFrame([kpis]).to_csv(f'{output_dir}/kpis.csv', index=False)
        
        # Monthly growth
        monthly = self.analyzer.calculate_growth_metrics()
        if not monthly.empty:
            monthly.to_csv(f'{output_dir}/monthly_growth.csv', index=False)
        
        # Top products
        top_products = self.analyzer.get_top_products(n=20)
        if not top_products.empty:
            top_products.to_csv(f'{output_dir}/top_products.csv', index=False)
        
        # Category analysis
        categories = self.analyzer.get_category_analysis()
        if not categories.empty:
            categories.to_csv(f'{output_dir}/category_analysis.csv', index=False)
        
        logger.info(f"CSV files saved to {output_dir}")


class Visualizer:
    """
    Kelas untuk membuat visualisasi interaktif dengan Plotly
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        """
        self.df = df.copy()
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly tidak tersedia. Gunakan matplotlib sebagai fallback.")
    
    def create_revenue_trend_chart(self) -> Any:
        """
        Create revenue trend chart
        
        Returns:
        --------
        plotly.graph_objects.Figure atau matplotlib.figure.Figure
        """
        if not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            daily = self.df.groupby(self.df['date'].dt.date)['revenue'].sum()
            ax.plot(daily.index, daily.values, linewidth=2)
            ax.set_title('Daily Revenue Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Revenue')
            plt.xticks(rotation=45)
            return fig
        
        daily = self.df.groupby(self.df['date'].dt.date)['revenue'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Daily Revenue Trend',
            xaxis_title='Date',
            yaxis_title='Revenue',
            hovermode='x unified'
        )
        
        return fig
    
    def create_top_products_chart(self, n: int = 10) -> Any:
        """
        Create top products bar chart
        
        Parameters:
        -----------
        n : int
            Jumlah produk
            
        Returns:
        --------
        plotly.graph_objects.Figure atau matplotlib.figure.Figure
        """
        if 'product' not in self.df.columns or 'revenue' not in self.df.columns:
            return None
        
        top = self.df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(n)
        
        if not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            top.plot(kind='barh', ax=ax)
            ax.set_title(f'Top {n} Products by Revenue')
            ax.set_xlabel('Revenue')
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top.values,
            y=top.index,
            orientation='h'
        ))
        
        fig.update_layout(
            title=f'Top {n} Products by Revenue',
            xaxis_title='Revenue',
            yaxis_title='Product',
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def create_category_pie_chart(self) -> Any:
        """
        Create category distribution pie chart
        
        Returns:
        --------
        plotly.graph_objects.Figure atau matplotlib.figure.Figure
        """
        if 'category' not in self.df.columns or 'revenue' not in self.df.columns:
            return None
        
        category_data = self.df.groupby('category')['revenue'].sum()
        
        if not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(category_data.values, labels=category_data.index, autopct='%1.1f%%')
            ax.set_title('Revenue by Category')
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=category_data.index,
            values=category_data.values,
            hole=0.4
        ))
        
        fig.update_layout(title='Revenue by Category')
        
        return fig
    
    def create_forecast_chart(self, forecast_df: pd.DataFrame) -> Any:
        """
        Create forecast chart
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            DataFrame forecast
            
        Returns:
        --------
        plotly.graph_objects.Figure atau matplotlib.figure.Figure
        """
        if not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical
            daily = self.df.groupby(self.df['date'].dt.date)['revenue'].sum()
            ax.plot(daily.index, daily.values, label='Historical', linewidth=2)
            
            # Forecast
            if 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
                ax.plot(forecast_df['date'], forecast_df['forecast'], 
                       label='Forecast', linewidth=2, linestyle='--')
            
            ax.set_title('Sales Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Revenue')
            ax.legend()
            plt.xticks(rotation=45)
            return fig
        
        fig = go.Figure()
        
        # Historical
        daily = self.df.groupby(self.df['date'].dt.date)['revenue'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['revenue'],
            mode='lines',
            name='Historical',
            line=dict(width=2)
        ))
        
        # Forecast
        if 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(width=2, dash='dash', color='red')
            ))
        
        fig.update_layout(
            title='Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue',
            hovermode='x unified'
        )
        
        return fig


def format_currency(value: float) -> str:
    """
    Format angka ke format mata uang Rupiah dengan K/M/B suffix biar ringkas
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "Rp 0"
    
    if abs(value) >= 1_000_000_000_000:
        return f"Rp {value/1_000_000_000_000:.1f}T"
    elif abs(value) >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"Rp {value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"Rp {value/1_000:.1f}K"
    else:
        return f"Rp {value:,.0f}"


def format_number(value: float) -> str:
    """Format angka biasa dengan K/M/B/T suffix"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "0"
    
    if abs(value) >= 1_000_000_000_000:
        return f"{value/1_000_000_000_000:.1f}T"
    elif abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def create_sample_data(n_records: int = 3000, output_path: str = 'data/sample_sales.csv'):
    """
    Generate sample data penjualan lengkap yang kompatibel dengan semua fitur dashboard.
    Kolom: tanggal, nama_produk, kategori, harga_satuan, hpp, qty, diskon, ongkir,
           grand_total, pelanggan, salesperson, nama_toko, kota, provinsi, channel,
           kurir, status
    """
    np.random.seed(42)

    products = [
        ('Laptop ProBook 14',       'Komputer',    8_500_000, 0.62),
        ('Laptop Gaming ASUS',      'Komputer',   14_000_000, 0.60),
        ('PC Desktop Core i7',      'Komputer',   12_000_000, 0.58),
        ('Monitor 27" 4K',          'Komputer',    4_200_000, 0.55),
        ('Keyboard Mechanical',     'Aksesoris',     850_000, 0.45),
        ('Mouse Wireless Logitech', 'Aksesoris',     450_000, 0.42),
        ('Headset Gaming RGB',      'Aksesoris',     750_000, 0.48),
        ('Webcam Full HD',          'Aksesoris',     650_000, 0.44),
        ('Smartphone Samsung A55',  'Handphone',   6_500_000, 0.65),
        ('Smartphone iPhone 15',    'Handphone',  18_000_000, 0.70),
        ('Smartphone Xiaomi 13',    'Handphone',   4_200_000, 0.60),
        ('Tablet iPad Air',         'Tablet',      9_800_000, 0.63),
        ('Tablet Samsung Tab S9',   'Tablet',      8_200_000, 0.61),
        ('Speaker Bluetooth JBL',   'Audio',       1_200_000, 0.50),
        ('Earbuds TWS Sony',        'Audio',       1_800_000, 0.52),
        ('Smart TV 43"',            'Elektronik',  5_500_000, 0.58),
        ('Smart TV 55"',            'Elektronik',  8_000_000, 0.57),
        ('Kamera DSLR Canon',       'Kamera',     12_500_000, 0.64),
        ('Kamera Mirrorless Sony',  'Kamera',     15_000_000, 0.65),
        ('Power Bank 20000mAh',     'Aksesoris',     380_000, 0.40),
    ]
    prod_w = np.array([6,4,3,5,8,9,7,6,5,2,7,4,4,8,7,4,3,2,2,9], dtype=float)
    prod_w /= prod_w.sum()

    salespersons = [
        'Budi Santoso','Siti Rahayu','Ahmad Fauzi','Dewi Lestari','Rizky Pratama',
        'Nina Kusuma','Hendra Wijaya','Rina Marlina','Doni Firmansyah','Yuli Astuti'
    ]
    customers = [f'CUST-{str(i).zfill(4)}' for i in range(1, 201)]
    stores = [
        ('Toko Pusat Jakarta Selatan', 'Jakarta',   'DKI Jakarta'),
        ('Toko Cabang Jakarta Barat',  'Jakarta',   'DKI Jakarta'),
        ('Toko Cabang Depok',          'Depok',     'Jawa Barat'),
        ('Toko Cabang Bekasi',         'Bekasi',    'Jawa Barat'),
        ('Toko Cabang Surabaya',       'Surabaya',  'Jawa Timur'),
        ('Toko Cabang Malang',         'Malang',    'Jawa Timur'),
        ('Toko Cabang Bandung',        'Bandung',   'Jawa Barat'),
        ('Toko Cabang Medan',          'Medan',     'Sumatera Utara'),
        ('Toko Cabang Makassar',       'Makassar',  'Sulawesi Selatan'),
        ('Toko Cabang Semarang',       'Semarang',  'Jawa Tengah'),
    ]
    channels   = ['Offline','Shopee','Tokopedia','Website','WhatsApp','Tiktok Shop']
    ch_w       = np.array([0.30,0.25,0.20,0.10,0.10,0.05])
    couriers   = ['JNE','J&T','SiCepat','Anteraja','Gosend']
    status_pool= ['Selesai']*6 + ['Diproses','Dikirim','Dikembalikan']

    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days

    records = []
    for i in range(n_records):
        day_off = np.random.randint(0, total_days)
        date    = start_date + timedelta(days=int(day_off))

        month_factor   = 1 + 0.3 * np.sin(2 * np.pi * (date.month - 3) / 12)
        weekend_factor = 1.25 if date.weekday() >= 5 else 1.0
        year_trend     = 1.15 if date.year == 2024 else 1.0

        pidx = np.random.choice(len(products), p=prod_w)
        prod_name, kategori, harga_base, hpp_pct = products[pidx]

        harga   = round(harga_base * np.random.uniform(0.92, 1.05), -2)
        hpp     = round(harga * hpp_pct, -2)
        max_qty = max(1, int(5_000_000 / harga_base))
        qty     = np.random.randint(1, max_qty + 1)

        channel = np.random.choice(channels, p=ch_w)
        sidx    = np.random.randint(0, len(stores))
        nama_toko, kota, provinsi = stores[sidx]

        if channel == 'Offline':
            salesperson = np.random.choice(salespersons)
            ongkir, courier = 0, '-'
        else:
            salesperson = np.random.choice(list(salespersons) + ['Online']*5)
            ongkir  = np.random.choice([15000,18000,20000,25000,35000])
            courier = np.random.choice(couriers)

        revenue     = max(harga, round(harga * qty * month_factor * weekend_factor * year_trend, -2))
        diskon      = round(revenue * np.random.choice([0,0,0,0,0,0.05,0.10,0.15]), -2)
        grand_total = revenue - diskon
        status      = np.random.choice(status_pool)
        customer    = np.random.choice(customers[:40] if np.random.random() < 0.30 else customers)
        no_order    = f"INV-{date.strftime('%Y%m%d')}-{str(i+1).zfill(5)}"

        records.append({
            'no_order': no_order, 'tanggal': date.strftime('%Y-%m-%d'),
            'nama_produk': prod_name, 'kategori': kategori,
            'harga_satuan': harga, 'hpp': hpp, 'qty': qty,
            'diskon': diskon, 'ongkir': ongkir, 'grand_total': grand_total,
            'pelanggan': customer, 'salesperson': salesperson,
            'nama_toko': nama_toko, 'kota': kota, 'provinsi': provinsi,
            'channel': channel, 'kurir': courier, 'status': status,
        })

    df = pd.DataFrame(records).sort_values('tanggal').reset_index(drop=True)

    # Inject anomali ~2%
    anom_idx = np.random.choice(len(df), size=max(1, int(len(df)*0.02)), replace=False)
    for idx in anom_idx:
        df.loc[idx, 'grand_total'] = round(df.loc[idx, 'grand_total'] * np.random.choice([0.10, 5.0]), -2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data generated: {len(df):,} records, {output_path}")
    return df


if __name__ == "__main__":
    print("Testing Utilities...")
    
    # Create sample data
    df = create_sample_data(n_records=500)
    
    # Test analyzer
    print("\n1. Testing SalesAnalyzer...")
    analyzer = SalesAnalyzer(df)
    
    kpis = analyzer.calculate_kpis()
    print(f"KPIs: {kpis}")
    
    insights = analyzer.generate_insights()
    print(f"\nInsights:\n" + "\n".join([f"- {i}" for i in insights]))
    
    # Test report generator
    print("\n2. Testing ReportGenerator...")
    reporter = ReportGenerator(analyzer)
    reporter.export_to_csv(output_dir='reports')
    
    # Test visualizer
    print("\n3. Testing Visualizer...")
    viz = Visualizer(df)
    fig = viz.create_revenue_trend_chart()
    print("Revenue trend chart created")
