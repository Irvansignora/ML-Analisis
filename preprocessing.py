"""
Preprocessing Module for Sales ML System
========================================
Modul untuk data cleaning, parsing, dan feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Kelas untuk preprocessing data penjualan
    """
    
    # Mapping kolom umum untuk standarisasi
    COLUMN_MAPPINGS = {
        # Date columns
        'date': ['date', 'tanggal', 'tgl', 'order_date', 'transaction_date', 'sale_date',
                 'timestamp', 'waktu', 'tanggal_transaksi', 'tanggal_penjualan',
                 'tanggal_order', 'tgl_transaksi', 'tgl_penjualan', 'tgl_order',
                 'created_at', 'updated_at', 'invoice_date', 'tanggal_faktur'],
        # Product columns
        'product': ['product', 'produk', 'product_name', 'nama_produk', 'item', 'item_name',
                    'nama_barang', 'barang', 'nama_item', 'deskripsi_produk', 'product_desc',
                    'sku', 'nama', 'goods', 'merchandise', 'komoditi', 'nama_produk/jasa'],
        # Quantity columns
        'quantity': ['quantity', 'qty', 'jumlah', 'amount', 'jml', 'jumlah_barang', 'unit',
                     'kuantitas', 'banyak', 'jml_barang', 'jumlah_unit', 'unit_terjual',
                     'qty_sold', 'terjual', 'jumlah_terjual', 'volume', 'pieces', 'pcs'],
        # Price columns
        'price': ['price', 'harga', 'unit_price', 'harga_satuan', 'price_per_unit', 'harga_jual',
                  'harga_per_unit', 'harga_item', 'unit_cost', 'selling_price',
                  'harga_pokok', 'hpp', 'cost', 'rate', 'tarif'],
        # Revenue/Sales columns
        'revenue': ['revenue', 'total', 'total_price', 'total_amount', 'penjualan', 'total_harga',
                    'sales', 'omset', 'total_revenue', 'subtotal', 'sub_total', 'grand_total',
                    'total_penjualan', 'total_bayar', 'total_pembayaran', 'nilai_penjualan',
                    'nilai', 'nominal', 'jumlah_total', 'total_transaksi', 'pendapatan',
                    'income', 'gross_sales', 'net_sales', 'total_sales', 'sale_amount',
                    'harga_total', 'total_harga_jual'],
        # Category columns
        'category': ['category', 'kategori', 'product_category', 'kategori_produk', 'type',
                     'jenis', 'tipe', 'divisi', 'division', 'group', 'grup', 'kelompok',
                     'product_group', 'product_type', 'jenis_produk', 'kelas'],
        # Customer columns
        'customer': ['customer', 'customer_id', 'pelanggan', 'customer_name', 'nama_pelanggan',
                     'buyer', 'pembeli', 'nama_pembeli', 'client', 'klien', 'konsumen',
                     'nama_customer', 'nama_klien', 'id_pelanggan'],
        # Region/Location columns
        'region': ['region', 'lokasi', 'location', 'area', 'city', 'kota', 'province',
                   'provinsi', 'wilayah', 'cabang', 'branch', 'store', 'toko', 'outlet',
                   'nama_toko', 'nama_cabang', 'alamat', 'address', 'district', 'kecamatan']
    }
    
    def __init__(self):
        self.original_columns = None
        self.mapped_columns = {}
        logger.info("DataPreprocessor initialized")
    
    def load_data(self, file_path: str, file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load data dari berbagai format file
        
        Parameters:
        -----------
        file_path : str
            Path ke file
        file_type : str, optional
            Tipe file ('csv', 'excel', 'json'). Auto-detect jika None
            
        Returns:
        --------
        pd.DataFrame
            DataFrame yang sudah dimuat
        """
        try:
            if file_type is None:
                # Auto-detect dari ekstensi
                if file_path.lower().endswith('.csv'):
                    file_type = 'csv'
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    file_type = 'excel'
                elif file_path.lower().endswith('.json'):
                    file_type = 'json'
                else:
                    raise ValueError(f"Format file tidak didukung: {file_path}")
            
            logger.info(f"Loading data from {file_path} (type: {file_type})")
            
            if file_type == 'csv':
                # Coba berbagai encoding
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Tipe file tidak didukung: {file_type}")
            
            self.original_columns = df.columns.tolist()
            logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standarisasi nama kolom ke format umum
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan kolom yang sudah distandarisasi
        """
        df = df.copy()
        original_cols = df.columns.tolist()
        new_columns = {}
        
        # Convert ke lowercase untuk matching
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        
        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            if standard_name in [v for v in new_columns.values()]:
                continue  # already mapped
            for col in df.columns:
                col_clean = col.lower().strip().replace(' ', '_')
                variations_lower = [v.lower().replace(' ', '_') for v in variations]
                # Exact match first
                if col_clean in variations_lower:
                    new_columns[col] = standard_name
                    self.mapped_columns[standard_name] = col
                    break
                # Partial/substring match as fallback
                for var in variations_lower:
                    if var in col_clean or col_clean in var:
                        if col not in new_columns:  # dont overwrite exact match
                            new_columns[col] = standard_name
                            self.mapped_columns[standard_name] = col
                        break
        
        df = df.rename(columns=new_columns)
        logger.info(f"Columns standardized: {new_columns}")
        return df
    
    def parse_date_column(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Parse kolom tanggal dengan berbagai format
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        date_column : str
            Nama kolom tanggal
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan kolom tanggal yang sudah diparse
        """
        df = df.copy()
        
        if date_column not in df.columns:
            logger.warning(f"Kolom {date_column} tidak ditemukan")
            return df
        
        # Jika sudah datetime, return
        if pd.api.types.is_datetime64_any_dtype(df[date_column]):
            logger.info(f"Kolom {date_column} sudah dalam format datetime")
            return df
        
        # Coba berbagai format tanggal
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%d-%m-%y', '%d/%m/%y',
            '%Y%m%d', '%d%m%Y',
            '%B %d, %Y', '%b %d, %Y',
            '%d %B %Y', '%d %b %Y'
        ]
        
        parsed = False
        for fmt in date_formats:
            try:
                df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                parsed = True
                logger.info(f"Tanggal diparse dengan format: {fmt}")
                break
            except:
                continue
        
        if not parsed:
            # Coba infer otomatis (BUG FIX: hapus infer_datetime_format yang deprecated di pandas 2.0+)
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                logger.info("Tanggal diparse dengan pd.to_datetime otomatis")
            except Exception as e:
                logger.error(f"Gagal parse tanggal: {str(e)}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values dengan berbagai strategi
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        strategy : dict
            Dictionary mapping kolom ke strategi ('mean', 'median', 'mode', 'drop', 'zero', 'ffill')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan missing values yang sudah dihandle
        """
        df = df.copy()
        
        # Default strategy
        default_strategy = {
            'quantity': 'median',
            'price': 'median',
            'revenue': 'median',
            'product': 'mode',
            'category': 'mode',
            'region': 'mode'
        }
        
        if strategy:
            default_strategy.update(strategy)
        
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values sebelum cleaning: {missing_before}")
        
        for col, strat in default_strategy.items():
            if col in df.columns and df[col].isnull().any():
                if strat == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strat == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif strat == 'mode':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                elif strat == 'drop':
                    df.dropna(subset=[col], inplace=True)
                elif strat == 'zero':
                    df[col].fillna(0, inplace=True)
                elif strat == 'ffill':
                    # BUG FIX: fillna(method='ffill') deprecated di pandas 2.0+, gunakan ffill()
                    df[col] = df[col].ffill()
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values setelah cleaning: {missing_after}")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        subset : list, optional
            Kolom yang dipertimbangkan untuk cek duplikat
            
        Returns:
        --------
        pd.DataFrame
            DataFrame tanpa duplikat
        """
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        after = len(df)
        
        removed = before - after
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def calculate_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kalkulasi revenue jika belum ada. Coba berbagai kombinasi kolom.
        """
        df = df.copy()
        
        if 'revenue' not in df.columns:
            if 'quantity' in df.columns and 'price' in df.columns:
                df['revenue'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0) *                                 pd.to_numeric(df['price'], errors='coerce').fillna(0)
                logger.info("Revenue dikalkulasi dari quantity * price")
            else:
                # Cari kolom numerik dengan nama yang mirip total/nilai/amount
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                revenue_keywords = ['total', 'amount', 'nilai', 'nominal', 'sales', 'omset',
                                    'pendapatan', 'income', 'bayar', 'harga_total']
                for col in numeric_cols:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in revenue_keywords):
                        df['revenue'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        logger.info(f"Revenue diambil dari kolom: {col}")
                        break
                else:
                    # Last resort: gunakan kolom numerik terbesar nilainya
                    if numeric_cols:
                        best_col = df[numeric_cols].mean().idxmax()
                        df['revenue'] = pd.to_numeric(df[best_col], errors='coerce').fillna(0)
                        logger.warning(f"Revenue fallback ke kolom: {best_col}")
                    else:
                        logger.warning("Tidak bisa kalkulasi revenue: tidak ada kolom numerik")
        else:
            # Pastikan revenue sudah numeric
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto feature engineering untuk data penjualan
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan fitur baru
        """
        df = df.copy()
        
        # Time-based features
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)  # BUG FIX: cast UInt32 → int
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month_name'] = df['date'].dt.month_name()
            df['year_month'] = df['date'].dt.to_period('M').astype(str)
            logger.info("Time-based features created")
        
        # Price-based features
        if 'price' in df.columns and 'quantity' in df.columns:
            # Revenue per unit analysis
            df['revenue_per_unit'] = df['revenue'] / df['quantity']
            
            # Price tier
            price_q25 = df['price'].quantile(0.25)
            price_q75 = df['price'].quantile(0.75)
            df['price_tier'] = pd.cut(
                df['price'],
                bins=[0, price_q25, price_q75, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
        
        # Product performance features
        if 'product' in df.columns:
            # Frequency encoding
            product_freq = df['product'].value_counts()
            df['product_freq'] = df['product'].map(product_freq)
            
            # Average revenue per product
            if 'revenue' in df.columns:
                product_avg_revenue = df.groupby('product')['revenue'].transform('mean')
                df['product_avg_revenue'] = product_avg_revenue
        
        # Lag features untuk time series
        if 'date' in df.columns and 'revenue' in df.columns:
            df = df.sort_values('date')
            df['revenue_lag_1'] = df['revenue'].shift(1)
            df['revenue_lag_7'] = df['revenue'].shift(7)
            df['revenue_rolling_mean_7'] = df['revenue'].rolling(window=7).mean()
            df['revenue_rolling_std_7'] = df['revenue'].rolling(window=7).std()
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def preprocess(self, df: pd.DataFrame, auto_clean: bool = True) -> pd.DataFrame:
        """
        Pipeline preprocessing lengkap
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        auto_clean : bool
            Jika True, lakukan cleaning otomatis
            
        Returns:
        --------
        pd.DataFrame
            DataFrame yang sudah dipreprocess
        """
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Standarisasi kolom
        df = self.standardize_columns(df)
        
        # 2. Parse tanggal
        df = self.parse_date_column(df)
        
        # 3. Handle missing values
        df = self.handle_missing_values(df)
        
        # 4. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 5. Kalkulasi revenue
        df = self.calculate_revenue(df)
        
        # 6. Feature engineering
        df = self.feature_engineering(df)
        
        logger.info("Preprocessing completed!")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary dari data
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
            
        Returns:
        --------
        dict
            Dictionary berisi summary data
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'date_range': None,
            'numeric_summary': None
        }
        
        if 'date' in df.columns:
            summary['date_range'] = {
                'min': df['date'].min().strftime('%Y-%m-%d'),
                'max': df['date'].max().strftime('%Y-%m-%d'),
                'span_days': (df['date'].max() - df['date'].min()).days
            }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return summary


def preprocess_file(file_path: str, file_type: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Fungsi helper untuk preprocess file secara langsung
    
    Parameters:
    -----------
    file_path : str
        Path ke file
    file_type : str, optional
        Tipe file
        
    Returns:
    --------
    tuple
        (DataFrame yang sudah dipreprocess, summary dict)
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(file_path, file_type)
    df = preprocessor.preprocess(df)
    summary = preprocessor.get_data_summary(df)
    return df, summary


if __name__ == "__main__":
    # Test dengan sample data
    print("Testing DataPreprocessor...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Tanggal': pd.date_range('2024-01-01', periods=100, freq='D'),
        'Nama Produk': ['Product A', 'Product B', 'Product C'] * 33 + ['Product A'],
        'Jumlah': np.random.randint(1, 50, 100),
        'Harga Satuan': np.random.uniform(10000, 100000, 100),
        'Kategori': ['Electronics', 'Clothing', 'Food'] * 33 + ['Electronics']
    })
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess(sample_data)
    
    print(f"\nOriginal columns: {sample_data.columns.tolist()}")
    print(f"Processed columns: {processed_df.columns.tolist()}")
    print(f"\nShape: {processed_df.shape}")
    print(f"\nFirst few rows:")
    print(processed_df.head())
