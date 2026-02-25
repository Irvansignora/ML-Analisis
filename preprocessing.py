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
    
    # ── COLUMN MAPPINGS ──────────────────────────────────────────────────────────
    # PENTING: urutan dict menentukan prioritas matching.
    # Kolom yang lebih spesifik (store, customer, region) harus didefinisikan
    # SEBELUM kolom yang lebih generik (product) agar tidak salah petakan.
    # Setiap keyword HANYA boleh ada di SATU mapping untuk hindari konflik.
    COLUMN_MAPPINGS = {
        # ── Date ──
        'date': [
            'date', 'tanggal', 'tgl', 'order_date', 'transaction_date', 'sale_date',
            'timestamp', 'waktu', 'tanggal_transaksi', 'tanggal_penjualan',
            'tanggal_order', 'tgl_transaksi', 'tgl_penjualan', 'tgl_order',
            'created_at', 'updated_at', 'invoice_date', 'tanggal_faktur',
            'tgl_order', 'tgl_transaksi', 'tgl_faktur', 'tgl_invoice',
        ],
        # ── Order / Invoice ──
        'order_id': [
            'no_pesanan', 'no_order', 'order_id', 'id_order', 'no_invoice',
            'invoice', 'no_faktur', 'faktur', 'ref', 'kode_transaksi',
            'nomor_pesanan', 'nomor_order', 'nomor_invoice', 'nomor_faktur',
            'transaction_id', 'id_transaksi',
        ],
        # ── Customer ── (sebelum product agar 'nama_pelanggan' tidak nyasar ke product)
        'customer': [
            'customer', 'customer_id', 'id_customer', 'customer_name',
            'nama_customer', 'nama_pelanggan', 'pelanggan', 'id_pelanggan',
            'buyer', 'nama_pembeli', 'pembeli',
            'client', 'klien', 'nama_klien', 'id_klien',
            'konsumen', 'nama_konsumen',
            'member', 'nama_member', 'id_member',
            'pemesan', 'nama_pemesan',
        ],
        # ── Store / Toko ── (sebelum product & region agar 'nama_toko' tidak nyasar)
        'store': [
            'store', 'nama_store', 'id_store',
            'toko', 'nama_toko', 'id_toko', 'kode_toko',
            'outlet', 'nama_outlet', 'id_outlet', 'kode_outlet',
            'gerai', 'nama_gerai',
            'shop', 'nama_shop',
            'warung', 'lapak',
        ],
        # ── Channel / Marketplace ── (sebelum category agar 'channel' tidak nyasar)
        'channel': [
            'channel', 'nama_channel', 'sales_channel',
            'marketplace', 'platform', 'nama_platform',
            'saluran', 'saluran_penjualan', 'sumber', 'sumber_penjualan',
            'media', 'media_penjualan',
            'shopee', 'tokopedia', 'lazada', 'bukalapak', 'blibli', 'tiktok_shop',
        ],
        # ── Region / Location ── (cabang/branch di sini, bukan di store)
        'region': [
            'region', 'nama_region', 'id_region', 'kode_region',
            'lokasi', 'location',
            'area', 'nama_area', 'kode_area',
            'city', 'kota', 'nama_kota', 'kode_kota',
            'province', 'provinsi', 'nama_provinsi',
            'wilayah', 'nama_wilayah', 'kode_wilayah',
            'cabang', 'nama_cabang', 'kode_cabang', 'id_cabang',
            'branch', 'nama_branch',
            'district', 'kecamatan', 'kelurahan',
            'daerah', 'zona', 'zone',
        ],
        # ── Category ──
        'category': [
            'category', 'kategori', 'nama_kategori', 'kode_kategori',
            'product_category', 'kategori_produk',
            'type', 'tipe', 'jenis', 'jenis_produk',
            'divisi', 'division',
            'group', 'grup', 'kelompok', 'product_group', 'product_type',
            'kelas', 'kelas_produk',
            'brand', 'merek', 'merk',
            'sub_category', 'sub_kategori',
        ],
        # ── Product ── (SETELAH store/customer/channel/region agar tidak salah tangkap)
        # TIDAK boleh pakai 'nama' saja karena terlalu generik → konflik dengan nama_toko dll
        'product': [
            'product', 'id_product', 'kode_product',
            'produk', 'nama_produk', 'id_produk', 'kode_produk',
            'product_name', 'product_desc', 'product_description',
            'item', 'item_name', 'item_desc', 'nama_item', 'kode_item',
            'barang', 'nama_barang', 'kode_barang', 'id_barang',
            'goods', 'merchandise',
            'sku', 'kode_sku',
            'komoditi', 'komoditas',
            'nama_produk/jasa', 'produk/jasa',
            'deskripsi_produk', 'deskripsi_barang',
        ],
        # ── Quantity ──
        'quantity': [
            'quantity', 'qty', 'jumlah', 'jml',
            'jumlah_barang', 'jml_barang', 'jumlah_unit', 'jml_unit',
            'jumlah_terjual', 'unit_terjual', 'qty_sold',
            'terjual', 'banyak', 'volume',
            'pieces', 'pcs', 'unit',
            'kuantitas',
        ],
        # ── Price ──
        'price': [
            'price', 'unit_price', 'price_per_unit', 'selling_price',
            'harga', 'harga_satuan', 'harga_jual', 'harga_per_unit',
            'harga_item', 'harga_produk', 'harga_barang',
            'unit_cost', 'harga_pokok', 'hpp', 'cost',
            'rate', 'tarif', 'tariff',
        ],
        # ── Revenue / Sales ──
        'revenue': [
            'revenue', 'total_revenue', 'gross_sales', 'net_sales',
            'total_sales', 'sale_amount', 'sales_amount',
            'grand_total', 'total_price', 'total_amount',
            'penjualan', 'total_penjualan', 'nilai_penjualan',
            'total_harga', 'harga_total', 'total_harga_jual',
            'total_bayar', 'total_pembayaran',
            'omset', 'pendapatan', 'income',
            'subtotal', 'sub_total',
            'nominal', 'nilai', 'jumlah_total', 'total_transaksi',
            'total',
        ],
        # ── Courier / Ekspedisi ──
        'courier': [
            'kurir', 'courier', 'ekspedisi', 'jasa_kirim', 'pengiriman',
            'shipping_courier', 'nama_kurir',
        ],
        # ── Salesperson ── (SEBELUM status agar tidak nyasar)
        'salesperson': [
            'salesperson', 'sales_person', 'sales', 'nama_sales', 'agen',
            'sales_name', 'nama_agen', 'nama_salesperson',
            'marketing', 'nama_marketing', 'marketer',
            'staff_penjualan', 'petugas_penjualan',
        ],
        # ── Status ──
        'status': [
            'status', 'status_order', 'status_transaksi', 'status_pembayaran',
            'kondisi', 'order_status', 'transaction_status', 'payment_status',
            'status_pengiriman', 'status_penjualan',
        ],
        # ── Discount ──
        'discount': [
            'diskon', 'discount', 'potongan', 'potongan_harga', 'promo',
            'voucher', 'cashback',
        ],
        # ── Shipping Cost ──
        'shipping': [
            'ongkir', 'ongkos_kirim', 'biaya_kirim', 'biaya_pengiriman',
            'shipping', 'shipping_cost', 'freight',
        ],
        # ── Tax ──
        'tax': [
            'pajak', 'tax', 'ppn', 'vat', 'tax_amount',
        ],
    }
    
    # Keyword yang DILARANG untuk partial match product
    # → hindari 'nama_toko', 'nama_pelanggan', 'nama_cabang' dll nyasar ke product
    _PRODUCT_PARTIAL_BLACKLIST = [
        'toko', 'store', 'outlet', 'gerai', 'warung', 'lapak', 'shop',
        'pelanggan', 'customer', 'pembeli', 'buyer', 'klien', 'konsumen',
        'member', 'pemesan',
        'cabang', 'branch', 'region', 'wilayah', 'area', 'kota', 'lokasi',
        'channel', 'marketplace', 'platform', 'saluran',
        'kurir', 'courier', 'ekspedisi',
        'status', 'kondisi',
        'tanggal', 'date', 'tgl', 'waktu',
        'harga', 'price', 'total', 'revenue', 'omset', 'bayar',
        'qty', 'jumlah', 'quantity',
        'diskon', 'discount', 'ongkir', 'pajak', 'tax',
        'invoice', 'faktur', 'order', 'transaksi',
        'sales', 'salesperson', 'agen', 'marketing',
    ]

    # Keyword yang DILARANG untuk partial match customer
    # → hindari 'salesperson', 'status' dll nyasar ke customer
    _CUSTOMER_PARTIAL_BLACKLIST = [
        'sales', 'salesperson', 'agen', 'marketing', 'marketer',
        'status', 'kondisi', 'channel', 'kurir', 'courier',
        'produk', 'product', 'barang', 'item',
        'toko', 'store', 'outlet',
    ]
    
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
        Standarisasi nama kolom ke format umum.
        
        Logika matching (berurutan, stop di match pertama per kolom):
        1. Exact match  → paling aman, langsung petakan
        2. Partial match → hati-hati, ada blacklist untuk 'product'
           agar 'nama_toko', 'nama_pelanggan' dll tidak nyasar ke product
        """
        df = df.copy()
        new_columns = {}   # col_clean → standard_name
        mapped_std  = set()  # standard_name yang sudah terpetakan

        # Normalisasi semua nama kolom: lowercase + underscore
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        all_cols = list(df.columns)

        # ── PASS 1: EXACT MATCH ──────────────────────────────────────────────
        # Prioritas tinggi: satu kolom → satu standard_name, first-win
        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            if standard_name in mapped_std:
                continue
            var_set = {v.lower().replace(' ', '_') for v in variations}
            for col in all_cols:
                if col in new_columns:
                    continue  # kolom ini sudah dipetakan
                if col in var_set:
                    new_columns[col] = standard_name
                    mapped_std.add(standard_name)
                    self.mapped_columns[standard_name] = col
                    break

        # ── PASS 2: PARTIAL / SUBSTRING MATCH ───────────────────────────────
        # Hanya untuk kolom & standard_name yang belum terpetakan
        blacklist = {b.lower() for b in self._PRODUCT_PARTIAL_BLACKLIST}

        # BUG FIX: kolom-kolom ini TIDAK BOLEH di-remap ke standard name lain
        # meski belum terpetakan di PASS 1 (misalnya jika tidak ada exact match).
        # Ini mencegah kolom 'status' nyasar ke 'product' lewat partial match.
        _HARD_RESERVED_COLS = {
            'status', 'status_order', 'status_transaksi', 'status_pembayaran',
            'order_status', 'transaction_status', 'payment_status',
            'kondisi', 'kurir', 'courier', 'ekspedisi',
            'diskon', 'discount', 'ongkir', 'shipping', 'pajak', 'tax',
        }

        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            if standard_name in mapped_std:
                continue
            var_set = [v.lower().replace(' ', '_') for v in variations]
            for col in all_cols:
                if col in new_columns:
                    continue

                # BUG FIX: jika kolom ini ada di hard-reserved list, skip total —
                # jangan biarkan diremapping ke standard name apapun via partial match
                if col in _HARD_RESERVED_COLS:
                    continue

                # Untuk 'product': tolak kolom yang mengandung kata dari blacklist
                if standard_name == 'product':
                    col_words = set(col.split('_'))
                    if col_words & blacklist:
                        continue  # skip - kolom ini bukan produk

                # Untuk 'customer': tolak kolom yang mengandung kata dari customer blacklist
                if standard_name == 'customer':
                    col_words = set(col.split('_'))
                    cust_bl = {b.lower() for b in self._CUSTOMER_PARTIAL_BLACKLIST}
                    if col_words & cust_bl:
                        continue  # skip - kolom ini bukan customer

                # Cek substring match (var ada di dalam col, atau col ada di dalam var)
                for var in var_set:
                    if len(var) >= 3 and (var in col or col in var):
                        new_columns[col] = standard_name
                        mapped_std.add(standard_name)
                        self.mapped_columns[standard_name] = col
                        break
                if standard_name in mapped_std:
                    break

        df = df.rename(columns=new_columns)
        logger.info(f"Columns standardized: {new_columns}")

        # ── POST-VALIDATION: pastikan kolom 'product' tidak berisi data status ──
        # Double-check: meskipun PASS 1 & 2 sudah diperketat, ini safety net terakhir.
        _STATUS_LIKE_VALUES = {
            'selesai', 'completed', 'complete', 'sukses', 'success',
            'dikirim', 'shipped', 'diproses', 'processing', 'pending',
            'menunggu', 'cancel', 'cancelled', 'dibatalkan',
            'return', 'returned', 'dikembalikan', 'refund',
            'gagal', 'failed', 'lunas', 'unpaid', 'paid',
        }
        if 'product' in df.columns:
            sample_vals = set(
                df['product'].dropna().astype(str).str.lower().str.strip().unique()[:30]
            )
            overlap = sample_vals & _STATUS_LIKE_VALUES
            # Jika >40% nilai unik di kolom 'product' adalah nilai status → kolom salah mapping
            if len(sample_vals) > 0 and len(overlap) / len(sample_vals) > 0.4:
                logger.warning(
                    f"BUG FIX: Kolom 'product' terdeteksi berisi data status ({overlap}). "
                    f"Mapping dibatalkan — kolom akan di-drop dan diisi fallback."
                )
                df = df.drop(columns=['product'])
                if 'product' in self.mapped_columns:
                    del self.mapped_columns['product']
                if 'product' in mapped_std:
                    mapped_std.discard('product')

        # ── POST-VALIDATION: pastikan kolom 'category' tidak berisi data status ──
        if 'category' in df.columns:
            sample_cat = set(
                df['category'].dropna().astype(str).str.lower().str.strip().unique()[:30]
            )
            overlap_cat = sample_cat & _STATUS_LIKE_VALUES
            if len(sample_cat) > 0 and len(overlap_cat) / len(sample_cat) > 0.4:
                logger.warning(
                    f"BUG FIX: Kolom 'category' terdeteksi berisi data status ({overlap_cat}). "
                    f"Mapping dibatalkan."
                )
                df = df.drop(columns=['category'])
                if 'category' in self.mapped_columns:
                    del self.mapped_columns['category']

        # ── FALLBACK: pastikan kolom wajib ada ───────────────────────────────
        # BUG FIX: fallback product WAJIB cek isi kolom — tidak boleh pakai
        # kolom yang isinya status/nilai transaksi.
        def _col_has_status_values(series) -> bool:
            """Kembalikan True jika kolom ini isinya nilai-nilai status transaksi."""
            sample = set(series.dropna().astype(str).str.lower().str.strip().unique()[:20])
            if len(sample) == 0:
                return False
            return len(sample & _STATUS_LIKE_VALUES) / len(sample) > 0.4

        if 'product' not in df.columns:
            placed = False
            for fallback in ['category', 'channel']:
                if fallback in df.columns:
                    col_dtype = df[fallback].dtype
                    if (col_dtype == object or str(col_dtype) == 'string') \
                            and not _col_has_status_values(df[fallback]):
                        df['product'] = df[fallback].astype(str)
                        logger.info(f"product proxy: '{fallback}'")
                        placed = True
                        break
            if not placed:
                # Tidak ada kolom yang layak jadi proxy product
                # Jangan paksa — biarkan chart produk tidak muncul daripada salah data
                logger.warning("Kolom 'product' tidak ditemukan dan tidak ada fallback yang valid. "
                               "Chart produk tidak akan ditampilkan.")
                # TIDAK di-set ke 'Unknown' agar _is_product_col_valid di app.py return False

        # category fallback — status TIDAK boleh jadi proxy category
        if 'category' not in df.columns:
            placed_cat = False
            for fallback in ['channel', 'courier']:
                if fallback in df.columns and not _col_has_status_values(df[fallback]):
                    df['category'] = df[fallback].astype(str)
                    logger.info(f"category proxy: '{fallback}'")
                    placed_cat = True
                    break
            if not placed_cat:
                df['category'] = 'General'
                logger.warning("Kolom 'category' tidak ditemukan, diisi 'General'")

        # region fallback: kalau ada store, bisa jadi proxy region juga
        if 'region' not in df.columns and 'store' in df.columns:
            df['region'] = df['store'].astype(str)
            logger.info("region proxy: 'store'")

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
                    # BUG FIX: fillna(inplace=True) silently fails di pandas 2.x (chained indexing)
                    # Gunakan assignment langsung agar perubahan benar-benar tersimpan
                    df[col] = df[col].fillna(df[col].mean())
                elif strat == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif strat == 'mode':
                    fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(fill_val)
                elif strat == 'drop':
                    df = df.dropna(subset=[col])
                elif strat == 'zero':
                    df[col] = df[col].fillna(0)
                elif strat == 'ffill':
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
    
    # ── STATUS SUCCESS VALUES ─────────────────────────────────────────────────
    # Hanya baris dengan status ini yang dihitung sebagai revenue.
    # Baris cancel/return/failed → revenue di-nol, tapi tetap ada di DataFrame
    # supaya chart distribusi status tetap akurat.
    STATUS_SUCCESS_VALUES = {
        'selesai', 'completed', 'complete', 'sukses', 'success',
        'berhasil', 'lunas', 'paid', 'delivered', 'terkirim',
        'diterima', 'confirmed', 'terkonfirmasi', 'approved', 'disetujui',
        'finish', 'done', 'selesai_kirim',
        'pesanan_selesai', 'order_selesai', 'transaksi_selesai',
        'order_completed', 'order_done',
    }

    def apply_status_revenue_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Jika ada kolom 'status', set revenue = 0 untuk baris yang bukan sukses/completed.

        - Baris cancel/return TETAP ADA → chart distribusi status akurat
        - Revenue di-nol-kan → KPI, trend, top produk hanya hitung transaksi berhasil
        - Kolom 'is_successful_transaction' (bool) ditambahkan sebagai marker
        """
        df = df.copy()

        if 'status' not in df.columns or 'revenue' not in df.columns:
            df['is_successful_transaction'] = True
            return df

        status_norm = df['status'].astype(str).str.lower().str.strip()
        is_success  = status_norm.isin(self.STATUS_SUCCESS_VALUES)

        # Fallback: jika tidak ada satu pun yang match, anggap semua valid
        if is_success.sum() == 0:
            logger.warning(
                f"apply_status_revenue_filter: tidak ada nilai status yang cocok. "
                f"Nilai unik: {status_norm.unique()[:10].tolist()}. "
                "Revenue tidak difilter (semua dianggap valid)."
            )
            df['is_successful_transaction'] = True
            return df

        n_zeroed = (~is_success).sum()
        if n_zeroed > 0:
            df.loc[~is_success, 'revenue'] = 0
            logger.info(
                f"apply_status_revenue_filter: {n_zeroed} baris non-sukses di-nol revenue-nya "
                f"(status: {status_norm[~is_success].value_counts().to_dict()})"
            )

        df['is_successful_transaction'] = is_success
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
            # BUG FIX: pd.cut gagal kalau q25 == q75 (semua harga sama) → pakai duplicates='drop'
            try:
                df['price_tier'] = pd.cut(
                    df['price'],
                    bins=[0, price_q25, price_q75, float('inf')],
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
            except Exception:
                df['price_tier'] = 'Medium'
        
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
        
        # 5b. BUG FIX: jika ada kolom 'status', zero-out revenue baris non-sukses.
        #     Baris cancel/return tetap ada untuk chart status, tapi revenue-nya 0
        #     sehingga KPI & trend hanya hitung transaksi berhasil.
        df = self.apply_status_revenue_filter(df)
        
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
