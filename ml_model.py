"""
Machine Learning Module for Sales Analysis
==========================================
Modul untuk predictive analytics, clustering, dan anomaly detection
Versi yang sudah di-upgrade dengan model terbaik dan bug fixes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, IsolationForest,
                               GradientBoostingRegressor, ExtraTreesRegressor,
                               VotingRegressor, StackingRegressor)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              silhouette_score, mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Optional imports dengan error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet tidak terinstall. Time series forecasting dengan Prophet tidak tersedia.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost tidak terinstall. XGBoost regressor tidak tersedia.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM tidak terinstall. LightGBM regressor tidak tersedia.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalesForecaster:
    """
    Kelas untuk forecasting penjualan dengan berbagai model.
    Mendukung: linear, ridge, lasso, random_forest, extra_trees,
               gradient_boosting, xgboost, lightgbm, prophet, ensemble, stacking
    """
    
    AVAILABLE_MODELS = [
        'linear', 'ridge', 'lasso',
        'random_forest', 'extra_trees', 'gradient_boosting',
        'xgboost', 'lightgbm', 'prophet', 'ensemble', 'stacking'
    ]
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize forecaster

        Parameters:
        -----------
        model_type : str
            Tipe model (lihat AVAILABLE_MODELS)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # Lebih robust terhadap outlier vs StandardScaler
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'revenue'
        self.metrics = {}
        self.is_fitted = False
        
        logger.info(f"SalesForecaster initialized with model: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'revenue') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features untuk training
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        target_col : str
            Nama kolom target
            
        Returns:
        --------
        tuple
            (X, y)
        """
        df = df.copy()
        
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target dan derived features
        exclude_cols = [target_col, 'revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_mean_7', 
                       'revenue_rolling_std_7', 'revenue_per_unit']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        return X, y
    
    def fit(self, df: pd.DataFrame, target_col: str = 'revenue', 
            test_size: float = 0.2, do_tuning: bool = False) -> Dict:
        """
        Train model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame training
        target_col : str
            Kolom target
        test_size : float
            Proporsi data test
        do_tuning : bool
            Jika True, lakukan hyperparameter tuning
            
        Returns:
        --------
        dict
            Metrics evaluasi
        """
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'prophet':
            return self._fit_prophet(df, target_col)
        
        # Prepare features
        X, y = self.prepare_features(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled_arr = self.scaler.fit_transform(X_train)
        X_test_scaled_arr  = self.scaler.transform(X_test)
        
        # BUG FIX: LightGBM & XGBoost ditraining dengan feature names (DataFrame),
        # tapi scaler.transform() mengembalikan numpy array -> menyebabkan warning.
        # Solusi: bungkus kembali ke DataFrame dengan nama kolom yang sama.
        X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns, index=X_train.index)
        X_test_scaled  = pd.DataFrame(X_test_scaled_arr,  columns=X_test.columns,  index=X_test.index)
        
        # Initialize model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0, max_iter=10000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=200, random_state=42,
                                               n_jobs=-1, min_samples_leaf=2)
        elif self.model_type == 'extra_trees':
            self.model = ExtraTreesRegressor(n_estimators=200, random_state=42,
                                             n_jobs=-1, min_samples_leaf=2)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=200, random_state=42,
                                                    learning_rate=0.05, max_depth=5,
                                                    subsample=0.8)
        elif self.model_type == 'xgboost':
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(n_estimators=300, random_state=42,
                                               n_jobs=-1, learning_rate=0.05,
                                               max_depth=6, subsample=0.8,
                                               colsample_bytree=0.8, tree_method='hist')
            else:
                logger.warning("XGBoost tidak tersedia, menggunakan GradientBoosting")
                self.model = GradientBoostingRegressor(n_estimators=200, random_state=42,
                                                        learning_rate=0.05)
        elif self.model_type == 'lightgbm':
            if LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMRegressor(n_estimators=300, random_state=42,
                                                n_jobs=-1, learning_rate=0.05,
                                                num_leaves=63, subsample=0.8,
                                                colsample_bytree=0.8, verbose=-1)
            else:
                logger.warning("LightGBM tidak tersedia, menggunakan GradientBoosting")
                self.model = GradientBoostingRegressor(n_estimators=200, random_state=42,
                                                        learning_rate=0.05)
        elif self.model_type == 'ensemble':
            # Voting Ensemble dari model-model terbaik
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42,
                                                  learning_rate=0.05)),
                ('et', ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1))
            ]
            if XGBOOST_AVAILABLE:
                estimators.append(('xgb', xgb.XGBRegressor(n_estimators=200, random_state=42,
                                                              n_jobs=-1, learning_rate=0.05,
                                                              tree_method='hist')))
            self.model = VotingRegressor(estimators=estimators, n_jobs=-1)
        elif self.model_type == 'stacking':
            # Stacking: base learners + meta learner Ridge
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42,
                                                  learning_rate=0.1)),
                ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            ]
            if XGBOOST_AVAILABLE:
                estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42,
                                                              n_jobs=-1, tree_method='hist')))
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                n_jobs=-1,
                cv=3
            )
        else:
            raise ValueError(f"Model type tidak dikenal: {self.model_type}. Pilihan: {self.AVAILABLE_MODELS}")
        
        # Hyperparameter tuning
        if do_tuning and self.model_type in ['random_forest', 'xgboost']:
            self._hyperparameter_tuning(X_train_scaled, y_train)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'train_r2': r2_score(y_train, y_pred_train),
            'train_mape': float(mean_absolute_percentage_error(y_train, y_pred_train) * 100),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mape': float(mean_absolute_percentage_error(y_test, y_pred_test) * 100),
        }
        
        self.is_fitted = True
        logger.info(f"Training completed. Test RMSE: {self.metrics['test_rmse']:.2f}")
        
        return self.metrics
    
    def _fit_prophet(self, df: pd.DataFrame, target_col: str) -> Dict:
        """
        Train Prophet model untuk time series
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet tidak terinstall")
        
        logger.info("Training Prophet model...")
        
        # Prepare data untuk Prophet
        prophet_df = df[['date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Split data
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        # Train
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        self.model.fit(train_df)
        
        # Predict
        future = self.model.make_future_dataframe(periods=len(test_df))
        forecast = self.model.predict(future)
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = forecast.iloc[split_idx:]['yhat'].values
        
        self.metrics = {
            'test_mae': mean_absolute_error(y_true, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'test_r2': r2_score(y_true, y_pred)
        }
        
        self.is_fitted = True
        logger.info(f"Prophet training completed. Test RMSE: {self.metrics['test_rmse']:.2f}")
        
        return self.metrics
    
    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Hyperparameter tuning dengan GridSearchCV
        """
        logger.info("Starting hyperparameter tuning...")
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif self.model_type == 'extra_trees':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_leaf': [1, 2]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.7, 0.9]
            }
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        else:
            return
        
        # Use TimeSeriesSplit for time-aware CV
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            self.model, param_grid,
            cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prediksi dengan model yang sudah trained
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame untuk prediksi
            
        Returns:
        --------
        np.ndarray
            Hasil prediksi
        """
        if not self.is_fitted:
            raise ValueError("Model belum di-training. Panggil fit() terlebih dahulu.")
        
        if self.model_type == 'prophet':
            future_periods = len(df)
            future = self.model.make_future_dataframe(periods=future_periods)
            forecast = self.model.predict(future)
            return forecast.iloc[-future_periods:]['yhat'].values
        
        # Prepare features - encode kategorik dulu kalau perlu
        df_pred = df.copy()
        for col, le in self.label_encoders.items():
            if col in df_pred.columns:
                # Handle nilai yang tidak dikenal saat training
                def safe_encode(val):
                    s = str(val)
                    return le.transform([s])[0] if s in le.classes_ else 0
                df_pred[col] = df_pred[col].apply(safe_encode)
        
        X = df_pred[self.feature_columns].fillna(0)
        # Pastikan semua kolom numerik
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled_arr = self.scaler.transform(X)
        # BUG FIX: wrap kembali ke DataFrame agar LightGBM/XGBoost tidak warning feature names
        X_scaled = pd.DataFrame(X_scaled_arr, columns=self.feature_columns, index=X.index)
        
        return self.model.predict(X_scaled)
    
    def forecast_future(self, df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        """
        Forecast untuk periode mendatang
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame historis
        periods : int
            Jumlah periode yang akan diforecast
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan forecast
        """
        if self.model_type == 'prophet' and PROPHET_AVAILABLE:
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        # Untuk model lain, generate future features
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Create future dataframe dengan features rata-rata
        future_df = pd.DataFrame({'date': future_dates})
        
        # Add time features
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['quarter'] = future_df['date'].dt.quarter
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['day_of_month'] = future_df['date'].dt.day
        future_df['week_of_year'] = future_df['date'].dt.isocalendar().week
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        
        # BUG FIX: isi feature columns dengan nilai yang sesuai tipenya
        # - numerik  → median (representatif)
        # - string/kategorikal → mode (nilai terbanyak), lalu label-encode
        for col in self.feature_columns:
            if col not in future_df.columns:
                if col not in df.columns:
                    future_df[col] = 0
                    continue
                
                col_dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(col_dtype):
                    future_df[col] = df[col].median()
                else:
                    # Kategorikal: pakai mode, lalu encode sama seperti training
                    mode_val = df[col].mode()
                    mode_str = str(mode_val.iloc[0]) if not mode_val.empty else 'Unknown'
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Kalau mode_str ada di classes, encode; kalau tidak, pakai 0
                        if mode_str in le.classes_:
                            future_df[col] = le.transform([mode_str])[0]
                        else:
                            future_df[col] = 0
                    else:
                        future_df[col] = 0
        
        # Predict
        predictions = self.predict(future_df)
        future_df['forecast'] = predictions
        
        return future_df[['date', 'forecast']]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance dari model
        
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model belum di-training")
        
        if self.model_type == 'prophet':
            return pd.DataFrame({'feature': ['Prophet does not provide feature importance'], 'importance': [0]})
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame({'feature': ['Model does not support feature importance'], 'importance': [0]})
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save model ke file
        
        Parameters:
        -----------
        filepath : str
            Path untuk menyimpan model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model dari file
        
        Parameters:
        -----------
        filepath : str
            Path ke file model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.metrics = model_data['metrics']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")


class ProductSegmenter:
    """
    Kelas untuk segmentasi produk dengan clustering
    """
    
    def __init__(self, n_clusters: int = 4):
        """
        Initialize segmenter
        
        Parameters:
        -----------
        n_clusters : int
            Jumlah cluster
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.cluster_profiles = None
        
        logger.info(f"ProductSegmenter initialized with {n_clusters} clusters")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features untuk clustering
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
            
        Returns:
        --------
        pd.DataFrame
            Features untuk clustering
        """
        # Aggregate per product
        product_stats = df.groupby('product').agg({
            'revenue': ['sum', 'mean', 'count'],
            'quantity': ['sum', 'mean'],
            'price': 'mean'
        }).reset_index()
        
        # Flatten column names
        product_stats.columns = ['product', 'total_revenue', 'avg_revenue', 'transaction_count',
                                'total_quantity', 'avg_quantity', 'avg_price']
        
        return product_stats
    
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit clustering model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan cluster labels
        """
        logger.info("Fitting product segmentation...")
        
        # Prepare features
        product_stats = self.prepare_features(df)
        self.product_ids = product_stats['product'].values
        
        # Select numeric features
        feature_cols = ['total_revenue', 'avg_revenue', 'transaction_count', 
                       'total_quantity', 'avg_quantity', 'avg_price']
        X = product_stats[feature_cols].fillna(0)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit
        self.model.fit(X_scaled)
        labels = self.model.labels_
        
        # Add labels
        product_stats['cluster'] = labels
        
        # Calculate cluster profiles
        self.cluster_profiles = product_stats.groupby('cluster')[feature_cols].mean()
        
        # Assign cluster names berdasarkan karakteristik
        cluster_names = self._assign_cluster_names(self.cluster_profiles)
        product_stats['segment'] = product_stats['cluster'].map(cluster_names)
        
        self.is_fitted = True
        self.product_stats = product_stats
        
        # BUG FIX: silhouette_score requires n_labels >= 2 AND n_labels < n_samples
        # Jika kondisi tidak terpenuhi, skip silhouette (jangan crash)
        n_samples = len(X_scaled)
        n_unique_labels = len(np.unique(labels))
        if 2 <= n_unique_labels <= n_samples - 1:
            silhouette = silhouette_score(X_scaled, labels)
            logger.info(f"Segmentation completed. Silhouette score: {silhouette:.3f}")
        else:
            logger.warning(
                f"Silhouette score tidak bisa dihitung: "
                f"n_samples={n_samples}, n_clusters={n_unique_labels}. "
                f"Perlu minimal 2 cluster DAN n_samples > n_clusters."
            )
        
        return product_stats
    
    def _assign_cluster_names(self, profiles: pd.DataFrame) -> Dict[int, str]:
        """
        Assign nama segment berdasarkan profil cluster
        """
        names = {}
        
        for cluster_id in profiles.index:
            profile = profiles.loc[cluster_id]
            
            # Determine characteristics
            high_revenue = profile['total_revenue'] > profiles['total_revenue'].median()
            high_frequency = profile['transaction_count'] > profiles['transaction_count'].median()
            high_price = profile['avg_price'] > profiles['avg_price'].median()
            
            if high_revenue and high_frequency:
                names[cluster_id] = 'Star Products'
            elif high_revenue and not high_frequency:
                names[cluster_id] = 'Premium Products'
            elif not high_revenue and high_frequency:
                names[cluster_id] = 'Volume Products'
            else:
                names[cluster_id] = 'Low Performers'
        
        return names
    
    def predict_segment(self, product_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict segment untuk produk baru
        
        Parameters:
        -----------
        product_data : pd.DataFrame
            Data produk
            
        Returns:
        --------
        pd.DataFrame
            Data dengan segment prediction
        """
        if not self.is_fitted:
            raise ValueError("Model belum di-fit")
        
        feature_cols = ['total_revenue', 'avg_revenue', 'transaction_count', 
                       'total_quantity', 'avg_quantity', 'avg_price']
        X = product_data[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        labels = self.model.predict(X_scaled)
        product_data['cluster'] = labels
        
        # Map ke nama segment
        cluster_names = self._assign_cluster_names(self.cluster_profiles)
        product_data['segment'] = product_data['cluster'].map(cluster_names)
        
        return product_data


class AnomalyDetector:
    """
    Kelas untuk anomaly detection pada data penjualan
    """
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05):
        """
        Initialize detector
        
        Parameters:
        -----------
        method : str
            Metode detection ('isolation_forest', 'zscore')
        contamination : float
            Proporsi outlier yang diharapkan
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.threshold = None
        self.is_fitted = False
        
        logger.info(f"AnomalyDetector initialized with method: {method}")
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Fit anomaly detection model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame input
        feature_cols : list
            Kolom yang digunakan untuk detection
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan anomaly labels
        """
        df = df.copy()
        
        if feature_cols is None:
            feature_cols = ['revenue', 'quantity']
        
        X = df[feature_cols].fillna(0)
        
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X)
            df['anomaly'] = self.model.predict(X)
            df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 0=normal, 1=anomaly
            df['anomaly_score'] = self.model.decision_function(X) * -1
            
        elif self.method == 'zscore':
            z_scores = np.abs((X - X.mean()) / X.std())
            df['anomaly'] = (z_scores > 3).any(axis=1).astype(int)
            df['anomaly_score'] = z_scores.max(axis=1)
            self.threshold = 3
        
        self.is_fitted = True
        
        n_anomalies = df['anomaly'].sum()
        logger.info(f"Anomaly detection completed. Found {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return df
    
    def detect_anomalies(self, new_data: pd.DataFrame,
                         feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies pada data baru menggunakan model yang sudah di-fit.
        BUG FIX: versi lama salah panggil self.fit() yang re-fit model dari scratch.
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            Data baru
        feature_cols : list, optional
            Kolom yang digunakan (harus sama dengan saat fit)
            
        Returns:
        --------
        pd.DataFrame
            Data dengan anomaly labels
        """
        if not self.is_fitted:
            raise ValueError("Model belum di-fit. Panggil fit() terlebih dahulu.")
        
        new_data = new_data.copy()
        
        if feature_cols is None:
            feature_cols = ['revenue', 'quantity']
        
        X = new_data[feature_cols].fillna(0)
        
        if self.method == 'isolation_forest':
            new_data['anomaly'] = self.model.predict(X)
            new_data['anomaly'] = new_data['anomaly'].map({1: 0, -1: 1})
            new_data['anomaly_score'] = self.model.decision_function(X) * -1
        elif self.method == 'zscore':
            # Gunakan threshold yang sudah ditentukan saat fit
            z_scores = np.abs((X - X.mean()) / X.std())
            new_data['anomaly'] = (z_scores > self.threshold).any(axis=1).astype(int)
            new_data['anomaly_score'] = z_scores.max(axis=1)
        
        return new_data


class ModelComparator:
    """
    Kelas untuk membandingkan performa berbagai model.
    Otomatis menguji semua model yang tersedia dan merekomendasikan yang terbaik.
    """
    
    def __init__(self):
        self.results = {}
        self.best_model_type = None
        self.best_model = None
        logger.info("ModelComparator initialized")
    
    def compare_models(self, df: pd.DataFrame, target_col: str = 'revenue') -> pd.DataFrame:
        """
        Bandingkan berbagai model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame training
        target_col : str
            Kolom target
            
        Returns:
        --------
        pd.DataFrame
            Comparison results diurutkan dari RMSE terkecil
        """
        models = ['linear', 'ridge', 'gradient_boosting', 'random_forest', 'extra_trees']
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models.append('lightgbm')
        if PROPHET_AVAILABLE:
            models.append('prophet')
        # Ensemble dan stacking butuh waktu lebih lama, tambahkan jika dataset cukup besar
        if len(df) >= 200:
            models.extend(['ensemble', 'stacking'])
        
        logger.info(f"Comparing {len(models)} models: {models}")
        
        trained_forecasters = {}
        for model_type in models:
            try:
                logger.info(f"Training {model_type}...")
                forecaster = SalesForecaster(model_type=model_type)
                metrics = forecaster.fit(df, target_col)
                self.results[model_type] = metrics
                trained_forecasters[model_type] = forecaster
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                self.results[model_type] = {'error': str(e)}
        
        # Convert ke DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        # Urutkan berdasarkan test_rmse (terkecil = terbaik)
        if 'test_rmse' in comparison_df.columns:
            valid_df = comparison_df[comparison_df['test_rmse'].notna()].copy()
            valid_df['test_rmse'] = pd.to_numeric(valid_df['test_rmse'], errors='coerce')
            comparison_df = comparison_df.reindex(
                valid_df.sort_values('test_rmse').index.tolist() +
                [i for i in comparison_df.index if i not in valid_df.index]
            )
            if not valid_df.empty:
                self.best_model_type = valid_df['test_rmse'].idxmin()
                self.best_model = trained_forecasters.get(self.best_model_type)
                logger.info(f"🏆 Best model: {self.best_model_type} "
                            f"(RMSE: {valid_df['test_rmse'].min():,.2f})")
        
        return comparison_df
    
    def get_best_model(self) -> Optional['SalesForecaster']:
        """
        Kembalikan model terbaik setelah compare_models dipanggil.
        
        Returns:
        --------
        SalesForecaster atau None
        """
        if self.best_model is None:
            logger.warning("Belum ada model yang dibandingkan. Panggil compare_models() dulu.")
        return self.best_model


def train_and_save_models(df: pd.DataFrame, output_dir: str = 'models'):
    """
    Fungsi helper untuk train dan save semua model
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame training
    output_dir : str
        Directory untuk menyimpan model
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train forecaster
    logger.info("Training sales forecaster...")
    forecaster = SalesForecaster(model_type='random_forest')
    forecaster.fit(df, do_tuning=True)
    forecaster.save_model(f'{output_dir}/forecaster.pkl')
    
    # Train segmenter
    logger.info("Training product segmenter...")
    segmenter = ProductSegmenter(n_clusters=4)
    segmenter.fit(df)
    
    # Save segmenter
    with open(f'{output_dir}/segmenter.pkl', 'wb') as f:
        pickle.dump(segmenter, f)
    
    # Train anomaly detector
    logger.info("Training anomaly detector...")
    detector = AnomalyDetector(method='isolation_forest')
    detector.fit(df)
    
    # Save detector
    with open(f'{output_dir}/detector.pkl', 'wb') as f:
        pickle.dump(detector, f)
    
    logger.info(f"All models saved to {output_dir}")


if __name__ == "__main__":
    # Test dengan sample data
    print("Testing ML Models...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 365),
        'quantity': np.random.randint(1, 100, 365),
        'price': np.random.uniform(10000, 100000, 365),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 365)
    })
    sample_data['revenue'] = sample_data['quantity'] * sample_data['price']
    
    # Add time features
    sample_data['year'] = sample_data['date'].dt.year
    sample_data['month'] = sample_data['date'].dt.month
    sample_data['day_of_week'] = sample_data['date'].dt.dayofweek
    
    # Test forecaster
    print("\n1. Testing SalesForecaster...")
    forecaster = SalesForecaster(model_type='random_forest')
    metrics = forecaster.fit(sample_data)
    print(f"Metrics: {metrics}")
    
    # Test feature importance
    importance = forecaster.get_feature_importance()
    print(f"\nTop 5 important features:\n{importance.head()}")
    
    # Test segmenter
    print("\n2. Testing ProductSegmenter...")
    segmenter = ProductSegmenter(n_clusters=4)
    segments = segmenter.fit(sample_data)
    print(f"Segments:\n{segments[['product', 'segment']].drop_duplicates()}")
    
    # Test anomaly detector
    print("\n3. Testing AnomalyDetector...")
    detector = AnomalyDetector(method='isolation_forest')
    results = detector.fit(sample_data)
    print(f"Anomalies detected: {results['anomaly'].sum()}")
    
    # Test model comparison
    print("\n4. Testing ModelComparator...")
    comparator = ModelComparator()
    comparison = comparator.compare_models(sample_data)
    print(f"\nModel Comparison:\n{comparison}")
