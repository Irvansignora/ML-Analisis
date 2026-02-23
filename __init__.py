"""
Sales ML Analytics System
=========================

A comprehensive machine learning system for sales data analysis,
forecasting, segmentation, and anomaly detection.

Modules:
--------
- preprocessing: Data cleaning and feature engineering
- ml_model: Machine learning models
- utils: Utilities for analysis and reporting
- app: Streamlit dashboard
- api: Flask REST API

Author: Sales ML Team
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Sales ML Team'

from preprocessing import DataPreprocessor
from ml_model import SalesForecaster, ProductSegmenter, AnomalyDetector, ModelComparator
from utils import SalesAnalyzer, ReportGenerator, Visualizer

__all__ = [
    'DataPreprocessor',
    'SalesForecaster',
    'ProductSegmenter',
    'AnomalyDetector',
    'ModelComparator',
    'SalesAnalyzer',
    'ReportGenerator',
    'Visualizer'
]
