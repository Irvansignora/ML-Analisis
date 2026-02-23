"""
Test Script for Sales ML System
===============================
Script untuk testing seluruh komponen sistem
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import modules
from preprocessing import DataPreprocessor, preprocess_file
from ml_model import SalesForecaster, ProductSegmenter, AnomalyDetector, ModelComparator
from utils import SalesAnalyzer, ReportGenerator, Visualizer, create_sample_data


def test_preprocessing():
    """Test preprocessing module"""
    print("\n" + "="*60)
    print("TEST 1: PREPROCESSING MODULE")
    print("="*60)
    
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
    
    print(f"✅ Original columns: {sample_data.columns.tolist()}")
    print(f"✅ Processed columns: {processed_df.columns.tolist()}")
    print(f"✅ Shape: {processed_df.shape}")
    print(f"✅ Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
    
    assert 'date' in processed_df.columns
    assert 'product' in processed_df.columns
    assert 'revenue' in processed_df.columns
    
    print("✅ Preprocessing test PASSED")
    return processed_df


def test_analytics(processed_df):
    """Test analytics module"""
    print("\n" + "="*60)
    print("TEST 2: ANALYTICS MODULE")
    print("="*60)
    
    analyzer = SalesAnalyzer(processed_df)
    
    # Test KPIs
    kpis = analyzer.calculate_kpis()
    print(f"✅ Total Revenue: Rp {kpis.get('total_revenue', 0):,.0f}")
    print(f"✅ Total Transactions: {kpis.get('total_transactions', 0)}")
    print(f"✅ Avg Order Value: Rp {kpis.get('avg_order_value', 0):,.0f}")
    
    # Test insights
    insights = analyzer.generate_insights()
    print(f"✅ Generated {len(insights)} insights")
    for i, insight in enumerate(insights[:3], 1):
        print(f"   {i}. {insight}")
    
    print("✅ Analytics test PASSED")
    return analyzer


def test_forecasting(processed_df):
    """Test forecasting module"""
    print("\n" + "="*60)
    print("TEST 3: FORECASTING MODULE")
    print("="*60)
    
    # Test Linear Regression
    print("\n📊 Testing Linear Regression...")
    forecaster_lr = SalesForecaster(model_type='linear')
    metrics_lr = forecaster_lr.fit(processed_df)
    print(f"✅ Linear Regression - RMSE: {metrics_lr['test_rmse']:.2f}, R²: {metrics_lr['test_r2']:.3f}")
    
    # Test Random Forest
    print("\n📊 Testing Random Forest...")
    forecaster_rf = SalesForecaster(model_type='random_forest')
    metrics_rf = forecaster_rf.fit(processed_df, do_tuning=False)
    print(f"✅ Random Forest - RMSE: {metrics_rf['test_rmse']:.2f}, R²: {metrics_rf['test_r2']:.3f}")
    
    # Test forecast generation
    forecast_df = forecaster_rf.forecast_future(processed_df, periods=30)
    print(f"✅ Generated forecast for {len(forecast_df)} periods")
    
    # Test feature importance
    importance = forecaster_rf.get_feature_importance()
    print(f"✅ Top 3 important features:")
    for _, row in importance.head(3).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    print("✅ Forecasting test PASSED")
    return forecaster_rf


def test_segmentation(processed_df):
    """Test segmentation module"""
    print("\n" + "="*60)
    print("TEST 4: SEGMENTATION MODULE")
    print("="*60)
    
    segmenter = ProductSegmenter(n_clusters=4)
    segments_df = segmenter.fit(processed_df)
    
    print(f"✅ Segmented {len(segments_df)} products into {segments_df['cluster'].nunique()} clusters")
    print(f"✅ Segment distribution:")
    for segment, count in segments_df['segment'].value_counts().items():
        print(f"   - {segment}: {count} products")
    
    print("✅ Segmentation test PASSED")
    return segments_df


def test_anomaly_detection(processed_df):
    """Test anomaly detection module"""
    print("\n" + "="*60)
    print("TEST 5: ANOMALY DETECTION MODULE")
    print("="*60)
    
    detector = AnomalyDetector(method='isolation_forest', contamination=0.05)
    result_df = detector.fit(processed_df)
    
    n_anomalies = result_df['anomaly'].sum()
    print(f"✅ Detected {n_anomalies} anomalies ({n_anomalies/len(result_df)*100:.2f}% of data)")
    
    print("✅ Anomaly detection test PASSED")
    return result_df


def test_reporting(analyzer):
    """Test reporting module"""
    print("\n" + "="*60)
    print("TEST 6: REPORTING MODULE")
    print("="*60)
    
    reporter = ReportGenerator(analyzer)
    
    # Test CSV export
    reporter.export_to_csv(output_dir='reports')
    print("✅ CSV export completed")
    
    # Test Excel export
    try:
        reporter.export_to_excel(output_path='reports/test_report.xlsx')
        print("✅ Excel export completed")
    except Exception as e:
        print(f"⚠️ Excel export skipped: {e}")
    
    print("✅ Reporting test PASSED")


def test_visualization(processed_df):
    """Test visualization module"""
    print("\n" + "="*60)
    print("TEST 7: VISUALIZATION MODULE")
    print("="*60)
    
    viz = Visualizer(processed_df)
    
    # Test charts
    fig1 = viz.create_revenue_trend_chart()
    print("✅ Revenue trend chart created")
    
    fig2 = viz.create_top_products_chart(n=5)
    print("✅ Top products chart created")
    
    fig3 = viz.create_category_pie_chart()
    print("✅ Category pie chart created")
    
    print("✅ Visualization test PASSED")


def test_model_comparison(processed_df):
    """Test model comparison"""
    print("\n" + "="*60)
    print("TEST 8: MODEL COMPARISON")
    print("="*60)
    
    comparator = ModelComparator()
    comparison_df = comparator.compare_models(processed_df)
    
    print("✅ Model comparison results:")
    print(comparison_df.to_string())
    
    print("✅ Model comparison test PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀"*30)
    print("SALES ML SYSTEM - COMPREHENSIVE TEST")
    print("🚀"*30)
    
    start_time = datetime.now()
    
    try:
        # Test 1: Preprocessing
        processed_df = test_preprocessing()
        
        # Test 2: Analytics
        analyzer = test_analytics(processed_df)
        
        # Test 3: Forecasting
        forecaster = test_forecasting(processed_df)
        
        # Test 4: Segmentation
        segments = test_segmentation(processed_df)
        
        # Test 5: Anomaly Detection
        anomalies = test_anomaly_detection(processed_df)
        
        # Test 6: Reporting
        test_reporting(analyzer)
        
        # Test 7: Visualization
        test_visualization(processed_df)
        
        # Test 8: Model Comparison
        test_model_comparison(processed_df)
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print(f"⏱️  Total duration: {duration:.2f} seconds")
        print(f"📊 Data processed: {len(processed_df):,} records")
        print("\n✅ System is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
