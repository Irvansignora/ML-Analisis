"""
REST API for Sales ML System
============================
API endpoints untuk prediksi real-time dan integrasi
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import pickle
import json
from pathlib import Path

# Import custom modules
from preprocessing import DataPreprocessor
from ml_model import SalesForecaster, ProductSegmenter, AnomalyDetector
from utils import SalesAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS untuk semua routes

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

# Global variables untuk menyimpan model yang sudah diload
loaded_models = {
    'forecaster': None,
    'segmenter': None,
    'detector': None
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_if_exists(model_type='forecaster'):
    """Load model dari file jika ada"""
    model_path = f"{MODEL_FOLDER}/{model_type}.pkl"
    
    if os.path.exists(model_path) and loaded_models[model_type] is None:
        try:
            with open(model_path, 'rb') as f:
                loaded_models[model_type] = pickle.load(f)
            logger.info(f"Model {model_type} loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {str(e)}")
    
    return loaded_models[model_type]


# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


# ==================== DATA UPLOAD & PREPROCESSING ====================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload dan preprocess file data penjualan
    
    Request:
        - file: File CSV/Excel/JSON
        
    Response:
        - summary: Summary dari data
        - preview: Preview data (first 10 rows)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Determine file type
            file_type = filename.rsplit('.', 1)[1].lower()
            if file_type == 'xls':
                file_type = 'excel'
            
            # Preprocess
            preprocessor = DataPreprocessor()
            df = preprocessor.load_data(filepath, file_type)
            df_processed = preprocessor.preprocess(df)
            
            # Get summary
            summary = preprocessor.get_data_summary(df_processed)
            
            # Convert summary untuk JSON serialization
            summary_json = {
                'total_rows': summary['total_rows'],
                'total_columns': summary['total_columns'],
                'columns': summary['columns'],
                'dtypes': {k: str(v) for k, v in summary['dtypes'].items()},
                'missing_values': summary['missing_values'],
                'memory_usage_mb': round(summary['memory_usage'], 2),
                'date_range': summary['date_range']
            }
            
            # Save processed data untuk digunakan nanti
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
            df_processed.to_csv(processed_path, index=False)
            
            # Preview data
            preview = df_processed.head(10).to_dict(orient='records')
            
            # Convert datetime untuk JSON
            for row in preview:
                for key, value in row.items():
                    if isinstance(value, pd.Timestamp):
                        row[key] = value.isoformat()
                    elif isinstance(value, np.integer):
                        row[key] = int(value)
                    elif isinstance(value, np.floating):
                        row[key] = float(value)
            
            return jsonify({
                'success': True,
                'message': 'File uploaded and processed successfully',
                'summary': summary_json,
                'preview': preview,
                'processed_file': 'processed_data.csv'
            })
        
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    """Get summary dari data yang sudah diupload"""
    try:
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        
        if not os.path.exists(processed_path):
            return jsonify({'error': 'No data uploaded yet'}), 404
        
        df = pd.read_csv(processed_path, parse_dates=['date'])
        analyzer = SalesAnalyzer(df)
        
        # Calculate KPIs
        kpis = analyzer.calculate_kpis()
        
        # Calculate growth
        monthly = analyzer.calculate_growth_metrics()
        
        # Top products
        top_products = analyzer.get_top_products(n=5)
        
        # Category analysis
        categories = analyzer.get_category_analysis()
        
        # Generate insights
        insights = analyzer.generate_insights()
        
        return jsonify({
            'success': True,
            'kpis': kpis,
            'monthly_growth': monthly.to_dict(orient='records') if not monthly.empty else [],
            'top_products': top_products.to_dict(orient='records') if not top_products.empty else [],
            'categories': categories.to_dict(orient='records') if not categories.empty else [],
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Error in get_data_summary: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== FORECASTING ====================

@app.route('/api/forecast/train', methods=['POST'])
def train_forecaster():
    """
    Train forecasting model
    
    Request Body:
        - model_type: 'linear', 'random_forest', 'xgboost', 'prophet'
        - do_tuning: boolean (optional)
        - test_size: float (optional, default 0.2)
        
    Response:
        - metrics: Model performance metrics
        - model_path: Path ke saved model
    """
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'random_forest')
        do_tuning = data.get('do_tuning', False)
        test_size = data.get('test_size', 0.2)
        
        # Load processed data
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        if not os.path.exists(processed_path):
            return jsonify({'error': 'No data uploaded yet'}), 404
        
        df = pd.read_csv(processed_path, parse_dates=['date'])
        
        # Train model
        forecaster = SalesForecaster(model_type=model_type)
        metrics = forecaster.fit(df, test_size=test_size, do_tuning=do_tuning)
        
        # Save model
        model_path = f"{MODEL_FOLDER}/forecaster.pkl"
        forecaster.save_model(model_path)
        
        # Update loaded models
        loaded_models['forecaster'] = forecaster
        
        return jsonify({
            'success': True,
            'message': f'Model {model_type} trained successfully',
            'metrics': metrics,
            'model_path': model_path
        })
        
    except Exception as e:
        logger.error(f"Error in train_forecaster: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/predict', methods=['POST'])
def predict_forecast():
    """
    Predict forecast untuk periode mendatang
    
    Request Body:
        - periods: int (jumlah hari untuk forecast, default 30)
        
    Response:
        - forecast: List of forecasted values
    """
    try:
        data = request.get_json()
        periods = data.get('periods', 30)
        
        # Load model
        forecaster = load_model_if_exists('forecaster')
        if forecaster is None:
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
        
        # Load data
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        df = pd.read_csv(processed_path, parse_dates=['date'])
        
        # Generate forecast
        forecast_df = forecaster.forecast_future(df, periods=periods)
        
        # Convert untuk JSON
        forecast_data = forecast_df.to_dict(orient='records')
        for row in forecast_data:
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    row[key] = value.isoformat()
                elif isinstance(value, np.integer):
                    row[key] = int(value)
                elif isinstance(value, np.floating):
                    row[key] = float(value)
        
        return jsonify({
            'success': True,
            'forecast': forecast_data,
            'periods': periods
        })
        
    except Exception as e:
        logger.error(f"Error in predict_forecast: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction untuk data yang dikirim
    
    Request Body:
        - data: List of records dengan features
        
    Response:
        - predictions: List of predicted values
    """
    try:
        data = request.get_json()
        input_data = data.get('data', [])
        
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Load model
        forecaster = load_model_if_exists('forecaster')
        if forecaster is None:
            return jsonify({'error': 'No trained model found'}), 404
        
        # Convert ke DataFrame
        df = pd.DataFrame(input_data)
        
        # Predict
        predictions = forecaster.predict(df)
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error in predict_batch: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance dari trained model"""
    try:
        forecaster = load_model_if_exists('forecaster')
        if forecaster is None:
            return jsonify({'error': 'No trained model found'}), 404
        
        importance_df = forecaster.get_feature_importance()
        
        return jsonify({
            'success': True,
            'feature_importance': importance_df.to_dict(orient='records')
        })
        
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== SEGMENTATION ====================

@app.route('/api/segmentation/train', methods=['POST'])
def train_segmentation():
    """
    Train product segmentation model
    
    Request Body:
        - n_clusters: int (jumlah cluster, default 4)
        
    Response:
        - segments: Product segments
    """
    try:
        data = request.get_json()
        n_clusters = data.get('n_clusters', 4)
        
        # Load data
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        if not os.path.exists(processed_path):
            return jsonify({'error': 'No data uploaded yet'}), 404
        
        df = pd.read_csv(processed_path, parse_dates=['date'])
        
        # Train segmenter
        segmenter = ProductSegmenter(n_clusters=n_clusters)
        segments_df = segmenter.fit(df)
        
        # Save model
        model_path = f"{MODEL_FOLDER}/segmenter.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(segmenter, f)
        
        loaded_models['segmenter'] = segmenter
        
        return jsonify({
            'success': True,
            'message': f'Segmentation with {n_clusters} clusters completed',
            'segments': segments_df.to_dict(orient='records'),
            'model_path': model_path
        })
        
    except Exception as e:
        logger.error(f"Error in train_segmentation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/segmentation/predict', methods=['POST'])
def predict_segment():
    """
    Predict segment untuk produk baru
    
    Request Body:
        - data: List of product records
        
    Response:
        - segments: Predicted segments
    """
    try:
        data = request.get_json()
        input_data = data.get('data', [])
        
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Load model
        segmenter = load_model_if_exists('segmenter')
        if segmenter is None:
            return jsonify({'error': 'No trained model found'}), 404
        
        # Convert ke DataFrame
        df = pd.DataFrame(input_data)
        
        # Predict
        result_df = segmenter.predict_segment(df)
        
        return jsonify({
            'success': True,
            'segments': result_df.to_dict(orient='records')
        })
        
    except Exception as e:
        logger.error(f"Error in predict_segment: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== ANOMALY DETECTION ====================

@app.route('/api/anomaly/detect', methods=['POST'])
def detect_anomalies():
    """
    Detect anomalies dalam data
    
    Request Body:
        - method: 'isolation_forest' atau 'zscore'
        - contamination: float (0.01 - 0.2)
        
    Response:
        - anomalies: List of detected anomalies
    """
    try:
        data = request.get_json()
        method = data.get('method', 'isolation_forest')
        contamination = data.get('contamination', 0.05)
        
        # Load data
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        if not os.path.exists(processed_path):
            return jsonify({'error': 'No data uploaded yet'}), 404
        
        df = pd.read_csv(processed_path, parse_dates=['date'])
        
        # Detect anomalies
        detector = AnomalyDetector(method=method, contamination=contamination)
        result_df = detector.fit(df)
        
        # Save model
        model_path = f"{MODEL_FOLDER}/detector.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(detector, f)
        
        loaded_models['detector'] = detector
        
        # Get anomalies only
        anomalies = result_df[result_df['anomaly'] == 1]
        
        # Convert untuk JSON
        anomalies_data = anomalies.to_dict(orient='records')
        for row in anomalies_data:
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    row[key] = value.isoformat()
                elif isinstance(value, np.integer):
                    row[key] = int(value)
                elif isinstance(value, np.floating):
                    row[key] = float(value)
        
        return jsonify({
            'success': True,
            'total_anomalies': len(anomalies),
            'anomaly_percentage': len(anomalies) / len(result_df) * 100,
            'anomalies': anomalies_data
        })
        
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== MODEL MANAGEMENT ====================

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List semua available models"""
    try:
        models = []
        model_dir = Path(MODEL_FOLDER)
        
        if model_dir.exists():
            for model_file in model_dir.glob('*.pkl'):
                stat = model_file.stat()
                models.append({
                    'name': model_file.stem,
                    'filename': model_file.name,
                    'size_kb': round(stat.st_size / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"Error in list_models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/load/<model_type>', methods=['POST'])
def load_model(model_type):
    """Load model dari file"""
    try:
        model_path = f"{MODEL_FOLDER}/{model_type}.pkl"
        
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 404
        
        with open(model_path, 'rb') as f:
            loaded_models[model_type] = pickle.load(f)
        
        return jsonify({
            'success': True,
            'message': f'Model {model_type} loaded successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("Starting Sales ML API Server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Model folder: {MODEL_FOLDER}")
    
    # Try load existing models
    for model_type in ['forecaster', 'segmenter', 'detector']:
        load_model_if_exists(model_type)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
