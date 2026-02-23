# 🎯 Sales ML Analytics System

Sistem Machine Learning lengkap untuk analisis data penjualan dengan forecasting, segmentasi produk, anomaly detection, dan dashboard interaktif.

## 📋 Fitur Utama

### 1. **Data Preprocessing**
- ✅ Multi-format support (CSV, Excel, JSON)
- ✅ Auto data cleaning (missing values, duplicates)
- ✅ Date parsing otomatis
- ✅ Column standardization
- ✅ Feature engineering (time-based, aggregations)

### 2. **Descriptive Analytics**
- 📊 KPI Dashboard (Revenue, Transactions, AOV)
- 📈 Growth Analysis (MoM, YoY)
- 🏆 Top/Bottom Products
- 📦 Category Analysis
- 📅 Weekly Patterns

### 3. **Predictive Analytics**
- 🔮 Multi-model Forecasting:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost (optional)
  - Prophet (time series)
- 🎯 Hyperparameter Tuning (GridSearchCV)
- 📊 Model Comparison
- 💾 Model Persistence (.pkl)

### 4. **Segmentasi Produk**
- 🎯 K-Means Clustering
- 🏷️ Auto segment naming (Star, Premium, Volume, Low Performers)
- 📊 Cluster visualization

### 5. **Anomaly Detection**
- 🚨 Isolation Forest
- 📉 Z-Score Method
- 🔍 Anomaly scoring

### 6. **Export & Reporting**
- 📄 PDF Reports dengan grafik
- 📊 Excel multi-sheet export
- 📁 CSV batch export
- 💡 Auto-generated insights

### 7. **Interfaces**
- 🖥️ Streamlit Dashboard (UI)
- 🔌 REST API (Flask)
- 🐳 Docker Support

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip atau conda

### 1. Clone & Setup

```bash
# Clone repository (atau extract folder)
cd sales_ml_system

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Dashboard (Streamlit)

```bash
streamlit run app.py
```

Dashboard akan tersedia di: **http://localhost:8501**

### 3. Jalankan API Server (Flask)

```bash
# Development mode
python api.py

# Production mode (dengan gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

API akan tersedia di: **http://localhost:5000**

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build untuk Streamlit
docker build --target production -t sales-ml-dashboard .

# Build untuk API only
docker build --target api-only -t sales-ml-api .

# Build untuk development
docker build --target development -t sales-ml-dev .
```

### Run Container

```bash
# Run Streamlit Dashboard
docker run -p 8501:8501 sales-ml-dashboard

# Run API Server
docker run -p 5000:5000 sales-ml-api

# Run Both Services
docker run -p 8501:8501 -p 5000:5000 sales-ml-dashboard both

# Run dengan volume untuk persistence
docker run -p 8501:8501 -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  sales-ml-dashboard both
```

### Docker Compose (Recommended)

```yaml
version: '3.8'

services:
  dashboard:
    build:
      context: .
      target: production
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    command: streamlit

  api:
    build:
      context: .
      target: api-only
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    command: api
```

Run dengan:
```bash
docker-compose up -d
```

---

## 📖 Usage Guide

### Streamlit Dashboard

1. **Upload Data**
   - Klik "Browse files" di sidebar
   - Pilih file CSV/Excel/JSON
   - Klik "Process Data"

2. **Overview Tab**
   - Lihat KPI cards
   - Analisis trend revenue
   - Top products chart
   - Auto-generated insights

3. **Forecasting Tab**
   - Pilih model (Random Forest/Linear/XGBoost/Prophet)
   - Enable hyperparameter tuning (opsional)
   - Klik "Train Model"
   - Lihat forecast chart dan metrics

4. **Segmentation Tab**
   - Pilih jumlah cluster (2-8)
   - Klik "Run Clustering"
   - Lihat segment distribution dan details

5. **Anomaly Detection Tab**
   - Pilih method (Isolation Forest/Z-Score)
   - Set contamination rate
   - Klik "Detect Anomalies"
   - Review anomaly list

6. **Reports Tab**
   - Generate PDF Report
   - Export ke Excel
   - Download CSV files

### REST API Endpoints

#### Health Check
```bash
GET /api/health
```

#### Upload & Preprocess
```bash
POST /api/upload
Content-Type: multipart/form-data

file: <your_file.csv>
```

#### Get Data Summary
```bash
GET /api/data/summary
```

#### Train Forecasting Model
```bash
POST /api/forecast/train
Content-Type: application/json

{
  "model_type": "random_forest",
  "do_tuning": true,
  "test_size": 0.2
}
```

#### Predict Forecast
```bash
POST /api/forecast/predict
Content-Type: application/json

{
  "periods": 30
}
```

#### Batch Prediction
```bash
POST /api/forecast/predict-batch
Content-Type: application/json

{
  "data": [
    {"quantity": 5, "price": 100000, "month": 6, ...},
    ...
  ]
}
```

#### Train Segmentation
```bash
POST /api/segmentation/train
Content-Type: application/json

{
  "n_clusters": 4
}
```

#### Detect Anomalies
```bash
POST /api/anomaly/detect
Content-Type: application/json

{
  "method": "isolation_forest",
  "contamination": 0.05
}
```

#### List Models
```bash
GET /api/models/list
```

---

## 📁 Project Structure

```
sales_ml_system/
│
├── app.py                  # Streamlit Dashboard
├── api.py                  # Flask REST API
├── preprocessing.py        # Data preprocessing module
├── ml_model.py            # ML models (forecasting, clustering, anomaly)
├── utils.py               # Utilities (analyzer, reports, visualization)
│
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-entrypoint.sh  # Docker entrypoint script
│
├── data/                 # Data folder
│   └── sample_sales.csv  # Sample dataset
│
├── models/               # Saved models
├── reports/              # Generated reports
├── uploads/              # Uploaded files
└── static/               # Static assets
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Flask API
export FLASK_ENV=production
export API_PORT=5000

# Streamlit
export PORT=8501

# Model Paths
export MODEL_PATH=./models
export DATA_PATH=./data
```

### Model Parameters

Edit di `ml_model.py`:

```python
# Forecasting
SalesForecaster(model_type='random_forest')  # atau 'linear', 'xgboost', 'prophet'

# Segmentation
ProductSegmenter(n_clusters=4)

# Anomaly Detection
AnomalyDetector(method='isolation_forest', contamination=0.05)
```

---

## 📊 Sample Dataset

Dataset sample tersedia di `data/sample_sales.csv` dengan format:

| Column | Description |
|--------|-------------|
| date | Tanggal transaksi |
| product | Nama produk |
| category | Kategori produk |
| region | Wilayah/region |
| quantity | Jumlah unit |
| price | Harga per unit |
| revenue | Total revenue |

### Generate Custom Sample Data

```python
from utils import create_sample_data

# Generate 1000 records
df = create_sample_data(n_records=1000, output_path='data/my_data.csv')
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# Test preprocessing
python preprocessing.py

# Test ML models
python ml_model.py

# Test utilities
python utils.py
```

### API Testing dengan cURL

```bash
# Health check
curl http://localhost:5000/api/health

# Upload file
curl -X POST -F "file=@data/sample_sales.csv" http://localhost:5000/api/upload

# Train model
curl -X POST -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "do_tuning": false}' \
  http://localhost:5000/api/forecast/train

# Get forecast
curl -X POST -H "Content-Type: application/json" \
  -d '{"periods": 30}' \
  http://localhost:5000/api/forecast/predict
```

---

## 🎯 Use Cases

### 1. Sales Performance Monitoring
- Upload daily sales data
- Monitor KPIs real-time
- Track growth trends

### 2. Demand Forecasting
- Train model dengan historical data
- Predict future sales
- Optimize inventory

### 3. Product Portfolio Analysis
- Segment products by performance
- Identify star products
- Find underperformers

### 4. Anomaly Detection
- Detect unusual sales patterns
- Identify potential fraud
- Monitor data quality

### 5. Automated Reporting
- Schedule daily/weekly reports
- Export ke PDF/Excel
- Share dengan stakeholders

---

## 🔍 Troubleshooting

### Common Issues

**1. Module Not Found**
```bash
pip install -r requirements.txt
```

**2. Port Already in Use**
```bash
# Find and kill process
lsof -ti:8501 | xargs kill -9
lsof -ti:5000 | xargs kill -9
```

**3. Memory Error dengan Large Dataset**
```python
# Gunakan chunking
df = pd.read_csv('large_file.csv', chunksize=10000)
```

**4. Prophet Installation Error**
```bash
# Install dependencies terlebih dahulu
pip install pystan
pip install prophet
```

**5. XGBoost Installation Error (Mac)**
```bash
brew install libomp
pip install xgboost
```

---

## 📈 Performance Tips

1. **Untuk Large Datasets (>100k rows)**
   - Gunakan sampling untuk training
   - Enable parallel processing
   - Use XGBoost untuk speed

2. **Model Optimization**
   - Enable hyperparameter tuning untuk accuracy
   - Use Prophet untuk time series dengan seasonality
   - Random Forest untuk general purpose

3. **API Optimization**
   - Use gunicorn dengan multiple workers
   - Enable caching untuk frequent queries
   - Batch predictions untuk efficiency

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

MIT License - feel free to use untuk personal dan commercial projects.

---

## 👨‍💻 Author

**Sales ML Analytics System**
Built with ❤️ menggunakan Python, scikit-learn, dan Streamlit.

---

## 📞 Support

Untuk questions dan support:
- 📧 Email: support@salesml.com
- 💬 Discord: [Join our community]
- 📚 Documentation: [Read the Docs]

---

## 🗺️ Roadmap

- [ ] Deep Learning models (LSTM, Transformer)
- [ ] Real-time streaming analytics
- [ ] Advanced visualization (3D charts, network graphs)
- [ ] Multi-language support
- [ ] Cloud deployment templates (AWS, GCP, Azure)
- [ ] Mobile app companion

---

**⭐ Star this repository jika bermanfaat!**
