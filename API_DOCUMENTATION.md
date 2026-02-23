# 📚 API Documentation

Dokumentasi lengkap REST API untuk Sales ML Analytics System.

**Base URL**: `http://localhost:5000/api`

---

## 🔐 Authentication

Saat ini API tidak memerlukan authentication. Untuk production, tambahkan:
- API Key
- JWT Token
- OAuth 2.0

---

## 📋 Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload & preprocess file |
| GET | `/data/summary` | Get data summary |
| POST | `/forecast/train` | Train forecasting model |
| POST | `/forecast/predict` | Generate forecast |
| POST | `/forecast/predict-batch` | Batch prediction |
| GET | `/forecast/feature-importance` | Get feature importance |
| POST | `/segmentation/train` | Train segmentation |
| POST | `/segmentation/predict` | Predict segment |
| POST | `/anomaly/detect` | Detect anomalies |
| GET | `/models/list` | List saved models |
| POST | `/models/load/{type}` | Load model |

---

## 🔍 Detailed Endpoints

### 1. Health Check

Check API status.

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000000",
  "version": "1.0.0"
}
```

---

### 2. Upload File

Upload dan preprocess data file.

```http
POST /api/upload
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | CSV, Excel, atau JSON file |

**Example (cURL):**
```bash
curl -X POST \
  -F "file=@data/sample_sales.csv" \
  http://localhost:5000/api/upload
```

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/upload"
files = {'file': open('data/sample_sales.csv', 'rb')}

response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "message": "File uploaded and processed successfully",
  "summary": {
    "total_rows": 33852,
    "total_columns": 10,
    "columns": ["date", "product", "category", "region", "quantity", "price", "revenue"],
    "memory_usage_mb": 15.5,
    "date_range": {
      "min": "2022-01-01",
      "max": "2023-12-31",
      "span_days": 729
    }
  },
  "preview": [...],
  "processed_file": "processed_data.csv"
}
```

---

### 3. Get Data Summary

Get summary dari data yang sudah diupload.

```http
GET /api/data/summary
```

**Response:**
```json
{
  "success": true,
  "kpis": {
    "total_revenue": 241987654321,
    "total_transactions": 33852,
    "avg_order_value": 7147231,
    "unique_products": 15,
    "unique_customers": 0,
    "avg_daily_revenue": 331944656
  },
  "monthly_growth": [...],
  "top_products": [...],
  "categories": [...],
  "insights": [
    "Total revenue: Rp 241,987,654,321",
    "Total transactions: 33,852",
    "Average order value: Rp 7,147,231",
    ...
  ]
}
```

---

### 4. Train Forecasting Model

Train model forecasting.

```http
POST /api/forecast/train
Content-Type: application/json
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| model_type | string | No | "random_forest" | "linear", "random_forest", "xgboost", "prophet" |
| do_tuning | boolean | No | false | Enable hyperparameter tuning |
| test_size | float | No | 0.2 | Test set proportion |

**Request:**
```json
{
  "model_type": "random_forest",
  "do_tuning": true,
  "test_size": 0.2
}
```

**Example (cURL):**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "do_tuning": false}' \
  http://localhost:5000/api/forecast/train
```

**Response:**
```json
{
  "success": true,
  "message": "Model random_forest trained successfully",
  "metrics": {
    "train_mae": 1523456.78,
    "train_rmse": 2134567.89,
    "train_r2": 0.9456,
    "test_mae": 1823456.90,
    "test_rmse": 2456789.12,
    "test_r2": 0.9234
  },
  "model_path": "models/forecaster.pkl"
}
```

---

### 5. Predict Forecast

Generate forecast untuk periode mendatang.

```http
POST /api/forecast/predict
Content-Type: application/json
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| periods | integer | No | 30 | Number of days to forecast |

**Request:**
```json
{
  "periods": 30
}
```

**Response:**
```json
{
  "success": true,
  "forecast": [
    {
      "date": "2024-01-01T00:00:00",
      "forecast": 52345678.90
    },
    {
      "date": "2024-01-02T00:00:00",
      "forecast": 53456789.01
    },
    ...
  ],
  "periods": 30
}
```

---

### 6. Batch Prediction

Predict untuk multiple records.

```http
POST /api/forecast/predict-batch
Content-Type: application/json
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | array | Yes | Array of feature objects |

**Request:**
```json
{
  "data": [
    {
      "quantity": 5,
      "price": 100000,
      "year": 2024,
      "month": 1,
      "day_of_week": 1,
      "is_weekend": 0
    },
    {
      "quantity": 3,
      "price": 150000,
      "year": 2024,
      "month": 1,
      "day_of_week": 2,
      "is_weekend": 0
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [5234567.89, 4789012.34]
}
```

---

### 7. Get Feature Importance

Get feature importance dari trained model.

```http
GET /api/forecast/feature-importance
```

**Response:**
```json
{
  "success": true,
  "feature_importance": [
    {"feature": "quantity", "importance": 0.4567},
    {"feature": "price", "importance": 0.3456},
    {"feature": "month", "importance": 0.1234},
    ...
  ]
}
```

---

### 8. Train Segmentation

Train product segmentation model.

```http
POST /api/segmentation/train
Content-Type: application/json
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| n_clusters | integer | No | 4 | Number of clusters |

**Request:**
```json
{
  "n_clusters": 4
}
```

**Response:**
```json
{
  "success": true,
  "message": "Segmentation with 4 clusters completed",
  "segments": [
    {
      "product": "Laptop Gaming",
      "total_revenue": 45678901234,
      "avg_revenue": 12345678,
      "transaction_count": 3698,
      "cluster": 0,
      "segment": "Star Products"
    },
    ...
  ],
  "model_path": "models/segmenter.pkl"
}
```

---

### 9. Predict Segment

Predict segment untuk produk baru.

```http
POST /api/segmentation/predict
Content-Type: application/json
```

**Request:**
```json
{
  "data": [
    {
      "product": "New Product A",
      "total_revenue": 5000000,
      "avg_revenue": 1000000,
      "transaction_count": 5,
      "total_quantity": 50,
      "avg_quantity": 10,
      "avg_price": 100000
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "segments": [
    {
      "product": "New Product A",
      "cluster": 2,
      "segment": "Volume Products"
    }
  ]
}
```

---

### 10. Detect Anomalies

Detect anomalies dalam data.

```http
POST /api/anomaly/detect
Content-Type: application/json
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| method | string | No | "isolation_forest" | "isolation_forest" atau "zscore" |
| contamination | float | No | 0.05 | Expected proportion of outliers |

**Request:**
```json
{
  "method": "isolation_forest",
  "contamination": 0.05
}
```

**Response:**
```json
{
  "success": true,
  "total_anomalies": 1692,
  "anomaly_percentage": 5.0,
  "anomalies": [
    {
      "date": "2022-03-15T00:00:00",
      "product": "Laptop Gaming",
      "revenue": 456789012,
      "quantity": 30,
      "anomaly": 1,
      "anomaly_score": 0.85
    },
    ...
  ]
}
```

---

### 11. List Models

List semua saved models.

```http
GET /api/models/list
```

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "name": "forecaster",
      "filename": "forecaster.pkl",
      "size_kb": 1250.5,
      "modified": "2024-01-15T10:30:00.000000"
    },
    {
      "name": "segmenter",
      "filename": "segmenter.pkl",
      "size_kb": 45.2,
      "modified": "2024-01-15T10:35:00.000000"
    }
  ]
}
```

---

### 12. Load Model

Load model dari file.

```http
POST /api/models/load/{model_type}
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| model_type | string | Yes | "forecaster", "segmenter", "detector" |

**Example:**
```bash
curl -X POST http://localhost:5000/api/models/load/forecaster
```

**Response:**
```json
{
  "success": true,
  "message": "Model forecaster loaded successfully"
}
```

---

## 🔄 Complete Workflow Example

### Scenario: Train Model dan Generate Forecast

```python
import requests
import json

BASE_URL = "http://localhost:5000/api"

# 1. Upload data
print("1. Uploading data...")
with open('data/sample_sales.csv', 'rb') as f:
    response = requests.post(f"{BASE_URL}/upload", files={'file': f})
print(response.json()['message'])

# 2. Train model
print("\n2. Training model...")
train_data = {
    "model_type": "random_forest",
    "do_tuning": True
}
response = requests.post(f"{BASE_URL}/forecast/train", json=train_data)
metrics = response.json()['metrics']
print(f"Model trained! RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.3f}")

# 3. Generate forecast
print("\n3. Generating forecast...")
predict_data = {"periods": 30}
response = requests.post(f"{BASE_URL}/forecast/predict", json=predict_data)
forecast = response.json()['forecast']
print(f"Generated {len(forecast)} days forecast")

# 4. Get feature importance
print("\n4. Feature importance:")
response = requests.get(f"{BASE_URL}/forecast/feature-importance")
importance = response.json()['feature_importance']
for feat in importance[:5]:
    print(f"  {feat['feature']}: {feat['importance']:.3f}")

# 5. Detect anomalies
print("\n5. Detecting anomalies...")
anomaly_data = {"method": "isolation_forest", "contamination": 0.05}
response = requests.post(f"{BASE_URL}/anomaly/detect", json=anomaly_data)
anomalies = response.json()
print(f"Found {anomalies['total_anomalies']} anomalies ({anomalies['anomaly_percentage']:.2f}%)")

print("\n✅ Workflow completed!")
```

---

## ⚠️ Error Handling

### Error Response Format

```json
{
  "error": "Error message description"
}
```

### Common Errors

| Status Code | Error | Solution |
|-------------|-------|----------|
| 400 | Bad Request | Check request parameters |
| 404 | Not Found | Upload data terlebih dahulu |
| 500 | Internal Error | Check server logs |

---

## 📊 Rate Limiting

Saat ini tidak ada rate limiting. Untuk production, tambahkan:

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/forecast/predict', methods=['POST'])
@limiter.limit("100 per hour")
def predict_forecast():
    ...
```

---

## 🔒 Security

### Production Checklist

- [ ] Enable authentication (API Key/JWT)
- [ ] Enable HTTPS/TLS
- [ ] Add rate limiting
- [ ] Validate all inputs
- [ ] Sanitize file uploads
- [ ] Add CORS restrictions
- [ ] Enable request logging
- [ ] Set up monitoring

---

## 📚 SDK Examples

### Python SDK

```python
class SalesMLClient:
    def __init__(self, base_url="http://localhost:5000/api"):
        self.base_url = base_url
    
    def upload(self, file_path):
        with open(file_path, 'rb') as f:
            return requests.post(f"{self.base_url}/upload", files={'file': f})
    
    def train_forecaster(self, model_type="random_forest"):
        return requests.post(
            f"{self.base_url}/forecast/train",
            json={"model_type": model_type}
        )
    
    def predict(self, periods=30):
        return requests.post(
            f"{self.base_url}/forecast/predict",
            json={"periods": periods}
        )

# Usage
client = SalesMLClient()
client.upload("data.csv")
client.train_forecaster("random_forest")
forecast = client.predict(30)
```

### JavaScript/Node.js SDK

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class SalesMLClient {
    constructor(baseUrl = 'http://localhost:5000/api') {
        this.baseUrl = baseUrl;
    }
    
    async upload(filePath) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));
        
        return axios.post(`${this.baseUrl}/upload`, form, {
            headers: form.getHeaders()
        });
    }
    
    async trainForecaster(modelType = 'random_forest') {
        return axios.post(`${this.baseUrl}/forecast/train`, {
            model_type: modelType
        });
    }
    
    async predict(periods = 30) {
        return axios.post(`${this.baseUrl}/forecast/predict`, {
            periods: periods
        });
    }
}

// Usage
const client = new SalesMLClient();
await client.upload('data.csv');
await client.trainForecaster('random_forest');
const forecast = await client.predict(30);
```

---

## 📞 Support

For API support:
- Email: api-support@salesml.com
- Docs: [Full Documentation](README.md)
- Issues: [GitHub Issues]

---

**Version**: 1.0.0  
**Last Updated**: 2024
