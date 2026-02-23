# 🚀 Installation Guide

Panduan lengkap instalasi Sales ML Analytics System.

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, atau Linux (Ubuntu 18.04+)
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Python**: 3.9, 3.10, atau 3.11
- **Internet**: Required untuk install dependencies

### Recommended Requirements
- **RAM**: 8 GB+
- **Storage**: 5 GB free space
- **CPU**: Multi-core processor
- **Python**: 3.11

---

## 🛠️ Installation Methods

### Method 1: Local Installation (Recommended untuk Development)

#### Step 1: Install Python

**Windows:**
1. Download Python dari [python.org](https://python.org)
2. Pilih versi 3.11.x
3. **IMPORTANT**: Centang "Add Python to PATH"
4. Install

**Verify installation:**
```bash
python --version
# Should show: Python 3.11.x
```

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Verify
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version
```

#### Step 2: Install Git (Optional tapi Recommended)

**Windows:**
Download dan install dari [git-scm.com](https://git-scm.com)

**macOS:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt install git
```

#### Step 3: Clone/Download Project

```bash
# Clone repository (jika ada)
git clone <repository-url>
cd sales_ml_system

# Atau extract ZIP file
cd sales_ml_system
```

#### Step 4: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify:**
```bash
which python
# Should show path to venv
```

#### Step 5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Troubleshooting dependencies:**

Jika ada error, install satu per satu:
```bash
# Core
pip install numpy pandas scikit-learn

# Web
pip install flask flask-cors streamlit

# Viz
pip install matplotlib seaborn plotly

# ML extras
pip install xgboost prophet

# Export
pip install openpyxl fpdf2
```

#### Step 6: Verify Installation

```bash
# Test modules
python preprocessing.py
python ml_model.py
python utils.py

# Run test suite
python test_system.py
```

#### Step 7: Run Application

```bash
# Terminal 1: Dashboard
streamlit run app.py

# Terminal 2: API
python api.py
```

---

### Method 2: Docker Installation (Recommended untuk Production)

#### Prerequisites
- Install Docker: [docker.com](https://docker.com)
- Install Docker Compose: [docs.docker.com/compose](https://docs.docker.com/compose)

#### Step 1: Verify Docker

```bash
docker --version
docker-compose --version
```

#### Step 2: Build Images

```bash
# Build all images
docker-compose build

# Atau build individual
docker build --target production -t sales-ml-dashboard .
docker build --target api-only -t sales-ml-api .
```

#### Step 3: Run Containers

```bash
# Run both services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Step 4: Access Applications

- Dashboard: http://localhost:8501
- API: http://localhost:5000
- API Docs: http://localhost:5000/api/health

#### Step 5: Stop Services

```bash
docker-compose down

# Stop dan hapus volumes
docker-compose down -v
```

---

### Method 3: Conda Installation (Alternative)

#### Step 1: Install Anaconda/Miniconda

Download dari [anaconda.com](https://anaconda.com) atau [conda.io](https://conda.io)

#### Step 2: Create Environment

```bash
# Create environment
conda create -n salesml python=3.11

# Activate
conda activate salesml
```

#### Step 3: Install Dependencies

```bash
# Install dari conda
conda install numpy pandas scikit-learn matplotlib seaborn

# Install dari pip
pip install -r requirements.txt
```

#### Step 4: Run Application

```bash
streamlit run app.py
```

---

## 🔧 Platform-Specific Instructions

### Windows

#### Common Issues

**1. "python" not recognized**
```bash
# Add to PATH atau gunakan:
py -3.11 -m venv venv
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run app.py
```

**2. Microsoft Visual C++ 14.0 required**
Download dari: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**3. XGBoost installation fails**
```bash
# Download pre-built wheel
pip install https://github.com/cgohlke/xgboost-wheels/releases/download/v2.0.0/xgboost-2.0.0-cp311-cp311-win_amd64.whl
```

### macOS

#### Common Issues

**1. "pip: command not found"**
```bash
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
```

**2. XGBoost installation**
```bash
# Install libomp terlebih dahulu
brew install libomp
pip install xgboost
```

**3. Prophet installation**
```bash
# Install dependencies
brew install pcre
pip install pystan
pip install prophet
```

### Linux (Ubuntu/Debian)

#### Common Issues

**1. Missing system dependencies**
```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip
sudo apt install -y libgomp1
```

**2. Permission denied**
```bash
# Jangan gunakan sudo dengan pip
python3 -m pip install --user -r requirements.txt

# Atau gunakan virtual environment
```

---

## 🧪 Testing Installation

### Quick Test

```bash
# Test Python imports
python -c "import pandas, numpy, sklearn, flask, streamlit; print('✅ All imports successful')"

# Test modules
python test_system.py
```

### Manual Test

```bash
# 1. Start API
python api.py

# 2. Test API (di terminal lain)
curl http://localhost:5000/api/health

# 3. Start Dashboard (di terminal lain)
streamlit run app.py

# 4. Open browser: http://localhost:8501
```

---

## 🐛 Troubleshooting

### Import Errors

```bash
# Module not found
pip install <module-name>

# Circular import
python -c "import sys; print(sys.path)"

# Wrong Python version
python --version
which python
```

### Memory Errors

```bash
# Untuk large datasets
export PYTHONUNBUFFERED=1

# Atau di Windows
set PYTHONUNBUFFERED=1
```

### Port Already in Use

```bash
# Find process
lsof -ti:8501  # Mac/Linux
netstat -ano | findstr :8501  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows
```

### Prophet Installation Issues

```bash
# Alternative installation
conda install -c conda-forge prophet

# Atau
pip install cmdstanpy
pip install prophet
```

---

## 📦 Dependency Versions

### Tested Versions

```
Python: 3.11.4
NumPy: 1.24.3
Pandas: 2.0.3
Scikit-learn: 1.3.0
Streamlit: 1.28.0
Flask: 2.3.3
Plotly: 5.17.0
XGBoost: 2.0.0
Prophet: 1.1.4
```

### Check Installed Versions

```bash
pip list

# Atau
python -c "import pandas; print(pandas.__version__)"
```

---

## 🔄 Updating

### Update Dependencies

```bash
# Update all
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade streamlit
```

### Update Project

```bash
# Git pull (jika menggunakan git)
git pull origin main

# Atau download dan extract ulang
```

---

## 🗑️ Uninstallation

### Remove Virtual Environment

```bash
# Deactivate
deactivate  # Mac/Linux
venv\Scripts\deactivate  # Windows

# Remove folder
rm -rf venv  # Mac/Linux
rmdir /s venv  # Windows
```

### Remove Docker Containers

```bash
docker-compose down -v
docker rmi sales-ml-dashboard sales-ml-api
```

### Remove Project

```bash
rm -rf sales_ml_system
```

---

## 💡 Tips & Tricks

### Speed Up Installation

```bash
# Gunakan mirror lokal (China)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Gunakan cache
pip install --cache-dir ~/.pip/cache -r requirements.txt
```

### Development Mode

```bash
# Install editable mode
pip install -e .

# Auto-reload
streamlit run app.py --server.runOnSave true
```

### Jupyter Notebook

```bash
# Install kernel
pip install ipykernel
python -m ipykernel install --user --name=salesml

# Run Jupyter
jupyter notebook
```

---

## 📞 Getting Help

### Documentation
- README.md - Overview dan usage
- PROJECT_SUMMARY.md - Technical details
- Code comments - Inline documentation

### Community
- GitHub Issues
- Stack Overflow
- Discord/Slack community

---

## ✅ Post-Installation Checklist

- [ ] Python installed correctly
- [ ] Virtual environment created
- [ ] Dependencies installed without errors
- [ ] Test suite passed
- [ ] Dashboard accessible di localhost:8501
- [ ] API accessible di localhost:5000
- [ ] Sample data loaded successfully
- [ ] Can upload and process files
- [ ] Can train models
- [ ] Can generate reports

---

**🎉 Installation Complete!**

Selanjutnya: Baca [README.md](README.md) untuk panduan penggunaan.
