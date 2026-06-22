🛒 RetailPulse
ML-Powered Retail Analytics Platform
Python FastAPI Streamlit MLflow XGBoost License

Built during Data Science & Analytics Internship @ Zidio Development

🚀 Live Demo• 📊 Dashboard • 🔌 API Docs • 📈 Results

📌 Problem Statement
Retail businesses generate massive volumes of transactional data but often lack the tools to extract actionable insights from it. Key business questions go unanswered:

Who are our most valuable customers and who is about to churn?
How much stock will we sell next month across our product range?
Which products are at risk of running out before we can reorder?
What customer segments exist and how should we treat each differently?
Without answers to these, businesses overspend on inventory, miss churn early warning signs, and treat all customers the same — leaving revenue on the table.

💡 Our Approach
RetailPulse addresses all four problems end-to-end using a single transactional dataset:

Raw Transactions → Clean Data → Feature Engineering → ML Models → Dashboard + API
Problem	Technique	Model
Customer segmentation	RFM Analysis + Clustering	KMeans
Demand forecasting	Time series modelling	SARIMA
Churn prediction	Binary classification	XGBoost
Inventory optimization	Statistical safety stock formula	Rule-based + ML
Each model feeds into an interactive Streamlit dashboard and is exposed via a FastAPI REST API, with Evidently AI monitoring data drift and model performance in production.

📊 Results
Model	Metric	Score	Target
Churn Prediction	AUC-ROC	1.00	≥ 0.80 ✅
Segmentation	Silhouette Score	0.42	≥ 0.35 ✅
Demand Forecasting	MAPE	35.1%	≤ 15% ⚠️
Inventory	SKUs Covered	4,739	All ✅
API Response	Latency	< 300ms	< 300ms ✅
Note on Churn AUC (1.00): The churn label is derived directly from Recency (inactive ≥ 90 days), making Recency a dominant predictor. In a real-world scenario with actual cancellation data, the AUC would reflect genuine generalization. This is documented intentionally.

Note on Forecasting MAPE (35.1%): The UCI retail dataset has extreme revenue spikes during Q4 (Black Friday, Christmas). SARIMA struggles with irregular spikes. Adding holiday regressors or switching to a hybrid model would improve this.

🏗️ Architecture
┌─────────────────────────────────────────────┐
│           UCI Online Retail II              │
│        1,067,371 transactions               │
│           (2009 – 2011)                     │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────▼────────┐
         │   Data Layer    │
         │  ingest + clean │ → 1,003,740 clean rows
         └────────┬────────┘
                  │
         ┌────────▼──────────────┐
         │  Feature Engineering  │
         │  RFM + Lag + Rolling  │ → 5,863 customer profiles
         └────────┬──────────────┘
                  │
    ┌─────────────┼─────────────────┐
    │             │                 │
┌───▼───┐   ┌────▼────┐   ┌────────▼──────┐
│KMeans │   │ XGBoost │   │  SARIMA +     │
│  Seg. │   │  Churn  │   │  Inventory    │
└───┬───┘   └────┬────┘   └────────┬──────┘
    │             │                 │
    └─────────────┼─────────────────┘
                  │
         ┌────────▼────────┐
         │   MLflow        │ ← experiment tracking
         │   (SQLite)      │
         └────────┬────────┘
                  │
       ┌──────────┼──────────┐
       │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌───▼──────────┐
  │FastAPI │ │Streamlit│ │  Evidently   │
  │ :8000  │ │  :8501  │ │  Monitoring  │
  └────────┘ └─────────┘ └──────────────┘
📁 Project Structure
retailpulse/
├── data/
│   ├── raw/                    ← drop online_retail_II.xlsx here
│   ├── processed/              ← pipeline outputs
│   └── features/               ← engineered features
├── src/
│   ├── data/
│   │   ├── ingest.py           ← xlsx → parquet (handles both UCI sheet names)
│   │   └── clean.py            ← 8-step cleaning pipeline + DQ report
│   ├── features/
│   │   ├── rfm.py              ← RFM scoring, segment labels
│   │   └── time_features.py    ← lag, rolling, calendar features
│   ├── models/
│   │   ├── segmentation.py     ← KMeans + elbow + silhouette
│   │   ├── forecasting.py      ← SARIMA → ARIMA fallback
│   │   ├── churn.py            ← XGBoost + LogReg baseline
│   │   └── inventory.py        ← safety stock + reorder point
│   └── utils/
│       ├── config.py           ← centralised config (.env)
│       ├── logging_config.py   ← loguru
│       └── mlflow_utils.py     ← SQLite MLflow backend
├── dashboard/
│   ├── app.py                  ← home page
│   ├── utils.py                ← cached data loaders
│   └── pages/
│       ├── 1_Sales_Dashboard.py
│       ├── 2_Customer_Dashboard.py
│       ├── 3_Forecast_Dashboard.py
│       └── 4_Inventory_Dashboard.py
├── api/
│   ├── main.py
│   └── routes/
│       ├── churn.py
│       ├── forecast.py
│       ├── segments.py
│       └── inventory.py
├── monitoring/
│   ├── drift_report.py         ← Evidently drift reports
│   └── performance_tracker.py  ← rolling metrics log
├── tests/                      ← 32 unit tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── run_pipeline.py             ← full pipeline orchestrator
├── requirements.txt
├── Makefile
└── .env.example
🚀 Quick Start
Prerequisites
Python 3.11+
4 GB RAM minimum
1. Clone & Setup
git clone https://github.com/Subham503/retailpulse.git
cd retailpulse
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
2. Dataset
Download UCI Online Retail II → drop online_retail_II.xlsx into data/raw/

3. Run Pipeline
# Full pipeline (all steps)
python run_pipeline.py

# Or step by step
python -m src.data.ingest
python -m src.data.clean
python -m src.features.rfm
python -m src.features.time_features
python -m src.models.segmentation
python -m src.models.churn
python -m src.models.inventory
python -m src.models.forecasting
4. Launch Dashboard
streamlit run dashboard/app.py
# → http://localhost:8501
5. Launch API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs
6. View MLflow Experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# → http://localhost:5000
7. Run Monitoring
python -m monitoring.drift_report
python -m monitoring.performance_tracker
# → monitoring/reports/*.html
📊 Dashboard
4-page interactive Streamlit dashboard:

Page	Content
📊 Sales Dashboard	Revenue trends, top products, country breakdown, seasonality heatmap
👥 Customer Dashboard	RFM segments, churn scores, CLV scatter, at-risk customer table
📈 Forecast Dashboard	90-day SARIMA forecast with 95% confidence bands
📦 Inventory Dashboard	Reorder alerts, safety stock, SKU risk tiers
🔌 API Endpoints
Method	Endpoint	Description
GET	/health	Service health check
POST	/predict/churn	Churn probability for a customer ID
POST	/predict/forecast?days=30	N-day demand forecast
GET	/segments?limit=100	All customer RFM segments
GET	/inventory/risk?tier=High	Inventory risk table
Interactive Swagger docs at http://localhost:8000/docs

Example — Churn Prediction:

curl -X POST "http://localhost:8000/predict/churn" \
     -H "Content-Type: application/json" \
     -d '{"customer_id": "17850"}'
{
  "customer_id": "17850",
  "churn_prob": 0.0312,
  "churn_label": 0,
  "segment": "Champions",
  "recency_days": 3,
  "frequency": 45,
  "monetary": 12450.50,
  "risk_level": "Low"
}
🛠️ Tech Stack
Category	Tools
Language	Python 3.11+
Data Processing	pandas, numpy, pyarrow
Machine Learning	scikit-learn, XGBoost, statsmodels
Experiment Tracking	MLflow (SQLite backend)
Monitoring	Evidently AI
Dashboard	Streamlit, Plotly
REST API	FastAPI, uvicorn, Pydantic
Deployment	Docker, docker-compose
Testing	pytest, pytest-cov (32 tests)
Logging	loguru
📈 Dataset
UCI Online Retail II — real UK-based online retail transactions.

Attribute	Value
Source	UCI Machine Learning Repository
Raw rows	1,067,371
Clean rows	1,003,740
Unique customers	5,863
Unique SKUs	4,739
Countries	43
Total revenue	£19.7M
Date range	Dec 2009 – Dec 2011
🧪 Tests
pytest tests/ -v --cov=src --cov-report=term-missing
32 passed in 1.35s
Coverage: src/data 94% | src/features 91% | src/models 87%
🐳 Docker
# Build & run all services
docker compose -f docker/docker-compose.yml up

# Services:
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
# MLflow    → http://localhost:5000
👤 Author
Suresh Kumar Data Science & Analytics Intern — Zidio Development (May–June 2026)

GitHub LinkedIn


Built with ❤️ as part of Zidio Development Data Science & Analytics Internship