# NLP Quality Analytics Demo Project

A comprehensive demonstration of NLP capabilities for quality and compliance analytics using Domino Data Lab platform.

## 🎯 Demo Overview

This project showcases enterprise-grade NLP workflows for quality and compliance analytics, featuring:

- **Semantic Search Engine**: BERT-powered search across quality findings
- **Topic Modeling**: Automated discovery of quality themes and patterns  
- **Category Classification**: Automated categorization of findings
- **Model APIs**: Production-ready model endpoints
- **Experiment Tracking**: MLflow-based model governance

## 📊 Key Metrics Demonstrated

- **170+ active users** across the organization
- **7,000+ unique searches** performed over 2.5 years
- **86% Top-1 accuracy** and **95% Top-3 accuracy** for category prediction
- **0.84 topic coherence score** for topic modeling
- **Real-time search** with < 50ms average query time

## 🏗️ Project Structure

```
NLP_Quality_Analytics/
├── apps/
│   ├── semantic_search/          # Main search application
│   │   ├── app.py               # Streamlit search interface
│   │   ├── app.sh               # Domino app launcher
│   │   └── requirements.txt
│   └── topic_explorer/          # Topic modeling interface
│       ├── app.py               # Topic analysis app
│       ├── app.sh               # Domino app launcher
│       └── requirements.txt
├── notebooks/
│   ├── 01_quick_data_overview.ipynb    # Data exploration
│   └── 02_model_validation.ipynb      # Model testing
├── models/
│   ├── model_api.py             # Flask API for model serving
│   └── train_models.py          # Model training scripts
├── experiments/
│   └── mlflow_tracking.py       # Experiment tracking demo
├── src/
│   ├── search_engine.py         # Search backend utilities
│   ├── topic_modeling.py        # Topic modeling functions
│   └── data_utils.py            # Data processing helpers
└── README.md
```

## 📁 Dataset Structure

The project uses the `quality_compliance_data` Domino dataset:

```
/mnt/data/quality_compliance_data/
├── audit_findings.csv              # 500+ quality findings
├── sample_queries.json             # Demo search queries
├── metadata.json                   # Dataset information
├── embeddings/
│   └── bert_embeddings.pkl         # Pre-computed embeddings
└── models/
    ├── topic_model.pkl             # Trained topic model
    └── category_classifier.pkl     # Classification model
```

## 🚀 Quick Start Guide

### 1. Launch Semantic Search App

The main demo application - a Google-like search interface for quality findings:

```bash
# Navigate to the app directory and launch
cd /mnt/code/apps/semantic_search
./app.sh
```

**Demo Features:**
- Natural language search across quality findings
- Real-time similarity scoring and ranking
- Advanced filtering by category, severity, and area
- Search analytics and trending queries
- Professional UI with responsive design

### 2. Launch Topic Explorer

Interactive topic modeling and analysis:

```bash
cd /mnt/code/apps/topic_explorer  
./app.sh
```

**Demo Features:**
- Topic distribution visualization
- Topic evolution over time
- Topic relationship network
- Detailed topic analysis with keywords

### 3. Model API Demo

Deploy and test the classification model API:

```bash
cd /mnt/code/models
python model_api.py
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information
- `GET /categories` - Available categories

### 4. Experiment Tracking

View MLflow experiments and model governance:

```bash
cd /mnt/code/experiments
python mlflow_tracking.py
```

## 🎪 Demo Flow (85 minutes)

### Section 1: NLP Quality & Compliance Pipeline (45 minutes)

#### Part 1.1: Project Setup & Core Workflow (8 minutes)
- **Show**: Project overview and dataset connections
- **Demonstrate**: 
  - Domino project structure and configuration
  - Dataset mounting and data access
  - Git integration and collaboration setup
  - Docker environment with NLP libraries

#### Part 1.2: Semantic Search Application (15 minutes)
- **Show**: Main interactive search demo
- **Demonstrate**:
  - Real-time semantic search with BERT embeddings
  - Similarity scoring and result ranking
  - Search analytics and trending queries
  - Advanced filtering and categorization
  - Professional UI showcasing enterprise readiness

#### Part 1.3: Model API & Deployment (7 minutes)
- **Show**: Production model deployment
- **Demonstrate**:
  - Model API endpoints and testing
  - Category prediction with confidence scores
  - Model versioning and performance metrics
  - API documentation and usage examples

#### Part 1.4: Topic Modeling Explorer (10 minutes)
- **Show**: Topic discovery and analysis
- **Demonstrate**:
  - Interactive topic visualization
  - Topic evolution over time
  - Machine-learned vs manual categorization
  - Topic relationship networks

#### Part 1.5: Experiment Tracking & Governance (5 minutes)
- **Show**: MLflow model governance
- **Demonstrate**:
  - Experiment tracking and comparison
  - Model performance metrics
  - Hyperparameter optimization results
  - Model registry and versioning

## 🛠️ Technical Stack

- **Frontend**: Streamlit for interactive applications
- **Backend**: Flask APIs for model serving
- **ML Models**: BERT, Sentence Transformers, LDA
- **Experiment Tracking**: MLflow
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, NetworkX
- **Infrastructure**: Domino Data Lab platform

## 🎯 Key Demo Points

### Business Value
- **Efficiency**: 2.5 years in production with 170+ users
- **Scale**: 7,000+ searches across quality compliance data
- **Accuracy**: 86% Top-1, 95% Top-3 classification accuracy
- **Speed**: Real-time search with enterprise-grade performance

### Technical Excellence
- **Enterprise Ready**: Production deployment with APIs
- **Scalable**: Handles large document collections efficiently  
- **Governed**: Full experiment tracking and model versioning
- **Collaborative**: Multi-user platform with role-based access

### Platform Capabilities
- **Rapid Deployment**: One-click app deployment with app.sh
- **Flexible Infrastructure**: GPU support, custom environments
- **Data Management**: Secure dataset mounting and sharing
- **Reproducibility**: Complete experiment tracking and versioning

## 📝 Notes for Demo

### Preparation Checklist
- [ ] Upload sample dataset to `quality_compliance_data`
- [ ] Test both Streamlit applications
- [ ] Verify model API endpoints
- [ ] Check MLflow experiment logs
- [ ] Prepare sample search queries
- [ ] Test all demo flows

### Demo Tips
- Focus on visual applications rather than code notebooks
- Emphasize business metrics and user adoption
- Show real-time interaction and responsiveness
- Highlight enterprise features (governance, security, scale)
- Keep notebook demos brief (2-3 minutes max)

### Fallback Options
- All applications include simulated data if datasets unavailable
- Model APIs work with mock predictions for reliability
- Search functionality uses deterministic similarity scoring
- Topic modeling includes pre-generated visualizations

## 🔧 Customization for Other Customers

This demo is designed to be customer-agnostic:

- **Generic branding**: No company-specific names or logos
- **Configurable data**: Easy to swap datasets and use cases
- **Flexible metrics**: Adjustable performance numbers
- **Reusable components**: Modular application structure
- **Professional styling**: Enterprise-ready UI/UX

Simply update the dataset and customize the sample queries to match any customer's quality/compliance use case.

---

**Demo Contact**: Update with your contact information
**Last Updated**: January 2025
**Version**: 1.0.0