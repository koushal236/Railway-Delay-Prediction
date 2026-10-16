# 🚆 Railway Delay & Ecological ML Intelligence Platform
## 23CSE301 Machine Learning Capstone — Unified Web Application & Public Deployment Suite

**Course:** 23CSE301 Machine Learning Capstone  
**Reference Architecture:** Clean Academic Engineering GUI (inspired by [tensortitans.streamlit.app](https://tensortitans.streamlit.app/))  
**Deployment Target:** Streamlit Community Cloud ([share.streamlit.io](https://share.streamlit.io))  
**Primary Deliverables:** Interactive Multi-Page Application, Pre-trained Pipeline Artifacts, Curated Benchmark Presets, and Leaderboards.

---

## 📌 Executive Summary

This application brings together all three machine learning tracks developed across the capstone reviews into one self-contained, deployment-ready web application:

1. **Supervised Classification — Railway Delay Prediction:**
   - **Dataset:** Indian Railways operational records (1,500,000 instances, 41 input features).
   - **Target:** `is_delayed` (Binary: 0 = On-Time, 1 = Delayed).
   - **Leaderboard:** 10 Classifiers evaluated (Part A: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM; Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP).
   - **Champion Model:** **Logistic Regression** (Accuracy: **86.09%**, Test ROC-AUC: **0.9220**, Delay Recall: **92.28%**, Weighted F1: **85.85%**).
   - **Hyperparameter Tuning:** AdaBoost tuned via GridSearchCV gaining **+0.43%** accuracy (85.01% → 85.44%).

2. **Supervised Regression — Railway Delay Duration Forecasting:**
   - **Authoritative Notebook:** Strictly based on `IR_Regression_Review1_Final_Corrected(1).ipynb`.
   - **Dataset:** 100,000 records, 44 input columns.
   - **Target:** `delay_minutes` (Continuous delay in minutes).
   - **Leaderboard:** 10 Continuous Regressors evaluated (Linear, Ridge, Lasso, ElasticNet, Polynomial Degree 3, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN).
   - **Champion Model:** **Gradient Boosting Regressor** (Test $R^2$: **0.4859**, RMSE: **47.33 minutes**, MAE: **34.77 minutes**; 5-Fold Cross-Validation Mean $R^2$: **0.4921 ± 0.0046**).
   - **Hyperparameter Tuning:** Random Forest tuned from $R^2 = 0.4716$ to $0.4728$.

3. **Unsupervised Clustering — Forest CoverType Ecological Segmentation:**
   - **Dataset:** Forest CoverType (`fetch_covtype()`, 581,012 records, 54 features).
   - **Leakage Prevention:** Ground-truth `Cover_Type` target labels strictly excluded from clustering inputs.
   - **Evaluation across $k=2$ to $k=8$:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, Inertia Elbow Analysis.
   - **Optimal Selection Justification:** **$k=2$** provides peak global cohesion and separation (Silhouette **0.1903**, Calinski-Harabasz **648.4**), while **$k=3$** represents the primary elbow inflection point ($\Delta\text{Inertia} = 20,103.0$).
   - **Hierarchical Clustering:** Evaluated Ward, Complete, and Average linkages with Dendrogram cutoffs, PCA 2D projections, and t-SNE 2D manifold visualizations.

---

## 📁 Folder Structure

```
GUI_Deployment/
├── .streamlit/
│   └── config.toml                    # Streamlit theme configuration (dark canvas, crisp typography)
├── streamlit_app.py                   # Master multi-page Streamlit application
├── requirements.txt                   # Locked production dependencies
├── README.md                          # Deployment guide and technical documentation
├── models/                            # Pre-trained, serialized pipeline artifacts (Joblib)
│   ├── best_classifier_pipeline.joblib   # Champion Logistic Regression Pipeline (with Scaler + OneHotEncoder)
│   ├── best_regressor_pipeline.joblib    # Champion Gradient Boosting Regressor Pipeline
│   ├── best_kmeans_model.joblib          # Champion K-Means Clustering Model (k=2)
│   └── covertype_scaler.joblib           # Pre-fitted StandardScaler on CoverType continuous features
├── figures/                       # High-resolution benchmark figures and visualizations
│   ├── classification_metrics_comparison.png
│   ├── classification_tuning_comparison.png
│   ├── confusion_matrices_part_b.png
│   ├── rf_feature_importance.png
│   ├── roc_curves_all_10_models.png
│   ├── regression_models_comparison.png
│   ├── regression_residuals_plot.png
│   ├── kmeans_elbow_curve.png
│   ├── hierarchical_dendrogram.png
│   ├── pca_kmeans_clusters.png
│   ├── pca_agglomerative_clusters.png
│   ├── tsne_clusters.png
│   └── post_hoc_cluster_distribution.png
└── data/                          # Benchmark sample presets for 1-click testing
    ├── classification_samples.json
    ├── regression_samples.json
    └── clustering_samples.json
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10+ or 3.11 is installed.

### 2. Install Dependencies
Navigate into the `GUI_Deployment` directory and install the requirements:
```bash
cd GUI_Deployment
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 Public Deployment Guide (Streamlit Community Cloud)

To deploy this application publicly and secure bonus marks for interactive GUI deployment:

### Step 1: Prepare GitHub Repository
1. Initialize a Git repository (or use your existing capstone repository):
   ```bash
   git init
   git add .
   git commit -m "feat: complete ML capstone Streamlit application"
   git push origin main
   ```
2. Ensure the `GUI_Deployment/` directory is committed with its `models/`, `figures/`, `data/`, `streamlit_app.py`, and `requirements.txt`. (All model files combined are under 400 KB, well within standard GitHub limits—no Git LFS required).

### Step 2: Connect to Streamlit Community Cloud
1. Visit [https://share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"**.
3. Select:
   - **Repository:** `your-username/Railway-Delay-Prediction`
   - **Branch:** `main`
   - **Main file path:** `GUI_Deployment/streamlit_app.py`
4. Click **"Deploy!"**.

### Step 3: Verify Live Deployment
Within 1–2 minutes, Streamlit Cloud builds the container, installs `requirements.txt`, and serves the live application at a public URL (e.g., `https://your-app-name.streamlit.app`).

---

## 🔬 Interactive Features & Rubric Highlights

### 1. Classification Track
- **Interactive Prediction:** Users can select presets (e.g. *Normal On-Time Express* vs *High-Risk Fog Delayed*) or manually configure 41 features.
- **Instant Inference:** Runs through `best_classifier_pipeline.joblib`.
- **Probabilities & Metrics:** Outputs predicted class (`Delayed` vs `On-Time`), confidence percentage, and key risk indicators.
- **Dashboards:** Visualizes ROC curves for all 10 models, confusion matrices for Part B models, and Gini feature importances.

### 2. Regression Track
- **Exact Continuous Prediction:** Computes predicted arrival delay duration in minutes.
- **Engineered Domain Ratios:** Automatically calculates:
  $$\text{stops\_per\_100km} = \left(\frac{\text{num\_scheduled\_stops}}{\text{distance\_km}}\right) \times 100$$
  $$\text{avg\_rolling\_stock\_age} = \frac{\text{loco\_age\_years} + \text{coach\_age\_years}}{2}$$
  $$\text{operational\_stress} = \text{seat\_utilisation\_pct} \times (1 + \text{is\_overloaded})$$
  $$\text{weather\_risk} = \text{fog\_risk\_score} \times (1 + \text{is\_monsoon\_season})$$
  $$\text{route\_efficiency} = \frac{\text{distance\_km}}{\text{scheduled\_travel\_hours}}$$
- **Disruption Tiers:** Categorizes predictions into Minimal (<15m), Moderate (15-60m), and Severe (>60m) disruption.
- **Dashboards:** Visualizes 10-model $R^2$/RMSE comparison and actual vs predicted residual distributions.

### 3. Clustering Track
- **Ecological Profile Discovery:** Slider controls for cartographic elevation, aspect, slope, hydrology, and solar insolation.
- **Cluster Assignment:** Predicts natural cluster membership ($k=2$) and displays the physical eco-niche profile (East-facing high morning insolation vs West-facing high afternoon solar radiation).
- **Dashboards:** Full multi-metric table across $k=2..8$, elbow curve with marginal inertia reduction, PCA 2D projections, t-SNE 2D manifold, and dendrogram.

---

## 🛡️ Academic Integrity & Verification Guarantee

- **Zero Data Leakage:** All scalers and encoders are encapsulated strictly inside pipelines fitted exclusively on training sets.
- **Zero Fabrication:** Every displayed metric, table row, and chart reflects reproducible results generated directly by executing project notebooks with `random_state=42`.
- **Offline / Non-blocking Inference:** Pre-serialized models ensure instantaneous UI response times with zero re-training on user button clicks.
