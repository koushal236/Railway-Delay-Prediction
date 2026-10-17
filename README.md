# 🚆 Railway Delay Prediction & Ecological ML Intelligence Platform
## 23CSE301 Machine Learning Capstone Project

A comprehensive, production-grade Machine Learning system integrating **Supervised Classification**, **Supervised Regression**, and **Unsupervised Clustering**, accompanied by reproducible Jupyter research notebooks, serialized model pipelines, publication-quality visualizations, and an interactive **Streamlit Web Application** ready for public cloud deployment.

---

## 📌 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Machine Learning Tracks & Methodologies](#-machine-learning-tracks--methodologies)
   - [Track 1: Railway Delay Classification](#track-1-railway-delay-classification)
   - [Track 2: Railway Delay Duration Regression](#track-2-railway-delay-duration-regression)
   - [Track 3: Forest CoverType Unsupervised Clustering](#track-3-forest-covertype-unsupervised-clustering)
3. [Empirical Results & Leaderboards](#-empirical-results--leaderboards)
4. [Dataset Specifications & Git LFS Handling](#-dataset-specifications--git-lfs-handling)
5. [Repository Folder Architecture](#-repository-folder-architecture)
6. [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
   - [Windows PowerShell Setup](#windows-powershell-setup)
   - [Dependency Installation](#dependency-installation)
7. [Running the Jupyter Notebooks](#-running-the-jupyter-notebooks)
8. [Running the Streamlit Web Application](#-running-the-streamlit-web-application)
9. [Streamlit Community Cloud Deployment Guide](#-streamlit-community-cloud-deployment-guide)
10. [GitHub ZIP Download & Portability Instructions](#-github-zip-download--portability-instructions)
11. [Academic Integrity & Audit Checklist](#-academic-integrity--audit-checklist)

---

## 📖 Project Overview & Objectives

In modern rail networks, delay propagation cascades through interdependent schedules, crew rosters, and physical route congestion. This capstone project formulates a unified predictive and analytical machine learning suite designed to:
- **Classify Binary Journey Risk:** Accurately determine whether a scheduled passenger journey will suffer an arrival delay (`is_delayed = 1` vs. `0`).
- **Forecast Continuous Delay Duration:** Quantify the exact expected delay magnitude in minutes (`delay_minutes`) using nonlinear ensemble and regularized regressors.
- **Discover Unsupervised Environmental Partitions:** Identify natural topographic and insolation eco-niches using the high-dimensional Forest CoverType benchmark without ground-truth label leakage.
- **Deliver Real-Time Accessible Decision Support:** Provide an intuitive, dark-mode, high-contrast Streamlit interface for dispatchers and transit researchers.

---

## 🔬 Machine Learning Tracks & Methodologies

### Track 1: Railway Delay Classification
- **Dataset:** `data/ir_train_siddu.csv` (1,500,000 journey records, 41 input features).
- **Target:** `is_delayed` (Binary: 0 = On-Time, 1 = Delayed).
- **Data Hygiene:** Target leakage columns (`delay_minutes`, `primary_delay_cause`) and raw identifiers (`journey_id`, `departure_date`) dropped prior to splitting.
- **Preprocessing:** Leakage-safe `ColumnTransformer` (Median imputation + `StandardScaler` on 34 continuous predictors; Mode imputation + `OneHotEncoder(drop='first')` on 7 categorical features). Fitted strictly on `X_train`.
- **Feature Engineering:** `scheduled_speed_kmph = distance_km / scheduled_travel_hours`.
- **Ten Algorithms Evaluated:**
  - *Part A:* Logistic Regression, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Decision Tree, Support Vector Machine (Linear SVC).
  - *Part B:* Random Forest, AdaBoost Classifier, Gradient Boosting Classifier, Bagging Classifier (Decision Tree base), Multi-Layer Perceptron (MLP Neural Network).
- **Champion Model:** **Logistic Regression** (Peak test ROC-AUC: `0.9220`, Accuracy: `86.09%`, Recall on delays: `92.28%`).
- **Hyperparameter Optimization:** AdaBoost tuned via `GridSearchCV` (`n_estimators=100`, `learning_rate=1.0`), achieving a measured `+0.43%` accuracy gain.

### Track 2: Railway Delay Duration Regression
- **Authoritative Workflow:** Strictly derived from `notebooks/regression.ipynb` (`IR_Regression_Review1_Final_Corrected.ipynb`).
- **Dataset:** `data/IR_regression_dataset.csv` (100,000 journey records, 44 input columns).
- **Target:** `delay_minutes` (Continuous numerical delay duration).
- **Partitioning:** Target-stratified 80:20 split using 10 quantile bins (`random_state=42`).
- **Five Domain Engineered Ratios:**
  $$\text{stops\_per\_100km} = \left(\frac{\text{num\_scheduled\_stops}}{\text{distance\_km}}\right) \times 100$$
  $$\text{avg\_rolling\_stock\_age} = \frac{\text{loco\_age\_years} + \text{coach\_age\_years}}{2.0}$$
  $$\text{operational\_stress} = \text{seat\_utilisation\_pct} \times (1.0 + \text{is\_overloaded})$$
  $$\text{weather\_risk} = \text{fog\_risk\_score} \times (1.0 + \text{is\_monsoon\_season})$$
  $$\text{route\_efficiency} = \frac{\text{distance\_km}}{\text{scheduled\_travel\_hours}}$$
- **Ten Regressors Evaluated:** Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression (Degree 3), Decision Tree, Random Forest, Gradient Boosting Regressor, Support Vector Regressor (SVR), K-Nearest Neighbors Regressor.
- **Champion Model:** **Gradient Boosting Regressor** ($R^2 = 0.4859$, $\text{RMSE} = 47.33\text{ min}$, $\text{MAE} = 34.77\text{ min}$, 5-Fold Cross-Validation Mean $R^2 = 0.4921 \pm 0.0046$).

### Track 3: Forest CoverType Unsupervised Clustering
- **Dataset:** Forest CoverType benchmark (`sklearn.datasets.fetch_covtype`, 581,012 records, 54 features).
- **Unsupervised Integrity:** Known target labels ($y$, Cover Type 1–7) are **strictly excluded** from clustering inputs.
- **Standardization:** Continuous cartographic variables (10 features) standardized via `StandardScaler`; binary indicator features (44 features: 4 wilderness areas, 40 soil types) preserved unscaled.
- **Algorithms:** K-Means clustering (evaluated across $k = 2$ through $8$) and Agglomerative Hierarchical clustering (Ward, Complete, and Average linkages).
- **Metric Suite:** Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index, and Within-Cluster Sum of Squares (Inertia).
- **Cluster Selection:** Mathematically optimal at **$k=2$** (Peak Silhouette: `0.1903`, Peak Calinski–Harabasz: `648.4`), with **$k=3$** representing the primary elbow inflection ($\Delta\text{Inertia} = 20,103.0$).
- **Visualizations:** Inertia Elbow Curve with marginal reduction bars, Ward Hierarchical Dendrogram, PCA 2D projections, and t-SNE 2D manifold embeddings.

---

## 📊 Empirical Results & Leaderboards

### Supervised Classification Leaderboard (Test Partition N = 30,000)
| Rank | Algorithm | Accuracy | Precision | Recall (Delay) | F1-Score | ROC-AUC | Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Logistic Regression** | **86.09%** | **88.80%** | **92.28%** | **85.85%** | **0.9220** | Linear Champion |
| 🥈 | Multi-Layer Perceptron (MLP) | 85.63% | 88.49% | 91.97% | 85.39% | 0.9160 | Neural Network |
| 🥉 | Gradient Boosting Classifier | 85.60% | 87.53% | 93.25% | 85.17% | 0.9162 | Boosting Ensemble |
| 4 | AdaBoost Classifier (Tuned) | 85.44% | 87.57% | 92.83% | 85.04% | 0.9150 | Boosting Ensemble |
| 5 | Random Forest Classifier | 84.79% | 88.16% | 90.79% | 84.50% | 0.9120 | Bagging Ensemble |
| 6 | Bagging (Decision Tree) | 84.39% | 87.97% | 90.35% | 84.07% | 0.9080 | Bagging Ensemble |
| 7 | Support Vector Machine (Linear) | 84.38% | 87.96% | 90.34% | 84.06% | 0.9075 | Maximum Margin |
| 8 | Gaussian Naive Bayes | 82.90% | 86.90% | 89.20% | 82.50% | 0.8910 | Probabilistic |
| 9 | Decision Tree Classifier | 82.89% | 86.88% | 89.18% | 82.49% | 0.8905 | Decision Trees |
| 10 | K-Nearest Neighbors (KNN) | 80.83% | 85.20% | 87.90% | 80.40% | 0.8750 | Instance-Based |

### Supervised Regression Leaderboard (Test Partition N = 20,000)
| Rank | Model | $R^2$ Score | RMSE (Minutes) | MAE (Minutes) | 5-Fold CV Mean $R^2$ | Paradigm |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting Regressor** | **0.4859** | **47.33** | **34.77** | **0.4921 ± 0.005** | Gradient Boosting |
| 🥈 | Random Forest Regressor (Tuned) | 0.4728 | 47.93 | 34.58 | 0.4802 ± 0.005 | Random Forest |
| 🥉 | Random Forest Regressor | 0.4716 | 47.98 | 34.62 | 0.4797 ± 0.005 | Random Forest |
| 4 | Polynomial Regression (Deg 3) | 0.4634 | 48.35 | 36.66 | — | Nonlinear Basis |
| 5 | Ridge Regression | 0.4602 | 48.50 | 36.88 | — | Regularized L2 |
| 6 | Linear Regression | 0.4601 | 48.50 | 36.88 | — | Ordinary Least Squares |
| 7 | Lasso Regression | 0.4595 | 48.53 | 36.93 | — | Regularized L1 |
| 8 | ElasticNet Regression | 0.4583 | 48.58 | 37.00 | — | Combined L1+L2 |
| 9 | Decision Tree Regressor | 0.4101 | 50.70 | 36.12 | — | Recursive Partitioning |
| 10 | K-Nearest Neighbors Regressor | 0.4089 | 50.75 | 38.45 | — | Metric Distance |
| 11 | Support Vector Regressor (SVR) | 0.3248 | 54.24 | 43.03 | — | Kernel Projection |

### Unsupervised Clustering Quality Metrics (Forest CoverType)
| Number of Clusters ($k$) | Cluster Status | Inertia (WCSS) | Marginal Gain ($\Delta\text{Inertia}$) | Silhouette Score | Davies–Bouldin Index | Calinski–Harabasz Index |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$k = 2$** | **Optimal Separation Peak** | 203,702.4 | — | **0.1903** | 1.9933 | **648.4** |
| **$k = 3$** | **Primary Elbow Inflection** | 183,599.4 | **20,103.0** | 0.1493 | 1.9734 | 542.3 |
| $k = 4$ | Intermediate Partition | 172,139.7 | 11,459.7 | 0.1264 | 1.9774 | 479.5 |
| $k = 5$ | Intermediate Partition | 162,943.8 | 9,195.9 | 0.1378 | 1.8213 | 459.6 |
| $k = 6$ | Davies–Bouldin Minimum | 155,758.1 | 7,185.7 | 0.1420 | **1.7793** | 435.7 |
| $k = 7$ | Fine-Grained Subcluster | 149,428.8 | 6,329.3 | 0.1345 | 1.8020 | 416.3 |
| $k = 8$ | Fine-Grained Subcluster | 144,383.1 | 5,045.7 | 0.1257 | 1.8541 | 397.0 |

---

## 📦 Dataset Specifications & Git LFS Handling

1. **`IR_regression_dataset.csv` (22.28 MB, 100,000 rows):**
   - **Tracked as a regular Git file**: Because it is well below GitHub's 100 MB hard limit, it is committed directly to Git. **When downloading the repository as a ZIP from GitHub, this file is fully included and immediately usable.**
2. **`ir_train_siddu.csv` (337.16 MB, 1,500,000 rows):**
   - **Tracked via Git LFS**: Exceeds GitHub's 100 MB file limit.
   - *For `git clone` users:* Simply run `git lfs pull` after cloning.
   - *For GitHub ZIP downloaders:* GitHub's web interface does not package large LFS assets inside ZIPs (it provides a small pointer stub). To verify your local datasets, execute:
     ```powershell
     python data/verify_datasets.py
     ```
3. **Streamlit GUI Independence:**
   - **The GUI does NOT require the 337 MB CSV file to run!** All pre-trained pipelines (`models/`) and curated benchmark test presets (`data/`) are pre-serialized inside `GUI_Deployment/`. The GUI starts up instantly and produces real predictions offline.

---

## 📁 Repository Folder Architecture

```
Railway-Delay-Prediction/
├── README.md                          # Master capstone documentation (this file)
├── requirements.txt                   # Root dependency specification for all workflows
├── .gitignore                         # Comprehensive ignore rules (caches, envs, checkpoints)
├── .gitattributes                     # Git LFS specification strictly for files >100 MB
├── notebooks/                         # Self-contained, portable Jupyter notebooks
│   ├── classification.ipynb           # Executed 10-classifier Review 2 notebook
│   ├── regression.ipynb               # Executed 10-regressor authoritative notebook
│   ├── clustering.ipynb               # Executed Forest CoverType clustering notebook
│   ├── IR_Classification_Review1.ipynb# Preserved Review 1 classification workflow
│   ├── IR_Regression_Review1_Final_Corrected.ipynb  # Authoritative regression notebook (canonical name)
│   └── IR_Regression_Archive.ipynb    # Preserved Review 1 exploratory regression notebook
├── data/                              # Primary dataset directory
│   ├── README.md                      # Dataset schemas, column dictionaries, and LFS notes
│   ├── verify_datasets.py             # Integrity checker for CSVs and LFS pointers
│   ├── IR_regression_dataset.csv      # Regression dataset (22.3 MB, fully bundled)
│   └── ir_train_siddu.csv             # Classification dataset (337.2 MB, Git LFS tracked)
├── GUI_Deployment/                    # Standalone, deployment-ready Streamlit application
│   ├── .streamlit/
│   │   └── config.toml                # Dark theme configuration (high contrast styling)
│   ├── streamlit_app.py               # Master multi-page application (5 tabs)
│   ├── requirements.txt               # Locked dependencies for Streamlit Cloud deployment
│   ├── README.md                      # Streamlit Community Cloud public deployment guide
│   ├── models/                        # Serialized ML pipeline artifacts (Instant inference)
│   │   ├── best_classifier_pipeline.joblib   # Champion Logistic Regression pipeline
│   │   ├── best_regressor_pipeline.joblib    # Champion Gradient Boosting Regressor pipeline
│   │   ├── best_kmeans_model.joblib          # Champion K-Means model (k=2)
│   │   └── covertype_scaler.joblib           # Pre-fitted CoverType continuous StandardScaler
│   ├── figures/                       # 13 high-resolution visualization figures
│   └── data/                          # Curated benchmark presets for 1-click UI evaluation
└── Review2_Final_Deliverables/        # Preserved submission directory from Capstone Review 2
```

---

## ⚙️ Prerequisites & Environment Setup

This project is tested and verified on **Python 3.10 and 3.11** on 64-bit Windows.

### Windows PowerShell Setup

Open PowerShell in the repository root directory:

```powershell
# 1. Create a clean virtual environment
python -m venv .venv

# 2. Allow local script execution (if restricted) and activate virtual environment
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### Dependency Installation

Upgrade `pip` and install all dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 📓 Running the Jupyter Notebooks

All notebooks feature **dynamic relative path resolution** and run seamlessly from any working directory:

### Running via Jupyter Lab / Notebook
```powershell
jupyter notebook
```
Navigate to the `notebooks/` folder and open:
- `notebooks/classification.ipynb`: Trains and benchmarks all 10 classifiers.
- `notebooks/regression.ipynb`: Trains and benchmarks all 10 regressors.
- `notebooks/clustering.ipynb`: Runs K-Means and Agglomerative clustering on Forest CoverType.

### Headless Execution via Command Line
To re-execute any notebook non-interactively:
```powershell
python -m nbconvert --to notebook --execute --inplace notebooks/regression.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/clustering.ipynb
```

---

## 🖥️ Running the Streamlit Web Application

Launch the unified GUI directly from the repository root:

```powershell
python -m streamlit run GUI_Deployment/streamlit_app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

### Highlights of the GUI:
- **🏠 Home / Project Overview:** Academic executive summary, interactive milestone tables, and rubric compliance badges.
- **🚆 Classification Page:** Real-time delay risk prediction (`is_delayed`) with probability breakdowns, curated presets (*Normal Express* vs. *Severe Fog Journey*), ROC curves, and confusion matrices.
- **⏱️ Regression Page:** Real-time continuous delay forecasting (`delay_minutes`), formatted hours/minutes duration, disruption severity badges (Minimal, Moderate, Severe), and residual scatter plots.
- **🌲 Clustering Page:** Interactive cartographic sliders (Elevation, Aspect, Slope, Hydrology), real-time cluster assignment, physical eco-niche profiling, elbow curve, PCA, and t-SNE projections.
- **📊 Model Results & Comparison:** Cross-paradigm leaderboard tables, 5-fold cross-validation metrics, and rubric compliance verification.

---

## 🌐 Streamlit Community Cloud Deployment Guide

To deploy this application publicly and secure bonus marks for interactive GUI deployment:

1. **Push your repository to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "feat: complete ML capstone project with Streamlit GUI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Railway-Delay-Prediction.git
   git push -u origin main
   ```
2. **Connect to Streamlit Cloud:**
   - Log into [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
   - Click **"Create app"** / **"New app"**.
   - Select your repository (`Railway-Delay-Prediction`) and branch (`main`).
   - Set **Main file path** to:
     ```text
     GUI_Deployment/streamlit_app.py
     ```
   - Click **"Deploy!"**.
3. Streamlit Cloud will detect `GUI_Deployment/requirements.txt`, install dependencies, and serve your live web application at a public URL (e.g. `https://railway-delay-prediction.streamlit.app`).

---

## 📥 GitHub ZIP Download & Portability Instructions

If downloading this repository as a `.zip` file from GitHub onto a new computer:

1. Extract the downloaded ZIP into any local folder (e.g. `C:\Users\Student\Railway-Delay-Prediction`).
2. Open PowerShell in the extracted folder.
3. Set up the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
4. **Immediate GUI Access:** You can launch the GUI immediately without downloading any large files:
   ```powershell
   python -m streamlit run GUI_Deployment/streamlit_app.py
   ```
5. **Immediate Regression Access:** `data/IR_regression_dataset.csv` is fully included in the ZIP, allowing `notebooks/regression.ipynb` to execute out of the box.
6. **For Classification Notebook:** If `data/ir_train_siddu.csv` appears as an LFS pointer, pull it via `git lfs pull` or refer to `data/README.md`.

---

## 🛡️ Academic Integrity & Audit Checklist

- [x] **Zero Data Leakage:** Preprocessing scalers (`StandardScaler`) and encoders (`OneHotEncoder`) are fitted strictly on `X_train` partitions inside Scikit-Learn `Pipeline` architectures.
- [x] **Strict Reproducibility:** Fixed `random_state=42` enforced across all splits, model initializations, and stochastic procedures.
- [x] **No Data Fabrication:** All metrics in this README and the web application reflect actual outputs produced by running the notebooks.
- [x] **Unsupervised Purity:** Ground-truth labels are excluded from clustering inputs and used solely for post-hoc validation.
- [x] **Portable Relative Paths:** No hardcoded absolute paths (`C:\Users\...`) exist in any active notebook, script, or configuration.
- [x] **Independent GUI Inference:** Models are pre-serialized into Joblib format; predictions execute instantly without retraining on button clicks.
