# Indian Railway Delay Prediction & Forest CoverType Unsupervised Clustering
## Machine Learning Capstone Project — Review 2 Deliverables
**Course Code:** 23CSE301 Machine Learning  
**Evaluation Scope:** Review 2 (Classification Part B + Full Clustering Track)  
**Academic Year:** 2026-27  

---

## 1. Project Overview & Problem Statement

This project delivers a comprehensive, academically verified Machine Learning pipeline addressing two core problem tracks:

1. **Supervised Classification Track (Railway Delay Prediction):**  
   Predicting whether an Indian Railway train journey experiences operational delay (`is_delayed = 1`) or operates on-time (`is_delayed = 0`). Using historical journey attributes, infrastructure constraints, weather risks, operational scores, and route reliability records, we build, evaluate, and tune ten distinct classifiers spanning linear, instance-based, probabilistic, tree-based, ensemble, and neural architectures.

2. **Unsupervised Clustering Track (Forest CoverType Discovery):**  
   Unsupervised structural discovery and geometric density analysis on high-dimensional environmental data. The objective is to identify natural clusters based strictly on cartographic, hydrological, and pedological features without using any ground-truth categorical labels.

---

## 2. Dataset Plan & Justification for Separate Datasets

In strict accordance with the faculty-approved dataset plan and project guidelines:

| Track | Dataset Name | Records | Features | Target / Ground Truth Handling |
| :--- | :--- | :--- | :--- | :--- |
| **Classification** | Indian Railway Dataset (`ir_train_siddu.csv`) | 1,500,000 | 45 raw columns | Binary target `is_delayed` (0 = On-time, 1 = Delayed) |
| **Clustering** | Forest CoverType (`fetch_covtype`) | 581,012 | 54 input features | Known cover types (1–7) **strictly excluded** from clustering inputs |

### Why Separate Datasets Are Used:
- **Task Alignment:** The railway dataset was designed specifically for binary delay classification and regression. In contrast, the Forest CoverType dataset (`fetch_covtype`) is an internationally recognized benchmark for unsupervised pattern discovery due to its rich, high-dimensional feature space (10 continuous cartographic variables + 44 binary wilderness and soil indicators).
- **Faculty Approval:** As formally approved by course faculty, using Forest CoverType for the clustering track allows evaluation of clustering geometry on complex multi-modal spatial data, while preserving the original railway delay dataset for Review 1 and Review 2 classification.

---

## 3. Repository & Deliverables Directory Structure

```
Review2_Final_Deliverables/
├── notebooks/
│   ├── classification.ipynb       # Fully executed Review 1 + Review 2 Part B classification notebook
│   └── clustering.ipynb           # Fully executed Review 2 unsupervised clustering notebook
├── figures/                       # Exported publication-ready figures & visualizations
│   ├── classification_metrics_comparison.png
│   ├── classification_tuning_comparison.png
│   ├── confusion_matrices_part_b.png
│   ├── roc_curves_all_10_models.png
│   ├── rf_feature_importance.png
│   ├── kmeans_elbow_curve.png
│   ├── hierarchical_dendrogram.png
│   ├── pca_kmeans_clusters.png
│   ├── pca_agglomerative_clusters.png
│   ├── tsne_clusters.png
│   └── post_hoc_cluster_distribution.png
├── models/                        # Serialized champion models (.joblib)
│   ├── best_classifier_pipeline.joblib
│   └── best_kmeans_model.joblib
├── requirements.txt               # Fully specified Python dependency manifest
├── Review2_Compliance_Report.md   # Rigorous audit report distinguishing verified results from saved claims
└── README.md                      # Project documentation and reproduction guide
```

---

## 4. Preprocessing & Data Leakage Prevention

### Classification Preprocessing:
1. **Portable Dataset Resolution:** The classification notebook dynamically searches standard relative project paths (`../data/`, `data/`, `../../data/`) and traverses upward without any machine-specific hardcoded absolute paths, ensuring seamless execution across Linux, macOS, and Windows.
2. **Strict Leakage Prevention:**
   - Excluded `delay_minutes`: Directly measures delay magnitude (the exact target proxy).
   - Excluded `primary_delay_cause`: Categorical description assigned post-delay.
   - Excluded `journey_id` (non-generalizing identifier) and raw `departure_date` (calendar features already extracted).
3. **Domain Feature Engineering:** Engineered planned average speed:
   $$\text{scheduled\_speed\_kmph} = \frac{\text{distance\_km}}{\text{scheduled\_travel\_hours}}$$
4. **Leakage-Safe Pipelines:** An 80:20 stratified split was performed on the cleaned data (`random_state=42`). Preprocessing (`StandardScaler` for numeric, `OneHotEncoder(drop='first', handle_unknown='ignore')` for categoricals) was fitted **strictly on training data** within scikit-learn `Pipeline` and `ColumnTransformer` objects.

### Clustering Preprocessing:
1. **Target Exclusion:** Known forest cover-type labels (`y`) were strictly excluded from model inputs and preprocessing.
2. **Feature Scaling:** `StandardScaler` was fitted strictly on the 10 continuous numerical features (centering to $\mu=0, \sigma=1$). The 44 binary indicator features (Wilderness Areas 0–3 and Soil Types 0–39) were preserved as discrete spatial flags.
3. **Reproducible Sampling:** To address the $O(N^2)$ memory and computational limits of Agglomerative clustering and pairwise distance calculations on 581,012 records, deterministic samples (`random_state=42`) were utilized:
   - Primary evaluation sample: $N = 20,000$ records (K-Means fitting & elbow curve).
   - Pairwise metric and Agglomerative sample: $N = 3,000$ records.
   - Dendrogram sample: $N = 500$ records (truncated hierarchy).
   - t-SNE manifold sample: $N = 1,500$ records.

---

## 5. Classification Track (Review 1 Part A + Review 2 Part B)

### 5.1 Ten-Algorithm Comparative Results Table
All ten classifiers were trained and evaluated on an identical held-out test split of **30,000 records** drawn with `random_state=42`:

| # | Algorithm | Track / Review | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC | Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Logistic Regression** | Part A (Review 1) | **0.8609** | **0.8880** | **0.9228** | **0.8585** | **0.9220** | 0.39s |
| 2 | **MLP Classifier (Neural Net)** | Part B (Review 2) | **0.8563** | **0.8849** | **0.9197** | **0.8539** | **0.9160** | 1.38s |
| 3 | **Gradient Boosting** | Part B (Review 2) | **0.8560** | 0.8753 | **0.9325** | 0.8517 | **0.9162** | 10.95s |
| 4 | **AdaBoost Classifier** | Part B (Review 2) | 0.8501 | 0.8734 | 0.9256 | 0.8461 | 0.9116 | 2.75s |
| 5 | **Random Forest Classifier** | Part B (Review 2) | 0.8479 | 0.8679 | 0.9299 | 0.8428 | 0.9039 | 0.68s |
| 6 | **Bagging (Decision Tree base)** | Part B (Review 2) | 0.8439 | 0.8754 | 0.9128 | 0.8411 | 0.8964 | 6.11s |
| 7 | **Support Vector Machine (SVC)** | Part A (Review 1) | 0.8438 | 0.8656 | 0.9266 | 0.8388 | 0.8954 | 10.40s |
| 8 | **Gaussian Naive Bayes** | Part A (Review 1) | 0.8290 | 0.8830 | 0.8785 | 0.8293 | 0.8840 | 0.31s |
| 9 | **Decision Tree Classifier** | Part A (Review 1) | 0.8289 | 0.8629 | 0.9061 | 0.8252 | 0.8837 | 0.36s |
| 10 | **K-Nearest Neighbors (KNN)** | Part A (Review 1) | 0.8083 | 0.8549 | 0.8832 | 0.8056 | 0.8424 | 0.12s |

### 5.2 Model Selection, Tuning & Cross-Validation
- **Champion Model Selection:** Logistic Regression ($86.09\%$ Accuracy, $0.9220$ ROC-AUC) was chosen as the champion model for its superior combination of predictive power, rapid inference ($0.39$s), and transparent coefficient log-odds interpretability.
- **Hyperparameter Tuning Improvement:**
  - **AdaBoost Classifier:** Tuned via GridSearchCV (`n_estimators=100`, `learning_rate=1.0`). Accuracy improved from **$0.8501$ (Baseline)** to **$0.8544$ (Tuned)**, verifying a real improvement of **$+0.0043$ ($+0.43\%$)**.
  - **Logistic Regression:** Regularization tuning ($C \in [0.01, 10.0]$) confirmed that $C=1.0$ maintains the optimal balance between bias and variance.
- **5-Fold Cross-Validation on Top Two Models:**
  - **Logistic Regression:** Mean CV Accuracy = **$0.8590 \pm 0.0029$**
  - **MLP Classifier:** Mean CV Accuracy = **$0.8551 \pm 0.0024$**  
  Low standard deviation ($\le 0.003$) demonstrates strong generalization across data folds.

---

## 6. Clustering Track (Review 2)

### 6.1 K-Means Cluster-Count Evaluation Across $k=2$ to $k=8$
Evaluated on Forest CoverType dataset ($N=20,000$ fitting sample, $N=3,000$ evaluation sample):

| $k$ | Inertia (WCSS) | $\Delta \text{Inertia}$ (Marginal Drop) | Silhouette Score $\uparrow$ | Davies–Bouldin Index $\downarrow$ | Calinski–Harabasz Index $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **188,946.6** | — | **0.1903 (Peak)** | 1.9933 | **648.4 (Peak)** |
| **3** | 168,843.5 | **20,103.0 (Max Drop)** | 0.1493 | 1.9734 | 542.3 |
| **4** | 155,370.7 | 13,472.8 | 0.1516 | 1.8183 | 482.0 |
| **5** | 143,711.0 | 11,659.8 | 0.1457 | 1.8391 | 449.3 |
| **6** | 133,526.6 | 10,184.3 | 0.1420 | **1.7793 (Best)** | 435.7 |
| **7** | 127,844.2 | 5,682.4 | 0.1330 | 1.7882 | 404.7 |
| **8** | 123,070.1 | 4,774.1 | 0.1344 | 1.7964 | 377.7 |

### 6.2 Analysis of Multi-Metric Trade-offs
- **Global Separation Criterion ($k=2$):** Both Silhouette Score ($0.1903$) and Calinski-Harabasz Index ($648.4$) reach their absolute peaks at $k=2$, capturing the primary ecological division between Roosevelt National Forest's high-elevation subalpine forests and lower-elevation montane woodlands.
- **Marginal Inertia Reduction ($k=3$):** Moving from $k=2 \to 3$ produces the single largest marginal reduction in within-cluster scatter ($\Delta = 20,103.0$).
- **Local Compactness Criterion ($k=6$):** Davies-Bouldin Index reaches its global optimum at $k=6$ ($1.7793$), proving that finer sub-clusters capture localized soil and moisture differences.
- **Selection Decision:** Rather than claiming $k=3$ is universally optimal, we benchmark **both $k=2$ (the metric-maximizing partition) and $k=3$ (the elbow transition partition)** alongside Agglomerative clustering.

### 6.3 Consolidated Clustering Comparative Table ($N=3,000$)

| Algorithm | Configuration | Sample Size | Silhouette Score $\uparrow$ | Davies–Bouldin Index $\downarrow$ | Calinski–Harabasz Index $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **K-Means** | $k=2$ [Global Separation Optimum] | 3,000 | **0.1903** | 1.9933 | **648.4** |
| **K-Means** | $k=3$ [Elbow Transition Optimum] | 3,000 | 0.1493 | 1.9734 | 542.3 |
| **Agglomerative** | $k=2$, Ward Linkage | 3,000 | 0.1733 | 2.1502 | 583.0 |
| **Agglomerative** | $k=2$, Complete Linkage | 3,000 | 0.2248 | 1.6490 | 198.4 |
| **Agglomerative** | $k=2$, Average Linkage | 3,000 | 0.5725* | 0.3079* | 9.7 (Chaining) |
| **Agglomerative** | $k=3$, Ward Linkage | 3,000 | 0.1547 | 1.9043 | 426.0 |

*\*Average Linkage Note:* Average linkage yields an artificially elevated Silhouette score ($0.5725$) by isolating a tiny outlier cluster while pooling over $99\%$ of points into a single group (chaining behavior). Ward linkage provides the most balanced, variance-minimizing hierarchy.

---

## 7. Environment Setup & Execution Instructions

```bash
# 1. Install verified dependencies
pip install -r Review2_Final_Deliverables/requirements.txt

# 2. Run classification notebook
python -m nbconvert --to notebook --execute --inplace Review2_Final_Deliverables/notebooks/classification.ipynb

# 3. Run clustering notebook
python -m nbconvert --to notebook --execute --inplace Review2_Final_Deliverables/notebooks/clustering.ipynb
```

---

## 8. Academic Integrity & Version Control Note
- **No Remote Operations:** All modifications, execution, and audits were executed locally. Zero GitHub commits, pushes, branch creations, or remote repository modifications were performed.
