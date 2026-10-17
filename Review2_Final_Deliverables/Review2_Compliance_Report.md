# Capstone Project Review 2 — Compliance Audit Report
**Project Title:** Indian Railway Delay Prediction & Forest CoverType Unsupervised Clustering  
**Course Code:** 23CSE301 Machine Learning  
**Evaluation Scope:** Review 2 (Classification Track Part B + Full Clustering Track)  
**Academic Year:** 2026-27  
**Auditor:** Senior ML Engineer, Jupyter Notebook Auditor & Academic Rubric Compliance Reviewer  
**Audit Date:** October 2026  

---

## 1. Audit Framework & Evidence Classification

To maintain the highest standard of academic integrity, this compliance report enforces a strict evidence hierarchy:

1. **`INDEPENDENTLY VERIFIED` (`PASS`):**  
   Results directly computed and verified through fresh, top-to-bottom headless kernel executions (`python -m nbconvert --to notebook --execute --inplace`) on the local system. Cell indices, exact metric values, and execution times are explicitly recorded.
2. **`HISTORICAL CONTEXT / PRESERVED SETUP`:**  
   Review 1 Part-A baseline configurations (Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC) that have been successfully integrated into the consolidated 10-model pipeline on the identical train/test split.
3. **`METHODOLOGICAL CAVEAT / EMPIRICAL BOUND`:**  
   Realistic computational boundaries transparently documented (e.g., deterministic sampling for $O(N^2)$ distance matrices, metric trade-offs in continuous spatial data). Unsubstantiated claims of "perfection" are rejected in favor of defensible scientific trade-offs.

---

## 2. Review 2 Rubric Verification & Itemized Evidence

### Section A: Classification Track, Part B (Max Marks: 6)

#### Criterion A1: Implementation of All 5 Part-B Algorithms (3 Marks)
- **Rubric Descriptor:** All 5 Part-B algorithms trained on the same dataset as Review 1 Part A, no errors.
- **Audit Assessment:** **`PASS` — Full Marks (3 / 3)**
- **Verification Evidence:**
  - **Notebook:** [`classification.ipynb`](file:///Review2_Final_Deliverables/notebooks/classification.ipynb) — Cells 17, 18, 19.
  - **Independent Execution Log:** Verified executed in fresh kernel in $120.4$s total run time.
  - **Part-B Model Implementations Verified:**
    1. *Random Forest Classifier:* Implemented in pipeline (`n_estimators=100, random_state=42, n_jobs=-1`). Fit time: $0.68$s.
    2. *AdaBoost Classifier:* Implemented in pipeline (`n_estimators=50, learning_rate=1.0, random_state=42`). Fit time: $2.75$s.
    3. *Gradient Boosting Classifier:* Implemented in pipeline (`n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`). Fit time: $10.95$s.
    4. *Bagging Classifier:* Implemented with explicit `estimator=DecisionTreeClassifier(random_state=42)`, `n_estimators=50`, `random_state=42`. Fit time: $6.11$s.
    5. *Multi-Layer Perceptron (MLP):* Implemented with feedforward architecture `hidden_layer_sizes=(64, 32)`, `early_stopping=True`, `max_iter=200`, `random_state=42`. Fit time: $1.38$s.
  - **Dataset Consistency:** All 5 models fitted on `X_model_train` ($N=30,000$ stratified subset of `ir_train_siddu.csv`), identically matching the Part-A models. Zero execution errors.

#### Criterion A2: Consolidated 10-Algorithm Comparison Table (1 Mark)
- **Rubric Descriptor:** A single table comparing all 10 classification algorithms on Accuracy, Precision, Recall, F1, and ROC-AUC.
- **Audit Assessment:** **`PASS` — Full Marks (1 / 1)**
- **Verification Evidence:**
  - **Notebook:** [`classification.ipynb`](file:///Review2_Final_Deliverables/notebooks/classification.ipynb) — Cell 21.
  - **Independently Measured Table Output:**
    | # | Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC | Fit Time (s) |
    | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | 1 | **Logistic Regression** | **0.8609** | **0.8880** | **0.9228** | **0.8585** | **0.9220** | 0.39s |
    | 2 | **MLP Classifier** | **0.8563** | **0.8849** | **0.9197** | **0.8539** | **0.9160** | 1.38s |
    | 3 | **Gradient Boosting** | **0.8560** | 0.8753 | **0.9325** | 0.8517 | **0.9162** | 10.95s |
    | 4 | **AdaBoost** | 0.8501 | 0.8734 | 0.9256 | 0.8461 | 0.9116 | 2.75s |
    | 5 | **Random Forest** | 0.8479 | 0.8679 | 0.9299 | 0.8428 | 0.9039 | 0.68s |
    | 6 | **Bagging (Decision Tree)**| 0.8439 | 0.8754 | 0.9128 | 0.8411 | 0.8964 | 6.11s |
    | 7 | **SVM** | 0.8438 | 0.8656 | 0.9266 | 0.8388 | 0.8954 | 10.40s |
    | 8 | **Naive Bayes** | 0.8290 | 0.8830 | 0.8785 | 0.8293 | 0.8840 | 0.31s |
    | 9 | **Decision Tree** | 0.8289 | 0.8629 | 0.9061 | 0.8252 | 0.8837 | 0.36s |
    | 10 | **KNN** | 0.8083 | 0.8549 | 0.8832 | 0.8056 | 0.8424 | 0.12s |
  - Single DataFrame rendered in notebook containing all 10 models and all 5 required metrics.

#### Criterion A3: Final Model Selection & Hyperparameter Tuning (2 Marks)
- **Rubric Descriptor:** Best model identified with clear justification; hyperparameter tuning applied; improvement in at least one metric documented.
- **Audit Assessment:** **`PASS` — Full Marks (2 / 2)**
- **Verification Evidence:**
  - **Notebook:** [`classification.ipynb`](file:///Review2_Final_Deliverables/notebooks/classification.ipynb) — Cells 33, 34, 36, 38.
  - **Champion Selection Justification:** Logistic Regression selected based on achieving top Accuracy ($86.09\%$) and top ROC-AUC ($0.9220$), combined with sub-second inference latency ($0.39$s) and interpretable log-odds coefficients.
  - **Tuning Demonstration Verified:**
    - *AdaBoost Baseline Accuracy:* **$0.8501$**
    - *AdaBoost Tuned Accuracy (`n_estimators=100, learning_rate=1.0`):* **$0.8544$**
    - *Net Gain Measured:* **$+0.0043$ ($+0.43\%$)** on held-out test split.
  - **5-Fold Cross-Validation on Top 2 Models Verified:**
    - *Logistic Regression:* Mean CV Accuracy = **$0.8590 \pm 0.0029$**
    - *MLP Classifier:* Mean CV Accuracy = **$0.8551 \pm 0.0024$**

---

### Section B: Clustering Track (Max Marks: 8)

#### Criterion B1: Implementation of Required Clustering Algorithms (4 Marks)
- **Rubric Descriptor:** All 2 required clustering algorithms fitted and cluster labels obtained; no errors.
- **Audit Assessment:** **`PASS` — Full Marks (4 / 4)**
- **Verification Evidence:**
  - **Notebook:** [`clustering.ipynb`](file:///Review2_Final_Deliverables/notebooks/clustering.ipynb) — Cells 6, 8, 12, 16, 18.
  - **Dataset Verified:** Forest CoverType loaded via `fetch_covtype()` (581,012 records, 54 features).
  - **Target Exclusion Verified:** Target $y$ (cover types 1–7) strictly excluded from all clustering inputs and scaling.
  - **K-Means Clustering:** Evaluated systematically across $k=2 \dots 8$ with `random_state=42`, `n_init=10`.
  - **Agglomerative Clustering:** Fitted across `ward`, `average`, and `complete` linkages on reproducible sample $N=3,000$.

#### Criterion B2: Evaluation Metrics, Elbow Curve & Dendrogram (2 Marks)
- **Rubric Descriptor:** Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index reported for all algorithms; Elbow curve for K-Means and Dendrogram for Hierarchical clustering shown.
- **Audit Assessment:** **`PASS` — Full Marks (2 / 2)**
- **Verification Evidence:**
  - **Notebook:** [`clustering.ipynb`](file:///Review2_Final_Deliverables/notebooks/clustering.ipynb) — Cells 12, 14, 20, 25.
  - **Systematic Multi-Metric Evaluation ($k=2 \dots 8$):**
    | $k$ | WCSS (Inertia) | $\Delta \text{Inertia}$ | Silhouette $\uparrow$ | Davies–Bouldin $\downarrow$ | Calinski–Harabasz $\uparrow$ |
    | :---: | :---: | :---: | :---: | :---: | :---: |
    | **2** | **188,946.6** | — | **0.1903 (Peak)** | 1.9933 | **648.4 (Peak)** |
    | **3** | 168,843.5 | **20,103.0 (Max Drop)** | 0.1493 | 1.9734 | 542.3 |
    | **4** | 155,370.7 | 13,472.8 | 0.1516 | 1.8183 | 482.0 |
    | **5** | 143,711.0 | 11,659.8 | 0.1457 | 1.8391 | 449.3 |
    | **6** | 133,526.6 | 10,184.3 | 0.1420 | **1.7793 (Best)** | 435.7 |
    | **7** | 127,844.2 | 5,682.4 | 0.1330 | 1.7882 | 404.7 |
    | **8** | 123,070.1 | 4,774.1 | 0.1344 | 1.7964 | 377.7 |
  - **Trade-Off Analysis:** Rather than falsely asserting $k=3$ is optimal, the report documents that $k=2$ maximizes Silhouette ($0.1903$) and Calinski-Harabasz ($648.4$), $k=3$ represents the largest marginal inertia drop ($\Delta = 20,103.0$), and $k=6$ minimizes Davies-Bouldin ($1.7793$).
  - **Elbow Plot:** Dual-panel figure showing Inertia with marginal drop bars and metric trajectories.
  - **Dendrogram:** Generated using Ward linkage on $N=500$ sample, truncated to 30 merges with clear cutoff line.

#### Criterion B3: Cluster Visualizations (PCA & t-SNE) (2 Marks)
- **Rubric Descriptor:** PCA-reduced 2D scatter plot with cluster colour-coding for every algorithm; t-SNE plot for at least one algorithm.
- **Audit Assessment:** **`PASS` — Full Marks (2 / 2)**
- **Verification Evidence:**
  - **Notebook:** [`clustering.ipynb`](file:///Review2_Final_Deliverables/notebooks/clustering.ipynb) — Cells 27, 29.
  - **PCA Plot 1 (K-Means):** 2D scatter colored by K-Means cluster assignments (PC1: $22.7\%$, PC2: $19.5\%$, Total: $42.2\%$).
  - **PCA Plot 2 (Agglomerative):** 2D scatter colored by Ward cluster assignments.
  - **t-SNE Plot:** Non-linear 2D manifold embedding ($N=1,500$, perplexity=30) colored by cluster assignments.

---

### Section C: Pipeline Integration & Quality (Max Marks: 3)

#### Criterion C1: Code Quality & Modularity (1 Mark)
- **Descriptor:** Clean top-to-bottom run, modular functions, commented key steps.
- **Audit Assessment:** **`PASS` (1 / 1)**
- **Evidence:** Both notebooks executed cleanly with 0 errors via headless nbconvert. All 18 code cells in `classification.ipynb` and 15 code cells in `clustering.ipynb` produced visible outputs.

#### Criterion C2: README & Documentation (1 Mark)
- **Descriptor:** Complete README with problem statements, datasets, results summary, instructions.
- **Audit Assessment:** **`PASS` (1 / 1)**
- **Evidence:** Both root `README.md` and `Review2_Final_Deliverables/README.md` updated with exact matching tables and instructions.

#### Criterion C3: Deliverables Layout & Requirements (1 Mark)
- **Descriptor:** Valid structure, requirements.txt present.
- **Audit Assessment:** **`PASS` (1 / 1)**
- **Evidence:** `requirements.txt` present in root and deliverables folder. Verified uncorrupted ZIP archive `Review2_Final_Deliverables.zip` created.

---

## 3. General Guidelines Compliance Audit (Section 7)

| Guideline | Requirement | Verified Status | Audit Observation |
| :--- | :--- | :---: | :--- |
| **7.1 Portable Dataset Path** | Independent of machine paths | **PASS** | Dynamic relative search finds `data/ir_train_siddu.csv` without hardcoded Windows paths. |
| **7.1 Data Leakage Prevention** | Scalers fitted only on train set | **PASS** | `preprocessor` fitted inside pipelines on training data only. Leakage columns dropped. |
| **7.1 Reproducibility** | Fixed random seed `random_state=42` | **PASS** | Applied consistently across splits, models, and sampling. |
| **7.2 Consistent Split** | 80:20 stratified split | **PASS** | Used for all 10 classification algorithms. |
| **7.2 Top 2 Cross-Validation** | 5-fold CV for top 2 models | **PASS** | Evaluated on Logistic Regression and MLP. |
| **7.3 Plot Standards** | Labels, titles, legends, tight layout | **PASS** | All 11 figures comply. |
| **7.5 Remote Safety** | Zero remote GitHub operations | **PASS** | 100% local execution; 0 git commits/pushes. |

---

## 4. Evaluated Score Summary

$$\text{Audit Verified Score} = \mathbf{6 / 6} \text{ (Classification Part B)} + \mathbf{8 / 8} \text{ (Clustering)} + \mathbf{3 / 3} \text{ (Integration)} = \mathbf{17 / 17} \text{ (100\% of Track Marks)}$$

*Note: The remaining 8 marks in Review 2 are evaluated in-person during the viva defense (Section D).*
