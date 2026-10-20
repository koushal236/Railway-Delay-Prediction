# Dataset Documentation & Integrity Guide

This directory houses the primary empirical datasets utilized across the Machine Learning Capstone workflow:

---

## 1. Dataset Inventory & Specifications

| Dataset Name | File Size | Dimensions | Target Column | ML Workflow | Storage & Git Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`IR_regression_dataset.csv`** | **22.3 MB** | 100,000 rows × 44 columns | `delay_minutes` (Continuous) | Supervised Regression (`notebooks/regression.ipynb`) | **Standard Git File** (under 100 MB limit, fully included in GitHub ZIP) |
| **`ir_train_siddu.csv`** | **337.2 MB** | 1,500,000 rows × 45 columns | `is_delayed` (Binary: 0 or 1) | Supervised Classification (`notebooks/classification.ipynb`) | **Git LFS Tracked** (exceeds GitHub 100 MB limit) |
| **Forest CoverType Benchmark** | Dynamically fetched | 581,012 rows × 54 features | None (Labels excluded from input) | Unsupervised Clustering (`notebooks/clustering.ipynb`) | **Scikit-Learn API** (`fetch_covtype()`, cached locally) |

---

## 2. Git Large File Storage (Git LFS) vs. GitHub ZIP Download

### For Git Clone Users:
If you clone this repository using Git:
```bash
git clone <repo-url>
cd Railway-Delay-Prediction
git lfs pull
```
`git lfs pull` automatically downloads the full 337 MB `ir_train_siddu.csv`.

### For GitHub "Download ZIP" Users:
GitHub's web interface **does not automatically package large Git LFS binary assets** into standard `.zip` downloads.
- `IR_regression_dataset.csv` (22.3 MB) is committed as a regular Git file, so it is **always present and usable** directly from the ZIP.
- `ir_train_siddu.csv` (337 MB) is tracked by Git LFS. If downloaded via ZIP, it may appear as a 130-byte text pointer stub unless LFS files were bundled.
- **The Streamlit GUI (`GUI_Deployment/`) is 100% self-contained**: It uses pre-serialized models in `GUI_Deployment/models/` and sample presets in `GUI_Deployment/data/`, meaning **the GUI runs completely without requiring the 337 MB training CSV!**

---

## 3. Dataset Verification Tool

To check whether your local datasets are fully downloaded and ready:
```bash
python data/verify_datasets.py
```
This utility scans the file sizes, verifies the schemas, ensures target columns exist, and warns you if any file is an unpulled Git LFS pointer.
