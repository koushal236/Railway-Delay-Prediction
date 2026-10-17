"""
Verification and integrity checker for Railway Delay Prediction datasets.
Checks whether datasets are real CSV files, LFS pointers, or missing.
"""
import os
import sys
import pandas as pd

def check_file(filename, expected_target=None):
    print(f"\n--- Checking: {filename} ---")
    candidates = [
        os.path.join("data", filename),
        filename,
        os.path.join("..", "data", filename),
        os.path.join("Railway-Delay-Prediction-main", "data", filename)
    ]
    resolved_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_path = os.path.abspath(c)
            break
            
    if not resolved_path:
        print(f"[MISSING] {filename} could not be located in any standard data path.")
        return False
        
    size_bytes = os.path.getsize(resolved_path)
    print(f"  Located at: {resolved_path}")
    print(f"  File size : {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
    
    # Check for Git LFS pointer
    with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
        
    if first_line.startswith("version https://git-lfs.github.com"):
        print("  [WARN] GIT LFS POINTER DETECTED")
        print("         This file is a Git LFS pointer stub (not the actual CSV data).")
        print("         To retrieve the actual data on a Git-cloned repository, run:")
        print("             git lfs pull")
        print("         If you downloaded the repository as a ZIP from GitHub, Git LFS files")
        print("         are not automatically bundled in the ZIP.")
        return False
        
    try:
        df_sample = pd.read_csv(resolved_path, nrows=5)
        print(f"  [OK] Valid CSV. Total columns: {len(df_sample.columns)}")
        print(f"       Sample columns: {df_sample.columns[:6].tolist()}...")
        if expected_target:
            if expected_target in df_sample.columns:
                print(f"  [OK] Target column '{expected_target}' verified present.")
            else:
                print(f"  [ERROR] Target column '{expected_target}' missing from schema.")
                return False
    except Exception as e:
        print(f"  [ERROR] Error reading CSV: {e}")
        return False
        
    print(f"  [OK] {filename} is ready and valid for machine learning workflows.")
    return True

if __name__ == "__main__":
    print("=" * 65)
    print("Railway Delay Prediction - Dataset Integrity Verification")
    print("=" * 65)
    
    reg_ok = check_file("IR_regression_dataset.csv", expected_target="delay_minutes")
    cls_ok = check_file("ir_train_siddu.csv", expected_target="is_delayed")
    
    print("\n" + "=" * 65)
    if reg_ok and cls_ok:
        print("[SUCCESS] All required datasets are present, verified, and ready!")
    else:
        print("[WARNING] Dataset verification completed with warnings above.")
    print("=" * 65)
