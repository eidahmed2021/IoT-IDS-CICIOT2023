import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64')

import numpy as np
import xgboost as xgb
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import gc

print("=" * 60)
print("XGBoost GPU Test - Full Dataset (Binary Classification)")
print("=" * 60)
print(">>> MEMORY-OPTIMIZED VERSION <<<")

PARQUET_PATH = 'processed/all_data_full.parquet'

# 1. Read parquet metadata
print("\n[1/5] Reading parquet metadata...")
pf = pq.ParquetFile(PARQUET_PATH)
total_rows = pf.metadata.num_rows
print(f"  Total rows: {total_rows:,}")

# 2. Read labels first (tiny memory - just 1 column)
print("\n[2/5] Reading labels and creating train/test indices...")
labels_table = pq.read_table(PARQUET_PATH, columns=['label'])
labels = labels_table.column('label').to_pylist()
del labels_table
gc.collect()

# Create binary labels
y = np.array([0 if l == 'BenignTraffic' else 1 for l in labels], dtype=np.int8)
del labels
gc.collect()

benign_count = (y == 0).sum()
attack_count = (y == 1).sum()
print(f"  Benign: {benign_count:,} | Attack: {attack_count:,}")

# Stratified split using indices only (memory efficient)
np.random.seed(42)
indices = np.arange(len(y))
np.random.shuffle(indices)

split_point = int(len(y) * 0.8)
train_idx = np.sort(indices[:split_point])
test_idx = np.sort(indices[split_point:])
del indices
gc.collect()

y_train = y[train_idx]
y_test = y[test_idx]
del y
gc.collect()
print(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")

# 3. Read features COLUMN-BY-COLUMN directly to float32 (skip pandas float64!)
print("\n[3/5] Loading features (memory-optimized, no pandas)...")

# Get feature column names (all except 'label')
schema = pf.schema_arrow
feature_cols = [f.name for f in schema if f.name != 'label']
n_features = len(feature_cols)
print(f"  Features: {n_features} columns")

t0 = time.time()

# ============================================================
# MEMORY-OPTIMIZED: Read directly from PyArrow to float32
# Old way: table.to_pandas().values.astype(np.float32)
#   -> Creates float64 pandas (~17GB) + float32 numpy (~8.5GB) = 25.5GB PEAK!
# New way: Read column by column directly to float32
#   -> Only float32 numpy (~8.5GB) in memory at any time
# ============================================================

# Pre-allocate train and test arrays directly (never hold full X in memory)
print(f"  Allocating train array: ({len(train_idx):,}, {n_features}) float32")
X_train = np.empty((len(train_idx), n_features), dtype=np.float32)
print(f"  Allocating test array: ({len(test_idx):,}, {n_features}) float32")
X_test = np.empty((len(test_idx), n_features), dtype=np.float32)

train_mem = X_train.nbytes / 1e9
test_mem = X_test.nbytes / 1e9
print(f"  Total memory: {train_mem + test_mem:.2f} GB (train: {train_mem:.2f} + test: {test_mem:.2f})")

# Read column by column and split directly into train/test
# This avoids ever having the full dataset in memory
print(f"  Reading columns: ", end="", flush=True)
for i, col_name in enumerate(feature_cols):
    col_table = pq.read_table(PARQUET_PATH, columns=[col_name])
    col_array = col_table.column(0).to_numpy().astype(np.float32)
    del col_table

    # Handle inf/nan per column
    col_array = np.nan_to_num(col_array, nan=0.0, posinf=1e10, neginf=-1e10)

    # Split directly into train/test (never hold full X)
    X_train[:, i] = col_array[train_idx]
    X_test[:, i] = col_array[test_idx]
    del col_array
    gc.collect()

    # Progress indicator
    if (i + 1) % 10 == 0 or i == n_features - 1:
        print(f"{i+1}/{n_features}", end=" ", flush=True)

print()  # newline
del train_idx, test_idx
gc.collect()

t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s")

# 4. Create DMatrix and train
print("\n[4/5] Creating DMatrix and training on GPU...")
dtrain = xgb.DMatrix(X_train, label=y_train)
del X_train, y_train
gc.collect()

dtest = xgb.DMatrix(X_test, label=y_test)
del X_test
gc.collect()

params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 8,
    'learning_rate': 0.1,
    'verbosity': 1
}

t0 = time.time()
bst = xgb.train(
    params, dtrain,
    num_boost_round=100,
    evals=[(dtest, 'test')],
    verbose_eval=25
)
t_train = time.time() - t0

# 5. Evaluate
print("\n[5/5] Evaluating...")
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')

print(f"\n{'='*60}")
print(f"RESULTS - XGBoost GPU (Binary: Benign vs Attack)")
print(f"{'='*60}")
print(f"  Dataset:    {total_rows:,} rows")
print(f"  Accuracy:   {acc*100:.4f}%")
print(f"  F1-Score:   {f1*100:.4f}%")
print(f"  Load time:  {t_load:.1f}s")
print(f"  Train time: {t_train:.1f}s (100 rounds on GPU)")
print(f"{'='*60}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
