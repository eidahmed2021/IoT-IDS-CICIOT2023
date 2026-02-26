import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64')

import numpy as np
import xgboost as xgb
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import gc
import json
from datetime import datetime

print("=" * 70)
print("  XGBoost GPU Training - Full CIC-IoT-2023 Dataset")
print("  Binary Classification: Benign vs Attack")
print("=" * 70)
print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Memory-Optimized Pipeline (column-by-column loading)")
print("=" * 70)

# ============================================================
# Configuration
# ============================================================
PARQUET_PATH = 'processed/all_data_full.parquet'
MODEL_SAVE_PATH = 'models/xgb_full_model.json'
RESULTS_SAVE_PATH = 'models/training_results.json'
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# XGBoost hyperparameters
XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],
    'max_depth': 8,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1,  # will be calculated from data
    'verbosity': 1,
    'seed': RANDOM_SEED
}
NUM_BOOST_ROUNDS = 300
EARLY_STOPPING_ROUNDS = 20

# Create output directory
os.makedirs('models', exist_ok=True)

# ============================================================
# Step 1: Read metadata
# ============================================================
print("\n" + "=" * 70)
print("[Step 1/6] Reading parquet metadata...")
print("=" * 70)

t_total_start = time.time()

pf = pq.ParquetFile(PARQUET_PATH)
total_rows = pf.metadata.num_rows
num_row_groups = pf.metadata.num_row_groups
schema = pf.schema_arrow
feature_cols = [f.name for f in schema if f.name != 'label']
n_features = len(feature_cols)

print(f"  File:          {PARQUET_PATH}")
print(f"  File size:     {os.path.getsize(PARQUET_PATH) / 1e9:.2f} GB")
print(f"  Total rows:    {total_rows:,}")
print(f"  Row groups:    {num_row_groups}")
print(f"  Features:      {n_features} columns")

# ============================================================
# Step 2: Read labels and create train/test split
# ============================================================
print("\n" + "=" * 70)
print("[Step 2/6] Reading labels and creating train/test split...")
print("=" * 70)

t0 = time.time()
labels_table = pq.read_table(PARQUET_PATH, columns=['label'])
labels = labels_table.column('label').to_pylist()
del labels_table
gc.collect()

# Create binary labels
y = np.array([0 if l == 'BenignTraffic' else 1 for l in labels], dtype=np.int8)
del labels
gc.collect()

benign_count = int((y == 0).sum())
attack_count = int((y == 1).sum())
print(f"  Benign:  {benign_count:,} ({benign_count/len(y)*100:.2f}%)")
print(f"  Attack:  {attack_count:,} ({attack_count/len(y)*100:.2f}%)")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = benign_count / attack_count if attack_count > 0 else 1.0
XGB_PARAMS['scale_pos_weight'] = round(scale_pos_weight, 4)
print(f"  Scale pos weight: {XGB_PARAMS['scale_pos_weight']:.4f}")

# Stratified-like random split using indices
np.random.seed(RANDOM_SEED)
indices = np.arange(len(y))
np.random.shuffle(indices)

split_point = int(len(y) * TRAIN_RATIO)
train_idx = np.sort(indices[:split_point])
test_idx = np.sort(indices[split_point:])
del indices
gc.collect()

y_train = y[train_idx]
y_test = y[test_idx]
del y
gc.collect()

print(f"  Train:   {len(train_idx):,} samples ({TRAIN_RATIO*100:.0f}%)")
print(f"  Test:    {len(test_idx):,} samples ({(1-TRAIN_RATIO)*100:.0f}%)")
print(f"  Labels read in {time.time() - t0:.1f}s")

# ============================================================
# Step 3: Load features (memory-optimized, column-by-column)
# ============================================================
print("\n" + "=" * 70)
print("[Step 3/6] Loading features (memory-optimized)...")
print("=" * 70)

t0 = time.time()

# Pre-allocate train and test arrays directly
train_mem_gb = len(train_idx) * n_features * 4 / 1e9
test_mem_gb = len(test_idx) * n_features * 4 / 1e9
print(f"  Allocating train: ({len(train_idx):,} x {n_features}) float32 = {train_mem_gb:.2f} GB")
print(f"  Allocating test:  ({len(test_idx):,} x {n_features}) float32 = {test_mem_gb:.2f} GB")
print(f"  Total memory needed: {train_mem_gb + test_mem_gb:.2f} GB")

X_train = np.empty((len(train_idx), n_features), dtype=np.float32)
X_test = np.empty((len(test_idx), n_features), dtype=np.float32)

# Read column by column - avoids holding full X in memory
print(f"\n  Loading columns: ", end="", flush=True)
for i, col_name in enumerate(feature_cols):
    col_table = pq.read_table(PARQUET_PATH, columns=[col_name])
    col_array = col_table.column(0).to_numpy().astype(np.float32)
    del col_table

    # Handle inf/nan
    col_array = np.nan_to_num(col_array, nan=0.0, posinf=1e10, neginf=-1e10)

    # Split directly into train/test
    X_train[:, i] = col_array[train_idx]
    X_test[:, i] = col_array[test_idx]
    del col_array
    gc.collect()

    # Progress
    if (i + 1) % 5 == 0 or i == n_features - 1:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_features - i - 1)
        print(f"\r  Loading columns: {i+1}/{n_features} ({(i+1)/n_features*100:.0f}%) | "
              f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s   ", end="", flush=True)

print()  # newline
del train_idx, test_idx
gc.collect()

t_load = time.time() - t0
print(f"  Features loaded in {t_load:.1f}s")

# ============================================================
# Step 4: Create DMatrix and train XGBoost
# ============================================================
print("\n" + "=" * 70)
print("[Step 4/6] Creating DMatrix and training with XGBoost GPU...")
print("=" * 70)

print(f"\n  XGBoost Parameters:")
for k, v in XGB_PARAMS.items():
    print(f"    {k}: {v}")
print(f"  Num boost rounds: {NUM_BOOST_ROUNDS}")
print(f"  Early stopping:   {EARLY_STOPPING_ROUNDS} rounds")

print(f"\n  Creating DMatrix (train)...")
t0 = time.time()
dtrain = xgb.DMatrix(X_train, label=y_train)
del X_train, y_train
gc.collect()
print(f"  DMatrix train created in {time.time() - t0:.1f}s")

print(f"  Creating DMatrix (test)...")
t0_dtest = time.time()
dtest = xgb.DMatrix(X_test, label=y_test)
del X_test
gc.collect()
print(f"  DMatrix test created in {time.time() - t0_dtest:.1f}s")

print(f"\n  {'='*50}")
print(f"  Starting GPU Training...")
print(f"  {'='*50}\n")

t_train_start = time.time()
bst = xgb.train(
    XGB_PARAMS,
    dtrain,
    num_boost_round=NUM_BOOST_ROUNDS,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=10
)
t_train = time.time() - t_train_start

best_iteration = bst.best_iteration
best_score = bst.best_score
actual_rounds = best_iteration + 1

print(f"\n  Training completed!")
print(f"  Best iteration: {best_iteration}")
print(f"  Best test score: {best_score:.6f}")
print(f"  Training time: {t_train:.1f}s ({t_train/60:.1f} min)")

# Free train DMatrix
del dtrain
gc.collect()

# ============================================================
# Step 5: Evaluate the model
# ============================================================
print("\n" + "=" * 70)
print("[Step 5/6] Evaluating model...")
print("=" * 70)

y_pred_prob = bst.predict(dtest, iteration_range=(0, best_iteration + 1))
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1_binary = f1_score(y_test, y_pred, average='binary')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

report = classification_report(y_test, y_pred, target_names=['Benign', 'Attack'])

print(f"\n  {'='*50}")
print(f"  RESULTS - XGBoost GPU (Binary Classification)")
print(f"  {'='*50}")
print(f"  Dataset:        {total_rows:,} rows x {n_features} features")
print(f"  Test samples:   {len(y_test):,}")
print(f"  Accuracy:       {acc*100:.4f}%")
print(f"  F1-Score (bin): {f1_binary*100:.4f}%")
print(f"  F1-Score (wtd): {f1_weighted*100:.4f}%")
print(f"  F1-Score (mac): {f1_macro*100:.4f}%")
print(f"  Precision:      {precision*100:.4f}%")
print(f"  Recall:         {recall*100:.4f}%")
print(f"  Specificity:    {specificity*100:.4f}%")
print(f"  FPR:            {fpr*100:.4f}%")
print(f"  {'='*50}")

print(f"\n  Confusion Matrix:")
print(f"                  Predicted")
print(f"                  Benign    Attack")
print(f"  Actual Benign   {tn:>10,}  {fp:>10,}")
print(f"  Actual Attack   {fn:>10,}  {tp:>10,}")

print(f"\n  Classification Report:")
print(report)

# Feature importance (top 20)
print(f"\n  Top 20 Most Important Features:")
importance = bst.get_score(importance_type='weight')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
for rank, (fname, score) in enumerate(sorted_imp, 1):
    # Map feature index to name
    if fname.startswith('f'):
        try:
            fidx = int(fname[1:])
            fname_actual = feature_cols[fidx] if fidx < len(feature_cols) else fname
        except ValueError:
            fname_actual = fname
    else:
        fname_actual = fname
    print(f"    {rank:2d}. {fname_actual:<30s} = {score:.0f}")

# ============================================================
# Step 6: Save model and results
# ============================================================
print("\n" + "=" * 70)
print("[Step 6/6] Saving model and results...")
print("=" * 70)

# Save model
bst.save_model(MODEL_SAVE_PATH)
model_size = os.path.getsize(MODEL_SAVE_PATH) / 1e6
print(f"  Model saved to:   {MODEL_SAVE_PATH} ({model_size:.1f} MB)")

# Save results as JSON
t_total = time.time() - t_total_start
results = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'path': PARQUET_PATH,
        'total_rows': total_rows,
        'n_features': n_features,
        'feature_names': feature_cols,
        'benign_count': benign_count,
        'attack_count': attack_count,
        'train_size': int(split_point),
        'test_size': int(total_rows - split_point)
    },
    'model': {
        'type': 'XGBoost',
        'params': {k: str(v) if isinstance(v, list) else v for k, v in XGB_PARAMS.items()},
        'num_boost_rounds_requested': NUM_BOOST_ROUNDS,
        'actual_rounds_trained': actual_rounds,
        'best_iteration': best_iteration,
        'best_score': float(best_score),
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'model_path': MODEL_SAVE_PATH
    },
    'metrics': {
        'accuracy': float(acc),
        'f1_binary': float(f1_binary),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'fpr': float(fpr),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    },
    'timing': {
        'data_loading_seconds': round(t_load, 1),
        'training_seconds': round(t_train, 1),
        'total_seconds': round(t_total, 1)
    },
    'feature_importance_top20': [
        {'feature': feature_cols[int(f[1:])] if f.startswith('f') and int(f[1:]) < len(feature_cols) else f,
         'importance': float(s)}
        for f, s in sorted_imp
    ]
}

with open(RESULTS_SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  Results saved to: {RESULTS_SAVE_PATH}")

# Final summary
print(f"\n{'='*70}")
print(f"  TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"  Total time:     {t_total:.1f}s ({t_total/60:.1f} min)")
print(f"  Data loading:   {t_load:.1f}s")
print(f"  GPU Training:   {t_train:.1f}s ({actual_rounds} rounds)")
print(f"  Accuracy:       {acc*100:.4f}%")
print(f"  F1-Score:       {f1_binary*100:.4f}%")
print(f"  Model:          {MODEL_SAVE_PATH}")
print(f"{'='*70}")
