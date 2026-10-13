import gc  # Garbage collection
import numpy as np
from sklearn.model_selection import KFold

# This code first merges the  train subset 80% + the validation subset 20% to have the original PulseDB training set
# Merge ECG train and validation
ecg1 = np.load('pulsedb_data_npy/train_ecg.npy')
ecg2 = np.load('pulsedb_data_npy/val_ecg.npy')
ecg = np.concatenate([ecg1, ecg2], axis=0)
print(ecg.shape)
del ecg1, ecg2
gc.collect()

# merge PPG training and validation
ppg1 = np.load('mydata/train_ppg.npy')
ppg2 = np.load('mydata/val_ppg.npy')
ppg = np.concatenate([ppg1, ppg2], axis=0)
print(ppg.shape)
del ppg1,ppg2
gc.collect()

# merge abp train and validation
abp1 = np.load('mydata/train_abp.npy')
abp2 = np.load('mydata/val_abp.npy')
abp = np.concatenate([abp1, abp2], axis=0)
print(abp.shape)
del abp1,abp2
gc.collect()

# merge label train and validation
y1 = np.load('mydata/train_Labels.npy')
y2 = np.load('mydata/val_Labels.npy')
y = np.concatenate([y1, y2], axis=0)
print(y.shape)
del y1, y2
gc.collect()

# then creates 5 folds from the PulseDb training set
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# split data using their index in train_idx and val_idx
for fold, (train_idx, val_idx) in enumerate(kf.split(ecg)):

    ecg_fold = ecg[val_idx]
    ppg_fold = ppg[val_idx]
    abp_fold = abp[val_idx]
    y_fold = y[val_idx]

    # Save the created folds on disk
    np.save(f'pulsedb_data_npy/fold{fold}_ecg.npy', ecg_fold)
    np.save(f'pulsedb_data_npy/fold{fold}_ppg.npy', ppg_fold)
    np.save(f'pulsedb_data_npy/fold{fold}_abp.npy', abp_fold)
    np.save(f'pulsedb_data_npy/fold{fold}_labels.npy', y_fold)
    del ecg_fold, ppg_fold, abp _fold, y_fold
    gc.collect()
