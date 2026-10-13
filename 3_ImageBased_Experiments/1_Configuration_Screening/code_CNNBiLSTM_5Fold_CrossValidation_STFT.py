# The purpose of this initial experiment was to implement five-fold
# cross-validation using STFT-based representations and to evaluate
# the experimental workflow, execution time, and memory requirements.

# Due to out-of-memory (OOM) errors, the five folds were processed separately.
# This script was executed five times, with a different FOLD_ID for each run.
import os
fold_id = int(os.environ.get('FOLD_ID', 0))

import numpy as np
import gc  # Garbage collection
from scipy.signal import spectrogram
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# min-max normalization for LABELS
def normalize_labels(y, sbp_min, sbp_max, dbp_min, dbp_max):
    y_scaled = y.copy()
    y_scaled[:, 0] = (y_scaled[:, 0] - sbp_min) / (sbp_max - sbp_min)
    y_scaled[:, 1] = (y_scaled[:, 1] - dbp_min) / (dbp_max - dbp_min)
    return y_scaled

# denormalize labels
def denormalize_labels(y_scaled, sbp_min, sbp_max, dbp_min, dbp_max):
    y = y_scaled.copy()
    y[:, 0] = y[:, 0] * (sbp_max - sbp_min) + sbp_min
    y[:, 1] = y[:, 1] * (dbp_max - dbp_min) + dbp_min
    return y

def signal_to_spectrogram(X,nperseg,noverlap):
    raw_specs = []
    f_list = []
    t_list = []
    spec_list = []

    for x in X:
        f, t, Sxx = spectrogram(x, fs=125, nperseg=nperseg, noverlap=noverlap)
        raw_specs.append(Sxx)
        f_list.append(f)
        t_list.append(t)

    # Global normalization across all samples and all pixels
    spec_array = np.array(raw_specs)
    f_array = np.array(f_list)
    t_array = np.array(t_list)
    return f_array, t_array, spec_array

def normalize_spectrogram (spec_array, spect_min,spect_max):
    spec_normalized_array = spec_array.copy()
    # To avoid division by zero
    spec_normalized_array = (spec_array - spect_min) / (spect_max - spect_min + 1e-8)
    spec_normalized_array= np.expand_dims(spec_normalized_array, axis=-1)  # shape: (batch, H, W, 1)
    return spec_normalized_array

# Define Model
from keras.saving import register_keras_serializable
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

@register_keras_serializable(package="Custom")
def CNN_BiLSTM_Spect_Model_report5(ecg_img_shape, ppg_img_shape):

    # Inputs
    ecg_input = Input(shape=ecg_img_shape, name='ecg_input')
    ppg_input = Input(shape=ppg_img_shape, name='ppg_input')

    # ECG CNN
    x1 = layers.Conv2D(64,  (2,2), activation='relu')(ecg_input)
    x1 = layers.Conv2D(64,  (3,3), activation='relu')(x1)
    x1 = layers.Conv2D(128, (4,4), activation='relu')(x1)
    x1 = layers.Conv2D(128, (5,5), activation='relu')(x1)
    x1 = layers.Conv2D(128, (2,2), activation='relu')(x1)

    # PPG CNN
    x2 = layers.Conv2D(64,  (2,2), activation='relu')(ppg_input)
    x2 = layers.Conv2D(64,  (3,3), activation='relu')(x2)
    x2 = layers.Conv2D(128, (4,4), activation='relu')(x2)
    x2 = layers.Conv2D(128, (5,5), activation='relu')(x2)
    x2 = layers.Conv2D(128, (2,2), activation='relu')(x2)

    # Merge features
    merged = layers.Concatenate(axis=-1)([x1, x2])

    # Reshape for LSTM
    shape = tf.keras.backend.int_shape(merged)  # (None, H, W, C)
    x = layers.Reshape((shape[1]*shape[2], shape[3]))(merged)

    # 8 stacked BiLSTM
    for _ in range(7):
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)

    # Fully Connected layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(2, activation=None)(x)

    model = Model(inputs=[ecg_input, ppg_input], outputs=output, name='CNN_BiLSTM_Spect_Model')
    return model

# Specify the folder containing data
myData='pulsedb_data_npy/'

# Prepare train and validation folds
train_ecg_list = []
train_ppg_list = []
train_y_list = []

# Concatenate 4 training folds and specify validation fold
for i in range(5):
    # i determines the validation fold id
    if i == fold_id:
        ecg_val = np.load(myData + f'fold{i}_ecg.npy')
        ppg_val = np.load(myData + f'fold{i}_ppg.npy')
        y_val = np.load(myData + f'fold{i}_labels.npy')
    else:
        train_ecg_list.append(np.load(myData + f'fold{i}_ecg.npy'))
        train_ppg_list.append(np.load(myData + f'fold{i}_ppg.npy'))
        train_y_list.append(np.load(myData + f'fold{i}_labels.npy'))

# convert train list to array
ecg_train = np.concatenate(train_ecg_list, axis=0)
ppg_train = np.concatenate(train_ppg_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)

# min-max normalization of labels---------------------------------------------------

# find min max of labels
sbp_max = np.max(y_train[:, 0])
dbp_max = np.max(y_train[:, 1])
sbp_min = np.min(y_train[:, 0])
dbp_min = np.min(y_train[:, 1])
print('sbp_max:', sbp_max)
print('dbp_max:', dbp_max)
print('sbp_min:', sbp_min)
print('dbp_min:', dbp_min)

# normalization
y_train_scaled = normalize_labels(y_train  , sbp_min, sbp_max, dbp_min, dbp_max)
y_val_scaled   = normalize_labels(y_val    , sbp_min, sbp_max, dbp_min, dbp_max)



# # STFT transformation parameters investigated during the initial experiments.
nperseg_list = [32, 64, 104, 125, 156, 208, 312, 625] #  1250/nperseg  => 1/12  1/10  1/8  1/6  1/4  1/2
overlap_percentages = [0.05, 0.10, 0.15, 0.25, 0.5, 0.75]

signal_length = 1250

results = []

# Five-Fold Cross-Validation on the 'fold_id' as validation and the rest of folds as training
for nperseg in nperseg_list:
    for percent in overlap_percentages:
        noverlap = int(nperseg * percent)
        if noverlap >= nperseg:
            continue

        # 1. Transform signals to spectrogram (training and validation)
        #    we do not plot the spectrogram from validation, so we do not need to keep frequency and time arrays
        # 2. Find the min and max across training
        # 3. Normalize ECG spectrograms

        # ECG
        f_ecg_arr, t_ecg_arr, ecg_rawspect_train = signal_to_spectrogram (ecg_train, nperseg, noverlap)
        _, _                , ecg_rawspect_val   = signal_to_spectrogram (ecg_val  , nperseg, noverlap)
        spect_min = np.min(ecg_rawspect_train)
        spect_max = np.max(ecg_rawspect_train)
        ecg_normspect_train = normalize_spectrogram (ecg_rawspect_train, spect_min, spect_max)
        ecg_normspect_val   = normalize_spectrogram (ecg_rawspect_val  , spect_min, spect_max)

        # PPG
        f_ppg_arr, t_ppg_arr, ppg_rawspect_train = signal_to_spectrogram (ppg_train, nperseg, noverlap)
        _, _                , ppg_rawspect_val   = signal_to_spectrogram (ppg_val  , nperseg, noverlap)
        spect_min = np.min(ppg_rawspect_train)
        spect_max = np.max(ppg_rawspect_train)
        ppg_normspect_train = normalize_spectrogram (ppg_rawspect_train, spect_min, spect_max)
        ppg_normspect_val   = normalize_spectrogram (ppg_rawspect_val  , spect_min, spect_max)

        # Plot examples of ECG and PPG spectrograms    
        idx= 800
        t = t_ecg_arr[idx]
        f = f_ecg_arr[idx]
        # Plot ECG Normalized spectrogram
        ecg_spect = ecg_normspect_train[idx].squeeze() # squeez here converts from H,W,1 to H,W
        print(f'ECG Normalized Spectrogram shape (idx={idx}) (nperseg={nperseg},noverlap={noverlap}): {ecg_spect.shape}')
        height, width = ecg_spect.shape
        aspect_ratio = width / height
        base_height = 4

        plt.figure(figsize=(base_height * aspect_ratio, base_height))
        plt.imshow(ecg_spect, aspect='auto', origin='lower', cmap='jet', extent=[t[0], t[-1], f[0], f[-1]])
        plt.colorbar()
        plt.title(f'ECG Normalized Spectrogram\nnperseg={nperseg}, noverlap={noverlap}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(f'results/example{idx}_ecg_norm_nps{nperseg}_nov{noverlap}.png', dpi=300)
        plt.close()

        # Plot PPG Normalized spectrogram
        ppg_spect = ppg_normspect_train[idx].squeeze() # squeez here converts from H,W,1 to H,W
        print(f'PPG Normalized Spectrogram shape (idx={idx}) (nperseg={nperseg},noverlap={noverlap}): {ppg_spect.shape}')
        height, width = ppg_spect.shape
        aspect_ratio = width / height
        base_height = 4

        t = t_ppg_arr[idx]
        f = f_ppg_arr[idx]
        plt.figure(figsize=(base_height * aspect_ratio, base_height))
        plt.imshow(ppg_spect, aspect='auto', origin='lower', cmap='jet', extent=[t[0], t[-1], f[0], f[-1]])
        plt.colorbar()
        plt.title(f'PPG Normalized Spectrogram\nnperseg={nperseg}, noverlap={noverlap}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(f'results/example{idx}_ppg_norm_nps{nperseg}_nov{noverlap}.png', dpi=300)
        plt.close()

        # Define input shape for different sizes of spectrogram images, to give to the same model architecture
        # height => spec_array.shape[1]  
        # width => spec_array.shape[2]   
        # input_shape => (height, width, 1)
        input_shape = ecg_normspect_train.shape[1:]  # (H, W, 1)
        print(f"Testing shape: {input_shape}")

        with tf.device(device):
            model = CNN_BiLSTM_Spect_Model_report5 (input_shape, input_shape)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error', metrics=['mae', 'mse'])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss',
                  patience=10,
                  restore_best_weights=True
           )
        # start training on the specific folds and fold i as validation   
        history = model.fit((ecg_normspect_train, ppg_normspect_train), y_train_scaled,
                    validation_data=((ecg_normspect_val, ppg_normspect_val), y_val_scaled),
                    epochs=50, batch_size=128, verbose=0, callbacks=[early_stopping])
        # Save model after training
        model.save(f'results/CNN_LSTM_fold{fold_id}_nps{nperseg}_nov{noverlap}.keras')

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = range(1, len(train_loss) + 1)

        y_pred_scaled = model.predict((ecg_normspect_val, ppg_normspect_val))
        y_pred = denormalize_labels(y_pred_scaled, sbp_min, sbp_max, dbp_min, dbp_max)

        # ME (Mean Error)
        me_sbp = np.mean(y_pred[:, 0] - y_val[:, 0])
        me_dbp = np.mean(y_pred[:, 1] - y_val[:, 1])

        # MSE
        mse_sbp = mean_squared_error(y_val[:, 0], y_pred[:, 0])
        mse_dbp = mean_squared_error(y_val[:, 1], y_pred[:, 1])

        # MAE
        mae_sbp = mean_absolute_error(y_val[:, 0], y_pred[:, 0])
        mae_dbp = mean_absolute_error(y_val[:, 1], y_pred[:, 1])

        # STD
        std_sbp = np.std(y_val[:, 0] - y_pred[:, 0])
        std_dbp = np.std(y_val[:, 1] - y_pred[:, 1])
        print(f"NPERSEG:{nperseg}, NOVERLAP:{noverlap}")
        print(f"fold{fold_id} - SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, STD={std_sbp:.2f}")
        print(f"fold{fold_id} - DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, STD={std_dbp:.2f}")

        plt.figure()
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Fold {fold_id} - nperseg={nperseg}, noverlap={noverlap}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/loss_fold{fold_id}_nps{nperseg}_nov{noverlap}.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(epochs, train_mae, label='Training MAE')
        plt.plot(epochs, val_mae, label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title(f'MAE Fold {fold_id} - nperseg={nperseg}, noverlap={noverlap}')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/'+ f'mae_fold{fold_id}_nps{nperseg}_nov{noverlap}.png', dpi=300, bbox_inches='tight')
        plt.close()

        del model, train_loss, val_loss, train_mae, val_mae, y_pred_scaled, y_pred, history
        del ecg_rawspect_train, ecg_rawspect_val, ecg_normspect_train, ecg_normspect_val
        del ppg_rawspect_train, ppg_rawspect_val, ppg_normspect_train, ppg_normspect_val
        tf.keras.backend.clear_session()
        gc.collect()
