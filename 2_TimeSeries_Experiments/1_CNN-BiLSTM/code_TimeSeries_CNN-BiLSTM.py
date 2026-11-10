import numpy as np
import gc  # Garbage collection
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

print('Data preparation started')

# Load training and validation data
myData='pulsedb_data_npy/'

# ECG training
ecg_train = np.load(myData + 'train_ecg.npy')
ecg_train = ecg_train.astype(np.float32)

# PPG training
ppg_train = np.load(myData + 'train_ppg.npy')
ppg_train = ppg_train.astype(np.float32)

# SBP and DBP training
y_train = np.load(myData + 'train_Labels.npy')

# ECG validation
ecg_val = np.load(myData + 'val_ecg.npy')
ecg_val = ecg_val.astype(np.float32)

# PPG validation
ppg_val= np.load(myData + 'val_ppg.npy')
ppg_val = ppg_val.astype(np.float32)

# SBP and DBP validation
y_val = np.load(myData + 'val_Labels.npy')

# print training shape
print(f'ecg train: {ecg_train.shape}')
print(f'y_train: {y_train.shape}')

# print validation shape
print(f'ecg val: {ecg_val.shape}')
print(f'y_val: {y_val.shape}')

# ECG is already normalised in PulseDB
ecg_train_scaled = ecg_train
ecg_val_scaled = ecg_val

# PPG is already normalised in PulseDB
ppg_train_scaled = ppg_train
ppg_val_scaled = ppg_val

# Free Memory
del ecg_train, ecg_val 
del ppg_train, ppg_val 
gc.collect()

# Function for Min-Max Normalisation of Labels
def min_max_scale(y, sbp_min, sbp_max, dbp_min, dbp_max):
    y_scaled = y.copy()
    y_scaled[:, 0] = (y_scaled[:, 0] - sbp_min) / (sbp_max - sbp_min)
    y_scaled[:, 1] = (y_scaled[:, 1] - dbp_min) / (dbp_max - dbp_min)
    return y_scaled

# Function to denormalise labels
def denormalize_labels(y_scaled, sbp_min, sbp_max, dbp_min, dbp_max):
    y = y_scaled.copy()
    y[:, 0] = y[:, 0] * (sbp_max - sbp_min) + sbp_min
    y[:, 1] = y[:, 1] * (dbp_max - dbp_min) + dbp_min
    return y

# get statistical parameters from training set to apply for label normalisation
sbp_max = np.max(y_train[:, 0])
dbp_max = np.max(y_train[:, 1])
sbp_min = np.min(y_train[:, 0])
dbp_min = np.min(y_train[:, 1])

# print them to be sure their values are correct
print('sbp_max:', sbp_max)
print('dbp_max:', dbp_max)
print('sbp_min:', sbp_min)
print('dbp_min:', dbp_min)

# Apply normalisation on training and validation sets
y_train_scaled         = min_max_scale(y_train , sbp_min, sbp_max, dbp_min, dbp_max)
y_val_scaled           = min_max_scale(y_val   , sbp_min, sbp_max, dbp_min, dbp_max)

del y_train, y_val
gc.collect()

now = datetime.now()
print('Model definition started:',now)

# Define Model
from keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable(package="Custom")
class CNN_BiLSTM_Raw_Model(tf.keras.Model):
    def __init__(self, ecg_raw_shape, ppg_raw_shape):
        super(CNN_BiLSTM_Raw_Model, self).__init__()
        
        self.ecg_raw_shape = ecg_raw_shape
        self.ppg_raw_shape = ppg_raw_shape

        # CNN for ECG Signal
        self.cnn_ecg_raw = tf.keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=ecg_raw_shape),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(2),

            layers.Conv1D(256, 5, activation='relu'),
            layers.Conv1D(256, 7, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 7, activation='relu'),
            layers.Conv1D(128, 5, activation='relu'),
            layers.Conv1D(128, 2, activation='relu')

        ])

        # CNN for PPG Signal
        self.cnn_ppg_raw = tf.keras.Sequential([
            layers.Conv1D(64, 5, activation='relu', input_shape=ppg_raw_shape),
            layers.Conv1D(128, 7, activation='relu'),
            layers.MaxPooling1D(2),

            layers.Conv1D(256, 9, activation='relu'),
            layers.Conv1D(256, 11, activation='relu'),
            layers.MaxPooling1D(2),

            layers.Conv1D(128, 7, strides=1, activation='relu')
        ])

        # LSTM layers for signals (stacked)
        self.bilstm_raw = tf.keras.Sequential([
	    layers.Bidirectional(layers.LSTM(128, return_sequences=True, recurrent_initializer='glorot_uniform', input_shape=(None, 256))), 
	    layers.Bidirectional(layers.LSTM(128, return_sequences=True, recurrent_initializer='glorot_uniform')),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, recurrent_initializer='glorot_uniform')),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, recurrent_initializer='glorot_uniform')),
            layers.Bidirectional(layers.LSTM(128, return_sequences=False, recurrent_initializer='glorot_uniform')),
        ])

        # Fully connected layers
        self.fc = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(2, activation=None)
        ])

    # How data flows through the layers in the defined architecture
    def call(self, inputs):
        ecg_raw, ppg_raw = inputs

        #  Signals
        ecg_raw_cnn_out = self.cnn_ecg_raw(ecg_raw)
        ppg_raw_cnn_out = self.cnn_ppg_raw(ppg_raw)
        
        #concatenate along the last axis (features):
        raw_cnn_features = tf.concat([ecg_raw_cnn_out, ppg_raw_cnn_out], axis=-1)

        # Take the last time step from the BiLSTM output
        last_lstm_out = self.bilstm_raw(raw_cnn_features)

        # Final fully connected layers
        final_output = self.fc(last_lstm_out)
        return final_output

    # functions to be able to load the model after saving it
    def get_config(self):
        config = super(CNN_BiLSTM_Raw_Model, self).get_config()
        config.update({
            'ecg_raw_shape': self.ecg_raw_shape,
            'ppg_raw_shape': self.ppg_raw_shape,
        })
        return config

    @classmethod
    def from_config(cls, config):
        ecg_raw_shape = config.pop('ecg_raw_shape')
        ppg_raw_shape = config.pop('ppg_raw_shape')
        return cls(ecg_raw_shape=ecg_raw_shape, ppg_raw_shape=ppg_raw_shape, **config)

# Example input shapes
ecg_raw_shape = (1250,1)
ppg_raw_shape = (1250,1)

# Instantiate the model
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(device):
    model = CNN_BiLSTM_Raw_Model(ecg_raw_shape, ppg_raw_shape)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error', metrics=['mae', 'mse'])

# print model summary
model.summary()

now = datetime.now()
print('Model train started:',now)

# Train the model 
batch_size = 256

# make ECG and PPG ready as inputs for CNNs
X_ecg = tf.expand_dims(ecg_train_scaled, axis=-1)
X_ppg = tf.expand_dims(ppg_train_scaled, axis=-1)
print('X_ecg:', X_ecg.shape)
print('X_ppg:',X_ppg.shape)

# make ECG and PPG ready for validation set
X_val_ecg = tf.expand_dims(ecg_val_scaled, axis=-1)
X_val_ppg = tf.expand_dims(ppg_val_scaled, axis=-1)

# Combine ECG and PPG inputs into a tuple
dataset = tf.data.Dataset.from_tensor_slices(((X_ecg,X_ppg), y_train_scaled))

# Shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)

# dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model_cnn_mychange.keras', save_best_only=True, monitor='val_loss')

# Prepare validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_ecg,X_val_ppg), y_val_scaled)).batch(batch_size)

# Train the model
history = model.fit(
    dataset,  # Training dataset
    validation_data=val_dataset,  # Validation dataset
    epochs=200,
    callbacks=[early_stopping, checkpoint]
)

# Print shapes of the training batches
for (batch_inputs, batch_targets) in dataset:
    ecg_raw_batch, ppg_raw_batch = batch_inputs
    print("Target Batch Shape:", batch_targets.shape)

now = datetime.now()
print('model training finished:',now)

# Save the trained model----------------------
model.save('Time-series_CNN_BiLSTM.keras')

# Plot train and validation curve ---------------------------------------
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# plot training and validation curve and save
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig('curve_timeseries_cnn_bilstm.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training and validation MAE and save
plt.figure(figsize=(4, 3))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE Curve')
plt.savefig('Timeseries_CNNBiLSTM_mae_curve.png', dpi=300, bbox_inches='tight')
plt.show()

del ecg_val_scaled, ppg_val_scaled, ecg_train_scaled, ppg_train_scaled
gc.collect()  # Force garbage collection

# Function to plot Baland_altman
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

def plot_bland_altman(true_sbp, true_dbp,  predicted_sbp, predicted_dbp, dataset_name='unknown'):

    labels = ['SBP', 'DBP']

    for k in range(2):  # 0 for SBP, 1 for DBP
        if (k==0):
           bp_target = true_sbp
           bp_output = predicted_sbp
        else:
           bp_target = true_dbp
           bp_output = predicted_dbp

        bp_mean = (bp_target + bp_output) / 2
        bp_diff = bp_target - bp_output
        bp_dev = np.std(bp_diff)

        # histogram2d
        counts, xedges, yedges = np.histogram2d(bp_mean, bp_diff, bins=300)

        x_bin = np.digitize(bp_mean, xedges) - 1
        y_bin = np.digitize(bp_diff, yedges) - 1
        z_real = np.zeros_like(bp_mean)
        for i in range(len(bp_mean)):
            if 0 <= x_bin[i] < counts.shape[0] and 0 <= y_bin[i] < counts.shape[1]:
                z_real[i] = counts[x_bin[i], y_bin[i]]
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(bp_mean, bp_diff, c=z_real, cmap='viridis', s=2, alpha=0.8, norm=Normalize())

        plt.axhline(np.mean(bp_diff), color='r', linestyle='--', linewidth=1, label='Mean Difference')
        plt.axhline(np.mean(bp_diff) + 1.96 * bp_dev, color='r', linestyle='--', linewidth=1, label='+1.96 SD')
        plt.axhline(np.mean(bp_diff) - 1.96 * bp_dev, color='r', linestyle='--', linewidth=1, label='-1.96 SD')

        cb = plt.colorbar(sc)
        cb.set_label('Number of points per pixel')
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xlabel('Average (Target + Output)/2 (mmHg)')
        plt.ylabel('Difference (Target - Output) (mmHg)')
        plt.title(f'Bland-Altman Plot - {labels[k]} ({dataset_name})')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"cnn_mychange_Blnatm_{dataset_name}_{labels[k]}.png", dpi=300, bbox_inches='tight')
        plt.close()


# Function to plot the correlation plot of estimated vs. target
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

def plot_estimated_vs_reference_with_density(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name='unknown'):
    labels = ['SBP', 'DBP']

    for i in range(2):  # 0 for SBP, 1 for DBP
        if (i==0): #SBP
            true = y_true_sbp
            pred = y_pred_sbp
        else:  #DBP
            true = y_true_dbp
            pred = y_pred_dbp

        # Density Calculation via histogram2d
        counts, xedges, yedges = np.histogram2d(true, pred, bins=300)

        x_bin = np.digitize(true, xedges) - 1
        y_bin = np.digitize(pred, yedges) - 1
        z_real = np.zeros_like(true)
        for j in range(len(true)):
            if 0 <= x_bin[j] < counts.shape[0] and 0 <= y_bin[j] < counts.shape[1]:
                z_real[j] = counts[x_bin[j], y_bin[j]]

        # Plotting
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(true, pred, c=z_real, cmap='viridis', s=2, alpha=0.8, norm=Normalize())
        cb = plt.colorbar(sc)
        cb.set_label('Number of points per pixel')
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Line of best fit
        coeffs = np.polyfit(true, pred, 1)
        poly_fn = np.poly1d(coeffs)
        x_vals = np.linspace(min(true), max(true), 100)
        plt.plot(x_vals, poly_fn(x_vals), color='red', linewidth=1, label=f'Line of Best Fit ({labels[i]})')

        # Line y = x
        plt.plot(x_vals, x_vals, color='limegreen', linestyle='dotted', linewidth=1, label='y = x')

        plt.xlabel("Target (mmHg)")
        plt.ylabel("Output (mmHg)")
        plt.title(f"Estimated vs. Reference {labels[i]} ({dataset_name})", fontsize=10)
        plt.xlim(0, 250)
        plt.ylim(0, 250)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"timeseries_cnn_bilstm_{dataset_name}_{labels[i]}.png", dpi=300, bbox_inches='tight')
        plt.close()


# function to evaluate model estimations on PulseDB test subsets
from sklearn.metrics import mean_squared_error, mean_absolute_error
def test_mymodel(y_scaled, y_true, ecg,ppg, dataset_name="unknown"):

    test_loss, test_mae, test_mse = model.evaluate((ecg,ppg), y_scaled)

    print(f" {dataset_name}: Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")

    y_pred = model.predict((ecg,ppg))
    y_pred = denormalize_labels(y_pred, sbp_min, sbp_max, dbp_min, dbp_max)

    # to convert shape (N,1) into shape (N,)
    y_true_sbp = y_true[:, 0]
    y_true_dbp = y_true[:, 1]
    y_pred_sbp = y_pred[:, 0]
    y_pred_dbp = y_pred[:, 1]

    y_true_sbp = y_true_sbp.astype(np.float32)
    y_true_dbp = y_true_dbp.astype(np.float32)
    y_pred_sbp = y_pred_sbp.astype(np.float32)
    y_pred_dbp = y_pred_dbp.astype(np.float32)

    # ME (Mean Error)
    me_sbp = np.mean(y_true_sbp - y_pred_sbp)
    me_dbp = np.mean(y_true_dbp - y_pred_dbp)

    # MSE
    mse_sbp = mean_squared_error(y_true_sbp, y_pred_sbp)
    mse_dbp = mean_squared_error(y_true_dbp, y_pred_dbp)

    # MAE
    mae_sbp = mean_absolute_error(y_true_sbp, y_pred_sbp)
    mae_dbp = mean_absolute_error(y_true_dbp, y_pred_dbp)

    # SDE
    std_sbp = np.std(y_true_sbp - y_pred_sbp)
    std_dbp = np.std(y_true_dbp - y_pred_dbp)

    print(f"{dataset_name} - SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, STD={std_sbp:.2f}")
    print(f"{dataset_name} - DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, STD={std_dbp:.2f}")

    print(type(y_true_sbp), np.shape(y_true_sbp))
    print(type(y_pred_sbp), np.shape(y_pred_sbp))

    # Plot Estimated vs. Reference SBP
    plot_estimated_vs_reference_with_density(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name=dataset_name)

    # Compute Bland-Altman plots for SBP and DBP
    plot_bland_altman(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name= dataset_name)


# Evaluate the model perfomance on test subsets

# define test sets
# AAMI Cal is added only for exploratory purposes, not for evaluation reports
test_subsets = ["Test_CalBased", "Test_CalFree", "AAMI_Cal"]

for subset in test_subsets:
    # Load test data

    # ECG test data
    ecg_test = np.load(myData + subset + '_ecg.npy', mmap_mode='r')
    ecg_test = ecg_test.astype(np.float32)

    # PPG test data
    ppg_test = np.load(myData + subset +'_ppg.npy', mmap_mode='r')
    ppg_test = ppg_test.astype(np.float32)

    # adjust the input dimensions
    ecg_test = np.expand_dims(ecg_test, axis=-1)
    ppg_test = np.expand_dims(ppg_test, axis=-1)

    # load SBP and DBP of test set
    y_test = np.load(myData + subset +'_Labels.npy' , mmap_mode='r')
    y_test = y_test.astype(np.float32)

    # Apply z-score normalization on subsets' Labels
    y_test_scaled = min_max_scale(y_test, sbp_min, sbp_max, dbp_min, dbp_max)

    # Test on test
    test_mymodel(y_test_scaled,
                 y_test,
                 ecg_test, ppg_test, subset)
