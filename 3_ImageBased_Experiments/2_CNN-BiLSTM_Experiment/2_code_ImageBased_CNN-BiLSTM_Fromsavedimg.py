import os
model_name = os.environ.get('MODELNAME', 'CNN')
batch_size = int (os.environ.get('BATCHSIZE', 256))

import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
print("Built with CUDA:", tf.test.is_built_with_cuda())

from tensorflow.keras import layers, models
import gc  # Garbage collection  to prevent memory usage error
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import spectrogram  #STFT
import pywt  # CWT  pip install PyWavelets
from pyts.image import GramianAngularField #GAF pip install pyts
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Function to Z-score normalisation for LABELS
def z_score_scale(y, mean, std):
    y_scaled = (y - mean) / std
    return y_scaled

# Function to Z-score De-normalization for LABELS
def denormalize_labels_zscore(y_scaled, mean, std):
    y = y_scaled * std + mean
    return y

# convert [0-1] into mmHg in loss function to minimise error in scale of mmHg, not scales between [0,1]
# MAE loss in mmHg with Z-score normalization

# Define a custom loss function
def mmhg_mae_loss(bp_mean, bp_std):

    def loss(y_true_scaled, y_pred_scaled):
        # de-normalize to mmHg
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    return loss

# Define metric function
def mmhg_mae_metric(bp_mean, bp_std):

    def metric(y_true_scaled, y_pred_scaled):
        # de-normalize from Z-score
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    metric.__name__ = "mae_mmhg"
    return metric

# mixed-precision to prevent Out Of Memory Error
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision policy:", mixed_precision.global_policy())

# Define model
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def CNN_BiLSTM_2D(ecg_img_shape, ppg_img_shape,head_mlp_units):

    # Inputs
    ecg_input = Input(shape=ecg_img_shape, name='ecg_input')
    ppg_input = Input(shape=ppg_img_shape, name='ppg_input')

    # ECG CNN
    x1 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ecg_input)
    x1 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x1)

    #instead of max pooling that decreases suddenly, reducing is learnable
    x1 = layers.Conv2D(64,  (5,5), strides=(2,1), padding='same', activation='relu') (x1) 
    x1 = layers.Conv2D(64,  (5,5), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(256, (5,5), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(256, (5,5), strides=(2,1), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(128, (5,5), padding ='same', activation='relu')(x1)
    x1 = layers.Conv2D(128, (5,5), strides=(2,1), padding='same', activation='relu')(x1) 
    
    # PPG CNN
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ppg_input)
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x2)

    #instead of max pooling that decrease suddenly, reducing is learnable
    x2 = layers.Conv2D(64,  (7,7), strides=(2,1), padding='same', activation='relu') (x2) 
    x2 = layers.Conv2D(64,  (7,7), padding='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(256, (7,7), padding='same', activation='relu') (x2)
    x2 = layers.Conv2D(256, (7,7), strides=(2,1), padding='same', activation='relu') (x2)
    x2 = layers.Conv2D(128, (5,5), strides = (2,1), padding ='same', activation='relu')(x2) 
    x2 = layers.Conv2D(128, (3,3), strides = (1,1), padding ='same', activation='relu') (x2)

     # This block: from (batch, frequency, time, channels)  ==> (batch, time, frequency, channels) ==> (batch, time, frequency * channels)

    # Merge features
    x = layers.Concatenate(axis=-1)([x1, x2]) # (batch,H, W, C)
    # Set time dimension as the main idea.  Put the time dimension in axis 1
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]))(x)   # (batch, W, H, C)
    # make features from  frequency and channels to achieve (batch, time, features)
    # Flatten freq and channels into features
    x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2] * tf.shape(t)[3])))(x) # (batch, time, features)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    # Shared learnable Fully Connected layers
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    # 6) Regression Head

    x_sbp = x
    for j, units in enumerate(head_mlp_units):
        x_sbp = layers.Dense(units, activation="relu", name=f"head_dense_sbp_{j}")(x_sbp)

    x_dbp = x
    for j, units in enumerate(head_mlp_units):
        x_dbp = layers.Dense(units, activation="relu", name=f"head_dense_dbp_{j}")(x_dbp)

    # in mixed-precision  dtype="float32",  was added
    sbp_output = layers.Dense(1, activation=None,  dtype="float32", name="sbp_output")(x_sbp)
    dbp_output = layers.Dense(1, activation=None,  dtype="float32", name="dbp_output")(x_dbp)

    model = Model(inputs=[ecg_input, ppg_input], outputs=[sbp_output,dbp_output], name='CNN_BiLSTM_2D')
    return model

# load data chunk by chunk of size 126 samples per file
import glob
from tensorflow.keras.utils import Sequence

# These functions are defined only to manage RAM, due to Out Of Memory errors. Also, to prevent GPU idling between loading data
# Function to load aligned ECG, PPG, SBP, and DBP data from disk in chunks to reduce RAM usage.
class ChunkSequence(Sequence):

    def __init__(self, data_dir, subset, shuffle=True):

        super().__init__()
        self.data_dir = data_dir
        self.subset = subset
        self.shuffle = shuffle

        self.ecg_files = sorted(glob.glob(f"{data_dir}/ecg_{subset}_*.npy"))
        self.ppg_files = sorted(glob.glob(f"{data_dir}/ppg_{subset}_*.npy"))
        self.sbp_files = sorted(glob.glob(f"{data_dir}/sbp_{subset}_*.npy"))
        self.dbp_files = sorted(glob.glob(f"{data_dir}/dbp_{subset}_*.npy"))
        
        print(
            f"{subset}: "
            f"ECG={len(self.ecg_files)}, "
            f"PPG={len(self.ppg_files)}, "
            f"SBP={len(self.sbp_files)}, "
            f"DBP={len(self.dbp_files)}"
          )

        if not (
             len(self.ecg_files)
                == len(self.ppg_files)
                == len(self.sbp_files)
                == len(self.dbp_files)
            ):
             raise ValueError(f"Chunk count mismatch in {subset}")

        self.indices = np.arange(len(self.ecg_files))
        self.on_epoch_end()
        print(f"{subset} chunks:", len(self.ecg_files))

    def __len__(self):
        return len(self.ecg_files)


    def __getitem__(self, idx):

        chunk_id = self.indices[idx]
        ecg_file = self.ecg_files[chunk_id]
        batch_num = os.path.basename(ecg_file).split("_")[-1].replace(".npy", "")
        ecg = np.load(f"{self.data_dir}/ecg_{self.subset}_{batch_num}.npy")
        ppg = np.load(f"{self.data_dir}/ppg_{self.subset}_{batch_num}.npy")
        sbp = np.load(f"{self.data_dir}/sbp_{self.subset}_{batch_num}.npy").astype(np.float32)
        dbp = np.load(f"{self.data_dir}/dbp_{self.subset}_{batch_num}.npy").astype(np.float32)

        return (
            {
                "ecg_input": ecg,
                "ppg_input": ppg
            },
            {
                "sbp_output": sbp,
                "dbp_output": dbp
            }
        )


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# To prevent losing data at the end of each chunk
# As data is  180432, the chunk size can be =>16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 112, 126, 144, 168, 252, 336, 504
import os
import glob
import numpy as np

def load_all_chunks(data_dir, subset):

    ecg_files = sorted(glob.glob(f"{data_dir}/ecg_{subset}_*.npy"))
    ppg_files = sorted(glob.glob(f"{data_dir}/ppg_{subset}_*.npy"))
    sbp_files = sorted(glob.glob(f"{data_dir}/sbp_{subset}_*.npy"))
    dbp_files = sorted(glob.glob(f"{data_dir}/dbp_{subset}_*.npy"))

    # Check number of files
    if not (
        len(ecg_files)
        == len(ppg_files)
        == len(sbp_files)
        == len(dbp_files)
    ):
        raise ValueError(f"Chunk count mismatch in {subset}")

    print(f"{subset} chunks:", len(ecg_files))

    # Determine total number of samples without loading data into RAM
    total_samples = 0

    for f in ecg_files:
        x = np.load(f, mmap_mode="r")
        total_samples += x.shape[0]

    # Get image shape
    sample = np.load(ecg_files[0], mmap_mode="r")
    image_shape = sample.shape[1:]

    print(f"{subset} total samples:", total_samples)
    print(f"{subset} image shape:", image_shape)

    # Allocate final arrays only once
    ecg_all = np.empty(
        (total_samples, *image_shape),
        dtype=np.float32
    )

    ppg_all = np.empty(
        (total_samples, *image_shape),
        dtype=np.float32
    )

    sbp_all = np.empty(
        (total_samples, 1),
        dtype=np.float32
    )

    dbp_all = np.empty(
        (total_samples, 1),
        dtype=np.float32
    )

    start = 0

    for ecg_file, ppg_file, sbp_file, dbp_file in zip(
        ecg_files,
        ppg_files,
        sbp_files,
        dbp_files
    ):

        ecg = np.load(ecg_file).astype(np.float32, copy=False)
        ppg = np.load(ppg_file).astype(np.float32, copy=False)
        sbp = np.load(sbp_file).astype(np.float32, copy=False)
        dbp = np.load(dbp_file).astype(np.float32, copy=False)

        n = ecg.shape[0]

        # alignment check
        assert (
            ppg.shape[0] == n
            and sbp.shape[0] == n
            and dbp.shape[0] == n
        )

        end = start + n

        ecg_all[start:end] = ecg
        ppg_all[start:end] = ppg
        sbp_all[start:end] = sbp
        dbp_all[start:end] = dbp

        start = end

    print(f"{subset} loaded into RAM.")

    return (
        {
            "ecg_input": ecg_all,
            "ppg_input": ppg_all
        },
        {
            "sbp_output": sbp_all,
            "dbp_output": dbp_all
        }
    )

from tensorflow.keras.utils import Sequence

# To prevent the GPU from idling between loading chunks
# This code increases the SM activity of the GPU usage, which means the allocated GPU is not idle between loading chunks
class RAMSequence(Sequence):

    def __init__(self, x, y, batch_size, shuffle=True):
        super().__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(x["ecg_input"]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):

        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))

        ids = self.indices[start:end]

        return (
            {
                "ecg_input": self.x["ecg_input"][ids],
                "ppg_input": self.x["ppg_input"][ids]
            },
            {
                "sbp_output": self.y["sbp_output"][ids],
                "dbp_output": self.y["dbp_output"][ids]
            }
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# Reduce the learning rate when validation loss stops improving
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

 
# Specify the folder path for loading the images that were already saved as npy files in size of 126
data_dir = "/scratch/username/cwt_32sr_5sec"

import time
load_start = time.perf_counter()

# Load training data into RAM
train_x, train_y = load_all_chunks( data_dir, "train")

# Create data generators for training and validation
train_generator = RAMSequence(
    train_x,
    train_y,
    batch_size=batch_size,
    shuffle=True
)

val_generator = ChunkSequence(
    data_dir=data_dir,
    subset="val",
    shuffle=False
)

# print loading time to obtain rough estimates of the load time required for resource allocation.
load_end = time.perf_counter()
print(f"Data loading time: {(load_end - load_start)/60:.2f} minutes")

# print data shapes to be sure all data are in correct format before training starts
print(train_x["ecg_input"].shape)
print(train_x["ppg_input"].shape)
print(train_y["sbp_output"].shape)
print(train_y["dbp_output"].shape)

input_shape = train_x["ecg_input"].shape[1:]
H, W, _ = input_shape
print("input_shape:", input_shape)
print("H:", H)
print("W:", W)

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(device):
    model = CNN_BiLSTM_2D(
               ecg_img_shape= input_shape,
               ppg_img_shape= input_shape,
               head_mlp_units=[1024, 1024, 512, 256]
               )

# load mean and std from file for denormalisation from training set
sbp_mean = np.float32( np.load(f"{data_dir}/sbp_mean.npy"))
sbp_std  = np.float32( np.load(f"{data_dir}/sbp_std.npy"))

dbp_mean = np.float32( np.load(f"{data_dir}/dbp_mean.npy"))
dbp_std  = np.float32( np.load(f"{data_dir}/dbp_std.npy"))

print("SBP mean/std:", sbp_mean, sbp_std)
print("DBP mean/std:", dbp_mean, dbp_std)

# Clarify the customised loss functions for the model
loss_fn_sbp = mmhg_mae_loss(sbp_mean, sbp_std)
loss_fn_dbp = mmhg_mae_loss(dbp_mean, dbp_std)

mae_mmhg_sbp = mmhg_mae_metric(sbp_mean, sbp_std)
mae_mmhg_dbp = mmhg_mae_metric(dbp_mean, dbp_std)

# Specify the initial Learning Rate
lr = 1e-4

model.compile (optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss={
                "sbp_output": loss_fn_sbp,
                "dbp_output": loss_fn_dbp
                },
              loss_weights={
                "sbp_output": 1.0,
                "dbp_output": 1.0
                },
              metrics={
                "sbp_output": [mae_mmhg_sbp],
                "dbp_output": [mae_mmhg_dbp]
                },
              jit_compile=False
              )

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_sbp_output_mae_mmhg',
    #monitor ='val_loss',
    mode='min',
    patience=8,
    restore_best_weights=True
    )

max_epochs = 200
train_start = time.perf_counter()

# Start training the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=max_epochs,
    verbose=2, # 2=> to show losses right after each epoch ends
    callbacks=[reduce_lr,early_stopping]
)

train_end = time.perf_counter()
print(f"Training time: {(train_end - train_start)/60:.2f} minutes")

# Save the trained model
model.save('img_' + model_name + f"_fromsavedimg.keras")

# Plot the curves
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(4, 3))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig('img_'+ model_name + f"_loadfromsavedimg_" +'_loss_curve.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.show()

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
        # felan comment, if error the you can uncomment it
        #bp_target = np.asarray(reshape(-1))
        #bp_output = np.asarray(reshape(-1))

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
        plt.savefig(f"img_{model_name}_Blnatm_{dataset_name}_{labels[k]}.png", dpi=300, bbox_inches='tight')
        plt.close()

from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

# Function to plot correlation plot: Estimated vs. Reference 
def plot_estimated_vs_reference_with_density(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name='unknown'):
    labels = ['SBP', 'DBP']

    for i in range(2):  # 0 for SBP, 1 for DBP
        if (i==0): #sbp
            true = y_true_sbp
            pred = y_pred_sbp
        else: #dbp
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
        plt.savefig(f"img_{model_name}_{dataset_name}_{labels[i]}.png", dpi=300, bbox_inches='tight')
        plt.close()

# Function to evaluate model performance on subset test 
def test_mymodel (y_pred_sbp, y_pred_dbp, y_true_sbp, y_true_dbp, dataset_name="unknown"):

    # flatten
    y_true_sbp = np.asarray(y_true_sbp[:, 0], dtype=np.float32)
    y_true_dbp = np.asarray(y_true_dbp[:, 0], dtype=np.float32)
    y_pred_sbp = np.asarray(y_pred_sbp[:, 0], dtype=np.float32)
    y_pred_dbp = np.asarray(y_pred_dbp[:, 0], dtype=np.float32)

    # metrics
    # Mean error
    me_sbp = np.mean(y_true_sbp - y_pred_sbp)
    me_dbp = np.mean(y_true_dbp - y_pred_dbp)
    
    # Mean Squared Error
    # MSE is calculated as a supplementary metric for
    # experimental logging and additional analysis.
    # The primary reported metrics are ME, MAE, and SDE.
    mse_sbp = mean_squared_error(y_true_sbp, y_pred_sbp)
    mse_dbp = mean_squared_error(y_true_dbp, y_pred_dbp)

    # Mean Absolute Error
    mae_sbp = mean_absolute_error(y_true_sbp, y_pred_sbp)
    mae_dbp = mean_absolute_error(y_true_dbp, y_pred_dbp)

    # Standard Deviation of Error
    std_sbp = np.std(y_true_sbp - y_pred_sbp)
    std_dbp = np.std(y_true_dbp - y_pred_dbp)

    print(f"{dataset_name} - SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, STD={std_sbp:.2f}")
    print(f"{dataset_name} - DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, STD={std_dbp:.2f}")

    plot_estimated_vs_reference_with_density(
        y_true_sbp, y_true_dbp,
        y_pred_sbp, y_pred_dbp,
        dataset_name=dataset_name
    )

    plot_bland_altman(
        y_true_sbp, y_true_dbp,
        y_pred_sbp, y_pred_dbp,
        dataset_name=dataset_name
    )

    return {
        "me_sbp": me_sbp,
        "me_dbp": me_dbp,
        "mae_sbp": mae_sbp,
        "mae_dbp": mae_dbp,
        "mse_sbp": mse_sbp,
        "mse_dbp": mse_dbp,
        "std_sbp": std_sbp,
        "std_dbp": std_dbp
    }

# Evaluate model on test subsets

# Define subset list
test_subsets = ["Test_CalBased", "Test_CalFree"]
for subset in test_subsets:

    data_dir_test = data_dir+ f"_test/{subset}/"
    test_generator = ChunkSequence(
        data_dir= data_dir_test,
        subset= subset,
        shuffle= False
    )

    y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict(
        test_generator,
        verbose=0
    )

    # evaluate batch-by-batch
    results = model.evaluate(test_generator, verbose=0)

    print("evaluate results:", results)

    # de-normalise predictions
    y_pred_sbp = denormalize_labels_zscore(y_pred_sbp_scaled, sbp_mean, sbp_std)
    y_pred_dbp = denormalize_labels_zscore(y_pred_dbp_scaled, dbp_mean, dbp_std)

    #load true labels of the subset
    y_true_sbp_scaled = []
    y_true_dbp_scaled = []

    for i in range(len(test_generator)):
        _, y_batch = test_generator[i]

        y_true_sbp_scaled.append(y_batch["sbp_output"])
        y_true_dbp_scaled.append(y_batch["dbp_output"])

    y_true_sbp_scaled = np.concatenate(y_true_sbp_scaled, axis=0)
    y_true_dbp_scaled = np.concatenate(y_true_dbp_scaled, axis=0)

    # denormalize true labels
    y_true_sbp = denormalize_labels_zscore(y_true_sbp_scaled, sbp_mean, sbp_std)
    y_true_dbp = denormalize_labels_zscore(y_true_dbp_scaled, dbp_mean, dbp_std)

    # test model
    test_mymodel(y_pred_sbp, y_pred_dbp, y_true_sbp, y_true_dbp, subset)
