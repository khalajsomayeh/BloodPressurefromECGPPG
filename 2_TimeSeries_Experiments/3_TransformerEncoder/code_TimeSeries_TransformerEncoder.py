import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gc  # Garbage collection  to prevent memory usage error
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to Z-score normalization for LABELS
def z_score_scale(y, mean, std):
    y_scaled = y.copy()
    y_scaled = (y_scaled - mean) / std
    return y_scaled

# Function to Z-score De-normalization for LABELS
def denormalize_labels_zscore(y_scaled, mean, std):
    y = y_scaled.copy()
    y = y * std + mean
    return y

# convert [0-1] into mmHg in loss function to mnimize error in scale of mmHg not percetage between [0,1]
# Function to define MAE loss in mmHg with Z-score normalization
def mmhg_mae_loss(bp_mean, bp_std):

    def loss(y_true_scaled, y_pred_scaled):
        # de-normalize to mmHg
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    return loss

# Function to define MAE metrics
def mmhg_mae_metric(bp_mean, bp_std):

    def metric(y_true_scaled, y_pred_scaled):
        # de-normalize from Z-score
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    metric.__name__ = "mae_mmhg"
    return metric

# Define Model
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class LearnableMultiHeadTemporalAttentionPooling(layers.Layer):
    def __init__(self, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.dim = input_shape[-1]  # D
        self.attn_dense = layers.Dense(self.num_heads)
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, D)

        # scores: (B, T, H)
        scores = self.attn_dense(x)

        # weights: (B, T, H)
        weights = tf.nn.softmax(scores, axis=1)

        # (B, T, 1, D)
        x_exp = tf.expand_dims(x, axis=2)

        # weighted sum → (B, H, D)
        pooled = tf.reduce_sum(
            x_exp * tf.expand_dims(weights, -1),
            axis=1
        )

        # average over heads → (B, D)
        return tf.reduce_mean(pooled, axis=1)

def mlp_block(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_vit_1d_regressor(
        seq_len,
        num_channels,
        projection_dim,
        transformer_layers,
        num_heads,
        encoder_mlp_units,
        head_mlp_units,
        dropout_rate
    ):
    
    input_ecg = Input(shape=(seq_len, num_channels), name="ecg_input")
    input_ppg = Input(shape=(seq_len, num_channels), name="ppg_input")

    # each time step = one token
    x_ecg = layers.Dense(projection_dim, name="token_projection_ecg")(input_ecg)
    x_ppg = layers.Dense(projection_dim, name="token_projection_ppg")(input_ppg)

    num_tokens = seq_len
    #  2) Positional embedding ‌
    positions = tf.range(start=0, limit=num_tokens, delta=1)
    pos_emb_ecg = layers.Embedding(
        input_dim=num_tokens,
        output_dim=projection_dim,
        name="pos_embedding_ecg"
    )

    pos_emb_ppg = layers.Embedding(
        input_dim=num_tokens,
        output_dim=projection_dim,
        name="pos_embedding_ppg"
    )

    pos_embedding_ecg = pos_emb_ecg(positions)  # (num_patches, projection_dim)
    pos_embedding_ppg = pos_emb_ppg(positions)  # (num_patches, projection_dim)
    x_ecg = x_ecg + pos_embedding_ecg  # broadcast on batch
    x_ppg = x_ppg + pos_embedding_ppg  # broadcast on batch


    #  3) Several layers of Transformer Encoder
    # --ECG
    for i in range(transformer_layers):
        # Self-Attention block
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_ecg_{i}")(x_ecg)
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            #key_dim=projection_dim,
            key_dim = projection_dim // num_heads,
            dropout=dropout_rate,
            name=f"mha_ecg_{i}"
        )(x1, x1)
        x2 = layers.Add(name=f"residual_attn_ecg_{i}")([x_ecg, attn_out])

        # MLP block (FNN inside Encoder)
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_mlp_ecg_{i}")(x2)
        x4 = mlp_block(x3, hidden_units=encoder_mlp_units, dropout_rate=dropout_rate)
        x4 = layers.Dense(x2.shape[-1], name=f"mlp_proj_ecg_{i}")(x4)
        x_ecg = layers.Add(name=f"residual_mlp_ecg_{i}")([x2, x4])


    #  PPG
    for i in range(transformer_layers):
        # Self-Attention block
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_ppg_{i}")(x_ppg)
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            #key_dim=projection_dim,
            key_dim = projection_dim // num_heads,
            dropout=dropout_rate,
            name=f"mha_ppg_{i}"
        )(x1, x1)
        x2 = layers.Add(name=f"residual_attn_ppg_{i}")([x_ppg, attn_out])

        # MLP block (FNN inside Encoder)
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_mlp_ppg_{i}")(x2)
        x4 = mlp_block(x3, hidden_units=encoder_mlp_units, dropout_rate=dropout_rate)
        x4 = layers.Dense(x2.shape[-1], name=f"mlp_proj_ppg_{i}")(x4)
        x_ppg = layers.Add(name=f"residual_mlp_ppg_{i}")([x2, x4])

    # 4) Normalization
    x_ecg = layers.LayerNormalization(epsilon=1e-6, name="ln_final_ecg")(x_ecg)
    x_ppg = layers.LayerNormalization(epsilon=1e-6, name="ln_final_ppg")(x_ppg)

    # 5)   sequence →  vector
    # shape: (batch, seq_len, 2 * projection_dim)
    x = layers.Concatenate(axis=-1, name="token_fusion_concat")([x_ecg, x_ppg])

    # bring it back to projection_dim
    x = layers.Dense(projection_dim, name="fusion_projection")(x)

    # fusion middle finds the relationship between ECG and PPG
    fusion_layers = 4

    for i in range(fusion_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_fusion_{i}")(x)

        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate,
            name=f"mha_fusion_{i}"
        )(x1, x1)

        x2 = layers.Add(name=f"residual_attn_fusion_{i}")([x, attn_out])
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_mlp_fusion_{i}")(x2)
        x4 = mlp_block(
            x3,
            hidden_units=encoder_mlp_units,
            dropout_rate=dropout_rate
        )

        x4 = layers.Dense(projection_dim, name=f"mlp_proj_fusion_{i}")(x4)
        x = layers.Add(name=f"residual_mlp_fusion_{i}")([x2, x4])
    x = layers.LayerNormalization(epsilon=1e-6, name="ln_final_fusion")(x)
    # sequence → vector
    x = LearnableMultiHeadTemporalAttentionPooling(
        num_heads=4,
        name="mh_attn_pool_fusion"
       )(x)

    print('x:x_ech, x_ppg',x.shape)

    # shared learnable regression head
    x_shared = layers.Dense(1024, activation="relu")(x)
    x_shared = layers.Dense(512, activation="relu")(x_shared)

    # 6) Regression Head
    # SBP
    x_sbp = x_shared
    for j, units in enumerate(head_mlp_units):
        x_sbp = layers.Dense(units, activation="relu", name=f"head_dense_sbp_{j}")(x_sbp)
        
    # DBP
    x_dbp = x_shared
    for j, units in enumerate(head_mlp_units):
        x_dbp = layers.Dense(units, activation="relu", name=f"head_dense_dbp_{j}")(x_dbp)

    output_sbp = layers.Dense(1, activation=None, name="sbp_output")(x_sbp)
    output_dbp = layers.Dense(1, activation=None, name="dbp_output")(x_dbp)

    model = Model(inputs=[input_ecg,input_ppg], outputs=[output_sbp,output_dbp], name="ViT_1D_BP_Regressor")
    return model

# Resegmentation from 10-second segments to 5 seconds
import numpy as np

def find_local_peaks_troughs(signal, min_distance=30):
    """
    Detect local peaks and troughs using shape only:
    peak   : x[i-1] < x[i] > x[i+1]
    trough : x[i-1] > x[i] < x[i+1]

    min_distance prevents detecting too many nearby extrema.
    """
    x = np.asarray(signal, dtype=np.float32)

    peak_idx = []
    trough_idx = []

    last_peak = -min_distance
    last_trough = -min_distance

    for i in range(1, len(x) - 1):
        # local peak
        if x[i-1] < x[i] and x[i] > x[i+1]:
            #if i - last_peak >= min_distance:
                peak_idx.append(i)
                last_peak = i

        # local trough
        if x[i-1] > x[i] and x[i] < x[i+1]:
            # if i - last_trough >= min_distance:
                trough_idx.append(i)
                last_trough = i

    return np.array(peak_idx), np.array(trough_idx)

# Function to recalculate SBP and DBP from resegmented ABPs
def compute_sbp_dbp_from_abp_segment(abp_5s, min_distance=30):
    peak_idx, trough_idx = find_local_peaks_troughs(abp_5s, min_distance=min_distance)

    if len(peak_idx) == 0 or len(trough_idx) == 0:
        return np.nan, np.nan

    sbp = np.mean(abp_5s[peak_idx])
    dbp = np.mean(abp_5s[trough_idx])
    return sbp, dbp

def shuffle_in_unions(ecg, ppg, y_sbp, y_dbp):
    idx = np.random.permutation(len(ecg))
    return ecg[idx], ppg[idx], y_sbp[idx], y_dbp[idx]

# Apply resegmentation on ECG, PPG, and ABP
def apply_resegmentation(seg_target, fs, ecg, ppg, abp, y_sbp, y_dbp):
    """
    ecg: (N, 1250)
    ppg: (N, 1250)
    y  : (N, 2)  or (N, 1)

    returns:
    ecg_new, ppg_new, y_new
    with shapes:
    (2N, 625), (2N, 625), (2N, ...)
    """
    if (seg_target==10):
        return ecg, ppg, y_sbp, y_dbp
    #ecg.shape[1] // sub_segment_time  # example 1250/2
    original_segment_length = 10 # seconds
    n  = int (fs * seg_target)
    num_parts = original_segment_length // seg_target
    ecg_list = []
    ppg_list = []
    abp_list = []
    sbp_list = []
    dbp_list = []

    for part in range(num_parts):
        start = part * n
        end = (part + 1) * n

        ecg_sub = ecg[:, start:end]
        ppg_sub = ppg[:, start:end]
        abp_sub = abp[:, start:end]

        ecg_list.append(ecg_sub)
        ppg_list.append(ppg_sub)
        abp_list.append(abp_sub)

        sbp_list.append(y_sbp)
        dbp_list.append(y_dbp)

    ecg_new = np.vstack(ecg_list)
    ppg_new = np.vstack(ppg_list)
    abp_new = np.vstack(abp_list)
    y_sbp_new = np.vstack(sbp_list)
    y_dbp_new = np.vstack(dbp_list)

    # re-calculate labels from abp signals for each segment
    for i in range(abp_new.shape[0]):
        sbp, dbp = compute_sbp_dbp_from_abp_segment (abp_new[i])
        if (not np.isfinite(sbp)) or (not np.isfinite(dbp)) or (sbp < 0) or (dbp < 0) or (sbp < dbp):
              continue
        else:
              y_sbp_new[i] = sbp
              y_dbp_new[i] = dbp

    # Shuffle new segments
    ecg_new, ppg_new, y_sbp_new, y_dbp_new = shuffle_in_unions (ecg_new, ppg_new, y_sbp_new, y_dbp_new)
    return ecg_new, ppg_new, y_sbp_new, y_dbp_new

# Specify hyperparameters
all_hyper = {
    "num_heads":         [8], 
    "transformer_layers":[4], 
    "projection_dim":    [128], 
    "encoder_mlp_units": [(512,256,128)], 
    "head_mlp_units":    [(512,256,128,64)], 
    "dropout_rate":      [0.001],
    "lr":                [1e-4],
    "batch_size":        [128],
    "max_epochs":        [200],
}

hyper_keys = list(all_hyper.keys())
hyper_values = list(all_hyper.values())

all_configs = list(product(*hyper_values))
print(f"Total configurations: {len(all_configs)}")

tf.keras.utils.disable_interactive_logging()

# Train the model on 80% train and 20% validation using the hyperparameters
# Load data
myData='pulsedb_data_npy/'

# load training data
ecg_train = np.load(myData + 'train_ecg.npy', mmap_mode='r')
ppg_train = np.load(myData + 'train_ppg.npy', mmap_mode='r')
abp_train = np.load(myData + 'train_abp.npy', mmap_mode='r')
y_train = np.load(myData + 'train_Labels.npy' , mmap_mode='r')

# laod validation data
ecg_val = np.load(myData + 'val_ecg.npy' , mmap_mode='r')
ppg_val = np.load(myData + 'val_ppg.npy' , mmap_mode='r')
abp_val = np.load(myData + 'val_abp.npy' , mmap_mode='r')
y_val = np.load(myData + 'val_Labels.npy' , mmap_mode='r')

# convert (y,2) into (SBP,1) and (DBP,1)
# for training set
y_train_sbp = y_train[:, 0:1]
y_train_dbp = y_train[:, 1:2]

# for validation set
y_val_sbp   = y_val[:, 0:1]
y_val_dbp   = y_val[:, 1:2]

# apply resegmentation from 10-second segments into 5-second segments for ECG, PPG, and ABP in training set
ecg_train_resegmented, ppg_train_resegmented, y_train_sbp_resegmented, y_train_dbp_resegmented = apply_resegmentation (
                                                                                                           5, 125,
                                                                                                           ecg_train, ppg_train, abp_train,
                                                                                                           y_train_sbp, y_train_dbp)

# apply resegmentation from 10-second segments into 5-second segments for ECG, PPG, and ABP in validation set
ecg_val_resegmented, ppg_val_resegmented, y_val_sbp_resegmented, y_val_dbp_resegmented  = apply_resegmentation (
                                                                                                           5, 125,
                                                                                                           ecg_val, ppg_val, abp_val,
                                                                                                           y_val_sbp,   y_val_dbp)

# resegmentation creates many arrays, the previous ones are deleted from memory
del ecg_train, ppg_train, abp_train, ecg_val, ppg_val, abp_val, y_train_sbp, y_train_dbp, y_val_sbp, y_val_dbp
gc.collect()

# assign the arrays above to the name used in the rest of the code, to prevent changing the whole code
ecg_train = ecg_train_resegmented
ppg_train = ppg_train_resegmented

y_train_sbp = y_train_sbp_resegmented
y_train_dbp = y_train_dbp_resegmented

ecg_val = ecg_val_resegmented
ppg_val = ppg_val_resegmented

y_val_sbp = y_val_sbp_resegmented
y_val_dbp = y_val_dbp_resegmented

# convert (N, seq_len) into (N, seq_len, num_channels=1)
ecg_train = ecg_train[..., np.newaxis]
ppg_train = ppg_train[..., np.newaxis]

ecg_val = ecg_val[..., np.newaxis]
ppg_val = ppg_val[..., np.newaxis]

# --------------------------------------------------------
# Get Z-score parameters from TRAIN ONLY
sbp_mean = np.mean(y_train_sbp)
sbp_std  = np.std(y_train_sbp)

dbp_mean = np.mean(y_train_dbp)
dbp_std  = np.std(y_train_dbp)

# print the statistics parameters 
print("SBP mean/std:", sbp_mean, sbp_std)
print("DBP mean/std:", dbp_mean, dbp_std)

# Apply z-score normalization on subsets' Labels
y_train_sbp_scaled = z_score_scale(y_train_sbp, sbp_mean, sbp_std)
y_train_dbp_scaled = z_score_scale(y_train_dbp, dbp_mean, dbp_std)

y_val_sbp_scaled   = z_score_scale(y_val_sbp, sbp_mean, sbp_std)
y_val_dbp_scaled   = z_score_scale(y_val_dbp, dbp_mean, dbp_std)

# print the values
print("SBP scaled mean/std:", y_train_sbp_scaled.mean(), y_train_sbp_scaled.std())
print("DBP scaled mean/std:", y_train_dbp_scaled.mean(), y_train_dbp_scaled.std())

print("ecg_train:", ecg_train.shape, ecg_train.min(), ecg_train.max(), ecg_train.mean(), ecg_train.std())
print("ppg_train:", ppg_train.shape, ppg_train.min(), ppg_train.max(), ppg_train.mean(), ppg_train.std())

print("y_train_sbp_scaled:", y_train_sbp_scaled.shape, y_train_sbp_scaled.mean(), y_train_sbp_scaled.std())
print("y_train_dbp_scaled:", y_train_dbp_scaled.shape, y_train_dbp_scaled.mean(), y_train_dbp_scaled.std())

# Prepare data to feed to the model
seq_len = ecg_train.shape[1]
num_channels = 1 

# define the device to run model training 
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print("Using device:", device)

for cfg_id, cfg in enumerate(all_configs, start=1):
    config = dict(zip(hyper_keys, cfg))

# print config to have the results alongside their config in log files
print (config)

# Build the model
with tf.device(device):
        model = build_vit_1d_regressor(
            seq_len= seq_len,
            num_channels= num_channels,
            projection_dim= best_config ["projection_dim"],
            transformer_layers= best_config ["transformer_layers"],
            num_heads= best_config ["num_heads"],
            encoder_mlp_units= best_config ["encoder_mlp_units"],
            head_mlp_units= best_config ["head_mlp_units"],
            dropout_rate= best_config ["dropout_rate"]
        )

# Clarify the functions for the model as loss functions
loss_fn_sbp = mmhg_mae_loss(sbp_mean, sbp_std)
loss_fn_dbp = mmhg_mae_loss(dbp_mean, dbp_std)

mae_mmhg_sbp = mmhg_mae_metric(sbp_mean, sbp_std)
mae_mmhg_dbp = mmhg_mae_metric(dbp_mean, dbp_std)

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_config["lr"]),
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
    }
    )

# Define early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_sbp_output_mae_mmhg',
    mode='min',
    patience=10,
    restore_best_weights=True
    )

# specify batch size and max epochs for training process
batch_size = best_config["batch_size"]
max_epochs = best_config["max_epochs"]

# start training the model
history = model.fit(
    x= [ecg_train, ppg_train],
    y = {
          "sbp_output": y_train_sbp_scaled,
          "dbp_output": y_train_dbp_scaled
        },
    validation_data = ([ecg_val, ppg_val],
    {
          "sbp_output": y_val_sbp_scaled,
          "dbp_output": y_val_dbp_scaled
          }),
     epochs= max_epochs,
     batch_size= batch_size,
     verbose=0, # Prevent showing epochs details in log
     callbacks=[early_stopping]
    )

# Save the trained model
model.save('encoder_transformer_5sec_fusionmidle.keras')

# Plot train and validation curve of the best model 
train_loss = history.history['sbp_output_mae_mmhg']
val_loss = history.history['val_sbp_output_mae_mmhg']
train_dbp = history.history['dbp_output_mae_mmhg']
val_dbp = history.history['val_dbp_output_mae_mmhg']
epochs = range(1, len(train_loss) + 1)

# plot the curve 
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig('encodertransformer_loss_curve.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.show()

# To prevent OOM (Out of Memory) Error, we remove the required arrays we do not need anymore from memory
del ecg_train, ppg_train, ecg_val, ppg_val, y_train_sbp_scaled, y_train_dbp_scaled, y_val_sbp_scaled, y_val_dbp_scaled 
del y_train_sbp, y_train_dbp, y_val_sbp, y_val_dbp
del y_train, y_val
gc.collect()

# Function to define Baland_altman
import matplotlib.pyplot as plt
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
        plt.savefig(f"timeseries_encodertransformer_Blnatm_{dataset_name}_{labels[k]}.png", dpi=300, bbox_inches='tight')
        plt.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

# Function to define the correlation plot of Estimated vs. references 
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
        plt.savefig(f"timeseries_encodertransformer_{dataset_name}_{labels[i]}.png", dpi=300, bbox_inches='tight')
        plt.close()

def test_mymodel(sbp_scaled, dbp_scaled, y_true_sbp, y_true_dbp, ecg,ppg, dataset_name="unknown"):

    # Evaluate the ECG and PPG test sets by model to estimate SBP and DBP
    y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict([ecg, ppg])

    # denormalize the scaled estimated values to mmHg scale
    y_pred_sbp = denormalize_labels_zscore(y_pred_sbp_scaled, sbp_mean, sbp_std)
    y_pred_dbp = denormalize_labels_zscore(y_pred_dbp_scaled, dbp_mean, dbp_std)    

    # to convert shape (N,1) into shape (N,)
    y_true_sbp = y_true_sbp[:, 0]
    y_true_dbp = y_true_dbp[:, 0]
    y_pred_sbp = y_pred_sbp[:, 0]
    y_pred_dbp = y_pred_dbp[:, 0]

    # ME (Mean Error)
    me_sbp = np.mean(y_pred_sbp - y_true_sbp)
    me_dbp = np.mean(y_pred_dbp - y_true_dbp)

    # MSE
    # MSE is calculated as a supplementary metric for
    # experimental logging and additional analysis.
    # The primary reported metrics are ME, MAE, and SDE.
    mse_sbp = mean_squared_error(y_pred_sbp, y_true_sbp)
    mse_dbp = mean_squared_error(y_pred_dbp, y_true_dbp)

    # MAE
    mae_sbp = mean_absolute_error(y_pred_sbp, y_true_sbp)
    mae_dbp = mean_absolute_error(y_pred_dbp, y_true_dbp)

    # SDE
    std_sbp = np.std(y_pred_sbp - y_true_sbp)
    std_dbp = np.std(y_pred_dbp - y_true_dbp)

    print(f"{dataset_name} - SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, STD={std_sbp:.2f}")
    print(f"{dataset_name} - DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, STD={std_dbp:.2f}")

    print(type(y_true_sbp), np.shape(y_true_sbp))
    print(type(y_pred_sbp), np.shape(y_pred_sbp))

    # Plot Estimated vs. Reference SBP
    plot_estimated_vs_reference_with_density(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name=dataset_name)

    # Compute Bland-Altman plots for SBP and DBP
    plot_bland_altman(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name= dataset_name)

# Evaluate model on test subsets
test_subsets = ["Test_CalBased", "Test_CalFree"]

for subset in test_subsets:
    # Load data
    ecg_test = np.load(myData + subset + '_ecg.npy', mmap_mode='r')
    ppg_test = np.load(myData + subset +'_ppg.npy', mmap_mode='r')
    abp_test = np.load(myData + subset +'_abp.npy', mmap_mode='r')
    y_test = np.load(myData + subset +'_Labels.npy' , mmap_mode='r')
    
    # convert (y,2) into (SBP,1) and (DBP,1)
    y_test_sbp = y_test[:, 0:1]
    y_test_dbp = y_test[:, 1:2]

    # Apply re-segmenteation on ECG, PPG, and ABP
    ecg_test_resegmented, ppg_test_resegmented, y_test_sbp_resegmented, y_test_dbp_resegmented = apply_resegmentation (
                                                                                                           5, 125,
                                                                                                           ecg_test, ppg_test, abp_test,
                                                                                                           y_test_sbp, y_test_dbp)
    # remove unnecessary arrays from memory
    del ecg_test, ppg_test, abp_test, y_test
    gc.collect()

    # convert (N, seq_len) into (N, seq_len, num_channels=1)
    ecg_test_resegmented = ecg_test_resegmented[..., np.newaxis]
    ppg_test_resegmented = ppg_test_resegmented[..., np.newaxis]

    # Apply z-score normalization on subsets' Labels
    y_test_sbp_scaled  = z_score_scale(y_test_sbp_resegmented, sbp_mean, sbp_std)
    y_test_dbp_scaled  = z_score_scale(y_test_dbp_resegmented, dbp_mean, dbp_std)
    
    # Evaluate on test
    test_mymodel(y_test_sbp_scaled, y_test_dbp_scaled,
                 y_test_sbp_resegmented, y_test_dbp_resegmented,
                 ecg_test_resegmented, ppg_test_resegmented, subset)
