import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gc  # Garbage collection  to prevent memory usage error
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to define Z-score normalization for LABELS
def z_score_scale(y, mean, std):
    y_scaled = y.copy()
    y_scaled = (y_scaled - mean) / std
    return y_scaled

# Function to define Z-score De-normalization for LABELS
def denormalize_labels_zscore(y_scaled, mean, std):
    y = y_scaled.copy()
    y = y * std + mean
    return y

# convert [0-1] into mmHg in loss function to minimise error in scale of mmHg, not scales between [0,1]

# Function to MAE loss in mmHg with Z-score normalization
def mmhg_mae_loss(bp_mean, bp_std):

    def loss(y_true_scaled, y_pred_scaled):
        # de-normalize to mmHg
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    return loss

# function to define mae metrics when normaliztio nis Z-score
def mmhg_mae_metric(bp_mean, bp_std):

    def metric(y_true_scaled, y_pred_scaled):
        # de-normalize using Z-score method
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    metric.__name__ = "mae_mmhg"
    return metric

    ppg_train = ppg_train[:N_SUB]
    y_train_sbp   = y_train_sbp[:N_SUB]
    y_train_dbp   = y_train_dbp[:N_SUB]

    N_SUB = 5000
    ecg_val = ecg_val[:N_SUB]
    ppg_val = ppg_val[:N_SUB]
    y_val_sbp  = y_val_sbp[:N_SUB]
    y_val_dbp  = y_val_dbp[:N_SUB]

    #temporary

    #X_train = np.stack([ecg_train, ppg_train], axis=-1)  # (N_train, T, 2)
    #X_val   = np.stack([ecg_val,  ppg_val],   axis=-1)  # (N_val,   T, 2)      
    #return X_train, y_train, X_val, y_val
    return ecg_train, ppg_train, y_train_sbp,y_train_dbp, ecg_val, ppg_val, y_val_sbp, y_val_dbp

# Define Model------------------------------------------------------------
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class MultiHeadAttentionPooling1D(layers.Layer):
    def __init__(self, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.dim = input_shape[-1]  
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
        patch_size,
        projection_dim,
        transformer_layers,
        num_heads,
        encoder_mlp_units,
        head_mlp_units,
        dropout_rate
    ):
    
    if seq_len % patch_size != 0:
        raise ValueError("seq_len must be divisible by patch_size")

    input_ecg = Input(shape=(seq_len, num_channels), name="ecg_input")
    input_ppg = Input(shape=(seq_len, num_channels), name="ppg_input")

    #  1) Patch embedding ‌with Conv1D
    # ECG
    x_ecg = layers.Conv1D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embedding_ecg"
    )(input_ecg)

    #  PPG
    x_ppg = layers.Conv1D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embedding_ppg"
    )(input_ppg)

    #  x: (batch, num_patches, projection_dim)
    num_patches = seq_len // patch_size
    #  2) Positional embedding ‌
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_emb_ecg = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim,
        name="pos_embedding_ecg"
    )

    pos_emb_ppg = layers.Embedding(
        input_dim=num_patches,
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
        x3 = mlp_block(x3, hidden_units=encoder_mlp_units, dropout_rate=dropout_rate)
        x3 = layers.Dense(x2.shape[-1], name=f"mlp_proj_ecg_{i}")(x3)
        x_ecg = layers.Add(name=f"residual_mlp_ecg_{i}")([x2, x3])


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
        x3 = mlp_block(x3, hidden_units=encoder_mlp_units, dropout_rate=dropout_rate)
        x3 = layers.Dense(x2.shape[-1], name=f"mlp_proj_ppg_{i}")(x3)
        x_ppg = layers.Add(name=f"residual_mlp_ppg_{i}")([x2, x3])

    # 4) Layer Normalization
    x_ecg = layers.LayerNormalization(epsilon=1e-6, name="ln_final_ecg")(x_ecg)
    x_ppg = layers.LayerNormalization(epsilon=1e-6, name="ln_final_ppg")(x_ppg)

    # 5)   sequence →  vector
    x_ecg = MultiHeadAttentionPooling1D(num_heads=4, name="mh_attn_pool_ecg")(x_ecg)
    x_ppg = MultiHeadAttentionPooling1D(num_heads=4, name="mh_attn_pool_ppg")(x_ppg)
    
    x = layers.Concatenate(name="fusion_concat")([x_ecg, x_ppg])
    
    print('x:x_ech, x_ppg',x.shape)

    # shared learnable layer between SBP and DBP regression head
    x_shared = layers.Dense(1024, activation="relu")(x)
    x_shared = layers.Dense(512, activation="relu")(x_shared)

    # 6) Regression Head

    # Regression for SBP
    x_sbp = x_shared
    for j, units in enumerate(head_mlp_units):
        x_sbp = layers.Dense(units, activation="relu", name=f"head_dense_sbp_{j}")(x_sbp)

    # Regression for DBP
    x_dbp = x_shared
    for j, units in enumerate(head_mlp_units):
        x_dbp = layers.Dense(units, activation="relu", name=f"head_dense_dbp_{j}")(x_dbp)

    # Output layers
    output_sbp = layers.Dense(1, activation=None, name="sbp_output")(x_sbp)
    output_dbp = layers.Dense(1, activation=None, name="dbp_output")(x_dbp)

    model = Model(inputs=[input_ecg,input_ppg], outputs=[output_sbp,output_dbp], name="ViT_1D_BP_Regressor")
    return model

# Specify hyperparameters
all_hyper = {
    "patch_size":        [25], # must be division by 1250
    "num_heads":         [8], 
    "transformer_layers":[8], 
    "projection_dim":    [512], 
    "encoder_mlp_units": [(512,256)],
    "head_mlp_units":    [(1024,1024,512,256)], 
    "dropout_rate":      [0.02], 
    "lr":                [1e-4],
    "batch_size":        [128],
    "max_epochs":        [200],
}

hyper_keys = list(all_hyper.keys())
hyper_values = list(all_hyper.values())

all_configs = list(product(*hyper_values))
print(f"Total configurations: {len(all_configs)}")


tf.keras.utils.disable_interactive_logging()

# Load data
myData='pulsedb_data_npy/'

# laod training data
ecg_train = np.load(myData + 'train_ecg.npy')
ppg_train = np.load(myData + 'train_ppg.npy')
y_train = np.load(myData + 'train_Labels.npy')


# laod validation data
ecg_val = np.load(myData + 'val_ecg.npy')
ppg_val = np.load(myData + 'val_ppg.npy')
y_val = np.load(myData + 'val_Labels.npy')


# Load test subsets
# Load calibration-based test set
ecg_test_calbased = np.load( myData + 'Test_CalBased_ecg.npy')
ppg_test_calbased = np.load(myData + 'Test_CalBased_ppg.npy')
y_test_calbased = np.load(myData + 'Test_CalBased_Labels.npy')

# Load calibration-free test set
ecg_test_calfree = np.load(myData + 'Test_CalFree_ecg.npy')
ppg_test_calfree = np.load(myData + 'Test_CalFree_ppg.npy')
y_test_calfree = np.load(myData + 'Test_CalFree_Labels.npy')

# separate SBP and DBP labels from (y,2) into (SBP,1) and (DBP,1)
# training
y_train_sbp = y_train[:, 0:1]
y_train_dbp = y_train[:, 1:2]

#validation
y_val_sbp   = y_val[:, 0:1]
y_val_dbp   = y_val[:, 1:2]

# calibration-based 
y_test_calbased_sbp   = y_test_calbased[:, 0:1]
y_test_calbased_dbp   = y_test_calbased[:, 1:2]

# calibration-free 
y_test_calfree_sbp   = y_test_calfree[:, 0:1]
y_test_calfree_dbp   = y_test_calfree[:, 1:2]

# Remove outliers ----------------------------------------
'''
def iqr_bounds(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# --- IQR bounds from TRAIN only ---
sbp_lo, sbp_hi = iqr_bounds(y_train_sbp)
dbp_lo, dbp_hi = iqr_bounds(y_train_dbp)

print("SBP IQR bounds:", sbp_lo, sbp_hi)
print("DBP IQR bounds:", dbp_lo, dbp_hi)

# Apply outlier removal on sets
# Train
y_train_sbp = np.clip(y_train_sbp, sbp_lo, sbp_hi)
y_train_dbp = np.clip(y_train_dbp, dbp_lo, dbp_hi)

# Validation
y_val_sbp = np.clip(y_val_sbp, sbp_lo, sbp_hi)
y_val_dbp = np.clip(y_val_dbp, dbp_lo, dbp_hi)

# Test (both cal-based & cal-free)
y_test_calbased_sbp = np.clip(y_test_calbased_sbp, sbp_lo, sbp_hi)
y_test_calbased_dbp = np.clip(y_test_calbased_dbp, dbp_lo, dbp_hi)

y_test_calfree_sbp = np.clip(y_test_calfree_sbp, sbp_lo, sbp_hi)
y_test_calfree_dbp = np.clip(y_test_calfree_dbp, dbp_lo, dbp_hi)
'''
# --------------------------------------------------------
# get Z-score parameters from TRAIN ONLY
sbp_mean = np.mean(y_train_sbp)
sbp_std  = np.std(y_train_sbp)

dbp_mean = np.mean(y_train_dbp)
dbp_std  = np.std(y_train_dbp)

print("SBP mean/std:", sbp_mean, sbp_std)
print("DBP mean/std:", dbp_mean, dbp_std)

# Apply z-score normalization on subsets' Labels
y_train_sbp_scaled = z_score_scale(y_train_sbp, sbp_mean, sbp_std)
y_train_dbp_scaled = z_score_scale(y_train_dbp, dbp_mean, dbp_std)

y_val_sbp_scaled   = z_score_scale(y_val_sbp, sbp_mean, sbp_std)
y_val_dbp_scaled   = z_score_scale(y_val_dbp, dbp_mean, dbp_std)

y_test_calbased_sbp_scaled  = z_score_scale(y_test_calbased_sbp, sbp_mean, sbp_std)
y_test_calbased_dbp_scaled  = z_score_scale(y_test_calbased_dbp, dbp_mean, dbp_std)

y_test_calfree_sbp_scaled  = z_score_scale(y_test_calfree_sbp, sbp_mean, sbp_std)
y_test_calfree_dbp_scaled  = z_score_scale(y_test_calfree_dbp, dbp_mean, dbp_std)

# print the statistics parameters
print("SBP scaled mean/std:", y_train_sbp_scaled.mean(), y_train_sbp_scaled.std())   
print("DBP scaled mean/std:", y_train_dbp_scaled.mean(), y_train_dbp_scaled.std())

# Build model based on hyperparameter configs
seq_len = ecg_train.shape[1] 
num_channels = 1 

# setup before training

# device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print("Using device:", device)

# print configs of hyperparameters to keep their log 
for cfg_id, cfg in enumerate(all_configs, start=1):
    config = dict(zip(hyper_keys, cfg)) 
print (config)

# build model
with tf.device(device):
        model = build_vit_1d_regressor(
            seq_len=seq_len,
            num_channels=num_channels,
            patch_size=best_config["patch_size"],
            projection_dim=best_config["projection_dim"], 
            transformer_layers=best_config["transformer_layers"],
            num_heads=best_config["num_heads"],
            encoder_mlp_units=best_config["encoder_mlp_units"],
            head_mlp_units=best_config["head_mlp_units"],
            dropout_rate=best_config["dropout_rate"]
        )

# Clarify the function for SBP and DBP loss for the model
loss_fn_sbp = mmhg_mae_loss(sbp_mean, sbp_std)
loss_fn_dbp = mmhg_mae_loss(dbp_mean, dbp_std)

mae_mmhg_sbp = mmhg_mae_metric(sbp_mean, sbp_std)
mae_mmhg_dbp = mmhg_mae_metric(dbp_mean, dbp_std)

# compile the model to make it ready for training
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

# define early stopping and specify to the model to prevent over fitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_sbp_output_mae_mmhg',
    mode='min',
    patience=10,
    restore_best_weights=True
    )

batch_size = config["batch_size"]
max_epochs = config["max_epochs"]

# train the model on 80% train and 20% validation using the hyperparameters
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
     epochs=max_epochs,
     batch_size=batch_size,
     verbose=0, # Prevent showing epoch details in log
     callbacks=[early_stopping]
    )

# Save the trained model
model.save('patchbased_transformer_Finalbest.keras')

# Plot train and validation curve of the best model 
train_loss = history.history['sbp_output_mae_mmhg']
val_loss = history.history['val_sbp_output_mae_mmhg']
epochs = range(1, len(train_loss) + 1)

import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig('patchbased_transformer_loss_curve.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.show()


# function to define Baland_altman
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
        plt.savefig(f"1dvit_Blnatm_{dataset_name}_{labels[k]}.png", dpi=300, bbox_inches='tight')
        plt.close()

# Function to define the correlation between estimated SBP and DBP and their reference values
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
        plt.savefig(f"{dataset_name}_{labels[i]}_raw.png", dpi=300, bbox_inches='tight')
        plt.close()

# Function to evaluate the model on test subsets
def test_mymodel(sbp_scaled, dbp_scaled, y_true_sbp, y_true_dbp, ecg,ppg, dataset_name="unknown"):
    results = model.evaluate(
        [ecg, ppg],
        {
            "sbp_output": sbp_scaled,
            "dbp_output": dbp_scaled
        },
        verbose=0,
        return_dict=True
    )

    print(results)

    y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict([ecg, ppg])

    y_pred_sbp = denormalize_labels_zscore(y_pred_sbp_scaled, sbp_mean, sbp_std)
    y_pred_dbp = denormalize_labels_zscore(y_pred_dbp_scaled, dbp_mean, dbp_std)    

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

# Test on calibration-based test
test_mymodel(y_test_calbased_sbp_scaled, y_test_calbased_dbp_scaled,
             y_test_calbased_sbp, y_test_calbased_dbp,
             ecg_test_calbased, ppg_test_calbased, "Cal_Based_Test")

# Test on calibration-free test
test_mymodel(y_test_calfree_sbp_scaled, y_test_calfree_dbp_scaled,
             y_test_calfree_sbp, y_test_calfree_dbp,
             ecg_test_calfree, ppg_test_calfree, "Cal_Free_Test")

