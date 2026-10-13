import os
# this code receives some parameters to run 5-fold cross-validation on a specified parameter because doing it all in one code takes too much time

# Receive the parameters when the code is going to be executed
fold_id = int(os.environ.get('FOLD_ID', 0)) # 0,1,2,3,4
resampling_rate = int(os.environ.get('RESAMPLING', 0)) # 16,32,64,125
resegmentation_seconds = int(os.environ.get('RESEG', 0)) # 2, 5, 10
patch_h = int(os.environ.get('PATCH_H', 0)) #  8, 16, 25
patch_w = int(os.environ.get('PATCH_W', 0)) # 8, 16 , 25
model_name = os.environ.get('MODELNAME', 'CNN') # CNN, Vit

import numpy as np
import tensorflow as tf

# check available GPUs before training starts
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# import required libraries
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

# Function to define MAE metric
def mmhg_mae_metric(bp_mean, bp_std):

    def metric(y_true_scaled, y_pred_scaled):
        # de-normalize from Z-score
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    metric.__name__ = "mae_mmhg"
    return metric

# Folder containing PulseDB data
myData='pulsedb_data_npy/'

# Function to create 5 folds
def getdata_5folds(fold_id):
    # prepare train and validation folds
    train_ecg_list = []
    train_ppg_list = []
    train_abp_list = []
    train_y_list = []

    # concatenate 4 training folds and specify validation fold
    for i in range(5):
        
        # the index "i" determines which fold is the validation fold
        if i == fold_id:
            ecg_val = np.load (myData + f'fold{i}_ecg.npy', mmap_mode='r')
            ppg_val = np.load (myData + f'fold{i}_ppg.npy', mmap_mode='r')
            abp_val = np.load (myData + f'fold{i}_abp.npy', mmap_mode='r')
            y_val = np.load (myData + f'fold{i}_labels.npy', mmap_mode='r')
        else:
            train_ecg_list.append (np.load(myData + f'fold{i}_ecg.npy', mmap_mode='r'))
            train_ppg_list.append (np.load(myData + f'fold{i}_ppg.npy', mmap_mode='r'))
            train_abp_list.append (np.load(myData + f'fold{i}_abp.npy', mmap_mode='r'))
            train_y_list.append (np.load(myData + f'fold{i}_labels.npy', mmap_mode='r'))

    # convert train list to array
    ecg_train = np.concatenate(train_ecg_list, axis=0)
    ppg_train = np.concatenate(train_ppg_list, axis=0)
    abp_train = np.concatenate(train_abp_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    # convert (y,1) into (SBP,1) and (DBP,1)
    # for training
    y_train_sbp = y_train[:, 0:1]
    y_train_dbp = y_train[:, 1:2]

    # for validation
    y_val_sbp   = y_val[:, 0:1]
    y_val_dbp   = y_val[:, 1:2]

    # print training shape
    print(f'ecg train: {ecg_train.shape}')
    print(f'y_train_sbp: {y_train_sbp.shape}')

    # print validation shape
    print(f'ecg val: {ecg_val.shape}')
    print(f'y_val_sbp: {y_val_sbp.shape}')

    # remove the arrays from memory
    del y_val, y_train
    gc.collect()

    return ecg_train, ppg_train, abp_train, y_train_sbp, y_train_dbp, ecg_val, ppg_val, abp_val, y_val_sbp, y_val_dbp
    #temporary

# function to signal-to-image STFT
def signal_to_spectrogram(X, cfg, fs):
    nperseg = int(cfg["nperseg"])
    overlap = float(cfg["overlap"])
    window  = cfg.get("window", "hann")

    # specify overlap
    noverlap = int(round(nperseg * overlap))

    raw_specs = []
    f_list = []
    t_list = []

    for x in X:
        f, t, Sxx = spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap, window=window)
        raw_specs.append(Sxx)
        f_list.append(f)
        t_list.append(t)

    # Global normalization across all samples and all pixels
    spec_array = np.array(raw_specs, dtype=np.float32)
    f_array = np.array(f_list)
    t_array = np.array(t_list)
    return f_array, t_array, spec_array

def normalize_spectrogram (spec_array, spect_min,spect_max):
    # To avoid division by zero
    spec_normalized_array = (spec_array - spect_min) / (spect_max - spect_min + 1e-8)
    spec_normalized_array= np.expand_dims(spec_normalized_array, axis=-1)  # shape: (batch, H, W, 1)
    return spec_normalized_array

# 3. CWT → scalogram
def signal_to_cwt_scalogram(X, cfg, fs):
    if pywt is None:
        raise ImportError("pywt is not installed; CWT is unavailable.")

    # ---- read cfg with defaults ----
    wavelet_name = cfg.get("wavelet", "morl")
    n_scales = int(cfg.get("n_scales", 64))
    scale_mode = cfg.get("scale_mode", "linear")         # "linear" / "log"
    voices_per_octave = int(cfg.get("voices_per_octave", 12))  # used in log mode
    fmin_hz = float(cfg.get("fmin_hz", 0.5))
    fmax_hz = float(cfg.get("fmax_hz", 10.0))
    power = cfg.get("power", "magnitude")                # "magnitude" / "power"
    log_scale = bool(cfg.get("log_scale", False))
    norm = cfg.get("norm", "per_sample")                 # "per_sample" / None
    pad_mode = cfg.get("pad_mode", "reflect")            # "reflect" / None
    use_coi_mask = bool(cfg.get("use_coi_mask", False))

    raw_scalos = []

    # frequency-based scale construction
    if scale_mode == "log":
        freqs = np.geomspace(fmin_hz, fmax_hz, n_scales)
    else:
        freqs = np.linspace(fmin_hz, fmax_hz, n_scales)

    scales = pywt.central_frequency(wavelet_name) * fs / freqs

    for x in X:
        # padding
        if pad_mode is not None:
            pad_len = len(x) // 2
            x = np.pad(x, pad_len, mode=pad_mode)

        coeffs, freqs_out = pywt.cwt(x, scales, wavelet_name, sampling_period=1/fs)

        # remove padding
        if pad_mode is not None:
            coeffs = coeffs[:, pad_len:-pad_len]

        # magnitude vs power
        if power == "power":
            scalo = np.abs(coeffs) ** 2
        else:
            scalo = np.abs(coeffs)

        # log scaling
        if log_scale:
            scalo = np.log1p(scalo)

        # COI masking (simple version)
        if use_coi_mask:
            T = scalo.shape[1]
            for i, s in enumerate(scales):
                edge = int(np.ceil(s))
                scalo[i, :edge] = 0
                scalo[i, T-edge:] = 0

        # normalization
        if norm == "per_sample":
            min_v = scalo.min()
            max_v = scalo.max()
            scalo = (scalo - min_v) / (max_v - min_v + 1e-8)

        raw_scalos.append(scalo.astype(np.float32))

    # (N, S, T)
    scalo_array = np.array(raw_scalos, dtype= np.float32)
    return scalo_array

# 4. GAF → Gramian Angular Field
def signal_to_gaf_images(X, cfg):
    if GramianAngularField is None:
        raise ImportError("pyts is not installed; GAF is unavailable.")

    # ---- read cfg with defaults ----
    image_size     = int(cfg.get("image_size", 128))
    kind           = cfg.get("kind", "GASF")                 # "GASF" / "GADF"
    reduce_method  = cfg.get("reduce_method", "paa")         # "paa" / "resample"
    normalize      = cfg.get("normalize", "minmax")          # "minmax" / "zscore"
    feature_range  = cfg.get("feature_range", (-1, 1))       # only for minmax
    norm_scope     = cfg.get("norm_scope", "per_sample")     # "per_sample" / "global"
    smooth         = cfg.get("smooth", None)                 # dict or None
    signal_variant = cfg.get("signal_variant", "raw")        # "raw" / "diff1"

    Xp = np.asarray(X, dtype=np.float32)

    Xp = Xp.astype(np.float32)

    # signal variant
    if signal_variant == "diff1":
        Xp = np.diff(Xp, axis=1)
    elif signal_variant != "raw":
        raise ValueError(f"Unknown signal_variant: {signal_variant}")

    # smoothing
    if smooth is not None:
        stype = smooth.get("type", None)
        if stype == "savgol":
            from scipy.signal import savgol_filter
            win = int(smooth.get("window", 11))
            poly = int(smooth.get("poly", 3))
            # Savgol requires odd window and window > poly
            if win % 2 == 0:
                win += 1
            if win <= poly:
                win = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
            Xp = savgol_filter(Xp, window_length=win, polyorder=poly, axis=1)
        else:
            raise ValueError(f"Unknown smooth type: {stype}")

    # reduce length to image_size BEFORE GAF
    if reduce_method == "paa":
        # PAA: piecewise mean pooling to length = image_size
        n = Xp.shape[1]
        if n != image_size:
            # compute bin edges
            idx = np.linspace(0, n, num=image_size + 1).astype(int)
            Xp_reduced = []
            for row in Xp:
                pooled = np.array([row[idx[i]:idx[i+1]].mean() if idx[i] < idx[i+1] else row[min(idx[i], n-1)]
                                   for i in range(image_size)], dtype=np.float32)
                Xp_reduced.append(pooled)
            Xp = np.stack(Xp_reduced, axis=0)

    elif reduce_method == "resample":
        # resample each sample to length = image_size
        from scipy.signal import resample
        if Xp.shape[1] != image_size:
            Xp = resample(Xp, num=image_size, axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unknown reduce_method: {reduce_method}")

    # normalization
    if normalize == "minmax":
        a, b = feature_range
        if norm_scope == "per_sample":
            mn = Xp.min(axis=1, keepdims=True)
            mx = Xp.max(axis=1, keepdims=True)
            Xp = (Xp - mn) / (mx - mn + 1e-8)
            Xp = Xp * (b - a) + a
        elif norm_scope == "global":
            mn = Xp.min()
            mx = Xp.max()
            Xp = (Xp - mn) / (mx - mn + 1e-8)
            Xp = Xp * (b - a) + a
        else:
            raise ValueError(f"Unknown norm_scope: {norm_scope}")
        sample_range = feature_range  # tell pyts the true range

    elif normalize == "zscore":
        if norm_scope == "per_sample":
            mu = Xp.mean(axis=1, keepdims=True)
            sd = Xp.std(axis=1, keepdims=True)
            Xp = (Xp - mu) / (sd + 1e-8)
        elif norm_scope == "global":
            mu = Xp.mean()
            sd = Xp.std()
            Xp = (Xp - mu) / (sd + 1e-8)
        else:
            raise ValueError(f"Unknown norm_scope: {norm_scope}")
        Xp = np.tanh(Xp)
        sample_range = (-1, 1)  # tell pyts the true range

    else:
        raise ValueError(f"Unknown normalize: {normalize}")

    # map kind -> pyts method
    if kind == "GASF":
        method = "summation"
    elif kind == "GADF":
        method = "difference"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    gaf = GramianAngularField(
        image_size=image_size,
        method=method,
        sample_range=sample_range
    )

    images = gaf.fit_transform(Xp)          # (N, size, size)
    images = np.expand_dims(images, axis=-1) # (N, size, size, 1)
    return images


# 5. VG → Visibility Graph adjacency matrix
def visibility_graph_adjacency(
    signal,
    cfg=None,                  
    mode="natural",
    tie_break="allow",
    weighted=False,
    weight_mode="abs_diff"
):
    if cfg is not None:
        mode = cfg.get("graph_type", mode)      # graph_type in config
        tie_break = cfg.get("tie_break", tie_break)
        weighted = cfg.get("weighted", weighted)
        weight_mode = cfg.get("weight_mode", weight_mode)

    x = np.asarray(signal, dtype=np.float32)
    N = len(x)
    A = np.zeros((N, N), dtype=np.float32)
    t = np.arange(N, dtype=np.float32)

    # jitter for tie_break
    if tie_break == "jitter_eps":
        eps = 1e-6
        x = x + eps * np.random.uniform(-1.0, 1.0, size=N).astype(np.float32)
    elif tie_break not in ("allow", "strict"):
        raise ValueError(f"Unknown tie_break: {tie_break}")

    for i in range(N):
        for j in range(i + 1, N):
            visible = True
            for k in range(i + 1, j):
                if mode == "natural":
                    y_ik = x[i] + (x[j] - x[i]) * (t[k] - t[i]) / (t[j] - t[i])

                    # strict vs allow handling
                    if tie_break == "strict":
                        if x[k] >= y_ik:
                            visible = False
                            break
                    else:  # allow
                        if x[k] > y_ik:
                            visible = False
                            break

                else:  # horizontal
                    thr = min(x[i], x[j])
                    if tie_break == "strict":
                        if x[k] >= thr:
                            visible = False
                            break
                    else:  # allow
                        if x[k] > thr:
                            visible = False
                            break

            if visible:
                if not weighted:
                    w = 1.0
                else:
                    # weights
                    if weight_mode == "abs_diff":
                        w = float(abs(x[j] - x[i]))
                    elif weight_mode == "slope":
                        w = float(abs((x[j] - x[i]) / (t[j] - t[i] + 1e-8)))
                    else:
                        raise ValueError(f"Unknown weight_mode: {weight_mode}")

                A[i, j] = w
                A[j, i] = w
    return A

def signal_to_vg_images(X, cfg):
    from scipy.signal import detrend as sp_detrend
    # ---- read cfg with defaults ----
    graph_type = cfg.get("graph_type", "natural")               # natural / horizontal
    downsample_len = int(cfg.get("downsample_len", 256))
    downsample_method = cfg.get("downsample_method", "mean_pool")  # mean_pool / median_pool
    detrend = bool(cfg.get("detrend", False))
    zscore = bool(cfg.get("zscore", True))
    tie_break = cfg.get("tie_break", "allow")                  # allow / strict / jitter_eps
    weighted = bool(cfg.get("weighted", False))
    weight_mode = cfg.get("weight_mode", "abs_diff")           # abs_diff / slope
    repr_type = cfg.get("repr_type", "adjacency")                   # adjacency / laplacian

    X = np.asarray(X, dtype=np.float32)
    images = []

    for x in X:
        x1 = x.astype(np.float32)

        # detrend
        if detrend:
            x1 = sp_detrend(x1, type="linear")

        # zscore
        if zscore:
            mu = x1.mean()
            sd = x1.std()
            x1 = (x1 - mu) / (sd + 1e-8)

        # downsample by pooling (instead of picking indices)
        n = len(x1)
        if n != downsample_len:
            edges = np.linspace(0, n, num=downsample_len + 1).astype(int)
            pooled = np.empty((downsample_len,), dtype=np.float32)
            for i in range(downsample_len):
                a, b = edges[i], edges[i + 1]
                if a >= b:
                    pooled[i] = x1[min(a, n - 1)]
                else:
                    seg = x1[a:b]
                    if downsample_method == "mean_pool":
                        pooled[i] = seg.mean()
                    elif downsample_method == "median_pool":
                        pooled[i] = np.median(seg)
                    else:
                        raise ValueError(f"Unknown downsample_method: {downsample_method}")
            x_ds = pooled
        else:
            x_ds = x1

        A = visibility_graph_adjacency(
            x_ds,
            cfg=cfg
        )

        # representation choice
        if repr_type == "adjacency":
            M = A
        elif repr_type == "laplacian":
            D = np.diag(A.sum(axis=1))
            M = D - A
        else:
            raise ValueError(f"Unknown repr: {repr_type}")

        images.append(M)

    images = np.array(images, dtype=np.float32)      # (N, L, L)
    images = np.expand_dims(images, axis=-1)         # (N, L, L, 1)
    return images

# define functions that generally receive all configs and, based on the transformation method, call the corresponding method 
def _make_stft_images(X, cfg, fs):
    _, _, raw = signal_to_spectrogram(X, cfg, fs)
    return raw

def _make_cwt_images(X, cfg, fs):
    # cfg keys match function params -> pass through
    return signal_to_cwt_scalogram(X, cfg, fs)  # (N,S,T)


def _make_gaf_images(X, cfg, fs):
    # cfg keys match function params -> pass through
    return signal_to_gaf_images(X, cfg)  # already (N,H,W,1)

def _make_vg_images(X, cfg, fs):
    # cfg keys match function params -> pass through
    return signal_to_vg_images(X, cfg)  # (N,L,L,1)

TRANSFORM_FNS = {
    "stft": _make_stft_images,
    "cwt":  _make_cwt_images,
    "gaf":  _make_gaf_images,
    "vg":   _make_vg_images,
}

# 7. transform_type and hyperparams to generate images of ECG and PPG
def make_images_for_transform(transform_type,
                              ecg_or_ppg,
                              cfg,
                              fs,
                              vmin = None, vmax= None
                              ):

    if transform_type not in TRANSFORM_FNS:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    fn = TRANSFORM_FNS[transform_type]

    ecg_or_ppg_img = fn(ecg_or_ppg, cfg, fs)
    # normalise images of STFT and CWT
    if transform_type=='stft' or transform_type=='cwt':
        if vmin is None or vmax is None:
           vmin, vmax = ecg_or_ppg_img.min(), ecg_or_ppg_img.max()

        ecg_or_ppg_img = normalize_spectrogram(ecg_or_ppg_img, vmin, vmax)  # (N,H,W,1)

    return ecg_or_ppg_img, vmin, vmax

# Define Model: CNN + BiLSTM
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def CNN_BiLSTM_2D(ecg_img_shape, ppg_img_shape,head_mlp_units):

    # Inputs
    ecg_input = Input(shape=ecg_img_shape, name='ecg_input')
    ppg_input = Input(shape=ppg_img_shape, name='ppg_input')

    # ECG CNN
    x1 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ecg_input)
    x1 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x1)
    
    # Instead of max pooling that decreases suddenly, reducing is learnable
    x1 = layers.Conv2D(64,  (5,5), strides=(2,1), padding='same', activation='relu') (x1) 
    x1 = layers.Conv2D(64,  (5,5), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(256, (5,5), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(256, (5,5), strides=(2,1), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(128, (5,5), padding ='same', activation='relu')(x1) 
    x1 = layers.Conv2D(128, (5,5), strides=(2,1), padding='same', activation='relu')(x1) 
    
    # PPG CNN
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ppg_input)
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x2)

    # Instead of max pooling that decreases suddenly, reducing is learnable
    x2 = layers.Conv2D(64,  (7,7), strides=(2,1), padding='same', activation='relu') (x2) 
    x2 = layers.Conv2D(64,  (7,7), padding='same', activation='relu')(x2)
    x2 = layers.Conv2D(256, (7,7), padding='same', activation='relu') (x2)
    x2 = layers.Conv2D(256, (7,7), strides=(2,1), padding='same', activation='relu') (x2)
    x2 = layers.Conv2D(128, (5,5), strides = (2,1), padding ='same', activation='relu')(x2)
    x2 = layers.Conv2D(128, (3,3), strides = (1,1), padding ='same', activation='relu') (x2)

    # This block: from (batch, frequency, time, channels)  ==> (batch, time, frequency, channels) ==> (batch, time, frequency * channels)

    # Merge features
    x = layers.Concatenate(axis=-1)([x1, x2]) # (batch,H, W, C)
    # set time dimension as main idea  Put time dimension in axis 1
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]))(x)   # (batch, W, H, C)
    # make features from  frequency and channels to achieve (batch, time, features)
    # Flatten freq and channels into features
    x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2] * tf.shape(t)[3])))(x) # (batch, time, features)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    # Shared learnable Fully connected layers
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

# Define Model: Vision Transformer
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class MultiHeadAttentionPooling1D(layers.Layer):
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads
        })
        return config
def mlp_block(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def get_2d_sincos_pos_embed(h, w, dim):
    '''
    Returns: (h*w, dim) float32
    dim must be divisible by 4.
    '''
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sin/cos embed, got dim={dim}")

    h = int(h)
    w = int(w)

    grid_h = tf.range(h, dtype=tf.float32)
    grid_w = tf.range(w, dtype=tf.float32)
    yy, xx = tf.meshgrid(grid_h, grid_w, indexing="ij")  # (h,w)

    # each axis gets dim/2, and within that: sin+cos => dim/4 frequencies
    dim_each = dim // 2
    omega_dim = dim_each // 2  # = dim/4

    omega = tf.range(omega_dim, dtype=tf.float32)
    omega = 1.0 / tf.pow(10000.0, omega / tf.cast(omega_dim, tf.float32))  # (dim/4,)

    # (h,w,dim/4)
    out_y = tf.einsum("hw,d->hwd", yy, omega)
    out_x = tf.einsum("hw,d->hwd", xx, omega)

    pos_emb = tf.concat(
        [tf.sin(out_y), tf.cos(out_y), tf.sin(out_x), tf.cos(out_x)],
        axis=-1
    )  # (h,w,dim)

    return tf.reshape(pos_emb, (h * w, dim))

def build_vit_2d_regressor(input_shape,
                        patch_h = 16,
                        patch_w = 10,
                        projection_dim=64,
                        transformer_layers=4,
                        num_heads=4,
                        mlp_units=(128, 64),
                        head_mlp_units=(128, 64),
                        dropout_rate=0.1):
    
    input_ecg = Input(shape=input_shape, name="ecg_input")  # (H,W,1)
    input_ppg = Input(shape=input_shape, name="ppg_input")  # (H,W,1)

    # Patch embedding: Conv2D with kernel=stride=patch_size  
    # ECG patch embedding
    x_ecg = layers.Conv2D(filters=projection_dim,
                      kernel_size=(patch_h, patch_w),
                      strides=(patch_h, patch_w),
                      padding="valid",
                      name = 'patch_embeddin_ecg')(input_ecg)

    # PPG patch embedding
    x_ppg = layers.Conv2D(filters=projection_dim,
                      kernel_size=(patch_h, patch_w),
                      strides=(patch_h, patch_w),
                      padding="valid",
                      name = 'patch_embeddin_ppg')(input_ppg)

    # shape: (batch, H_p, W_p, projection_dim)
    H, W, _= input_shape
    H_p = H // patch_h
    W_p = W // patch_w
    num_patches = H_p * W_p

    x_ecg = layers.Reshape((num_patches, projection_dim))(x_ecg)
    x_ppg = layers.Reshape((num_patches, projection_dim))(x_ppg)

    # Positional embedding
    pos_embedding_ecg = get_2d_sincos_pos_embed(H_p, W_p, projection_dim)
    x_ecg = x_ecg + pos_embedding_ecg

    pos_embedding_ppg = get_2d_sincos_pos_embed(H_p, W_p, projection_dim)
    x_ppg = x_ppg + pos_embedding_ppg

    # Transformer encoders for ECG
    for i in range(transformer_layers):
        # LayerNorm + MHA
        x1 = layers.LayerNormalization(epsilon=1e-6)(x_ecg)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=projection_dim//num_heads,
                                                     dropout=dropout_rate,
                                                     name =f"mha_ecg_{i}")(x1, x1)
        x2 = layers.Add(name=f"residual_attn_ecg_{i}")([attention_output, x_ecg])

        # MLP block
        x3 = layers.LayerNormalization(epsilon=1e-6, name= f"ln_mlp_ecg_{i}")(x2)
        x3 = mlp_block(x3, hidden_units=mlp_units, dropout_rate=dropout_rate)
        x3 = layers.Dense(x2.shape[-1], name =f"mlp_proj_ecg_{i}")(x3)
        x_ecg = layers.Add(name=f"residual_mlp_ecg_{i}")([x2, x3])


    # Transformer encoders for PPG
    for i in range(transformer_layers):
        # LayerNorm + MHA
        x1 = layers.LayerNormalization(epsilon=1e-6)(x_ppg)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=projection_dim//num_heads,
                                                     dropout=dropout_rate,
                                                     name =f"mha_ppg_{i}")(x1, x1)
        x2 = layers.Add(name=f"residual_attn_ppg_{i}")([attention_output, x_ppg])

        # MLP block
        x3 = layers.LayerNormalization(epsilon=1e-6, name= f"ln_mlp_ppg_{i}")(x2)
        x3 = mlp_block(x3, hidden_units=mlp_units, dropout_rate=dropout_rate)
        x3 = layers.Dense(x2.shape[-1], name =f"mlp_proj_ppg_{i}")(x3)
        x_ppg = layers.Add(name=f"residual_mlp_ppg_{i}")([x2, x3])

    x_ecg = layers.LayerNormalization(epsilon=1e-6,name =f"ln_final_ecg")(x_ecg)
    x_ppg = layers.LayerNormalization(epsilon=1e-6,name =f"ln_final_ppg")(x_ppg)

    # convert (B, T, D) into (B, D)
    # T= number of patch tokens
    # project dimension
    # this code does token aggregation, instead of GlobalAveragePooling1D
    x_ecg = MultiHeadAttentionPooling1D(num_heads=4, name="mh_attn_pool_ecg")(x_ecg)
    x_ppg = MultiHeadAttentionPooling1D(num_heads=4, name="mh_attn_pool_ppg")(x_ppg)

    # Fusion concatenation
    x = layers.Concatenate(name="fusion_concat")([x_ecg, x_ppg])

    # shared learnable fully conncted between SBP and DBP
    x_shared = layers.Dense(1024, activation="relu")(x)
    x_shared = layers.Dense(512, activation="relu")(x_shared)
    
    # 6) Regression Head
    # SBP branch
    x_sbp = x_shared
    for j, units in enumerate(head_mlp_units):
        x_sbp = layers.Dense(units, activation="relu", name=f"head_dense_sbp_{j}")(x_sbp)

    # DBP branch
    x_dbp = x_shared
    for j, units in enumerate(head_mlp_units):
        x_dbp = layers.Dense(units, activation="relu", name=f"head_dense_dbp_{j}")(x_dbp)

    output_sbp = layers.Dense(1, activation=None, name="sbp_output")(x_sbp)
    output_dbp = layers.Dense(1, activation=None, name="dbp_output")(x_dbp)
    model = Model(inputs=[input_ecg,input_ppg], outputs=[output_sbp,output_dbp], name="ViT_2D_BP_Regressor")
    return model

# Sequencing to prevent Out of Memory Error
from tensorflow.keras.utils import Sequence

class BPImageSequence(Sequence):
    def __init__(self,
                 ecg, ppg,
                 y_sbp, y_dbp,
                 batch_size,
                 transform_type,
                 cfg,
                 fs,
                 ecg_vmin, ecg_vmax,
                 ppg_vmin, ppg_vmax,
                 sbp_mean, sbp_std,
                 dbp_mean, dbp_std,
                 shuffle=True):

        super().__init__()
        self.ecg = ecg
        self.ppg = ppg
        self.y_sbp = y_sbp
        self.y_dbp = y_dbp
        self.batch_size = batch_size
        self.transform_type = transform_type
        self.cfg = cfg
        self.fs = fs
        self.ecg_vmin = ecg_vmin
        self.ecg_vmax = ecg_vmax
        self.ppg_vmin = ppg_vmin
        self.ppg_vmax = ppg_vmax

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std

        self.shuffle = shuffle
        self.indices = np.arange(len(self.y_sbp))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        ecg_batch = self.ecg[batch_idx]
        ppg_batch = self.ppg[batch_idx]
        y_sbp_batch = self.y_sbp[batch_idx]
        y_dbp_batch = self.y_dbp[batch_idx]

        ecg_img, _, _ = make_images_for_transform(
            self.transform_type, ecg_batch, self.cfg, self.fs,
            self.ecg_vmin, self.ecg_vmax
        )
        ppg_img, _, _ = make_images_for_transform(
            self.transform_type, ppg_batch, self.cfg, self.fs,
            self.ppg_vmin, self.ppg_vmax
        )

        ecg_img = ecg_img.astype(np.float32)
        ppg_img = ppg_img.astype(np.float32)

        y_sbp_scaled = z_score_scale(y_sbp_batch, self.sbp_mean, self.sbp_std).astype(np.float32)
        y_dbp_scaled = z_score_scale(y_dbp_batch, self.dbp_mean, self.dbp_std).astype(np.float32)

        return (
            {"ecg_input": ecg_img, "ppg_input": ppg_img},
            {"sbp_output": y_sbp_scaled, "dbp_output": y_dbp_scaled}
        )

def compute_global_minmax(X, cfg,fs, transform_type, batch_size=128):
    global_min = np.inf
    global_max = -np.inf

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = X[start:end]

        if transform_type == "stft":
            imgs = _make_stft_images(xb, cfg, fs)
        elif transform_type == "cwt":
            imgs = _make_cwt_images(xb, cfg, fs)
        elif transform_type == "gaf":
            imgs = _make_gaf_images(xb, cfg, fs)
        elif transform_type == "vg":
            imgs = _make_vg_images(xb, cfg, fs)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

        batch_min = imgs.min()
        batch_max = imgs.max()

        if batch_min < global_min:
            global_min = batch_min
        if batch_max > global_max:
            global_max = batch_max

        del imgs
        gc.collect()

    return float(global_min), float(global_max)
# When the size of images are different in 5-fold cross validation, we cannot specify a fixed patch size
# because it may be too small or large, so we define a flexible way to select the patch size being selected based on the image size
def choose_patch_hw(H, W) :
    best_ph = H
    best_pw = W
    # Height
    if (H<=32):
       m = H-1 # min(H, 25)
    else:
       m = 32
    # 5 because in spectrogram we have 12*65 which and 6 is good for example. 25 has been best patch so far
    for p in range(m, 4, -1):  # from 25 to 5
        if H % p == 0:
            best_ph = p
            break

    # Width
    if (W<32):
       m = W-1
    else:
       m = 32
    # 5 because in spectrogram we have 12*65 which and 6 is good for example. 25 has beeen best patch so far
    for p in range(m, 4, -1):  # from 25 to 5
        if W % p == 0:
            best_pw = p
            break
    return best_ph, best_pw

# Function to Resegmentation
import numpy as np

def find_local_peaks_troughs(signal, min_distance=30):
    '''
    Detect local peaks and troughs using shape only:
    peak   : x[i-1] < x[i] > x[i+1]
    trough : x[i-1] > x[i] < x[i+1]

    min_distance prevents detecting too many nearby extrema.
    '''
    x = np.asarray(signal, dtype=np.float32)

    peak_idx = []
    trough_idx = []

    last_peak = -min_distance
    last_trough = -min_distance

    for i in range(1, len(x) - 1):
        # local peak
        if x[i-1] < x[i] and x[i] > x[i+1]:
            if i - last_peak >= min_distance:
                peak_idx.append(i)
                last_peak = i

        # local trough
        if x[i-1] > x[i] and x[i] < x[i+1]:
            if i - last_trough >= min_distance:
                trough_idx.append(i)
                last_trough = i

    return np.array(peak_idx), np.array(trough_idx)

# Function to shuffle data 
def shuffle_array(array, idx):
    return array[idx]

# Function to recalculate the SBP and DBP from ABP segments
def compute_sbp_dbp_from_abp_segment(abp, min_distance=30):
    min_distance = max(1, int(round(0.4 * fs)))
    peak_idx, trough_idx = find_local_peaks_troughs(abp, min_distance=min_distance)

    if len(peak_idx) == 0 or len(trough_idx) == 0:
        return np.nan, np.nan

    sbp = np.mean(abp[peak_idx])
    dbp = np.mean(abp[trough_idx])
    return sbp, dbp

# Function to apply resegmentation labels
def apply_resegmentation_labels(seg_target, fs, abp_resegmented, y_sbp, y_dbp):

    # 10-second segments are the default long in PulseDB, so we skip this step
    if (seg_target==10):
        return y_sbp, y_dbp

    # labels
    y_new_sbp = np.vstack([y_sbp, y_sbp])
    y_new_dbp = np.vstack([y_dbp, y_dbp])
    
    for i in range(y_new_sbp.shape[0]):
        sbp, dbp = compute_sbp_dbp_from_abp_segment (abp_resegmented[i])
        if (not np.isfinite(sbp)) or (not np.isfinite(dbp)) or (sbp < 0) or (dbp < 0) or (sbp < dbp):
              continue  # keep the sbp and dbp of pulsedb
        else:
              y_new_sbp[i] = sbp  # re-assign sbp fro new-size segment
              y_new_dbp[i] = dbp
    
    return y_new_sbp, y_new_dbp

# Function to apply resegmentation on signal; the signal can be ECG, PPG, or ABP when this function is called
def apply_resegmentation_signal(seg_target, fs, signal):
    '''
    ecg: (N, 1250)
    ppg: (N, 1250)
    y  : (N, 2)  or (N, 1)

    returns:
    ecg_new, ppg_new, y_new
    with shapes:
    (2N, 625), (2N, 625), (2N, ...)
    '''

    if (seg_target==10):
        return signal
    print("resegmentation started")
    #ecg.shape[1] // sub_segment_time  # example 1250/2

    n  = fs * seg_target
    signal_first_half  = signal[:, :n]
    signal_second_half = signal[:, n:]

    # Stack first part and second part on top of each other , so number of samples will be doubled
    signal_resegmented = np.vstack([signal_first_half, signal_second_half]) #np.concatenate([ecg_first, ecg_second], axis=0)

    return signal_resegmented

#------------------------------------------------------------- resampling
import numpy as np
from scipy.signal import resample_poly

# Function to apply resampling on input signal; the signal can be ECG, PPG, or ABP
def apply_resampling(
    fs_target,
    signal_segments):
    '''
    Resample a batch of signals from fs_original to fs_target.

    Parameters
    ----------
    X : np.ndarray
        Input signals.
        Examples:
            (N, L) for one signal type
            (N, C, L) if multiple channels
    fs_original : int
        Original sampling rate.
    fs_target : int
        Target sampling rate.
    segment_seconds : int or float
        Duration of each segment in seconds.
    axis : int
        Axis corresponding to time dimension.
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    signal_resampled : np.ndarray
        Resampled signals with target length = fs_target * segment_seconds
    '''
    axis = 1
    dtype = np.float32
    fs_original = 125
    segment_seconds = signal_segments.shape[1] // fs_original
    target_len = int(round(fs_target * segment_seconds))
    original_len_expected = int(round(fs_original * segment_seconds))

    # No resampling needed if target sampling rate=125; which is the default sampling rate in PulseDB
    if fs_original == fs_target:
        return signal_segments

    # Rational resampling
    # signal_resampled = resample_poly(signal_segments, up=fs_target, down=fs_original, axis= axis) # axis is important to resample over segments not samples
    # The above code caused OOM Error. 
    # To prevent OOM make batches and resample over batches
    print("resampling started")
    n_samples = signal_segments.shape[0]
    batch_size = 100
    signal_resampled = np.empty((n_samples, target_len), dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch = signal_segments[start:end].astype(np.float32)

        batch_resampled = resample_poly(
            batch,
            up=fs_target,
            down=fs_original,
            axis=axis
        )
        signal_resampled[start:end] = batch_resampled.astype(np.float32)

    # Safety: force exact target length
    current_len = signal_resampled.shape[axis]
    if current_len != target_len:
        slicer = [slice(None)] * signal_resampled.ndim
        if current_len > target_len:
            slicer[axis] = slice(0, target_len)
            signal_resampled = signal_resampled[tuple(slicer)]
        else:
            pad_width = [(0, 0)] * signal_resampled.ndim
            pad_width[axis] = (0, target_len - current_len)
            signal_resampled = np.pad(signal_resampled, pad_width=pad_width, mode="edge")

    return signal_resampled.astype(dtype, copy=False)

# Specify candidate hyperparameters and configurations
active_image_configs = {
    # STFT configurations
    "stft": [
        {"nperseg": 64,  "overlap": 0.25, "window": "hann"},
        {"nperseg": 64,  "overlap": 0.25, "window": "hamming"},
        {"nperseg": 64,  "overlap": 0.50, "window": "hann"},
        {"nperseg": 64,  "overlap": 0.50, "window": "hamming"},
        {"nperseg": 64,  "overlap": 0.15, "window": "hamming"},
        {"nperseg": 128, "overlap": 0.15, "window": "hann"},  
        {"nperseg": 128, "overlap": 0.25, "window": "hann"},
        {"nperseg": 128, "overlap": 0.25, "window": "blackman"},
        {"nperseg": 128, "overlap": 0.50, "window": "hann"},
        {"nperseg": 128, "overlap": 0.50, "window": "blackman"},
        {"nperseg": 256, "overlap": 0.15, "window": "hann"},
        {"nperseg": 256, "overlap": 0.25, "window": "hann"},
        {"nperseg": 256, "overlap": 0.25, "window": "blackman"},
        {"nperseg": 256, "overlap": 0.50, "window": "hann"},
        {"nperseg": 256, "overlap": 0.50, "window": "blackman"},
        {"nperseg": 512, "overlap": 0.25, "window": "hann"},
        {"nperseg": 512, "overlap": 0.50, "window": "blackman"},
    ],
    # CWT configurations
    "cwt": [
        {"wavelet": "morl", "n_scales": 64, "scale_mode": "log", "voices_per_octave": 12,
        "fmin_hz": 0.5, "fmax_hz": 10, "power": "power", "log_scale": True,
        "norm": None, "pad_mode": "reflect", "use_coi_mask": True},

        {"wavelet": "mexh", "n_scales": 64, "scale_mode": "linear",
        "fmin_hz": 0.5, "fmax_hz": 10, "power": "magnitude", "log_scale": False,
        "norm": None, "pad_mode": "reflect", "use_coi_mask": False},

        {"wavelet": "cmor1.5-1.0", "n_scales": 64, "scale_mode": "log", "voices_per_octave": 12,
        "fmin_hz": 0.5, "fmax_hz": 10, "power": "power", "log_scale": True,
        "norm": None, "pad_mode": "reflect", "use_coi_mask": True},
    ],

    # GAF configurations
    "gaf": [
        {"kind": "GASF", "image_size": 128, "reduce_method": "paa",
        "normalize": "zscore", "feature_range": (-1, 1), "norm_scope": "global", #before was minmax per_sample",
        "smooth": None, "signal_variant": "raw"},

        {"kind": "GADF", "image_size": 128, "reduce_method": "paa",
        "normalize": "zscore", "feature_range": (-1, 1), "norm_scope": "global", #before was minmax "per_sample",
        "smooth": {"type":"savgol","window":11,"poly":3}, "signal_variant": "raw"},

        {"kind": "GASF", "image_size": 224, "reduce_method": "resample",
        "normalize": "zscore", "norm_scope": "global",
        "smooth": None, "signal_variant": "diff1"},
    ],

    # Visibility Graph configurations
    "vg": [
        {"graph_type": "natural", "downsample_len": 128, "downsample_method": "mean_pool",
        "detrend": True, "zscore": True, "tie_break": "jitter_eps",
        "weighted": False, "repr_type": "adjacency"},

        {"graph_type": "horizontal", "downsample_len": 128, "downsample_method": "median_pool",
        "detrend": True, "zscore": True, "tie_break": "strict",
        "weighted": False, "repr_type": "adjacency"},

        {"graph_type": "natural", "downsample_len": 256, "downsample_method": "mean_pool",
        "detrend": False, "zscore": True, "tie_break": "allow",
        "weighted": True, "weight_mode": "slope", "repr_type": "adjacency"},

        {"graph_type": "natural", "downsample_len": 256, "downsample_method": "mean_pool",
        "detrend": True, "zscore": True, "tie_break": "jitter_eps",
        "weighted": True, "weight_mode": "abs_diff", "repr_type": "laplacian"},
    ]
}

resampling_config = [ resampling_rate] 
resegmentation_config =  [resegmentation_seconds] 

results = []
history_folds= []
fold_metrics =[]

# Apply 5-fold cross-validation on current config
# Due to OOM errors, we changed the code to be called for each fold separately, each resampling rate, resegmentation and model separately
# To prevent changes in the rest of the code, "i = fold_id" is added
# In case OOM does not happen, we can change the 1 to 5 and remove the "fold_id" parameter from the beginning of the code 
for i in range(1):    
    i= fold_id
    # Create folds and make data ready for cross-validation
    # i presents the validation fold number
    ecg_train, ppg_train, abp_train, y_train_sbp, y_train_dbp, ecg_val, ppg_val, abp_val, y_val_sbp, y_val_dbp = getdata_5folds(i)
    
    # loop for resampling list
    for fs_target in resampling_config:
        
        # apply resampling
        ecg_train_resampled = apply_resampling(fs_target, ecg_train)
        
        # to prevent OOM:
        # del no-need objects
        del ecg_train
        gc.collect()

        ppg_train_resampled = apply_resampling(fs_target, ppg_train)
        del ppg_train
        gc.collect()

        abp_train_resampled = apply_resampling(fs_target, abp_train)
        del abp_train
        gc.collect()

        ecg_val_resampled = apply_resampling(fs_target, ecg_val)
        del ecg_val
        gc.collect()

        ppg_val_resampled = apply_resampling(fs_target, ppg_val)
        del ppg_val
        gc.collect()

        abp_val_resampled = apply_resampling(fs_target, abp_val)
        del abp_val
        gc.collect()

        # loop for re-segmentation
        for seg_target in resegmentation_config:
            
            # apply resegmentation on training set
            
            # ECG
            ecg_train_resampled_resegmented = apply_resegmentation_signal ( seg_target, fs_target, ecg_train_resampled)
            del ecg_train_resampled
            gc.collect()

            # PPG
            ppg_train_resampled_resegmented = apply_resegmentation_signal ( seg_target, fs_target, ppg_train_resampled)
            del ppg_train_resampled
            gc.collect()

            # ABP
            abp_train_resampled_resegmented = apply_resegmentation_signal ( seg_target, fs_target, abp_train_resampled)
            del abp_train_resampled
            gc.collect()

            # SBP and DBP
            y_train_sbp_resegmented, y_train_dbp_resegmented = apply_resegmentation_labels (seg_target, fs_target, abp_train_resampled_resegmented, y_train_sbp, y_train_dbp)
            
            del y_train_sbp, y_train_dbp, abp_train_resampled_resegmented
            gc.collect()

            # apply resegmentation on validation set

            # ECG
            ecg_val_resampled_resegmented = apply_resegmentation_signal ( seg_target, fs_target, ecg_val_resampled)
            del ecg_val_resampled
            gc.collect()

            # PPG
            ppg_val_resampled_resegmented = apply_resegmentation_signal ( seg_target, fs_target, ppg_val_resampled)
            del ppg_val_resampled
            gc.collect()

            # ABP
            abp_val_resampled_resegmented = apply_resegmentation_signal ( seg_target, fs_target, abp_val_resampled)
            del abp_val_resampled
            gc.collect()

            # SBP and DBP
            y_val_sbp_resegmented, y_val_dbp_resegmented = apply_resegmentation_labels (seg_target, fs_target, abp_val_resampled_resegmented,y_val_sbp, y_val_dbp)
            del y_val_sbp, y_val_dbp, abp_val_resampled_resegmented
            gc.collect()

            # shuffle data after resegmentation
            # shuffle ONLY in training set
            idx = np.random.permutation(len(y_train_sbp_resegmented))
            ecg_train_resampled_resegmented = shuffle_array(ecg_train_resampled_resegmented, idx)
            ppg_train_resampled_resegmented = shuffle_array(ppg_train_resampled_resegmented, idx)
            y_train_sbp_resegmented = shuffle_array(y_train_sbp_resegmented, idx)
            y_train_dbp_resegmented = shuffle_array(y_train_dbp_resegmented, idx)

            # normalisation ECG and PPG signals after resampling
            ecg_mean = np.mean (ecg_train_resampled_resegmented)
            ecg_std  = np.std (ecg_train_resampled_resegmented)
            ppg_mean = np.mean (ppg_train_resampled_resegmented)
            ppg_std  = np.std (ppg_train_resampled_resegmented)

            # Apply z-score normalization on ECG training set
            ecg_train_resampled_resegmented_scaled = z_score_scale(ecg_train_resampled_resegmented, ecg_mean, ecg_std)
            del ecg_train_resampled_resegmented
            gc.collect()

            # Apply z-score normalization on PPG training set
            ppg_train_resampled_resegmented_scaled = z_score_scale(ppg_train_resampled_resegmented, ppg_mean, ppg_std)
            del ppg_train_resampled_resegmented
            gc.collect()

            # Apply z-score normalization on ECG validation set
            ecg_val_resampled_resegmented_scaled = z_score_scale(ecg_val_resampled_resegmented, ecg_mean, ecg_std)
            del ecg_val_resampled_resegmented
            gc.collect()

            # Apply z-score normalization on PPG validation set
            ppg_val_resampled_resegmented_scaled = z_score_scale(ppg_val_resampled_resegmented, ppg_mean, ppg_std)
            del ppg_val_resampled_resegmented
            gc.collect()

            # Get Z-score parameters from TRAIN ONLY from SBP and DBP (labels)
            sbp_mean = np.mean (y_train_sbp_resegmented)
            sbp_std  = np.std (y_train_sbp_resegmented)

            dbp_mean = np.mean (y_train_dbp_resegmented)
            dbp_std  = np.std (y_train_dbp_resegmented)

            # Apply z-score normalization on training SBP
            y_train_sbp_resegmented_scaled = z_score_scale (y_train_sbp_resegmented, sbp_mean, sbp_std)
            del y_train_sbp_resegmented
            gc.collect()

            # Apply z-score normalization on training DBP
            y_train_dbp_resegmented_scaled = z_score_scale (y_train_dbp_resegmented, dbp_mean, dbp_std)
            del y_train_dbp_resegmented
            gc.collect()
            
            # Apply z-score normalization on validation SBP; based on the statistics parameters from training
            y_val_sbp_resegmented_scaled  = z_score_scale(y_val_sbp_resegmented, sbp_mean, sbp_std)

            # Apply z-score normalization on validation DBP; based on the statistics parameters from training
            y_val_dbp_resegmented_scaled  = z_score_scale(y_val_dbp_resegmented, dbp_mean, dbp_std)

            # Loop for config items specific for each signal-to-image transformation already defined in "active_image_configs"
            for transform_type, config_list in active_image_configs.items():
                for cfg in config_list:
                    sig_len = fs_target *  seg_target

                    # recognise invalid configurations
                    if transform_type == "stft" and cfg["nperseg"] > sig_len:
                         print(f"Skip invalid STFT cfg for sig_len={sig_len}: {cfg}")
                         continue # skip this invalid combination of configuration
                    
                    # find min max values for image normalization over the entire training data; global normalisation
                    
                    # ECG
                    train_ecg_vmin, train_ecg_vmax = compute_global_minmax (ecg_train_resampled_resegmented_scaled, cfg, fs_target, transform_type, batch_size=100)

                    # PPG
                    train_ppg_vmin, train_ppg_vmax = compute_global_minmax (ppg_train_resampled_resegmented_scaled, cfg, fs_target, transform_type, batch_size=100)

                    # only transform the first sample to obtain the shape of data further in code
                    train_ecg_img, _, _ = make_images_for_transform (transform_type, ecg_train_resampled_resegmented_scaled[1:3], cfg, fs_target, train_ecg_vmin, train_ecg_vmax)
                    #if transform_type=="stft" and train_ecg_img.shape[1]< cfg["nperseg"]:
                    #    continue
                    train_ppg_img, _, _ = make_images_for_transform (transform_type, ppg_train_resampled_resegmented_scaled[1:3], cfg, fs_target, train_ppg_vmin, train_ppg_vmax)

                    # Plot examples of ECG and PPG only for one fold, one example

                    if (fold_id==0):
                        idx= 0
                        ecg_img = train_ecg_img[idx, :, :]
                        ppg_img = train_ppg_img[idx, :, :]

                        plt.figure(figsize=(6,3))
                        plt.subplot(1,2,1)
                        plt.imshow(ecg_img, cmap="jet"); plt.title("ECG"); plt.axis("off")

                        plt.subplot(1,2,2)
                        plt.imshow(ppg_img, cmap="jet"); plt.title("PPG"); plt.axis("off")

                        plt.suptitle(f"{transform_type} | {cfg}")
                        plt.tight_layout()
                        cfg_tag = "_".join([f"{k}-{v}" for k, v in cfg.items()])
                        plt.savefig(f"img_fs{fs_target}_seg{seg_target}_{transform_type}.png", dpi=300)
                        plt.close()

                    input_shape = train_ecg_img.shape[1:]  # (H,W,1)
                    H, W, _ = input_shape

                    # In cases when the patch needs to be automatically selected, uncomment this code
                    #patch_h, patch_w = choose_patch_hw(H, W)
                    
                    # In ECG/PPG, the time dimension (W=160) is usually more important than the frequency dimension (H=64). 
                    # That's why we thought this option is even more than  8x16 (h*w):
                    print ("patch_h=", patch_h, ", patch_w=", patch_w)

                    # define the device  that runs the code
                    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'                    
                    if (model_name=='CNN'):
                         with tf.device(device):
                             model = CNN_BiLSTM_2D(
                             ecg_img_shape= train_ecg_img.shape[1:],
                             ppg_img_shape= train_ppg_img.shape[1:],
                             head_mlp_units=[1024, 1024, 512, 256]
                         )
                    else:
                         with tf.device(device):
                             model = build_vit_2d_regressor(
                                input_shape=input_shape,
                                patch_h = patch_h,
                                patch_w = patch_w,
                                projection_dim=512,
                                transformer_layers=4,
                                num_heads=8,
                                head_mlp_units= (1024, 1024, 512, 256),
                                mlp_units=(512, 256),
                                dropout_rate=0.02
                         )

                    # Clarify loss functions for the models
                    loss_fn_sbp = mmhg_mae_loss(sbp_mean, sbp_std)
                    loss_fn_dbp = mmhg_mae_loss(dbp_mean, dbp_std)

                    # Clarify MAE function for the models
                    mae_mmhg_sbp = mmhg_mae_metric(sbp_mean, sbp_std)
                    mae_mmhg_dbp = mmhg_mae_metric(dbp_mean, dbp_std)

                    if (model_name=='CNN'):
                        # H, W, C = ecg_img_shape
                        # kh1, kw1 = safe_kernel_size(H, W, (3, 3))
                        lr = 1e-4
                    else:
                        lr = 1e-4
                    model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
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

                    # Early stopping to prevent overfitting
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_sbp_output_mae_mmhg',
                        #monitor ='val_loss',
                        mode='min',
                        patience=10,
                        restore_best_weights=True
                    )
                   
                    # Due to OOM errors, the next code is added
                    chunk_size = 126
                    # to prevent losing data of end of each chunk
                    # as data is  180432 the chnk siz can be =>16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 112, 126, 144, 168, 252, 336, 504
                    # To be sure that the prediction works well the validation size % chunk size ==> 0 else 
                    #     this code makes errors: y_val_sbp_resegmented - y_pred_sbp  because those arrays are from different sizes
                    #start training
                    # training data generator due to OOM Error
                    def train_generator():
                        for i in range(0, len(ecg_train_resampled_resegmented_scaled), chunk_size):
                            train_ecg_chunk = ecg_train_resampled_resegmented_scaled [i:i+chunk_size]
                            train_ppg_chunk = ppg_train_resampled_resegmented_scaled [i:i+chunk_size]
                            if i == 0:
                                print('train_ecg_chunk.shape:', train_ecg_chunk.shape)
                                print('train_ppg_chunk.shape:', train_ppg_chunk.shape)

                            y_sbp_scaled_chunk = y_train_sbp_resegmented_scaled [i:i+chunk_size]
                            y_dbp_scaled_chunk = y_train_dbp_resegmented_scaled [i:i+chunk_size]

                            train_ecg_img_chunk,_,_ = make_images_for_transform (transform_type, train_ecg_chunk, cfg, fs_target, train_ecg_vmin, train_ecg_vmax)
                            train_ppg_img_chunk,_,_ = make_images_for_transform (transform_type, train_ppg_chunk, cfg, fs_target, train_ppg_vmin, train_ppg_vmax)
                            # for cwt when sampling rate=32 and segment length = 5 second the shape must be = (5, 64, 160, 1)
                            if i == 0:
                                print('train_ecg_img_chunk.shape:', train_ecg_img_chunk.shape)
                                print('train_ppg_img_chunk.shape:', train_ppg_img_chunk.shape)

                            yield (
                                {"ecg_input": train_ecg_img_chunk, "ppg_input": train_ppg_img_chunk},
                                {"sbp_output": y_sbp_scaled_chunk, "dbp_output": y_dbp_scaled_chunk}
                            )

                    train_dataset = tf.data.Dataset.from_generator(
                        train_generator,
                        output_signature=(
                            {
                                "ecg_input": tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                                "ppg_input": tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                            },
                            {
                                "sbp_output": tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                                "dbp_output": tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                            }
                        )
                    )
                    
                    # validation generator due to OOM Error
                    def val_generator():
                        for i in range(0, len(ecg_val_resampled_resegmented_scaled), chunk_size):
                            val_ecg_chunk = ecg_val_resampled_resegmented_scaled [i:i+chunk_size]
                            val_ppg_chunk = ppg_val_resampled_resegmented_scaled [i:i+chunk_size]
                            if i == 0:
                                print('val_ecg_chunk.shape:', val_ecg_chunk.shape)
                                print('val_ppg_chunk.shape:', val_ppg_chunk.shape)

                            y_sbp_scaled_chunk = y_val_sbp_resegmented_scaled [i:i+chunk_size]
                            y_dbp_scaled_chunk = y_val_dbp_resegmented_scaled [i:i+chunk_size]

                            val_ecg_img_chunk,_,_ = make_images_for_transform (transform_type, val_ecg_chunk, cfg, fs_target, train_ecg_vmin, train_ecg_vmax)
                            val_ppg_img_chunk,_,_ = make_images_for_transform (transform_type, val_ppg_chunk, cfg, fs_target, train_ppg_vmin, train_ppg_vmax)
                            # for cwt when sampling rate=32 and segment length = 5 second the shape must be = (5, 64, 160, 1)
                            if i == 0:
                                print('val_ecg_img_chunk.shape:', val_ecg_img_chunk.shape)
                                print('val_ppg_img_chunk.shape:', val_ppg_img_chunk.shape)

                            yield (
                                {"ecg_input": val_ecg_img_chunk, "ppg_input": val_ppg_img_chunk},
                                {"sbp_output": y_sbp_scaled_chunk, "dbp_output": y_dbp_scaled_chunk}
                            )

                    val_dataset = tf.data.Dataset.from_generator(
                        val_generator,
                        output_signature=(
                            {
                                "ecg_input": tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                                "ppg_input": tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                            },
                            {
                                "sbp_output": tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                                "dbp_output": tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                            }
                        )
                    )
                    
                    import math

                    training_steps_per_epoch = int (math.ceil(len(ecg_train_resampled_resegmented_scaled) / chunk_size))
                    val_steps = int (math.ceil(len(ecg_val_resampled_resegmented_scaled) / chunk_size))

                    # start training the model
                    max_epochs= 100
                    history = model.fit(
                        train_dataset,
                        steps_per_epoch=training_steps_per_epoch,
                        epochs=max_epochs,
                        validation_data=val_dataset,
                        validation_steps=val_steps,
                        #validation_data = (
                        #    {"ecg_input": val_ecg_img, "ppg_input": val_ppg_img},
                        #    {"sbp_output": y_val_sbp_resegmented_scaled, "dbp_output": y_val_dbp_resegmented_scaled}
                        #),
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    # To keep the history of all folds
                    # history_folds.append(history_current_fold)

                    #y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict({"ecg_input": val_ecg_img, "ppg_input": val_ppg_img}, verbose=0)
                    y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict (val_dataset, steps=val_steps, verbose=0)

                    y_pred_sbp = denormalize_labels_zscore(y_pred_sbp_scaled, sbp_mean, sbp_std)
                    y_pred_dbp = denormalize_labels_zscore(y_pred_dbp_scaled, dbp_mean, dbp_std)

                    #y_val_sbp = y_val_sbp[:, 0]
                    #y_val_dbp = y_val_dbp[:, 0]
                    #y_pred_sbp = y_val_sbp[:, 0]
                    #y_pred_dbp = y_val_dbp[:, 0]

                    # ME (Mean Error)
                    me_sbp = np.mean(y_val_sbp_resegmented - y_pred_sbp)
                    me_dbp = np.mean(y_val_dbp_resegmented - y_pred_dbp)

                    # MSE
                    # MSE is calculated as a supplementary metric for
                    # experimental logging and additional analysis.
                    # The primary reported metrics are ME, MAE, and SDE.
                    mse_sbp = mean_squared_error(y_val_sbp_resegmented, y_pred_sbp)
                    mse_dbp = mean_squared_error(y_val_dbp_resegmented, y_pred_dbp)

                    # MAE
                    mae_sbp = mean_absolute_error(y_val_sbp_resegmented, y_pred_sbp)
                    mae_dbp = mean_absolute_error(y_val_dbp_resegmented, y_pred_dbp)

                    # SDE
                    std_sbp = np.std(y_val_sbp_resegmented - y_pred_sbp)
                    std_dbp = np.std(y_val_dbp_resegmented - y_pred_dbp)

                    fold_metrics.append({
                        "fold": i,
                        "fs_target": fs_target,
                        "seg_target": seg_target,
                        "transform": transform_type,
                        "hyperparams": cfg,
                        "me_sbp": me_sbp,
                        "me_dbp": me_dbp,
                        "mae_sbp": mae_sbp,
                        "mae_dbp": mae_dbp,
                        "mse_sbp": mse_sbp,
                        "mse_dbp": mse_dbp,
                        "std_sbp": std_sbp,
                        "std_dbp": std_dbp
                    })                    
                    print(f"\n====================================================================================")
                    print(f"\n Fold:{i}\n train_ecg_shape:{train_ecg_img.shape}\n sampling rate:{fs_target}\n segment length:{seg_target}\n Transform method: {transform_type}\n config: {cfg}")
                    print(f"SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, SDE={std_sbp:.2f}")
                    print(f"DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, SDE={std_dbp:.2f}")

                    # In scenarios where there are no memory errors, and the entire five-fold cross-validation within this code is executed in a single run,                       # the following lines of code can be uncommented for further out-of-memory issues. 
                    
                    # Release memory allocated for this model
                    #tf.keras.backend.clear_session()
                    #del model, train_ecg_img, train_ppg_img, val_ecg_img, val_ppg_img
                    #gc.collect()

            #del ecg_train_resampled_resegmented_scaled, ppg_train_resampled_resegmented_scaled
            #del ecg_val_resampled_resegmented_scaled, ppg_val_resampled_resegmented_scaled
            #del y_train_sbp_resegmented_scaled, y_train_dbp_resegmented_scaled, y_val_sbp_resegmented_scaled, y_val_dbp_resegmented_scaled
            #del y_train_sbp_resegmented, y_train_dbp_resegmented, y_val_sbp_resegmented, y_val_dbp_resegmented
            #gc.collect()

        #delete no-need objects
        #del ecg_train_resampled, ppg_train_resampled, abp_train_resampled, ecg_val_resampled, ppg_val_resampled, abp_val_resampled
        #gc.collect()
    #del no-need objects
    #del ecg_train, ppg_train, abp_train, ecg_val, ppg_val, abp_val
    #del y_train_sbp, y_train_dbp, y_val_sbp, y_val_dbp
    #gc.collect()
    
