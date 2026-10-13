import os
# this code receives some parameters to run 5-fold cross-validation on a specified parameter because doing it all in one code takes too much time

# Receive the parameters when the code is going to be executed
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

# Function to load or generate indices for a specific fold
def LoadIndices(fold_id, total_samples=100000):

    index_file = f'{myData}/indices_fold_{fold_id}_{total_samples}.npy'

    # Load indices if already saved
    if os.path.isfile(index_file):
        indices = np.load(index_file)

    else:
        # Generate and save indices
        samples_per_fold = total_samples // 5
        seed = 42

        ecg = np.load(myData + f'fold{fold_id}_ecg.npy', mmap_mode='r')
        n_samples = ecg.shape[0]
        rng = np.random.default_rng(seed + fold_id)
        indices = rng.choice(n_samples, size=samples_per_fold, replace=False)
        indices = np.sort(indices)
        np.save(index_file, indices)

    return indices

# Function to create 5 folds
def getdata_5folds(fold_id, total_samples=100000):
    '''
    parameters:
        fold_id: determines which fold number is validation. The rest of the folds are considered as training folds
        total_samples: in case when 5-fold is not run on the complete dataset, this parameter determines the total number of samples
                       in the subset. For example, 20,000 or 100,000 total samples.
                       This requires the indices to be already saved. If not, the function to generate random indices is called before.
    '''
    # prepare train and validation folds
    train_ecg_list = []
    train_ppg_list = []
    train_abp_list = []
    train_y_list = []

    # concatenate 4 training folds and specify validation fold
    for i in range(5):

        # load the indices of the fold i        
        indices = LoadIndices(i, total_samples)
        
        # the index "i" determines which fold is the validation fold
        if i == fold_id:
            ecg_val = np.load (myData + f'fold{i}_ecg.npy', mmap_mode='r') [indices]
            ppg_val = np.load (myData + f'fold{i}_ppg.npy', mmap_mode='r') [indices]
            abp_val = np.load (myData + f'fold{i}_abp.npy', mmap_mode='r') [indices]
            y_val = np.load (myData + f'fold{i}_labels.npy', mmap_mode='r') [indices]
        else:
            train_ecg_list.append (np.load(myData + f'fold{i}_ecg.npy', mmap_mode='r') [indices])
            train_ppg_list.append (np.load(myData + f'fold{i}_ppg.npy', mmap_mode='r') [indices])
            train_abp_list.append (np.load(myData + f'fold{i}_abp.npy', mmap_mode='r') [indices])
            train_y_list.append (np.load(myData + f'fold{i}_labels.npy', mmap_mode='r') [indices])

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


results = []

# Loop for config items
for transform_type, config_list in active_image_configs.items():
    for cfg in config_list:
        print(f"\n=== Transform: {transform_type}, config: {cfg} ===")

        # Apply 5-fold cross-validation on current config
        history_folds= []
        fold_metrics =[]
        for i in range(5):
            ecg_train, ppg_train, y_train_sbp, y_train_dbp, ecg_val,ppg_val, y_val_sbp, y_val_dbp = getdata_5folds(i, 100000)

            # 1) ECG+PPG
            train_ecg_img, train_ppg_img, val_ecg_img, val_ppg_img = make_images_for_transform(
                transform_type,
                ecg_train, ppg_train,
                ecg_val, ppg_val,
                cfg
            )

            fold_id = 1
            # Plot examples of ECG and PPG only for one fold, one example
            if (fold_id==1):
                idx= 800
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
                plt.savefig(f"myresults/examples/{transform_type}_{cfg_tag}_idx{idx}.png", dpi=300)
                plt.close()
                 
            # Z-score parameters from TRAIN ONLY
            sbp_mean = np.mean(y_train_sbp)
            sbp_std  = np.std(y_train_sbp)

            dbp_mean = np.mean(y_train_dbp)
            dbp_std  = np.std(y_train_dbp)
            
            # Apply z-score normalization on subsets' Labels for training and validation
            y_train_sbp_scaled = z_score_scale(y_train_sbp, sbp_mean, sbp_std)
            y_train_dbp_scaled = z_score_scale(y_train_dbp, dbp_mean, dbp_std)

            y_val_sbp_scaled   = z_score_scale(y_val_sbp, sbp_mean, sbp_std)
            y_val_dbp_scaled   = z_score_scale(y_val_dbp, dbp_mean, dbp_std)
                                    
            input_shape = train_ecg_img.shape[1:]  # (H,W,1)
            H, W, _ = input_shape
            patch_h, patch_w = choose_patch_hw(H, W)
            
            print(f"Transform: {transform_type}, cfg:{cfg},input_shape={input_shape}, patch(h,w)=({patch_h}, {patch_w})")
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
            loss_fn_sbp = mmhg_mae_loss(sbp_mean, sbp_std)
            loss_fn_dbp = mmhg_mae_loss(dbp_mean, dbp_std)
            
            mae_mmhg_sbp = mmhg_mae_metric(sbp_mean, sbp_std)
            mae_mmhg_dbp = mmhg_mae_metric(dbp_mean, dbp_std)
            model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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

            history_current_fold = model.fit(
                x={"ecg_input": train_ecg_img, "ppg_input": train_ppg_img}, 
	        y = {
	          "sbp_output": y_train_sbp_scaled,
	          "dbp_output": y_train_dbp_scaled
	        },
                validation_data=( {"ecg_input": val_ecg_img, "ppg_input": val_ppg_img}, 
	        {
	          "sbp_output": y_val_sbp_scaled,
	          "dbp_output": y_val_dbp_scaled
	        }),
                epochs=max_epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping]
            )         
        
            # To keep the history of all folds
            history_folds.append(history_current_fold)               

            y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict({"ecg_input": val_ecg_img, "ppg_input": val_ppg_img}, verbose=0)

            y_pred_sbp = denormalize_labels_zscore(y_pred_sbp_scaled, sbp_mean, sbp_std)
            y_pred_dbp = denormalize_labels_zscore(y_pred_dbp_scaled, dbp_mean, dbp_std)  
  
            # ME (Mean Error)
            me_sbp = np.mean(y_val_sbp - y_pred_sbp)
            me_dbp = np.mean(y_val_dbp - y_pred_dbp)
            
            # MSE 
            # MSE is calculated as a supplementary metric for
            # experimental logging and additional analysis.
            # The primary reported metrics are ME, MAE, and SDE. 
            mse_sbp = mean_squared_error(y_val_sbp, y_pred_sbp)
            mse_dbp = mean_squared_error(y_val_dbp, y_pred_dbp)

            # MAE
            mae_sbp = mean_absolute_error(y_val_sbp, y_pred_sbp)
            mae_dbp = mean_absolute_error(y_val_dbp, y_pred_dbp)

            # SDE
            std_sbp = np.std(y_val_sbp - y_pred_sbp)
            std_dbp = np.std(y_val_dbp - y_pred_dbp)

            print(f"fold{i} - SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, STD={std_sbp:.2f}")
            print(f"fold{i} - DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, STD={std_dbp:.2f}")

            fold_metric = {}

            fold_metrics.append({
                "fold": i,
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
        
            # Release memory allocated for this model
            tf.keras.backend.clear_session()
            del model, train_ecg_img, train_ppg_img, val_ecg_img, val_ppg_img
            gc.collect()

        # Check if not all 5 folds are executed due to error in configs
        if len(history_folds) < 5:
            print(f"[SKIP] cfg {cfg}: only {len(history_folds)}/5 folds trained")
            continue

        # Calculate the mean metrics obtained from all 5 folds  
        mean_metrics = {}
        for k in ["me_sbp","me_dbp","mae_sbp","mae_dbp","mse_sbp","mse_dbp","std_sbp","std_dbp"]:
            mean_metrics[k] = float(np.mean([m[k] for m in fold_metrics]))

        print(mean_metrics)

        # Calculate meanval loss    
        mean_val_mae_mmhg = float(np.mean([min(h.history["val_sbp_output_mae_mmhg"]) for h in history_folds]))

        results.append({
            "transform": transform_type,
            "config": cfg,
            "mean_val_mae_mmhg": mean_val_mae_mmhg,
            **mean_metrics
        })

# Select the best signal-to-image transform method and its transformation parameters
best_run = min(results, key=lambda x: x["mean_val_mae_mmhg"])
print("\nBEST CONFIG:")
print(best_run["config"])
for k, v in best_run["config"].items():
    print(f"{k}: {v}")
print("Mean CV val_mae_mmhg:", best_run["mean_val_mae_mmhg"])

