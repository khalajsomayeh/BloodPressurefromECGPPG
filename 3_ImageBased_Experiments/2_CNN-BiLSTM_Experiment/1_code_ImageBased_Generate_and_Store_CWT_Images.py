# To make the code structure, definitions, and variables easier to understand, they have been designed to resemble the previous code as much as possible.

import os
subset = os.environ.get('SUBSET', 'train')

import numpy as np
import tensorflow as tf

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

myData='pulsedb_data_npy/'

# function to load data
def load_data():
    y_sbp = [0]
    y_dbp = [0]
    data_subset = np.load(myData + f'{subset}_{signal}.npy', mmap_mode='r')
    if signal=="abp":
       y = np.load(myData + f'{subset}_Labels.npy', mmap_mode='r')
       y_sbp = y[:, 0:1]
       y_dbp = y[:, 1:2]

    # training shape
    print(f'{signal} {subset} : {data_subset.shape}')
    if signal=="abp":
       print(f'y_sbp: {y_sbp.shape}')
       print(f'y_dbp: {y_dbp.shape}')

    return data_subset,y_sbp, y_dbp

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

    # 🔹 NEW: frequency-based scale construction
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

        # NEW: magnitude vs power
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

    # scalo_array = np.array(raw_scalos)  # (N, S, T)
    scalo_array = np.array(raw_scalos, dtype= np.float32)
    return scalo_array

def _make_cwt_images(X, cfg, fs):
    # cfg keys match function params -> pass through
    return signal_to_cwt_scalogram(X, cfg, fs)  # (N,S,T)

TRANSFORM_FNS = {
    "cwt":  _make_cwt_images
}

def normalize_images(ecg_or_ppg_img, vmin, vmax):

    # normalaize images of STFT and CWT
    if transform_type=='stft' or transform_type=='cwt':
        ecg_or_ppg_img = normalize_spectrogram(ecg_or_ppg_img, vmin, vmax)  # (N,H,W,1)

    return ecg_or_ppg_img

# 7. transform_type and hyperparams to generate images of ECG and PPG
def make_images_for_transform(transform_type,
                              ecg_or_ppg,
                              cfg,
                              fs
                              ):

    if transform_type not in TRANSFORM_FNS:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    fn = TRANSFORM_FNS[transform_type]
    ecg_or_ppg_img = fn(ecg_or_ppg, cfg, fs)    
    
    return ecg_or_ppg_img

#sequencing to prevent Out of Memory Error
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

# Resegmentation
import numpy as np

# Function to find peaks and troughs from ABP signal segment
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
            if i - last_peak >= min_distance:
                peak_idx.append(i)
                last_peak = i

        # local trough
        if x[i-1] > x[i] and x[i] < x[i+1]:
            if i - last_trough >= min_distance:
                trough_idx.append(i)
                last_trough = i

    return np.array(peak_idx), np.array(trough_idx)

# Shuffle samples 
def shuffle_array(array, idx):
    return array[idx]

# Function to recalculate SBP an DBP from peaks and troughs of ABP
def compute_sbp_dbp_from_abp_segment(abp_5s, min_distance=30):
    peak_idx, trough_idx = find_local_peaks_troughs(abp_5s, min_distance=min_distance)

    if len(peak_idx) == 0 or len(trough_idx) == 0:
        return np.nan, np.nan

    sbp = np.mean(abp_5s[peak_idx])
    dbp = np.mean(abp_5s[trough_idx])
    return sbp, dbp

# Function to define resegmentation
def apply_resegmentation_labels(seg_target, fs, abp_resegmented, y_sbp, y_dbp):

    if (seg_target==10):
        return y_sbp, y_dbp

    # label
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

    n  = fs * seg_target
    signal_first_half  = signal[:, :n]
    signal_second_half = signal[:, n:]

    # Stack first part and second part on top of each other , so number of samples will be doubled
    signal_resegmented = np.vstack([signal_first_half, signal_second_half])

    return signal_resegmented

#------------------------------------------------------------- resampling
import numpy as np
from scipy.signal import resample_poly

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

    # No resampling needed
    if fs_original == fs_target:
        return signal_segments

    # Rational resampling
    # signal_resampled = resample_poly(signal_segments, up=fs_target, down=fs_original, axis= axis) # axis is important to resample over segments not samples
    # The above code caused an Out Of Memory (OOM) Error. 
    # To prevent OOM, make batches and resample over batches
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

# The final best configuration found
active_image_configs = {
    "cwt": [
        {"wavelet": "morl", "n_scales": 64, "scale_mode": "log", "voices_per_octave": 12,
        "fmin_hz": 1, "fmax_hz": 10, "power": "power", "log_scale": True,
        "norm": None, "pad_mode": None, "use_coi_mask": False}
    ]
}
resampling_config = [32]
resegmentation_config =  [5]

# load data from the specified subset received as a parameter at the beginning of the code
signal_subset, y_sbp, y_dbp = load_data()

# Resampling
fs_target = resampling_config

# Apply resampling
signal_subset_resampled = apply_resampling(fs_target, signal_subset)
del signal_subset
gc.collect()

# Re-segmentation
seg_target = resegmentation_config

# Apply resegmentation
signal_subset_resampled_resegmented = apply_resegmentation_signal (seg_target, fs_target, signal_subset_resampled)
del signal_subset_resampled
gc.collect()

# Only if signal is ABP recalculate the labels
if signal=='abp':
     y_sbp_resegmented, y_dbp_resegmented = apply_resegmentation_labels (seg_target, fs_target, signal_subset_resampled_resegmented,y_sbp, y_dbp)
     del y_sbp, y_dbp
     gc.collect()

# Get final transformation method and its configuration 
transform_type, configs = next(iter(active_image_configs.items()))
cfg = configs[0]
    
# Load parameters for normalisation signals before transforming to images and saving them
save_dir = f"/scratch/username/{transform_type}_{fs_target}sr_{seg_target}sec_test/{subset}"
os.makedirs(save_dir, exist_ok=True)

# normalisation of ECG and PPG after resampling
if subset!="train":
   try:
      if signal in ('ecg', 'ppg'):
         signal_mean = np.load(f"{save_dir}/{signal}_mean.npy") # mean extracted from ecg signals
         signal_std  = np.load(f"{save_dir}/{signal}_std.npy")  #std extracted from ecg signals
         signal_vmin = np.load(f"{save_dir}/{signal}_vmin.npy") #vmin extracted from ecg images
         signal_vmax = np.load(f"{save_dir}/{signal}_vmax.npy") #vmax extracted from ecg images
      elif signal=="abp":
         sbp_mean = np.load(f"{save_dir}/sbp_mean.npy")
         sbp_std = np.load(f"{save_dir}/sbp_std.npy")
         dbp_mean = np.load(f"{save_dir}/dbp_mean.npy")
         dbp_std = np.load(f"{save_dir}/dbp_std.npy")

      print("Train normalization parameters loaded.")

   except Exception as e:
      print("ERROR: Train normalization parameters are not available. First call this code for train subset, then validation, after test subsets")
      print(e)
      raise SystemExit
elif subset=="train" and signal in ('ecg', 'ppg'):
   signal_mean = np.mean (signal_subset_resampled_resegmented)
   signal_std  = np.std (signal_subset_resampled_resegmented)

# Apply z-score normalization on only ecg and ppg
if signal in ('ecg', 'ppg'):
   signal_subset_resampled_resegmented_scaled = z_score_scale(signal_subset_resampled_resegmented, signal_mean, signal_std)
   del signal_subset_resampled_resegmented
   gc.collect()

# Z-score parameters from TRAIN ONLY
if subset=="train" and signal=="abp":
   sbp_mean = np.mean (y_sbp_resegmented)
   sbp_std  = np.std (y_sbp_resegmented)
   dbp_mean = np.mean (y_dbp_resegmented)
   dbp_std  = np.std (y_dbp_resegmented)

sig_len = fs_target *  seg_target

chunk_size = 126
# to prevent losing data of end of each chunk
# as data is  180432 the chnk siz can be =>16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 112, 126, 144, 168, 252, 336, 504
# To be sure that the prediction works well, the validation size % chunk size ==> 0 else 
#     this code makes errors: y_val_sbp_resegmented - y_pred_sbp  because those arrays are from different sizes
#start training
# training data generator due to OOM Error
def data_generator():
    if signal in ('ecg', 'ppg'):
         data_length = len(signal_subset_resampled_resegmented_scaled)

    elif signal == 'abp':
         data_length = len(y_sbp_resegmented)

    for i in range(0, data_length, chunk_size):
        if signal in ('ecg', 'ppg'):
           signal_chunk = signal_subset_resampled_resegmented_scaled [i:i+chunk_size]
           signal_img_chunk = make_images_for_transform (transform_type, signal_chunk, cfg, fs_target)
           if i == 0:
              print('data_chunk.shape:', signal_chunk.shape)
              print('signal_img_chunk.shape:', signal_img_chunk.shape)
              # for cwt when sampling rate=32 and segment length = 5 second the shape must be = (5, 64, 160, 1)
           yield {"signal": signal_img_chunk}


        elif signal=='abp':
           y_sbp_chunk = y_sbp_resegmented [i:i+chunk_size]
           y_dbp_chunk = y_dbp_resegmented [i:i+chunk_size]
           if i == 0:
              print('y_sbp_chunk.shape:', y_sbp_chunk.shape)
              print('y_dbp_chunk.shape:', y_dbp_chunk.shape)
           yield {
                "sbp": y_sbp_chunk,
                "dbp": y_dbp_chunk
            }
     
# generate images

# keep vmin vmax as the generator creates images
global_vmin = np.inf
global_vmax = -np.inf

signal_imgs = []
y_sbp_s = []
y_dbp_s = []
for batch_num, (batch) in enumerate(data_generator()):
    if signal in ('ecg', 'ppg'):
       signal_img = batch["signal"]
       signal_imgs.append(signal_img)
       # calculate vmin and vmax for each batch, to further find the global vmin-vmax
       if subset == "train":
          batch_vmin = signal_img.min()
          batch_vmax = signal_img.max()
          if batch_vmin < global_vmin:
             global_vmin = batch_vmin
          if batch_vmax > global_vmax:
             global_vmax = batch_vmax

    elif signal == 'abp':
         y_sbp = batch["sbp"]
         y_dbp = batch["dbp"]
         y_sbp_s.append(y_sbp)
         y_dbp_s.append(y_dbp)

if signal in ('ecg', 'ppg') and subset=="train":
   signal_vmin = float(global_vmin)
   signal_vmax = float(global_vmax)


# save data on disk
if subset == "train":
   if signal in ('ecg', 'ppg'):
      np.save(f"{save_dir}/{signal}_mean.npy", signal_mean)
      np.save(f"{save_dir}/{signal}_std.npy", signal_std)
      np.save(f"{save_dir}/{signal}_vmin.npy", signal_vmin)
      np.save(f"{save_dir}/{signal}_vmax.npy", signal_vmax)
   if signal=="abp":
      np.save(f"{save_dir}/sbp_mean.npy", sbp_mean)
      np.save(f"{save_dir}/sbp_std.npy", sbp_std)
      np.save(f"{save_dir}/dbp_mean.npy", dbp_mean)
      np.save(f"{save_dir}/dbp_std.npy", dbp_std)
print("Train normalization parameters saved.")


if signal in ('ecg', 'ppg'):
    for batch_num, signal_img in enumerate(signal_imgs):
        # normalize images
        normalized_img = normalize_images(signal_img, signal_vmin, signal_vmax)
        # save images on disk
        np.save(f"{save_dir}/{signal}_{subset}_{batch_num:05d}.npy", normalized_img)

elif signal == 'abp':
    for batch_num, (y_sbp, y_dbp) in enumerate(zip(y_sbp_s, y_dbp_s)):
        # Apply z-score normalization on subsets' Labels
        y_sbp_scaled = z_score_scale (y_sbp, sbp_mean, sbp_std)
        y_dbp_scaled = z_score_scale (y_dbp, dbp_mean, dbp_std)

        np.save(f"{save_dir}/sbp_{subset}_{batch_num:05d}.npy", y_sbp_scaled)
        np.save(f"{save_dir}/dbp_{subset}_{batch_num:05d}.npy", y_dbp_scaled)

print("All images/labels saved successfully.")
raise SystemExit
