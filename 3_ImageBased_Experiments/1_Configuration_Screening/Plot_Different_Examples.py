import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
import pywt
from pyts.image import GramianAngularField  # for GAF (GASF)

# ----------------------------
# Load data
# ----------------------------
subid = np.load('train_subjectids.npy')
ecg = np.load('train_ecg.npy').astype(np.float32) 
ppg = np.load('train_ppg.npy').astype(np.float32)

sample_index = 600
ecg_seg = ecg[sample_index]
ppg_seg = ppg[sample_index]

print("Subject ID:", subid[sample_index])
print("ECG seg shape:", ecg_seg.shape, "PPG seg shape:", ppg_seg.shape)


# Parameters for each signal-to-image transformation method

# STFT
WIN_LEN = 125
OVERLAP_PCT = 0.15
NOVERLAP = int(round(WIN_LEN * OVERLAP_PCT))  # ~19

#  The default sampling rate in PulseDB
FS = 125

# CWT
WAVELET = "morl"   # Morlet
N_SCALES = 64
SCALES = np.arange(1, N_SCALES + 1)

# GAF (GASF = summation)
gaf = GramianAngularField(method="summation")  # GASF

# Visibility Graph
VG_LEN = 256

# normalize to [0,1]

def minmax01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# Visibility Graph (Natural) -> adjacency matrix
def downsample_to_len(x, target_len=256):
    if len(x) == target_len:
        return x.copy()
    idx = np.linspace(0, len(x) - 1, target_len).astype(int)
    return x[idx]

def visibility_graph_adjacency(x):
    N = len(x)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            visible = True
            for k in range(i + 1, j):
                if x[k] >= x[i] + (x[j] - x[i]) * (k - i) / (j - i):
                    visible = False
                    break
            if visible:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


# Compute representations for ECG
# RAW: just plot

# STFT: magnitude spectrogram
f_e, t_e, Sxx_e = spectrogram(ecg_seg, fs=FS, nperseg=WIN_LEN, noverlap=NOVERLAP, mode="magnitude")
Sxx_e = np.log1p(Sxx_e)  # better contrast for display

# CWT: magnitude scalogram
cwt_e, freqs_e = pywt.cwt(ecg_seg, SCALES, WAVELET)
cwt_e = np.abs(cwt_e)

# GASF
gasf_e = gaf.fit_transform(ecg_seg.reshape(1, -1))[0]

# VG adjacency (downsample to 256)
ecg_ds = downsample_to_len(ecg_seg, VG_LEN)
vg_e = visibility_graph_adjacency(ecg_ds)

# Compute representations for PPG
f_p, t_p, Sxx_p = spectrogram(ppg_seg, fs=FS, nperseg=WIN_LEN, noverlap=NOVERLAP, mode="magnitude")
Sxx_p = np.log1p(Sxx_p)

cwt_p, freqs_p = pywt.cwt(ppg_seg, SCALES, WAVELET)
cwt_p = np.abs(cwt_p)

gasf_p = gaf.fit_transform(ppg_seg.reshape(1, -1))[0]

ppg_ds = downsample_to_len(ppg_seg, VG_LEN)
vg_p = visibility_graph_adjacency(ppg_ds)

# ----------------------------
# Plot: 5 rows x 2 columns  (ECG left, PPG right)
# (a,b) Raw
# (c,d) STFT
# (e,f) CWT
# (g,h) GASF
# (i,j) VG
# ----------------------------
fig, axes = plt.subplots(5, 2, figsize=(12, 16))

# (a) Raw ECG
axes[0, 0].plot(ecg_seg)
axes[0, 0].set_title("(a) ECG")
axes[0, 0].set_xlabel("Time step")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True, alpha=0.3)

# (b) Raw PPG
axes[0, 1].plot(ppg_seg)
axes[0, 1].set_title("(b) PPG")
axes[0, 1].set_xlabel("Time step")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].grid(True, alpha=0.3)

# (c) STFT ECG
im = axes[1, 0].imshow(Sxx_e, aspect="auto", origin="lower")
axes[1, 0].set_title(f"(c) STFT ECG (win={WIN_LEN}, overlap={int(OVERLAP_PCT*100)}%)")
axes[1, 0].set_xlabel("Time frames")
axes[1, 0].set_ylabel("Frequency bins")
fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

# (d) STFT PPG
im = axes[1, 1].imshow(Sxx_p, aspect="auto", origin="lower")
axes[1, 1].set_title(f"(d) STFT PPG (win={WIN_LEN}, overlap={int(OVERLAP_PCT*100)}%)")
axes[1, 1].set_xlabel("Time frames")
axes[1, 1].set_ylabel("Frequency bins")
fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

# (e) CWT ECG
im = axes[2, 0].imshow(cwt_e, aspect="auto", origin="lower")
axes[2, 0].set_title(f"(e) CWT ECG (Morlet, scales={N_SCALES})")
axes[2, 0].set_xlabel("Time steps")
axes[2, 0].set_ylabel("Scales")
fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

# (f) CWT PPG
im = axes[2, 1].imshow(cwt_p, aspect="auto", origin="lower")
axes[2, 1].set_title(f"(f) CWT PPG (Morlet, scales={N_SCALES})")
axes[2, 1].set_xlabel("Time steps")
axes[2, 1].set_ylabel("Scales")
fig.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)

# (g) GASF ECG
im = axes[3, 0].imshow(gasf_e, aspect="auto", origin="lower")
axes[3, 0].set_title("(g) GASF ECG (min–max, summation)")
axes[3, 0].set_xlabel("Time")
axes[3, 0].set_ylabel("Time")
fig.colorbar(im, ax=axes[3, 0], fraction=0.046, pad=0.04)

# (h) GASF PPG
im = axes[3, 1].imshow(gasf_p, aspect="auto", origin="lower")
axes[3, 1].set_title("(h) GASF PPG (min–max, summation)")
axes[3, 1].set_xlabel("Time")
axes[3, 1].set_ylabel("Time")
fig.colorbar(im, ax=axes[3, 1], fraction=0.046, pad=0.04)

# (i) Visibility Graph ECG
im = axes[4, 0].imshow(vg_e, aspect="auto", origin="lower")
axes[4, 0].set_title(f"(i) Visibility Graph ECG (downsampled length={VG_LEN})")
axes[4, 0].set_xlabel("Node index")
axes[4, 0].set_ylabel("Node index")
fig.colorbar(im, ax=axes[4, 0], fraction=0.046, pad=0.04)

# (j) Visibility Graph PPG
im = axes[4, 1].imshow(vg_p, aspect="auto", origin="lower")
axes[4, 1].set_title(f"(j) Visibility Graph PPG (downsampled length={VG_LEN})")
axes[4, 1].set_xlabel("Node index")
axes[4, 1].set_ylabel("Node index")
fig.colorbar(im, ax=axes[4, 1], fraction=0.046, pad=0.04)

plt.suptitle(f"Sample {sample_index} | Subject ID: {subid[sample_index]}", y=0.995)
plt.tight_layout()
plt.show()


# Save a high-resolution figure 
outpath = f"Sample_{sample_index}_ECG_PPG_4methods.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight")
print("Saved figure to:", outpath)
