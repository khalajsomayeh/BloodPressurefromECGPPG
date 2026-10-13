import os
batch_size = 256

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
# split subject-wise functions
import os
import glob
import numpy as np


def iter_selected_from_subset(
    image_root,
    subset,
    selected_indices
):
    """
    Read OLD train/val chunks and return only selected samples.

    Parameters
    ----------
    image_root:
        Main directory containing train/ and val/

    subset:
        "train" or "val"

    selected_indices:
        Global 5-sec indices belonging to the desired NEW subset.
        Example:
        train_old_to_new_train.npy
    """

    selected_indices = np.asarray(selected_indices)
    selected_indices = np.sort(selected_indices)

    data_dir = image_root #os.path.join(image_root, subset)

    ecg_files = sorted(
        glob.glob(
            os.path.join(data_dir, f"ecg_{subset}_*.npy")
        )
    )

    ppg_files = sorted(
        glob.glob(
            os.path.join(data_dir, f"ppg_{subset}_*.npy")
        )
    )

    sbp_files = sorted(
        glob.glob(
            os.path.join(data_dir, f"sbp_{subset}_*.npy")
        )
    )

    dbp_files = sorted(
        glob.glob(
            os.path.join(data_dir, f"dbp_{subset}_*.npy")
        )
    )

    assert len(ecg_files) > 0, f"No ECG files found for {subset}"

    assert (
        len(ecg_files)
        == len(ppg_files)
        == len(sbp_files)
        == len(dbp_files)
    ), "Number of ECG/PPG/SBP/DBP chunks does not match."

    global_start = 0

    for ecg_file, ppg_file, sbp_file, dbp_file in zip(
        ecg_files,
        ppg_files,
        sbp_files,
        dbp_files
    ):

        # فقط shape را می‌خوانیم
        ecg_mmap = np.load(ecg_file, mmap_mode="r")
        n_samples = ecg_mmap.shape[0]

        global_end = global_start + n_samples

        # پیدا کردن selected indexهایی که داخل این chunk هستند
        left = np.searchsorted(
            selected_indices,
            global_start,
            side="left"
        )

        right = np.searchsorted(
            selected_indices,
            global_end,
            side="left"
        )

        if right > left:

            # global index → local index inside current chunk
            local_indices = (
                selected_indices[left:right]
                - global_start
            )

            ecg = np.asarray(
                ecg_mmap[local_indices],
                dtype=np.float32
            )

            ppg = np.asarray(
                np.load(ppg_file, mmap_mode="r")[local_indices],
                dtype=np.float32
            )

            sbp = np.asarray(
                np.load(sbp_file, mmap_mode="r")[local_indices],
                dtype=np.float32
            )

            dbp = np.asarray(
                np.load(dbp_file, mmap_mode="r")[local_indices],
                dtype=np.float32
            )

            yield ecg, ppg, sbp, dbp

        global_start = global_end


# ---------------------------------------------------------
# STEP 2: Test reading OLD TRAIN samples for NEW TRAIN
# ---------------------------------------------------------

# must print
# Expected selected samples from old train: 1150000 bayad mosavi bashand
#Actually read samples from old train: 1150000
#OLD TRAIN -> NEW TRAIN check passed.
image_root = "/scratch/skhalaj/cwt_32sr_5sec"

# indices that belong to NEW TRAIN, coming from OLD TRAIN
train_old_to_new_train = np.load(
    "mydata/train_old_to_new_train.npy"
)

print(
    "Expected selected samples from old train:",
    len(train_old_to_new_train)
)
'''
count = 0

for ecg, ppg, sbp, dbp in iter_selected_from_subset(
    image_root=image_root,
    subset="train",
    selected_indices=train_old_to_new_train
):

    count += len(ecg)

print(
    "Actually read samples from old train:",
    count
)

assert count == len(train_old_to_new_train)

print("OLD TRAIN -> NEW TRAIN check passed.")
'''
# ---------------------------------------------------------
# STEP 3: Test reading OLD VAL samples for NEW TRAIN
# ---------------------------------------------------------

val_old_to_new_train = np.load(
    "mydata/val_old_to_new_train.npy"
)

print(
    "Expected selected samples from old val:",
    len(val_old_to_new_train)
)

'''
count_val_to_train = 0

for ecg, ppg, sbp, dbp in iter_selected_from_subset(
    image_root=image_root,
    subset="val",
    selected_indices=val_old_to_new_train
):

    count_val_to_train += len(ecg)

print(
    "Actually read samples from old val:",
    count_val_to_train
)

assert count_val_to_train == len(val_old_to_new_train)

print("OLD VAL -> NEW TRAIN check passed.")
'''


#   next step
def load_new_train_to_ram(
    image_root,
    train_old_to_new_train,
    val_old_to_new_train
):
    # Total number of NEW TRAIN samples
    total_samples = (
        len(train_old_to_new_train)
        + len(val_old_to_new_train)
    )

    print("TOTAL NEW TRAIN samples:", total_samples)

    # --------------------------------------------------
    # First generator: OLD TRAIN -> NEW TRAIN
    # --------------------------------------------------
    gen_train = iter_selected_from_subset(
        image_root=image_root,
        subset="train",
        selected_indices=train_old_to_new_train
    )

    # Read first selected chunk only to determine shapes
    first_ecg, first_ppg, first_sbp, first_dbp = next(gen_train)

    print("ECG sample shape:", first_ecg.shape[1:])
    print("PPG sample shape:", first_ppg.shape[1:])
    print("SBP sample shape:", first_sbp.shape[1:])
    print("DBP sample shape:", first_dbp.shape[1:])

    # --------------------------------------------------
    # Preallocate RAM ONCE
    # --------------------------------------------------
    ecg_new_train = np.empty(
        (total_samples, *first_ecg.shape[1:]),
        dtype=np.float32
    )

    ppg_new_train = np.empty(
        (total_samples, *first_ppg.shape[1:]),
        dtype=np.float32
    )

    sbp_new_train = np.empty(
        (total_samples, *first_sbp.shape[1:]),
        dtype=np.float32
    )

    dbp_new_train = np.empty(
        (total_samples, *first_dbp.shape[1:]),
        dtype=np.float32
    )

    # --------------------------------------------------
    # Put first chunk into RAM
    # --------------------------------------------------
    pos = 0
    n = len(first_ecg)

    ecg_new_train[pos:pos+n] = first_ecg
    ppg_new_train[pos:pos+n] = first_ppg
    sbp_new_train[pos:pos+n] = first_sbp
    dbp_new_train[pos:pos+n] = first_dbp

    pos += n

    # --------------------------------------------------
    # Continue OLD TRAIN -> NEW TRAIN
    # --------------------------------------------------
    for ecg, ppg, sbp, dbp in gen_train:

        n = len(ecg)

        ecg_new_train[pos:pos+n] = ecg
        ppg_new_train[pos:pos+n] = ppg
        sbp_new_train[pos:pos+n] = sbp
        dbp_new_train[pos:pos+n] = dbp

        pos += n

        if pos % 100000 < n:
            print("Loaded into NEW TRAIN:", pos)

    print(
        "Finished OLD TRAIN -> NEW TRAIN:",
        pos
    )

    # --------------------------------------------------
    # OLD VAL -> NEW TRAIN
    # --------------------------------------------------
    for ecg, ppg, sbp, dbp in iter_selected_from_subset(
        image_root=image_root,
        subset="val",
        selected_indices=val_old_to_new_train
    ):

        n = len(ecg)

        ecg_new_train[pos:pos+n] = ecg
        ppg_new_train[pos:pos+n] = ppg
        sbp_new_train[pos:pos+n] = sbp
        dbp_new_train[pos:pos+n] = dbp

        pos += n

        if pos % 100000 < n:
            print("Loaded into NEW TRAIN:", pos)

    print(
        "Finished OLD VAL -> NEW TRAIN:",
        pos
    )

    # --------------------------------------------------
    # Final safety check
    # --------------------------------------------------
    assert pos == total_samples, (
        f"Loaded {pos}, expected {total_samples}"
    )

    print("NEW TRAIN loaded into RAM successfully.")

    print("ECG:", ecg_new_train.shape)
    print("PPG:", ppg_new_train.shape)
    print("SBP:", sbp_new_train.shape)
    print("DBP:", dbp_new_train.shape)

    return (
        ecg_new_train,
        ppg_new_train,
        sbp_new_train,
        dbp_new_train
    )



# ---------------------------------------------------------
# STEP 4: Load NEW subject-wise TRAIN into RAM
# ---------------------------------------------------------

train_old_to_new_train = np.load(
    "mydata/train_old_to_new_train.npy"
)

val_old_to_new_train = np.load(
    "mydata/val_old_to_new_train.npy"
)

(
    train_ecg_img,
    train_ppg_img,
    train_sbp,
    train_dbp
) = load_new_train_to_ram(
    image_root=image_root,
    train_old_to_new_train=train_old_to_new_train,
    val_old_to_new_train=val_old_to_new_train
)

print("NEW TRAIN shapes:")
print("ECG:", train_ecg_img.shape)
print("PPG:", train_ppg_img.shape)
print("SBP:", train_sbp.shape)
print("DBP:", train_dbp.shape)

assert len(train_ecg_img) == len(train_ppg_img)
assert len(train_ecg_img) == len(train_sbp)
assert len(train_ecg_img) == len(train_dbp)

print("NEW TRAIN RAM check passed.")


# ---------------------------------------------------------
# STEP 5: Prepare NEW TRAIN for RAMSequence
# ---------------------------------------------------------

train_x = {
    "ecg_input": train_ecg_img,
    "ppg_input": train_ppg_img
}

train_y = {
    "sbp_output": train_sbp,
    "dbp_output": train_dbp
}

print("train_x ECG:", train_x["ecg_input"].shape)
print("train_x PPG:", train_x["ppg_input"].shape)

print("train_y SBP:", train_y["sbp_output"].shape)
print("train_y DBP:", train_y["dbp_output"].shape)

# all above steps for validation
def iter_new_val_batches(
    image_root,
    train_old_to_new_val,
    val_old_to_new_val,
    batch_size=256
):
    """
    Build NEW subject-wise validation batches from:
      1) OLD TRAIN samples assigned to NEW VAL
      2) OLD VAL samples assigned to NEW VAL

    Nothing is saved to disk.
    Entire validation set is NOT loaded into RAM.
    """

    ecg_buffer = []
    ppg_buffer = []
    sbp_buffer = []
    dbp_buffer = []

    buffered = 0

    # --------------------------------------------------
    # NEW VAL comes from BOTH old train and old val
    # --------------------------------------------------
    sources = [
        ("train", train_old_to_new_val),
        ("val",   val_old_to_new_val)
    ]

    for subset, selected_indices in sources:

        print(
            f"Reading OLD {subset} -> NEW VAL: "
            f"{len(selected_indices)} samples"
        )

        for ecg, ppg, sbp, dbp in iter_selected_from_subset(
            image_root=image_root,
            subset=subset,
            selected_indices=selected_indices
        ):

            ecg_buffer.append(ecg)
            ppg_buffer.append(ppg)
            sbp_buffer.append(sbp)
            dbp_buffer.append(dbp)

            buffered += len(ecg)

            # ------------------------------------------
            # As soon as >= batch_size samples exist,
            # combine buffer and yield full batches
            # ------------------------------------------
            if buffered >= batch_size:

                ecg_buffer = [
                    np.concatenate(ecg_buffer, axis=0)
                ]

                ppg_buffer = [
                    np.concatenate(ppg_buffer, axis=0)
                ]

                sbp_buffer = [
                    np.concatenate(sbp_buffer, axis=0)
                ]

                dbp_buffer = [
                    np.concatenate(dbp_buffer, axis=0)
                ]

                while len(ecg_buffer[0]) >= batch_size:

                    batch_ecg = ecg_buffer[0][:batch_size]
                    batch_ppg = ppg_buffer[0][:batch_size]
                    batch_sbp = sbp_buffer[0][:batch_size]
                    batch_dbp = dbp_buffer[0][:batch_size]

                    yield (
                        {
                            "ecg_input": batch_ecg,
                            "ppg_input": batch_ppg
                        },
                        {
                            "sbp_output": batch_sbp,
                            "dbp_output": batch_dbp
                        }
                    )

                    # keep remaining samples
                    ecg_buffer[0] = ecg_buffer[0][batch_size:]
                    ppg_buffer[0] = ppg_buffer[0][batch_size:]
                    sbp_buffer[0] = sbp_buffer[0][batch_size:]
                    dbp_buffer[0] = dbp_buffer[0][batch_size:]

                buffered = len(ecg_buffer[0])

    # --------------------------------------------------
    # Last incomplete batch
    # --------------------------------------------------
    if buffered > 0:

        batch_ecg = np.concatenate(
            ecg_buffer,
            axis=0
        )

        batch_ppg = np.concatenate(
            ppg_buffer,
            axis=0
        )

        batch_sbp = np.concatenate(
            sbp_buffer,
            axis=0
        )

        batch_dbp = np.concatenate(
            dbp_buffer,
            axis=0
        )

        yield (
            {
                "ecg_input": batch_ecg,
                "ppg_input": batch_ppg
            },
            {
                "sbp_output": batch_sbp,
                "dbp_output": batch_dbp
            }
        )

# ---------------------------------------------------------
# STEP 6: Load subject-wise validation indices
# ---------------------------------------------------------

train_old_to_new_val = np.load(
    "mydata/train_old_to_new_val.npy"
)

val_old_to_new_val = np.load(
    "mydata/val_old_to_new_val.npy"
)

print(
    "NEW VAL samples from OLD TRAIN:",
    len(train_old_to_new_val)
)

print(
    "NEW VAL samples from OLD VAL:",
    len(val_old_to_new_val)
)

new_val_total = (
    len(train_old_to_new_val)
    + len(val_old_to_new_val)
)

print("TOTAL NEW VAL samples:", new_val_total)

# ---------------------------------------------------------
# STEP 7: Build subject-wise validation Dataset
# ---------------------------------------------------------

val_dataset = tf.data.Dataset.from_generator(
    lambda: iter_new_val_batches(
        image_root=image_root,
        train_old_to_new_val=train_old_to_new_val,
        val_old_to_new_val=val_old_to_new_val,
        batch_size=256
    ),
    output_signature=(
        {
            "ecg_input": tf.TensorSpec(
                shape=(None, 64, 160, 1),
                dtype=tf.float32
            ),
            "ppg_input": tf.TensorSpec(
                shape=(None, 64, 160, 1),
                dtype=tf.float32
            )
        },
        {
            "sbp_output": tf.TensorSpec(
                shape=(None, 1),
                dtype=tf.float32
            ),
            "dbp_output": tf.TensorSpec(
                shape=(None, 1),
                dtype=tf.float32
            )
        }
    )
)

val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)








# Function to Z-score normalization for LABELS
def z_score_scale(y, mean, std):
    y_scaled = (y - mean) / std
    return y_scaled

# Function to Z-score De-normalization for LABELS
def denormalize_labels_zscore(y_scaled, mean, std):
    y = y_scaled * std + mean
    return y

# convert [0-1] into mmHg in loss function to mnimize error in scale of mmHg not percetage between [0,1]
# MAE loss in mmHg with Z-score normalization

def mmhg_mae_loss(bp_mean, bp_std):

    def loss(y_true_scaled, y_pred_scaled):
        # de-normalize to mmHg
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    return loss

def mmhg_mae_metric(bp_mean, bp_std):

    def metric(y_true_scaled, y_pred_scaled):
        # de-normalize from Z-score
        y_true_bp = y_true_scaled * bp_std + bp_mean
        y_pred_bp = y_pred_scaled * bp_std + bp_mean

        # MAE in mmHg
        return tf.reduce_mean(tf.abs(y_pred_bp - y_true_bp))

    metric.__name__ = "mae_mmhg"
    return metric

# mixed-precision
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

print("Mixed precision policy:", mixed_precision.global_policy())
# mixed-precision

from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def CNN_BiLSTM_2D(ecg_img_shape, ppg_img_shape,head_mlp_units):

    # Inputs
    ecg_input = Input(shape=ecg_img_shape, name='ecg_input')
    ppg_input = Input(shape=ppg_img_shape, name='ppg_input')

    #H, W, C = ecg_input
    #kh1, kw1 = safe_kernel_size(H, W, (3, 3))
    # padding='same' causes bad results
    # ECG CNN
    '''
    x1 = layers.Conv2D(32,  (3,3), padding='same', activation='relu')(ecg_input)
    x1 = layers.Conv2D(32,  (3,3), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(64,  (3,3), strides=(1,1), padding='same', activation='relu') (x1) #instead of max pooling that decrease suddenly, reducing is learnable
    x1 = layers.Conv2D(64,  (3,3), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu')(x1) # optional small freq reduction

    # PPG CNN
    x2 = layers.Conv2D(32,  (3,3), padding='same', activation='relu')(ppg_input)
    x2 = layers.Conv2D(32,  (3,3), padding='same', activation='relu')(x2)
    x2 = layers.Conv2D(64,  (3,3), strides=(1,1), padding='same', activation='relu') (x2) #instead of max pooling that decrease suddenly, reducing is learnable
    x2 = layers.Conv2D(64,  (3,3), padding='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(128, (3,3), strides = (1,1), padding ='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(128, (3,3), strides = (1,1), padding ='same', activation='relu') (x2)
    '''
    # ECG CNN
    x1 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ecg_input)
    x1 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(64,  (5,5), strides=(2,1), padding='same', activation='relu') (x1) #instead of max pooling that decrease suddenly, reducing is learnable
    x1 = layers.Conv2D(64,  (5,5), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(256, (5,5), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(256, (5,5), strides=(2,1), padding='same', activation='relu') (x1)
    x1 = layers.Conv2D(128, (5,5), padding ='same', activation='relu')(x1) # to increase receptive field
    x1 = layers.Conv2D(128, (5,5), strides=(2,1), padding='same', activation='relu')(x1) # optional small freq reduction

    # PPG CNN
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ppg_input)
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x2)
    x2 = layers.Conv2D(64,  (7,7), strides=(2,1), padding='same', activation='relu') (x2) #instead of max pooling that decrease suddenly, reducing is learnable
    x2 = layers.Conv2D(64,  (7,7), padding='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(256, (7,7), padding='same', activation='relu') (x2)
    x2 = layers.Conv2D(256, (7,7), strides=(2,1), padding='same', activation='relu') (x2)
    x2 = layers.Conv2D(128, (5,5), strides = (2,1), padding ='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(128, (3,3), strides = (1,1), padding ='same', activation='relu') (x2)

    '''
    # PPG CNN
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(ppg_input)
    x2 = layers.Conv2D(32,  (5,5), padding='same', activation='relu')(x2)
    x2 = layers.Conv2D(64,  (5,5), strides=(2,2), padding='same', activation='relu') (x2) #instead of max pooling that decrease suddenly, reducing is learnable
    x2 = layers.Conv2D(64,  (5,5), padding='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(128, (5,5), strides = (2,2), padding ='same', activation='relu')(x2) # to increase receptive field
    x2 = layers.Conv2D(128, (5,5), strides = (1,1), padding ='same', activation='relu') (x2)
    '''
    # This block: from (batch, frequency, time, channels)  ==> (batch, time, frequency, channels) ==> (batch, time, frequency * channels)

    # Merge features
    x = layers.Concatenate(axis=-1)([x1, x2]) # (batch,H, W, C)
    # set time dimension as main idea  Put time dimension in axis 1
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]))(x)   # (batch, W, H, C)
    # make features from  frequency and channels to achieve (batch, time, features)
    # Flatten freq and channels into features
    x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2] * tf.shape(t)[3])))(x) # (batch, time, features)

    # 8 stacked BiLSTM
    for _ in range(1): #5):
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    # Fully Connected layers
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
        
        # temporary
        #self.ecg_files = self.ecg_files[:400]
        #self.ppg_files = self.ppg_files[:400]
        #self.sbp_files = self.sbp_files[:400]
        #self.dbp_files = self.dbp_files[:400]
        #
        
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

'''
#max_epochs = 30
# to prevent losing data of end of each chunk
# as data is  180432 the chnk siz can be =>16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 112, 126, 144, 168, 252, 336, 504
chunk_size = 126

train_generator = ChunkSequence(
    data_dir=data_dir,
    subset="train",
    shuffle=True
)

val_generator = ChunkSequence(
    data_dir=data_dir,
    subset="val",
    shuffle=False
)
sample_x, sample_y = train_generator[0]
input_shape = sample_x["ecg_input"].shape[1:]

'''
import os
import glob
import numpy as np

def load_all_chunks(data_dir, subset):

    ecg_files = sorted(glob.glob(f"{data_dir}/ecg_{subset}_*.npy"))
    ppg_files = sorted(glob.glob(f"{data_dir}/ppg_{subset}_*.npy"))
    sbp_files = sorted(glob.glob(f"{data_dir}/sbp_{subset}_*.npy"))
    dbp_files = sorted(glob.glob(f"{data_dir}/dbp_{subset}_*.npy"))


    #ecg_files = ecg_files[:400]
    #ppg_files = ppg_files[:400]
    #sbp_files = sbp_files[:400]
    #dbp_files = dbp_files[:400]
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

#----------------------------------------------------------------
from tensorflow.keras.utils import Sequence

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
'''
#------------------------------------------------------------
def split_train_val_subject_wise():

    myData = "mydata/"

    # -----------------------------------------
    # 1) Load original 10-sec subject IDs
    # -----------------------------------------
    train_sid_10s = np.load(myData + "train_subjectids.npy")
    val_sid_10s   = np.load(myData + "val_subjectids.npy")

    print("train 10s:", train_sid_10s.shape)
    print("val   10s:", val_sid_10s.shape)


    # -----------------------------------------
    # 2) Convert IDs to the SAME ordering as
    #    your 5-sec resegmentation
    #
    # Original:
    # A B C ...
    #
    # After your vstack:
    # A B C ... A B C ...
    # -----------------------------------------
    train_sid_5s = np.concatenate(
        [train_sid_10s, train_sid_10s],
        axis=0
    )

    val_sid_5s = np.concatenate(
        [val_sid_10s, val_sid_10s],
        axis=0
    )

    print("train 5s:", train_sid_5s.shape)
    print("val   5s:", val_sid_5s.shape)


    # -----------------------------------------
    # 3) Find ALL unique subjects
    # -----------------------------------------
    all_subjects = np.unique(
        np.concatenate([train_sid_10s, val_sid_10s])
    )

    print("Total unique subjects:", len(all_subjects))


    # -----------------------------------------
    # 4) Subject-wise random split
    #    Example: 80% train, 20% validation
    # -----------------------------------------
    rng = np.random.default_rng(42)

    subjects_shuffled = all_subjects.copy()
    rng.shuffle(subjects_shuffled)

    n_val_subjects = int(round(0.20 * len(subjects_shuffled)))

    new_val_subjects = subjects_shuffled[:n_val_subjects]
    new_train_subjects = subjects_shuffled[n_val_subjects:]

    print("New train subjects:", len(new_train_subjects))
    print("New val subjects:", len(new_val_subjects))

    print(
        "Common subjects:",
        len(np.intersect1d(new_train_subjects, new_val_subjects))
    )
    train_old_to_new_train = np.isin(
        train_sid_5s,
        new_train_subjects
    )

    train_old_to_new_val = np.isin(
        train_sid_5s,
        new_val_subjects
    )

    val_old_to_new_train = np.isin(
        val_sid_5s,
        new_train_subjects
    )

    val_old_to_new_val = np.isin(
        val_sid_5s,
        new_val_subjects
    )
    # save indx
    np.save(
        "mydata/new_train_subjects.npy",
        new_train_subjects
    )

    np.save(
        "mydata/new_val_subjects.npy",
        new_val_subjects
    )

    np.save(
        "mydata/train_old_to_new_train.npy",
        np.where(train_old_to_new_train)[0]
    )

    np.save(
        "mydata/train_old_to_new_val.npy",
        np.where(train_old_to_new_val)[0]
    )

    np.save(
        "mydata/val_old_to_new_train.npy",
        np.where(val_old_to_new_train)[0]
    )

    np.save(
        "mydata/val_old_to_new_val.npy",
        np.where(val_old_to_new_val)[0]
    )
    print("\nSamples from OLD TRAIN:")
    print(
        " -> new train:",
        np.sum(train_old_to_new_train)
    )
    print(
        " -> new val:",
        np.sum(train_old_to_new_val)
    )

    print("\nSamples from OLD VAL:")
    print(
        " -> new train:",
    np.sum(val_old_to_new_train)
    )
    print(
        " -> new val:",
        np.sum(val_old_to_new_val)
    )

    assert (
        np.sum(train_old_to_new_train) +
        np.sum(train_old_to_new_val)
        == len(train_sid_5s)
    )

    assert (
        np.sum(val_old_to_new_train) +
        np.sum(val_old_to_new_val)
        == len(val_sid_5s)
    )

    print("All samples assigned correctly.")
'''
#------------------------------------------------------------
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

#----------------------------------------------------------------

# split subject_wise for training and validation
# because segmentation is first half of all subject+ second half of all object
# like : A1, B1, C1,......, A2,B2,C2
# split_train_val_subject_wise()

data_dir = "/scratch/skhalaj/cwt_32sr_5sec"

import time
load_start = time.perf_counter()

#train_x, train_y = load_all_chunks( data_dir, "train")
## val_x, val_y = load_all_chunks( data_dir, "val")


train_generator = RAMSequence(
    train_x,
    train_y,
    batch_size=batch_size,
    shuffle=True
)


'''
batch_x, batch_y = train_generator[0]

print("Batch ECG:", batch_x["ecg_input"].shape)
print("Batch PPG:", batch_x["ppg_input"].shape)

print("Batch SBP:", batch_y["sbp_output"].shape)
print("Batch DBP:", batch_y["dbp_output"].shape)
'''
'''
# reday validarion 

val_generator_test = iter_new_val_batches(
    image_root=image_root,
    train_old_to_new_val=train_old_to_new_val,
    val_old_to_new_val=val_old_to_new_val,
    batch_size=256
)

val_batch_x, val_batch_y = next(val_generator_test)

print("VAL Batch ECG:", val_batch_x["ecg_input"].shape)
print("VAL Batch PPG:", val_batch_x["ppg_input"].shape)

print("VAL Batch SBP:", val_batch_y["sbp_output"].shape)
print("VAL Batch DBP:", val_batch_y["dbp_output"].shape)
'''
validation_steps = int(np.ceil(new_val_total / 256))

print("(This print must be like Validation steps: 1410)Validation steps:", validation_steps)

#raise SystemExit


#val_generator = ChunkSequence(
#    data_dir=data_dir,
#    subset="val",
#    shuffle=False
#)

load_end = time.perf_counter()
print(f"Data loading time: {(load_end - load_start)/60:.2f} minutes")

print(train_x["ecg_input"].shape)
print(train_x["ppg_input"].shape)
print(train_y["sbp_output"].shape)
print(train_y["dbp_output"].shape)

#print(val_x["ecg_input"].shape)
#print(val_x["ppg_input"].shape)

input_shape = train_x["ecg_input"].shape[1:]
H, W, _ = input_shape
print("input_shape:", input_shape)
print("H:", H)
print("W:", W)
# patch_h = 16 # 8  # 8
# patch_w = 8  # 8  # 16
print ("patch_h=", patch_h, ", patch_w=", patch_w)

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

if (model_name=='CNN'):
   with tf.device(device):
        model = CNN_BiLSTM_2D(
                   ecg_img_shape= input_shape,
                   ppg_img_shape= input_shape,
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

# load mean and std from file for denormalization
sbp_mean = np.float32( np.load(f"{data_dir}/sbp_mean.npy"))
sbp_std  = np.float32( np.load(f"{data_dir}/sbp_std.npy"))

dbp_mean = np.float32( np.load(f"{data_dir}/dbp_mean.npy"))
dbp_std  = np.float32( np.load(f"{data_dir}/dbp_std.npy"))

print("SBP mean/std:", sbp_mean, sbp_std)
print("DBP mean/std:", dbp_mean, dbp_std)

loss_fn_sbp = mmhg_mae_loss(sbp_mean, sbp_std)
loss_fn_dbp = mmhg_mae_loss(dbp_mean, dbp_std)

mae_mmhg_sbp = mmhg_mae_metric(sbp_mean, sbp_std)
mae_mmhg_dbp = mmhg_mae_metric(dbp_mean, dbp_std)

if model_name=='CNN':
   # H, W, C = ecg_img_shape
   # kh1, kw1 = safe_kernel_size(H, W, (3, 3))
   lr = 1e-4
else:
   lr = 1e-4

#model.compile (optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
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

'''
sample_x, sample_y = train_generator[0]

print("ECG dtype:", sample_x["ecg_input"].dtype)
print("PPG dtype:", sample_x["ppg_input"].dtype)
print("SBP dtype:", sample_y["sbp_output"].dtype)
print("DBP dtype:", sample_y["dbp_output"].dtype)

print("sbp_mean dtype:", sbp_mean.dtype)
print("sbp_std dtype:", sbp_std.dtype)
print("dbp_mean dtype:", dbp_mean.dtype)
print("dbp_std dtype:", dbp_std.dtype)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=max_epochs,
    verbose=0,
    callbacks=[early_stopping]
)
'''







'''
# -----------------test--------------- cudo
@tf.function
def check_forward(ecg, ppg):
    return model(
        {
            "ecg_input": ecg,
            "ppg_input": ppg
        },
        training=True
    )

concrete_func = check_forward.get_concrete_function(
    tf.TensorSpec(
        shape=(None, *input_shape),
        dtype=tf.float32
    ),
    tf.TensorSpec(
        shape=(None, *input_shape),
        dtype=tf.float32
    )
)

graph_def = concrete_func.graph.as_graph_def()

cudnn_ops = []

# Check main graph
for node in graph_def.node:
    if "Cudnn" in node.op or "cudnn" in node.name.lower():
        cudnn_ops.append((node.name, node.op))

# Check functions inside the graph
for func in graph_def.library.function:
    for node in func.node_def:
        if "Cudnn" in node.op or "cudnn" in node.name.lower():
            cudnn_ops.append((node.name, node.op))

print("\n==============================")
print("cuDNN CHECK")
print("==============================")

if cudnn_ops:
    print("cuDNN operations found:")
    for name, op in sorted(set(cudnn_ops)):
        print(op, ":", name)
else:
    print("NO explicit cuDNN operation found")
# -------------------------------------
'''



max_epochs = 200

train_start = time.perf_counter()


print("Train samples:", len(train_x["ecg_input"]))
print("Train batches:", len(train_generator))
print("Validation samples:", new_val_total)

'''
Above print must be like:
Train samples: 1443600
Train batches: 5640
Validation samples: 360720
'''


# the best the last fit 
history = model.fit(
    train_generator,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=max_epochs,
    callbacks=[reduce_lr, early_stopping],
    verbose=2
)

# this code is correct just comment to load
#history = model.fit(
#    train_generator,
#    validation_data=val_generator,
#    epochs=max_epochs,
#    verbose=2, # 0 = show nothing  2 show right after each epoch
#    callbacks=[reduce_lr,early_stopping]
#)
'''
error:
history = model.fit(
    train_x,
    train_y,
    validation_data=val_generator, #(val_x, val_y),
    batch_size=126,
    epochs=max_epochs,
    verbose=1,
    callbacks=[early_stopping],
    shuffle=True
)
'''
train_end = time.perf_counter()
print(f"Training time: {(train_end - train_start)/60:.2f} minutes")

'''
# evaluate model on validation subset after training
# denormalize true labels
y_val_sbp = denormalize_labels_zscore(val_y ["sbp_output"], sbp_mean, sbp_std)
y_val_dbp = denormalize_labels_zscore(val_y ["dbp_output"], dbp_mean, dbp_std)

# predict labels by the model
y_pred_sbp_scaled, y_pred_dbp_scaled = model.predict(val_generator, verbose=0)

# denormalize predicted labels by the model
y_pred_sbp = denormalize_labels_zscore(y_pred_sbp_scaled, sbp_mean, sbp_std)
y_pred_dbp = denormalize_labels_zscore(y_pred_dbp_scaled, dbp_mean, dbp_std)

# check if the shape are the same
print("True SBP :", y_val_sbp.shape)
print("Pred SBP :", y_pred_sbp.shape)

print("True DBP :", y_val_dbp.shape)
print("Pred DBP :", y_pred_dbp.shape)

assert len(y_val_sbp) == len(y_pred_sbp)
assert len(y_val_dbp) == len(y_pred_dbp)

# evaluate true labels vs. predicted lables
# ME (Mean Error)
me_sbp = np.mean(y_val_sbp - y_pred_sbp)
me_dbp = np.mean(y_val_dbp - y_pred_dbp)

# MSE
mse_sbp = mean_squared_error(y_val_sbp, y_pred_sbp)
mse_dbp = mean_squared_error(y_val_dbp, y_pred_dbp)

# MAE
mae_sbp = mean_absolute_error(y_val_sbp, y_pred_sbp)
mae_dbp = mean_absolute_error(y_val_dbp, y_pred_dbp)

# SDE
std_sbp = np.std(y_val_sbp - y_pred_sbp)
std_dbp = np.std(y_val_dbp - y_pred_dbp)

# (None, 64, 160, 1)
#         ↑   ↑
#      scales time
# (batch, frequency/scales, time, channel)
print(f"\n====================================================================================")
print(f"SBP: ME={me_sbp:.2f}, MAE={mae_sbp:.2f}, MSE={mse_sbp:.2f}, SDE={std_sbp:.2f}")
print(f"DBP: ME={me_dbp:.2f}, MAE={mae_dbp:.2f}, MSE={mse_dbp:.2f}, SDE={std_dbp:.2f}")
'''
# Save the trained model----------------------
model.save('Subj_img_' + model_name + f"_fromsavedimg_256batch.keras")

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(4, 3))
# when chunk plt.plot(epochs, train_loss_epochs, label='Training Loss')
#plt.plot(epochs, val_loss_epochs, label='Validation Loss')
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig('Subj_img_'+ model_name + f"_loadfromsavedimg_" +'_loss_curve.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.show()


#raise SystemExit
'''
#--------------- load model beacause we used manual early stopping and weights must be the best model, not the model currenlty in RAM
best_model_path = "img_CNN_fromsavedimg_256batch.keras"

try:
   model = tf.keras.models.load_model(
        best_model_path,
        custom_objects={
            "CNN_BiLSTM_2D": CNN_BiLSTM_2D
        },
        compile=False
    )

   print(f"Best model loaded from {best_model_path}")
except Exception as e:
   print(f"WARNING: could not load best model from disk: {e}")
   print("Using in-memory model for testing instead.")
'''

# Baland_altman
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
        plt.savefig(f"Subj_img_{model_name}_Blnatm_{dataset_name}_{labels[k]}.png", dpi=300, bbox_inches='tight')
        plt.close()


from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

def plot_estimated_vs_reference_with_density(y_true_sbp, y_true_dbp, y_pred_sbp, y_pred_dbp, dataset_name='unknown'):
    labels = ['SBP', 'DBP']

    for i in range(2):  # 0 for SBP, 1 for DBP
        if (i==0): #sbp
            true = y_true_sbp
            pred = y_pred_sbp
        else: #dbp
            true = y_true_dbp
            pred = y_pred_dbp

        # felan comment, if error the you can uncomment it
        #true = np.asarray(reshape(-1))
        #pred = np.asarray(reshape(-1))

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
        plt.savefig(f"Subj_img_{model_name}_{dataset_name}_{labels[i]}.png", dpi=300, bbox_inches='tight')
        plt.close()

def test_mymodel (y_pred_sbp, y_pred_dbp, y_true_sbp, y_true_dbp, dataset_name="unknown"):

    # flatten
    y_true_sbp = np.asarray(y_true_sbp[:, 0], dtype=np.float32)
    y_true_dbp = np.asarray(y_true_dbp[:, 0], dtype=np.float32)
    y_pred_sbp = np.asarray(y_pred_sbp[:, 0], dtype=np.float32)
    y_pred_dbp = np.asarray(y_pred_dbp[:, 0], dtype=np.float32)

    # metrics
    me_sbp = np.mean(y_true_sbp - y_pred_sbp)
    me_dbp = np.mean(y_true_dbp - y_pred_dbp)

    mse_sbp = mean_squared_error(y_true_sbp, y_pred_sbp)
    mse_dbp = mean_squared_error(y_true_dbp, y_pred_dbp)

    mae_sbp = mean_absolute_error(y_true_sbp, y_pred_sbp)
    mae_dbp = mean_absolute_error(y_true_dbp, y_pred_dbp)

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

# Evalluate model on test subsets

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

    # de-normalize predictions
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
