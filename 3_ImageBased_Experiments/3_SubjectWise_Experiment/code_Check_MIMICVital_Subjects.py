import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# PATH
DATA_DIR = "pulsedb_data_npy"

# 1. LOAD ORIGINAL SUBJECT IDs

train_subjects_10s = np.load( f"{DATA_DIR}/train_subjectids.npy").reshape(-1)

val_subjects_10s = np.load( f"{DATA_DIR}/val_subjectids.npy").reshape(-1)


print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print("Original train segments:", len(train_subjects_10s))
print("Original val segments:", len(val_subjects_10s))

print(
    "Original train unique subjects:",
    len(np.unique(train_subjects_10s))
)

print(
    "Original val unique subjects:",
    len(np.unique(val_subjects_10s))
)

# 2. LOAD SUBJECT-INDEPENDENT SPLIT INDEX FILES


tr_tr = np.load(f"{DATA_DIR}/train_old_to_new_train.npy")

tr_va = np.load( f"{DATA_DIR}/train_old_to_new_val.npy")

va_tr = np.load(f"{DATA_DIR}/val_old_to_new_train.npy")

va_va = np.load( f"{DATA_DIR}/val_old_to_new_val.npy")


# ============================================================
# 3. CONVERT SUBJECT IDs FROM 10 s TO 5 s
#
# IMPORTANT:
# Correct ordering is concatenate, NOT repeat.
# ============================================================

train_subjects_5s = np.concatenate([
    train_subjects_10s,
    train_subjects_10s
])

val_subjects_5s = np.concatenate([
    val_subjects_10s,
    val_subjects_10s
])


print("\n" + "=" * 70)
print("5-SECOND SUBJECT ARRAYS")
print("=" * 70)

print("Old train 5s:", len(train_subjects_5s))
print("Old val 5s:", len(val_subjects_5s))


# ============================================================
# 4. REBUILD NEW SUBJECT-INDEPENDENT TRAIN
# ============================================================

new_train_subjects_from_old_train = train_subjects_5s[tr_tr]

new_train_subjects_from_old_val = val_subjects_5s[va_tr]

new_train_subjects = np.concatenate([
    new_train_subjects_from_old_train,
    new_train_subjects_from_old_val
])


# ============================================================
# 5. REBUILD NEW SUBJECT-INDEPENDENT VALIDATION
# ============================================================

new_val_subjects_from_old_train = train_subjects_5s[tr_va]

new_val_subjects_from_old_val = val_subjects_5s[va_va]

new_val_subjects = np.concatenate([
    new_val_subjects_from_old_train,
    new_val_subjects_from_old_val
])


# ============================================================
# 6. UNIQUE SUBJECTS
# ============================================================

train_unique = np.unique(new_train_subjects)

val_unique = np.unique(new_val_subjects)

shared = np.intersect1d(
    train_unique,
    val_unique
)


print("\n" + "=" * 70)
print("SUBJECT-INDEPENDENT SPLIT CHECK")
print("=" * 70)

print("New Train samples:", len(new_train_subjects))
print("New Val samples:", len(new_val_subjects))

print("New Train subjects:", len(train_unique))
print("New Val subjects:", len(val_unique))

print("Shared subjects:", len(shared))


# ============================================================
# 7. COUNT _0 AND _1 SUBJECTS
# ============================================================

train_0 = np.sum(
    np.char.endswith(train_unique, "_0")
)

train_1 = np.sum(
    np.char.endswith(train_unique, "_1")
)

val_0 = np.sum(
    np.char.endswith(val_unique, "_0")
)

val_1 = np.sum(
    np.char.endswith(val_unique, "_1")
)


train_other = np.sum(
    ~np.char.endswith(train_unique, "_0")
    &
    ~np.char.endswith(train_unique, "_1")
)

val_other = np.sum(
    ~np.char.endswith(val_unique, "_0")
    &
    ~np.char.endswith(val_unique, "_1")
)


# ============================================================
# 8. PERCENTAGES
# ============================================================

train_0_pct = 100 * train_0 / len(train_unique)

train_1_pct = 100 * train_1 / len(train_unique)

val_0_pct = 100 * val_0 / len(val_unique)

val_1_pct = 100 * val_1 / len(val_unique)


print("\n" + "=" * 70)
print("SUBJECT DISTRIBUTION BY SUFFIX")
print("=" * 70)

print("\nNEW TRAIN")

print("Total:", len(train_unique))
print("_0:", train_0)
print("_1:", train_1)
print("Other:", train_other)

print("_0 %:", train_0_pct)
print("_1 %:", train_1_pct)


print("\nNEW VALIDATION")

print("Total:", len(val_unique))
print("_0:", val_0)
print("_1:", val_1)
print("Other:", val_other)

print("_0 %:", val_0_pct)
print("_1 %:", val_1_pct)


print("\nPercentage-point difference:")

print(
    "_0 Val - Train:",
    val_0_pct - train_0_pct
)

print(
    "_1 Val - Train:",
    val_1_pct - train_1_pct
)


# ============================================================
# 9. SAMPLE-LEVEL COUNTS
#
# Since every subject should have 720 samples,
# these ratios should be essentially identical.
# But we verify it explicitly.
# ============================================================

train_sample_0 = np.sum(
    np.char.endswith(new_train_subjects, "_0")
)

train_sample_1 = np.sum(
    np.char.endswith(new_train_subjects, "_1")
)

val_sample_0 = np.sum(
    np.char.endswith(new_val_subjects, "_0")
)

val_sample_1 = np.sum(
    np.char.endswith(new_val_subjects, "_1")
)


print("\n" + "=" * 70)
print("SAMPLE DISTRIBUTION BY SUFFIX")
print("=" * 70)


print("\nNEW TRAIN")

print("_0 samples:", train_sample_0)
print("_1 samples:", train_sample_1)

print(
    "_0 %:",
    100 * train_sample_0 / len(new_train_subjects)
)

print(
    "_1 %:",
    100 * train_sample_1 / len(new_train_subjects)
)


print("\nNEW VALIDATION")

print("_0 samples:", val_sample_0)
print("_1 samples:", val_sample_1)

print(
    "_0 %:",
    100 * val_sample_0 / len(new_val_subjects)
)

print(
    "_1 %:",
    100 * val_sample_1 / len(new_val_subjects)
)


# ============================================================
# 10. STATISTICAL TEST:
#     IS SUFFIX DISTRIBUTION DIFFERENT BETWEEN
#     TRAIN AND VALIDATION?
# ============================================================

contingency_table = np.array([
    [train_0, train_1],
    [val_0, val_1]
])


chi2, p_value, dof, expected = chi2_contingency(
    contingency_table
)


print("\n" + "=" * 70)
print("CHI-SQUARE TEST")
print("=" * 70)

print("Observed table:")
print(contingency_table)

print("\nExpected table:")
print(expected)

print("\nChi-square:", chi2)
print("p-value:", p_value)
print("Degrees of freedom:", dof)


# ============================================================
# 11. CHECK ORIGINAL FULL DATASET DISTRIBUTION
# ============================================================

all_original_subjects = np.unique(
    np.concatenate([
        train_subjects_10s,
        val_subjects_10s
    ])
)


all_0 = np.sum(
    np.char.endswith(all_original_subjects, "_0")
)

all_1 = np.sum(
    np.char.endswith(all_original_subjects, "_1")
)


print("\n" + "=" * 70)
print("FULL ORIGINAL DATASET")
print("=" * 70)

print("Total subjects:", len(all_original_subjects))

print("_0:", all_0)

print("_1:", all_1)

print(
    "_0 %:",
    100 * all_0 / len(all_original_subjects)
)

print(
    "_1 %:",
    100 * all_1 / len(all_original_subjects)
)


# ============================================================
# 12. CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "Set": [
        "Full_dataset",
        "New_Train",
        "New_Validation"
    ],

    "Total_subjects": [
        len(all_original_subjects),
        len(train_unique),
        len(val_unique)
    ],

    "Suffix_0_subjects": [
        all_0,
        train_0,
        val_0
    ],

    "Suffix_0_percent": [
        100 * all_0 / len(all_original_subjects),
        train_0_pct,
        val_0_pct
    ],

    "Suffix_1_subjects": [
        all_1,
        train_1,
        val_1
    ],

    "Suffix_1_percent": [
        100 * all_1 / len(all_original_subjects),
        train_1_pct,
        val_1_pct
    ]
})


print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))


# ============================================================
# 13. SAVE RESULTS
# ============================================================

summary.to_csv(
    "source_split_summary.csv",
    index=False
)


pd.DataFrame({
    "subject": train_unique,
    "suffix": np.where(
        np.char.endswith(train_unique, "_0"),
        "_0",
        "_1"
    ),
    "set": "Train"
}).to_csv(
    "new_train_subject_source.csv",
    index=False
)


pd.DataFrame({
    "subject": val_unique,
    "suffix": np.where(
        np.char.endswith(val_unique, "_0"),
        "_0",
        "_1"
    ),
    "set": "Validation"
}).to_csv(
    "new_val_subject_source.csv",
    index=False
)


print("\nSaved:")
print("source_split_summary.csv")
print("new_train_subject_source.csv")
print("new_val_subject_source.csv")

print("\nDONE")
