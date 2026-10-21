# Cuffless Blood Pressure Estimation from ECG and PPG Using Deep Learning

This repository contains Python implementations associated with a master's thesis on cuffless blood pressure (BP) estimation using electrocardiogram (ECG) and photoplethysmogram (PPG) signals. The research investigates time-series and image-based input representations for estimating systolic blood pressure (SBP) and diastolic blood pressure (DBP) with deep learning.

## 

## Research overview

The work includes two main studies:

* **Study I — Time-series-based models:** ECG and PPG signals are provided directly as one-dimensional sequences to CNN–BiLSTM, patch-based Transformer encoder, and time-series Transformer encoder models.
* **Study II — Image-based representations:** ECG and PPG signals are transformed into images, with experiments involving representation selection, cross-validation, and CNN–BiLSTM modeling. The image-based investigation includes continuous wavelet transform (CWT) representations.

## 

## Dataset

The research uses **PulseDB**, which provides synchronized ECG, PPG, and arterial blood pressure (ABP) recordings derived from the MIMIC-III and VitalDB sources. ABP is used to obtain reference SBP and DBP values.

**Dataset files are not included in this repository.** Users must obtain PulseDB through its official distribution channels and follow the applicable access conditions and licenses. Local dataset paths in the scripts may need to be updated before use.

The study considers calibration-based (seen-subject) and calibration-free (unseen-subject) evaluation. The evaluation protocols and exact dataset partitions should be taken from the thesis, rather than inferred solely from file names.

## 

## Repository Structure



The repository is organized into four main directories, covering data preparation, time-series experiments, image-based experiments, and synthetic sample data.



BloodPressurefromECGPPG/

│

├── 1\_Data\_Preparation/

│

├── 2\_TimeSeries\_Experiments/

│   ├── 1\_CNN-BiLSTM/

│   ├── 2\_PatchBased\_Transformer/

│   └── 3\_TransformerEncoder/

│

├── 3\_ImageBased\_Experiments/

│   ├── 1\_Configuration\_Screening/

│   ├── 2\_CNN-BiLSTM\_Experiment/

│   └── 3\_SubjectWised\_Experiments/

│

├── Synthetic\_Sample\_Data/

│

└── README.md



**1. Data Preparation**



The '1\_Data\_Preparation' directory contains Python scripts for preparing the PulseDB dataset for deep learning experiments.



These scripts cover:



* Converting the original MATLAB files into NumPy arrays.
* Extracting statistical information from the dataset subsets.
* Creating five folds from the PulseDB training set.
* Analyzing age differences in the dataset.



**2. Time-Series Experiments**



The '2\_TimeSeries\_Experiments' directory contains the implementations and experimental results of three deep learning architectures for blood pressure estimation directly from ECG and PPG time-series signals.



The three architectures are:



* **CNN-BiLSTM:** A dual-branch architecture combining convolutional feature extraction and bidirectional LSTM layers.
* **Patch-Based Transformer:** A Transformer architecture that processes ECG and PPG signals as sequences of temporal patches.
* **Transformer Encoder:** A dual-branch Transformer architecture for processing ECG and PPG time-series signals.



Each model is organized in a separate directory to facilitate reproducibility and comparison.



**3. Image-Based Experiments**



The '3\_ImageBased\_Experiments' directory contains the implementations and results of experiments using image-based representations of ECG and PPG signals.



It includes three main components:



**Configuration Screening**



Scripts for investigating different signal representations and experimental configurations, including sampling frequencies, segment lengths, image transformation methods, and model architectures.



The investigated signal-to-image methods include STFT, CWT, GAF, and Visibility Graphs.






The 'Example\_Generated\_Images' directory contains examples of generated signal representations.



**CNN-BiLSTM Experiment**



Implementation and evaluation of the image-based CNN-BiLSTM architecture using the selected signal representation and experimental settings.



**Subject-Wise Experiments**



Additional experiments related to subject-wise evaluation of blood pressure estimation models.



**4. Synthetic Sample Data**



The 'Synthetic\_Sample\_Data' directory provides artificially generated ECG, PPG, and blood pressure label arrays for demonstrating the expected data structure.



These files are intended only for testing data-loading procedures and understanding the required input formats.



Note: The synthetic samples are not derived from real patients and are not used for model training, validation, or performance evaluation reported in the thesis.





**Software and computational environment**

The research code was developed in Python using deep learning and scientific computing libraries, including TensorFlow/Keras, NumPy, and SciPy. Experiments were run using GPU computing resources provided by the Digital Research Alliance of Canada.

The research environment included Python 3.11.5 and CUDA 12.2. **These are reported research-environment details, not a verified installation specification for every script.** Consult the imports and environment settings in each script before running it. A complete, tested dependency list is not provided here.

## 

## Running the code

1. Obtain the required PulseDB data and prepare the data directories.
2. Review each script's input/output paths and adapt any cluster-specific paths to your system.
3. Install the packages required by the scripts you plan to run, using versions compatible with your hardware and TensorFlow setup.
4. Consult the thesis for the dataset partitions, preprocessing procedures, experiment settings, and model evaluation protocols.
5. Run only the script or group of scripts needed for your chosen experiment. Some scripts may require intermediate files generated by other scripts.

**Important:** The scripts may contain paths or settings specific to the original high-performance computing environment. This repository has not been presented as a verified one-command reproduction package.



## Scope of the repository

This repository provides the main implementations and selected experimental configurations relevant to the methods and results described in the thesis. The research also involved additional exploratory experiments to investigate input representations, sampling rates, segment lengths, model architectures, and training settings. **It is not intended as a complete archive of every exploratory experiment or training log.**

The thesis is the primary reference for the reported results, model comparisons, and selection procedures.



## Thesis and citation

**Thesis title:** *Cuffless Blood Pressure Estimation from Photoplethysmogram and Electrocardiogram using Deep Learning*

Author, institution, year, and a permanent thesis link can be added after the final thesis record is available.



### Synthetic Sample Data

The 'Synthetic\_Sample\_Data/' folder contains 10 randomly generated synthetic samples to demonstrate the expected data formats and shapes.



|File|Shape|Description|
|-|-|-|
|Synthetic\_Sample\_ECG.npy|(10, 1250)|Synthetic ECG samples|
|Synthetic\_Sample\_PPG.npy|(10, 1250)|Synthetic PPG samples|
|Synthetic\_Sample\_ABP.npy|(10, 1250)|Synthetic ABP samples|
|synthetic\_labels.npy|(10, 2)|Synthetic SBP and DBP labels in mmHg|



Each sample contains 1,250 data points, corresponding to a 10-second segment at 125 Hz.

**Note:** These data are randomly generated and do not represent real physiological signals. They are provided only to demonstrate the expected data structure and are not suitable for model training or evaluation.

