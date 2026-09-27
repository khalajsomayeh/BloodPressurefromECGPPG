from mat73 import loadmat
import h5py
import numpy as np
import gc
import pandas as pd
from datetime import datetime

# function to recieves data from each subset, calculates their statistics and print them
def compute_dataset_statistics(ecg,ppg,abp,age,gender,height,weight,subjectids, y_data, dataset_title="Unknown"):
    unique_subjects = len(np.unique(subjectids))

    sbp = y_data[:, 0]
    dbp = y_data[:, 1]

    male_indices = (gender == 1)  # exampl male_indices => [true, true, false, true,....]
    female_indices = (gender == 0)

    duration_men = (np.sum(male_indices) * 10) / 60 / 60   # 10seconds /60 => min /60 => hour
    duration_women = (np.sum(female_indices) * 10) / 60 / 60 

    print(f"Statistics for {dataset_title}:")
    temp = np.array(ecg)
    print("ecg:", temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(ppg)
    print("ppg:", temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(abp)
    print("abp:", temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(age)
    print("age:", temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(gender)
    print("gender:",temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(height)
    print("height:",temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(weight)
    print("weight:",temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(subjectids)
    print("subjectids:",temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    temp = np.array(y_data)
    print("y_data:",temp.shape)
    print("    first element:", temp[0])
    print("    middle element (index", temp.shape[0] // 2, "):",  temp[temp.shape[0] // 2])
    print("    last element:", temp[-1])
    del temp
    gc.collect()

    df = pd.DataFrame({
        'subjectid': subjectids.flatten(),
        'age': age.flatten(),
        'gender': gender.flatten(),
        'height': height.flatten(),
        'weight': weight.flatten()
    })

    df_unique = df.drop_duplicates(subset='subjectid')

    print("Number of Subjects of dataframe:", df_unique.shape)
    print(f"Number of Subjects: {unique_subjects}")

    gender_counts = df_unique['gender'].value_counts()
    print("Number of Male Subjects:", gender_counts.get(1, 0))
    print("Number of Female Subjects", gender_counts.get(0, 0))

    # convert subject ids into string
    subject_strs = df_unique['subjectid'].astype(str)

    # subject ids end with _ 0 MIMIC  subject ids end with _1 VitalDB
    num_mimic = subject_strs.str.endswith('_0').sum()
    num_vital = subject_strs.str.endswith('_1').sum()

    print("Subjects from MIMIC:", num_mimic)
    print("Subjects from VitalDB:", num_vital)


    print(f"Age (year, mean ± SD):{df_unique['age'].mean():.2f} ± {df_unique['age'].std():.2f}")
    print(f"Minimum age (year): {np.min(age)}")
    print(f"Maximum age (year): {np.max(age)}")

    print(f"Signal duration - Male (hour): { duration_men:.2f}")
    print(f"Signal duration - Female (hour): { duration_women:.2f}")

    print(f"SBP (mmHg, mean ± SD): {np.mean(sbp):.2f} ± {np.std(sbp):.2f}")
    print(f"DBP (mmHg, mean ± SD): {np.mean(dbp):.2f} ± {np.std(dbp):.2f}")
    
    del sbp, dbp
    gc.collect() # garbage colletor to free memory

# Load to check data ----------------------------------------------------------------

def load_to_check(dataset_name='unknown.npy'):

    fields = ['ecg', 'ppg', 'abp', 'age', 'gender', 'height', 'weight','subjectids','Labels']
    data = {}
    for f in fields:
        data[f] = np.load(f'{dataset_name}_{f}.npy')
        print(f'{dataset_name}_{f} loaded: {data[f].shape}')

    compute_dataset_statistics(data['ecg'],data['ppg'],data['abp'],data['age'], data['gender'],data['height'],data['weight'],data['subjectids'], data['Labels'], dataset_title=dataset_name)
    return data


import numpy as np
# the address of the folder containing all .npy files
myData= 'mydata/'
mytrain = load_to_check(myData + 'train')
myval= load_to_check(myData + 'val')

# Combine again my train subset +  my validation subset which will be the original PulseDB training set
pulseDB = {}
for key in mytrain.keys():
    pulseDB[key] = np.concatenate((mytrain[key], myval[key]), axis=0)

compute_dataset_statistics(pulseDB['ecg'], pulseDB['ppg'], pulseDB['abp'],
                           pulseDB['age'], pulseDB['gender'], pulseDB['height'],
                           pulseDB['weight'], pulseDB['subjectids'], pulseDB['Labels'],
                           dataset_title="Train+Val Combined")


# get statistics from each subset by calling load to check function
load_to_check(myData + 'AAMI_Test')
load_to_check(myData + 'AAMI_Cal')
load_to_check(myData + 'Test_CalFree')
load_to_check(myData + 'Test_CalBased')

print('finished.')

