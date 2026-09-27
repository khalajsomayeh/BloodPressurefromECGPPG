from mat73 import loadmat
import h5py
import numpy as np
import gc  # Garbage collection

def Build_Dataset(Path, FieldName='Subset', dataset_name='unknown'):
        Data = loadmat(Path)
        Subject_ids = np.array(Data['Subset']['Subject']).astype(str)

        Signals = np.array(Data[FieldName]['Signals'])

        ECG = Signals[:, 0, :]  # 902160 * 1250

        PPG = Signals[:, 1, :]  # 902160 * 1250

        ABP = Signals[:, 2, :]  # (1250, 902160)

        # Load labels and demographic info
        SBP = np.array(Data[FieldName]['SBP'])
        SBP= SBP.reshape(-1, 1)

        DBP = np.array(Data[FieldName]['DBP'])
        DBP= DBP.reshape(-1, 1)

        Age = np.array(Data[FieldName]['Age'])
        Age= Age.reshape(-1, 1)

        Height = np.array(Data[FieldName]['Height'])
        Height = Height.reshape(-1, 1)

        Weight = np.array(Data[FieldName]['Weight'])
        Weight = Weight.reshape(-1, 1)

        # Convert Gender to numerical (0 for Female, 1 for Male)
        Gender=np.array(Data[FieldName]['Gender'])

        Gender=(Gender=='M').astype(float)
        # heigh=1 1 data point
        # number of  features= 1250+ 1250 + 1+ 1+ 1+ 1 = 2504
        # number of samples =902160
        # final array size= number of rows * number of features = 902160 * 2504

        y_train = np.hstack([SBP, DBP])

        del Signals, SBP, DBP
        gc.collect()  # Force garbage collection
        return ECG, PPG, ABP, Age, Gender, Height, Weight, Subject_ids, y_train

def prepare_save(ecg,ppg,abp,age,gender,height,weight,subjectids,Labels, dataset_name="Unknown"):

    # Save subset in format of .npy
    if "Train" not in dataset_name:
        np.save(myData +  dataset_name +'_ecg.npy',ecg)
        np.save(myData +  dataset_name +'_ppg.npy', ppg)
        np.save(myData +  dataset_name +'_abp.npy', abp)
        np.save(myData +  dataset_name +'_age.npy', age)
        np.save(myData +  dataset_name +'_gender.npy', gender)
        np.save(myData +  dataset_name +'_height.npy', height)
        np.save(myData +  dataset_name +'_weight.npy', weight)
        np.save(myData +  dataset_name +'_subjectids.npy', subjectids)
        np.save(myData +  dataset_name +'_Labels.npy', Labels)
    else:
        # Split Data
        print('Data splitting started')
        from sklearn.model_selection import train_test_split

        # Create indices for splitting
        all_indices = np.arange(len(subjectids))
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

        train_ecg = ecg[train_idx]
        val_ecg =   ecg[val_idx]

        train_ppg = ppg[train_idx]
        val_ppg =   ppg[val_idx]

        train_abp = abp[train_idx]
        val_abp =   abp[val_idx]

        train_age = age[train_idx]
        val_age =   age[val_idx]

        train_gender = gender[train_idx]
        val_gender =   gender[val_idx]

        train_height = height[train_idx]
        val_height =   height[val_idx]

        train_weight = weight[train_idx]
        val_weight =   weight[val_idx]

        train_subjectids = subjectids[train_idx]
        val_subjectids = subjectids[val_idx]

        train_y = Labels[train_idx]
        val_y = Labels[val_idx]

        # Save
        np.save(myData + 'train_ecg.npy', train_ecg)
        np.save(myData + 'train_ppg.npy', train_ppg)
        np.save(myData + 'train_abp.npy', train_abp)
        np.save(myData + 'train_age.npy', train_age)
        np.save(myData + 'train_gender.npy', train_gender)
        np.save(myData + 'train_height.npy', train_height)
        np.save(myData + 'train_weight.npy', train_weight)
        np.save(myData + 'train_subjectids.npy', train_subjectids)
        np.save(myData + 'train_Labels.npy', train_y)

        np.save(myData + 'val_ecg.npy', val_ecg)
        np.save(myData + 'val_ppg.npy', val_ppg)
        np.save(myData + 'val_abp.npy', val_abp)
        np.save(myData + 'val_age.npy', val_age)
        np.save(myData + 'val_gender.npy', val_gender)
        np.save(myData + 'val_height.npy', val_height)
        np.save(myData + 'val_weight.npy', val_weight)
        np.save(myData + 'val_subjectids.npy', val_subjectids)
        np.save(myData + 'val_Labels.npy', val_y)

        # Garbage
        del train_ecg,train_ppg, train_abp,train_age, train_gender,train_height,train_weight,train_subjectids,train_y
        del val_ecg,val_ppg,val_abp, val_age, val_gender,val_height,val_weight,val_subjectids,val_y
        gc.collect()

    # garbage memory
    del ecg, ppg,abp, age, gender, height, weight, subjectids,Labels
    gc.collect()

# specify the forlder name where the  npy file will be saved
myData= 'mydata/'

# The folder name consists of MAT files
data_folder = 'PulseDB/Subset_Files/'
Train_File = data_folder+'Train_Subset.mat'
Test_CalBased_File = data_folder+'CalBased_Test_Subset.mat'
Test_CalFree_File = data_folder+'CalFree_Test_Subset.mat'
AAMI_Test_File = data_folder+'AAMI_Test_Subset.mat'
AAMI_Cal_File = data_folder+'AAMI_Cal_Subset.mat'

print('Data preparation started.')

# call the function that recieves MAT files and saves the npy files
# Test CalBased Data-----------------------------------------------------------------
ecg,ppg,abp,age,gender,height,weight,subjectids,Labels = Build_Dataset(Test_CalBased_File,'Subset','test_calbased')
prepare_save(ecg,ppg,abp,age,gender,height,weight,subjectids, Labels, dataset_name="Test_CalBased")

# Test CalFree Data-----------------------------------------------------------------
ecg,ppg,abp,age,gender,height,weight,subjectids,Labels = Build_Dataset(Test_CalFree_File,'Subset','test_calfree')
prepare_save(ecg,ppg,abp,age,gender,height,weight,subjectids, Labels, dataset_name="Test_CalFree")

# AAMI Test Data-----------------------------------------------------------------
ecg,ppg,abp,age,gender,height,weight,subjectids,Labels  = Build_Dataset(AAMI_Test_File,'Subset','AAMI_test')
prepare_save(ecg,ppg,abp,age,gender,height,weight,subjectids,Labels,dataset_name="AAMI_Test")

# AAMI Cal Data-----------------------------------------------------------------
ecg,ppg,abp,age,gender,height,weight,subjectids,Labels = Build_Dataset(AAMI_Cal_File,'Subset','AAMI_Cal')
prepare_save(ecg,ppg,abp,age,gender,height,weight,subjectids, Labels, dataset_name="AAMI_Cal")

# Train Data -----------------------------------------------------------------
ecg,ppg,abp,age,gender,height,weight,subjectids,Labels = Build_Dataset(Train_File,'Subset','trainval')
prepare_save(ecg,ppg,abp,age,gender,height,weight,subjectids, Labels, dataset_name="Train")

print('finished.')
