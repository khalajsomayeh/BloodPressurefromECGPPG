import numpy as np
import pandas as pd
from mat73 import loadmat
import gc

Data = loadmat('Subset_Files/AAMI_Cal_Subset.mat')
Subject_ids = np.array(Data['Subset']['Subject']).astype(str)
Age = np.array(Data['Subset']['Age'])
df= pd.DataFrame({'subjectid':Subject_ids.flatten(),'age':Age.flatten()})
grouped= df.groupby('subjectid') ['age'].nunique()
print(grouped[grouped>1])

del Subject_ids
del Age
del df
del grouped
gc.collect()

Data = loadmat('Subset_Files/Train_Subset.mat')
Subject_ids = np.array(Data['Subset']['Subject']).astype(str)
Age = np.array(Data['Subset']['Age'])
df= pd.DataFrame({'subjectid':Subject_ids.flatten(),'age':Age.flatten()})
grouped= df.groupby('subjectid') ['age'].nunique()
print(grouped[grouped>1])

