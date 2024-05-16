import numpy as np
import itk
import json
import pandas as pd
import volume

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

df = np.zeros((50,6))
count = 0

for patient in patients.keys():
    print(patient,"\n")

    label = itk.imread(f"/home/bcatez/data/Dataset002_glands/labelsTr/" + patients[patient] + "_0000.nii.gz")

    df[count] = volume.volume_calculator(label=label)
    count +=1

df = pd.DataFrame(df)
df.columns = ['Glande_Lacrim_D', 'Glande_Lacrim_G', 'Glnd_Submand_L', 'Parotid_R', 'Glnd_Submand_R', 'Parotid_L']
df.to_csv("/home/bcatez/Documents/project/gland_segmentation_nnunet/Dataviz/ROITable.csv")
print(df)