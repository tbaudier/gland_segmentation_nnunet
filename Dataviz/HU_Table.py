import numpy as np
import itk
import json
import pandas as pd

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

df = np.zeros((50,6))
count = 0

for patient in patients.keys():
    print(patient)

    label = itk.GetArrayViewFromImage(itk.imread(f"/home/bcatez/data/Dataset002_glands/labelsTr/" + patients[patient] + "_0000.nii.gz"))
    image = itk.GetArrayViewFromImage(itk.imread(f"/home/bcatez/data/Dataset002_glands/imagesTr/" + patients[patient] + "_0000.nii.gz"))
    for i in range(6):
        index = np.where(label == i+1)
        roi = image[index]
        df[count][i] = roi.mean()
        print("Mean number ", i+1, " saved")
    count +=1
    print("___________________________\n")

df = pd.DataFrame(df)
df.columns = ['Glande_Lacrim_D', 'Glande_Lacrim_G', 'Glnd_Submand_L', 'Parotid_R', 'Glnd_Submand_R', 'Parotid_L']
df.to_csv("/home/bcatez/Documents/project/gland_segmentation_nnunet/Dataviz/HUTable.csv")
print(df)