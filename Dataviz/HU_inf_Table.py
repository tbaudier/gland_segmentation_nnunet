import numpy as np
import itk
import json
import pandas as pd

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

df = np.zeros((5,6))
count = 0

for patient in ["12","16","21","22","47"]:
    print(patient)

    inf = itk.GetArrayViewFromImage(itk.imread("/home/bcatez/data/idris/nnUNet_raw/Dataset001_glands/imagesTs_pred_/p0" + patient + "_psma.nii.gz"))
    image = itk.GetArrayViewFromImage(itk.imread("/home/bcatez/data/idris/nnUNet_raw/Dataset001_glands/imagesTs/p0" + patient + "_psma_0000.nii.gz"))
    for i in range(6):
        index = np.where(inf == i+1)
        roi = image[index]
        df[count][i] = roi.mean()
        print("Mean number ", i+1, " saved")
    count +=1
    print("___________________________\n")

df = pd.DataFrame(df)
df.columns = ['Glande_Lacrim_D', 'Glande_Lacrim_G', 'Glnd_Submand_L', 'Parotid_R', 'Glnd_Submand_R', 'Parotid_L']
df.to_csv("/home/bcatez/Documents/project/gland_segmentation_nnunet/Dataviz/HU_inf_Table.csv")
print(df)