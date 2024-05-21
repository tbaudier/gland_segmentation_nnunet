import gatetools as gt
import json
import itk
from nnunetv2.dataset_conversion import generate_dataset_json

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

origin_path = "/home/bcatez/data/nnUNet_raw/Dataset003_glands"
folder_path = "/home/bcatez/data/nnUNet_raw/Dataset004_glands"

for patient in patients.keys():
    print(patient,"\n")
    # open the ct
    image = itk.imread(origin_path + "/imagesTr/" + patients[patient] + "_0000.nii.gz")
    label = itk.imread(origin_path + "/labelsTr/" + patients[patient] + "_0000.nii.gz")
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()

    # Origin to center the ct
    centerorigin = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        centerorigin[i] = -spacing[i]*size[i]/2 + 0.5*spacing[i]

    # new spacing and size 
    newspacing = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        newspacing[i] = 9.0

    newsize = itk.Size[3]()
    newsize[0] = 67
    newsize[1] = 67
    newsize[2] = 42

    print("CT...")
    # Center the image
    image.SetOrigin(centerorigin)
    # resize and save
    image_output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(image_output, folder_path + "/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n")

    print("Label...")
    # Center the label
    label.SetOrigin(centerorigin)
    # resize and save
    label_output = gt.applyTransformation(input=label,newspacing=newspacing,neworigin=centerorigin, newsize=newsize,pad=0 ,interpolation_mode="NN",force_resample=True)
    itk.imwrite(label_output, folder_path + "/labelsTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n______________________________________")

generate_dataset_json.generate_dataset_json(output_folder=folder_path,
                                            channel_names={0:"CT"}, 
                                            labels={"background" : 0,
                                                    "Glande_Lacrim_D":1,
                                                    "Glande_Lacrim_G":2,
                                                    "Glnd_Submand_L":3,
                                                    "Parotid_R":4,
                                                    "Glnd_Submand_R":5,
                                                    "Parotid_L":6},
                                            num_training_cases=50,
                                            file_ending=".nii.gz")
print("Dataset.json generated succesfully.")