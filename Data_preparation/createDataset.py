import glob
import gatetools as gt
import json
import itk
import numpy as np
import pydicom
from nnunetv2.dataset_conversion import generate_dataset_json

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

folder_path = "/home/bcatez/data/nnUNet_raw/Dataset002_glands"
skull_path = "/home/bcatez/data/Skull_folder"

for patient in patients.keys():
    print(patient, "\n")
    # open the ct
    dcmFiles = glob.glob("/home/bcatez/data/originalData/" + patient + "/*/*.dcm")
    image = gt.read_dicom(dcmFiles)
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()

    # open skulls
    skull = itk.imread(skull_path + "/skull_resized/" + patients[patient] + "_0000.nii.gz")
    # Origin to center the ct
    centerorigin = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        centerorigin[i] = -spacing[i]*size[i]/2 + 0.5*spacing[i]

    # new spacing, origin and size of the extanded ct
    newspacing = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        newspacing[i] = 1.0

    newsize = itk.Size[3]()
    newsize[0] = 600
    newsize[1] = 600
    newsize[2] = 2050

    neworigin = itk.Vector[itk.D, 3]()
    neworigin[0] = -299.5
    neworigin[1] = -299.5
    neworigin[2] = -1024.5

    #create the empty image to add stuctures
    array = np.zeros([newsize[2], newsize[1], newsize[0]], dtype=np.int32) # inverted Z and x
    index = 1

    print("Label...")
    #get structures
    structFile = glob.glob("/home/bcatez/data/segmentations/" + patient + "/*__Studies/*/*.dcm")
    structset = pydicom.read_file(structFile[0])
    for r in ['Glande_Lacrim_D', 'Glande_Lacrim_G', 'Glnd_Submand_L', 'Parotid_R', 'Glnd_Submand_R', 'Parotid_L']:
        aroi = gt.region_of_interest(structset, r)
        mask = aroi.get_mask(image, corrected=False)
        mask.SetOrigin(centerorigin)
        output = gt.applyTransformation(input = mask, newspacing = newspacing, neworigin=neworigin, newsize = newsize, pad=0, interpolation_mode="NN", force_resample=True)
        structArray = itk.GetArrayFromImage(output)
        array += index*structArray
        index += 1
    structImage = itk.image_from_array(array)
    structImage.SetSpacing(newspacing)
    structImage.SetOrigin(neworigin)
    itk.imwrite(structImage, folder_path + "/labelsTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n")

    print("CT...")  
    # Center the CT
    image.SetOrigin(centerorigin)
    # resize and save the CT
    image_output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=neworigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(image_output, folder_path + "/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n")

    print("Skull...")
    # Center the skull
    skull.SetOrigin(centerorigin)
    # resize and save the skull
    skull_output = gt.applyTransformation(input = skull, newspacing = newspacing, neworigin=neworigin, newsize = newsize, pad=0, force_resample=True)
    itk.imwrite(skull_output, skull_path + "/skull_fullsized/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n______________________________________")
    print("\n")


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