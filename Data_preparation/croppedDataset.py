import gatetools as gt
import json
import itk
import definecrop
from nnunetv2.dataset_conversion import generate_dataset_json

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

origin_path = "/home/bcatez/data/nnUNet_raw/Dataset002_glands"
folder_path = "/home/bcatez/data/nnUNet_raw/Dataset003_glands"
skull_path = "/home/bcatez/data/Skull_folder"

z = definecrop.find_crop_height(skull_path=skull_path + "/skull_fullsized/", patients=patients)

print("Height of the cropped image : ",z)

for patient in patients.keys():
    print(patient, "\n")

    mni , mxi = definecrop.find_skull_limits(skull_path=skull_path + "/skull_fullsized/", patient=patients[patient], z=z)

    print(f"Crop from " + str(mni) + " to " + str(mxi) + "\n")

    # open skulls
    skull = itk.imread(skull_path + "/skull_fullsized/" + patients[patient] + "_0000.nii.gz")
    skull = itk.GetImageViewFromArray(skull[int(mni):int(mxi),:,:])

    # open the ct
    image = itk.imread(origin_path + "/imagesTr/" + patients[patient] + "_0000.nii.gz")
    image = itk.GetImageViewFromArray(image[int(mni):int(mxi),:,:])
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()

    # open the label
    label = itk.imread(origin_path + "/labelsTr/" + patients[patient] + ".nii.gz")
    label = itk.GetImageViewFromArray(label[int(mni):int(mxi),:,:])
    label = label.astype(itk.SS)

    # Origin to center the ct
    centerorigin = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        centerorigin[i] = -spacing[i]*size[i]/2 + 0.5*spacing[i]

    # new spacing and size of the croped ct
    newspacing = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        newspacing[i] = 3.0

    newsize = itk.Size[3]()
    newsize[0] = 200 # 600 when spacing = 1mm
    newsize[1] = 200 # 600 when spacing = 1mm
    newsize[2] = round(z/newspacing[0]) # z when spacing = 1mm

    print("Label...")
    # center the label
    label.SetOrigin(centerorigin)
    # resize and save the label
    label_output = gt.applyTransformation(input = label, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=0, interpolation_mode="NN", force_resample=True)
    itk.imwrite(label_output, folder_path + "/labelsTr/" + patients[patient] + ".nii.gz", compression=True)
    print("Saved\n")

    print("CT...")  
    # Center the CT
    image.SetOrigin(centerorigin)
    # resize and save the CT
    image_output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(image_output, folder_path + "/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n")

    print("Skull...")
    # Center the skull
    skull.SetOrigin(centerorigin)
    # resize and save the skull
    skull_output = gt.applyTransformation(input = skull, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=0, force_resample=True)
    itk.imwrite(skull_output, skull_path + "/skull_cropped/" + patients[patient] + "_0000.nii.gz", compression=True)
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
                                            file_ending=".nii.gz",
                                            dataset_name="Dataset003_glands")
print("Dataset.json generated succesfully.")