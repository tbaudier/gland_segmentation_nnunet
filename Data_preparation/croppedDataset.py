import gatetools as gt
import json
import itk
from Data_preparation import definecrop
from Data_preparation import center_skull

f = open('patient.json')
patients = json.load(f)
f.close()

path = "/home/bcatez/data/Dataset002_glands/skull/"

z = definecrop.find_crop_height(skull_path=path, patients=patients)

print("Height of the cropped image : ",z)

for patient in patients.keys():
    print(patient)

    mni , mxi = definecrop.find_skull_limits(skull_path=path, patient=patient, z=z)

    # open skulls
    skull = itk.imread(f"/home/bcatez/data/Dataset002_glands/skull/" + patients[patient] + "_0000.nii.gz")
    skull = itk.GetImageViewFromArray(skull[mni:mxi,:,:])

    # open the ct
    image = itk.imread(f"/home/bcatez/data/Dataset002_glands/imagesTr/" + patients[patient] + "_0000.nii.gz")
    image = itk.GetImageViewFromArray(image[mni:mxi,:,:])
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()

    # open the label
    label = itk.imread("/home/bcatez/data/Dataset002_glands/labelsTr/" + patients[patient] + "_0000.nii.gz")
    label = itk.GetImageViewFromArray(label[mni:mxi,:,:])
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


    # center the label
    label.SetOrigin(centerorigin)
    # resize and save the label
    label_output = gt.applyTransformation(input = label, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=0, interpolation_mode="NN", force_resample=True)
    itk.imwrite(label_output, "Dataset003_1_glands/labelsTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("label saved")
        
    # Center the CT
    image.SetOrigin(centerorigin)
    # resize and save the CT
    image_output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(image_output, "Dataset003_1_glands/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("CT saved")

    # Center the skull
    skull.SetOrigin(centerorigin)
    # resize and save the skull
    skull_output = gt.applyTransformation(input = skull, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=0, force_resample=True)
    itk.imwrite(skull_output, "Dataset003_1_glands/skulls/" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Skull saved")