import gatetools as gt
import json
import itk

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

for patient in patients.keys():
    print(patient)
    # open the ct
    image = itk.imread(f"/home/bcatez/data/Dataset003_1_glands/imagesTr/" + patients[patient] + "_0000.nii.gz")
    label = itk.imread(f"/home/bcatez/data/Dataset003_1_glands/labelsTr/" + patients[patient] + "_0000.nii.gz")
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

  
    # Center the image
    image.SetOrigin(centerorigin)
    # resize and save
    image_output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(image_output, "/home/bcatez/data/Dataset004_glands/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)

    # Center the label
    label.SetOrigin(centerorigin)
    # resize and save
    label_output = gt.applyTransformation(input=label,newspacing=newspacing,neworigin=centerorigin, newsize=newsize,pad=0 ,interpolation_mode="NN",force_resample=True)
    itk.imwrite(label_output, "/home/bcatez/data/Dataset004_glands/labelsTr/" + patients[patient] + "_0000.nii.gz", compression=True)