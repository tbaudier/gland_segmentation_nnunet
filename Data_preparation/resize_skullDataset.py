import glob
import gatetools as gt
import json
import itk

f = open('/home/bcatez/data/patient.json')
patients = json.load(f)
f.close()

for patient in patients.keys():
    print(patient)
    # open the ct
    dcmFiles = glob.glob("/home/bcatez/data/originalData/" + patient + "/*/*.dcm")
    image = gt.read_dicom(dcmFiles)
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()
    origin = image.GetOrigin()

    # get the skull segmentation
    skull = itk.imread(f"/home/bcatez/data/segmentations/"+patient+"/image_CT_Ga68_1/segmentations/skull.nii.gz")
    skull_origin = skull.GetOrigin()
    
    # Origin to center the ct
    centerorigin = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        centerorigin[i] = -spacing[i]*size[i]/2 + 0.5*spacing[i]

    # new spacing, origin and size of the extanded ct
    newspacing = itk.Vector[itk.D, 3]()
    for i in range(0, 3):
        newspacing[i] = spacing[i]

    neworigin = itk.Vector[itk.D, 3]()
    neworigin[0] = centerorigin[0]
    neworigin[1] = centerorigin[1]
    neworigin[2] = centerorigin[2] + origin[2] - skull_origin[2]
  
    # Center the skull
    skull.SetOrigin(centerorigin)
    # resize and save
    output = gt.applyTransformation(input = skull, newspacing = newspacing, neworigin=neworigin, newsize = size, pad=0, force_resample=True)
    itk.imwrite(output, "/home/bcatez/data/Skull_folder/skull_resized" + patients[patient] + "_0000.nii.gz", compression=True)
    print("Saved\n______________________________________")


