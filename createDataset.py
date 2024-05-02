import glob
import gatetools as gt
import json
import itk
import numpy as np
import pydicom

f = open('patient.json')
patients = json.load(f)
f.close()

for patient in patients.keys():
    print(patient)
    # open the ct
    dcmFiles = glob.glob("../data/originalData/" + patient + "/*/*.dcm")
    image = gt.read_dicom(dcmFiles)
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()

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
    for i in range(0, 3):
        neworigin[i] = -newspacing[i]*newsize[i]/2.0 + 0.5*newsize[i]

    #create the empty image to add stuctures
    array = np.zeros([newsize[2], newsize[1], newsize[0]], dtype=np.int32) # inverted Z and x
    index = 1

    #get structures
    structFile = glob.glob("../data/segmentations/" + patient + "/*__Studies/*/*.dcm")
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
    itk.imwrite(structImage, "../data/Dataset001_glands/labelsTr/" + patients[patient] + "_0000.nii.gz", compression=True)
        
    # Center the CT
    image.SetOrigin(centerorigin)
    # resize and save
    output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=neworigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(output, "../data/Dataset001_glands/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)


