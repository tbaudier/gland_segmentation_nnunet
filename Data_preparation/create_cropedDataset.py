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

    # open skulls
    skull = itk.imread(f"/home/bcatez/data/Datasettest_glands/skull/" + patients[patient] + "_0000.nii.gz")
    crop_skull = itk.GetArrayViewFromImage(skull)
    mni, mxi = np.where(crop_skull)[0][0] - 50 , np.where(crop_skull)[0][-1] + 50
    skull = itk.GetImageViewFromArray(skull[mni:mxi,:,:])

    # open the ct
    dcmFiles = glob.glob("/home/bcatez/data/originalData/" + patient + "/*/*.dcm")
    image_o = gt.read_dicom(dcmFiles)
    image = itk.GetImageViewFromArray(image_o[mni:mxi,:,:])
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()

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
    newsize[2] = 125 # 375 when spacing = 1mm


    #create the empty image to add stuctures
    array = np.zeros([newsize[2], newsize[1], newsize[0]], dtype=np.int32) # inverted Z and x
    index = 1

    #get structures
    structFile = glob.glob("/home/bcatez/data/segmentations/" + patient + "/*__Studies/*/*.dcm")
    structset = pydicom.read_file(structFile[0])
    for r in ['Glande_Lacrim_D', 'Glande_Lacrim_G', 'Glnd_Submand_L', 'Parotid_R', 'Glnd_Submand_R', 'Parotid_L']:
        aroi = gt.region_of_interest(structset, r)
        mask = aroi.get_mask(image_o, corrected=False)
        mask = itk.GetImageViewFromArray(mask[mni:mxi,:,:])
        mask.SetOrigin(centerorigin)
        output = gt.applyTransformation(input = mask, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=0, interpolation_mode="NN", force_resample=True)
        structArray = itk.GetArrayFromImage(output)
        array += index*structArray
        index += 1
    structImage = itk.image_from_array(array)
    structImage.SetSpacing(newspacing)
    structImage.SetOrigin(centerorigin)
    itk.imwrite(structImage, "Dataset003_glands/labelsTr/" + patients[patient] + "_0000.nii.gz", compression=True)
        
    # Center the CT
    image.SetOrigin(centerorigin)
    # resize and save the CT
    image_output = gt.applyTransformation(input = image, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=-1024, force_resample=True)
    itk.imwrite(image_output, "Dataset003_glands/imagesTr/" + patients[patient] + "_0000.nii.gz", compression=True)

    # Center the skull
    skull.SetOrigin(centerorigin)
    # resize and save the skull
    skull_output = gt.applyTransformation(input = skull, newspacing = newspacing, neworigin=centerorigin, newsize = newsize, pad=0, force_resample=True)
    itk.imwrite(skull_output, "Dataset003_glands/skulls/" + patients[patient] + "_0000.nii.gz", compression=True)
