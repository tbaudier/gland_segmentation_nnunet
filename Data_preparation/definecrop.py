import itk
import numpy as np

def find_crop_limits(skull_path = str , patients = dict):
    '''
    Find the highest and lowest limit to crop for the entire dataset.

    Args:
        skull_path (str): enter the path where the skull are stored. (e.g '/home/bcatez/data/DatasetSkull_glands/skull/')
        patients (disct): enter the patients list ( a dict found in the patient.json file)

    Returns:
        int: the biggest skull size + 100.
    '''
    z_list = []


    for patient in patients.keys():
        print(patient)
        skull = itk.imread(skull_path + patients[patient] + "_0000.nii.gz")
        z_list.append((max(np.where(skull)[0]))-(min(np.where(skull)[0])))

    z_list.sort()

    print("limits found")

    return z_list[-1] + 100
