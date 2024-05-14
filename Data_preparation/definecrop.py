import itk
import numpy as np

def find_crop_height(skull_path = str , patients = dict):
    '''
    Find the highest and lowest limit to crop for the entire dataset.

    Args:
        skull_path (str): path where the skull are stored. (e.g '/home/bcatez/data/DatasetSkull_glands/skull/')
        patients (dict): patients list ( a dict found in the patient.json file)

    Returns:
        int: the biggest skull size + 100.
    '''
    # Initiate the list
    z_list = []
    # Look into all patients
    for patient in patients.keys():
        print(patient)
        # Get the skull
        skull = itk.imread(skull_path + patients[patient] + "_0000.nii.gz")
        # Append height in the list
        z_list.append((max(np.where(skull)[0]))-(min(np.where(skull)[0])))
    # Sort the list in order to send the highest number at the end of the list
    z_list.sort()

    print("limits found")

    return z_list[-1] + 100 # Return the highest height + 100



def find_skull_limits(skull_path = str, patient = str, z = int):
    """
    Find the center of the skull

    Args:
        skull_path (str): path where the skull are stored. (e.g '/home/bcatez/data/Dataset002_glands/skull/')
        patient (str): patient id
        z (int): biggest skull height

    Returns:
        tuple: lower limit and upper limit of the crop
    """
    # Get the skull
    skull = itk.imread(skull_path + patient + "_0000.nii.gz")
    # Find skull limits
    mni = min(np.where(skull)[0]) - 50
    mxi = max(np.where(skull)[0]) + 50
    # Find skull center
    center = ((mxi-mni)/2) + mni
    # Find distance between skull center and image upper limit
    distance = (skull.shape[0])-center

    # If distance is greater than the max distance
    if distance > round(z/2):
          # Define limits as they are
          return mni, mxi
    else:
          # Define limits from max distance
          return skull.shape[0] - z , skull.shape[0]
