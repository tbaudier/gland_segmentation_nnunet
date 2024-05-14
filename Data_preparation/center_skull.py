import numpy as np
import itk


def find_skull_center(skull_path = str, patient = dict):
    """
    Find the center of the skull

    Args:
        skull_path (str): path where the skull are stored. (e.g '/home/bcatez/data/Dataset002_glands/skull/')
        patient (str): patient id 

    Returns:
        int: center of the skull
    """
    skull = itk.imread(skull_path + patient + "_0000.nii.gz")
    mni = min(np.where(skull)[0])
    mxi = max(np.where(skull)[0])
    center = ((mxi-mni)/2) + mni

    return center

def crop_process(skull_path = str, patient = dict, center = int, z = int):
        """
        Define the cropping process

        Args:
            skull_path (str): path where the skull are stored. (e.g '/home/bcatez/data/Dataset002_glands/skull/')
            patient (str): patient id 
            center (int): center of the skull
            z (int): biggest skull height

        Returns:
            bool: true or false
        """

        skull = itk.imread(skull_path + patient + "_0000.nii.gz")
        distance = (skull.shape[0])-center

        if distance > z:
              return True
        else:
              return False
