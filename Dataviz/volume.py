import numpy as np
import itk


def volume_calculator(label = any):
    """
    Give the volume in mm3 of each ROI

    Args:
        label (any): image from Dataset002

    Returns:
        tuple: in order the volume of each ROI
    """
    volume = itk.GetArrayViewFromImage(label)  
    vol_1 = volume.copy()
    vol_2 = volume.copy()
    vol_3 = volume.copy()
    vol_4 = volume.copy()
    vol_5 = volume.copy()
    vol_6 = volume.copy()
    vol_1[vol_1 != 1] = 0
    vol_2[vol_2 != 2] = 0
    vol_3[vol_3 != 3] = 0
    vol_4[vol_4 != 4] = 0
    vol_5[vol_5 != 5] = 0
    vol_6[vol_6 != 6] = 0
    vol_1 = np.count_nonzero(vol_1)
    vol_2 = np.count_nonzero(vol_2)
    vol_3 = np.count_nonzero(vol_3)
    vol_4 = np.count_nonzero(vol_4)
    vol_5 = np.count_nonzero(vol_5)
    vol_6 = np.count_nonzero(vol_6)
    return vol_1, vol_2, vol_3, vol_4, vol_5, vol_6
    