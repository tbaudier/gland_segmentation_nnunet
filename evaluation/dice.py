import itk
import numpy as np

def dice(ar1, ar2):

    ar1 = itk.GetArrayFromImage(itk.imread(ar1))
    ar2 = itk.GetArrayFromImage(itk.imread(ar2))

    im1 = np.asarray(ar1).astype(np.bool_)
    im2 = np.asarray(ar2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    
    # Compute Dice coefficient
    intersection = np.logical_and(im1,im2)

    return 2.*intersection.sum()/(im1.sum() + im2.sum())

