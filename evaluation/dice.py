import itk
import numpy as np

def dice(inf, label):

    result = []
    
    inf = itk.GetArrayFromImage(itk.imread(inf))
    label = itk.GetArrayFromImage(itk.imread(label))


    for val in range(1,7):
        inf_ = inf.copy()
        label_ = label.copy()
        inf_[inf_ != val] = 0
        label_[label_ != val] = 0
        inf_bool = np.asarray(inf_).astype(np.bool_)
        label_bool = np.asarray(label_).astype(np.bool_)

        if inf_bool.shape != label_bool.shape:
            raise ValueError("Shape mismatch: inference and label must have the same shape.")
    
    # Compute Dice coefficient
        intersection = np.logical_and(inf_bool,label_bool)

        result.append(2.*intersection.sum()/(inf_bool.sum() + label_bool.sum()))

    return result
