import itk
import numpy as np
import json

def dice(inference, label):

    result = []
    
    inf = itk.GetArrayFromImage(itk.imread(inference))
    lab = itk.GetArrayFromImage(itk.imread(label))


    for val in range(6):
        inf_ = inf.copy()
        lab_ = lab.copy()
        inf_[inf_ != (val+1)] = 0
        lab_[lab_ != (val+1)] = 0
        inf_bool = np.asarray(inf_).astype(np.bool_)
        lab_bool = np.asarray(lab_).astype(np.bool_)

        if inf_bool.shape != lab_bool.shape:
            raise ValueError("Shape mismatch: inference and label must have the same shape.")
    
    # Compute Dice coefficient
        intersection = np.logical_and(inf_bool,lab_bool)

        result.append(2.*intersection.sum()/(inf_bool.sum() + lab_bool.sum()))

    return result


def dice_result(imagesTs :list[int], dataset_name : str):
    dice_results = np.zeros((len(imagesTs),6))

    for i in range(len(imagesTs)):
        dice_results[i] = dice(inference="/home/bcatez/data/idris/nnUNet_raw/" + dataset_name + "/imagesTs_pred_/p0" + str(imagesTs[i]) + "_psma.nii.gz",
                            label="/home/bcatez/data/idris/nnUNet_raw/" + dataset_name + "/labelsTr/p0" + str(imagesTs[i]) + "_psma.nii.gz")

    file = dice_results.tolist()
    with open('/home/bcatez/data/nnUNet_raw/' + dataset_name + '/imagesTs_pred_/dice_results.json','w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, separators=(',\n', ','))

# change list according to patients ID in imagesTs

if __name__ == "__main__": 
    import sys
    dice_result(sys.argv[2:7], sys.argv[1])