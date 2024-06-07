import itk
import numpy as np
import json

def dice(inf, label):

    result = []
    
    inf = itk.GetArrayFromImage(itk.imread(inf))
    label = itk.GetArrayFromImage(itk.imread(label))


    for val in range(6):
        inf_ = inf.copy()
        label_ = label.copy()
        inf_[inf_ != (val+1)] = 0
        label_[label_ != (val+1)] = 0
        inf_bool = np.asarray(inf_).astype(np.bool_)
        label_bool = np.asarray(label_).astype(np.bool_)

        if inf_bool.shape != label_bool.shape:
            raise ValueError("Shape mismatch: inference and label must have the same shape.")
    
    # Compute Dice coefficient
        intersection = np.logical_and(inf_bool,label_bool)

        result.append(2.*intersection.sum()/(inf_bool.sum() + label_bool.sum()))

    return result


def dice_result(imagesTs :list[int], dataset_name : str):
    dice_results = np.zeros((len(imagesTs),6))

    for i in range(len(imagesTs)):
        dice_results[i] = dice(inf="/home/bcatez/data/nnUNet_raw/" + dataset_name + "/imagesTs_pred_/p0" + str(imagesTs[i]) + "_psma.nii.gz",
                            label="/home/bcatez/data/nnUNet_raw/" + dataset_name + "/labelsTr/p0" + str(imagesTs[i]) + "_psma.nii.gz")

    file = dice_results.tolist()
    with open('/home/bcatez/data/nnUNet_raw/' + dataset_name + '/imagesTs_pred_/dice_results.json','w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, separators=(',\n', ','))

# change list according to patients ID in imagesTs

if __name__ == "__main__": 
    import sys
    dice_result(sys.argv[2:-1], sys.argv[1])