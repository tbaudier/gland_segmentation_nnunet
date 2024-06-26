from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from multiprocessing import freeze_support, Process
import sys

def inference(dataset_name : str, Unet_configuration : str = "3d_fullres", epoch : str = None, fold : int = 4):    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, dataset_name + '/nnUNetTrainer'+ epoch + '__nnUNetPlans__' + Unet_configuration),
        use_folds=(fold,),
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, dataset_name + '/imagesTs'),
                                 join(nnUNet_raw, dataset_name + '/imagesTs_pred_'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    

if __name__ == '__main__':
    freeze_support()
    Process(target=inference(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4])).start()