from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def inference(tile_step_size : float = 0.5,
            use_gaussian : bool = True,
            use_mirroring : bool = True,
            perform_everyting_on_divice : bool = True,
            device : tuple = ('cuda', 0),
            verbose : bool = False,
            verbose_preprocessing : bool =False,
            allow_tqdm : bool = True,
            model_training_output_dir: str = 'Dataset003_glands/nnUNetTrainer__nnUNetPlans__3d_fullres',
            use_folds : tuple[int | str] = (0,),
            checkpoint_name : str = 'checkpoint_final.pth',
            list_of_lists_or_source_folder: str | list[list[str]] = 'Dataset003_glands/imagesTs',
            output_folder_or_list_of_truncated_output_files: str | list[str] | None = 'Dataset003_glands/imagesTs_pred_lowres',
            save_probabilities : bool = False,
            overwrite : bool = False,
            num_processes_preprocessing : int = 2,
            num_processes_segmentation_export : int = 2,
            folder_with_segs_from_prev_stage : str = None,
            num_parts : int = 1,
            part_id : int = 0):
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size = tile_step_size,
        use_gaussian = use_gaussian,
        use_mirroring = use_mirroring,
        perform_everything_on_device = perform_everyting_on_divice,
        device = torch.device(device),
        verbose = verbose,
        verbose_preprocessing = verbose_preprocessing,
        allow_tqdm = allow_tqdm
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, model_training_output_dir),
        use_folds = use_folds,
        checkpoint_name = checkpoint_name
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, list_of_lists_or_source_folder),
                                join(nnUNet_raw, output_folder_or_list_of_truncated_output_files),
                                save_probabilities = save_probabilities,
                                overwrite = overwrite,
                                num_processes_preprocessing = num_processes_preprocessing,
                                num_processes_segmentation_export = num_processes_segmentation_export,
                                folder_with_segs_from_prev_stage = folder_with_segs_from_prev_stage,
                                num_parts = num_parts, part_id = part_id)