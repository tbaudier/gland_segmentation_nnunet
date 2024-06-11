#!/bin/bash
#SBATCH --job-name=my_job            # nom du job
#SBATCH --nodes=1                    # on demande 1 node
#SBATCH --ntasks-per-node=1          # nombre total de taches (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (2/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (2/4 du noeud 4-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=16:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=my_job%j.out      # nom du fichier de sortie
#SBATCH --error=my_job%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

#Run this job with
# sbatch runNNUnet.slurm "Dataset001_glands" "3d_fullres" "4" "-tr nnUNetTrainer" "_100epochs" "12" "16" "21" "22" "47"

grpFolder="vax"
DATASET_NAME=$1
UNET_CONFIGURATION=$2
FOLD=4
OPTION=$3
EPOCHS=$4
IMAGESTS1=$5
IMAGESTS2=$6
IMAGESTS3=$7
IMAGESTS4=$8
IMAGESTS5=$9

module purge

# See available module with: module avail
module load gcc/8.3.1 cuda/12.1.0 #python/3.10.4

# Activate venv with this installation and python
#python -m venv env_phd
#pip install cupy itk
cd /gpfsscratch/rech/$grpFolder/commun
source venv_nnunet/bin/activate

export nnUNet_raw='/gpfsscratch/rech/vax/commun/nnUNet_raw'
export nnUNet_preprocessed='/gpfsscratch/rech/vax/commun/nnUNet_preprocessed'
export nnUNet_results='/gpfsscratch/rech/vax/commun/nnUNet_results'

# separate planning/ preprocessing if needed
# nnUNetv2_preprocess -d ${DATASET_NAME:7:3}

# run to plan and preprocess your dataset
# nnUNetv2_plan_and_preprocess -d ${DATASET_NAME:7:3} -c $UNET_CONFIGURATION --verify_dataset_integrity

# train the model
# nnUNetv2_train ${DATASET_NAME:7:3} $UNET_CONFIGURATION $FOLD $OPTION$EPOCHS # example epoch = "_100epochs"

# evaluate the model
mkdir -p nnUNet_raw/$DATASET_NAME/imagesTs_pred_
# inference
python3 gland_segmentation_nnunet/evaluation/inference.py $DATASET_NAME $UNET_CONFIGURATION $EPOCHS $FOLD

# dice
python3 gland_segmentation_nnunet/evaluation/dice.py $DATASET_NAME $IMAGESTS1 $IMAGESTS2 $IMAGESTS3 $IMAGESTS4 $IMAGESTS5
