#!/bin/bash
#SBATCH --output="/fp/homes01/u01/ec-balintl/FEDformer/HPC/run-fedformer.log"
#SBATCH --job-name="fedformer"
#SBATCH --time=00:00:00     # walltime
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=8G   # memory per CPU core
#SBATCH --ntasks=1
#SBATCH --account=ec69

module purge

cd /fp/homes01/u01/ec-balintl/FEDformer

apptainer exec --nv HPC/fedformer-runner.sif python3 run.py --is_training 1 --root_path data/RAVE_encoded_datasets --data_path vctk_rave_encoded.h5 --csv_path vctk_rave_encoded.csv --task_id RaveVctk --model FEDformer --data RAVEnc --seq_len 32 --label_len 16 --pred_len 8 --enc_in 8 --dec_in 8 --c_out 8 --embed token_only --itr 1 --moving_avg 6 --train_epochs 3 --num_workers 0 --batch_size 128