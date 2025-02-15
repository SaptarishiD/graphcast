#! /bin/bash
#PBS -N graphcast_test
#PBS -o /home/saptarishi.dhanuka_asp25/capstone/graphcast/out1.log
#PBS -e /home/saptarishi.dhanuka_asp25/capstone/graphcast/out1.log
#PBS -l ncpus=20
#PBS -q gpu
#PBS -l host=compute3 
#PBS -k oe

module load compiler/anaconda3

conda init

source ~/.bashrc

conda activate graphcast

cd /home/saptarishi.dhanuka_asp25/capstone/graphcast/local_files

python3 -u run_graphcast_train_one_step.py --model_levels 13 --model_resolution 1.0