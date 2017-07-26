#!/bin/bash
# Change to the submission directory
cd $PBS_O_WORKDIR  

# choose gpu
export CUDA_VISIBLE_DEVICES=1

# Perform tasks
python generate_runs.py mnist ff 0.5 adadelta 100 50
