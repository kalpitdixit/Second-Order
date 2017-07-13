#!/bin/bash

# Change to the submission directory
cd $PBS_O_WORKDIR  

# Perform tasks
python tf_run.py
