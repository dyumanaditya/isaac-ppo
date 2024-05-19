#!/bin/bash

#-------------------------------------------------------------------------
#SBATCH -G a100:1              # number of GPUs
#SBATCH --cpus-per-gpu 20
#SBATCH --mem=50G              # memory in GB
#SBATCH -t 0-12:00:00          # time in d-hh:mm:ss
#SBATCH -o ./slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e ./slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL   # Send an e-mail when a job stops, or fails
#SBATCH --mail-user=%u@asu.edu # Mail-to address
#SBATCH --export=NONE          # Purge the job-submitting shell environment
#-------------------------------------------------------------------------