#!/bin/bash -l
    
#SBATCH --partition=daily  # Set this to partition=general if time is more than 24h
#SBATCH --time=03:30:00
#SBATCH --clusters=merlin6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/user/grosche_w/XAI/XAI/Code/Duffing/datagen/out_%j.log
#SBATCH --job-name=Wilke_datagen

NOW=$(date +"%m-%d-%Y")
NOW2=$(date +"%r")
echo "Starting time: $NOW, $NOW2"
echo ""
START=$(date +%s)

# Here activate an environment if you need to, and load modules
module use unstable
module load anaconda
conda activate /data/user/grosche_w/myenv



python3 data_generator.py


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
NOW=$(date +"%m-%d-%Y")
NOW2=$(date +"%r")
echo "Ending time: $NOW, $NOW2"
echo ""