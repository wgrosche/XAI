#!/bin/bash -l
    
#SBATCH --partition=general  # Set this to partition=general if time is more than 24h
#SBATCH --time=23:30:00
#SBATCH --clusters=merlin6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000
#SBATCH --output=/data/user/grosche_w/XAI/XAI/FinalRuns/Logs/random_feature_out_%j.log
#SBATCH --job-name=Wilke_run_1

NOW=$(date +"%m-%d-%Y")
NOW2=$(date +"%r")
echo "Starting time: $NOW, $NOW2"
echo ""
START=$(date +%s)

# Here activate an environment if you need to, and load modules
module use unstable
module load anaconda
conda activate /data/user/grosche_w/myenv



python3 RandomFeature.py


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
NOW=$(date +"%m-%d-%Y")
NOW2=$(date +"%r")
echo "Ending time: $NOW, $NOW2"
echo ""
