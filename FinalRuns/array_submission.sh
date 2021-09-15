#!/bin/bash

# We assume running this from the script directory
job_directory=$PWD/.job
data_dir="${SCRATCH}/project/LizardLips"

Settings=("Base" "Random" "Energy" "Gamma")

for Setting in ${Settings[@]}; do

    job_file="${job_directory}/${lizard}.job"

    echo "#!/bin/bash
#SBATCH --partition=general  # Set this to partition=general if time is more than 24h
#SBATCH --time=48:00:00
#SBATCH --clusters=merlin6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000
#SBATCH --output=/data/user/grosche_w/XAI/XAI/FinalRuns/Logs/${Setting}_${Params}_out_%j.log
#SBATCH --job-name=${Setting}_${Params}_run
NOW=$(date +"%m-%d-%Y")
NOW2=$(date +"%r")
echo "Starting time: $NOW, $NOW2"
echo ""
START=$(date +%s)

# Here activate an environment if you need to, and load modules
module use unstable
module load anaconda
conda activate /data/user/grosche_w/myenv



python3 ${Setting}Feature.py --


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
NOW=$(date +"%m-%d-%Y")
NOW2=$(date +"%r")
echo "Ending time: $NOW, $NOW2"
echo ""
 > $job_file
    sbatch $job_file

done