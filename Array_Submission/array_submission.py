#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = "%s/.job" %os.getcwd()
scratch = os.environ['SCRATCH']
data_dir = os.path.join(scratch, '/project/ExplainabilityArray')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

Settings=["Base", "Random", "Energy","Gamma"]
Models=["True", "Complex", "Simple"]


lizards=["LizardA","LizardB"]

for Setting in Settings:
    for Model in Models:
        job_file=os.path.join(job_directory, Setting+Model+"s.job")
        explainer_data = os.path.join(data_dir, Setting+Model)
        

    # Create lizard directories
    mkdir_p(explainer_data)

    with open(job_file) as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --partition=daily\n")
        fh.writelines("#SBATCH --time=23:30:00\n")
        fh.writelines("#SBATCH --clusters=merlin6\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --#SBATCH --output=/data/user/grosche_w/XAI/XAI/FinalRuns/Logs/"+Setting+Params+"_out_%j.log\n")
        fh.writelines("#SBATCH --job-name="+{Setting}+{Params}+"_run\n")
        fh.writelines("""NOW=$(date +"%m-%d-%Y")\n""")
        fh.writelines("""echo "Starting time: $NOW, $NOW2"\n""")
        fh.writelines("""echo ""\n""")
        fh.writelines("""START=$(date +%s)\n""")
        fh.writelines("""module use unstable\n""")
        fh.writelines("""module load anaconda\n""")
        fh.writelines("""conda activate /data/user/grosche_w/myenv\n""")
        fh.writelines("python3 "+Setting+"Feature.py\n")
        fh.writelines("""END=$(date +%s)\n""")
        fh.writelines("""DIFF=$(( $END - $START ))\n""")
        fh.writelines("""echo "It took $DIFF seconds"\n""")
        fh.writelines("""NOW=$(date +"%m-%d-%Y")\n""")
        fh.writelines("""NOW2=$(date +"%r")""")
        fh.writelines("""echo "Ending time: $NOW, $NOW2"\n""")
        fh.writelines("""echo """"")
       
    os.system("sbatch %s" %job_file)