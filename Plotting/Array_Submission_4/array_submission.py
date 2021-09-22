#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = "%s/.job" %os.getcwd()
#scratch = os.environ['SCRATCH']
#data_dir = os.path.join(scratch, '/project/ExplainabilityArray')

# Make top level directories
mkdir_p(job_directory)
#mkdir_p(data_dir)

Settings=["Base", "Random", "Energy","Gamma"]
Models=["Complex", "Simple"]# ["True", 
Params = [{'alpha' : 1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2}, 
          {'alpha' : 1.0, 'beta' : -0.5, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2},
          {'alpha' : 1.0, 'beta' : -0.5, 'gamma' : 0.37, 'delta' : 1.0, 'omega' : 1.2}, 
          {'alpha' : 1.0, 'beta' : -0.5, 'gamma' : 0.5, 'delta' : 0.3, 'omega' : 1.2},
          {'alpha' : 1.0, 'beta' : -0.5, 'gamma' : 0.37, 'delta' : 0.0, 'omega' : 1.2},
          {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2},
          {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 1.0, 'omega' : 1.2}, 
          {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.5, 'delta' : 0.3, 'omega' : 1.2},
          {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.0, 'delta' : 0.3, 'omega' : 0.0},
          {'alpha' : -1.0, 'beta' : -1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2},
          {'alpha' : 0.0, 'beta' : 0.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2}]

#lizards=["LizardA","LizardB"]

for Setting in Settings:
    for Model in Models:
        for param_idx, Param in enumerate(Params):
            job_file=os.path.join(job_directory, Setting+Model+".job")
            #explainer_data = os.path.join(data_dir, Setting+Model)


            # Create lizard directories
            #mkdir_p(explainer_data)

            with open(job_file, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --partition=daily\n")
                fh.writelines("#SBATCH --time=23:30:00\n")
                fh.writelines("#SBATCH --clusters=merlin6\n")
                fh.writelines("#SBATCH --nodes=1\n")
                fh.writelines("#SBATCH --ntasks=1\n")
                fh.writelines("#SBATCH --ntasks-per-node=1\n")
                fh.writelines("#SBATCH --cpus-per-task=1\n")
                fh.writelines("#SBATCH --output=/data/user/grosche_w/XAI/XAI/Array_Submission/Logs/"+Setting+"_"+Model+"_"+str(param_idx)+"_out_%j.log\n")
                fh.writelines("#SBATCH --job-name="+Setting+"_"+str(param_idx)+"_run\n")
                fh.writelines("""NOW=$(date +"%m-%d-%Y")\n""")
                fh.writelines("""NOW2=$(date +"%r")\n""")
                fh.writelines("""echo "Starting time: $NOW, $NOW2"\n""")
                fh.writelines("""echo ""\n""")
                fh.writelines("""START=$(date +%s)\n""")
                fh.writelines("""module use unstable\n""")
                fh.writelines("""module load anaconda\n""")
                fh.writelines("""conda activate /data/user/grosche_w/myenv\n""")
                fh.writelines("python3 ArraySubmission.py "+str(param_idx)+" "+Model+" "+Setting+"\n")
                fh.writelines("""END=$(date +%s)\n""")
                fh.writelines("""DIFF=$(( $END - $START ))\n""")
                fh.writelines("""echo "It took $DIFF seconds"\n""")
                fh.writelines("""NOW=$(date +"%m-%d-%Y")\n""")
                fh.writelines("""NOW2=$(date +"%r")\n""")
                fh.writelines("""echo "Ending time: $NOW, $NOW2"\n""")
                fh.writelines("""echo "" """)

            os.system("sbatch %s" %job_file)