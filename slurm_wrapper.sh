#!/bin/bash
###
# File: train_panoptic.sh
# Created Date: Thursday, November 19th 2020, 12:06:36 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


## Default command arguments

DEFAULT_ARGS=""

## Argument handling

helpFunction()
{
   echo ""
   echo "Usage: $0 -n jobname -o outdir -m exec_module -a '<args>'"
   echo -e "\t-n Job name"
   echo -e "\t-o Output directory to hold experiment artefacts"
   echo -e "\t-m Executable module"
   echo -e "\t-a Arguments to the executable"
   exit 1 # Exit script after printing help
}

while getopts "n:o:m:a:" opt
do
   case "$opt" in
      n ) jobname="$OPTARG" ;;
      o ) outdir="$OPTARG" ;;
      m ) module="$OPTARG" ;;
      a ) args="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if  [ -z "$jobname" ] || [ -z "$outdir" ] || [ -z "$module" ] || [ -z "$args" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo -e "Slurm Parameters :\n jobname\t - $jobname\n outdir\t\t - $outdir"

# Construct python command
PYTHON_CMD="python -m $module $DEFAULT_ARGS $args"
echo -e "Python command :\n $PYTHON_CMD"

# Create output directory if it doesn't exist
mkdir -p $outdir

# Get directory of current script and change to parent
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )
cd "$parent_path/.." || exit

# Set conda installation dir
CONDA_ROOT="/tudelft.net/staff-bulk/ewi/insy/SPCLab/craman/miniconda3/"

## Wrap and execute slurm script
sbatch <<EOT
#!/bin/bash
# Setup Slurm options
#SBATCH --job-name=$jobname                     # short name for the job
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --ntasks=1                              # total number of tasks across all nodes
#SBATCH --cpus-per-task=12                      # cpu-cores per task
#SBATCH --gres=gpu:pascal:1                     # use GPU
#SBATCH --mem=8G                                # total memory per cpu-core
#SBATCH --time=6-00:00                          # time (D-HH:MM)
#SBATCH --output=$outdir/%j.out                 # output log
#SBATCH --error=$outdir/%j.err                  # error log
#SBATCH --mail-type=begin                       # send mail when job begins
#SBATCH --mail-type=end                         # send mail when job ends
#SBATCH --mail-type=fail                        # send mail if job fails
#SBATCH --mail-user=c.a.raman@tudelft.nl        # email id

# Load the modules
module use /opt/insy/modulefiles
module load cuda/11.0
module load cudnn/

# Setup conda
export PATH="$CONDA_ROOT/bin:$PATH"
source $CONDA_ROOT/etc/profile.d/conda.sh

# Activate virtual environment
conda activate ml

# Execute the python command
$PYTHON_CMD
EOT