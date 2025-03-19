#!/bin/bash
#SBATCH --job-name=cdo_crop
#SBATCH --array=0-42%4
#SBATCH --mail-user=laratt@umich.edu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cdo_crop_%A_%a.out
#SBATCH --error=logs/cdo_crop_%A_%a.err
#SBATCH --account=drewgron0
#SBATCH --partition=standard

# Load necessary modules
module load cdo

# Global paths
PRODUCT="NLDAS"
DATAPATH="/nfs/turbo/seas-hydro/laratt/NLDAS_monthly/NOAH"
MASKPATH="/nfs/turbo/seas-hydro/laratt/MGU_25/data/csvfile_0.csv"

# Ensure directories exist
mkdir -p $PRODUCT/processed
#mkdir -p $PRODUCT/logs

# Get list of files and select one based on the Slurm array index
FILES=($PRODUCT/symlinks/*.nc)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# Process file
filename=$(basename "$FILE")
output_file="$PRODUCT/processed/${filename%.nc}_PROCESSED.nc"

echo "Processing $filename..."
cdo -maskregion,$MASKPATH "$FILE" "$output_file"

if [[ $? -eq 0 ]]; then
    echo "Successfully processed: $output_file"
else
    echo "Error processing: $filename" >&2
fi
