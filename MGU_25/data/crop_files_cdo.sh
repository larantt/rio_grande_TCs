#!/bin/bash

#--------------------------------------------------#
# crop_files_cdo.sh                                #
#                                                  #
# Bash script to crop files to the RGB using CDO   #
#--------------------------------------------------#

# Global paths
PRODUCT="NLDAS"
DATAPATH="/nfs/turbo/seas-hydro/laratt/NLDAS_monthly/NOAH"
MASKPATH="/nfs/turbo/seas-hydro/laratt/MGU_25/data/csvfile_0.csv"

# Ensure directory is correct
cd /nfs/turbo/seas-hydro/laratt/MGU_25/data

# Create necessary directories
mkdir -p $PRODUCT/symlinks
mkdir -p $PRODUCT/processed

# Remove old symlinks (if any)
rm -f $PRODUCT/symlinks/*

# Create symbolic links to all files you want to process
ln -s ${DATAPATH}/*.nc $PRODUCT/symlinks/

# Loop over all files and process
#for file in $PRODUCT/symlinks/*.nc; do
#    filename=$(basename "$file")
#    output_file="$PRODUCT/processed/${filename%.nc}_PROCESSED.nc"
#    
#    echo "Processing $filename..."
#    cdo -maskregion,$MASKPATH "$file" "$output_file"
#    
#    if [[ $? -eq 0 ]]; then
#        echo "Successfully processed: $output_file"
#    else
#        echo "Error processing: $filename" >&2
#    fi
#done

echo "All files processed."
