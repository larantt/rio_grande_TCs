"""
Script subtracts precipitation from premade swaths out of the
larger NetCDF file and saves these new files in a separate directory.

Currently only tested using IMERG3B for 500km radius storms
"""

import xarray as xr
from pathlib import Path
import numpy as np
import os

## GLOBALS
COMBINE_FILES = "/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/IMERG_3B_TIMES_REPAIRED/merged_out.nc"

# Path to the IMERG files and TC merged output
imerg_daily_path = "/nfs/turbo/seas-hydro/laratt/IMERG_3B/total_precip"  # Update with actual path
swaths_path = "/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/IMERG_3B_TIMES_REPAIRED/"
output_path = "/nfs/turbo/seas-hydro/laratt/IMERG_3B/precip_removed_new"  # Update with actual output folder path

if COMBINE_FILES == True:
    # Define the directory containing your datasets
    data_directory = '/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/IMERG_3B_TIMES_REPAIRED'

    # Initialize a list to store the datasets
    datasets = []

    # Loop through all files in the directory
    for filename in os.listdir(data_directory):
        # Check if the file is a NetCDF file (or other formats you are using)
        if filename.endswith('.nc'):  # Change the extension if needed
            filepath = os.path.join(data_directory, filename)
            # Load the dataset
            ds = xr.open_dataset(filepath)
            datasets.append(ds)

    # Combine the datasets using combine_first
    combined_ds = datasets[0]
    for ds in datasets[1:]:
        combined_ds = combined_ds.combine_first(ds)
else:
    # open the combined file
    combined_ds = xr.open_dataset(COMBINE_FILES)


tc_times = [tstep.astype('datetime64[us]').item().strftime("%Y%m%d") for tstep in combined_ds.time.values]
for t in tc_times:
        print(f'Processing time {t}')
        fp = f'{imerg_daily_path}/3B-DAY.MS.MRG.3IMERG.{t}*'
        # open the dataset
        imerg = xr.open_mfdataset(fp)
        # now extract the day from the TC swath
        prcp = combined_ds.sel(time=t)
        prcp['precipitation'] = prcp.precipitation.fillna(0)
        # subtract the precip from this file
        result = imerg - prcp
        result = result.compute()
        result.to_netcdf(f'{output_path}/3B-DAY.MS.MRG.3IMERG.{t}.TCREMOVED.nc')
