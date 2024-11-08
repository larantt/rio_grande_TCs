"""
remove_precip.y
author - laratt@umich.edu

Script runs a parallelized local cluster with dask to process and extract
TC precipitation for the Rio Grande Basin, using a shapefile to subset geographically.

This is the second step in processing TC precipitation for the basin, after TC_swaths.py
and should be executed before RGB_timeseries.py.

ROUTINES
--------
* COMBINE_FILES : script will take TC swath netcdfs, which have been generated in TC_swaths.py
                  and combine all TCs into a timeseries file. This will allow duplicate TCs to
                  exist at a single timestep in the overall file.
* REMOVE_TCS    : fills all NaNed areas in the TC swaths with 0s for subtraction. Note that this 
                  is NOT a redundant step that could be remedied in TC_swaths.py, because it is 
                  also important for individual case studies to have isolated swaths of precip for
                  each TC.
* ANNUAL_FILES  : Creates an annual file for final processing from the intermediates. This file
                  contains one precipitation field with TCs and one precipitation field with TCs
                  removed from the total precipitation.
                  >> note, this can be run in parallel on an interactive node using dask's local
                  cluster, which is helpful given the IO intensive nature of creating intermediate
                  files etc. To enable this, set PARALLEL to true and request an interactive node.

OUTPUTS
-------
* if REMOVE_TCS   : NetCDF files with TCs associated with a swath file masked out of the data and set to 0
                    for the IMERG-3B files, this will create ~437 files as there are 437 TC days from 2000-2019
* if ANNUAL_FILES : NetCDF for each year with total precip and precip without TCs, subset to a shapefile of the
                    Rio Grande Basin. Note each file is pretty big (~600MB)

                    
EXECUTION
---------
if the script is being run in parallel, set the PARALLEL global to true, then launch an interactive node
using some command similar to this (note this is not optimized to this code, just my default for WRF analysis aliased in my bashrc):
    ```
    salloc  --account=climate588s001f24_class --nodes=1 --ntasks-per-node=1 --mem=50GB --cpus-per-task=1 --time=01:30:00
    ```
This will put you on an interactive node. Activate your desired python environment, ensure you have installed dask.
Then, run the following command:
>>> python remove_precip.py
This code will take advantage of the number of tasks per node and cpus per task to execute your processes in parallel.
If you do not specify parallel, the script will just loop over anything passed to dask futures.

You will see a bunch of logging commands linked to the terminal. First a test will be executed to ensure that your dask workers
are doing what you want. This could be removed, but I find it is helpful for debugging if dask crashes (which happens a lot on Great Lakes).
Then, the mask is created from the shapefile, which is annoyingly slow and I could probably speed up but havent. These are the fiona logging
commands.
Finally, you will enter the main processing script : logging will give you some indication of progress

DASK DASHBOARD
--------------
I find that the dask dashboard is extremely tempremental. I almost definietly just don't know how to use it right. However, if and when I get
it to work, it is really nice for keeping track and optimizing your code (especially when IO bound as it tracks reads and writes)
To (maybe) access the dask dashboard:
1. Check in the terminal output for a line similar to this:
  ```
  2024-11-08 08:37:20,846 - __main__ - INFO - Dask dashboard available at http://127.0.0.1:8787/status
  ```
If you are in JupyterLab and have Bokeh installed, you should just be able to launch it. Unfortunately, I hate JupyterLab more than I like
the Dask dashboard, so... over SSH:
2. open a new terminal on your local machine, and ssh with port forwarding into great lakes again:
  ```
  ssh -L 8787:localhost:8787 laratt@greatlakes.arc-ts.umich.edu
  ```
3. try opening http://localhost:8787 in your browser. If it doesnt work try the link in the notebook
4. if it still doesnt work, from your new terminal, ssh into the interactive node e.g.:
   ```
   ssh gl1234
   ```
5. if it still doesnt work give up and just use the logging on the terminal :) 

LOGGING
-------
The following steps describe logging output for expected behavior of this script
1. You will see worker nodes launch with the following message: 
    2024-11-08 08:37:20,117 - asyncio - DEBUG - Using selector: EpollSelector
2. The dask dashboard launches (godspeed hope it works):
    2024-11-08 08:37:20,846 - __main__ - INFO - Dask dashboard available at http://127.0.0.1:8787/status
3. The test (which does 10+1=11) executes. Expect a huge ugly dump of dictionaries, which you can check on later if necessary
    look for the result: 2024-11-08 08:37:25,865 - __main__ - INFO - Test result: 11
4. A bunch of GDAL and fiona logging messages about subsetting the data to a shapefile print
5. the subsetting by year starts:
    2024-11-08 08:44:27,906 - __main__ - INFO - Starting processing for year 2000
    some progress bars might also appear to keep track of these writes
6. the writes complete and the individual workers shut down once they have completed their tasks:
    2024-11-08 09:07:08,593 - __main__ - INFO - Task get_yearly_data-358937597f1407940ad4dc7ad4732c6d completed successfully.
    2024-11-08 09:07:12,704 - distributed.nanny - WARNING - Worker process still alive after 4.0 seconds, killing
yay! The script probably ran as expected. Now check the files and cry if they are wrong!
"""

### IMPORTS ###
import xarray as xr
from pathlib import Path
import os
import time
import pickle
import geopandas as gpd
import numpy as np
import dask
from shapely.geometry import Point
from dask.distributed import Client, LocalCluster, progress, wait
import logging
import gc

# Setup logger
log_file = 'script_log.log'
# log everything until you are 1000000% sure it works, then could change level to warning
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
# start logging
logger = logging.getLogger(__name__)

### GLOBALS ###
COMBINE_FILES = None    # should be a filepath to the files that would otherwise be combined, True, or None
ANNUAL_FILES = '/nfs/turbo/seas-hydro/laratt/historic_precip/combined_precip.pkl' # None of filepath to total precip with no TCs
REMOVE_TCS = False  # if True, subtract TCs out of data
SUBSET_RGB = True   # if True, subset using the rio grande basin shapefile
PARALLEL = True     # if True, subset files to RGB using dask parallelization
outpath = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/combined_precip' # where should combined precip files be saved

# Path to the IMERG files and TC merged output
imerg_daily_path = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/total_precip'   # path to IMERG files
swaths_path = '/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/IMERG_3B_TIMES_REPAIRED/' # paths to TC swaths
output_path = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/precip_removed'    # where to output files with TC precip removed
output_annual = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/combined_precip' # where to output annual precip files
rgb_shp_path = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp' # where is the shapefile for the basin

def gen_mask(yearly_data, shpfile):
    """
    Function creates a mask of the basin for the rio grande.

    Inputs
    ------
    yearly_data : xr.DataArray
        data containing lats and lons to be masked
    shpfile : shapefile
        shapefile for the basin
    
    Returns
    -------
    mask_xr : xr.DataArray 
        mask in the form of an xarray dataset
    """
    # get the shapefile geometry
    polygon = shpfile['geometry'].values[0]
    # get lats and lons (these should get passed in)
    lon = yearly_data['lon'].values
    lat = yearly_data['lat'].values

    # create a meshgrid of these & flatten to match netcdf
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    # get the lat lon points contained in the shapefile for the basin
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)])
    # get the points inside the polygon
    mask = points.within(polygon)
    logger.info('Masked to basin')

    # reshape and create xarray dataset to be used for subsetting
    mask_2d = mask.values.reshape(lon2d.shape)
    mask_xr = xr.DataArray(mask_2d, dims=('lat', 'lon'))
    return mask_xr

def get_yearly_data(infile, year, mask, outpath=output_annual):
    """
    Function subsets data for the rio grande basin and creates an annual
    file containing total and non-TC precipitation for the basin.

    Parameters
    ----------
    infile : xr.DataSet
        xarray dataset of TC removed precip for all times
    year : int
        integer of the year to subset data to
    mask : xr.DataArray
        data array to mask the data to
    outpath : string
        where to save the netcdf file
    """
    # log to terminal that file is being written
    logger.info(f'Starting processing for year {year}')
    # try to execute the subsetting routine
    try:
        # subset the combined precip data to the selected year
        yearly_data = infile.sel(time=str(year))
        # open the associated section of the total precipitation data with TCs included
        yearly_data_total = xr.open_mfdataset(f'{imerg_daily_path}/3B*{year}*')
        # enforce the subselection to a given year to be sure
        yearly_data_total = yearly_data_total.sel(time=str(year))

        # mask to the basin
        precip_masked = yearly_data['precipitation'].where(mask)
        precip_masked_total = yearly_data_total['precipitation'].where(mask)

        # reassign the masked data to the dataframe
        yearly_data['precipitation'] = precip_masked  # precip with TCs removed
        yearly_data['total'] = precip_masked_total    # total precip including TCs

        # make the directory if it doesn't exist already
        Path(outpath).mkdir(parents=True, exist_ok=True)
        filename = f'{outpath}/precipitation_{year}.nc'
        # write the data to disk
        yearly_data.to_netcdf(filename)
        logger.info(f'File for {year} written to {outpath}')
    except Exception as e:
        # if the process can't be executedf for whatever reason, the logger will raise an execption
        # and that individual task will fail. If all tasks fail, you probably have a dask issue. If only
        # one task fails, the issue is probably with that worker and it will try again on another thread.
        logger.exception(f'Exception occurred while processing year {year}: {e}')

def log_result(future):
    """
    Function logs the completion of a dask futures task

    Parameters
    ----------
    future : da.futures
        dask futures object with get_yearly_data tasks
    """
    try:
        result = future.result()  # Trigger exception here if there was a problem
        logger.info(f'Task {future.key} completed successfully.')
    except Exception as e:
        logger.exception(f'Task {future.key} failed with exception: {e}')

if __name__ == '__main__':
    # launch a local dask cluster to run extraction in parallel
    if PARALLEL:
        # Setting up a LocalCluster using 10 workers with 6GB memory each - can be optimized
        cluster = LocalCluster(n_workers=10, threads_per_worker=1, memory_limit='6GB')
        client = Client(cluster)    # set up dask dashboard (praying rn Xxx)
        logger.info(f'Dask dashboard available at {client.dashboard_link}')
        
        time.sleep(5)   # sleep to allow all workers to launch and synch with head node
        # find and print the active workers (helpful for debugging and keeping track)
        active_workers = client.scheduler_info()['workers']
        logger.info(f"Active workers: {active_workers}")
        
        if not active_workers:
            # this will indicate that the workers did not launch correctly, check the dask config
            logger.error("No workers found. Check error logs for details.")
            raise RuntimeError("No workers found. Check error logs for details.")

        # Submit a test task to verify worker functionality
        def test_task(x):
            """
            Small test function to ensure the dask workers are doing
            what they are supposed to do.

            Just adds 1 to a float.

            Parameters
            -----------
            x : float
                float to add one to

            Returns
            -------
                : float
                x + 1
            """
            return x + 1

        future = client.submit(test_task, 10) # test 10 + 1 = 11
        result = future.result() # evaluate the test task
        logger.info(f'Test result: {result}') # if this isn't 11, godspeed debugging this...

    if COMBINE_FILES:
        # if you need to make a combined file with all TC precip removed 
        # (NOTE: this should be parallelized better than just chunking w dask soon)
        if COMBINE_FILES == True:
            # if True, will combine all the swaths into a single file
            data_directory = '/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/IMERG_3B_TIMES_REPAIRED'
            # list all the TC swath files for IMERG-3B (make flexible in future)
            datasets = [xr.open_dataset(os.path.join(data_directory, filename)) for filename in os.listdir(data_directory) if filename.endswith('.nc')]
            # combine the datasets, start with data at 0
            combined_ds = datasets[0]
            for ds in datasets[1:]:
                # loop through and combine_first. This ensures that if there are two swaths at one time
                # aka two TCs active at the same time, they are combined into one file for that day
                combined_ds = combined_ds.combine_first(ds)
                # dump as pickle (Not long term storage, just useful until I optimize this step better)
                with open(f'{output_path}/combined_precip.pkl', 'wb') as f:
                    pickle.dump(combined_ds, f)
        else:
            # NOTE: right now this is a really bad redundant code block bc legacy code. NEED TO CHANGE!!!!!
            # NOTE: probably best to keep this and change the ANNUAL_FILES part to do the same thing. For now, only specify one.
            # if combined_ds is a filepath to combined data, then open the file
            try: 
                # try an xarray dataset (this means I did good thing and wrote the netcdf or that swaths are yet to be removed)
                # this depends on context (e.g. did I specify REMOVE_TCs = True)
                combined_ds = xr.open_dataset(COMBINE_FILES)
            except:
                # I am a bad person and kept it as a pickle dump
                with open(COMBINE_FILES, 'rb') as f:
                    combined_ds = pickle.load(f)

    # if we want to mask TCs out completely (aka we only have swaths rn) 
    # NOTE: this NEEDS to be parallelized with dask. Very simple task - do before processing TRMMM and PERSIANN-CDR
    if REMOVE_TCS:
        ### NOTE TO NICK - this is an example of how you could add a calculation. I didn't parallelize this step yet, but
        ### you just wrap this in a function the same way I did the next clause for if ANNUAL_FILES:

        # get all times that TCs were active
        tc_times = [tstep.astype('datetime64[us]').item().strftime('%Y%m%d') for tstep in combined_ds.time.values]
        # loop over times that TCs are active
        for t in tc_times:
            logger.info(f'Processing time {t}')
            # open IMERG file(s) for time TC is active
            fp = f'{imerg_daily_path}/3B-DAY.MS.MRG.3IMERG.{t}*'
            imerg = xr.open_mfdataset(fp)
            # from combined data, get the total precipitation
            prcp = combined_ds.sel(time=t)
            prcp['precipitation'] = prcp.precipitation.fillna(0) # fill NaNs in the swath with 0s
            # subtract the TC swath from the full precip field
            result = imerg - prcp
            # compute the result before writing (can remove when parallelized!)
            result = result.compute()
            # write to netcdf (note that TCs have been removed, could add name but kinda cba)
            result.to_netcdf(f'{output_path}/3B-DAY.MS.MRG.3IMERG.{t}.TCREMOVED.nc')

    # do we want to write files with annual totals (probably thats the main goal of this study)
    if ANNUAL_FILES:
        # this is the terrible redundant bit I need to fix - rly the same as combined_ds here
        # this file is just the annual precip with TCs removed
        if not os.path.exists(ANNUAL_FILES):
            # check if the file exists
            raise FileNotFoundError(f'Annual file not found: {ANNUAL_FILES}')

        # read in file if it hasn't already been made (again, need to fix this to be cleaner)
        with open(ANNUAL_FILES, 'rb') as f:
            loaded_combined_precip = pickle.load(f)

        # get individual years in the annual precipitation file
        years = np.unique(loaded_combined_precip['time.year'])

        # do we want to execute the script in parallel with a dask local cluster?
        if PARALLEL:
            # open the shapefile for the basin (lowkey I think I need to fix the crs here but anyway)
            rgb_shp = gpd.read_file(rgb_shp_path)
            # create the mask using just the first time slice
            mask_xr = gen_mask(loaded_combined_precip.isel(time=0), rgb_shp)

            # create a lisk of tasks for dask to distribute to workers
            futures = []
            for year in years:
                # loop over years and submit each task
                # NOTE: I tried to do this more efficiently by scattering tasks but for some reason it didn't work
                # NOTE: but this does, and its relatively quick. It gives one warning about memory but it doesn't seem
                # NOTE: to cause any noticable slowdown to me and behaves more predictably sooooooo........
                future = client.submit(get_yearly_data, loaded_combined_precip, year, mask_xr)
                future.add_done_callback(log_result)  # terminal logging command
                futures.append(future)   # append to list of running tasks to keep track
            
            # get nice progress bar (its not that nice might remove)
            progress(futures)
            # wait until all tasks are complete and all files have been written
            wait(futures)
            # close client and kill workers
            client.close()
        else:
            # if not running in parallel, just does the same thing but in a slow and painful loop :)
            rgb_shp = gpd.read_file(rgb_shp_path)
            mask_xr = gen_mask(loaded_combined_precip.isel(time=0), rgb_shp)
            for year in years:
                start_time = time.time()
                get_yearly_data(loaded_combined_precip, year, mask_xr)
                end_time = time.time()
                logger.info(f'Processed {year} in {end_time - start_time:.2f} seconds')