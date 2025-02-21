#!/usr/bin/env python3
"""
Author: Lara Tobias-Tarsh (laratt@umich.edu)
Created: October 2nd 2024

TC_swaths.py

This script extracts TC precipitation swaths for all TCs entering the 
Rio Grande Basin from a given dataset. At time of creation, this is 
tested with TRMM/GPM IMERG merged data. The next dataset to be added
will be MSWEP.

The output of this script is intended to create a database of 
lifetime TC associated precipitation for TCs affecting the basin, including
precipitation that did not fall in the basin.

Swaths are created using TC-best track data from the extended best track 
database, and consider a 500km radius around the TC center. Hopefully,
this will be extended through the use of tracking algorithms to include
the track of TC remnants in the future.

Further processing in a seperate script will be performed to calculate
the contribution to overall basin precipitation, and to remove the TC
associated precipitation from the historical timeseries.
"""

#############
## IMPORTS ##
#############
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import utils as md
import matplotlib.colors as mcolors
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import os
import matplotlib.colors as mcolors
import pyproj

###########
# GLOBALS #
###########

# filepaths to data
IMERG_PATH = "/nfs/turbo/seas-hydro/laratt/daymet/"
IMERG_EXTN = '_basin.nc'
SHP_RGB = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp'
EBT_AL = '/nfs/turbo/seas-hydro/laratt/TCdata/track_files/EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt'
EBT_EP = '/nfs/turbo/seas-hydro/laratt/TCdata/track_files/EBTRK_EP_final_1949-2021_new_format_02-Sep-2022.txt'
SWATH_OUTPATH = "/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/daymet"

# debugging/testing globals - if none, will just do the entire record of storms read in
PLOT_SWATHS = False        # bool - plot each swath for verification if True (not recommended for long periods)
TRACKS_FROM_DISK = False #"/nfs/turbo/seas-hydro/laratt/TCdata/track_files/2024-10-03_tracks_2000-2020.pkl"    # string filepath - read in clean data that has been saved to disk to skip IO and cleaning EBT overhead
SAVE_CLEAN_TRACKS = False   # bool - should the cleaned EBT data be saved to disk as a csv? False if TRACKS_FROM_DISK specified

# globals for swath extraction
SOURCE = 'Daymet'          # precipitation product being used
TC_RADIUS = 500            # radius in km about which to extract TC precip
PREC_VAR = 'prcp'          # variable in precipitation data for precip
LAT_VAR = 'x'              # variable for latitude in precipitation data
LON_VAR = 'y'              # variable for longitude in precipitation data
START_YR = 1980            # year to start data extraction
END_YR = 2020              # year to end data extraction
STORM_ID =  'AL012010' #None            # run test for individual TC - use the ATCF ID from the EBTRAK dataset (e.g. 'AL012010' for Alex)
BASIN = None               # EBT_AL or EBT_EP - basin to extract swaths for (None will extract both)

#--------------- begin extraction ---------------#
print('beginning extraction')
# get the date of the run for use in attrs
today = dt.date.today()

## read in TC track data and basin shapefile
rgb_shp = gpd.read_file(SHP_RGB)

# if a cleaned track file has been supplied, read in now
if TRACKS_FROM_DISK:
    database = pd.read_pickle(TRACKS_FROM_DISK)
# if a cleaned track file has not been supplied, clean the EBT files    
else:
    # if basin, read in specific basin
    if BASIN:
        track_df = md.ReadEBTFile(BASIN)
    # else read in both and merge into one dataframe
    else:
        AL_df = md.ReadEBTFile(EBT_AL)
        EP_df = md.ReadEBTFile(EBT_EP)
        track_df = pd.concat([AL_df,EP_df])

    # clean data and extract intersecting TCs
    database, intersecting = md.clean_data(track_df,rgb_shp,TC_RADIUS,START_YR,END_YR)
    print('TC tracks read in')

# save track file if specified
if SAVE_CLEAN_TRACKS:
    # now pickle the dataframe
    database.to_pickle(f'/nfs/turbo/seas-hydro/laratt/TCdata/track_files/{today}_tracks_{START_YR}-{END_YR}.pkl')
    print(f'TC tracks saved to: {today}_tracks_{START_YR}-{END_YR}.pkl ')

## if STORM_ID, subset data to specific storm
if STORM_ID:
    database = database[database.track_id == STORM_ID]
    print(f'subset to {STORM_ID}')

## make a list of all days TC occuring precip is needed for each TC
imerg_times = pd.to_datetime(database['date']).dt.strftime('%Y%m%d').unique().tolist()
# now format the filepaths to all necessary IMERG files (NOTE this needs to be changed depending on ur source)
#imerg_files = [f'{IMERG_PATH}3B-DAY.MS.MRG.3IMERG.{time}{IMERG_EXTN}' for time in imerg_times]
imerg_files = f'{IMERG_PATH}*{IMERG_EXTN}'

## lazy load data for all active precip days using Dask and xarray
# for some reason it is much faster to decode cf after loading...
precip_ds = xr.open_mfdataset(imerg_files,engine='netcdf4')
#precip_ds = xr.open_mfdataset(imerg_files,decode_times=False,decode_cf=False,engine='netcdf4',parallel=True)

print('Loaded precip data')
print(precip_ds.info())

### EXTRACTION ###
## loop over all the TCs in the database
for track in database.track_id.unique():
    # get the data for specific storm
    storm = database[database.track_id == track]
    # get the name, used to save file later (probably a better way than this but oh well)
    name = storm.name.unique()[0]
    print(f'Processing {track} ({name})')

    # get all of the TC centres for a given storm
    centres_rect = list(zip(storm['lon'], storm['lat']))
    # convert centres to daymet lcc grid
    lcc_crs = pyproj.CRS.from_wkt(precip_ds.lambert_conformal_conic.attrs['crs_wkt'])
    centres = [md.latlon_to_lcc(lat, lon, lcc_crs) for lon, lat in centres_rect]
    # get all the times the TC is active
    times = pd.to_datetime(storm['date'])
    # extract the precip swath for that TC
    # note that the output has the same dimensions as an individual IMERG output file
    ds = daily_radial_precip_lcc(centres, times, precip_ds, 
                                precip_var=PREC_VAR, lat_var=LAT_VAR, lon_var=LON_VAR, radius=TC_RADIUS)

    # assign attributes to dataset
    ds.attrs = {
    "title": f"Precipitation Swath for {track} (TC {name})",
    "description": f"Dataset contains accumulated precipitation swath data extracted from {SOURCE} and Extended Best Track Data",
    "start_date": f"{times.values[0]}",
    "end_date": f"{times.values[-1]}",
    "history": f"Created on {today}",
    }

    # save the file as a netcdf
    ds.to_netcdf(f'{SWATH_OUTPATH}/daily_precip_{track}_{name}.nc')

 
# create figure to plot on if we are plotting the precip swaths for verification
if PLOT_SWATHS:
    # set the bbox for the rio grande valley
    lon_min, lat_min, lon_max, lat_max = -121.98, 20.66, -84.64, 40.68
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add map features
    ax.coastlines()
    # Set the extent of the map to the bounding box
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    datasets = []

    # Load all NetCDF files from the directory
    for filename in os.listdir(SWATH_OUTPATH):
        if filename.endswith('.nc'):
            filepath = os.path.join(SWATH_OUTPATH, filename)
            ds = xr.open_dataset(filepath)
            datasets.append(ds)

    # Combine all datasets along the time dimension
    combined_swaths = xr.concat([ds.precipitation for ds in datasets], dim='time')


    # Calculate the sum of precipitation across time dimension
    total_precipitation = combined_swaths.sum(dim='time')
    # total_precipitation = precip_masked.mean(dim='time')

    # Define the levels (start from 1 mm up to 1500 mm, with the last level for > 1500 mm)
    levels = [0] + np.linspace(1, 1500, num=20).tolist() + [5000]  # Explicitly add 0 and 2000

    # Create a colormap where the first color (for 0 mm) is white, and others are from 'Blues'
    cmap = plt.get_cmap('Blues', 20)  # Get a colormap with 20 levels
    colors = [(1, 1, 1)] + [cmap(i) for i in range(cmap.N)]  # Add white as the first color
    new_cmap = mcolors.ListedColormap(colors)  # Create a new colormap with white at the start

    # Create a normalization so that 0 is white and 1+ mm gets colors from the colormap
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=new_cmap.N)


    # Plot the total precipitation swath
    p = ax.contourf(
        total_precipitation.lon, total_precipitation.lat, total_precipitation.T,
        transform=ccrs.PlateCarree(),  # Transform coordinates
        cmap=new_cmap,  # Use the custom colormap
        levels=levels,  # Use the defined levels
        alpha=0.9,  # Set transparency
        norm=norm  # Apply the normalization
    )

    # Add the colorbar and ensure it extends beyond 1500
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Precipitation (mm)')

    # Set the colorbar ticks
    cbar.set_ticks([0] + np.linspace(1, 1500, num=6).tolist())  # Include 0, then regular intervals
    cbar.set_ticklabels(['0'] + [f'{int(tick)}' for tick in np.linspace(1, 1500, num=6)])  # Format tick labels


    resol = '50m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    land = cfeature.NaturalEarthFeature('physical', 'land', \
            scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    lakes = cfeature.NaturalEarthFeature('physical', 'lakes', \
            scale=resol, edgecolor='lightsteelblue', facecolor=cfeature.COLORS['water'])
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
            scale=resol, edgecolor='lightsteelblue', facecolor='none')

    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

    rgb_shp.plot(ax=ax, transform=ccrs.PlateCarree(), color="grey", alpha=0.3, zorder=1000)
    # Set title
    fig.suptitle(f'TC Associated Precipitation from {SOURCE} ({START_YR}-{END_YR})')

    # set tight layout
    fig.tight_layout()
    # Show the plot
    plt.show()