#usr/bin/env python3

"""
RGB_timeseries.py 

Creates a timeseries of precipitation in the Rio Grande Basin using a 
from a specified product, then removes TC precipitation swaths created in
the TC_swaths.py script from the historical record.

Final output generated:
1. monthly timeseries plots of accumulated TC precipitation in the basin
2. timeseries of all precipitation in the RGB
    1. total average
    2. spatial average
3. timeseries of all non TC precipitation in the RGB
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
import dask.array as da
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import os
import matplotlib.colors as mcolors
from shapely.geometry import Point
import dask

#############
## GLOBALS ##
#############

# filepaths to necessary data sources
PRECIP_PRODUCT = "IMERG_3B"                                                                  # name of precipiation product
SWATHS_PATH = f"/nfs/turbo/seas-hydro/laratt/TCdata/precip_swaths/{PRECIP_PRODUCT}"          # path to the precipitation swaths
PRECIP_PATH = f"/nfs/turbo/seas-hydro/laratt/{PRECIP_PRODUCT}"                               # path to the precipitation product
SHP_RGB = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp'
EBT_AL = '/nfs/turbo/seas-hydro/laratt/TCdata/track_files/EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt'
EBT_EP = '/nfs/turbo/seas-hydro/laratt/TCdata/track_files/EBTRK_EP_final_1949-2021_new_format_02-Sep-2022.txt'

# globals for swath extraction
SOURCE = 'IMERG 3B'        # precipitation product being used
TC_RADIUS = 500            # radius in km about which to extract TC precip
PREC_VAR = 'precipitation' # variable in precipitation data for precip
LAT_VAR = 'lat'            # variable for latitude in precipitation data
LON_VAR = 'lon'            # variable for longitude in precipitation data
START_YR = 2000            # year to start data extraction
END_YR = 2020              # year to end data extraction
STORM_ID = None            # run test for individual TC - use the ATCF ID from the EBTRAK dataset (e.g. 'AL012010' for Alex 2010)
TIMESERIES = "monthly"     # how to do the time series: "annual", "monthly", or "daily"
WRITE_PRECIP = True       # should precip files be written to disk
PLOT_ALL_SWATHS = False     # should a plot of all swaths be made?
EXTRACT_MASK = True

rgb_shp = gpd.read_file(SHP_RGB)
"""if EXTRACT_MASK:
    # first read in the total dataset of precipitation
    precip_ds = xr.open_mfdataset(f"{PRECIP_PATH}/3B-DAY*",engine='netcdf4')  #decode_times=False,decode_cf=False,
    print('precip data read in')
    datasets = []

    # Load all NetCDF files from the directory
    for filename in os.listdir(SWATHS_PATH):
        if filename.endswith('.nc'):
            filepath = os.path.join(SWATHS_PATH, filename)
            ds = xr.open_dataset(filepath)
            datasets.append(ds)
    
    print('loaded precip data')

    # Combine all datasets along the time dimension
    combined_swaths = xr.concat([ds.precipitation for ds in datasets], dim='time')
    print('swaths combined')

    # subtract out precipitation from TCs here
    # Create an empty array to hold the results
    aligned_precip = precip_ds.sel(time=combined_swaths['time'])
    print('datasets aligned')

    # Perform the subtraction
    result_precipitation = aligned_precip - combined_swaths
    print('subtracted precip')

    # Ensure result_precipitation is aligned with the full precipitation dataset
    result_precip_aggregated = result_precipitation.groupby('time').mean(dim='time', skipna=True)
    common_times = np.intersect1d(precip_ds['time'].values, result_precip_aggregated['time'].values)
    print('common times extracted')"""

if EXTRACT_MASK:
    # Read in the total dataset of precipitation with Dask
    precip_ds = xr.open_mfdataset(f"{PRECIP_PATH}/3B-DAY*", engine='netcdf4', chunks={'time': 1})
    print('precip data read in')
    
    # Create a list of delayed tasks for loading each NetCDF file
    delayed_datasets = []
    for filename in os.listdir(SWATHS_PATH):
        if filename.endswith('.nc'):
            filepath = os.path.join(SWATHS_PATH, filename)
            delayed_ds = dask.delayed(xr.open_dataset)(filepath, chunks={'time': 1})
            delayed_datasets.append(delayed_ds)

    # Compute all delayed datasets in parallel
    datasets = dask.compute(*delayed_datasets)
    print('loaded precip data')

    # Combine all datasets along the time dimension
    combined_swaths = xr.concat([ds.precipitation for ds in datasets], dim='time')
    print('swaths combined')

    # Align precipitation dataset with combined swaths
    aligned_precip = precip_ds.sel(time=combined_swaths['time'])
    print('datasets aligned')

    # Perform the subtraction
    result_precipitation = aligned_precip - combined_swaths
    print('subtracted precip')

    # Ensure result_precipitation is aligned with the full precipitation dataset
    result_precip_aggregated = result_precipitation.groupby('time').mean(dim='time', skipna=True)
    
    # Extract common times efficiently
    common_times = np.intersect1d(precip_ds['time'].values, result_precip_aggregated['time'].values)
    print('common times extracted')

    # Loop through the common times and replace the precipitation values
    updated_precip_masked = precip_ds.copy()
    # Convert your DataArrays to Dask arrays if they aren't already
    result_precip_aggregated_dask = result_precip_aggregated.chunk({'time': 1})  # Adjust chunk size as needed
    print('chunked data')

    result_precip_aggregated = result_precip_aggregated.compute()
    
    # Use Dask to handle the replacement in chunks
    for time in common_times:
        time_slice = result_precip_aggregated_dask.precipitation.sel(time=time)
        updated_precip_masked['precipitation'].loc[dict(time=time)] = time_slice
        print(f'slice {time_slice} extracted')

    updated_precip_masked.to_netcdf("/nfs/turbo/seas-hydro/laratt/historic_precip/imerg3b_swaths_removed.nc")

    # then subset to basin
    polygon = rgb_shp['geometry'].values[0]

    # Reproject if necessary (change 'epsg:4326' to your dataset CRS)
    #if rgb_shp.crs != "EPSG:4326":
    #    rgb_shp = rgb_shp.to_crs("EPSG:4326")

    # next mask all of them to only the RGB
    # Extract lat/lon from the dataset - note that swaths necessarily have the same spatial coords as the satellite product
    lon = precip_ds['lon'].values
    lat = precip_ds['lat'].values

    # Create a meshgrid of lat and lon
    lon2d, lat2d = np.meshgrid(lon, lat)
    # Flatten lat and lon arrays
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    # Create a list of shapely Points
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)])

    # Create a mask (True for points inside the polygon)
    mask = points.within(polygon)
    print('masked to basin')

    # Reshape the mask back to the shape of lon2d/lat2d
    mask_2d = mask.values.reshape(lon2d.shape)

    # Create a DataArray mask (time dimension is broadcasted across lon/lat)
    mask_xr = xr.DataArray(mask_2d, dims=("lat", "lon"))

    # Mask the precipitation data
    precip_masked = precip_ds['precipitation'].where(mask_xr)
    # Assign masked precipitation back to the dataset
    precip_ds['precipitation'] = precip_masked

    swaths_masked = combined_swaths.where(mask_xr)

    # next mask the swaths
    #swaths_masked = mask_to_shp(rgb_shp,lon2d,lat2d,combined_swaths)
    swaths_masked = combined_swaths.where(mask_xr)
    print('processing complete')

    """# Create an empty array to hold the results
    aligned_precip = precip_masked.sel(time=swaths_masked['time'])

    # Perform the subtraction
    result_precipitation = aligned_precip - swaths_masked

    # Ensure result_precipitation is aligned with the full precipitation dataset
    result_precip_aggregated = result_precipitation.groupby('time').mean(dim='time', skipna=True)
    common_times = np.intersect1d(precip_masked['time'].values, result_precip_aggregated['time'].values)

    # Loop through the common times and replace the precipitation values
    updated_precip_masked = precip_masked.copy()
    # Convert your DataArrays to Dask arrays if they aren't already
    result_precip_aggregated_dask = result_precip_aggregated.chunk({'time': 1})  # Adjust chunk size as needed

    # Use Dask to handle the replacement in chunks
    for time in common_times:
        time_slice = result_precip_aggregated_dask.precipitation.sel(time=time)
        updated_precip_masked['precipitation'].loc[dict(time=time)] = time_slice

    updated_precip_masked.to_netcdf("/nfs/turbo/seas-hydro/laratt/historic_precip/rgb_imerg3b_swaths_removed.nc")"""

else:
    precip_masked = xr.open_dataset('/nfs/turbo/seas-hydro/laratt/historic_precip/rgb_imerg3b.nc')
    swaths_masked = xr.open_dataset('/nfs/turbo/seas-hydro/laratt/historic_precip/rgb_imerg3b_swaths.nc')
    updated_precip_masked = xr.open_dataset('/nfs/turbo/seas-hydro/laratt/historic_precip/rgb_imerg3b_swaths_removed.nc')

mon_sum_total = precip_masked.resample(time='1M').sum(skipna=False)
mon_sum_no_TC = updated_precip_masked.resample(time='1M').sum(skipna=False)
# take average of days with two tracks (should do this more thoroughly later)
swaths_masked_aligned = swaths_masked.groupby('time').mean(dim='time', skipna=False)
mon_sum_TC_only = swaths_masked_aligned.resample(time='1M').sum(skipna=False)
    
# now take a basin-wide sum and average for each month
basin_total = mon_sum_total.sum(dim=('lat','lon'))
basin_total_no_TC = mon_sum_no_TC.sum(dim=('lat','lon'))
basin_total_TC_only = mon_sum_TC_only.sum(dim=('lat','lon'))

basin_avg = mon_sum_total.mean(dim=('lat','lon'))
basin_avg_no_TC = mon_sum_no_TC.mean(dim=('lat','lon'),skipna=True)
basin_avg_TC_only = mon_sum_TC_only.mean(dim=('lat','lon'))


fig, ax = plt.subplots(2,1,figsize=(12,8))
# first plot the basin monthly totals
ax[0].plot(basin_total.time.values,basin_total.precipitation.values,c='#648FFF',label='Total Accumulated Precip')
ax[0].plot(basin_total_no_TC.time.values,basin_total_no_TC.precipitation.values,c='#DC267F',label='No TCs')
ax[0].set_xlim(basin_total_TC_only.time.min(),basin_total_TC_only.time.max())
ax[0].set_ylabel('Total Precipitation (mm/day)')
ax[0].legend()
ax[0].set_title(f'IMERG 3b Total Monthly Precipitation in Rio Grande Basin (Jun 2000-Dec 2020)',loc='left',fontweight='bold',fontsize=16)
# next plot the basin monthly averages
ax[1].plot(basin_avg.time.values,basin_avg.precipitation.values,c='#648FFF',label='Average Accumulated Precip')
ax[1].plot(basin_avg_no_TC.time.values,basin_avg_no_TC.precipitation.values,c='#DC267F',label='No TCs')
ax[1].set_xlim(basin_avg_TC_only.time.min(),basin_avg_TC_only.time.max())
ax[1].legend()
ax[1].set_ylabel('Average Precipitation (mm/day)')
ax[1].set_title(f'IMERG 3b Average Monthly Precipitation in Rio Grande Basin (Jun 2000-Dec 2020)',loc='left',fontweight='bold',fontsize=16)
fig.savefig('basin_total_precip.png')



if PLOT_ALL_SWATHS:
    # set the bbox for the rio grande valley
    lon_min, lat_min, lon_max, lat_max = -121.98, 20.66, -84.64, 40.68
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add map features
    ax.coastlines()
    # Set the extent of the map to the bounding box
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Calculate the sum of precipitation across time dimension
    total_precipitation = updated_precip_masked.sum(dim='time')
    # total_precipitation = precip_masked.mean(dim='time')

    # Define the levels (start from 1 mm up to 1500 mm, with the last level for > 1500 mm)
    levels = [0] + np.linspace(1, 1000, num=20).tolist() + [1500]  # Explicitly add 0 and 2000

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
    cbar.set_ticks([0] + np.linspace(1, 1000, num=6).tolist())  # Include 0, then regular intervals
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

## construct a timeseries of swaths ##
# take average o
# make overall timeseries

# make TC precip timeseries

# now subtract out all of the TC precip 

# take an average over the length of the desired timeseries

# make overall timeseries of precip minus TCs
