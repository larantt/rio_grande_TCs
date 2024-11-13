import time
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
from dask.distributed import Client, LocalCluster

# Set up a local Dask cluster
cluster = LocalCluster()
client = Client(cluster)
print(client)  # This will output the dashboard link


# read in datasets
fp_precip = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/combined_precip'
SHP_RGB = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp'
# read in data
ds = xr.open_mfdataset(f'{fp_precip}/precipitation*.nc')
#ds = xr.open_dataset(f'{fp_precip}/precipitation_2010.nc')

# Group by month across all years and calculate the mean for each month
# Step 1: Resample to get monthly precipitation sums
ds_monthly_sum = ds.resample(time='1M').sum()

# Step 2: Calculate the long-term mean of these monthly sums
ds_monthly_mean = ds_monthly_sum.groupby('time.month').mean('time')

# Define a list of month names for labeling
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']


# Function to create a plot using Cartopy
def plot_cartopy(ax, data, title,cmap='Blues',vmax=10,vmin=0.01):
    ax.coastlines()
    # Define the levels (start from 1 mm up to 1500 mm, with the last level for > 1500 mm)
    levels = np.linspace(vmin, vmax, num=20).tolist()

    # Create a colormap where the first color (for 0 mm) is white, and others are from 'Blues'
    if cmap == 'Blues':
        cmap = plt.get_cmap(cmap, 20)  # Get a colormap with 20 levels
        colors = [(1, 1, 1)] + [cmap(i) for i in range(cmap.N)]  # Add white as the first color
        new_cmap = mcolors.ListedColormap(colors)  # Create a new colormap with white at the start

        # Create a normalization so that 0 is white and 1+ mm gets colors from the colormap
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=new_cmap.N)
    else:
        cmap = plt.get_cmap(cmap, 20)  # Get a colormap with 20 levels
        colors = [cmap(i) for i in range(cmap.N)] + [(1, 1, 1)]  # Add white as the first color
        new_cmap = mcolors.ListedColormap(colors)  # Create a new colormap with white at the start

        # Create a normalization so that 0 is white and 1+ mm gets colors from the colormap
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=new_cmap.N)


    # Plot the total precipitation swath
    p = ax.contourf(
        data.lon, data.lat, data.T,
        transform=ccrs.PlateCarree(),  # Transform coordinates
        cmap=new_cmap,  # Use the custom colormap
        levels=levels,  # Use the defined levels
        alpha=0.9,  # Set transparency
        norm=norm  # Apply the normalization
    )

    # Add the colorbar and ensure it extends beyond 1500
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal', pad=0.05, shrink=0.9)
    cbar.set_label('Precipitation (mm)')

    resol = '50m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
            scale=resol, edgecolor='lightsteelblue', facecolor='none')

    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)
    ax.set_title(title)

# Loop through each month to create the 3-panel plot
for month in np.arange(5,12):  # Iterate over each month
    data = ds_monthly_mean.isel(month=month)
    
    total = data['total']
    precipitation = data['precipitation']
    difference = precipitation - total
    diff_max = np.ceil(np.nanmax(total.values))
    diff_min = np.floor(np.nanmin(difference.values)) if np.floor(np.nanmin(difference.values)) < 0 else -0.1
    print(f'plotting {month}')
    # Create the 3-panel plot
    fig, axs = plt.subplots(1, 3, figsize=(19, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    plot_cartopy(axs[0], total, f'Average Total Precip - {months[month]}',vmax=diff_max)
    plot_cartopy(axs[1], precipitation, f'Average Non TC Precip  - {months[month]}',vmax=diff_max)
    plot_cartopy(axs[2], difference, f'Average TC Precip Loss - {months[month]}', cmap='gist_heat',vmax=0.0,vmin=diff_min)
    
    fig.tight_layout()
    fig.savefig(f'/nfs/turbo/seas-hydro/laratt/historic_precip/figures/precip_{months[month]}_spatial.png')

client.close()




# now sum and average the data
mon_sum = ds.resample(time='1M').sum(skipna=False)

# figures to make - monthly timeseries in each sub basin, esp conchos
# spatial plot for monthly mean precip (precip without - precip with)
# spatial plot for 95th percentile precip (precip without - precip with)

basin_total = mon_sum.sum(dim=('lat','lon'))
basin_avg = mon_sum.mean(dim=('lat','lon'))
# take average of days with two tracks (should do this more thoroughly later)
#swaths_masked_aligned = swaths_masked.groupby('time').mean(dim='time', skipna=False)
#mon_sum_TC_only = swaths_masked_aligned.resample(time='1M').sum(skipna=False)

# now make plots
fig, ax = plt.subplots(2,1,figsize=(12,8))
# first plot the basin monthly totals
ax[0].plot(basin_total.time.values,basin_total.precipitation.values,c='#648FFF',label='Accumulated Precip, TCs Removed',lw=1.5)
ax[0].plot(basin_total.time.values,basin_total.total.values,c='#DC267F',label='Accumulated Precip', lw=1.5)
ax[0].set_xlim(basin_total.time.min(),basin_total.time.max())
ax[0].set_ylabel('Total Precipitation (mm/day)')
ax[0].legend()
ax[0].set_title(f'IMERG 3b Total Monthly Precipitation in Rio Grande Basin (Jun 2000-Dec 2020)',loc='left',fontweight='bold',fontsize=16)
# next plot the basin monthly averages
ax[1].plot(basin_avg.time.values,basin_avg.precipitation.values,c='#648FFF',label='Average Accumulated Precip, TCs Removed',lw=1.5)
ax[1].plot(basin_avg.time.values,basin_avg.total.values,c='#DC267F',label='Average Accumulated Precip',lw=1.5)
ax[1].set_xlim(basin_avg.time.min(),basin_avg.time.max())
ax[1].legend()
ax[1].set_ylabel('Average Precipitation (mm/day)')
ax[1].set_title(f'IMERG 3b Average Monthly Precipitation in Rio Grande Basin (Jun 2000-Dec 2020)',loc='left',fontweight='bold',fontsize=16)
fig.savefig('basin_total_precip.png')

# not averaged
# now make plots
fig, ax = plt.subplots(2,1,figsize=(30,5))
basin_total_day = ds.sum(dim=('lat','lon'))
basin_avg_day = ds.mean(dim=('lat','lon'))
# first plot the basin monthly totals
ax[0].plot(basin_total_day.time.values,basin_total_day.precipitation.values,c='#648FFF',label='Accumulated Precip, TCs Removed',lw=1.5)
ax[0].plot(basin_total_day.time.values,basin_total_day.total.values,c='#DC267F',label='Accumulated Precip', lw=1.5)
ax[0].set_xlim(basin_total_day.time.min(),basin_total_day.time.max())
ax[0].set_ylabel('Total Precipitation (mm/day)')
ax[0].legend()
ax[0].set_title(f'IMERG 3b Total Monthly Precipitation in Rio Grande Basin (Jun 2000-Dec 2020)',loc='left',fontweight='bold',fontsize=16)
# next plot the basin monthly averages
ax[1].plot(basin_avg_day.time.values,basin_avg_day.precipitation.values,c='#648FFF',label='Average Accumulated Precip, TCs Removed',lw=1.5)
ax[1].plot(basin_avg_day.time.values,basin_avg_day.total.values,c='#DC267F',label='Average Accumulated Precip',lw=1.5)
ax[1].set_xlim(basin_avg_day.time.min(),basin_avg_day.time.max())
ax[1].legend()
ax[1].set_ylabel('Average Precipitation (mm/day)')
ax[1].set_title(f'IMERG 3b Average Monthly Precipitation in Rio Grande Basin (Jun 2000-Dec 2020)',loc='left',fontweight='bold',fontsize=16)
fig.savefig('basin_total_precip_daily.png')

# now make spatial plots for each year
fig3, ax3 = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': ccrs.PlateCarree()})