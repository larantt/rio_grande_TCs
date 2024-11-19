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
import matplotlib.colors as mcolors
from shapely.geometry import Point
from matplotlib.ticker import MaxNLocator, FuncFormatter     
from dask.distributed import Client, LocalCluster
import matplotlib.dates as mdates

IBM_BLUE = "#648FFF"
IBM_PURPLE = "#785EF0"
IBM_PINK = "#DC267F"
IBM_ORANGE = "#FE6100"
IBM_YELLOW = "#FFB000"

# Set up a local Dask cluster
cluster = LocalCluster()
client = Client(cluster)
print(client)  # This will output the dashboard link

def gen_mask(yearly_data, shpfile):
    """
    Function creates a mask of the basin for the Rio Grande.

    Inputs
    ------
    yearly_data : xr.DataArray
        Data containing latitudes and longitudes to be masked.
    shpfile : GeoDataFrame
        Shapefile for the basin geometry.

    Returns
    -------
    mask_xr : xr.DataArray 
        Mask in the form of an xarray DataArray, where points inside or on the boundary 
        of the basin polygon are True (1), and points outside the basin are False (0).
    """
    # Ensure the shapefile is in the same CRS as the lat/lon grid (EPSG:4326)
    if shpfile.crs != 'EPSG:4326':
        shpfile = shpfile.to_crs('EPSG:4326')

    # Get the shapefile geometry (assume the basin is represented by a single polygon or multipolygon)
    polygon = shpfile['geometry'].values[0]  # Assuming only one geometry for simplicity
    
    # Get latitudes and longitudes from the xarray DataArray
    lon = yearly_data['lon'].values
    lat = yearly_data['lat'].values

    # Create a 2D meshgrid of longitude and latitude
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Flatten the 2D meshgrid to get 1D arrays of latitudes and longitudes
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    # Create Shapely Point objects for each (lon, lat) pair
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)])

    # Check if each point intersects (inside or on the boundary) the basin polygon
    mask = points.intersects(polygon)

    # Reshape the mask to the original grid shape (lat, lon)
    mask_2d = mask.values.reshape(lon2d.shape)
    
    # Convert the mask to an xarray DataArray for easy integration with your dataset
    mask_xr = xr.DataArray(mask_2d, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})

    return mask_xr

# read in datasets
fp_precip = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/combined_precip'
SHP_RGB = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp'
SHP_MEX = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_mexico-shp/combine_mexico_side_shapefile.shp'
DS_PATH = f'{fp_precip}/precipitation*.nc'
# read in data
ds = xr.open_mfdataset(f'{fp_precip}/precipitation*.nc').sel(time=slice('2000-01-01', '2019-12-31'))
mexico_basin = gpd.read_file(SHP_MEX)
#ds = xr.open_dataset(f'{fp_precip}/precipitation_2010.nc')

# subset to Mexican region of the basin
mask_xr = gen_mask(ds, mexico_basin)
mexico_data = ds.where(mask_xr)

# Group by month across all years and calculate the mean for each month
# Step 1: Resample to get monthly precipitation sums
ds_monthly_sum = ds.resample(time='1ME').sum()
ds_annual_sum = ds.resample(time='1YE').sum()

# Step 2: Calculate the long-term mean of these monthly sums
ds_monthly_mean = ds_monthly_sum.groupby('time.month').mean('time')
ds_monthly_median = ds_monthly_sum.groupby('time.month').median('time')

#ds = ds.chunk({'time': -1})
#ds_monthly_q90 = ds.groupby('time.month').quantile(0.9,dim='time')

# Define a list of month names for labeling
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

def plot_cartopy(ax, data, title, cmap='Blues', vmax=50, vmin=0, levs=11):
    """
    Function to create a plot of precipitation from a plot file.
    """
    ax.coastlines()
    # get spacing
    space = vmax/(levs-1)
    # Define the color levels for the colorbar (increments of 5 from 0 to 50)
    levels = np.arange(vmin, vmax + space, space).tolist()  # Levels from 0 to 50 in increments of 5
    
    # Create the colormap (Blues)
    cmap = plt.get_cmap(cmap, levs + 1)  # Get the 'Blues' colormap with 'levs' number of colors
    colors = [cmap(i) for i in range(cmap.N)]  # Generate the colormap colors
    new_cmap = mcolors.ListedColormap(colors)  # Create a new colormap

    # Plot the total precipitation swath
    p = ax.contourf(
        data.lon, data.lat, data.T,
        transform=ccrs.PlateCarree(),  # Transform coordinates
        cmap=new_cmap,  # Use the custom colormap
        levels=levels,  # Use the defined levels
        alpha=0.9,  # Set transparency
        vmin=vmin,
        vmax=vmax,
        extend='max'
    )

    # Add the colorbar and ensure it extends beyond 50
    resol = '50m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', 
            scale=resol, edgecolor='lightsteelblue', facecolor='none')

    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)
    ax.set_title(title)
    return p

for year in np.arange(2000,2020,1):
    fig,ax = plt.subplots(1,1,figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
    pct_contrib = ((ds_annual_sum['total'] - ds_annual_sum['precipitation'])/ds_annual_sum['total']) * 100
    p = plot_cartopy(ax, pct_contrib.sel(time=f'{year}').isel(time=0), f'Percentage Contribution of TCs to Total Precipitation {year}',cmap='Blues',levs=11)
    cb1 = fig.colorbar(p, ax=ax, extend='max')
    fig.savefig(f'/nfs/turbo/seas-hydro/laratt/historic_precip/figures/total_yrs/annual_{year}.png')

    # Loop through each month to create the 6-panel plot
    fig, ax = plt.subplots(2, 3, figsize=(19, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    # get percentage contribution of TCs to precipitation
    pct_contrib = ((ds_monthly_sum['total'] - ds_monthly_sum['precipitation'])/ds_monthly_sum['total']) * 100

    for month,axs in zip(np.arange(5,12),ax.flatten()):  # Iterate over each month
        data = pct_contrib.sel(time=f'{year}-{month + 1}') # take a monthly mean
        print(f'plotting {months[month]}, year {year}')
        # Create the 3-panel plot
        
        p = plot_cartopy(axs, data.isel(time=0), f'{months[month]}',cmap='Blues',levs=11)

    fig.subplots_adjust(hspace=0.1,wspace=0.1,right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.01, 0.78])
    cb1 = fig.colorbar(p, cax=cbar_ax, extend='max') 
    cb1.set_label('TC Contribution to Total Precipitation (%)')
    fig.suptitle(f'IMERG-3B TC Contribution (%) to Total Precipitation ({year})',fontweight='bold',fontsize=16)
    fig.savefig(f'/nfs/turbo/seas-hydro/laratt/historic_precip/figures/total_yrs/monthly_{year}.png')

# spatially calculate trend in percentage contribution data

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


def clean_IBWC_csvs(dat):
    """
    Cleans a csv dataset from the IBWC
    """
    try:
        dat.drop('End of Interval (UTC-06:00)', inplace=True, axis='columns')
        dat["Start of Interval (UTC-06:00)"] = pd.to_datetime(dat["Start of Interval (UTC-06:00)"])
    except:
        dat["Timestamp (UTC-06:00)"] = pd.to_datetime(dat["Timestamp (UTC-06:00)"])
    dat.rename(columns={"Timestamp (UTC-06:00)":"date","Start of Interval (UTC-06:00)":"date", "Value (TCM)":"flow",
                        "Average (TCM)":"flow", "Average (m^3/d)":"flow","Value":"flow","Value (m)":"flow"}, inplace=True)
    dat["date"] = pd.to_datetime(dat["date"])
    dat = dat[(dat['date'] >= '1962-01-01') & (dat['date'] <= '2023-12-31')]
    return dat

# plot the runoff at each of the mexican tributaries with duration of TC in Mexican half of basin
# plot the water level at Amistad and Falcon against the daily total and TC precipitation for 2010 cycles and 2015-2020 cycles
amistad_elev = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/amistad_storage.csv',skiprows=4,skipfooter=1,engine='python')
falcon_elev = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/falcon_storage.csv',skiprows=4,skipfooter=1,engine='python')
laboq_elev = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/la_boquilla_storage.csv',skiprows=4,skipfooter=1,engine='python')
deliveries = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/deliveries.csv')
targets = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/target_deliveries.csv',skiprows=4,skipfooter=1,engine='python')

names = ['Amistad','Falcon','La Boquilla']
rivers = [amistad_elev,falcon_elev,laboq_elev]
cleaned_rivers = {name:clean_IBWC_csvs(river) for name, river in zip(names,rivers)}
cleaned_deliveries = {'Deliveries (m^3/d)':clean_IBWC_csvs(deliveries)}
targets = clean_IBWC_csvs(targets)

# falcon 91.7 meters
# now make better plot for choice water cycles
def elevation_vs_precip(elev, precip, start_time='2002-10-01', end_time='2007-09-30', 
                        ax=None, nonTC=False,ylab='Storage (TCM)',target=False):
    """
    Function makes a hyetograph plot of (TC) associated precipitation with reservoir 
    elevation.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Plot the reservoir elevation
    colors = [IBM_PINK, IBM_ORANGE, IBM_PURPLE, IBM_YELLOW]
    for (res, data), color in zip(elev.items(),colors):
        data = data[(data['date'] >= start_time) & (data['date'] <= end_time)]
        ax.plot(data['date'], data['flow'], lw=2, label=res,c=color)  # Example IBM_PINK
    
    # plot the target deliveries
    if target:
        ax.plot(target['date'],target['flow'], ls='--', c='k', label=f'Target', lw=2)

    ax2 = ax.twinx()

    # Now get the precipitation and average
    precip = precip.sel(time=slice(start_time, end_time))
    precip_TC = precip.precipitation.sum(dim=('lat', 'lon'))
    
    # Plot precipitation with negative values to make bars go below the axis
    ax2.bar(precip_TC.time, precip_TC.values, 0.25, label='TC Precipitation Falling in Mexico', ec=IBM_BLUE)  # Example IBM_BLUE
    #ax2.plot(precip_TC.time, precip_TC.values, label='TC Precipitation Falling in Mexico', color='k', ls = '--', lw=0.5)
    #ax2.scatter(precip_TC.time, precip_TC.values, label='TC Precipitation Falling in Mexico', color='k', lw=0.5)

    # Invert the y-axis so the precipitation bars are above the x-axis
    ax2.invert_yaxis()

    if nonTC:
        precip_total = precip.total.sum(dim=('lat', 'lon'))
        ax2.bar(precip.time, -precip_total.values, 0.1, alpha=0.5, zorder=1, label='Total Precipitation Falling in Mexico')

    ax.set_xlim(start_time,end_time)
    ax2.set_xlim(start_time,end_time)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=12))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Yearly ticks for the start of each year

    # DateFormatter for the ticks, you can customize the format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate the x-tick labels for better readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    # Labels and title
    ax.set_ylabel(ylab)
    ax2.set_ylabel('Daily Basin Total Precipitation (mm)')
    ax.set_title(f'Mexican TC precipitation vs {res} Elevation', loc='left')
    ax.set_title(f'Cycle {start_time.strftime("%Y-%m-%d")} - {end_time.strftime("%Y-%m-%d")}', loc='right')
    
    # Add legend
    ax.legend()


# make subplot figures of 4 select
    
client.close()



#### FINAL PROJECT FIGURES ####
# 1. TC Track Density & Trend in TC tracks affecting RGB basin (split bar by hurricane basin)
# 2. TC percentage contribution to precipitation in basin (average, median, 95th pctl) & total TC precip
# 3. Example delivery cycles w precipitation timeseries, reservoir storage
# 4. Example delivery cycles w precipitation timeseries, total deliveries to US
# 5. Correlation between total TC precipitation and total deliveries vs non TC precipitation and total deliveries
# 6. Maybe an artifical gauge network &/or plots from Mo's class?
# 7. V's vertical delivery timeseries vs target, but with %age TC contribution overlaid
# 8. correlation between elevation change at each reservoir and TC occurrence


#### Spatial Plots ####


#### Example delivery cycles vs reservoir storage #####
fig1, ax1 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_rivers,mexico_data,ax=ax1)
fig1.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/reservoirs_2002-10-01_2007-09-30.png")

fig2, ax2a,ax2b = plt.subplots(1,2,figsize=(18,8)) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_rivers,mexico_data,ax=ax2a,start_date='2007-10-01',end_date='2008-08-10')
elevation_vs_precip(cleaned_rivers,mexico_data,ax=ax2b,start_date='2010-07-13',end_date='2010-10-24')
fig1.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/reservoirs_2007-10-01_2010-10-24.png")


#### Example delivery cycles vs total deliveries #####
fig3, ax3 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax3,ylab='Deliveries (m^3/day)',target=targets)
fig3.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/deliveries_2002-10-01_2007-09-30.png")

fig4, ax4a,ax4b = plt.subplots(1,2,figsize=(18,8)) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax4a,start_date='2007-10-01',end_date='2008-08-10',ylab='Deliveries (m^3/day)',target=targets)
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax4b,start_date='2010-07-13',end_date='2010-10-24',ylab='Deliveries (m^3/day)',target=targets)
fig4.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/reservoirs_2007-10-01_2010-10-24.png")


#### Correlation Plots ####