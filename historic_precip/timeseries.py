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

def clean_IBWC_csvs(dat,start='1962-01-01',end='2023-12-31'):
    """
    Cleans a csv dataset from the IBWC
    """
    try:
        dat.drop('End of Interval (UTC-06:00)', inplace=True, axis='columns')
        dat["Start of Interval (UTC-06:00)"] = pd.to_datetime(dat["Start of Interval (UTC-06:00)"])
    except:
        dat["Timestamp (UTC-06:00)"] = pd.to_datetime(dat["Timestamp (UTC-06:00)"])
    dat.rename(columns={"Timestamp (UTC-06:00)":"date","Start of Interval (UTC-06:00)":"date", "Value (TCM)":"flow","Total (TCM)":"flow",
                        "Average (TCM)":"flow", "Average (m^3/d)":"flow","Value":"flow","Value (m)":"flow"}, inplace=True)
    dat["date"] = pd.to_datetime(dat["date"])
    dat = dat[(dat['date'] >= start) & (dat['date'] <= end)]
    return dat

def calculate_min_max_normalized_correlation(dataset1, dataset2, window_size=20, ax=None):
    """
    Calculates the Min-Max normalized cross-correlation between two datasets, 
    and plots the result for a specified window around the maximum lag.
    
    Parameters:
        dataset1: First dataset (array-like or xarray.DataArray)
        dataset2: Second dataset (array-like or xarray.DataArray)
        window_size: Size of the window around the maximum lag to display (default is 20)
        plot: Boolean flag to indicate whether to plot the result (default is True)
        
    Returns:
        max_lag: The lag with the maximum correlation
    """
    # Calculate cross-correlation
    correlation = correlate(dataset1, dataset2, mode='full')

    # Get the full range of lags
    lags = np.arange(-len(dataset2) + 1, len(dataset2))  # Full range of lags (-len + 1 to len)

    # Only consider non-negative lags (0 to len(dataset2)-1)
    non_negative_lags = lags[lags >= 0]
    correlation_non_negative = correlation[lags >= 0]

    # Min-Max normalization: scale to [0, 1]
    min_corr = np.min(correlation_non_negative)
    max_corr = np.max(correlation_non_negative)

    # Min-Max normalize the correlation
    min_max_normalized_correlation = (correlation_non_negative - min_corr) / (max_corr - min_corr)

    # Find the lag with maximum Min-Max correlation
    max_lag = non_negative_lags[np.argmax(min_max_normalized_correlation)]
    print(f"Maximum Min-Max normalized correlation occurs at lag: {max_lag} days")

    # Define a window of 20 days around the max lag
    window_start = max_lag - window_size // 2
    window_end = max_lag + window_size // 2 + 1

    # Ensure the window is within the bounds of the available non-negative lags
    window_start = max(window_start, 0)  # Ensure no negative lags
    window_end = min(window_end, len(min_max_normalized_correlation))  # Ensure it doesn't exceed the correlation array length

    # Crop the lags and Min-Max normalized correlation to the window of interest
    lags_window = non_negative_lags[(non_negative_lags >= window_start) & (non_negative_lags < window_end)]
    correlation_window = min_max_normalized_correlation[(non_negative_lags >= window_start) & (non_negative_lags < window_end)]

    # Plot if required
    if ax is not None:

        # Plot the Min-Max normalized cross-correlation for the 20-day window
        ax.plot(lags_window, correlation_window, label='Min-Max Normalized Cross-Correlation', color='tab:blue')

        # Plot the maximum correlation line
        ax.axvline(max_lag, color='tab:pink', linestyle='--', label=f'Max Min-Max Normalized Correlation at Lag {max_lag} days')

    # Return the max lag for further use
    return max_lag

def enso_phase(mei):
    if mei >= 0.5:
        return 'El Niño'
    elif mei <= -0.5:
        return 'La Niña'
    else:
        return 'Neutral'
    
def shade_by_index(ax,df,start_time='2002-10-01', end_time='2007-09-30',):
    """
    Shades the timeseries plot by index (e.g. ENSO/MJO phase)
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
    # Apply the function to the MEI values
    df['Phase'] = df['MEI'].apply(enso_phase)
    print(df)

    for i in range(len(df) - 1):
        x_start = df['date'].iloc[i]
        x_end = df['date'].iloc[i + 1]
        y_min = min(df['MEI'].min(), -0.6)  # Extend a little below La Niña for shading
        y_max = max(df['MEI'].max(), 0.6)  # Extend a little above El Niño for shading
    
        # Shade based on phase
        if df['Phase'].iloc[i] == 'El Niño':
            ax.axvspan(x_start, x_end, color='mistyrose', alpha=0.4, label='El Niño' if i == 0 else "")
        elif df['Phase'].iloc[i] == 'La Niña':
            ax.axvspan(x_start, x_end, color='aliceblue', alpha=0.4, label='La Niña' if i == 0 else "")
        else:
            ax.axvspan(x_start, x_end, color='whitesmoke', alpha=0.4, label='Neutral' if i == 0 else "")

def elevation_vs_precip(elev, precip, start_time='2002-10-01', end_time='2007-09-30', 
                        ax=None, nonTC=False,ylab='Storage (TCM)',target=None,
                        title='Mexican Basin TC Precipitation vs Reservoir Storage',capacities=None):
    """
    Function makes a hyetograph plot of (TC) associated precipitation with reservoir 
    elevation.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    
    ax2 = ax.twinx()

    # Now get the precipitation and average
    precip = precip.sel(time=slice(start_time, end_time))

    if nonTC:
        precip_nonTC = precip.precipitation.sum(dim=('lat', 'lon'))
        ax2.stem(precip_nonTC.time, precip_nonTC.values,markerfmt=" ",basefmt=" ",
                    label='Non-TC Precipitation Falling in Mexico', linefmt=(0.8, 0.8, 0.8, 0.3))
    

    precip = precip['total'] - precip['precipitation']
    precip_TC = precip.sum(dim=('lat', 'lon'))

    # Plot precipitation with negative values to make bars go below the axis
    ax2.stem(precip_TC.time, precip_TC.values, markerfmt=" ",basefmt=" ",
             label='TC Precipitation Falling in Mexico',linefmt=IBM_BLUE)  # Example IBM_BLUE
    #ax2.plot(precip_TC.time, precip_TC.values, label='TC Precipitation Falling in Mexico', color='k', ls = '--', lw=0.5)
    #ax2.scatter(precip_TC.time, precip_TC.values, label='TC Precipitation Falling in Mexico', color='k', lw=0.5)

    # Plot the reservoir elevation
    # plot the target deliveries
    colors = [IBM_PINK, IBM_ORANGE, IBM_PURPLE, IBM_YELLOW]
    if target:
        for (res, data), color in zip(elev.items(),colors):
            data = data[(data['date'] >= start_time) & (data['date'] <= end_time)]
            for i, cycle in enumerate(target):
                datas = data[(data['date'] >= cycle[0]) & (data['date'] <= cycle[1])]
                cumulative = np.cumsum(datas['flow'])
                if i == 0:
                    ax.plot(datas['date'], cumulative, lw=2, label=res,c=color,zorder=1000)  # Example IBM_PINK
                else:
                    ax.plot(datas['date'], cumulative, lw=2,c=color,zorder=1000)  # Example IBM_PINK
    else:
        for (res, data), color in zip(elev.items(),colors):
            data = data[(data['date'] >= start_time) & (data['date'] <= end_time)]
            ax.plot(data['date'], data['flow'], lw=2, label=res,c=color,zorder=1000)  # Example IBM_PINK
            if capacities:
                ax.axhline(capacities[res],ls='dashdot',lw=2,c=color,label=f'{res} Capacity')


    # Invert the y-axis so the precipitation bars are above the x-axis
    ax2.set_ylim(bottom=0)
    ax2.invert_yaxis()

    ax.set_xlim(start_time,end_time)
    ax2.set_xlim(start_time,end_time)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=12))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Yearly ticks for the start of each year

    # DateFormatter for the ticks, you can customize the format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate the x-tick labels for better readability
    #for label in ax.get_xticklabels():
    #    label.set_rotation(45)

    # Labels and title
    ax.set_ylabel(ylab)
    ax2.set_ylabel('Daily Basin Total Precipitation (mm)')
    ax.set_title(title, loc='left')
    ax.set_title(f'Cycle {start_time.strftime("%Y-%m-%d")} - {end_time.strftime("%Y-%m-%d")}', loc='right')
    
    # Add legend
    if ax is None:
        ax.legend(ncols=5, frameon=False, loc='upper right',bbox_to_anchor=(-0.09,1.147))

def plot_targets(targets, start_date='2002-09-01', end_date='2007-09-30', resets=['2008-08-10', '2008-10-28', '2009-02-28'], ax=None, yscale=4.6e6, dday=2):
    """
    Plots the targets and any resets for a given year
    """
    if resets:
        # If there's only one reset, handle it differently
        if len(resets) == 1:
            reset = resets[0]
            dates_1 = pd.date_range(start_date, reset,inclusive='left')
            dates_2 = pd.date_range(reset,end_date,inclusive='left')
            print(dates_2, len(dates_2))
            ax.axvline(pd.to_datetime(reset), lw=1.5, ls='--', c='darkgrey', label='Cycle Reset')
            ax.plot(dates_1, targets.flow[0:len(dates_1)], ls='--', c='k',label='Target Deliveries')
            ax.plot(dates_2, targets.flow[0:len(dates_2)], ls='--', c='k')
            # Add annotation for the reset
            ax.annotate(reset, 
                        xy=(pd.to_datetime(reset), yscale),  # The position of the vertical line
                        xytext=(pd.to_datetime(reset) + pd.Timedelta(days=dday), yscale),  # Text position (to the right of the line)
                        horizontalalignment='left',  # Align text to the left of the given position
                        verticalalignment='center',  # Align text vertically to the center
                        rotation=90,
                        color='darkgrey')
        else:
            # Multiple resets: process each reset one by one
            for i, reset in enumerate(resets):
                if i == 0:
                    dates = pd.date_range(start_date, reset,inclusive='left')
                    ax.axvline(pd.to_datetime(reset), lw=1.5, ls='--', c='darkgrey', label='Cycle Reset')
                    ax.plot(dates, targets.flow[0:len(dates)], ls='--', c='k',label='Target Deliveries')
                if reset == resets[-1]:
                    dates = pd.date_range(reset, end_date,inclusive='left')
                    ax.axvline(pd.to_datetime(reset), lw=1.5, ls='--', c='darkgrey')
                    ax.plot(dates, targets.flow[0:len(dates)], ls='--', c='k')
                else:
                    dates = pd.date_range(reset, resets[i+1],inclusive='left')
                    ax.axvline(pd.to_datetime(reset), lw=1.5, ls='--', c='darkgrey')
                    # Add annotation for the reset
                    ax.plot(dates, targets.flow[0:len(dates)], ls='--', c='k')
                ax.annotate(reset, 
                            xy=(pd.to_datetime(reset), yscale),  # The position of the vertical line
                            xytext=(pd.to_datetime(reset) + pd.Timedelta(days=dday), yscale),  # Text position (to the right of the line)
                            horizontalalignment='left',  # Align text to the left of the given position
                            verticalalignment='center',  # Align text vertically to the center
                            rotation=90,
                            color='darkgrey')
    else:
        # No resets: plot the entire range
        dates = pd.date_range(start_date, end_date,inclusive='left')
        ax.plot(dates, targets.flow[0:len(dates)], ls='--', c='k')
        
def plot_resets(ax,resets=['2008-10-08','2008-10-28','2009-02-28'],yscale=4.6e6,dday=2):
    """
    Plots the resets of the delivery cycle on a figure
    """
    for i, reset in enumerate(resets):
        if i == 0:
            ax.axvline(pd.to_datetime(reset),lw=1.5,ls='--',c='darkgrey',label='Cycle Reset')
        else:
            ax.axvline(pd.to_datetime(reset),lw=1.5,ls='--',c='darkgrey')
        
        # now add annotation
        ax.annotate(reset, 
             xy=(pd.to_datetime(reset), yscale),  # The position of the vertical line
             xytext=(pd.to_datetime(reset) + pd.Timedelta(days=dday),yscale),  # Text position (to the right of the line)
             horizontalalignment='left',  # Align text to the left of the given position
             verticalalignment='center',  # Align text vertically to the center
             rotation=90,
             color='darkgrey')

# falcon 91.7 meters
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


# plot the runoff at each of the mexican tributaries with duration of TC in Mexican half of basin
# plot the water level at Amistad and Falcon against the daily total and TC precipitation for 2010 cycles and 2015-2020 cycles
amistad_elev = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/amistad_storage.csv',skiprows=4,skipfooter=1,engine='python')
falcon_elev = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/falcon_storage.csv',skiprows=4,skipfooter=1,engine='python')
laboq_elev = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/la_boquilla_storage.csv',skiprows=4,skipfooter=1,engine='python')
deliveries = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/deliveries_TCM.csv',skiprows=4,skipfooter=1,engine='python')
targets = pd.read_csv('/nfs/turbo/seas-hydro/laratt/TCdata/conchos_reservoirs/target_deliveries.csv')
MEI = pd.read_csv("/nfs/turbo/seas-hydro/laratt/TCdata/MEI.csv")
MEI["date"] = pd.to_datetime(MEI["date"])

names = ['Amistad','Falcon','La Boquilla']
rivers = [amistad_elev,falcon_elev,laboq_elev]
cleaned_rivers = {name:clean_IBWC_csvs(river) for name, river in zip(names,rivers)}
cleaned_deliveries = {'Deliveries (TCM)':clean_IBWC_csvs(deliveries)}
targets = clean_IBWC_csvs(targets,end='2025-10-24')

# define the capacities (TCM)
cons_storage = {'Amistad':3980096,
                'La Boquilla':2893571,
                'Falcon':3288726 }

dead_storage = {'Amistad':40358,
                'La Boquilla':106087,
                'Falcon':12 }

flood_storage = {'Amistad':6055720,
                'La Boquilla':2893571,
                'Falcon':3923322 }

"""
mexico_basin_precip = {'time':mexico_data.time.values,
                        'total':mexico_data.total.sum(dim=['lat','lon']).values,
                        'non_tc':mexico_data.precipitation.sum(dim=['lat','lon']).values,
                        'tc':mexico_data.total.sum(dim=['lat','lon']).values -mexico_data.precipitation.sum(dim=['lat','lon']).values }


"""



# now make better plot for choice water cycles

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


#### Timeseries Plots ####
figa, axa = plt.subplots(2,1,figsize=(18,8))
# precipitation contribution in mexican portion of basin
mexico_TCs = mexico_data.total - mexico_data.precipitation
# get the means
monthly_mexico_TCs = mexico_TCs.resample(time='1ME').mean().mean(dim=('lat', 'lon'))
monthly_mexico_precip = mexico_data.total.resample(time='1MS').mean().mean(dim=('lat', 'lon'))
axa[0].plot(monthly_mexico_TCs.time,monthly_mexico_TCs.values)
axa[1].plot(monthly_mexico_precip.time,monthly_mexico_precip.values)




#### Example delivery cycles vs reservoir storage #####
fig1, ax1 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_rivers,mexico_data,ax=ax1,start_time='2002-09-01',nonTC=True,capacities=cons_storage)
plot_resets(ax1,['2002-10-01'],yscale=3e6,dday=4)
ax1.legend(ncols=10, frameon=False, loc='upper left',bbox_to_anchor=(0.21,1.045))
fig1.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/reservoirs_2002-10-01_2007-09-30.png")


fig2, (ax2,ax2b) = plt.subplots(2,1,figsize=(18,8),sharex=True) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_rivers,mexico_data,ax=ax2,start_time='2007-10-01',end_time='2010-10-24',nonTC=True,capacities=cons_storage)
plot_resets(ax2)
ax2.legend(ncols=10, frameon=False, loc='upper left',bbox_to_anchor=(0.31,1.095))

elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax2b,ylab='Cumulative Cycle Deliveries (TCM)',
                    start_time='2007-10-01',end_time='2010-10-24',nonTC=True,title='Mexican Basin TC Precipitation vs Cumulative Deliveries',
                    target=[('2007-10-01','2008-10-08'),('2008-10-08','2008-10-28'),('2008-10-28','2009-02-28'),('2009-02-28','2010-10-24')])
# '2008-08-10','2008-10-28','2009-02-28'
plot_targets(targets=targets,start_date='2007-10-01',end_date='2010-10-24',ax=ax2b,resets=['2008-10-08','2008-10-28','2009-02-28'],yscale=2.0e6,dday=4)
ax2b.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.67,1.095))
fig2.tight_layout()

fig2.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/combined_2007-10-01_2010-10-24.png")

fig8, ax8 = plt.subplots(1,1,figsize=(18,8)) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_rivers,mexico_data,ax=ax8,start_time='2010-10-25',end_time='2015-10-25',nonTC=True,capacities=cons_storage)
plot_resets(ax8)
ax8.legend(ncols=10, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
fig8.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/reservoirs_2010-10-25_2015-10-25.png")


#### Cumulative Plots for each cycle with precip ####
fig9, ax9 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax9,ylab='Deliveries (TCM)',title='Mexican Basin TC Precipitation vs Cumulative Deliveries',
                    start_time='2010-10-25',end_time='2015-10-25',nonTC=True,target=[('2010-10-25','2015-10-25')])
plot_targets(targets=targets,start_date='2010-10-25',end_date='2015-10-24',ax=ax9,yscale=2.1e6,dday=4,resets=None)
ax9.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax5,MEI,start_time='2002-09-01')
fig9.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/cumulative_deliveries_2010-10-25_2015-10-25.png")


#### Example delivery cycles vs total deliveries #####
fig3, ax3 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax3,ylab='Deliveries (m^3/day)',start_time='2002-09-01',nonTC=True)
plot_resets(ax3,['2002-10-01'],yscale=3.2e7,dday=4)
ax3.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax3,MEI,start_time='2002-09-01')
fig3.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/deliveries_2002-10-01_2007-09-30.png")

fig4, ax4 = plt.subplots(1,1,figsize=(18,8)) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax4,start_time='2007-10-01',nonTC=True,
                    end_time='2010-10-24',ylab='Deliveries (m^3/day)',title='Mexico Basin TC Precipitation vs Deliveries')
plot_resets(ax4,yscale=0.8e8)
ax4.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax4,MEI,start_time='2007-10-01',end_time='2010-10-24')
fig4.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/deliveries_2007-10-01_2010-10-24.png")

#### Cumulative Plots for each cycle with precip ####
fig5, ax5 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax5,ylab='Deliveries (TCM)',title='Mexican Basin TC Precipitation vs Cumulative Deliveries',
                    start_time='2002-09-01',nonTC=True,target=[('2002-09-01','2002-10-01'),('2002-10-01','2007-09-30')])
plot_targets(targets=targets,start_date='2002-09-01',ax=ax5,resets=['2002-10-01'],yscale=2.1e6,dday=4)
ax5.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax5,MEI,start_time='2002-09-01')
fig5.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/cumulative_deliveries_2002-10-01_2007-09-30.png")


fig6, ax6 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,mexico_data,ax=ax6,ylab='Cumulative Cycle Deliveries (TCM)',
                    start_time='2007-10-01',end_time='2010-10-24',nonTC=True,title='Mexican Basin TC Precipitation vs Cumulative Deliveries',
                    target=[('2007-10-01','2008-08-10'),('2008-08-10','2008-10-28'),('2008-10-28','2009-02-28'),('2009-02-28','2010-10-24')])
# '2008-08-10','2008-10-28','2009-02-28'
plot_targets(targets=targets,start_date='2007-10-01',end_date='2010-10-24',ax=ax6,resets=['2008-08-10','2008-10-28','2009-02-28'],yscale=2.0e6,dday=4)
ax6.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax5,MEI,start_time='2002-09-01')
fig6.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/just_mexico/cumulative_deliveries_2007-10-01_2010-10-24.png")


#### Correlation Plots ####

#client.close()






#### Example delivery cycles vs reservoir storage #####
fig1, ax1 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_rivers,ds,ax=ax1,start_time='2002-09-01',nonTC=True,title='Total Basin TC Precipitation vs Reservoir Levels')
plot_resets(ax1,['2002-10-01'],yscale=3e6,dday=4)
ax1.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
fig1.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/full_basin/reservoirs_2002-10-01_2007-09-30.png")


fig2, ax2 = plt.subplots(1,1,figsize=(18,8)) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_rivers,ds,ax=ax2,start_time='2007-10-01',end_time='2010-10-24',nonTC=True,title='Total Basin TC Precipitation vs Reservoir Levels')
plot_resets(ax2)
ax2.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
fig2.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/full_basin/reservoirs_2007-10-01_2010-10-24.png")


#### Example delivery cycles vs total deliveries #####
fig3, ax3 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,ds,ax=ax3,ylab='Deliveries (m^3/day)',start_time='2002-09-01',nonTC=True,title='Total Basin TC Precipitation vs Deliveries')
plot_resets(ax3,['2002-10-01'],yscale=3.2e7,dday=4)
ax3.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax3,MEI,start_time='2002-09-01')
fig3.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/full_basin/deliveries_2002-10-01_2007-09-30.png")

fig4, ax4 = plt.subplots(1,1,figsize=(18,8)) # 2007-10-01 - 10-08-2008, 07-13-2010 to 10-24-2010
elevation_vs_precip(cleaned_deliveries,ds,ax=ax4,start_time='2007-10-01',nonTC=True,
                    end_time='2010-10-24',ylab='Deliveries (m^3/day)',title='Total Basin TC Precipitation vs Total Deliveries to US')
plot_resets(ax4,yscale=0.8e8)
ax4.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax4,MEI,start_time='2007-10-01',end_time='2010-10-24')
fig4.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/full_basin/deliveries_2007-10-01_2010-10-24.png")

#### Cumulative Plots for each cycle with precip ####
fig5, ax5 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,ds,ax=ax5,ylab='Deliveries (TCM)',title='Total Basin TC Precipitation vs Cumulative Deliveries',
                    start_time='2002-09-01',nonTC=True,target=[('2002-09-01','2002-10-01'),('2002-10-01','2007-09-30')])
plot_targets(targets=targets,start_date='2002-09-01',ax=ax5,resets=['2002-10-01'],yscale=2.1e6,dday=4)
ax5.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax5,MEI,start_time='2002-09-01')
fig5.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/full_basin/cumulative_deliveries_2002-10-01_2007-09-30.png")


fig6, ax6 = plt.subplots(1,1,figsize=(18,8)) # 2002-10-01 to 2007-09-30
elevation_vs_precip(cleaned_deliveries,ds,ax=ax6,ylab='Cumulative Cycle Deliveries (TCM)',
                    start_time='2007-10-01',end_time='2010-10-24',nonTC=True,title='Total Basin TC Precipitation vs Cumulative Deliveries',
                    target=[('2007-10-01','2008-08-10'),('2008-08-10','2008-10-28'),('2008-10-28','2009-02-28'),('2009-02-28','2010-10-24')])
# '2008-08-10','2008-10-28','2009-02-28'
plot_targets(targets=targets,start_date='2007-10-01',end_date='2010-10-24',ax=ax6,resets=['2008-08-10','2008-10-28','2009-02-28'],yscale=2.0e6,dday=4)
ax6.legend(ncols=5, frameon=False, loc='upper left',bbox_to_anchor=(0.63,1.045))
#shade_by_index(ax5,MEI,start_time='2002-09-01')
fig6.savefig("/nfs/turbo/seas-hydro/laratt/historic_precip/figures/water_resources/full_basin/cumulative_deliveries_2007-10-01_2010-10-24.png")

