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



# read in datasets
fp_precip = '/nfs/turbo/seas-hydro/laratt/IMERG_3B/combined_precip'
SHP_RGB = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp'
# read in data
ds = xr.open_mfdataset(f'{fp_precip}/precipitation*.nc')
#ds = xr.open_dataset(f'{fp_precip}/precipitation_2010.nc')


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












df = pd.read_csv("/nfs/turbo/seas-hydro/laratt/historic_precip/mexico_basin_totals.csv")
df['time'] = pd.to_datetime(df['time'])
# Extract the year from the 'time' column
df['year'] = df['time'].dt.year
# Group by 'year' and sum the values for each category
annual_precipitation = df.groupby('year')[['Total', 'Non TC', 'TC']].sum()

deliveries['year'] = deliveries['date'].dt.year
# Filter for years between 2000 and 2019
filtered_deliveries = deliveries[(deliveries['date'] >= '2000-06-01') & (deliveries['date'] <= '2019-12-31')]
# Group by 'year' and sum the 'flow' values to get the total flow per year
annual_flow = filtered_deliveries.groupby('year')['flow'].sum().reset_index()

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
# Plot 'non_tc' and 'tc' as stacked bars on top of each other
annual_precipitation[['Non TC', 'TC']].plot(kind='bar', stacked=True, ax=ax, position=0, width=0.3,
                                            color=[IBM_BLUE, IBM_PURPLE],label=['Non TC', 'TC'])
ax.axhline(np.mean(annual_precipitation.Total),ls='--',label='Average Total',color='k')
# Set plot labels and title
ax.set_title('Annual Total Precipitation in Mexican Portion',loc='left')
ax.set_xlabel('Year')
ax.set_ylabel('Precipitation (mm)')
# Rotate x-tick labels for better readability
#ax.set_xticklabels(rotation=45)
# Add legend to the plot
ax.legend(ncols=4,frameon=False,loc='upper left',bbox_to_anchor=(0.63,1.11))
# Create a twin axis for plotting a second bar chart
ax2 = ax.twinx()
# Plot the 'total' as a bar chart on the secondary axis (twin axis)
annual_flow['flow'].plot(kind='bar', ax=ax2, color=IBM_PINK, width=0.3, position=1, label='Annual Deliveries')
# Set labels for the secondary axis
ax2.set_ylabel('Annual Deliveries MCM)', color=IBM_PINK)
ax2.tick_params(axis='y', labelcolor=IBM_PINK)

fig.savefig('precip_test.png')


# Assuming 'filtered_deliveries' and 'df' are already defined
# Calculate cross-correlation
correlation = correlate(filtered_deliveries.flow, df.TC, mode='full')

# Get the full range of lags
lags = np.arange(-len(df.TC) + 1, len(df.TC))  # Full range of lags (-len + 1 to len)

# Only consider non-negative lags (0 to len(df.TC)-1)
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
window_size = 20
window_start = max_lag - window_size // 2
window_end = max_lag + window_size // 2 + 1

# Ensure the window is within the bounds of the available non-negative lags
window_start = max(window_start, 0)  # Ensure no negative lags
window_end = min(window_end, len(min_max_normalized_correlation))  # Ensure it doesn't exceed the correlation array length

# Crop the lags and Min-Max normalized correlation to the window of interest
lags_window = non_negative_lags[(non_negative_lags >= window_start) & (non_negative_lags < window_end)]
correlation_window = min_max_normalized_correlation[(non_negative_lags >= window_start) & (non_negative_lags < window_end)]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 6))

# Plot the Min-Max normalized cross-correlation for the 20-day window
ax.plot(lags_window, correlation_window, label='Min-Max Normalized Cross-Correlation', color='tab:blue')

# Plot the maximum correlation line
ax.axvline(max_lag, color='tab:pink', linestyle='--', label=f'Max Min-Max Normalized Correlation at Lag {max_lag} days')

# Set labels and title
ax.set_title('Cross-Correlation Between Daily Deliveries and Daily Mexican TC Precipitation', fontsize=14)
ax.set_xlabel('Lag (days)', fontsize=12)
ax.set_ylabel('Min-Max Normalized Correlation', fontsize=12)

# Add a legend
ax.legend()

# Save the figure
fig.savefig('corr_20_day_window.png')




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

