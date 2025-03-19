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
import dask
from pyproj import Proj, transform
import netCDF4 as nc  
from dask.distributed import Client, LocalCluster
import matplotlib.dates as mdates
import cmasher as cmr
import pymannkendall as mk
from tqdm import tqdm

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
    print('Extracting shapefile geometry')
    polygon = shpfile['geometry'].values[0]  # Assuming only one geometry for simplicity
    
    # Get latitudes and longitudes from the xarray DataArray
    lon = yearly_data['lon'].values
    lat = yearly_data['lat'].values

    # Create a 2D meshgrid of longitude and latitude
    print('Generating grid')
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Flatten the 2D meshgrid to get 1D arrays of latitudes and longitudes
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    # Create Shapely Point objects for each (lon, lat) pair
    print('Extracting points')
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)])

    # Check if each point intersects (inside or on the boundary) the basin polygon
    print('Creating mask')
    mask = points.intersects(polygon)

    # Reshape the mask to the original grid shape (lat, lon)
    mask_2d = mask.values.reshape(lon2d.shape)
    
    print('Converting mask to netcdf')
    # Convert the mask to an xarray DataArray for easy integration with your dataset
    mask_xr = xr.DataArray(mask_2d, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})

    return mask_xr

def plot_on_map(ax, data, projection, title, cmap='cmr.arctic_r', colorbar_label=None, vmax=None,div=False):
    # Convert x and y coordinates to lat/lon using pyproj transformation
    x_coords = data.coords['lon'].values
    y_coords = data.coords['lat'].values
    lons, lats = np.meshgrid(x_coords, y_coords)
    #lons, lats = transform(projection, ccrs.PlateCarree(), lons, lats)
    
    # Plot the data
    if vmax:
        if div:
            p = ax.contourf(lons, lats, data.values, cmap=cmap, transform=ccrs.PlateCarree(),vmin=-1*vmax,vmax=vmax)
        else:
            p = ax.contourf(lons, lats, data.values, cmap=cmap, transform=ccrs.PlateCarree(),vmin=0,vmax=vmax)
    else:
        p = ax.contourf(lons, lats, data.values, cmap=cmap, transform=ccrs.PlateCarree())
    
    # Add coastlines and borders
    #ax.coastlines(resolution='10m')
    #ax.add_feature(feature.BORDERS, linestyle=':', edgecolor='black', linewidth=2)
    
    # Set the title
    ax.set_title(title,loc='left')
    
    
    return p

def plot_on_map_with_stipple(fig, ax, xy, trend, p_values, title, cmap='coolwarm', colorbar_label=None, vmax=None, div=False, threshold=0.05, cb_orient='vertical'):
    # Convert x and y coordinates to lat/lon using pyproj transformation

    x_coords = xy.coords['lon'].values
    y_coords = xy.coords['lat'].values
    lons, lats = np.meshgrid(x_coords, y_coords)
    #lons, lats = transform(projection, ccrs.PlateCarree(), lons, lats)
    
    # Transform from projection coordinates to lat/lon (using pyproj)
    #transformer = Transformer.from_crs(projection, ccrs.PlateCarree(), always_xy=True)
    #lons, lats = transformer.transform(lons, lats)
    
    # Plot the trend data
    if vmax:
        if div:
            p = ax.pcolormesh(lons, lats, trend, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-1 * vmax, vmax=vmax)
        else:
            p = ax.pcolormesh(lons, lats, trend, cmap=cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)
    else:
        p = ax.pcolormesh(lons, lats, trend, cmap=cmap, transform=ccrs.PlateCarree())
    
    # Stippling where p-value < threshold (e.g., 0.05)
    p_masked = p_values < threshold
    #stipple_lons, stipple_lats = lons[p_masked], lats[p_masked]
    stipple_lons, stipple_lats = lons[p_masked][::35], lats[p_masked][::35]
    
    # Overlay stipples on the plot (small black dots for significant points)
    ax.scatter(stipple_lons, stipple_lats, color='black', s=0.01, transform=ccrs.PlateCarree())
    
    # Add coastlines and borders
    #ax.coastlines(resolution='10m')
    #ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=2)
    
    # Set the title
    ax.set_title(title, loc='left')
    
    # Add colorbar if needed
    if colorbar_label:
        if cb_orient == 'vertical':
            fig.colorbar(p, ax=ax, orientation=cb_orient, label=colorbar_label,pad=0.02,shrink=0.9,extend='both')
        else:
            fig.colorbar(p, ax=ax, orientation=cb_orient, label=colorbar_label,pad=0.002,shrink=0.3,extend='both')
    
    return p

def spatial_mann_kendall(ds,var,lonvar='lon',latvar='lat'):
    """
    Conducts a Mann-Kendall test for non-parametric trend analysis, with 
    significance determined via Theil-Sen slope. Adapted from Ombadi and Risser, 2022

    See https://github.com/mombadi/Ombadi-Risser-2022-_Increasing-Trends-in-Extreme-Volatility-of-Daily-Max-Temperature/blob/main/NOAA20CR.ipynb

    Parameters
    ----------
    ds : xr.DataArray
        NetCDF containing the data to be tested
    var : string
        variable to test mann-kendall trend for
    lonvar : string, default = lon
        variable name for longitude in dataset
    latvar : string, default = lat
        variable for latitude in the dataset

    Returns
    -------
    slope : np.ndarray
        array of size (lat,lon) containing theil-sen's slope at each grid cell
    p_value : np.ndarray
        array of size (lat,lon) containing the p-value for ts slope at each grid cell
    """
    # convert xarray dataset to numpy arrays
    ds_npz = ds[var].to_numpy()
    # extract the shape of the data, exlcuding the time dimension bc this is a temporal trend
    lon_len = len(ds[lonvar])
    lat_len = len(ds[latvar])
    # create empty numpy arrays to store data from mann-kendall test
    slope, p_value = np.empty([lat_len, lon_len]), np.empty([lat_len, lon_len])
    # fill with nans to start
    slope[:] = np.nan
    p_value[:] = np.nan
    # Loop over all cells and calculate Mann-Kendall with progress bar
    for lat in tqdm(range(ds_npz.shape[1]), desc="Latitude Progress"):
        for lon in range(ds_npz.shape[2]):
            if ~np.isnan(ds_npz[0, lat, lon]):
                x = ds_npz[:, lat, lon]
                result = mk.original_test(x)
                slope[lat, lon] = result.slope
                p_value[lat, lon] = result.p

    return slope, p_value

def convert_kgm2_mm(ds,var,cheat=True):
    """
    Function converts units of Kg/m^2 to mm for NLDAS data. A simple
    calculation assumes that the density of water in the soil is 1000 kg/m^2.
    This calculation is the "cheat" calculation. A more thorough calculation uses
    soil temperature instead. 

    Parameters
    ----------
    ds : xr.Dataset
        xarray dataset containing data to convert
    var : string
        the variable to be converted
    cheat : bool
        should we assume the density of water

    Returns
    -------

    """
    # determine density of water
    if cheat:
        rho_w = 1000 # kg/m^3
        ds[var] = ds[var] * 1000/1000
        ds[var]['units'] = 'mm'
    #else:
    #    rho_w = (1000)/(1 + ds[var]['TSOIL']

SHP_RGB = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp'
SHP_MEX = '/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_mexico-shp/combine_mexico_side_shapefile.shp'
PATH_NLDAS = '/nfs/turbo/seas-hydro/laratt/NLDAS_monthly/NOAH'
TEST_PLOT = False
MKVAR = 'EVP'

ds = xr.open_mfdataset(f'{PATH_NLDAS}/*.nc',chunks={'time':1},engine='netcdf4')
rg_basin = gpd.read_file(SHP_RGB)
mask_xr = gen_mask(ds, rg_basin)
rgb_data = ds.where(mask_xr)
rgb_annual_sum = rgb_data.groupby('time.year').sum(dim='time')

# extract a small test dataset
if TEST_PLOT:
    test_data = rgb_data['EVP'].isel(time=0)
    fig, ax = plt.subplots(1, 1, figsize=(15, 8), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-100.0, central_latitude=42.5)})

    projection = Proj(proj='lcc', lat_1=25, lat_2=60, lon_0=-100, lat_0=42.5)
    p = plot_on_map(ax, test_data, 
                    projection=projection, 
                    title='', cmap='cmr.arctic_r', colorbar_label='Total Evaporation')
    
## - conduct mann-kendall significance test - ##
slope,p_value = spatial_mann_kendall(rgb_annual_sum,MKVAR)

# plot the data
fig, ax = plt.subplots(1, 1, figsize=(15, 8), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-100.0, central_latitude=42.5)})
resol = '50m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', 
            scale=resol, edgecolor='lightsteelblue', facecolor='none')

ax.add_feature(rivers, linewidth=0.5)
ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)
ax.set_extent([-115.485103, -96.017329, 23.412489, 38.624836], crs=ccrs.PlateCarree())

plot_on_map_with_stipple(fig, ax, rgb_annual_sum[MKVAR], 
                         slope, p_value,  
                         'NLDAS (1979-2020)', vmax=-10, div=True, 
                         cmap='cmr.viola', colorbar_label='Total Evaporation Trend (mm/yr)')

ax.set_title(f'min: {np.nanmin(slope):.2f}  max: {np.nanmax(slope):.2f}',loc='right')
fig.suptitle('Theil-Sen ET trend & Mann-Kendall Significance < 0.05 (stippling)',fontweight='bold', ha='center',x=.55)
fig.tight_layout()
fig.savefig('NLDAS_ET_trend.png')


## - conduct mann-kendall significance test - ##
slope,p_value = spatial_mann_kendall(rgb_annual_sum,"AVSFT")

# plot the data
fig, ax = plt.subplots(1, 1, figsize=(15, 8), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-100.0, central_latitude=42.5)})
resol = '50m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', 
            scale=resol, edgecolor='lightsteelblue', facecolor='none')

ax.add_feature(rivers, linewidth=0.5)
ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)
ax.set_extent([-115.485103, -96.017329, 23.412489, 38.624836], crs=ccrs.PlateCarree())

plot_on_map_with_stipple(fig, ax, rgb_annual_sum['AVSFT'], 
                         slope, p_value,  
                         'NLDAS (1979-2020)', vmax=-2.5, div=True, 
                         cmap='cmr.viola', colorbar_label='Average Surface Skin Temperature Trend (K)')

ax.set_title(f'min: {np.nanmin(slope):.2f}  max: {np.nanmax(slope):.2f}',loc='right')
fig.suptitle('Theil-Sen Average Skin Temperature trend & Mann-Kendall Significance < 0.05 (stippling)',fontweight='bold', ha='center',x=.55)
fig.tight_layout()
fig.savefig('NLDAS_AVST_trend.png')


data_80_99 = rgb_data.sel(time=slice('1980-01-01','1999-12-31')).mean(dim='time')
data_00_20 = rgb_data.sel(time=slice('2000-01-01','2020-12-31')).mean(dim='time')

EVP_diff = data_00_20.EVP - data_80_99.EVP

fig, ax = plt.subplots(1, 1, figsize=(15, 8), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-100.0, central_latitude=42.5)})

projection = Proj(proj='lcc', lat_1=25, lat_2=60, lon_0=-100, lat_0=42.5)
p = plot_on_map(ax, EVP_diff, 
                projection=projection, 
                title='', cmap='cmr.arctic_r', colorbar_label='Total Evaporation')


test_cell = rgb_annual_sum.sel(lat=26.443216,lon=-99.102621,method='nearest')
fig2,ax2 = plt.subplots(1,1,figsize=(12,8))
ax2.plot(test_cell.year,test_cell.EVP.values,lw=3,c='indianred',label='Annual Total ET at 26.44,-99.10')
ax2.set_title('Annual Total ET at 26.44,-99.10',loc='left',fontweight='bold')
ax2.set_xlabel('year')
ax2.set_ylabel('ET (mm/yr)')
fig2.savefig('ET_La_Minita_timeseries.png')