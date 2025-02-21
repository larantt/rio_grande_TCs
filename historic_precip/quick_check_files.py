#usr/bin/envs python3
"""
Script contains a plotting function and script to do a quick
check of a netcdf file, creating a map and pcolor plot.
"""
import xarray as xr
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

def quick_plot(ds, var, time, lonname='lon', latnname='lat'):
    """
    Function creates a quick map to visualize netcdf output
    for a netcdf in the terminal
    """
    fig, ax = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': ccrs.PlateCarree()})
    
    # subset precipitation to time if necessary
    if type(time) == str:
        data = ds.sel(time=time)
    elif type(time) == int:
        data = ds.isel(time=time)
    else:
        data = ds
    
    plotted = data[var].plot.pcolormesh(
        ax=ax,
        cmap='cividis_r',
        transform = ccrs.PlateCarree(),
        x=lonname,
        y=latnname,
        add_colorbar=False
    )
    plt.colorbar(plotted,orientation='horizontal',pad=0.05, label=var)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    return fig,ax

def main():
    inpath = input("Enter full path to file as string:")
    plot_time = input("Enter time or time index of plot frame:")
    plot_var = input("Enter variable to plot:")
    lon_name = input("Enter name of longitude variable:")
    lat_name = input("Enter name of latitude variable:")
    save_fig = input("To save figure, enter outpath, otherwise enter False:")

    # open dataset
    df = xr.open_dataset(inpath)
    fig,ax = quick_plot(df,plot_var,plot_time,lon_name,lat_name)

    if save_fig:
        fig.savefig(f'{save_fig}/quick_plot_{plot_var}_{plot_time}.png')

if __name__ == '__main__':
    main()
