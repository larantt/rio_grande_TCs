import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import shapely
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import xarray as xr

#---------------------------#
# Data Processing Functions #
#---------------------------#

def normalize_longitude(lon):
    """Normalize longitude to be within the range [-180, 180]."""
    lon = lon % 360  # Wrap to [0, 360)
    if lon > 180:
        lon -= 360
    return lon

def clean_data(data, shapefile, fixed_rad=500, start_yr=1900, end_yr=2021, start_mon=1):
    """ Cleans data to be used in plotting

    Parameters
    ----------
    data : pd.DataFrame
        standardised TC track dataframe created in statUtils.py
    shapefile : gpd.GeoDataFrame
        shapefile to filter data
    fixed_rad : float
        fixed radius for buffer region
    start_yr : int
        year to start data at
    end_yr : int
        year to end data at
    start_mon : int
        month to start data at
    
    Returns
    -------
    data : pd.DataFrame
        cleaned data
    """
    # Convert longitudes to the range [-180, 180]
    data['lon'] = data['lon'].apply(normalize_longitude)

    # Filter data by year and month
    data = data[(data['year'] >= start_yr) & (data['year'] < end_yr) & (data['month'] >= start_mon)]

    # Create Point geometries
    data['geometry'] = [Point(lon, lat) for lon, lat in zip(data['lon'], data['lat'])]
    data_gdf = gpd.GeoDataFrame(data, geometry='geometry')

    # Create buffer geometries (circles) based on rsize and convert to the same CRS as the shapefile
    data_gdf['buffer'] = data_gdf.apply(
        #lambda row: row.geometry.buffer(row.rsize / 111.0), axis=1
        lambda row: row.geometry.buffer(fixed_rad / 111.0), axis=1
    )
    data_gdf = data_gdf.set_geometry('buffer')

    # Check if any buffer intersects with the shapefile
    intersecting_points = data_gdf[data_gdf.intersects(shapefile.unary_union)]
    intersecting_tracks = data_gdf[data_gdf.intersects(shapefile.unary_union)].track_id.unique()
    print(intersecting_tracks)
    #print(intersecting_points)
    # Filter the original data to keep only the intersecting tracks
    data = data[data['track_id'].isin(intersecting_tracks)]

    # Reset the geometry to the original points and drop the buffer column
    #data = data.drop(columns=['geometry', 'buffer'])
    data['geometry'] = [Point(lon, lat) for lon, lat in zip(data['lon'], data['lat'])]
    data_gdf = gpd.GeoDataFrame(data, geometry='geometry')

    # Reset index and downcast numeric columns
    data_gdf.reset_index(drop=True, inplace=True)
    intersecting_points.reset_index(drop=True, inplace=True)
    data_gdf = data_gdf.apply(pd.to_numeric, errors='ignore', downcast='integer')
    #dtime = datetime.strptime(datestring, "%m-%d-%Y %H:%M")

    return data_gdf, intersecting_points

def get_storm_influence(EBT_df,track_id):
    """
    Generates the geometries to get a swath of influence
    for a specific storm. Designed to be called in the precip
    extraction scripts.

    Parameters
    ----------
    EBT_df : pd.DataFrame
        extended best track dataframe
    track_id : string
        unique identifier for a TC

    Returns
    -------
    daily_influence : gpd.gdf
        geodataframe containing the influence swaths
    """
    storm_df = EBT_df[EBT_df.track_id == track_id]

    # convert the date column to datetime
    storm_df['date'] = pd.to_datetime(storm_df['date'])
    # extract only the date part
    storm_df['date_only'] = storm_df['date'].dt.date
    # dissolve geometries by the date_only column
    daily_influence = storm_df.dissolve(by='date_only')

    # reset the index if needed
    daily_influence = daily_influence.reset_index()
    # transform crs of ghcn to match daily_influence
    daily_influence.set_crs(epsg=4326, inplace=True)
    
    return daily_influence

def get_storm_influence_2(EBT_df, track_id):
    """
    Generates the geometries to get a swath of influence
    for a specific storm. Designed to be called in the precip
    extraction scripts.

    Parameters
    ----------
    EBT_df : pd.DataFrame
        extended best track dataframe
    track_id : string
        unique identifier for a TC

    Returns
    -------
    daily_influence : gpd.GeoDataFrame
        GeoDataFrame containing the influence swaths
    """
    # Filter the dataframe for the given track_id
    storm_df = EBT_df[EBT_df.track_id == track_id]

    # Convert the date column to datetime and extract the date part
    storm_df['date'] = pd.to_datetime(storm_df['date'])
    storm_df['date_only'] = storm_df['date'].dt.date

    # Convert DataFrame to GeoDataFrame
    storm_df['geometry'] = storm_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    gdf_storm = gpd.GeoDataFrame(storm_df, geometry='geometry')
    
    # Set CRS to WGS84 (EPSG:4326)
    gdf_storm.set_crs(epsg=4326, inplace=True)

    # Convert to a projected CRS (meters) for buffering
    gdf_storm = gdf_storm.to_crs(epsg=3857)  # Web Mercator for meter units

    # Create 500 km buffer around each point
    gdf_storm['buffer'] = gdf_storm.geometry.buffer(500000)  # 500 km buffer in meters

    # Dissolve geometries by the date_only column
    daily_influence = gdf_storm.dissolve(by='date_only', aggfunc='first')  # Assuming aggregation by first geometry
    
    # Reset index and transform CRS back to WGS84
    daily_influence = daily_influence.reset_index()
    daily_influence = daily_influence.to_crs(epsg=4326)

    return daily_influence


def find_intersecting_stations_ghcn(stations_df,precip_df,EBT_df, track_id,shp_influence,return_swath=False):
    """
    Function identifies stations within a TC's radius of influence for a
    given storm. 

    Parameters
    ----------
    stations_df : pd.DataFrame
        dataframe containing GHCN station coordinates
    precip_df : xr.DataSet
        xarray dataset containing precipitation for a given year
    EBT_df : pd.DataFrame
        dataframe containing TC extended best tracks
    track_id : string
        unqiue track identification string for a given storm
    shp_influence : shapefile
        shapefile defining an irregular area of influence
    return_swath : bool
        whether to return shapefile with influence swath

    Returns
    -------
    individual_days : List(xr.DataSet)
        list of datasets containing the daily precipitation at each station
        in the TC's radius of influence
    total_precip : xr.DataSet
        dataset containing the sum of the precipitation at all stations over the
        time period within the TC's radius of influence
    daily_influence : gpd.gdf
        geopandas dataframe containing radii of influence geometries
    """
    daily_influence = get_storm_influence(EBT_df,track_id)
    stations_gdf = gpd.GeoDataFrame(
        stations_df, 
        geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat)
    )
    # set crs
    stations_gdf.set_crs(epsg=4326, inplace=True)  # assuming EPSG:4326 (WGS 84)

    # get all stations for the basin in one dataset
    all_stations_basin = gpd.sjoin(stations_gdf, daily_influence, how='inner', predicate='within')
    all_stations_basin = all_stations_basin.drop(['index_right','lat_left','lat_right','lon_left','lon_right'], axis=1)
    all_stations_basin = gpd.sjoin(all_stations_basin, shp_influence, how='inner', predicate='within')
    # add coordinates back into dataframe
    all_stations_basin['lon'] = all_stations_basin['geometry'].x
    all_stations_basin['lat'] = all_stations_basin['geometry'].y
    
    stations_days = []
    # loop through days in the dataframe to get influence on each day
    for day in daily_influence.date_only:
        # perform a spatial join between stations and daily_influence
        stations_within_influence = gpd.sjoin(stations_gdf, daily_influence[daily_influence.date_only == day], how='inner', predicate='within')
        stations_within_influence = stations_within_influence.drop(['index_right','lat_left','lat_right','lon_left','lon_right'], axis=1)
        stations_within_influence = gpd.sjoin(stations_within_influence, shp_influence, how='inner', predicate='within')
        if len(stations_within_influence) != 0:
            # don't append an empty list if no stations intersect
            stations_days.append(stations_within_influence)

    # create a list of precipitation on individual days
    individual_days = [precip_df.reindex(station_id=stations_days[ix].station_id.unique()).sel(date=stations_days[ix].date_only.unique(),method='nearest') 
                       for ix in range(len(stations_days))]
    
    # sum precip over the full period
    combined_ds = xr.concat(individual_days, dim='date').fillna(0)
    # sum precipitation for each station_id across dates
    summed_ds = combined_ds.groupby('station_id').sum('date')
    # drop duplicate stations
    all_stations_basin = all_stations_basin.drop_duplicates(subset='station_id')

    coords_ds = xr.Dataset(
    {
        'lat': ('station_id', all_stations_basin['lat'].values),
        'lon': ('station_id', all_stations_basin['lon'].values)
    },
    coords={'station_id': all_stations_basin['station_id'].values}
    )
    # reassign coordinates
    summed_ds = summed_ds.assign_coords(station_id=summed_ds.station_id.values)

    # merge datasets
    total_precip = xr.merge([summed_ds, coords_ds])

    if return_swath:
        return individual_days, total_precip, daily_influence
    else:
        return individual_days, total_precip
    

def mask_TC_precip(precip_df,EBT_df,track_id,shp_influence,return_swath=False):
    """
    Function identifies stations within a TC's radius of influence for a
    given storm. 

    Parameters
    ----------
    precip_df : xr.DataSet
        xarray dataset containing precipitation for a given year
    EBT_df : pd.DataFrame
        dataframe containing TC extended best tracks
    track_id : string
        unqiue track identification string for a given storm
    shp_influence : shapefile
        shapefile defining an irregular area of influence
    return_swath : bool
        whether to return shapefile with influence swath

    Returns
    -------
    total_precip : xr.DataSet
        dataset containing the sum of the precipitation at all stations over the
        time period within the TC's radius of influence
    """
    daily_influence = get_storm_influence(EBT_df,track_id)
    
    # get all stations for the basin in one dataset
    all_precip = gpd.sjoin(precip_df, daily_influence, how='inner', predicate='within')
    return all_precip
    

##############################################################################################

#------------------------------#
# Data Visualisation Functions #
#------------------------------#

def plot_tracks(data,ax, lower_norm=10, upper_norm=50, map='ws',col='lightsteelblue'):
    """
    Makes a plot of any given TC tracks in a track dataframe
    """

    ax.coastlines()
    ax.set_extent([-120, -30, 7, 55])
    
    # Iterate through the unique track_ids and plot each trac
    scalarmappable = None
    unique_track_ids = data['track_id'].unique()

    for track_id in unique_track_ids:
        # Filter the DataFrame for the current track_id
        track_df = data[data['track_id'] == track_id]
        
        # Get the latitude and longitude values for the track
        latitudes = track_df['lat'].values
        longitudes = track_df['lon'].values
        if map:
            wind_speeds = track_df[map].astype(float).values

            if map == 'mslp':
                max_wind_speed_index = wind_speeds.argmin()
                max_wind_speed = wind_speeds.min()
                max_lon, max_lat = longitudes[max_wind_speed_index], latitudes[max_wind_speed_index]

            else:
                max_wind_speed_index = wind_speeds.argmax()
                max_wind_speed = wind_speeds.max()
                max_lon, max_lat = longitudes[max_wind_speed_index], latitudes[max_wind_speed_index]

            norm = Normalize(vmin=lower_norm, vmax=upper_norm)

            if track_df.name.unique() == 'ALEX_2010':
                for i in range(len(longitudes) - 1):
                # Choose color based on wind speed
                    color = 'indianred'  # Choose colormap
                
                    # Plot line segment
                    ax.plot([longitudes[i], longitudes[i + 1]], [latitudes[i], latitudes[i + 1]], color=color, linewidth=2, transform=ccrs.PlateCarree())
                    # plot circle at around lat lon centre based on rsize
                    if track_df.rsize.values[i] > 0:
                        # Calculate the radius in degrees
                        radius_lat_deg = track_df.rsize.values[i] / 111.0  # Conversion from km to degrees latitude
                        radius_lon_deg = track_df.rsize.values[i] / (111.0 * np.cos(np.deg2rad(latitudes[i])))  # Conversion from km to degrees longitude based on latitude

                        # Plot circle at the center
                        circle = plt.Circle((longitudes[i], latitudes[i]), radius_lon_deg, edgecolor='blue', facecolor='none', transform=ccrs.PlateCarree())
                        ax.add_patch(circle)

            else:
                for i in range(len(longitudes) - 1):
                    # Choose color based on wind speed
                    color = plt.cm.viridis(norm(wind_speeds[i]))  # Choose colormap
                    
                    # Plot line segment
                    ax.plot([longitudes[i], longitudes[i + 1]], [latitudes[i], latitudes[i + 1]], color=color, linewidth=2, transform=ccrs.PlateCarree(),alpha=0.75)

            #ax.scatter(max_lon, max_lat, edgecolors='palevioletred', s=15, facecolors='none', transform=ccrs.PlateCarree(),zorder=100)

                
        if map == None:
        # Plot the track on the single map
            ax.plot(longitudes, latitudes,alpha=0.5, c=col,lw='2')

    if scalarmappable is None:
        scalarmappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis,norm=norm)
        scalarmappable.set_array(wind_speeds)

    return scalarmappable


###############################################################################

#---------------------------#
# Data Extraction Functions #
#---------------------------#

def ReadEBTFile(fname):
    """Read a specific carthe file
       returns a pandas DataFrame containing the days, times, latd, and lond of drifter observations.
       usage is ddata = ReadEBTFile(fname) where fname is a string containing the path to the file
    """
    # Read the file
    with open(fname, 'r') as fileObj:
        lines = fileObj.readlines()

    # Number of header lines (set this to the correct number if there are header lines)
    nbofHeaderLines = 0

    # Number of lines in the file
    nbofLines = len(lines)
    ndata = nbofLines - nbofHeaderLines  # Number of data lines in the file
    print("file has", nbofLines, "lines and", ndata, "data lines")

    # Initialize lists for storing data
    data = {
        'track_id': [],
        'name': [],
        'year': [],
        'month': [],
        'date': [],
        'lat': [],
        'lon': [],
        'ws': [],
        'mslp': []
    }

    # Define column indices based on the expected format of the file
    icode = 0
    iname = 1
    iyear = 3
    idate = 2
    ilat = 4
    ilon = 5
    iwmx = 6
    islp = 7

    for j in range(nbofHeaderLines, nbofLines):
        linedata = lines[j].split()  # Split line into a word list
        hcode = linedata[icode]
        hname = linedata[iname]
        hyear = linedata[iyear]
        hmdt = linedata[idate]
        hmm = hmdt[0:2]
        hdd = hmdt[2:4]
        hhh = str(hmdt[4:6])
        datestring = f"{hmm}-{hdd}-{hyear} {hhh}:00"  # Changed this to include hours 3-10
        hlat = float(linedata[ilat])
        hlon = -float(linedata[ilon])
        if hlon > 180.0:
            hlon -= 360.0
        if hlon < -180.0:
            hlon += 360.0
        hwmx = float(linedata[iwmx])
        hslp = float(linedata[islp])

        # Append data to lists
        data['track_id'].append(hcode)
        data['name'].append(hname)
        data['year'].append(int(hyear))
        data['month'].append(int(hmm))
        data['date'].append(datestring)
        data['lat'].append(hlat)
        data['lon'].append(hlon)
        data['ws'].append(hwmx  * 0.514444) # convert from kts to m/s
        data['mslp'].append(hslp)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    return df

def ghcn_to_nc(df_stations,df_coords,outpath='/Users/laratobias-tarsh/Documents/tcRGB/data/gchn/ncs',year=None):
    """
    Function to create a netcdf for a GHCN daily file in csv format

    Parameters
    ----------ß
    df_stations : pd.DataFrame
        dataframe containing ghcn data for each station
    df_coords : pd.DataFrame
        dataframe containing coordinates for each station
    outpath : string
        path to write netcdf file to
    year : int
        year to generate NetCDF file for
    
    """
    df_stations_long = df_stations.drop('idx',axis=1)
    df_stations_long['date'] = pd.to_datetime(df_stations['date'])
    if year:
        print(f'working on year {year}')
        df_stations_long = df_stations_long[df_stations_long.date.dt.year == year]
    else:
        year = 'full_record'

    df_stations_long = df_stations_long.melt(id_vars=['date'], var_name='station_id', value_name='precipitation')
    df_stations_long = df_stations_long.dropna(subset=['precipitation'])
    df_stations_long['station_id'] = df_stations_long['station_id'].astype(str)
    df_stations_long['precipitation'] = df_stations_long['precipitation'].astype(float)

    # Merge the station data with the long-format precipitation data
    merged_df = pd.merge(df_stations_long, df_coords, on='station_id')
    print(f'data for {year} merged')

    # Create an xarray Dataset
    ds = xr.Dataset.from_dataframe(merged_df.set_index(['station_id', 'date']))

    # Move the lon and lat from data variables to coordinates
    ds = ds.set_coords(['lon', 'lat'])

    # Reorganize to ensure the dataset is well-structured
    ds = ds.transpose('station_id', 'date')

    print(f'Writing netcdf for {year}')
    ds.to_netcdf(f'{outpath}/ghcn_precip_{year}.nc')



def plot_TCP(central_lon,central_lat,masked_precip,lon_array,lat_array,map_extent=[-170, -50, 5, 85]):
    # Plotting with Cartopy
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines and other features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    # Plot the tp data
    # Ensure lon/lat are 2D arrays to match the shape of tp_masked if needed
    lon_2d, lat_2d = np.meshgrid(lon_array, lat_array)
    plot = ax.pcolormesh(lon_2d, lat_2d, masked_precip, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(plot, orientation='vertical', label='Total Precipitation (m)')

    # Mark the target point
    ax.plot(central_lon,central_lat, 'ro', markersize=3, transform=ccrs.PlateCarree(), label='Target Point (-91, 20)')

    # Add gridlines
    ax.gridlines(draw_labels=True)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    # Set title
    plt.title('Total Precipitation within 500km of TC Center')
    return fig, ax


def plot_swath(lon_array, lat_array, masked_precip, central_track,plot_track=True,map_extent=[-170, -50, 5, 85]):
    """
    Plot the total precipitation swath on a 2D map.

    Parameters
    ----------
    lon : arrayLike
        Array of longitudes
    lat : arrayLike
        Array of latitudes
    total_precip : arrayLike
        Total precipitation swath
    central_track : list(tuple)
        List of (lon, lat) tuples representing the track
    """

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines and other features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    # Plot the tp data
    # Ensure lon/lat are 2D arrays to match the shape of tp_masked if needed
    lon_2d, lat_2d = np.meshgrid(lon_array, lat_array)
    plot = ax.pcolormesh(lon_2d, lat_2d, masked_precip, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(plot, orientation='vertical', label='Total Precipitation (m)')

    # Mark the target point
    if plot_track:
        track_lons, track_lats = zip(*central_track)
        ax.plot(track_lons, track_lats, 'r-', marker='o', label='Storm Track', markersize=3, transform=ccrs.PlateCarree())
        #ax.plot(central_lon,central_lat, 'ro', markersize=3, transform=ccrs.PlateCarree(), label='Target Point (-91, 20)')

    # Add gridlines
    ax.gridlines(draw_labels=True)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    # Set title
    plt.title('Total Precipitation within 500km of TC Center')


def daily_radial_precip(centers, times, ds, precip_var='precipitation', lat_var='lat', lon_var='lon', radius=500):
    """
    Function calculates daily accumulated precipitation within the area covered
    by all storm centers (6-hourly) for each day, without double-counting overlapping areas.
    
    Parameters
    ----------
    centers : list of tuples
        List of (lon, lat) tuples for each storm center.
    times : pandas.Series
        Series of datetime objects corresponding to each storm center.
    precip_data : xarray.DataArray
        Daily precipitation data array from netcdf.
    radius : float
        Radius to extract precipitation around each center (in km).
    
    Returns
    -------
    total_precip : xarray.DataArray
        Total accumulated precipitation masked by the full area covered by the storm swaths.
    daily_precip_dataset : xarray.Dataset
        Dataset with time, lat, lon dimensions for daily precipitation.
    """
    # Group storm centers by day
    unique_days = times.dt.floor('D').unique()

    # Extract lons and lats from dataset
    lon_array = ds[lon_var].values
    lat_array = ds[lat_var].values

    # Initialize an empty list to store daily masked precipitation data
    daily_precip_list = []

    # Iterate over each day and apply radial masking
    for i, day in enumerate(unique_days):
        # Get the centers for the current day
        day_centers = [(lon, lat) for (lon, lat), time in zip(centers, times) if time.floor('D') == day]

        # Extract total precipitation for the current day
        tp = ds[precip_var].sel(time=day, method='nearest').values
        
        # Initialize a mask for the current day (False means no precipitation to start)
        daily_mask = np.zeros_like(tp, dtype=bool)
        
        # Apply radial masking for each center within the same day
        for lon, lat in day_centers:
            mask = radial_mask(lon, lat, lon_array, lat_array, radius).T
            daily_mask = np.logical_or(daily_mask, mask)  # Combine the masks without overlap

        # Mask the daily precip
        precip_masked = np.where(daily_mask, tp, np.nan)
        
        # Append the masked precipitation for the current day to the list
        daily_precip_list.append(precip_masked)

    # Convert the list of daily precipitation into an xarray DataArray
    daily_precip_da = xr.DataArray(
        np.stack(daily_precip_list, axis=0),
        coords={"time": unique_days, lon_var: ds[lon_var], lat_var: ds[lat_var]},
        dims=["time", lon_var, lat_var]
    )

    # Create a dataset with daily precipitation
    daily_precip_dataset = xr.Dataset({precip_var: daily_precip_da})

    return daily_precip_dataset



# Create a radial mask for each storm center
def radial_mask(central_lon, central_lat, lon_array, lat_array, radius=500):
    """
    Creates a radial mask for a given storm center.
    
    Parameters
    ----------
    central_lon : float
        Longitude of the storm center.
    central_lat : float
        Latitude of the storm center.
    lon_array : xarray.DataArray
        Array of longitude values from netcdf.
    lat_array : xarray.DataArray
        Array of latitude values from netcdf.
    radius : float
        Radius in kilometers around the storm center.
    
    Returns
    -------
    mask : np.array
        Boolean array where True represents points within the radius.
    """
    # Convert lat/lon to radians
    lon_rad = np.radians(lon_array)
    lat_rad = np.radians(lat_array)

    # Convert target point to radians
    target_lon, target_lat = np.radians([central_lon, central_lat])

    # Earth's radius in km
    earth_radius = 6371.0

    # Calculate Cartesian coordinates for grid points
    x = earth_radius * np.cos(lat_rad[:, None]) * np.cos(lon_rad)
    y = earth_radius * np.cos(lat_rad[:, None]) * np.sin(lon_rad)
    z = earth_radius * np.sin(lat_rad[:, None])

    # Cartesian coordinates for target point
    x0 = earth_radius * np.cos(target_lat) * np.cos(target_lon)
    y0 = earth_radius * np.cos(target_lat) * np.sin(target_lon)
    z0 = earth_radius * np.sin(target_lat)

    # Calculate distances to the target point
    distances = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    # Mask points greater than radius
    mask = distances <= radius

    return mask



def mask_to_shp(shapefile,lon2d,lat2d,data_in,var='precipitation'):
    """
    Function masks a meshgrid of data to a shapefile

    Parameters
    ----------
    shapefile : geopandas.shp
        shapefile to mask data to
    lon2d : np.ndarray
        2d longitude array from meshgrid
    lat2d : np.ndarray
        2d latitude array from meshgrid
    data_in : np.nadarray
        2d array of data to mask

    Returns
    -------
    masked_data : np.ndarray
        2d array of data masked to the shapefile
    """

    polygon = shapefile['geometry'].values[0]

    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    # Create a list of shapely Points
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)])

    # Create a mask (True for points inside the polygon)
    mask = points.within(polygon)

    # Reshape the mask back to the shape of lon2d/lat2d
    mask_2d = mask.values.reshape(lon2d.shape)
    
    if isinstance(data_in, xr.DataArray):
        # If data_in is an xarray DataArray
        # Create a DataArray mask (time dimension is broadcasted across lon/lat)
        mask_xr = xr.DataArray(mask_2d, dims=("lat", "lon"))

        # Mask the data
        masked_data = data_in.where(mask_xr)
    elif isinstance(data_in, np.ndarray):
        # If data_in is a numpy array
        masked_data = np.where(mask_2d, data_in, np.nan)  # Replace masked values with NaN
    else:
        raise TypeError("data_in must be either a numpy array or an xarray DataArray.")

    return masked_data


def memory_check():
    import psutil
    available_memory = psutil.virtual_memory().available
    available_memory_gb = available_memory / (1024 ** 3)  # Convert bytes to GB
    print(available_memory_gb)