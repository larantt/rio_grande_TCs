"""
process_datasets.py

Contains tools to process datasets used in making MGU 2025 poster. Mostly just functions that are
convenient and can be called in other data cleaning scripts with tools like CDO etc. for optimization.

NOTE: this probably doesn't work with Daymet data. There should be an R script to deal with that.
"""

###########
# IMPORTS #
###########
import os
import geopandas as gpd

##################
# USEFUL GLOBALS #
##################
RGB_SHP_PATH = "/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp" # Full RGB
MX_SHP_PATH = "/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_mexico-shp/combine_mexico_side_shapefile.shp"  # RGB (just Mexico)

def csv_from_shp(shp_file):
    """
    Function creates a csv of points in a shapefile polygon. Created to use for masking netcdf data with CDO.
    CREDIT: Karin Meier-Flescher (https://code.mpimet.mpg.de/boards/2/topics/11284)

    when file has been generated call: cdo -maskregion,csvfile_0.csv infile outfile

    Parameters
    ----------
    shp_file : str
        path to shapefile you want to convert
    """
    reader = gpd.read_file(shp_file, index_col=0)

    geoms = [i for i in reader.geometry.explode(index_parts=True)]
    len_geoms = len(geoms)

    x, y = geoms[0].exterior.coords.xy

    csv_file = ''
    for np in (range(len_geoms)):
        fname = f'csvfile_{np:00d}.csv'
        with open(fname, 'w') as f:
            x0, y0 = geoms[np].exterior.coords.xy

            for i in range(len(x)-1):
                f.write(f'{x[:][i]} {y[:][i]}\n')

            if np == 0:
                csv_file = fname
            else:
                csv_file = csv_file + ',' + fname