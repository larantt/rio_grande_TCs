;-- copy shapefile_utils.ncl from 
;-- https://www.ncl.ucar.edu/Applications/Scripts/shapefile_utils.ncl
load "$HOME/NCL/shapefiles/shapefile_utils.ncl" 

shpname  = "/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RG_Watershed_Boundary-shp/RG_Watershed_Boundary.shp" 
maskname = "/nfs/turbo/seas-hydro/laratt/TCdata/shapefiles/RGB_NLDAS_mask.nc" 

print_shapefile_info(shpname)

;-- open data file to get the grid to be used
f = addfile("/nfs/turbo/seas-hydro/laratt/NLDAS_monthly/NOAH/NLDAS_NOAH0125_M.A202311.002.nc","r")

;-- read variable
var                    =  f->LAI(0,:,:)
var@lat1d              =  f->LAT
var@lon1d              =  f->LON

;-- shapefile mask resources
opt             =  True
opt@return_mask =  True    ;-- this forces the return of a 0s and 1s mask array

;-- create the mask based on the given shapefile
mask_array             =  shapefile_mask_data(var, shpname, opt)
mask_array!0           = "y" 
mask_array!1           = "x" 
mask_array@coordinates = "LAT LON" 

;-- create new netCDF file and write mask array
system("rm -f " + maskname)
fout = addfile(maskname,"c")

fout->mask_array =  mask_array
fout->LAT        =  f->LAT
fout->LON        =  f->LON