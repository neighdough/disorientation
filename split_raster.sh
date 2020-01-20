#!/bin/sh

cd $HOME/temp

for fname1 in ./tracts/*.gpkg; 
  do
    fname2="$(basename ${fname1%.*})"
    gdalwarp -dstnodata -9999 -cutline $fname1 -crop_to_cutline -of GTiff ./parcel_luc.tif ./split_raster/$fname2.tif
  done
