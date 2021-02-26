;************************************************
; pop2lat_1.ncl
;************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"    
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"  
;
; export NCL_POP_REMAP=$... to where remap file is located
; here we use map_gx1v6_to_1x1d_bilin_da_100716.nc
; https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/ncl/
;
; This file still needs to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/popRemap.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl" 
;************************************************
;begin

  in  = addfile("/glade/scratch/molina/amoc_exp/b.e11.B1850LENS.f09_g16.FWPaSalP04Sv.pop.h.nday1.SST.03010101-03501231.nc","r")
  ;printVarSummary(in)
  sst = in->SST
  printVarSummary(sst)

;************************************************
; convert pop to a 1x1 degree grid
;************************************************

  remap = PopLatLon(sst,"gx1v6","1x1d","bilin","da","100716")
  printVarSummary(remap)

  system("/bin/rm -f /glade/scratch/molina/amoc_exp/b1d.e11.B1850LENS.f09_g16.FWPaSalP04Sv.pop.h.nday1.SST.03010101-03501231.nc")   ; remove any pre-existing file
  
  ncdf = addfile("/glade/scratch/molina/amoc_exp/b1d.e11.B1850LENS.f09_g16.FWPaSalP04Sv.pop.h.nday1.SST.03010101-03501231.nc" ,"c")  ; open output netCDF file
  filedimdef(ncdf,"time",-1,True)
;===================================================================
; output variables directly; NCL will call appropriate functions
; to write the meta data associated with each variable
;===================================================================
  ncdf->SST  = remap                          ; 4D  
