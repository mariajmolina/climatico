;*************************************************
; spec_1.ncl
;
; Concepts illustrated:
;   - Calculating and plotting spectra
; From: https://www.ncl.ucar.edu/Applications/Scripts/spec_1.ncl
;
;************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl" 
;************************************************
begin
;************************************************
; variable and file handling
;************************************************
   fn  = "amoc_collapse_G2Sv.nc" ; define filename
   in  = addfile(fn,"r")                                 ; open netcdf file
   sst  = in->SST                                        ; get data
;************************************************
; set function arguments
;************************************************
; detrending opt: 0=>remove mean 1=>remove mean and detrend
  d = 0
; smoothing periodogram: (0 <= sm <= ??.) should be at least 3 and odd
  sm = 0
; percent tapered: (0.0 <= pct <= 1.0) 0.10 common. 
  pct = 0.0
;************************************************
; calculate spectrum
;************************************************
  spec = specx_anal(sst,d,sm,pct)
;************************************************
; plotting
;************************************************
   wks  = gsn_open_wks("png","spec")               ; send graphics to PNG file 

   res = True                              ; plot mods desired
   res@tiMainString = "SSTs Nino3.4 G2Sv (yrs 200-500)"                ; title
   res@tiXAxisString = "Frequency (cycles/month)"  ; xaxis
   res@tiYAxisString = "Variance"                  ; yaxis

   plot=gsn_csm_xy(wks,spec@frq,spec@spcx,res)     ; create plot   
;***********************************************
end
