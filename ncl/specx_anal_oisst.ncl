;------------------- Unsupported: DJS -------------------
;--------------------------------------------------------
undef("SPECX_ANAL")
function SPECX_ANAL(X[*]:numeric, iopt[1]:integer, jave[1]:integer, pct[1]:numeric)
;
; created to illustrate the various steps used by the builtin function: specx_anal
;
local x, N, xVari, xVaro, cf, cr, ci, px, spcx, total_area, df, frq, wgts, sclVar, sdof, acr, tapcf
begin

   ; Reminder: X is input time series to perform specx_anal on.
   ; X is a one-dimensional array containing the data. Missing values are not allowed.
   
   N       = dimsizes(X)                ; returns the dimension sizes of the input variable
   
   xVari   = variance(X)*(N-1.)/N       ; input variance
   
   if (iopt.ge.0) then                  ; remove mean if iopt option set
       x = X - avg(X)
   else
       x = X
   end if
   
   if (iopt.ge.1) then                  ; detrend if iopt option set
       x = dtrend(x, False)             ; Estimates and removes the least squares linear trend 
   end if                               ; of the rightmost dimension from all grid points. False for no attributes returned.
   
   xVaro = variance(x)*(N-1.)/N         ; calculate input sample variance; population variance after pre-whitening
   
   x     = taper(x, pct, 0)             ; Applies split-cosine-bell tapering to one or more series across the 
                                        ; rightmost dimension.
                                      
   tapcf = 0.5*(128-93*pct)/(8-5*pct)^2 ; tapering correction factor Bloomfield
   
   cf    = ezfftf(x)                    ; Perform a Fourier analysis on a real periodic sequence. Foward FFT.
   cr    = cf(0, :)                     ; clarity ... real coef
   ci    = cf(1, :)                     ;             imaginary coef
   delete( cf )
   
   px    = cr^2 + ci^2                  ; periodogram of "x"
   
   wgts         = fspan(1,1,jave)       ; Creates an array of evenly-spaced floating point numbers.
   wgts(0)      = 0.5
   wgts(jave-1) = 0.5
   wgts         = wgts/sum(wgts)        ; normalize sum of wgts to one
   
   spcx         = wgt_runave(px,wgts,1) ; weighted run ave
   
   delete(px)
   
   ; step 7: normalize the area under the curve [6] to the variance.
   ;         Remember the first and last bandwidths are only 0.5*df.
   
   df           = 1./N
   total_area   = (spcx(0) + spcx(N/2-1))*(df/2) + sum(spcx(1:N/2-2))*df
   sclVar       = xVaro/total_area
   spcx         = spcx*sclVar
   
   frq          = fspan(df, 0.5, N/2)
   
   sdof         = 2/tapcf*sum(wgts^2)   ; sum squares of normalized wgts to calculate dof. bw: jones pg 202: murphy+katz
   
   sdof         = 2/(tapcf*sum(wgts^2))
   
   sdof@bw      = 0.5*sdof*df
   sdof@xvari   = xVari
   sdof@xvaro   = xVaro
   sdof@frq     = frq
   sdof@spcx    = spcx
   acr          = esacr(x,1)
   sdof@xlag1   = acr(1)
   
   system ("rm ~/python_scripts/climatico/ncl/spcx.nc")            ; remove any pre-existing file
   ncdf = addfile("~/python_scripts/climatico/ncl/spcx.nc" , "c")  ; open output netCDF file
   ncdf->spcx  = spcx  
   printVarSummary(spcx)
   
   system ("rm ~/python_scripts/climatico/ncl/frq.nc")            ; remove any pre-existing file
   ncdf = addfile("~/python_scripts/climatico/ncl/frq.nc" , "c")  ; open output netCDF file
   ncdf->frq   = frq
   printVarSummary(frq)
   
   return( sdof )
end
   
diri  = "~/python_scripts/climatico/ncl/"

fili  = "box_sst.nc"

in    = addfile(diri+fili,"r")
SST   = in->SST

iopt  = 1    ; remove series mean
;jave  = 7    ; 
jave  = 3    ; for oisst -- confirmed!
pct   = 0.1  ; taper 10% of the data

spec = SPECX_ANAL (SST,iopt,jave,pct)

;wks  = gsn_open_wks("png","spec")               ; send graphics to PNG file 

;res = True                              ; plot mods desired
;res@tiMainString = "SSTs Nino3.4 G2Sv (yrs 200-500)"                ; title
;res@tiXAxisString = "Frequency (cycles/month)"  ; xaxis
;res@tiYAxisString = "Variance"                  ; yaxis

;plot=gsn_csm_xy(wks,spec@frq,spec@spcx,res)     ; create plot 
