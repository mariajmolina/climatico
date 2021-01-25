;----------------------------------------------------------------------
; indices_oni_1.ncl
;
; Concepts illustrated:
;   - Computing the Oceanic Nino Index
;   - Drawing a time series plot
;   From: https://www.ncl.ucar.edu/Applications/Scripts/indices_oni_1.ncl
;----------------------------------------------------------------------
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl" 
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl" 
;---------------------------------------------------------------------------
; NOAA's operational definitions of El Nino±o and La Nino±a conditions are based
; upon the Oceanic Nino±o Index [ONI]. The ONI is defined as the 3-month running
; means of SST anomalies in the Nino±o 3.4 region [5N-5S, 120-170W]. The anomalies
; are derived from the 1971-2000 SST climatology.
;
; The Nino±o 3.4 anomalies may be thought of as representing the average equatorial
; SSTs across the Pacific from about the dateline to the South American coast.
; To be classified as a full-fledged El Nino±o and La Nino±a episode the ONI must excee
; d +0.5 [El Nino±o] or -0.5 [La Nino±a] for at least five consecutive months. 
;---------------------------------------------------------------------------
;                         User input
;---------------------------------------------------------------------------

  latS   = -5.0
  latN   =  5.0
  lonL   = 190.0
  lonR   = 240.0
  
  nrun   = 3             ; length of running average

  ymStrt = 070101
  ymLast = 075012        ; last full year on file

  clStrt = 072001        ; climatology start    
  clLast = 074012        ;             last

  yrStrt = ymStrt/100
  yrLast = ymLast/100

  pltType = "png"       ; send graphics to PNG file
  pltDir  = "./"        ; dir to which plots are sent
  
  pltName = "ONI."+(ymStrt/100)+"-"+(ymLast/100)

  pltTitle= "ONI: "+(ymStrt/100)+"-"+(ymLast/100) \
          + ": Base "+(clStrt/100)+"-"+(clLast/100)

;-------------------- End User Input ---------------------------------------

  diri  = "./"
  
  fili  = "simple_G2Sv_700.nc" ;
  
  in    = addfile(diri+fili,"r")

  tStrt = 0                  ; indices of selected times
  tLast = 599                ; ind(YYYYMM.eq.ymLast)
  
  x     = in->SST(:,0,{latS:latN},{lonL:lonR})

  date  = in->time(:)
  yyyymm= date/100
  ntim  = dimsizes(date)

;*********************************
; time indices for base climatology 
;*********************************

  iClmStrt = 239 
  iClmLast = 478 

;*********************************
; Climatology and anomalies from base climatology   
;*********************************

  xClm     = clmMonTLL(x(iClmStrt:iClmLast,:,:))

  xAnom    = calcMonAnomTLL (x,  xClm ) 
  xAnom@long_name = "SST Anomalies"

;*********************************
; Unweighted areal average anomalies (time series)
; Small latitudinal extent so no need to weight    
;*********************************

  xAnom_avg = wgt_areaave_Wrap(xAnom, 1.0, 1.0, 1)
  xAnom_avg@long_name = "areal avg anomalies"

;*********************************
; Perform an unweighted 'nrun' month running average
;*********************************

  xAnom_avg = runave_n_Wrap (xAnom_avg, nrun, 1, 0)

;*********************************
; plot graph
;*********************************

  wks = gsn_open_wks(pltType, pltDir+pltName)     

  res                  = True       
  res@gsnMaximize      = True
  res@gsnYRefLine      = 0.0            ; create a reference line 
  res@gsnAboveYRefLineColor = "red"              ; above ref line fill red
  res@gsnBelowYRefLineColor = "blue"             ; below ref line fill blue

  res@vpHeightF        = 0.4            ; change aspect ratio of plot
  res@vpWidthF         = 0.8            
  res@vpXF             = 0.1            ; start plot at x ndc coord 
  
  res@trYMinF          = -3.0           ; min value on y-axis
  res@trYMaxF          =  3.0           ; max value on y-axis

  res@tiMainString     = pltTitle                            
  
  res@tiYAxisString    = "Anomalies (C)"    ; y-axis label 
  plot = gsn_csm_xy (wks, yyyymm, xAnom_avg, res)
