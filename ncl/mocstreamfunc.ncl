;*****************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/popRemap.ncl"
;; Provided by Aixue Hu
;*****************************************************
begin

data=1
if data.eq.0 then
path1="/datalocal/ccpb/ocn_ahu_diagnostics/FWAtSal/"
path2="/datalocal/ccpb/ocn_ahu_diagnostics/Large_Ensemble/TVTS/"
f0=addfile(path2+"TVTS_b.e11.B1850C5CN.f09_g16.005.pop.h.0400-2200.ann_ave.nc","r")

    amsfc=f0->amsf
    pmsfc=f0->pmsf
    bervc=f0->BeringV
;    drakvc=f0->DrakeV
    indtvc=f0->InthrouV

fin3=systemfunc("ls "+path1+"TVTS_b.e11.B1850LENS.f09_g16.FWAtSalP02Sv.pop.h.????-????.ann.nc")
fin4=systemfunc("ls "+path1+"TVTS_b.e11.B1850LENS.f09_g16.FWAtSalP04Sv.pop.h.????-????.ann.nc")
fin1=systemfunc("ls "+path1+"TVTS_b.e11.B1850LENS.f09_g16.FWAtSalG02Sv.pop.h.????-????.ann.nc")
fin2=systemfunc("ls "+path1+"TVTS_b.e11.B1850LENS.f09_g16.FWAtSalG04Sv.pop.h.????-????.ann.nc")
fin5=systemfunc("ls "+path1+"TVTS_b.e11.B1850LENS.f09_g16.FWPaSalP04Sv.pop.h.????-????.ann.nc")


    f1=addfiles(fin1,"r")
    ListSetType (f1, "cat")
    f2=addfiles(fin2,"r")
    ListSetType (f2, "cat")
    f3=addfiles(fin3,"r")
    ListSetType (f3, "cat")
    f4=addfiles(fin4,"r")
    ListSetType (f4, "cat")
   f5=addfiles(fin5,"r")
    ListSetType (f5, "cat")


    amsf1=f1[:]->amsf
    dimz1=dimsizes(amsf1)
    amsf=new((/6,dimz1(1),dimz1(2)/),float)
    pmsf=new((/6,dimz1(1),dimz1(2)/),float)
    bering=new((/5,810/),float)
    drakep=new((/5,810/),float)
    indthg=new((/5,810/),float)

    amsf!0="case"
    amsf!1="lev"
    amsf!2="lat"
    amsf&case=ispan(1,6,1)
    amsf&lev=amsf1&lev
    amsf&lat=amsf1&lat
    pmsf!0="case"
    pmsf!1="lev"
    pmsf!2="lat"
    pmsf&case=ispan(1,6,1)
    pmsf&lev=amsf1&lev
    pmsf&lat=amsf1&lat

    
    pmsf1=f1[:]->pmsf
    berv1=f1[:]->BeringV
    drakv1=f1[:]->DrakeV
    indtv1=f1[:]->InthrouV

    amsfc@_FillValue = -999.
    pmsfc@_FillValue = -999.
    amsfc@_FillValue = amsf1@_FillValue
    pmsfc@_FillValue = pmsf1@_FillValue
printVarSummary(amsfc)
printVarSummary(amsf)

    print(dimz1(0))
    amsf(0,:,112:330)=(/amsfc(0,:,112:330)/)
    amsf(1,:,:)=dim_avg_n(amsf1(200:499,:,:),0)
    pmsf(0,:,112:330)=(/pmsfc(0,:,112:330)/)
    pmsf(1,:,:)=dim_avg_n(pmsf1(200:499,:,:),0)
    bering(0,0:dimz1(0)-1)=berv1
    drakep(0,0:dimz1(0)-1)=drakv1
    indthg(0,0:dimz1(0)-1)=indtv1
    delete(amsf1)
    delete(pmsf1)

    
    amsf2=f2[:]->amsf
    pmsf2=f2[:]->pmsf
    berv2=f2[:]->BeringV
    drakv2=f2[:]->DrakeV
    indtv2=f2[:]->InthrouV
    dimz1=dimsizes(amsf2)
    amsf(2,:,:)=dim_avg_n(amsf2(200:499,:,:),0)
    pmsf(2,:,:)=dim_avg_n(pmsf2(200:499,:,:),0)
    bering(1,0:dimz1(0)-1)=berv2
    drakep(1,0:dimz1(0)-1)=drakv2
    indthg(1,0:dimz1(0)-1)=indtv2
    delete(amsf2)
    delete(pmsf2)

    amsf3=f3[:]->amsf
    pmsf3=f3[:]->pmsf
    berv3=f3[:]->BeringV
    drakv3=f3[:]->DrakeV
    indtv3=f3[:]->InthrouV
    dimz1=dimsizes(amsf3)
    amsf(3,:,:)=dim_avg_n(amsf3(200:499,:,:),0)
    pmsf(3,:,:)=dim_avg_n(pmsf3(200:499,:,:),0)
    bering(2,0:dimz1(0)-1)=berv3
    drakep(2,0:dimz1(0)-1)=drakv3
    indthg(2,0:dimz1(0)-1)=indtv3
    delete(amsf3)
    delete(pmsf3)


    amsf4=f4[:]->amsf
    pmsf4=f4[:]->pmsf
    berv4=f4[:]->BeringV
    drakv4=f4[:]->DrakeV
    indtv4=f4[:]->InthrouV
    dimz1=dimsizes(amsf4)
    amsf(4,:,:)=dim_avg_n(amsf4(200:499,:,:),0)
    pmsf(4,:,:)=dim_avg_n(pmsf4(200:499,:,:),0)
    bering(3,0:dimz1(0)-1)=berv4
    drakep(3,0:dimz1(0)-1)=drakv4
    indthg(3,0:dimz1(0)-1)=indtv4
    delete(amsf4)
    delete(pmsf4)

    amsf5=f5[:]->amsf
    pmsf5=f5[:]->pmsf
    berv5=f5[:]->BeringV
    drakv5=f5[:]->DrakeV
    indtv5=f5[:]->InthrouV
    dimz1=dimsizes(amsf5)
    amsf(5,:,:)=dim_avg_n(amsf5(100:249,:,:),0)
    pmsf(5,:,:)=dim_avg_n(pmsf5(100:249,:,:),0)
    bering(4,0:dimz1(0)-1)=berv5
    drakep(4,0:dimz1(0)-1)=drakv5
    indthg(4,0:dimz1(0)-1)=indtv5
    delete(amsf5)
    delete(pmsf5)

;fcc=addfile("FW_msf_trans_12032018.nc","c")
fcc=addfile("FW_msf_trans_080562020.nc","c")
fcc->amsf=amsf
fcc->pmsf=pmsf
fcc->bering=bering
fcc->drakep=drakep
fcc->indthg=indthg
fcc->bervc=bervc
fcc->indtvc=indtvc
else
;fcc=addfile("FW_msf_trans_12032018.nc","r")
fcc=addfile("FW_msf_trans_080562020.nc","r")
amsf=fcc->amsf
pmsf=fcc->pmsf
bering=fcc->bering
drakep=fcc->drakep
indthg=fcc->indthg
bervc=fcc->bervc
indtvc=fcc->indtvc
end if

bering=runave(bering,10,1)
drakep=runave(drakep,10,1)
indthg=runave(indthg,10,1)

amsfd=amsf
pmsfd=pmsf

do i=1,5
  amsfd(i,:,:)=amsfd(i,:,:)-amsf(0,:,:)
  pmsfd(i,:,:)=pmsfd(i,:,:)-pmsf(0,:,:)
end do

;bervc=runave(bervc,10,1)
;indtvc=runave(indtvc,10,1)


 wks = gsn_open_wks("ps","FW_streamfunction_time_new.ps")
  resR                        = True
  resR@vpHeightF=0.5
  resR@vpWidthF = 0.75
  resR@gsnFrame = False
  resR@gsnDraw  = False
  resR@xyLineThicknesses      = (/3,3,3,3/)
  resR@xyLineColors = (/"black","red","blue","green"/)
  resR@xyDashPatterns = (/0,0,0,0/)
  resR@pmLegendDisplayMode    = "Always"            ; turn on legend
  resR@pmLegendSide           = "Top"               ; Change location of
  resR@pmLegendParallelPosF   = .33                  ; move units right
  resR@pmLegendOrthogonalPosF = -0.52                ; move units down
  resR@pmLegendWidthF         = 0.15                ; Change width and
  resR@pmLegendHeightF        = 0.18                ; height of legend.
  resR@lgPerimOn              = False               ; turn off box around
  resR@lgLabelFontHeightF     = .03                 ; label font height
  resR@xyExplicitLegendLabels = (/"0.2 Sv GLOBx","0.4 Sv GLOBx","0.2 Sv NPAC","0.4 Sv NPAC"/)         ; create explicit labels

  resR@tiYAxisString = "Sv (10~S~6~N~ m~S~3~N~s~S~-1~N~)"  ; axis string
  resR@tiXAxisString = "Model year"  ; axis string

  resR@trXMaxF = 800.

  time=ispan(1,810,1)
  resR@tiMainString      = "Bering Strait transport"
  plota=gsn_csm_xy(wks,time,bering,resR)
  resR@pmLegendDisplayMode    = "Never"            ; turn on legend
  resR@tiMainString      = "Drake Passage transport"
  plotb=gsn_csm_xy(wks,time,drakep,resR)
  resR@tiMainString      = "Indonesia Throughflow transport"
  plotc=gsn_csm_xy(wks,time,indthg,resR)

pres=True
gsn_panel(wks,(/plota,plotb,plotc/),(/3,1/),pres)

  res                     = True               ; plot mods desired
  res@gsnDraw             = False              ; don't draw yet
  res@gsnFrame            = False              ; don't advance frame yet
  res@cnFillOn         = True
  res@cnLinesOn        = False
  res@cnLineLabelsOn      = False
  res@gsnSpreadColors     = True               ; use full color ma
  res@cnInfoLabelOn       = False
  res@gsnYAxisIrregular2Linear = True
  res@gsnXAxisIrregular2Linear = True
  res@trYReverse          =True
  res@lbLabelStride    = 4                  ; every ot
  res@lbLabelBarOn    = False                  ; every ot
  res@lbOrientation =   "vertical"
  res@cnLevelSelectionMode =  "ManualLevels"   ; manual levels
  res@cnMinLevelValF       = -24.               ; min level
  res@cnMaxLevelValF       = 26.               ; max level
  res@cnLevelSpacingF      =  2.               ; interval
  res@vpHeightF = 0.5
  res@vpWidthF  = 0.75
  res@gsnContourZeroLineThicknessF  = 2
  res@gsnContourNegLineDashPattern  = 1
  res@tiYAxisString = "" ;Depth (meter)"
 res@tiYAxisString = "" ;Depth (meter)"
   cmap = RGBtoCmap("../jma.rgb")
   gsn_define_colormap(wks,cmap)

plot=new(12,graphic)
plotd=new(12,graphic)
plotst=new(12,graphic)
plotsd=new(12,graphic)

tit=(/"Control AMOC","Control PMOC","0.2 Sv GLOBx AMOC","0.2 Sv GLOBx PMOC",\
      "0.4 Sv GLOBx AMOC","0.4 Sv GLOBx PMOC",\
      "0.2 Sv NPAC AMOC","0.2 Sv NPAC PMOC",\
      "0.4 Sv NPAC AMOC","0.4 Sv NPAC PMOC",\
      "PACSALT AMOC","PACSALT PMOC"/)

amsf=mask(amsf,abs(amsf).le.1.e-5,False)
pmsf=mask(pmsf,abs(pmsf).le.1.e-5,False)
amsfd=mask(amsfd,abs(amsfd).le.1.e-5,False)
pmsfd=mask(pmsfd,abs(pmsfd).le.1.e-5,False)
do i=0,5
  res@cnMinLevelValF       = -24.               ; min level
  res@cnMaxLevelValF       = 26.               ; max level
  res@cnLevelSpacingF      =  2.               ; interval
res@gsnLeftString = tit(2*i)
plot(2*i)=gsn_csm_contour(wks,amsf(i,:,100:349),res)
res@gsnLeftString = tit(2*i+1)
plot(2*i+1)=gsn_csm_contour(wks,pmsf(i,:,100:349),res)
if i.gt.0 then
  res@cnMinLevelValF       = -24.               ; min level
  res@cnMaxLevelValF       = 26.               ; max level
  res@cnLevelSpacingF      =  2.               ; interval
res@gsnLeftString = tit(2*i)
plotd(2*i)=gsn_csm_contour(wks,amsfd(i,:,100:349),res)
res@gsnLeftString = tit(2*i+1)
plotd(2*i+1)=gsn_csm_contour(wks,pmsfd(i,:,100:349),res)
end if
end do

tit=(/"Control ASTC","Control PSTC","0.2 Sv GLOBx ASTC","0.2 Sv GLOBx PSTC",\
      "0.4 Sv GLOBx ASTC","0.4 Sv GLOBx PSTC",\
      "0.2 Sv NPAC ASTC","0.2 Sv NPAC PSTC",\
      "0.4 Sv NPAC ASTC","0.4 Sv NPAC PSTC",\
      "PACSALT ASTC","PACSALT PSTC"/)

do i=0,5
  res@cnMinLevelValF       = -24.               ; min level
  res@cnMaxLevelValF       = 26.               ; max level
  res@cnLevelSpacingF      =  2.               ; interval
res@gsnLeftString = tit(2*i)
plotst(2*i)=gsn_csm_contour(wks,amsf(i,0:34,100:260),res)
res@gsnLeftString = tit(2*i+1)
plotst(2*i+1)=gsn_csm_contour(wks,pmsf(i,0:34,100:260),res)

if i.gt.0 then
  res@cnMinLevelValF       = -24.               ; min level
  res@cnMaxLevelValF       = 26.               ; max level
  res@cnLevelSpacingF      =  2.               ; interval
res@gsnLeftString = tit(2*i)
plotsd(2*i)=gsn_csm_contour(wks,amsfd(i,0:34,100:260),res)
res@gsnLeftString = tit(2*i+1)
plotsd(2*i+1)=gsn_csm_contour(wks,pmsfd(i,0:34,100:260),res)
end if
end do

  pres@gsnPanelLabelBar  =  True
  pres@lbLabelStride    = 4
pres@gsnPanelFigureStringsPerimOn = "False"
pres@gsnPanelFigureStrings  = (/"a)","b)","c)","d)","e)","f)","g)","h)"/)
pres@gsnPanelFigureStringsFontHeightF = 0.03
pres@amJust = "BottomRight"
pres@gsnPanelFigureStringsBackgroundFillColor = "transparent"

gsn_panel(wks,plot,(/6,2/),pres)
gsn_panel(wks,plot(2:5),(/2,2/),pres)
gsn_panel(wks,(/plot(0),plot(1),plot(6),plot(7),plot(8),plot(9)/),(/3,2/),pres)

gsn_panel(wks,plotd,(/6,2/),pres)
gsn_panel(wks,plotd(2:5),(/2,2/),pres)
gsn_panel(wks,(/plot(0),plot(1),plotd(6),plotd(7),plotd(8),plotd(9)/),(/3,2/),pres)

gsn_panel(wks,plotst,(/6,2/),pres)
gsn_panel(wks,plotst(2:5),(/2,2/),pres)
gsn_panel(wks,(/plotst(0),plotst(1),plotst(6),plotst(7),plotst(8),plotst(9)/),(/3,2/),pres)

gsn_panel(wks,plotsd,(/6,2/),pres)
gsn_panel(wks,plotsd(2:5),(/2,2/),pres)
gsn_panel(wks,(/plotst(0),plotst(1),plotsd(6),plotsd(7),plotsd(8),plotsd(9)/),(/3,2/),pres)

gsn_panel(wks,(/plot(10),plot(11),plotst(10),plotst(11)/),(/2,2/),pres)
gsn_panel(wks,(/plotd(10),plotd(11),plotsd(10),plotsd(11)/),(/2,2/),pres)

end


