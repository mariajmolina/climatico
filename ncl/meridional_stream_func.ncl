; computation of MOC
; provided by Aixue Hu

path="/glade/scratch/molina/amoc_exp/"
fin1=systemfunc("ls "+path+"b.e11.B1850LENS.f09_g16.FWAtSalP04Sv.pop.h.MOC.??????-??????.nc")

    f1=addfiles(fin1,"r")
    printVarSummary(fin1)
    ListSetType (f1, "cat")

moc=f1[:]->MOC
moc=runave_n(moc,12,0,0)
moc1=moc(5::12,0:1,0,:,:)  ; annual mean, global and atlantic, Eularian mean
printVarSummary(moc1)
delete(moc)
mocg02=dim_max(dim_max(moc1(:,:,29:59,280:390)))
mocg02(:,0)=mocg02(:,0)-mocg02(:,1)
delete(moc1)

printVarSummary(mocg02)

system ("rm ~/python_scripts/climatico/ncl/tempmoc.nc")            ; remove any pre-existing file
ncdf = addfile("~/python_scripts/climatico/ncl/tempmoc.nc" , "c")  ; open output netCDF file
ncdf->mocg02  = mocg02
