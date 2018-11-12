tic()
@everywhere using PyPlot
@everywhere using DistributedArrays
@everywhere using HDF5   
@everywhere using Devectorize
@everywhere using Optim
@everywhere root="/home/data/DATA/ISC/IT90_3/"  #folder
@everywhere file1=string(root,"declust_z0193")   #input files - (Density,Dark_Matter_Density,Temperature)
@everywhere file2=string(root,"declust_v_z0193")    #input files - (vx,vy,vz)
@everywhere file3=string(root,"deDD0193.conv2")  #input files - (line4:convd, line5: convv)  
@everywhere const ncell=320
@everywhere const kb=1.38*10.^(-16.)
@everywhere const mp=1.67e-24
@everywhere const gamma=1.66667 
@everywhere const fac=gamma*kb/mp
@everywhere const mthr=1.3

toc()
tic()



@everywhere function Smooth(data::Array{Float64,3},w::Int) #only for cubic data set and even lenght windows

n=size(data)
ris=similar(data)

@fastmath low=1+w/2
low=convert(Int,low)
@fastmath high=n[2]-w/2
high=convert(Int,high)
@fastmath wmez=w/2
wmez=convert(Int,wmez)

@inbounds  for j in low:high, k in low:high
           ris1d=(data[:,j,k])
           risx=smox(wmez,ris1d,low,high,ncell)
           ris[:,j,k]=risx
           end
           ris[1:low,:,:]=data[1:low,:,:]
           ris[high:end,:,:]=data[high:end,:,:]
@inbounds  for i in low:high, k in low:high
           ris1d=(ris[i,:,k])
           risy=smox(wmez,ris1d,low,high,ncell)
           ris[i,:,k]=risy
           end
           ris[:,1:low,:]=data[:,1:low,:]
           ris[:,high:end,:]=data[:,high:end,:]
@inbounds  for i in low:high, j in low:high
           ris1d=(ris[i,j,:])
           risz=smox(wmez,ris1d,low,high,ncell)
           ris[i,j,:]=risz
           end
           ris[:,:,1:low]=data[:,:,1:low]
           ris[:,:,high:end]=data[:,:,high:end]
    
id1=find(x-> (x <= 0),ris) #using to avoid the zero value (if plot is set in log space)
ris[id1]=1.e-35    
    return ris
end


@everywhere function smox(wmez::Int,ris1d::Array{Float64},low::Int,high::Int,ncell::Int)
   risx=Array{Float64}(ncell)
    @inbounds    for i in low:high
    @fastmath  risx[i]=mean(ris1d[i-wmez:i+wmez])
           end

   return risx
end

@everywhere function maps(field,window::Int,lim1::Int,lim2::Int)   #function to make the maps of project fields 

 if lim1 != 0 && lim2 != 0 
    n=size(field)
    proj=Array{Float64}(n[2],n[2])
@fastmath  lim1=1      #using these parameters to make projection or slice
@fastmath  lim2=n[2]
    lim1=convert(Int,lim1)
    lim2=convert(Int,lim2)
    
@inbounds for i=1:n[2], j=1:n[2] 
@fastmath proj[i,j]=sum(field[i,j,lim1:lim2])   #projection along z-axis
          end
    
    if window == 0      #in order to distangle unsmoothed-smoothed velocity field (edge problem) 
 @fastmath proj=proj/n[2]
    else window != 0
 @fastmath proj=proj/(n[2]-window)
    end    
     return proj
 else
      n=size(field)
     proj=Array{Float64}(n[2],n[2])
     
    lim1=convert(Int,lim1)
    lim2=convert(Int,lim2)
    
@inbounds for i=1:n[2], j=1:n[2] 
@fastmath proj[i,j]=sum(field[i,j,lim1:lim2])   #projection along z-axis
          end
    
    if window == 0      #in order to distangle unsmoothed-smoothed velocity field (edge problem) 
 @fastmath proj=proj/n[2]
    else window != 0
 @fastmath proj=proj/(n[2]-window)
    end    
     return proj
 end    
end


@everywhere function vmod(vx::Array{Float32,3},vy::Array{Float32,3},vz::Array{Float32,3},ncell::Int) #make velocity field
    
 v=Array{Float64}(ncell,ncell,ncell)
        @inbounds for i in 1:ncell, j in 1:ncell, k in 1:ncell
        @fastmath  v[j,i,k]=sqrt(vx[i,j,k]^2+vy[i,j,k]^2+vz[i,j,k]^2)
        end
        return v
        end

@everywhere function vturb(vtot::Array{Float64,3},vsmooth::Array{Float64,3},ncell::Int)   #make turbulence velocity field
    vturb=similar(vtot)
#=@inbounds for i in 1:ncell, j in 1:ncell, k in 1:ncell 
@fastmath  vturb[i,j,k]=vtot[i,j,k]-vsmooth[i,j,k]
           end =#
    vturb .= vtot .- vsmooth
    id1=find(x-> (x <= 0),vturb) #using to avoid the zero value (if plot is set in log space)
    vturb[id1]=1.e-35   
   return vturb
end

######## Shock Finder ########

@everywhere function histog(mm::Vector{Float64},hm::Vector{Float64},macx::Array{Float64})

@fastmath for i in eachindex(mm)
   tag=find( x->(x >= mm[i]),macx)
   nshock=size(tag)
   hm[i]=nshock[1]
  end
return hm
end


@everywhere  function(psi_m_all,mpsi,f)
# ............reads in the Psi(M) function of Hoeft & Bruggen
fhb="mach_psi_table.txt"
nt=13
f1=string(fhb)
ff=readdlm(f1)

n=size(v)
mpsi=v[:,1]
ff=v[:,2:n[2]]

 end




@everywhere function psi_sel(mach,t,mpsi,f)

const nt=13
const n33=244

th=[1.00e-04,3.162e-04,1.000e-03,3.162e-03,1.000e-02,3.162e-02,1.000e-01,3.162e-01,1.000e+00,3.162e+00,1.000e+01,3.162e+01,1.000e+02]
#...................selection from the HB efficiency function for Psi(M)
to=t

@inbounds @fastmath for ii in 1:nt-1 
if to > th[ii] && to < th[ii+1]
tsel=ii
end
end


@inbounds @fastmath for ii in 1:n33-1
if m >= mpsi[ii] && m < mpsi[ii+1]
msel=ii
end
end

if to < th[1] 
tsel=1
end
if to > th[nt] 
tsel=nt
end
if m < mpsi[1] 
msel=1
end

if m > mpsi[n33] 
msel=n33
end

psi=f(tsel,msel)

 return psi

end

@everywhere function shocks(d::Array{Float32,3},t::Array{Float32,3},vx::Array{Float32,3},vy::Array{Float32,3},vz::Array{Float32,3})
#::Array{Float64},t::Array{Float64},vx::Array{Float64},vy::Array{Float64},vz::Array{Float64},bx::Array{Float64},by::Array{Float64},bz::Array{Float64})

#::Array{Float64},t::Array{Float64},vx::Array{Float64},vy::Array{Float64},vz::Array{Float64},bx::Array{Float64},by::Array{Float64},bz::Array{Float64},mach::Array{Float64})

n3=size(d)

 macx=Array{Float64}(n3[1],n3[2],n3[3])
 macy=Array{Float64}(n3[1],n3[2],n3[3])
 macz=Array{Float64}(n3[1],n3[2],n3[3])
 mach=Array{Float64}(n3[1],n3[2],n3[3])

# macx[:,:,:]=0.
# macy[:,:,:]=0.
# macz[:,:,:]=0.
# mach[:,:,:]=0.

  div=Array{Float64}(n3[1],n3[2],n3[3])
#  div[:,:,:]=0.

 @inbounds for i in 2:n3[1]-1, j in 2:n3[2]-1, l in 2:n3[3]-1
 @fastmath div[i,j,l]=0.5*(vx[i+1,j,l]-vx[i-1,j,l]+vy[i,j+1,l]-vy[i,j-1,l]+vz[i,j,l+1]-vz[i,j,l-1])
  end
    
  tag=find( x->(x <= -1.e6), div)
    nshock=size(tag)
  ijk=ind2sub((n3[1],n3[2],n3[3]),tag)
   println(nshock, "candidate shocks") 
   v1=Vector{Float64}(2)

@simd for i in 1:nshock[1]
 # in eachindex(candidate)
    ijk=ind2sub((n3[1],n3[2],n3[3]),tag[i])
    
  a=ijk[1]
  b=ijk[2]
  c=ijk[3]
 
@inbounds  dvx=-1.*(vx[a-1,b,c]-vx[a+1,b,c])  #...we need velocity in cm/s here --> k\m/s
@inbounds  dvy=-1.*(vy[a,b-1,c]-vy[a,b+1,c])
@inbounds  dvz=-1.*(vz[a,b,c-1]-vz[a,b,c+1])

@inbounds  if dvx < 0. && t[a+1,b,c] > t[a-1,b,c]
   vvx=dvx/(sqrt(fac*t[a-1,b,c]))
   mx=(4.*abs(vvx)+sqrt(16.*vvx*vvx+36.))*0.166666    
   v1[1]=mx
   v1[2]=macx[a+1,b,c]
   macx[a+1,b,c]=maximum(v1)
   elseif dvx <0. && t[a+1,b,c] <  t[a-1,b,c] 
   mx=abs(dvx)/(sqrt(fac*t[a+1,b,c]))
   mx=(4.*mx+sqrt(16.*mx*mx+36.))*0.166666
   v1[1]=mx
   v1[2]=macx[a-1,b,c]
   macx[a-1,b,c]=maximum(v1)
  end

@inbounds if dvy < 0. && t[a,b+1,c] > t[a,b-1,c]
@fastmath   vvy=dvy/(sqrt(fac*t[a,b-1,c]))
@fastmath   my=(4.*abs(vvy)+sqrt(16.*vvy*vvy+36.))*0.166666
   v1[1]=my
   v1[2]=macy[a,b+1,c]
   macy[a,b+1,c]=maximum(v1)
   elseif dvy <0 && t[a,b+1,c] <  t[a,b-1,c]
@fastmath   my=abs(dvy)/(sqrt(fac*t[a,b+1,c]))
@fastmath   my=(4.*my+sqrt(16.*my*my+36.))*0.166666
   v1[1]=my
   v1[2]=macy[a,b-1,c]
   macy[a,b-1,c]=maximum(v1)
  end

@inbounds if dvz < 0. && t[a,b,c+1] > t[a,b,c-1]
@fastmath   vvz=dvz/(sqrt(fac*t[a,b,c-1]))
@fastmath   mz=(4.*abs(vvz)+sqrt(16.*vvz*vvz+36.))*0.166666
   v1[1]=mz
   v1[2]=macz[a,b,c+1]
   macz[a,b,c+1]=maximum(v1)
   elseif dvz <0. && t[a,b,c+1] <  t[a,b,c-1]
 @fastmath  mz=abs(dvz)/(sqrt(fac*t[a,b,c+1]))
 @fastmath  mz=(4.*mz+sqrt(16.*mz*mz+36.))*0.166666
   v1[1]=mz
   v1[2]=macz[a,b,c-1]
   macz[a,b,c-1]=maximum(v1)
  end
@inbounds @fastmath   mach[i]=sqrt(macx[i]^2. +macy[i]^2.+macz[i]^2.) 
  
 end

#@fastmath   for i in eachindex(macx)
#@inbounds   mach[i]=sqrt(macx[i]^2. +macy[i]^2.+macz[i]^2.)
#    end
    println(minimum(mach)," ",maximum(mach))

#...cleaning for shock thickness

  tag0=find(x->(x <= mthr), mach)
  mach[tag0]=0.  

  tag=find( x->(x >= mthr), mach)
  
  nshock=size(tag)
  ijk=ind2sub((n3[1],n3[2],n3[3]),tag)

 v1=Vector{Float64}(3) 
 
@simd for i in eachindex(mach[tag])
@inbounds a=ijk[1][1]
@inbounds b=ijk[2][1]
@inbounds c=ijk[3][1]

  x1=a-1
  x2=a
  x3=a+1
  y1=b-1
  y2=b
  y3=b+1
  z1=c-1
  z2=c
  z3=c+1
 
  if x1 < 1 
  x1+=1
  end  
 
  if x3 > n3[1]+1
  x3-=1
  end
  if y1 < 1
  y1+=1
  end
  if y3 > n3[2]+1  
  y3-=1
  end 
  if z1 < 1
  z1+=1
  end
  if z3 > n3[3]+1
  z3-=1
  end  

@inbounds      v1[1]=macx[x1,y2,z2]
@inbounds      v1[2]=macx[x2,y2,z2]
@inbounds      v1[3]=macx[x3,y2,z2]
      max3=maximum(v1)
    
@inbounds     if macx[x1,y2,z2] != max3 
@inbounds     macx[x1,y2,z2]=0.
    end
@inbounds     if macx[x2,y2,z2] != max3 
@inbounds     macx[x2,y2,z2]=0.
    end
@inbounds     if macx[x3,y2,z2] != max3 
@inbounds     macx[x3,y2,z2]=0.
    end

@inbounds    v1[1]=macy[x2,y1,z2]
@inbounds    v1[2]=macy[x2,y2,z2]
@inbounds    v1[3]=macy[x2,y3,z2]

     max3=maximum(v1)
@inbounds     if macy[x2,y1,z2] != max3 
@inbounds     macy[x2,y1,z2]=0.
    end   
@inbounds     if macy[x2,y2,z2] != max3
@inbounds     macy[x2,y2,z2]=0.
    end
@inbounds     if macy[x2,y3,z2] != max3
@inbounds     macy[x2,y3,z2]=0.
    end

@inbounds     v1[1]=macz[x2,y2,z1]
@inbounds     v1[1]=macz[x2,y2,z2]
@inbounds     v1[1]=macz[x2,y2,z3]

    max3=maximum(v1)

@inbounds    if macz[x2,y2,z1] != max3 
@inbounds    macz[x2,y2,z1]=0.
   end   

@inbounds     if macz[x2,y2,z2] != max3
@inbounds     macz[x2,y2,z2]=0.
    end

@inbounds     if macz[x2,y2,z3] != max3 
@inbounds     macz[x2,y2,z3]=0.
    end


   end


@simd   for i in eachindex(macx[tag])
@fastmath   macx[i]=sqrt(macx[i]^2. +macy[i]^2.+macz[i]^2.)

   end
   println(minimum(macx)," ",maximum(macx))
   clear!(:macy)
   clear!(:macz)
   clear!(:div)
   clear!(:mach)   
#  mach=abs.(div)
  return macx
end

########## Shock Finder ###########


toc()
tic()

d=h5read(file1,"Density")
dm=h5read(file1,"Dark_Matter_Density")
temp=h5read(file1,"Temperature")

vx=h5read(file2,"x-velocity")   #read velocity file
vy=h5read(file2,"y-velocity")
vz=h5read(file2,"z-velocity")

info=readdlm(file3, '\t', Float64, '\n') #read enzo's conversion factors

toc()
tic()

d .= d .* info[4]
dm .= dm .* info[4]


vx .= vx .* info[5]
vy .= vy .* info[5]
vz .= vz .* info[5]

v=vmod(vx,vy,vz,ncell)   #velocity field
macx=shocks(d,temp,vx,vy,vz)

win=10
res=Smooth(v,win)    #smoothed velocity field
turb=vturb(v,res,ncell)    #turbulence velocity field


#ds=maps(v,0)     #velocity field's map
#dssm=maps(res,win)
ds=maps(d,0,150,170)
dsm=maps(macx,0,150,170)
dst=maps(turb,win,150,170)

toc()

#=figure(1,figsize=(15,10))
subplot(131)
axis(:equal)
box(:off)
title("Unsmoothed")
pcolor(ds, norm=matplotlib[:colors][:LogNorm](vmin=minimum(2.4e7), vmax=maximum(5e7)), cmap="jet")
xticks([])
yticks([])
colorbar(orientation="horizontal", fraction=0.4, label=L"\mathrm{cm \ s^{-1}}")
savefig("/Users/matteoangelinelli/Desktop/v_field.png")

figure(2,figsize=(20,20))
subplot(132)
axis(:equal)
box(:off)
title("Smoothed")
pcolor(dssm, norm=matplotlib[:colors][:LogNorm](vmin=minimum(2.4e7), vmax=maximum(5e7)), cmap="jet")
xticks([])
yticks([])
colorbar(orientation="horizontal", fraction=0.4, label=L"\mathrm{cm \ s^{-1}}")
savefig("/Users/matteoangelinelli/Desktop/v_smooth.png")
=#

figure(1,figsize=(15,10))
subplot(131)
axis(:equal)
box(:off)
title("Gas Density")
pcolor(ds, norm=matplotlib[:colors][:LogNorm](vmin=minimum(1e-31), vmax=maximum(ds)), cmap="PuBu_r")
xticks([])
yticks([])
colorbar(orientation="horizontal", fraction=0.4, label=L"\mathrm{g \ cm^{-3}}")

subplot(132)
axis(:equal)
box(:off)
title("Shocks")
pcolor(dsm, norm=matplotlib[:colors][:LogNorm](vmin=minimum(0.1), vmax=maximum(11)), cmap="PuBu_r")
xticks([])
yticks([])
colorbar(orientation="horizontal", fraction=0.4, label=L"\mathrm{Mach \ Number}")


subplot(133)
axis(:equal)
box(:off)
title("Turbulent Velocity")
pcolor(dst, norm=matplotlib[:colors][:LogNorm](vmin=minimum(1.1e6), vmax=maximum(6.1e6)), cmap="PuBu_r")
xticks([])
yticks([])
colorbar(orientation="horizontal", fraction=0.4, label=L"\mathrm{cm \ s^{-1}}")

suptitle("Projected of central slide of IT90_3")

savefig("/home/STUDENTI/matteo.angelinelli/Julia/abst.png")



#show()

#center=find(x-> (x == maximum(d.+dm)),(d.+dm))
#println(center)






