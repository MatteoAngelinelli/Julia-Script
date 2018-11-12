tic()
println("Packages")

@everywhere using PyPlot
@everywhere using DistributedArrays
@everywhere using HDF5   
@everywhere using Devectorize
@everywhere using Optim
@everywhere const ncell=320
@everywhere const kb=1.38*10.^(-16.)
@everywhere const mh=1.67*10.^(-24.)
@everywhere const mu=1.1
@everywhere const mue=0.59
@everywhere const mp=1.67e-24
@everywhere const vol=3*log10(20)+3*log10(3.086*10^21)
@everywhere const gamma=1.66667 
@everywhere const fac=gamma*kb/mp
@everywhere const mthr=1.3
@everywhere const msol=2*1e33


@everywhere root="/home/data/DATA/ISC/IT90_3/"  #folder
@everywhere snap=["0023" "0043" "0063" "0083" "0103" "0113" "0133" "0153" "0173" "0193"]
#@everywhere file1=string(root,"declust_z",snap[i])   #input files - (Density,Dark_Matter_Density,Temperature)
#@everywhere file2=string(root,"declust_v_z",snap[i])    #input files - (vx,vy,vz)
#@everywhere file3=string(root,"deDD",snap[i],".conv2")  #input files - (line4:convd, line5: convv)  

toc()
tic()
println("Functions")
@everywhere function readfile(root::String,snap::String)

 file1=string(root,"declust_z",snap)   #input files - (Density,Dark_Matter_Density,Temperature)
 file2=string(root,"declust_v_z",snap)    #input files - (vx,vy,vz)
 file3=string(root,"deDD",snap,".conv2")  #input files - (line4:convd, line5: convv)  

file=[file1 file2 file3]

return file

end


@everywhere function Smooth(data::Array{Float32,3},w::Int) #only for cubic data set and even lenght windows

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


@everywhere function smox(wmez::Int,ris1d::Array{Float32},low::Int,high::Int,ncell::Int)
   risx=Array{Float32}(ncell)
    @inbounds    for i in low:high
    @fastmath  risx[i]=mean(ris1d[i-wmez:i+wmez])
           end

   return risx
end

@everywhere function maps(field::Array{Float32,3},window::Int,lim1::Int,lim2::Int)   #function to make the maps of project fields 

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
    
 v=Array{Float32}(ncell,ncell,ncell)
        @inbounds for i in 1:ncell, j in 1:ncell, k in 1:ncell
        @fastmath  v[j,i,k]=sqrt(vx[i,j,k]^2+vy[i,j,k]^2+vz[i,j,k]^2)
        end
        return v
        end

@everywhere function vturb(vtot::Array{Float32,3},vsmooth::Array{Float32,3},ncell::Int)   #make turbulence velocity field
    vturb=Array{Float32}(ncell,ncell,ncell)
    vturb = vtot .- vsmooth
    id1=find(x-> (x <= 0),vturb) #using to avoid the zero value (if plot is set in log space)
    vturb[id1]=1.e-35   
   return vturb
end

@everywhere function conversion(data::Array{Float32,3},conv::Float64)
    data1=Array{Float32}(ncell,ncell,ncell)
    conv=convert(Float32,conv)
    data1 = data .* conv
  return data1
end

@everywhere function sumfield(data1::Array{Float32,3},data2::Array{Float32,3},ncell::Int)
    data=Array{Float32}(ncell,ncell,ncell)
    data .= data1 .+ data2
  return data
end

@everywhere function convind(center::Int,ncell::Int)

    cc=Array{Int}(3)
    center=convert(Float32,center)
    ncell=convert(Float32,ncell)
    
  @fastmath  xz=center/(ncell*ncell)+1
  @fastmath  xy=ncell*(xz-floor(xz))+1
  @fastmath  xx=ncell*(xy-floor(xy))
  @fastmath  xx=floor(xx)
  @fastmath  xy=floor(xy)
  @fastmath  xz=floor(xz)

    xx=convert(Int,xx)
    xy=convert(Int,xy)
    xz=convert(Int,xz)

    cc=[xx,xy,xz]

   return cc
end

@everywhere function gridspher(ncell::Int,posc1::Int,posc2::Int,posc3::Int)

   rgrid=Array{Float64}(ncell,ncell,ncell)
    
@inbounds  for i in 1:ncell, j in 1:ncell, k in 1:ncell  
 @fastmath   rgrid[i,j,k]=sqrt((i-posc1)^2+(j-posc2)^2+(k-posc3)^2)
           end
    return rgrid
    
end

@everywhere function mean_profile(data,rmax::Int,rgrid::Array{Float64,3})

    prof=Array{Float64}(rmax,2)
    prof[:,:]=0.
    tic()
 @inbounds   for i in eachindex(data) #1:ncell, j in 1:ncell, k in 1:ncell 
        ri=convert(Int,(floor(rgrid[i])))
    @fastmath     prof[ri,1]+=data[i]
   @fastmath     prof[ri,2]+=1.
    end
    toc()
  @inbounds  for i in 1:rmax
   @fastmath     prof[i,1]=prof[i,1]/prof[i,2]
    end
return prof[:,1]
    end 



@everywhere function rvir(r::Vector{Float64},d::Vector{Float64},dm::Vector{Float64},overd::Int, rhoc::Float64)
  
d .= d ./ rhoc                 
dm .= dm ./ rhoc                
nr=size(r)
vshell=Vector{Float64}(nr[1]+1)
radius=0
a=0
for i in 1:nr[1]+1
vshell[i]=4.*pi*i^2. 
end
dens_old=1e6
for i in 2:nr[1]
 dens1=sum(vshell[1:i] .* (d[1:i] .+ dm[1:i])) ./ (sum(vshell[1:i])) 
 a=i
 if dens1 < overd && dens_old > overd 
   @goto esc 
  dens_old=dens1
 end
end
  @label esc
   radius=a
  return radius
end

@everywhere function mass_tot(res::Float64,r200::Int,overd::Int,rhoc::Float64)

vol=3.* log10.(res)+3.*log10.(3.086*10.^21.)
m_tot=log10.((4./3.)*pi)+3.*log10.(r200)+vol+log10.(rhoc*overd)

return m_tot

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
println("Reading")
open("/home/STUDENTI/matteo.angelinelli/Julia/prova_mass_90-3.txt", "w") do f
s=1
while s<=10
println(snap[s])
file=readfile(root,snap[s])
d=h5read(file[1],"Density")     #read density and temperature file (array{Float32,3})
dm=h5read(file[1],"Dark_Matter_Density")
temp=h5read(file[1],"Temperature")
vx=h5read(file[2],"x-velocity")   #read velocity file (array{Float32,3})
vy=h5read(file[2],"y-velocity")
vz=h5read(file[2],"z-velocity")

info=readdlm(file[3], '\t', Float64, '\n') #read enzo's conversion factors {Float64}

info[4]=info[4]/((1+info[3])^3)

toc()
tic()
println("Field")
d=conversion(d,info[4])
dm=conversion(dm,info[4])
vx=conversion(vx,info[5])
vy=conversion(vy,info[5])
vz=conversion(vz,info[5])

#ds=maps(d,0,0,0)
#figure(1,figsize=(10,10))
#axis(:equal)
#box(:off)
#pcolor(ds, norm=matplotlib[:colors][:LogNorm](vmin=minimum(1e-32), vmax=maximum(ds)), cmap="jet")
#xticks([])
#yticks([])
#colorbar()
#show()
#savefig("/home/STUDENTI/matteo.angelinelli/Julia/map_"snap[s]".png")
#close(1)
v=vmod(vx,vy,vz,ncell)   #velocity field

@everywhere const dx=20.0   #resolution of simulations
@everywhere const win=10  #window used to make the smoothed field
res=Smooth(v,win)    #smoothed velocity field
turb=vturb(v,res,ncell)    #turbulence velocity field

toc()
tic()
println("Shock")

macx=shocks(d,temp,vx,vy,vz)

#ds=maps(macx,0,150,170)

#figure(1,figsize=(10,10))
#axis(:equal)
#box(:off)
#pcolor(ds, norm=matplotlib[:colors][:LogNorm](vmin=minimum(1), vmax=maximum(10)), cmap="jet")
#xticks([])
#yticks([])
#colorbar()
#show()
#savefig("/home/STUDENTI/matteo.angelinelli/Julia/map_mach.png")

toc()
tic()

println("Grid")
dens=sumfield(d,dm,ncell)
maxdens=maximum(dens)
center=find(x-> (x == maxdens),dens)
posc=convind(center[1],ncell) #position of barycenter in x-y-z

rgrid=gridspher(ncell,posc[1],posc[2],posc[3]) #spherical grid centered in posc[] 

rmax=maximum(rgrid)
rmax=floor(rmax)
rmax=convert(Int,rmax)

r=Array{Float64}(rmax)

@inbounds for i in 1:rmax
@fastmath    r[i]=dx*0.5+i*dx
          end

toc()
tic()
println("Profile")

pd=mean_profile(d,rmax,rgrid) 
pdm=mean_profile(dm,rmax,rgrid)
ptemp=mean_profile(temp,rmax,rgrid) 
#pvt=mean_profile(turb,rmax,rgrid)
#pvs=mean_profile(v,rmax,rgrid)
#pvv=mean_profile(res,rmax,rgrid)

overden=100

rhoc=(10.^(-29.063736))*((1.+info[3])^3)  #cosmological check 

R100=rvir(r,pd,pdm,overden,rhoc) #give the r of overdensity in cell's unit
M100=mass_tot(dx,R100,overden,rhoc) #give the mass within overdensity radius in log10 unit
M100_sol=10.^(M100)/msol
R100_Mpc=(R100*dx)/1e3

r .= r ./ (R100_Mpc*1e3)
pd .= pd .* rhoc
pdm .= pdm .* rhoc


open("/home/STUDENTI/matteo.angelinelli/Julia/prova_90-3_"snap[s]".txt", "w") do f1
    writedlm(f1, [r[1:rmax] pd[1:rmax] ptemp[1:rmax]],',')
end
     writedlm(f, [snap[s] info[3] R100_Mpc M100_sol],',')
s=s+1

end


end
toc()

