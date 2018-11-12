@everywhere using PyPlot
@everywhere using Devectorize
@everywhere using Optim
@everywhere using LaTeXStrings

@everywhere snap=["0023" "0043" "0063" "0083" "0103" "0113" "0133" "0153" "0173" "0193"]
s=1
col=["red","blue","green","gold","black","grey","magenta","brown","darkcyan","deeppink"]
while s<=10
info=readdlm("/home/STUDENTI/matteo.angelinelli/Julia/prova_90-3_gas_"snap[s]".txt",',', Float64)

#plot(info[1:end-1,1], info[1:end-1,2], color="blue", title="Density", xlabel="r [kpc]", ylabel=L"\mathrm{Dens [g \ cm^{-3}]}", xscale=:log10, yscale=:log10, grid=:off,legend=:false)

figure(s)
loglog(info[1:end-1,1], info[1:end-1,2])
xlabel(L"\frac{\mathrm{r}}{\mathrm{[r_{100}]}}")
ylabel(L"\mathrm{Dens [g \ cm^{-3}]}")
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_"snap[s]".png")
close(s)

figure(20)
loglog(info[1:end-1,1], info[1:end-1,2],color=col[s],label=snap[s])
xlabel(L"\frac{\mathrm{r}}{\mathrm{[r_{100}]}}")
ylabel(L"\mathrm{Dens [g \ cm^{-3}]}")
legend()
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_all.png")

s=s+1
end

close(20)

info2=readdlm("/home/STUDENTI/matteo.angelinelli/Julia/prova_mass_gas_90-3.txt",',', Float64)
figure(11)
for s in 1:10
scatter(info2[s,1], info2[s,4],color=col[s],label=snap[s])
end
xlabel("Snap")
ylabel(L"\mathrm{\frac{M}{[M_{o}]}}")
legend()
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_mass_snap.png")
close(20)

figure(21)
for s in 1:10
scatter(info2[s,1], info2[s,3],color=col[s],label=snap[s])
end
xlabel("Snap")
ylabel(L"\mathrm{\frac{r_{100}}{[Mpc]}}")
legend()
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_r100_snap.png")
close(21)

figure(22)
for s in 1:10
scatter(info2[s,1], info2[s,2],color=col[s],label=snap[s])
end
xlabel("Snap")
ylabel("z")
legend()
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_z_snap.png")
close(22)

figure(23)
for s in 1:10
scatter(info2[s,2], info2[s,3],color=col[s],label=snap[s])
end
xlabel("z")
ylabel(L"\mathrm{\frac{r_{100}}{[Mpc]}}")
legend()
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_z_r100.png")
close(23)

figure(24)
for s in 1:10
scatter(info2[s,2], info2[s,4],color=col[s],label=snap[s])
end
xlabel("z")
ylabel(L"\mathrm{\frac{M}{[M_{o}]}}")
legend()
grid()
savefig("/home/STUDENTI/matteo.angelinelli/Julia/prova_gas_z_mass.png")
close(24)


