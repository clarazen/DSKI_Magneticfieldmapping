includet("../DSKI scalar potential/functionsBasic.jl")
includet("../DSKI scalar potential/functionsKernels.jl")
# modules for KISS-GP:
includet("../DSKI scalar potential/functionsMVM.jl")
includet("../DSKI scalar potential/functionsPCD.jl")
includet("../DSKI scalar potential/functionsInterpolation.jl")
includet("../DSKI scalar potential/functionsSKI.jl")    # depends on the other modules
includet("../DSKI scalar potential/functionsExperiments.jl")
includet("../DSKI scalar potential/gpSKI.jl")

using Images
using LinearAlgebra
using Plots
using Random
using StatsBase
using DelimitedFiles
using .functionsBasic
using .functionsKernels
using .functionsInterpolation
using .functionsSKI
using .functionsExperiments
using .gpSKI

#%% experiment with real magnetic field data collected with the suit to show
#   the SKI framework for magnetic field modeling approximates full Gaussian
#   process regression well
#
#   the SKI framework uses preconditioned conjugate gradients and LOVE to
#   estimate the magnetic field for the shared and scalar potential models
#
#   the data set used consists of N = 21,931 measurements recorded by the 
#   pelvis sensor (can be changed) in the one of the halls of the 3mE building
#   of Delft University of Technology

# load data
X                   = readdlm("X1.csv")
y                   = readdlm("y1.csv")

# remove 2 outliers
X                   = hcat(X[:,1:522],X[:,524:9858],X[:,9860:end])
y                   = hcat(y[:,1:522],y[:,524:9858],y[:,9860:end])

# visualize data
plot(y[1,:],label="x")
plot!(y[2,:],label="y")
plot!(y[3,:],label="z")
plot!([norm(y[:,i]) for i=1:size(y,2)],label="magnitude",color="black")
plot!(legend=:outertop, legendcolumns=4)
title!("magnetometer data")

# Regression with SKI, LOVE and Kronecker algebra
# initialization
resCG               = 1e-8;
rankPCD             = 3;
nLanczos            = 500;
deg                 = 3;
griddims            = (400, 40, 4);

# generate grid of test point
xtest21 = range(-185, -175, 100);
xtest22 = range(17, 22, 50);
xtest23 = range(mean(X[3,:]), mean(X[3,:]),1);
xrange2 = [xtest21, xtest22, xtest23];
Xtest   = shapeGrid(xrange2);

sensors             = [1];
Ngrid               = -1;
Xgrid, Ygrid, ygrid, Ygridmean, YgridZM, ygridZM    = sampleData(X, y, Ngrid, sensors);
Xinducing, Xinducingdim                             = createInducingGrid(Xgrid, Xtest, griddims, deg);
Xinducingdimnorm                                    = covNorm.(Xinducingdim, Xinducingdim);

Wgrid, dWgrid, Wgriddim, dWgriddim                  = interpMatrixDerivative(Xgrid, Xinducingdim, deg);
Wtest, dWtest, Wtestdim, dWtestdim                  = interpMatrixDerivative(Xtest, Xinducingdim, deg);

ℓ, σ_f, σ_n                         =  [.5,.2,0.01]
Kdimcurl, acurl, Rcurl, Rprimecurl  = trainCurlSKI(Xinducingdimnorm, dWgrid, ygridZM, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos);
μcurlkissZM, stdcurlkiss            = testCurlSKI(dWtest, dWtestdim, Wtestdim, Kdimcurl, acurl, Rcurl, Rprimecurl);
μcurlkiss                           = correctZeroMean(μcurlkissZM, Ygridmean);

# plotting map of smaller area
img                 = load("Figures/wingzoom.png");
imgx                = range(-193, -123.5, size(img,1));
imgy                = range(8.5, 33.5, size(img,2));

# plots of data 
plt_data = plot(aspect_ratio = :equal, axis = ([], false),legend = false); #  
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
ymag     = [norm(y[:,i]) for i=1:size(y,2)]
clims    = (minimum(ymag), maximum(ymag));
scatter!(X[1,:], X[2,:], zcolor=ymag,clim=clims,markerstrokewidth=0,colorbar=true,markersize=1,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25));
# x
plt_data_x = plot(aspect_ratio = :equal, axis = ([], false),legend = false); #  
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
clims                               = (minimum(y[1,:]), maximum(y[1,:]));
scatter!(X[1,:], X[2,:], zcolor=y[1,:],clim=clims,markerstrokewidth=0,colorbar=true,markersize=1,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25));
# y
plt_data_y = plot(aspect_ratio = :equal, axis = ([], false),legend = false); #  
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
clims                               = (minimum(y[2,:]), maximum(y[2,:]));
scatter!(X[1,:], X[2,:], zcolor=y[2,:],clim=clims,markerstrokewidth=0,colorbar=true,markersize=1,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25));
# z
plt_data_z = plot(aspect_ratio = :equal, axis = ([], false),legend = false); #  
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
clims                               = (minimum(y[3,:]), maximum(y[3,:]));
scatter!(X[1,:], X[2,:], zcolor=y[3,:],clim=clims,markerstrokewidth=0,colorbar=true,markersize=1,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25));
# layout plot
plot(plt_data,plt_data_x,plt_data_y,plt_data_z,layout=(2,2))

# map of magnitude
μcurlmag2                           = norm.(eachrow(reduce(hcat, μcurlkiss)))
clims                               = (minimum(μcurlmag2), maximum(μcurlmag2));
sd                                  = scaleUncertainty(stdcurlkiss[3], 0.05)
plt_mapmag = plot(aspect_ratio = :equal, axis = ([], false), legend = false, colorbar = true);
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
scatter!(Xtest[:,1], Xtest[:,2], zcolor = μcurlmag2, clim = clims, markersize = 1.7, markerstrokewidth = 0, markeralpha = sd, markershape = :square,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25))
savefig(plt_mapmag, "Figures/mapzoomed_mag.png")

# map of x-component
clims = (minimum(μcurlkiss[1]), maximum(μcurlkiss[1]));
sd    = scaleUncertainty(stdcurlkiss[1], 0.05)
plt_mapx = plot(aspect_ratio = :equal, axis = ([], false), legend = false, colorbar = true);
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
scatter!(Xtest[:,1], Xtest[:,2], zcolor = μcurlkiss[1],clim=clims, markersize = 1.7, markerstrokewidth = 0, markeralpha = sd, markershape = :square,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25))
savefig(plt_mapx, "Figures/mapzoomed_x.png")

# map of y-component
clims = (minimum(μcurlkiss[2]), maximum(μcurlkiss[2]));
sd    = scaleUncertainty(stdcurlkiss[2], 0.05)
plt_mapy = plot(aspect_ratio = :equal, axis = ([], false), legend = false, colorbar = true);
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
scatter!(Xtest[:,1], Xtest[:,2], zcolor = μcurlkiss[2],clim=clims, markersize = 1.7, markerstrokewidth = 0, markeralpha = sd, markershape = :square,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25))
savefig(plt_mapy, "Figures/mapzoomed_y.png")

# map of z-component
clims = (minimum(μcurlkiss[3]), maximum(μcurlkiss[3]));
sd    = scaleUncertainty(stdcurlkiss[3], 0.05)
plt_mapz = plot(aspect_ratio = :equal, axis = ([], false), legend = false, colorbar = true);
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
scatter!(Xtest[:,1], Xtest[:,2], zcolor = μcurlkiss[3],clim=clims, markersize = 1.7, markerstrokewidth = 0, markeralpha = sd, markershape = :square,c=cgrad(:default, rev=true));
plot!(xlim = (-185, -175), ylim = (15, 25))
savefig(plt_mapz, "Figures/mapzoomed_z.png")

p = plot(plt_mapmag,plt_mapx,plt_mapy,plt_mapz,layout=(2,2))
savefig(p, "Figures/mapzoomedin.png")

