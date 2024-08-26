includet("../DSKI scalar potential/functionsBasic.jl")
includet("../DSKI scalar potential/functionsKernels.jl")
includet("../DSKI scalar potential/functionsMVM.jl")
includet("../DSKI scalar potential/functionsPCD.jl")
includet("../DSKI scalar potential/functionsInterpolation.jl")
includet("../DSKI scalar potential/functionsSKI.jl")    # depends on the other modules
includet("../DSKI scalar potential/functionsExperiments.jl")
includet("../DSKI scalar potential/gpSKI.jl")

using Images
using LinearAlgebra
using Optim
using Plots
using Random
using StatsBase
using DelimitedFiles
using ColorSchemes
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

plt_data = plot(y[1,:],label="x");
plot!(y[2,:],label="y");
plot!(y[3,:],label="z");
plot!([norm(y[:,i]) for i=1:size(y,2)],label="magnitude",color="black");
plot!(legend=:outertop, legendcolumns=4);
title!("magnetometer data");
plot(plt_data)

# Regression with SKI, LOVE and Kronecker algebra
# initialization
resCG               = 1e-8;
rankPCD             = 3;
nLanczos            = 500;
deg                 = 3;
griddims            = (400, 40, 4);

# generate grid of test point
xtest1              = range(-192, -124, 250);
xtest2              = range(16, 25, 30);
xtest3              = range(mean(X[3,:]), mean(X[3,:]),1);
xrange              = [xtest1, xtest2, xtest3];
Xtest               = shapeGrid(xrange);

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

# plotting
img                                 = load("Figures/wingzoom-modified.png");
imgx                                = range(-193, -123.5, size(img,1));
imgy                                = range(8.5, 33.5, size(img,2));
clims                               = (0.3, 1.25);
# generate data plot
plt_data = plot(aspect_ratio = :equal, axis = ([], false),legend = false); #  
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
scatter!(X[1,:], X[2,:], zcolor=[norm(y[:,i]) for i=1:size(y,2)],clim=clims,markerstrokewidth=0,colorbar=true,markersize=1,c=cgrad(:default, rev=true));

# generate map plot
sd    = scaleUncertainty(stdcurlkiss[3], 0.1);
plt_map = plot(aspect_ratio = :equal, axis = ([], false), legend = false, colorbar = false);
plot!(imgx, imgy, reverse(img, dims=1), yflip = false, dpi = 1000);
μcurlmag = norm.(eachrow(reduce(hcat, μcurlkiss)));
scatter!(Xtest[:,1], Xtest[:,2], zcolor = μcurlmag, clim = clims, markersize = 2, markerstrokewidth = 0, markeralpha = sd, markershape = :square,c=cgrad(:default, rev=true));
scatter!(X[1,:], X[2,:], markersize = .5, markerstrokewidth = 0,markercolor="red");
plot(plt_map)
savefig(plt_map, "Figures/map20000.png")

# plots
plot(plt_data,plt_map,layout=(2,1))
