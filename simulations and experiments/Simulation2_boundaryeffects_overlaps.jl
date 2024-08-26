using Pkg
Pkg.activate("onlineSKI")
using LinearAlgebra
using SparseArrays
using Metrics
using Distributions
using Plots 

######### Marnix code
includet("../DSKI scalar potential/functionsBasic.jl")
using .functionsBasic
includet("../DSKI scalar potential/functionsKernels.jl")
using .functionsKernels
includet("../DSKI scalar potential/functionsInterpolation.jl")
using .functionsInterpolation
includet("../DSKI scalar potential/functionsMVM.jl")
includet("../DSKI scalar potential/functionsPCD.jl")
includet("../DSKI scalar potential/functionsSKI.jl") 
includet("../DSKI scalar potential/gpSKI.jl")
using .gpSKI
############

############ Clara's code
includet("../Basis functions/gensynthdata.jl")
using .synthdata
includet("../Basis functions/colectofbasisfunc_dxyz.jl")
using .basisfunctions
############

N                       = 6000;
D                       = 3;
hyp                     = [5^2*ones(D), 1., 0.01]; 
XL                      = 20 # area is 20 x 20 x 0.01
X                       = hcat(rand(Uniform(-XL,XL),N,2),0.01*randn(N));

# training and testing data sampled from GP prior with curl-free kernel
Kcurl                   = covcurlfree(X,X,hyp);       
Lchol                   = cholesky(Kcurl+(hyp[3]+sqrt(eps(1.0)))*Matrix(I,size(Kcurl))).L;
df                      = Lchol*randn(3N);
f                       = reshape(df,3,N)';
y                       = f + sqrt(hyp[3])*randn(N,3)
yvec                    = [norm(f[i,:]) for i=1:size(f,1)]
                        scatter(X[:,1],X[:,2],marker_z=yvec,markerstrokewidth=0,markersize=5)

Xstar                   = Matrix(hcat(gengriddata(100,2,[-XL,-XL],[XL,XL]),zeros(10000)));

# divide training and testing data into four regions
X_ll    = Any[]; # lower left
y_ll    = Any[];
f_ll    = Any[];
X_lr    = Any[]; # lower right
y_lr    = Any[];
f_lr    = Any[];
X_ul    = Any[]; # upper left
y_ul    = Any[];
f_ul    = Any[];
X_ur    = Any[]; # upper right
y_ur    = Any[];
f_ur    = Any[];

for n = 1:N
    if X[n,1] < 0  && X[n,2] < 0
        push!(X_ll,X[n,:])
        push!(y_ll,y[n,:])
        push!(f_ll,df[n,:])
    end
    if X[n,1] > 0 && X[n,2] < 0
        push!(X_lr,X[n,:])
        push!(y_lr,y[n,:])
        push!(f_lr,df[n,:])
    end
    if X[n,1] < 0  && X[n,2] > 0
        push!(X_ul,X[n,:])
        push!(y_ul,y[n,:])
        push!(f_ul,df[n,:])
    end
    if X[n,1] > 0  && X[n,2] > 0
        push!(X_ur,X[n,:])
        push!(y_ur,y[n,:])
        push!(f_ur,df[n,:])
    end
end
# downsample in some regions
X_ll                    = Matrix(reduce(hcat,X_ll)')
y_ll                    = Matrix(reduce(hcat,y_ll)')
f_ll                    = Matrix(reduce(hcat,f_ll)')
X_lr                    = Matrix(reduce(hcat,X_lr)')[1:3:end,:]
y_lr                    = Matrix(reduce(hcat,y_lr)')[1:3:end,:]
f_lr                    = Matrix(reduce(hcat,f_lr)')[1:3:end,:]
X_ur                    = Matrix(reduce(hcat,X_ur)')[1:20:end,:]
y_ur                    = Matrix(reduce(hcat,y_ur)')[1:20:end,:]
f_ur                    = Matrix(reduce(hcat,f_ur)')[1:20:end,:]
X_ul                    = Matrix(reduce(hcat,X_ul)')[1:50:end,:]
y_ul                    = Matrix(reduce(hcat,y_ul)')[1:50:end,:]
f_ul                    = Matrix(reduce(hcat,f_ul)')[1:50:end,:]

plt     = Vector{Plots.Plot{Plots.GRBackend}}(undef,5)
scatter(X_ll[:,1],X_ll[:,2],marker_z=[norm(y_ll[i,:]) for i=1:size(y_ll,1)],markerstrokewidth=0,markersize=1,legend=false)
scatter!(X_lr[:,1],X_lr[:,2],marker_z=[norm(y_lr[i,:]) for i=1:size(y_lr,1)],markerstrokewidth=0,markersize=1,legend=false)
scatter!(X_ur[:,1],X_ur[:,2],marker_z=[norm(y_ur[i,:]) for i=1:size(y_ur,1)],markerstrokewidth=0,markersize=1,legend=false)
scatter!(X_ul[:,1],X_ul[:,2],marker_z=[norm(y_ul[i,:]) for i=1:size(y_ul,1)], dpi=300,markerstrokewidth=0,markersize=1,legend=false,aspect_ratio=1.0, axis=([], false),border=:nothing)
plot!([-20; 20], [0; 0], lw=1, lc=:black, legend=false)
plot!([-20; 20], [20; 20], lw=1, lc=:black, legend=false)
plot!([-20; 20], [-20; -20], lw=1, lc=:black, legend=false)

plot!([0; 0], [-20; 20], lw=1, lc=:black, legend=false)
plot!([-20; -20], [-20; 20], lw=1, lc=:black, legend=false)
plt[1] = plot!([20; 20], [-20; 20], lw=1, lc=:black, legend=false)



#savefig(plt1,"simulations and experiments/Figures/data_regions.png")

X_downsampled           = vcat(X_ll,X_lr,X_ur,X_ul)
y_downsampled           = vcat(y_ll,y_lr,y_ur,y_ul)
f_downsampled           = vcat(f_ll,f_lr,f_ur,f_ul)
scatter(X_downsampled[:,1],X_downsampled[:,2],marker_z=[norm(y_downsampled[i,:]) for i=1:size(y_downsampled,1)],markerstrokewidth=0,legend=false,aspect_ratio=1.0, axis=([], false),border=:nothing)

# D-SKI
dimsInducing                            = (40,40,10);
Xinducing, Xinducingdim                 = createInducingGrid(X_downsampled, Xstar, dimsInducing, 3);
Xinducingdimnorm                        = covNorm.(Xinducingdim, Xinducingdim);
Wtrain, dWtrain, Wtraindim, dWtraindim  = interpMatrixDerivative(X_downsampled, Xinducingdim, 3);
Wtest, dWtest, Wtestdim, dWtestdim      = interpMatrixDerivative(Xstar, Xinducingdim, 3);
            
ℓ, σ_f, σ_n                             = sqrt.([hyp[1][1],hyp[2],hyp[3]])
resCG                                   = 1e-8;
rankPCD                                 = 5;
nLanczos                                = 500;
ytrain                                  = hcat(y_downsampled[:,1], y_downsampled[:,2], y_downsampled[:,3])

Kdimcurl, acurl, Rcurl, Rprimecurl,iter = trainCurlSKI(Xinducingdimnorm, dWtrain, ytrain'[:], rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
μcurl, stdcurl                          = testCurlSKI(dWtest, dWtestdim, Wtestdim, Kdimcurl, acurl, Rcurl, Rprimecurl)
mski                                    = hcat(μcurl[1],μcurl[2],μcurl[3])

plt[2]  = scatter(Xstar[:,1],Xstar[:,2],marker_z=[norm(mski[i,:]) for i=1:size(mski,1)], dpi=300,markerstrokewidth=0,legend=false,markersize=2, axis=([], false),border=:nothing,aspect_ratio=1.0)

overlap = [0,0.1,0.3]

#for i = 1:length(overlap)
i = 1
    # add measurement from neighboring regions when within α distance
    X_ll    = Any[]; # lower left
    y_ll    = Any[];
    f_ll    = Any[];
    X_lr    = Any[]; # lower right
    y_lr    = Any[];
    f_lr    = Any[];
    X_ul    = Any[]; # upper left
    y_ul    = Any[];
    f_ul    = Any[];
    X_ur    = Any[]; # upper right
    y_ur    = Any[];
    f_ur    = Any[];

    αℓ = sqrt(hyp[1][1])*overlap[i];
    for n = 1:size(X_downsampled,1)
        if X_downsampled[n,1] < αℓ  && X_downsampled[n,2] < αℓ
            push!(X_ll,X_downsampled[n,:])
            push!(y_ll,y_downsampled[n,:])
            push!(f_ll,f_downsampled[n,:])
        end
        if X_downsampled[n,1] > -αℓ && X_downsampled[n,2] < αℓ
            push!(X_lr,X_downsampled[n,:])
            push!(y_lr,y_downsampled[n,:])
            push!(f_lr,f_downsampled[n,:])
        end
        if X_downsampled[n,1] < αℓ  && X_downsampled[n,2] > -αℓ
            push!(X_ul,X_downsampled[n,:])
            push!(y_ul,y_downsampled[n,:])
            push!(f_ul,f_downsampled[n,:])
        end
        if X_downsampled[n,1] > -αℓ  && X_downsampled[n,2] > -αℓ
            push!(X_ur,X_downsampled[n,:])
            push!(y_ur,y_downsampled[n,:])
            push!(f_ur,f_downsampled[n,:])
        end
    end
    X_ll                    = Matrix(reduce(hcat,X_ll)')
    y_ll                    = Matrix(reduce(hcat,y_ll)')
    f_ll                    = Matrix(reduce(hcat,f_ll)')
    X_lr                    = Matrix(reduce(hcat,X_lr)')
    y_lr                    = Matrix(reduce(hcat,y_lr)')
    f_lr                    = Matrix(reduce(hcat,f_lr)')
    X_ur                    = Matrix(reduce(hcat,X_ur)')
    y_ur                    = Matrix(reduce(hcat,y_ur)')
    f_ur                    = Matrix(reduce(hcat,f_ur)')
    X_ul                    = Matrix(reduce(hcat,X_ul)')
    y_ul                    = Matrix(reduce(hcat,y_ul)')
    f_ul                    = Matrix(reduce(hcat,f_ul)')

    scatter(X_ll[:,1],X_ll[:,2],marker_z=[norm(y_ll[i,:])  for i=1:size(y_ll,1)],markerstrokewidth=0,legend=false)
    scatter!(X_lr[:,1],X_lr[:,2],marker_z=[norm(y_lr[i,:]) for i=1:size(y_lr,1)],markerstrokewidth=0,legend=false)
    scatter!(X_ur[:,1],X_ur[:,2],marker_z=[norm(y_ur[i,:]) for i=1:size(y_ur,1)],markerstrokewidth=0,legend=false)
    scatter!(X_ul[:,1],X_ul[:,2],marker_z=[norm(y_ul[i,:]) for i=1:size(y_ul,1)],markerstrokewidth=0,legend=false,aspect_ratio=1.0)

    # reconstruct map in four regions with training data that goes a bit over the region
    M                       = [40,40,10];
    MR                      = 1000;
    # lower left

    Xs_ll                   = Matrix(hcat(gengriddata(50,2,[-XL,-XL],[0,0]),zeros(2500)));
    #X_ll_min                = minimum(X_ll,dims=1);
    #X_ll_max                = maximum(X_ll,dims=1);
    #X_ll                    = X_ll .+ (X_ll_max - X_ll_min) ./ 2
    m_ll                    = reshape(BF(X_ll,y_ll,Xs_ll,hyp,M,MR),size(Xs_ll,1),3);
    Xs_ll                   = gengriddata(50,2,[-XL,-XL],[0,0])
    scatter(Xs_ll[:,1],Xs_ll[:,2],marker_z=[norm(m_ll[i,:]) for i=1:size(m_ll,1)],markerstrokewidth=0, dpi=300,legend=false,markersize=2, axis=([], false),border=:nothing)

    # lower right
    Xs_lr                   = Matrix(hcat(gengriddata(50,2,[-XL,0],[0,XL]),zeros(2500)));
    #X_lr_min                = minimum(X_lr,dims=1);
    #X_lr_max                = maximum(X_lr,dims=1);
    #X_lr                    = X_lr .+ (X_lr_max - X_lr_min) ./ 2
    m_lr                    = reshape(BF(X_lr,y_lr,Xs_lr,hyp,M,MR),size(Xs_lr,1),3);
    Xs_lr                   = gengriddata(50,2,[-XL,0],[0,XL]);
    scatter!(Xs_lr[:,1],Xs_lr[:,2],marker_z=[norm(m_lr[i,:]) for i=1:size(m_lr,1)],markerstrokewidth=0, dpi=300,legend=false,markersize=2, axis=([], false),border=:nothing)

    # upper left
    Xs_ul                   = Matrix(hcat(gengriddata(50,2,[0,-XL],[XL,0]),zeros(2500)));
    #X_ul_min                = minimum(X_ul,dims=1);
    #X_ul_max                = maximum(X_ul,dims=1);
    #X_ul                    = X_ul .+ (X_ul_max - X_ul_min) ./ 2
    m_ul                    = reshape(BF(X_ul,y_ul,Xs_ul,hyp,M,MR),size(Xs_ul,1),3);
    Xs_ul                   = gengriddata(50,2,[0,-XL],[XL,0]);    
    scatter!(Xs_ul[:,1],Xs_ul[:,2],marker_z=[norm(m_ul[i,:]) for i=1:size(m_ul,1)],markerstrokewidth=0, dpi=300,legend=false,markersize=2, axis=([], false),border=:nothing)

    # upper right
    Xs_ur                   = Matrix(hcat(gengriddata(50,2,[0,0],[XL,XL]),zeros(2500)));
    #X_ur_min                = minimum(X_ur,dims=1);
    #X_ur_max                = maximum(X_ur,dims=1);
    #X_ur                    = X_ur .+ (X_ur_max - X_ur_min) ./ 2
    m_ur  = reshape(BF(X_ur,y_ur,Xs_ur,hyp,M,MR),size(Xs_ur,1),3);
    Xs_ur = gengriddata(50,2,[0,0],[XL,XL]);
    plt[i+2] = scatter!(Xs_ur[:,1],Xs_ur[:,2],marker_z=[norm(m_ur[i,:]) for i=1:size(m_ur,1)], dpi=300,markerstrokewidth=0,legend=false,markersize=2,aspect_ratio=1.0, axis=([], false),border=:nothing)
#end
plot(plt[1], plt[2], plt[3], plt[4],plt[5], layout = (1,5))

        
        





