using LinearAlgebra
using SparseArrays
using Metrics
using Distributions
using Plots 
using SparseArrays
using DelimitedFiles
using CSV
using DataFrames
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
hyp                     = [2^2*ones(D), 1., 0.01]; 
XL                      = 20 # area is 20 x 20 x 0.1
X                       = hcat(rand(Uniform(-XL,XL),N,2),rand(Uniform(-.01,.01),N,1));

# data sampled from GP prior with curl-free kernel
Kcurl                   = covcurlfree(X,X,hyp);       
Lchol                   = cholesky(Kcurl+(hyp[3]+sqrt(eps(1.0)))*Matrix(I,size(Kcurl))).L;
df                      = Lchol*randn(3N);
f                       = reshape(df,3,N)';
y                       = f + sqrt(hyp[3])*randn(N,3)
yvec                    = [norm(f[i,:]) for i=1:size(f,1)];

#data1 = vcat(hcat("x","y","label"),hcat(X[:,1],X[:,2],yvec))
#writedlm("data_scatter1.csv", data1[1:1000,:], ',')

plt0 = scatter(X[:,1],X[:,2],marker_z=yvec,markerstrokewidth=0,legend=false,aspect_ratio=1)
savefig(plt0,"Figures/data.png")

# compute RMSE for different domain sizes and methods
DS                      = [20,40]; # domain sizes
M12                     = [10,20,40,80,100,200] # number of basis functions in first and second dimension

RMSE_bf                 = zeros(3,length(M12));
RMSE_dski               = zeros(3,length(M12));
RMSE_full               = zeros(3);
RMSE_down               = zeros(3,length(M12));
Oind                    = zeros(3,length(M12));
Ntrain                  = zeros(3);
Ntest                   = zeros(3);
MR                      = Int.(zeros(3,length(M12)));
N_ds                    = Int.(zeros(3,length(M12)));
for i = 1:2
#i = 1
    # getting data for the specified domain size
    ds      = DS[i];
    X_ds    = Any[];
    y_ds    = Any[];
    f_ds    = Any[];
    for n = 1:N
        if X[n,1] < ds/2 && X[n,1] > -ds/2 && X[n,2] < ds/2 && X[n,2] > -ds/2
            push!(X_ds,X[n,:])
            push!(y_ds,y[n,:])
            push!(f_ds,f[n,:])
        end
    end
    X_ds                    = Matrix(reduce(hcat,X_ds)');
    y_ds                    = Matrix(reduce(hcat,y_ds)');
    f_ds                    = Matrix(reduce(hcat,f_ds)');
    Xtrain                  = X_ds[1:Int(ceil(0.8*size(X_ds,1))),:];
    Xtest                   = X_ds[Int(ceil(0.8*size(X_ds,1)))+1:end,:];
    ytrain                  = y_ds[1:Int(ceil(0.8*size(X_ds,1))),:];
    ftest                   = f_ds[Int(ceil(0.8*size(X_ds,1)))+1:end,:];
    Ntrain[i]               = size(Xtrain,1)
    Ntest[i]                = size(Xtest,1)
### curl-free KISS inference
for j = 1:length(M12)
#j = 1
    dimsInducing                            = (M12[j],M12[j],5);
    Xinducing, Xinducingdim                 = createInducingGrid(Xtrain, Xtest, dimsInducing, 3);
    Xinducingdimnorm                        = covNorm.(Xinducingdim, Xinducingdim);
    Wtrain, dWtrain, Wtraindim, dWtraindim  = interpMatrixDerivative(Xtrain, Xinducingdim, 3);
    Wtest, dWtest, Wtestdim, dWtestdim      = interpMatrixDerivative(Xtest, Xinducingdim, 3);
                
    ℓ, σ_f, σ_n                             = sqrt.([hyp[1][1],hyp[2],hyp[3]])
    resCG                                   = 1e-8;
    rankPCD                                 = 5;
    nLanczos                                = 500;
    YMean                                   = [mean(ytrain[1,:]),mean(ytrain[2,:]) ,mean(ytrain[3,:])];
    ytrain                                  = hcat(ytrain[:,1], ytrain[:,2], ytrain[:,3])

    Kdimcurl, acurl, Rcurl, Rprimecurl,iter = trainCurlSKI(Xinducingdimnorm, dWtrain, ytrain'[:], rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    μcurl, stdcurl                          = testCurlSKI(dWtest, dWtestdim, Wtestdim, Kdimcurl, acurl, Rcurl, Rprimecurl)
    RMSE_dski[i,j]                          = mse([μcurl[1];μcurl[2];μcurl[3]],ftest[:])
    Oind[i,j]                               = (iter * (length(Xtrain) + M12[j]^2*5*(2*M12[j]+5)))
end

### full GP
    K                       = covcurlfree(Xtrain,Xtrain,hyp);  
    Ks                      = covcurlfree(Xtest,Xtrain,hyp);
    Kss                     = covcurlfree(Xtest,Xtest,hyp);         
    Lchol                   = cholesky(K+(hyp[3]+sqrt(eps(1.0)))*Matrix(I,length(ytrain),length(ytrain))).L;
    α                       = Lchol'\(Lchol\ytrain'[:]);
    mstar                   = Ks*α;
    RMSE_full[i]            = mse(mstar,ftest'[:])

### basis functions     
M                           = [M12[end],M12[end],10]; # different numbers of inducing inputs
L                           = [XL+2sqrt(hyp[1][1]),XL+2sqrt(hyp[1][1]),.1+2sqrt(hyp[1][1])];
Λ                           = eigenval(M,hyp,L);
for j = 1:length(M12)     
    MR[i,j]                 = Int(ceil(sqrt(Oind[i,j]/length(Xtrain))))
    m̃star                   = BF(Xtrain,ytrain,Xtest,hyp,M,MR[i,j])
    RMSE_bf[i,j]            = mse(m̃star,ftest)
end

### downsampling
if i == 2
    for j = 1:length(M12)
        N_ds[i,j]                   = Int.(ceil(Oind[i,j]^(1/3)/3))
        if N_ds[i,j] < Ntrain[i]
            X_ds                        = Xtrain[1:N_ds[i,j],:]
            y_ds                        = ytrain[1:N_ds[i,j],:]
        else
            X_ds                        = Xtrain
            y_ds                        = ytrain
        end
        Kcurl                       = covcurlfree(X_ds,X_ds,hyp);  
        Ks                          = covcurlfree(Xtest,X_ds,hyp);
        Kss                         = covcurlfree(Xtest,Xtest,hyp);         
        Lchol                       = cholesky(Kcurl+(hyp[3]+sqrt(eps(1.0)))*Matrix(I,length(y_ds),length(y_ds))).L;
        α                           = Lchol'\(Lchol\y_ds'[:]);
        mstar                       = Ks*α;
        RMSE_down[i,j]              = mse(mstar,ftest'[:])
    end
end
print("*****************\n")
end

    

plot(RMSE_bf[1,:],xlabel="Mbf / Mind",ylabel="RMSE",labels=false,color="red",linewidth=1)
plot!(RMSE_bf[2,:],xlabel="Mbf / Mind",ylabel="RMSE",labels=false,color="red",linewidth=2)
#plot!(RMSE_bf[3,:],xlabel="Mbf / Mind",ylabel="RMSE",labels=false,color="red",linewidth=3)
plot!(RMSE_dski[1,:],xlabel="",ylabel="RMSE",labels=false,color="blue",linewidth=1)
plot!(RMSE_dski[2,:],xlabel="",ylabel="RMSE",labels=false,color="blue",linewidth=2)
#plot!(RMSE_dski[3,:],xlabel="",ylabel="RMSE",labels=false,color="blue",linewidth=3)
plot!(RMSE_down[1,:],xlabel="",ylabel="RMSE",labels=false,color="green",linewidth=1)
plot!(RMSE_down[2,:],xlabel="",ylabel="RMSE",labels=false,color="green",linewidth=2)
#plot!(RMSE_down[3,:],xlabel="",ylabel="RMSE",labels=false,color="green",linewidth=3)

scatter!(RMSE_bf[1,:],xlabel="Mbf / Mind",ylabel="RMSE",labels="basis functions",color="red",linewidth=1)
scatter!(RMSE_bf[2,:],xlabel="Mbf / Mind",ylabel="RMSE",labels=false,color="red",linewidth=2)
#scatter!(RMSE_bf[3,:],xlabel="Mbf / Mind",ylabel="RMSE",labels=false,color="red",linewidth=3)
scatter!(RMSE_dski[1,:],xlabel="",ylabel="RMSE",labels="D-SKI",color="blue",linewidth=1)
scatter!(RMSE_dski[2,:],xlabel="",ylabel="RMSE",labels=false,color="blue",linewidth=2)
#scatter!(RMSE_dski[3,:],xlabel="",ylabel="RMSE",labels=false,color="blue",linewidth=3)
scatter!(RMSE_down[1,:],xlabel="",ylabel="RMSE",labels="downsampled",color="green",linewidth=1)
scatter!(RMSE_down[2,:],xlabel="",ylabel="RMSE",labels=false,color="green",linewidth=2)
#scatter!(RMSE_down[3,:],xlabel="",ylabel="RMSE",labels=false,color="green",linewidth=3)

plt = hline!([RMSE_full[1]],color=:black,linestyle=:dash,labels="full GP",legend=:best)
savefig(plt,"RMSEvsM.png")