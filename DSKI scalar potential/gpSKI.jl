module gpSKI

using LinearAlgebra
using ..functionsBasic
using ..functionsKernels
using ..functionsInterpolation
using ..functionsMVM
using ..functionsSKI

export trainSharedSKI, testSharedSKI, gpSharedSKI, gpSharedSKI2
export trainCurlSKI, testCurlSKI, gpCurlSKI, gpCurlSKI2

function trainSharedSKI(Xnorm, Wtrain, Ytrain, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # inducing covariance matrix per dimension
    Kdim = normToSE.(Xnorm, ℓ, σ_f^(1/length(Xnorm)))
    # predictive mean precomputations
    α = solveSKI(Kdim, Wtrain, Ytrain, rankPCD, σ_n, resCG)
    a = [kronMVM(Kdim, transpose(Wtrain) * α) for α in α]
    # predictive variance precomputations
    R, Rprime = love(Kdim, Wtrain, nLanczos, σ_n)
    return Kdim, a, R, Rprime
end

function testSharedSKI(Wtest, Wtestdim, Kdim, a, R, Rprime)
    # predictive mean
    μ = [Wtest * a for a in a]
    #KttSKI = Wtest * krSKI(Kdim, Wtestdim)
    KttSKI = krSKI2(Kdim, Wtest, Wtestdim)
    # predictive variance
    std = sqrt.(KttSKI .- diag(transpose(R*transpose(Wtest))*(Rprime*transpose(Wtest))))
    return μ, std
end

function gpSharedSKI2(Xnorm, Wtrain, Ytrain, Wtest, Wtestdim, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # train shared KISS-GP approximation with SE kernel
    Kdim, a , R, Rprime = trainSharedSKI(Xnorm, Wtrain, Ytrain, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # compute KISS-GP approximation predictions at test points
    μ, std = testSharedSKI(Wtest, Wtestdim, Kdim, a, R, Rprime)
    return μ, std
end

function gpSharedSKI(Xinducingdim, Xtrain, Ytrain, Xtest, deg, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    Xinducingdimnorm = covNorm.(Xinducingdim, Xinducingdim);
    Wtrain, Wtraindim = interpMatrix(Xtrain, Xinducingdim, deg)
    Wtest, Wtestdim = interpMatrix(Xtest, Xinducingdim, deg)
    # train shared KISS-GP approximation with SE kernel
    Kdim, a , R, Rprime = trainSharedSKI(Xinducingdimnorm, Wtrain, Ytrain, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # compute KISS-GP approximation predictions at test points
    μ, std = testSharedSKI(Wtest, Wtestdim, Kdim, a, R, Rprime)
    return μ, std
end

function trainCurlSKI(Xnorm, dWtrain, ytrain, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # inducing covariance matrix per dimension
    Kdim = normToSE.(Xnorm, ℓ, σ_f^(1/length(Xnorm)))
    # predictive mean precomputations
    F = solveSKI(Kdim, dWtrain, ytrain, rankPCD, σ_n, resCG)
    α = F[1][1]
    iter = F[1][2]
    a = kronMVM(Kdim, transpose(dWtrain) * α)
    # predictive variance precomputations
    R, Rprime = love(Kdim, dWtrain, nLanczos, σ_n)
    return Kdim, a, R, Rprime,iter
end

function testCurlSKI(dWtest, dWtestdim, Wtestdim, Kdim, a, R, Rprime)
    # number of outputs
    n_output = div(size(dWtest,1), size.(Wtestdim,1)[1])
    # predictive mean
    μ = dWtest * a
    # predictive variance
    #dKttSKI = dWtest * krDSKI(Kdim, Wtestdim, dWtestdim) # use Khatri-Rao algebra
    dKttSKI = krDSKI2(Kdim, dWtest, Wtestdim, dWtestdim)
    std = sqrt.(dKttSKI .- diag(transpose(R*transpose(dWtest))*(Rprime*transpose(dWtest))))
    return separate(μ, n_output), separate(std, n_output)
end

function gpCurlSKI2(Xnorm, dWtrain, ytrain, dWtest, dWtestdim, Wtestdim, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # train curl-free KISS-GP approximation with SE kernel
    Kdim, a, R, Rprime = trainCurlSKI(Xnorm, dWtrain, ytrain, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # compute KISS-GP approximation predictions at test points
    μ, std = testCurlSKI(dWtest, dWtestdim, Wtestdim, Kdim, a, R, Rprime)
    return μ, std
end

function gpCurlSKI(Xinducingdim, Xtrain, ytrain, Xtest, deg, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    Xinducingdimnorm = covNorm.(Xinducingdim, Xinducingdim);
    Wtrain, dWtrain, Wtraindim, dWtraindim = interpMatrixDerivative(Xtrain, Xinducingdim, deg)
    Wtest, dWtest, Wtestdim, dWtestdim = interpMatrixDerivative(Xtest, Xinducingdim, deg)
    # train curl-free KISS-GP approximation with SE kernel
    Kdim, a, R, Rprime = trainCurlSKI(Xinducingdimnorm, dWtrain, ytrain, rankPCD, ℓ, σ_f, σ_n, resCG, nLanczos)
    # compute KISS-GP approximation predictions at test points
    μ, std = testCurlSKI(dWtest, dWtestdim, Wtestdim, Kdim, a, R, Rprime)
    return μ, std
end

end