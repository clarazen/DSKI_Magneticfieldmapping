module lmlSKI

using LinearAlgebra
using ..functionsBasic
using ..functionsKernels
using ..functionsMVM
using ..functionsSKI

export lmlSharedSKI, lmlCurlSKI, logdetSLQ, DlogdetSLQ

function lmlSharedSKI(hyp, Xnorm, Ytrain, Wtrain, rankPCD, resCG)
    Kdim = normToSE.(Xnorm, exp(hyp[1]/2), exp(hyp[2]/(2*length(Xnorm))))

    # computations for data fit term
    α = solveSKI(Kdim, Wtrain, Ytrain, rankPCD, exp(hyp[3]/2), resCG)
    dataFit = sum([transpose(Ytrain[:,d])*α[d] for d in eachindex(α)])
    #println(dataFit)
    
    # computations for complexity penalty term
    nz, nLanczos = 3, 15
    complexityPenalty = size(Ytrain,2)*logdetSLQ(Kdim, Wtrain, nz, nLanczos, exp(hyp[3]/2))
    
    # objective function
    return - 1/2*dataFit - 1/2*complexityPenalty - prod(size(Ytrain))/2*log(2π)
end

function lmlCurlSKI(hyp, Xnorm, ytrain, dWtrain, rankPCD, resCG)
    Kdim = normToSE.(Xnorm, exp(hyp[1]/2), exp(hyp[2]/(2*length(Xnorm))))

    # computations for data fit term
    α = solveSKI(Kdim, dWtrain, ytrain, rankPCD, exp(hyp[3]/2), resCG)[1]
    dataFit = transpose(ytrain) * α

    # computations for complexity penalty term
    nz, nLanczos = 2, 25
    complexityPenalty = logdetSLQ(Kdim, dWtrain, nz, nLanczos, exp(hyp[3]/2))

    # objective function
    return - 1/2*dataFit - 1/2*complexityPenalty - length(ytrain)/2*log(2π)
end

# logdet approximation via Stochastic Lanczos Quadrature (SLQ)
function logdetSLQ(Kdim, Wtrain, nz, nLanc, σ_n)
    Γ = 0
    for _ = 1:nz
        z = (rand(size(Wtrain,1)) .< 0.5) .* 2.0 .- 1.0
        qprobe = z ./ norm(z)
        T, _ = lanczos(Kdim, Wtrain, qprobe, nLanc, σ_n)
        eigT = eigen(Matrix(T))
        τ = eigT.vectors[1,:]
        Γ += sum(τ.^2 .* log.(eigT.values))
    end
    return size(Wtrain,1)/nz * Γ
end

# logdet approximation via Stochastic Lanczos Quadrature (SLQ)
function DlogdetSLQ(Kdim, F1, D2Kdim, Wtrain, nz, nLanc, σ_n)
    Γ = 0
    D1Γ = 0;
    D2Γ = 0;
    D3Γ = 0;
    for _ = 1:nz
        # Lanczos decomposition
        z = (rand(size(Wtrain,1)) .< 0.5) .* 2.0 .- 1.0
        qprobe = z ./ norm(z)
        T, Q = lanczos(Kdim, Wtrain, qprobe, nLanc, σ_n)

        # logdet approximation
        eigT = eigen(Matrix(T))
        τ = eigT.vectors[1,:]
        Γ += sum(τ.^2 .* log.(eigT.values))

        # logdet approximation of gradients
        ez = zeros(size(T,1)); ez[1] = norm(z)
        QTez = Q * (T \ ez)
        D1Γ += transpose(QTez) * Wtrain*circMVM(F1, transpose(Wtrain)*z)
        D2Γ += transpose(QTez) * Wtrain*kronMVM(D2Kdim, transpose(Wtrain)*z)
        D3Γ += transpose(QTez) * 2σ_n * z
    end
    # average logdets
    ld = size(Wtrain,1)/nz * Γ
    D1ld = 1/nz * D1Γ
    D2ld = 1/nz * D2Γ
    D3ld = 1/nz * D3Γ
    return ld, D1ld, D2ld, D3ld
end

end