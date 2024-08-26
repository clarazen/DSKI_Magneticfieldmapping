module lmlFull

using LinearAlgebra
using ..functionsBasic
using ..functionsKernels

export lmlCurlFull

function lmlCurlFull(hyp, Xtrain, ytrain, Xnorm, Xtrainouter)
    K_ff    = normToSE(Xnorm, exp(hyp[1]/2), exp(hyp[2]/2))
    Icurl   = fill(diagm(fill(exp(hyp[1]), size(Xtrain,2))), size(Xtrainouter))
    M       = (Icurl .- Xtrainouter) ./ exp(2*hyp[1])
    K       = hvcat(size(K_ff,2), (K_ff .* M)...) + exp(hyp[3])*I
    L       = cholesky(K).L
    α       = transpose(L)\(L\ytrain)
    
    return -1/2*transpose(ytrain)*α - sum(log.(diag(L))) - length(ytrain)/2*log(2π)
end

#=
function lmlCurlFull(hyp, Xtrain, ytrain, Xnorm, Xtrainouter)
    ℓ²    = exp(hyp[1])
    σ_f²  = exp(hyp[2])
    σ_n²  = exp(hyp[2])

    K_ff  = normToSE(Xnorm, sqrt(ℓ²), sqrt(σ_f²))

    Icurl = fill(diagm(fill(exp(hyp[1]), size(Xtrain,2))), size(Xtrainouter))
    M     = (Icurl .- Xtrainouter) ./ exp(2*hyp[1])
    K     = hvcat(size(K_ff,2), (K_ff .* M)...) + exp(hyp[3])*I
    L     = cholesky(K).L
    α     = L'\(L\ytrain)
    return -1/2* (transpose(ytrain)*α + sum(log.(diag(L))) + length(ytrain)*log(2π))
    #return -1/2* (transpose(ytrain)*α + log(det(K)) + length(ytrain)*log(2π))
end
=#
end