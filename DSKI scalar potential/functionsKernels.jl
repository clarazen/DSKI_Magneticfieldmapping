module functionsKernels

using Base
using LinearAlgebra

export covNorm, covSE, covDSE1, covDSE2, normToSE, normToDSE1, normToDSE2

function covNorm(x1, x2)
    return [norm(x1[i,:] - x2[j,:]) for i = eachindex(x1[:,1]), j = eachindex(x2[:,1])]
end

function covSE(x1::AbstractArray, x2::AbstractArray, ℓ::Float64, σ_f::Float64)
    tmp = [norm(x1[i,:] - x2[j,:]) for i = eachindex(x1[:,1]), j = eachindex(x2[:,1])]
    return σ_f^2 .* exp.(-1/2 * tmp.^2 ./ ℓ^2)
end

function covDSE1(x1::AbstractArray, x2::AbstractArray, ℓ::Float64, σ_f::Float64, D)
    tmp = [norm(x1[i,:] - x2[j,:]) for i = eachindex(x1[:,1]), j = eachindex(x2[:,1])]
    return σ_f^(2/D) .* exp.(-1/2 * tmp.^2 ./ ℓ^2) .* (tmp.^2 ./ ℓ^3)
end

function covDSE2(x1::AbstractArray, x2::AbstractArray, ℓ::Float64, σ_f::Float64, D)
    tmp = [norm(x1[i,:] - x2[j,:]) for i = eachindex(x1[:,1]), j = eachindex(x2[:,1])]
    return 2^(1/D) * σ_f^(1/D) .* exp.(-1/2 * tmp.^2 ./ ℓ^2)
end

function normToSE(xNorm, ℓ, σ_f)
    return σ_f^2 .* exp.(-1/2 * xNorm.^2 ./ ℓ^2)
end

function normToDSE1(xNorm, ℓ, σ_f, D)
    return σ_f^(2/D) .* exp.(-1/2 * xNorm.^2 ./ ℓ^2) .* (xNorm.^2 ./ ℓ^3)
end

function normToDSE2(xNorm, ℓ, σ_f, D)
    return 2^(1/D) * σ_f^(1/D) .* exp.(-1/2 * xNorm.^2 ./ ℓ^2)
end

end