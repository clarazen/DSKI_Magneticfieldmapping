module functionsMetrics

using LinearAlgebra
using StatsBase

export computeRMSE, computeMSLL

function computeRMSE(Yestimate, Yreference)
    if typeof(Yreference) <: Vector
        Yreference = reduce(hcat, Yreference)
    end
    difference = reduce(hcat, Yestimate) .- Yreference
    dRMSE = [sum(difference[:,d].^2) ./ size(difference,1) for d in axes(difference,2)]
    return dRMSE
end

function computeMSLL(μ, Yref, std, σ_n)
    if typeof(Yref) <: Vector
        Yref = reduce(hcat, Yref)
    end
    if size(std,1) == size(Yref,1); std = [std, std, std] end
    MSLL = Array{Float64}(undef, length(μ))
    for d in eachindex(μ)
        var_n = std[d].^2 .+ σ_n^2
        term1 = 1/2 .* log.(2π .* var_n)
        term2 = (μ[d] .- Yref[:,d]).^2 ./ (2 .* var_n)
        MSLL[d] = mean(term1 + term2)
    end
    return MSLL
end

end