module functionsInterpolation

using Base
using LinearAlgebra
using SparseArrays

export interpMatrix, interpMatrixDerivative, interpMatrixDim, interpMatrixDimDerivative

function cubicKernel(s::Float64)
    if 0 ≤ abs(s) < 1
        u = 1.5*abs(s)^3 - 2.5*abs(s)^2 + 1
    elseif 1 ≤ abs(s) < 2
        u = -0.5*abs(s)^3 + 2.5*abs(s)^2 - 4*abs(s) + 2
    else
        u = 0.0
    end
    return u
end

function cubicKernelDerivative(s::Float64)
    if 0 ≤ abs(s) < 1
        u = 4.5*s*abs(s) - 5*s
    elseif 1 ≤ abs(s) < 2
        u = -1.5*s*abs(s) + 5*s - 4*s/abs(s)
    else
        u = 0.0
    end
    return u
end

function quinticKernel(s::Float64)
    if 0 ≤ abs(s) < 1 # Coefficients:  -0.84375, 1.96875, 0, -2.125, 0, 1
        u = -0.84375*abs(s)^5 + 1.96875*abs(s)^4 - 2.125*abs(s)^2 + 1
    elseif 1 ≤ abs(s) < 2 # Coefficients: 0.203125, -1.3125, 2.65625, -0.875, -2.578125, 1.90625
        u = 0.203125*abs(s)^5 - 1.3125*abs(s)^4 + 2.65625*abs(s)^3 - 0.875*abs(s)^2 - 2.578125*abs(s) + 1.90625
    elseif 2 ≤ abs(s) < 3 # Coefficients: 0.046875, -0.65625, 3.65625, -10.125, 13.921875, -7.59375
        u = 0.046875*abs(s)^5 - 0.65625*abs(s)^4 + 3.65625*abs(s)^3 - 10.125*abs(s)^2 + 13.921875*abs(s) - 7.59375
    else
        u = 0.0
    end
    return u
end

function quinticKernelDerivative(s::Float64)
    if 0 ≤ abs(s) < 1 # Coefficients:  -4.21875, 7.875, 0, -4.25, 0
        u = -4.21875*s*abs(s)^3 + 7.875*s*abs(s)^2 - 4.25*s
    elseif 1 ≤ abs(s) < 2 # Coefficients: 1.015625, -5.25, 7.96875, -1.75, -2.578125
        u = 1.015625*s*abs(s)^3 - 5.25*s^3 + 7.96875*s*abs(s) - 1.75*s - 2.578125*s/abs(s)
    elseif 2 ≤ abs(s) < 3 # Coefficients: 0.234375, -2.625, 10.96875, -20.25, 13.921875
        u = 0.234375*s*abs(s)^3 - 2.625*s^3 + 10.96875*s*abs(s) - 20.25*s + 13.921875*s/abs(s)
    else
        u = 0.0
    end
    return u
end

function cubicWeights(point::Float64, xInterp::AbstractArray{Float64}, nInterp::Int64, h::Float64)
    k = findInterpIndex(point, xInterp)
    s = (point - xInterp[k])/h
    sdim = [s+1, s, s-1, s-2]
    u = cubicKernel.(sdim)
    # if-else with boundary conditions
    if k == 1
        j = [k k+1 k+2 k+3]
        u = [u[2]+3u[1] u[3]-3u[1] u[1]+u[4] 0.0]
    elseif k == nInterp-1
        j = [k-2 k-1 k k+1]
        u = [0.0 u[1]+u[4] u[2]-3u[4] u[3]+3u[4]]
    elseif k == nInterp
        j = [k-3 k-2 k-1 k]
        u = [0.0 0.0 0.0 u[2]]
    else
        j = [k-1 k k+1 k+2]
        u = [u[1] u[2] u[3] u[4]]
    end
    return j, u
end

function cubicWeightsDerivative(point::Float64, xInterp::AbstractArray{Float64}, nInterp::Int64, h::Float64)
    k = findInterpIndex(point, xInterp)
    s = (point - xInterp[k])/h
    sdim = [s+1, s, s-1, s-2]
    wi = spzeros(nInterp)
    u = (cubicKernelDerivative.(sdim)) ./ h
    # if-else with boundary conditions
    if k == 1
        j = [k k+1 k+2 k+3]
        u = [u[2]+3u[1] u[3]-3u[1] u[1]+u[4] 0.0]
    elseif k == nInterp-1
        j = [k-2 k-1 k k+1]
        u = [0.0 u[1]+u[4] u[2]-3u[4] u[3]+3u[4]]
    elseif k == nInterp
        j = [k-3 k-2 k-1 k]
        u = [0.0 0.0 0.0 u[2]]
    else
        j = [k-1 k k+1 k+2]
        u = [u[1] u[2] u[3] u[4]]
    end
    return j, u
end

function quinticWeights(point::Float64, xInterp::AbstractArray{Float64}, nInterp::Int64, h::Float64)
    k = findInterpIndex(point, xInterp)
    s = (point - xInterp[k])/h
    sdim = [s+2, s+1, s, s-1, s-2, s-3]
    wi = spzeros(nInterp)
    u = quinticKernel.(sdim)
    j = [k-2 k-1 k k+1 k+2 k+3]
    return j, u
end

function quinticWeightsDerivative(point::Float64, xInterp::AbstractArray{Float64}, nInterp::Int64, h::Float64)
    k = findInterpIndex(point, xInterp)
    s = (point - xInterp[k])/h
    sdim = [s+2, s+1, s, s-1, s-2, s-3]
    wi = spzeros(nInterp)
    u = (quinticKernelDerivative.(sdim)) ./ h
    j = [k-2 k-1 k k+1 k+2 k+3]
    return j, u
end

function findInterpIndex(point::Float64, xInterp::AbstractArray{Float64})
    xDifference = point .- collect(xInterp)
    minVal = Inf
    k = undef
    for i in eachindex(xDifference)
        if 0 <= xDifference[i] < minVal
            minVal = xDifference[i]
            k = i
        end
    end
    return k
end

function interpMatrixDim(Xvar, Xinducing, deg)
    h = (maximum(Xinducing) - minimum(Xinducing)) / (length(Xinducing) .- 1)
    Jdim = Array{Int64}(undef, length(Xvar), deg+1)
    Cdim = Array{Float64}(undef, length(Xvar), deg+1)
    if deg == 3
        for i in eachindex(Xvar)
            Jdim[i,:], Cdim[i,:] = cubicWeights(Xvar[i], Xinducing, length(Xinducing), h)
        end
    elseif deg == 5
        for i in eachindex(Xvar)
            Jdim[i,:], Cdim[i,:] = quinticWeights(Xvar[i], Xinducing, length(Xinducing), h)
        end
    end
    return Jdim, Cdim
end

function interpMatrixDimDerivative(Xvar, Xinducing, deg)
    h = (maximum(Xinducing) - minimum(Xinducing)) / (length(Xinducing) .- 1)
    dJdim = Array{Int64}(undef, length(Xvar), deg+1)
    dCdim = Array{Float64}(undef, length(Xvar), deg+1)
    if deg == 3
        for i in eachindex(Xvar)
            dJdim[i,:], dCdim[i,:] = cubicWeightsDerivative(Xvar[i], Xinducing, length(Xinducing), h)
        end
    elseif deg == 5
        for i in eachindex(Xvar)
            dJdim[i,:], dCdim[i,:] = quinticWeightsDerivative(Xvar[i], Xinducing, length(Xinducing), h)
        end
    end
    return dJdim, dCdim
end

function interpMatrix(Xvar, Xinducingdim, deg)
    JCdim = [interpMatrixDim(Xvar[:,d], Xinducingdim[d], deg) for d in eachindex(Xinducingdim)]
    Jdim = [JCdim[d][1] for d in eachindex(JCdim)]
    Cdim = [JCdim[d][2] for d in eachindex(JCdim)]
    Idim = [repeat(1:size(Xvar,1), 1, size(Cdim[d],2)) for d in eachindex(Cdim)]
    inducingdims = length.(Xinducingdim)
    J = undef
    C = undef
    for d in eachindex(Xinducingdim)
        Jd = Jdim[d]
        Cd = Cdim[d]
        if d == 1
            J = Jd
            C = Cd
        else
            pd = prod(inducingdims[1:d-1])
            J = repeat(J, 1, deg+1) + repeat((Jd .- 1) .* pd, inner = (1,size(J,2)))
            C = repeat(C, 1, deg+1) .* repeat(Cd, inner = (1,size(C,2)))
        end
    end
    Wdim = [sparse(vec(Idim[d]), vec(Jdim[d]), vec(Cdim[d]), size(Xvar,1), inducingdims[d]) for d in eachindex(Xinducingdim)]
    Isp = repeat(1:size(Xvar,1), 1, size(C,2))
    W = sparse(vec(Isp), vec(J), vec(C), size(Xvar,1), prod(inducingdims))
    return W, Wdim
end

function interpMatrixDerivative(Xvar, Xinducingdim, deg)
    inducingdims = length.(Xinducingdim)

    JCdim = [interpMatrixDim(Xvar[:,d], Xinducingdim[d], deg) for d in eachindex(Xinducingdim)]
    Jdim = [JCdim[d][1] for d in eachindex(JCdim)]
    Cdim = [JCdim[d][2] for d in eachindex(JCdim)]
    J = undef
    C = undef

    dJCdim = [interpMatrixDimDerivative(Xvar[:,d], Xinducingdim[d], deg) for d in eachindex(Xinducingdim)]
    dJdim = [dJCdim[d][1] for d in eachindex(dJCdim)]
    dCdim = [dJCdim[d][2] for d in eachindex(dJCdim)]
    dC = [Matrix{Float64}(undef, size(Xvar,1), deg+1) for d in eachindex(Xinducingdim)]
    
    for d in eachindex(Xinducingdim)
        Jd = Jdim[d]
        Cd = Cdim[d]
        dCd = dCdim[d] 
        if d == 1
            J = Jd
            C = Cd
            dC[d] = dCd
            dC[2:end] = [Cd for i = 1:length(dC[2:end])]
        else
            pd = prod(inducingdims[1:d-1])
            J = repeat(J, 1, deg+1) + repeat((Jd .- 1) .* pd, inner = (1,size(J,2)))
            C = repeat(C, 1, deg+1) .* repeat(Cd, inner = (1,size(C,2)))
            for j in eachindex(Xinducingdim)
                if j == d
                    dC[j] = repeat(dC[j], 1, deg+1) .* repeat(dCd, inner = (1,size(dC[j],2)))
                else
                    dC[j] = repeat(dC[j], 1, deg+1) .* repeat(Cd, inner = (1,size(dC[j],2)))
                end
            end
        end
    end
    dI = reduce(vcat, [repeat(1:size(Xvar,1), 1, size(C,2)) .* length(Xinducingdim) .- length(Xinducingdim) .+ d for d in eachindex(Xinducingdim)])
    dJ = repeat(J, length(Xinducingdim))
    dC = reduce(vcat, dC)
    
    Isp = repeat(1:size(Xvar,1), 1, size(C,2))
    W = sparse(vec(Isp), vec(J), vec(C), size(Xvar,1), prod(inducingdims))
    dW = sparse(vec(dI), vec(dJ), vec(dC), length(Xvar), prod(inducingdims))

    Idim = [repeat(1:size(Xvar,1), 1, size(Cdim[d],2)) for d in eachindex(Cdim)]
    Wdim = [sparse(vec(Idim[d]), vec(Jdim[d]), vec(Cdim[d]), size(Xvar,1), inducingdims[d]) for d in eachindex(Xinducingdim)]
    dWdim = [sparse(vec(Idim[d]), vec(dJdim[d]), vec(dCdim[d]), size(Xvar,1), inducingdims[d]) for d in eachindex(Xinducingdim)]
    return W, dW, Wdim, dWdim
end

end