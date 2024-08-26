module functionsSKI

using LinearAlgebra
using SparseArrays
using ..functionsBasic
using ..functionsMVM
using ..functionsPCD

export solveSKI, kroneckerPCG
export lanczos, love
export krSKI, krDSKI
export krSKI2, krDSKI2

function solveSKI(Kdim::AbstractArray, Wtrain, Ytrain, rankPCD, σ_n, resCG)
    Ldim = PCD.(Kdim, rankPCD)
    WL = Wtrain * kron((L for L in reverse(Ldim))...)
    
    return [kroneckerPCG(Kdim, Wtrain, Ytrain[:,d], WL, σ_n, resCG) for d in axes(Ytrain,2)]
end

# preconditioned conjugate gradient (PCG) with fast Kronecker MVMs
function kroneckerPCG(Kdim, WTrain, b::AbstractArray{Float64}, WL, σ_n, res::Float64)
    WLTWL = I + (1/σ_n^2) * transpose(WL)*WL
    Lm = cholesky(WLTWL).L

    x = zeros(length(b))
    r = b - (WTrain * kronMVM(Kdim, transpose(WTrain) * x) + σ_n^2 * x)
    z = (1/σ_n^2)*r - (1/σ_n^4) * WL * (transpose(Lm) \ (Lm\(transpose(WL)*r)))
    p = z
    rzold = transpose(r)*z

    iterations = 0
    for i = 1:length(b)
        Ap = WTrain * kronMVM(Kdim, transpose(WTrain) * p) + σ_n^2 * p
        α = rzold / (transpose(p) * Ap)
        x = x + α * p
        r = r - α * Ap
        znew = (1/σ_n^2)*r - (1/σ_n^4) * WL * (transpose(Lm) \ (Lm\(transpose(WL)*r)))
        rznew = transpose(r) * znew
        if sqrt(rznew) < res
            break
        end
        p = znew + (rznew / rzold) * p
        rzold = rznew
        iterations = i
    end
    println("Used $iterations conjugate gradient iterations.")
    return x, iterations
end

function lanczos(Kdim, WTrain::AbstractArray{Float64}, qProbe::AbstractArray{Float64}, nLanczos::Int64, σ_n)
    # Lanczos tridiagonalization is not valid if ran for more iterations than the number of training points,
    # so introduce threshold if more iterations are specified
    if nLanczos > size(WTrain,1); nLanczos = size(WTrain,1) end
    # initialization
    α = Float64[]
    β = [1.0]
    r = qProbe/norm(qProbe)
    Q = zeros(length(qProbe), nLanczos)

    # Lanczos tridiagonalization
    for i = 1:nLanczos
        q = r/β[i]
        if i > 1 # perform full reorthogonalization once
            q = q - Q*transpose(Q) * q
            q = q/norm(q)
        end
        Q[:,i] = q
        Aq = WTrain * kronMVM(Kdim, transpose(WTrain) * q) + σ_n^2*q
        push!(α, transpose(q) * Aq)
        if i == 1
            r = Aq - α[i]*q
            push!(β,norm(r))
        elseif i < nLanczos
            r = Aq - α[i]*q - β[i]*Q[:,i-1]
            push!(β,norm(r))
        end
    end
    T = Diagonal(α) + sparse(diagm(-1 => β[2:length(β)])) + sparse(diagm(1 => β[2:length(β)]))
    return T, Q
end

function love(Kdim, Wtrain, nLanczos, σ_n)
    # prevent more iterations than training points (invalid tridiagonalization)
    if nLanczos > size(Wtrain,1); nLanczos = size(Wtrain,1) end
    # Lanczos tridiagonalization with probe vector qprobe
    qprobe = Wtrain * kronMVM(Kdim, ones(size(Wtrain,2)))
    T, Q = lanczos(Kdim, Wtrain, qprobe, nLanczos, σ_n)

    # compute R and Rprime according to LOVE paper
    # use \ as T is sparse symmetric tridiagonal (fast computation)
    R = transpose(kronMMM(Kdim, transpose(Wtrain) * Q))
    Rprime = T \ R
    return R, Rprime
end

function krSKIDraft(Kdim, Wdim)
    KWdim = Kdim .* transpose.(reduce.(vcat, Wdim))
    KW = Array{Float64}(undef, (prod(size.(KWdim,1)), size.(KWdim,2)[1]))
    for i in axes(KW,2)
        KW[:,i] = kron((kw[:,i] for kw in reverse(KWdim))...)
    end
    return KW
end

function krDSKIDraft(Kdim, Wdim, dWdim)
    KWdim = Kdim .* transpose.(reduce.(vcat, Wdim))
    dKWdim = Kdim .* transpose.(reduce.(vcat, dWdim))
    dKW = Array{Float64}(undef, (prod(size.(KWdim,1)), sum(size.(dKWdim,2))))
    for i = 1:size(KWdim[1],2)
        for d in eachindex(KWdim)
            kw = undef
            if d == 1
                kw = dKWdim[1][:,i]
            else
                kw = KWdim[1][:,i]
            end
            for j in eachindex(KWdim)
                if j == 1
                    # do nothing
                elseif j == d
                    kw = kron(dKWdim[j][:,i], kw)
                else
                    kw = kron(KWdim[j][:,i], kw)
                end
            end
            dKW[:,length(Kdim)*i-length(Kdim)+d] = kw
        end
    end
    return dKW
end

function krSKI(Kdim, Wdim)
    KWdim = Kdim .* transpose.(Wdim)
    KW = Array{Float64}(undef, (prod(size.(KWdim,1)), size.(KWdim,2)[1]))
    for i in axes(KW,2)
        KW[:,i] = kron((kw[:,i] for kw in reverse(KWdim))...)
    end
    return KW
end

function krDSKI(Kdim, Wdim, dWdim)
    KWdim = Kdim .* transpose.(Wdim)
    dKWdim = Kdim .* transpose.(dWdim)
    dKW = Array{Float64}(undef, (prod(size.(KWdim,1)), sum(size.(dKWdim,2))))
    for i = 1:size(KWdim[1],2)
        for d in eachindex(KWdim)
            kw = undef
            if d == 1
                kw = dKWdim[1][:,i]
            else
                kw = KWdim[1][:,i]
            end
            for j in eachindex(KWdim)
                if j == 1
                    # do nothing
                elseif j == d
                    kw = kron(dKWdim[j][:,i], kw)
                else
                    kw = kron(KWdim[j][:,i], kw)
                end
            end
            dKW[:,length(Kdim)*i-length(Kdim)+d] = kw
        end
    end
    return dKW
end

function krSKI2(Kdim, W, Wdim)
    KWdim = Kdim .* transpose.(Wdim)
    WKW = Array{Float64}(undef, size(W,1))
    for i in axes(W,1)
        WKW[i] = transpose(W[i,:]) * kron((kw[:,i] for kw in reverse(KWdim))...)
    end
    return WKW
end

function krDSKI2(Kdim, dW, Wdim, dWdim)
    KWdim = Kdim .* transpose.(Wdim)
    dKWdim = Kdim .* transpose.(dWdim)
    dWKW = Array{Float64}(undef, sum(size.(dKWdim,2)))
    for i = 1:size(KWdim[1],2)
        for d in eachindex(KWdim)
            kw = undef
            if d == 1
                kw = dKWdim[1][:,i]
            else
                kw = KWdim[1][:,i]
            end
            for j in eachindex(KWdim)
                if j == 1
                    # do nothing
                elseif j == d
                    kw = kron(dKWdim[j][:,i], kw)
                else
                    kw = kron(KWdim[j][:,i], kw)
                end
            end
            dWKW[length(Kdim)*i-length(Kdim)+d] = transpose(dW[(i-1)*length(Kdim)+d,:]) * kw
        end
    end
    return dWKW
end

end