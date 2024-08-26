module functionsMVM

using FFTW
using LinearAlgebra
using SparseArrays

export kronMVM, kronMMM
export circNorm, circDFT, circMVM

function unfold(tens::AbstractArray, mode::Int64, dims)
    if mode == 1
        tmp = permutedims(tens, (2, mode, 3))
        return transpose(reshape(tmp, (:, dims[mode])))
    elseif mode == 2
        return reshape(tens, (dims[mode], :))
    elseif mode == 3
        tmp = permutedims(tens, (2, 1, mode))
        return reshape(tmp, (dims[mode], :))
    end
end

function refold(vec::AbstractArray, mode::Int64, dims)
    if mode == 1
        tmp = reshape(transpose(vec), (dims[3], dims[2], dims[mode]))
        return permutedims(tmp, (2, mode, 3))
    elseif mode == 2 
        return reshape(vec, (dims[mode], dims[3], dims[1]))
    elseif mode == 3
        tmp = reshape(vec, (dims[mode], dims[2], dims[1]))
        return permutedims(tmp, (2, 1, mode))
    end
end

# slightly faster than kronMMM (identity matrix expansion approach)
function kronMVM(As::AbstractArray, v::AbstractArray)
    if length(As) == 7  # multidimensional index approach (only works for D = 3)
        As = reverse(As)
        dims = reshape(size.(As,1), length(As))
        vt = permutedims(reshape(v, Tuple(reverse(dims))), (2,1,3))
        for (i, A) in enumerate(As)
            vt = refold(A * unfold(vt, i, dims), i, dims)
        end
        Av = reshape(permutedims(vt, (2,1,3)), prod(dims))
    else                # Wilson's approach as described in the thesis
        dims = (size.(As,1))
        KU = v
        for (i, K) in enumerate(As)
            U = sparse(reshape(KU, dims[i], Int(prod(dims)/dims[i])))
            KU = transpose(K * U)
        end
        Av = vec(KU)
    end
    return Vector(Av)
end

# multiple dispatch of kronMVM() for use with composite
# Kronecker decomposable kernels
function kronMVM(K::Tuple, v::AbstractArray{Float64})
    return kronMVM(K[1],v) + kronMVM(K[2],v)
end

# identity matrix expansion approach
# allows the computation of A*X, where X is a matrix
function kronMMM(A::AbstractArray, X::AbstractArray{Float64})
    A = reverse(A)
    dims = size.(A,1)
    x = copy(X)
    N = length(A)
    nleft = prod(dims[1:N-1])
    nright = 1

    for i = N:-1:1
        base = 0
        jump = dims[i] * nright
        for k = 1:nleft
            for j = 1:nright
                index1 = base + j
                index2 = index1 + nright * (dims[i]-1)
                x[index1:nright:index2,:] = A[i] * x[index1:nright:index2,:]
            end
            base += jump
        end
        nleft /= dims[max(i-1, 1)]
        nright *= dims[i]
    end
    return x
end

# multiple dispatch of kronMMM() for use with composite
# Kronecker decomposable kernels
function kronMMM(K::Tuple, X::AbstractArray{Float64})
    return kronMMM(K[1],X) + kronMMM(K[2],X)
end

#%% Toeplitz/BTTB matrix fast MVMs
# only used for MVMs with the derivative covariance matrix w.r.t. ℓ
# for gradient information for hyperparameter optimization
function circNorm(xDim)
    ndim = length.(xDim)
    wdim = [x[end] - x[1] for x in xDim]
    n = [floor(n-1/2)+1 for n in ndim]
    xc = [[1:n[i];n[i]-2*ndim[i]+2:0] for i in eachindex(ndim)]
    hdim = wdim ./ (ndim .- 1)
    gridTensor = collect.(Iterators.product((x for x in xc)...))
    gridVector = reshape(gridTensor, length(gridTensor))
    circInd = transpose(reduce(hcat, gridVector))
    return [norm((circInd[i,:] .- 1.0) .* hdim) for i in axes(circInd,1)]
end

function circDFT(c, ndim)
    return fft(reshape(c, Tuple(2 .* ndim .- 1)))
end

function circMVM(F, B)
    ng = div.(size(F) .+ 1, 2)
    b = reshape(B, ng)
    bpad = zeros(size(F))
    if length(ng) == 2
        bpad[1:ng[1],1:ng[2]] = b
    elseif length(ng) == 3
        bpad[1:ng[1],1:ng[2],1:ng[3]] = b
    elseif length(ng) == 4
        bpad[1:ng[1],1:ng[2],1:ng[3],1:ng[4]] = b
    end
    b = fft(bpad)
    b = F .* b
    b = real.(ifft(b))
    for i in eachindex(ng)
        b = reshape(b, (prod(ng[1:i-1]), 2*ng[i] - 1, prod(2 .* ng[i+1:end] .- 1)))
        b = b[:,1:ng[i],:]
    end
    return b = reshape(b,:)
end

end