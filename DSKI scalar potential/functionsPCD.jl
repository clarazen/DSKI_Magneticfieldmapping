module functionsPCD

using LinearAlgebra

export PCD

function PCD(S, k)
    Q = []
    Perm = [diagm(ones(size(S,1)))]
    if k >= size(S,1); k = size(S,1)-1 end
    for i = 1:k
        S, q, perm = computeSchur(S, i)
        push!(Q,q)
        if i < k
            permhat = diagm(ones(length(q)))
            permhat[i+1:end,i+1:end] = perm
            push!(Perm, Perm[i] * permhat)
        end
    end
    return reduce(hcat, Perm .* Q)
end

function computeSchur(S, i)
    b = S[2:end, 1]
    L11 = sqrt(S[1,1])
    L21 = 1/L11 * b
    S = S[2:end,2:end] - 1/L11^2 * b *transpose(b)
    S, perm = permSchur(S)

    q = [zeros(i-1); L11; L21]
    
    return S, q, perm
end

function permMatrix(S, permIndex)
    perm = diagm(ones(size(S,1)))
    idata = perm[1,:]
    perm[1,:] = perm[permIndex,:]
    perm[permIndex,:] = idata
    return perm
end

function permSchur(S)
    ind = findmax(diag(S))[2]
    perm = permMatrix(S, ind)
    return perm*S*perm, perm
end

end