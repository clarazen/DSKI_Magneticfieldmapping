function colectofbasisfunc(M::Vector,X::Matrix{Float64},ℓ::Vector,σ_f::Float64,L::Vector,R::Int)
        # computes M^D basis functions and keeps the R leading ones
        N = size(X,1)
        D = size(X,2)
        Λ = 1;
        Φ = ones(N,1);
        for d = D:-1:1
            w    = collect(1:M[d])';
            Λ    = kron(Λ,spdiagm(σ_f^(1/D)*sqrt(2π*ℓ[d]) .* exp.(- ℓ[d]/2 .* ((π.*w')./(2L[d])).^2 )))
            Φ    = KhatriRao(Φ,(1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w),1);
        end

        p  = sortperm(diag(Λ),rev=true)[1:R] # first R values of sorting permutation
        ΛR = Λ[p,p] # first R eigenvalues
        ΦR = Φ[:,p]    

       return ΦR,ΛR
   end

   function KhatriRao(A::Matrix{Float64},B::Matrix{Float64},dims::Int64)
    if dims == 1 # row-wise
        C = zeros(size(A,1),size(A,2)*size(B,2));
        @inbounds @simd for i = 1:size(A,1)
            @views kron!(C[i,:],A[i,:],B[i,:])
        end
    elseif dims == 2 # column-wise
        C = zeros(size(A,1)*size(B,1),size(A,2));
        @inbounds @simd for i = 1:size(A,2)
            @views kron!(C[:,i],A[:,i],B[:,i])
        end
    end

    return C
end