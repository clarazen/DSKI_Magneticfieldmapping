module basisfunctions
using SparseArrays
using LinearAlgebra

    export eigenval,leadingeigenval,leadingeigenfunc_dxyz,leadingeigenfunc,BF

    function eigenval(M::Vector,hyp,L::Vector)
        ℓ   = hyp[1][1]
        σ_f = hyp[2]
        ev  = zeros(M...)
        for j₁ = 1:M[1]
            for j₂ = 1:M[2]
                for j₃ = 1:M[3]
                    λ            = π^2/4 * ( j₁^2/L[1]^2 + j₂^2/L[2]^2 + j₃^2/L[3]^2 )
                    ev[j₁,j₂,j₃] = σ_f*sqrt(2π*ℓ)^3 .* exp.(- ℓ/2 * λ ) 
                end
            end
        end
        return ev
    end

    function leadingeigenval(Λ::Array{Float64,3},R::Int)

        p   = sortperm(Λ[:],rev=true)[1:R] # first R values of sorting permutation
        ΛR  = Λ[p] # first R eigenvalues
        ind = Float64.(Matrix(reshape(reinterpret(Int,CartesianIndices((1:size(Λ,1),1:size(Λ,2),1:size(Λ,3)))[p]),3,R)))

        return ΛR,ind
    end

    function leadingeigenval(M,Λ::SparseMatrixCSC,R::Int)

        p   = sortperm(diag(Λ),rev=true)[1:R] # first R values of sorting permutation
        ΛR  = Λ[p,p] # first R eigenvalues
        ind = Float64.(Matrix(reshape(reinterpret(Int,CartesianIndices((1:M[1],1:M[2],1:M[3]))[p]),3,R)))

        return ΛR,ind
    end



    function eigenval(M::Vector,ℓ::Vector,σ_f::Float64,L::Vector)
        D  = 3
        Λ  = 1;
        for d = D:-1:1
            w    = collect(1:M[d])';
            tmp  = σ_f^(1/D)*sqrt(2π*ℓ[d]) .* exp.(- ℓ[d]/2 .* ((π.*w')./(2L[d])).^2 )
            Λ    = kron(Λ,spdiagm(tmp))
        end
        
        return Λ
    end

    function leadingeigenfunc_dxyz(X::Matrix{Float64},L::Vector,ind::Matrix)
        D     = size(X,2)
        dΦdx  = Vector{Matrix}(undef,D);
        dΦdy  = Vector{Matrix}(undef,D);
        dΦdz  = Vector{Matrix}(undef,D);

        dΦdx[1] = (π*ind[1,:]'/(2*L[1]*sqrt(L[1]))) .*cospi.(  ((X[:,1].+L[1])./2L[1]).*ind[1,:]');
        dΦdx[2] = (1/sqrt(L[2])) .*sinpi.(  ((X[:,2].+L[2])./2L[2]).*ind[2,:]');
        dΦdx[3] = (1/sqrt(L[3])) .*sinpi.(  ((X[:,3].+L[3])./2L[3]).*ind[3,:]');
        dΦdx_   = dΦdx[1] .* dΦdx[2] .* dΦdx[3]

        dΦdy[1] = (1/sqrt(L[1])) .*sinpi.(  ((X[:,1].+L[1])./2L[1]).*ind[1,:]');
        dΦdy[2] = (π*ind[2,:]'/(2*L[2]*sqrt(L[2]))) .*cospi.(  ((X[:,2].+L[2])./2L[2]).*ind[2,:]');
        dΦdy[3] = (1/sqrt(L[3])) .*sinpi.(  ((X[:,3].+L[3])./2L[3]).*ind[3,:]');
        dΦdy_   = dΦdy[1] .* dΦdy[2] .* dΦdy[3]

        dΦdz[1] = (1/sqrt(L[1])) .*sinpi.(  ((X[:,1].+L[1])./2L[1]).*ind[1,:]');
        dΦdz[2] = (1/sqrt(L[2])) .*sinpi.(  ((X[:,2].+L[2])./2L[2]).*ind[2,:]');
        dΦdz[3] = (π*ind[3,:]'/(2*L[3]*sqrt(L[3]))) .*cospi.(  ((X[:,3].+L[3])./2L[3]).*ind[3,:]');
        dΦdz_   = dΦdz[1] .* dΦdz[2] .* dΦdz[3]
        
        return dΦdx_,dΦdy_,dΦdz_
    end

    function leadingeigenfunc(X::Matrix{Float64},L::Vector,ind::Matrix)

        Φ    = Vector{Matrix}(undef,D);
        Φ[1] = (π*ind[1,:]'/(2*L[1]*sqrt(L[1]))) .*cospi.(  ((X[:,1].+L[1])./2L[1]).*ind[1,:]');
        Φ[2] = (π*ind[2,:]'/(2*L[2]*sqrt(L[2]))) .*cospi.(  ((X[:,2].+L[2])./2L[2]).*ind[2,:]');
        Φ[3] = (π*ind[3,:]'/(2*L[3]*sqrt(L[3]))) .*cospi.(  ((X[:,3].+L[3])./2L[3]).*ind[3,:]');
        
        return Φ[1] .* Φ[2] .* Φ[3]
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

    function BF(X,y,Xtest,hyp,M,MR)
        # X and Xstar need to be centered around 0 in interval [-L, L]
        boundsMin_Xtest         = [minimum(Xtest[:,1]), minimum(Xtest[:,2]), minimum(Xtest[:,3])];
        boundsMax_Xtest         = [maximum(Xtest[:,1]), maximum(Xtest[:,2]), maximum(Xtest[:,3])];
        boundsMin_X             = [minimum(X[:,1]), minimum(X[:,2]), minimum(X[:,3])];
        boundsMax_X             = [maximum(X[:,1]), maximum(X[:,2]), maximum(X[:,3])];

        boundsMin               = zeros(size(X,2))
        boundsMax               = zeros(size(X,2))
        for d = 1:size(X,2)
            if boundsMin_Xtest[d] < boundsMin_X[d]
                boundsMin[d] = boundsMin_Xtest[d]
            else
                boundsMin[d] = boundsMin_X[d]
            end
            if boundsMax_Xtest[d] > boundsMax_X[d]
                boundsMax[d] = boundsMax_Xtest[d]
            else
                boundsMax[d] = boundsMax_X[d]
            end
        end

        L = zeros(size(X,2))
        for d = 1:size(X,2)
            if abs(boundsMax[d]) > abs(boundsMin[d])
                L[d] = abs(boundsMax[d]) + 2*sqrt(hyp[1][d]);
            else
                L[d] = abs(boundsMin[d]) + 2*sqrt(hyp[1][d]);
            end
        end

        #L                       = ((boundsMax.-boundsMin)./2)  + 2*sqrt.(hyp[1]);
        Λ                       = eigenval(M,hyp,L)
        ΛR,ind                  = leadingeigenval(Λ,MR);
        if maximum(ind[3,:])<2
            print("Warning: Choose more basis functions. \n")
        end

        dΦdx,dΦdy,dΦdz          = leadingeigenfunc_dxyz(X,L,ind);
        dΦsdx,dΦsdy,dΦsdz       = leadingeigenfunc_dxyz(Xtest,L,ind);
        invΛ                    = Diagonal(1 ./ ΛR)
        ∇Φs                     = [dΦsdx; dΦsdy; dΦsdz]
        ∇Φ                      = [dΦdx; dΦdy; dΦdz]
        return reshape(∇Φs * ((∇Φ'*∇Φ + hyp[3]*invΛ)\(∇Φ'*y[:])),size(Xtest))
    end

end