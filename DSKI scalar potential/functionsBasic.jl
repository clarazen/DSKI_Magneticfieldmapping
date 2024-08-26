module functionsBasic

using Base
using LinearAlgebra
using StatsBase

export separate, correctZeroMean, Xouter
export shapeGrid, createGrid, createInducingGrid, createTestGrid

function separate(componentD, outputD)
    componentDdim = [Array{Float64}(undef,0) for _ = 1:outputD]
    [push!(componentDdim[d], componentD[outputD*i-(outputD-d)]) for i = 1:div(length(componentD),outputD), d = 1:outputD]
    return componentDdim
end

function correctZeroMean(μZM, Ymean)
    return [μZM[d] .+ Ymean[d] for d in eachindex(Ymean)]
end

# elementwise outer product of X with itself (x-x')(x-x')^T
function Xouter(Xvar)
    Xdist = [Xvar[i,:] - Xvar[j,:] for i in axes(Xvar,1), j in axes(Xvar,1)]
    return Xdist .* transpose.(Xdist)
end

function shapeGrid(Xrange)
    Xcartproduct = collect.(Iterators.product((x for x in Xrange)...))
    return transpose(reduce(hcat, reshape(Xcartproduct, length(Xcartproduct))))
end

# incorporate shapeGrid() into createGrid()
function createGrid(Xtrain::AbstractArray{Float64}, margin::Float64, dims::Tuple)
    if size(Xtrain,2) != length(dims)
        throw(error("Input and inducing/test point dimensionalities not equal; change dims."))
    end
    xMin = transpose(minimum(Xtrain, dims = 1))
    xMax = transpose(maximum(Xtrain, dims = 1))
    xRange = xMax - xMin
    xArray = range.(xMin - margin .* xRange, xMax + margin .* xRange, dims)
    gridMatrix = collect.(Iterators.product((x for x in xArray)...))
    gridVector = reshape(gridMatrix,length(gridMatrix))
    xMatrix = transpose(reduce(hcat, gridVector))
    return xMatrix, xArray
end

# add ::Tuple again for dims
function createInducingGrid(Xtrain::AbstractArray{Float64}, Xtest::AbstractArray{Float64}, dims, deg::Int)
    Xtrainmin = minimum(Xtrain, dims = 1)
    Xtrainmax = maximum(Xtrain, dims = 1)
    Xtestmin = minimum(Xtest, dims = 1)
    Xtestmax = maximum(Xtest, dims = 1)
    Xmin = transpose(minimum([Xtrainmin; Xtestmin], dims = 1))
    Xmax = transpose(maximum([Xtrainmax; Xtestmax], dims = 1))
    Xdistance = Xmax - Xmin
    if deg == 5
        uextra = 2
        umargin = 1.01 .* (uextra ./ (dims .- (2uextra + 1))) #small offset (1.01) for roundoff errors
    elseif deg == 3
        uextra = 1
        umargin = 0.01 .* (uextra ./ (dims .- (2uextra + 1))) #small offset (1.01) for roundoff errors
    else; throw(error("Not a valid degree of interpolation; choose deg = 3 or deg = 5."))
    end
    Xrange = range.(Xmin - Xdistance .* umargin, Xmax + Xdistance .* umargin, dims)
    Xmatrix = shapeGrid(Xrange)
    return Xmatrix, Xrange
end

function createTestGrid(xTrain::AbstractArray{Float64}, dims::Tuple)
    xMin = transpose(minimum(xTrain, dims = 1))
    xMax = transpose(maximum(xTrain, dims = 1))
    xRange = xMax - xMin
    uExtra = 3
    uMargin = (uExtra ./ (dims .- (2uExtra + 1)))
    xArray = range.(xMin - xRange .* uMargin, xMax + xRange .* uMargin, dims)
    gridMatrix = collect.(Iterators.product((x for x in xArray)...))
    gridVector = reshape(gridMatrix,length(gridMatrix))
    xMatrix = transpose(reduce(hcat, gridVector))
    return xMatrix, xArray
end

end