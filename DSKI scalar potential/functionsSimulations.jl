module functionsSimulations

using Random
using StatsBase

export processData
export generate2DFunctionData, generate2DRandomData
export generate3DFunctionData, generate3DRandomData
export processMetric

function processData(X, Y, N, samplingFactor)
    Xtrain = X[1:samplingFactor:N,:]
    Ytrain = Y[1:samplingFactor:N,:]
    vec_Ytrain = Vector(vec(transpose(Ytrain)))
    Ymean = sum(Ytrain, dims = 1) ./ size(Ytrain,1)
    YtrainZM = reduce(hcat,[Ytrain[:,d] .- Ymean[d] for d in axes(Ytrain,2)])
    vec_YtrainZM = Vector(vec(transpose(YtrainZM)))
    sampleSize = size(Xtrain,1)
    println("Sampled $sampleSize of $N data points.")
    return Xtrain, Ytrain, vec_Ytrain, Ymean, YtrainZM, vec_YtrainZM
end

#%% Himmelblau's function with derivatives
# commented parts are the 2D Rosenbrock function
function himmelblau(x1, x2)
    return (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
    #return (1 - x1)^2 + 100 * (x2 - x1^2)^2
end
function d1himmelblau(x1, x2)
    return 4x1 * (x1^2 + x2 - 11) + 2*(x1 + x2^2 - 7)
    #return 2 * (200 * x1^3 - 200 * x1 * x2 + x1 -1)
end
function d2himmelblau(x1, x2)
    return 2 * (x1^2 + x2 - 11) + 4x2*(x1 + x2^2 - 7)
    #return 200 * (x2 - x1^2)
end

#%% compute (noised) versions of the derivatives of Himmelblau's function
# uncomment f, feval_sample and y_noised to include the actual function
# compute derivatives on a 2D grid
function generate2DFunctionData(points, x1min, x1max, x2min, x2max)
    x1 = (x1min:(x1max-x1min)/(points-1):x1max)
    x2 = (x2min:(x2max-x2min)/(points-1):x2max)
    #f = feval.(x1',x2)
    df1 = d1himmelblau.(x1',x2)
    df2 = d2himmelblau.(x1',x2)
    return x1, x2, df1, df2
end
# compute noised derivatives at random points
function generate2DRandomData(n, seed, x1min, x1max, x2min, x2max, mrg, noise)
    x1_sample = rand(seed, n-4) .* (x1max-x1min) .+ x1min
    x2_sample = shuffle(seed, x1_sample)
    push!(x1_sample, x1min+mrg, x1min+mrg, x1max-mrg, x2max-mrg)
    push!(x2_sample, x2min+mrg, x2max-mrg, x2min+mrg, x2max-mrg)
    #feval_sample = f.(x1_sample, x2_sample)
    df1_sample = d1himmelblau.(x1_sample, x2_sample)
    df2_sample = d2himmelblau.(x1_sample, x2_sample)
    #y_noised = feval_sample .+ noise * randn(seed, size(feval_sample))
    dy1_noised = df1_sample .+ noise * randn(seed, size(df1_sample))
    dy2_noised = df2_sample .+ noise * randn(seed, size(df2_sample))
    return x1_sample, x2_sample, dy1_noised, dy2_noised
end

#%% Curl-free vector field with derivatives
# potential function of conservative vector field including derivatives
# f(x) = x1 * x2^2 * x3^3
function feval(x)
    return x[1] * x[2]^2 * x[3]^3
    #return (1 - x1)^2 + 100 * (x2 - x1^2)^2
end
function df1eval(x)
    return x[2]^2 * x[3]^3
    #return 2 * (200 * x1^3 - 200 * x1 * x2 + x1 -1)
end
function df2eval(x)
    return 2 * x[1] * x[2] * x[3]^3
    #return 200 * (x2 - x1^2)
end
function df3eval(x)
    return 3 * x[1] * x[2]^2 * x[3]^2
end

#%% compute (noised) versions of the vector field
# uncomment f, feval_sample and y_noised to include the actual vector field
# compute vector field on a 3D grid
function generate3DFunctionData(X)
    #f = feval.(x1',x2)
    df1 = - df1eval.(eachrow(X))
    df2 = - df2eval.(eachrow(X))
    df3 = - df3eval.(eachrow(X))
    return df1, df2, df3
end
# compute noised versions of the vector field at random points
function generate3DRandomData(n, seed, Xmin, Xmax, noise)
    Xsample = reduce(hcat, [rand(seed, n) .* (Xmax[d]-Xmin[d]) .+ Xmin[d] for d in eachindex(Xmin)])
    #feval_sample = f.(x1_sample, x2_sample)
    df1_sample = - df1eval.(eachrow(Xsample))
    df2_sample = - df2eval.(eachrow(Xsample))
    df3_sample = - df3eval.(eachrow(Xsample))
    #y_noised = feval_sample .+ noise * randn(seed, size(feval_sample))
    dy1_noised = df1_sample .+ noise * randn(seed, size(df1_sample))
    dy2_noised = df2_sample .+ noise * randn(seed, size(df2_sample))
    dy3_noised = df3_sample .+ noise * randn(seed, size(df3_sample))
    return Xsample, reduce(hcat, [dy1_noised, dy2_noised, dy3_noised])
end

function processMetric(errorMetric)
    mMetric = reduce.(vcat, [mean.(errorMetric[deg], dims=1) for deg in eachindex(errorMetric)])
    sdMetric = reduce.(vcat, [std.(errorMetric[deg], dims=1) for deg in eachindex(errorMetric)])
    return mMetric, sdMetric
end

end