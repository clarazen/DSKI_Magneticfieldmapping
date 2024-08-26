module functionsExperiments

using ColorSchemes
using Images
using LinearAlgebra
using StatsBase

export splitDataSet, shortenData, retrieveSensorData, sampleData, rotateData
export distancePath, findSplitIndex
export constructUndefErrors
export scaleUncertainty


function splitDataSet(X, Y, interval)
    if interval[1] > size(X,2); interval[1] = 1
        println("Interval start is larger than number of measurements, setting start at first measurement.") 
    elseif interval[1] < 1; interval[1] = 1 end
    if interval[end] > size(X,2); interval[end] = size(X,2)
        println("Interval end is larger than number of measurements, setting end at last measurement.")
    end
    Xtrainsplit = hcat(X[:, 1:interval[1]-1], X[:, interval[end]+1:end])
    Xvalsplit = X[:, interval[1]:interval[end]]
    Ytrainsplit = hcat(Y[:, 1:interval[1]-1], Y[:, interval[end]+1:end])
    Yvalsplit = Y[:, interval[1]:interval[end]]
    return Xtrainsplit, Xvalsplit, Ytrainsplit, Yvalsplit
end

function shortenData(X, Y, interval)
    if interval[1] > size(X,2); interval[1] = 1
        println("Interval start is larger than number of measurements, setting start at first measurement.") 
    elseif interval[1] < 1; interval[1] = 1 end
    if interval[end] > size(X,2); interval[end] = size(X,2)
        println("Interval end is larger than number of measurements, setting end at last measurement.")
    end
    Xshort = X[:, interval[1]:interval[end]]
    Yshort = Y[:, interval[1]:interval[end]]
    Nshort = size(Xshort,2)
    println("Shortened data set from ",size(X,2)," to $Nshort data points.")
    return Xshort, Yshort, Nshort
end

function retrieveSensorData(X, Y, sensors)
    # Nm: number of measurements per sensor, Ns: number of sensors
    Xsensor = Array{Float64}(undef, (3*length(sensors),size(X,2)))
    Ysensor = Array{Float64}(undef, (3*length(sensors),size(Y,2)))
    for i in eachindex(sensors)
        Xsensor[3i-2:3i,:] = X[3*sensors[i]-2:3*sensors[i],:];
        Ysensor[3i-2:3i,:] = Y[3*sensors[i]-2:3*sensors[i],:];
    end
    return Xsensor, Ysensor
end

function sampleData(X, Y, N, sensors)
    if N == -1; N = size(X,2) * length(sensors)
    elseif N > size(X,2) * length(sensors); N = size(X,2) * length(sensors)
        println("Specified number of measurements larger than number of measurents, taking all measurements.")
    end
    Nsensor = floor(Int, N/length(sensors))
    sampleIndex = round.(Int,LinRange(1,size(X,2),Nsensor))
    Xsample = X[:, sampleIndex]
    Ysample = Y[:, sampleIndex]
    Xstack = matrixToStack(Xsample, length(sensors))
    Ystack = matrixToStack(Ysample, length(sensors))
    Yvec = Vector(vec(transpose(Ystack)))
    Ymean = mean(Ystack, dims = 1)
    YstackZM = reduce(hcat, [Ystack[:,d] .- Ymean[d] for d in axes(Ystack,2)])
    YvecZM = Vector(vec(transpose(YstackZM)))
    Ntotal = Nsensor * length(sensors)
    println("Sampled $Ntotal of ",size(X,2) * length(sensors)," data points.")
    return Xstack, Ystack, Yvec, Ymean, YstackZM, YvecZM
end

function rotateData(X, Y, θ)
    R = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]
    Xrot = Array{Float64}(undef, size(X))
    Yrot = Array{Float64}(undef, size(Y))
    for i in axes(X,2)
        Xrot[:,i] = R*(X[:,i]-X[:,1]) + X[:,1]
        Yrot[:,i] = R*Y[:,i]
    end
    return Xrot, Yrot
end

function findSplitIndex(Xtrain)
    b = 0.0
    index = undef
    for i in axes(Xtrain,1)
        if i > 1; 
            c = norm(Xtrain[i,:] - Xtrain[i-1,:])
            if c > b; b = c
                index = i-1
            end
        end
    end
    return index
end

function distancePath(X, sensors)
    Xdist = Array{Float64}(undef, size(X,1), length(sensors))
    for i in axes(X,1)
        if i == 1; Xdist[i,:] = zeros(length(sensors))
        else; Xdist[i] = Xdist[i-1] + norm(X[i,:] .- X[i-1,:]) end
    end
    return Xdist
end

function constructUndefErrors(Ntrain, degree, Yval)
    RMSEindval = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    RMSEcurlval = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    RMSEindsample = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    RMSEcurlsample = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    MSLLindval = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    MSLLcurlval = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    MSLLindsample = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    MSLLcurlsample = [Array{Float64}(undef, (length(Ntrain), size(Yval,2))) for _ in eachindex(degree)];
    return RMSEindval, RMSEcurlval, RMSEindsample, RMSEcurlsample, MSLLindval, MSLLcurlval, MSLLindsample, MSLLcurlsample
end

# stackToMatrix() and matrixToStack() do the same thing
# the process gets reversed if ran twice
function stackToMatrix(X, Ns)
    # Ns: number of sensors
    Xr = reshape(transpose(X), (size(X,2), :, Ns))
    return reshape(permutedims(Xr, (1,3,2)), (Ns*size(X,2),:))
end

function matrixToStack(X, Ns)
    # Ns: number of sensors
    Xr = reshape(transpose(X), (size(X,2), :, Ns))
    return reshape(permutedims(Xr, (1,3,2)), (Ns*size(X,2), :))
end

function scaleUncertainty(sd, sdmin)
    sdvec = copy(sd)
    sdrev = 1 .- sdvec
    sd = (1.0 - sdmin)/(maximum(sdrev)-minimum(sdrev)) * (sdrev .- maximum(sdrev)) .+ 1
    return sd
end

end