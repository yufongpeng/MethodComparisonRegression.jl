abstract type LLSPred{T} end

"""
    RatioPred{T} <: LLSPred{T}

Ratio predictor. 

# Fields 
* `x`: input predictor data
* `β`: ratio estimator
* `invXtX`: `1 / xᵀx` for computing sttandard error
"""
struct RatioPred{T} <: LLSPred{T}
    x::Vector{T}
    β::T
    invXtX::T
end

function RatioPred(x::AbstractVector{L}, β::U, invXtX::V) where {L, U, V}
    T = promote_type(L, U, V)
    RatioPred(convert(Vector{T}, x), convert(T, β), convert(T, invXtX))
end

"""
    LinPred{T} <: LLSPred{T}

Linear predictor. 

# Fields 
* `x`: input predictor data
* `β`: intercept and slope
* `invXtX`: `(XᵀX)⁻¹` for computing sttandard error
"""
struct LinPred{T} <: LLSPred{T}
    x::Vector{T}
    β::Vector{T}
    invXtX::Matrix{T}
end


function LinPred(x::AbstractVector{L}, β::AbstractVector{U}, invXtX::AbstractMatrix{V}) where {L, U, V}
    T = promote_type(L, U, V)
    LinPred(convert(Vector{T}, x), convert(Vector{T}, β), convert(Matrix{T}, invXtX))
end

abstract type DemingPred{T} end

"""
    DemingLinPred{T} <: DemingPred{T}

Linear predictor for Deming regression. 

# Fields 
* `x`: input predictor data
* `β`: intercept and slope
* `μ`: estimates of predictors 
* `λ`: the ratio of x variance to y variance
"""
mutable struct DemingLinPred{T} <: DemingPred{T}
    x::Vector{T}
    β::Vector{T}
    μ::Vector{T}
    λ::T
end

function DemingLinPred(x::AbstractVector{L}, β::AbstractVector{U}, μ::AbstractVector{V}, λ::S) where {L, U, V, S}
    T = promote_type(L, U, V, S)
    DemingLinPred(convert(Vector{T}, x), convert(Vector{T}, β), convert(Vector{T}, μ), convert(T, λ))
end

"""
    DemingRatioPred{T} <: DemingPred{T}

Linear predictor for Deming regression without fitting intercept. 

# Fields 
* `x`: input predictor data
* `β`: slope
* `μ`: estimates of predictors 
* `λ`: the ratio of x variance to y variance
"""
mutable struct DemingRatioPred{T} <: DemingPred{T}
    x::Vector{T}
    β::T
    μ::Vector{T}
    λ::T
end

function DemingRatioPred(x::AbstractVector{L}, β::U, μ::AbstractVector{V}, λ::S) where {L, U, V, S}
    T = promote_type(L, U, V, S)
    DemingRatioPred(convert(Vector{T}, x), convert(T, β), convert(Vector{T}, μ), convert(T, λ))
end

"""
    PBPred{T}

Linear predictor for Passing-Bablok regression. 

# Fields 
* `x`: input predictor data
* `β`: intercept and slope
* `β1s`: all valid slopes
* `negid`: the last index of negative elements of `β1s`
* `K`: shift of median of slopes
"""
mutable struct PBPred{T}
    x::Vector{T}
    β::Vector{T}
    β1s::Vector{T}
    negid::Union{Nothing, Int}
    K::Int
end

function PBPred(x::AbstractVector{L}, β::AbstractVector{U}, β1s::AbstractVector{V}, negid::Union{Nothing, <: Integer}, K::Integer) where {L, U, V}
    T = promote_type(L, U, V)
    PBPred(convert(Vector{T}, x), convert(Vector{T}, β), convert(Vector{T}, β1s), isnothing(negid) ? nothing : convert(Int, negid), convert(Int, K))
end

"""
    LinResp{T}

Encapsulates the response for a linear model.

# Fields
* `y`: input response data
* `μ`: estimates of responses
* `wts`: weights 
"""
mutable struct LinResp{T}
    y::Vector{T}
    μ::Vector{T}
    wts::Vector{T}
end

function LinResp(y::AbstractVector{L}, μ::AbstractVector{U}, wts::AbstractVector{V}) where {L, U, V}
    T = promote_type(L, U, V)
    LinResp(convert(Vector{T}, y), convert(Vector{T}, μ), convert(Vector{T}, wts))
end