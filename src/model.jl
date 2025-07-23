abstract type LinearRegressionModel <: RegressionModel end
abstract type LLSModel <: LinearRegressionModel end

"""
    MeanRatioModel{L, T} <: LLSModel

Model constructed by "Mean of Ratios", i.e. `mean(y ./ x)`. It is equivalent to fitting an ordinary linear model without intercept using weight "1/x²".

# Fields
* `rr`: `LinResp`, response data
* `pp`: `RatioPred`, predictor data
"""
struct MeanRatioModel{L <: LinResp, T <: RatioPred} <: LLSModel
    rr::L
    pp::T
end

"""
    RatioMeanModel{L, T} <: LLSModel{T}

Model constructed by "Ratio of Means", i.e. `mean(y) / mean(x)`. It is equivalent to fitting an ordinary linear model without intercept using weight "1/x".

# Fields
* `rr`: `LinResp`, response data
* `pp`: `RatioPred`, predictor data
"""
struct RatioMeanModel{L <: LinResp, T <: RatioPred} <: LLSModel
    rr::L
    pp::T
end

"""
    LinearModel{L, T} <: LLSModel

Ordinary or weighted linear regression model. 

# Fields
* `rr`: `LinResp`, response data
* `pp`: `LLSPred`, predictor data
* `wfn`: `nothing` or a weight function 
"""
struct LinearModel{L <: LinResp, T <: LLSPred} <: LLSModel
    rr::L
    pp::T
    wfn::Union{Nothing, Function}
end

"""
    DemingModel{L, T} <: LinearRegressionModel

Deming regression model. 

# Fields
* `rr`: `LinResp`, response data
* `pp`: `DemingPred`, predictor data
* `fit_λ`: whether to fit the ratio of x variance to y variance
* `maxiter`: maximal number of iteration 
* `atol`: absolute tolerance of estimate diffrence between iterations
* `rtol`: relative tolerance of estimate diffrence between iterations
* `id`: `nothing` or a vector of integer where each integer repressents a experimental or individual sample (not measurement duplicate)
* `wfn`: `nothing` or a weight function 

# Reference
1. Linnet K (1993). "Evaluation of regression procedures for methods comparison studies". Clinical Biochemistry. 39 (3): 424–32. doi:10.1093/clinchem/39.3.424
2. Linnet K (1990). "Estimation of the linear relationship between the measurements of two methods with proportional errors". Statistics in Medicine. 9: 1463–1473. doi:10.1002/sim.4780091210
"""
mutable struct DemingModel{L <: LinResp, T <: DemingPred} <: LinearRegressionModel
    rr::L
    pp::T
    fit_λ::Bool
    maxiter::Int
    atol::Float64
    rtol::Float64
    id::Union{Nothing, Vector{Int}}
    wfn::Union{Nothing, Function}
end

"""
    PassingBablokModel{L, T} <: LinearRegressionModel

Passing-Bablok regression model. 

# Fields
* `rr`: `LinResp`, response data
* `pp`: `PBPred`, predictor data
* `τ`: Kendall rank correlation coefficient
* `maxiter`: maximal number of iteration 

# Reference
1. Passing H, Bablok W (1983). "A new biometrical procedure for testing the equality of measurements from two different analytical methods. Application of linear regression procedures for method comparison studies in Clinical Chemistry, Part I". Journal of Clinical Chemistry and Clinical Biochemistry. 21 (11): 709–20. doi:10.1515/cclm.1983.21.11.709
2. Passing H, Bablok W (1984). "Comparison of several regression procedures for method comparison studies and determination of sample sizes. Application of linear regression procedures for method comparison studies in Clinical Chemistry, Part II" (PDF). Journal of Clinical Chemistry and Clinical Biochemistry. 22 (6): 431–45. doi:10.1515/cclm.1984.22.6.431
3. Bablok W, Passing H, Bender R, Schneider B (1988). "A general regression procedure for method transformation. Application of linear regression procedures for method comparison studies in clinical chemistry, Part III" (PDF). Journal of Clinical Chemistry and Clinical Biochemistry. 26 (11): 783–90. doi:10.1515/cclm.1988.26.11.783
"""
mutable struct PassingBablokModel{L <: LinResp, T <: PBPred} <: LinearRegressionModel
    rr::L
    pp::T
    τ::Float64
    maxiter::Int
end

abstract type SEMethod end
struct AnalyticalSE <: SEMethod end 
"""
"""
struct JackknifeSE{T} 
    βs::Vector{T}
end

"""
    MCRModel{M, S} <: RegressionModel

A wrapper for method comparison regression model.

# Fields 
* `model`: `LinearRegressionModel`
* `stderror`: `SEMethod`, contains method and data for standard error computation.
* `range`: `Range`, contains practical and theoretical x and y ranges.
* `wnm`: the name of weighting function (keys of `WFN`).
* `conversion`: determines whether converting from y to x (`:x`), x to y (`:y`), or no conversion(`:na`).
* `reference`: determines the reference measurement, `:x`, `:y`, or no reference (`:na`).
"""
mutable struct MCRModel{M, S} 
    model::M
    stderror::S
    range::Range
    wnm::String
    conversion::Symbol
    reference::Symbol
end

MCRModel(t::Type{<: Union{LinearModel, DemingModel}}, x, y; 
            practicalx = ri"(0, Inf)", 
            practicaly = ri"(0, Inf)", 
            theoreticalx = ri"[0, Inf)", 
            theoreticaly = ri"[0, Inf)",
            wnm = "1",
            wfn = WFN[wnm], 
            conversion = :na, 
            reference = :na, 
            stderror = t == DemingModel ? :JackknifeSE : :AnalyticalSE, 
            kwargs...) = 
    MCRModel(fit(t, x, y; wfn, kwargs...); practicalx, practicaly, theoreticalx, theoreticaly, wnm, conversion, reference, stderror)

MCRModel(t::Type{<: LinearRegressionModel}, x, y; 
            practicalx = ri"(0, Inf)", 
            practicaly = ri"(0, Inf)", 
            theoreticalx = ri"[0, Inf)", 
            theoreticaly = ri"[0, Inf)", 
            conversion = :na, 
            reference = :na, 
            stderror = :AnalyticalSE,
            kwargs...) = 
MCRModel(fit(t, x, y; kwargs...); practicalx, practicaly, theoreticalx, theoreticaly, wnm = "1", conversion, reference, stderror)

function MCRModel(
    m::LinearRegressionModel;
    practicalx = ri"(0, Inf)", 
    practicaly = ri"(0, Inf)", 
    theoreticalx = ri"[0, Inf)", 
    theoreticaly = ri"[0, Inf)", 
    wnm = "1", 
    conversion = :na, 
    reference = :na, 
    stderror = :AnalyticalSE,
    kwargs...) 
    if stderror == :AnalyticalSE
        stderror = AnalyticalSE()
    elseif stderror == :JackknifeSE
        stderror = JackknifeSE(map(eachindex(m.pp.x)) do i
            id = setdiff(eachindex(m.pp.x), i)
            fit(m, m.pp.x[id], m.rr.y[id]).pp.β
        end)
    end
    MCRModel(m, stderror, Range((x=practicalx, y=practicaly), (x=theoreticalx, y=theoreticaly)), wnm, conversion, reference)
end