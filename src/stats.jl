ncoef(m::MeanRatioModel) = 1
ncoef(m::RatioMeanModel) = 1
ncoef(m::LinearModel{L, <: LinPred}) where L = 2
ncoef(m::LinearModel{L, <:  RatioPred}) where L = 1
ncoef(m::DemingModel{L, <: DemingLinPred}) where L = 2
ncoef(m::DemingModel{L, <:  DemingRatioPred}) where L = 1
ncoef(m::PassingBablokModel) = 2

coef(m::LinearRegressionModel) = ncoef(m) == 1 ? [0, m.pp.β] : m.pp.β
coef(mcr::MCRModel) = coef(mcr.model)
dof(m::LinearRegressionModel) = length(m.pp.x) - ncoef(m)
dof(mcr::MCRModel) = dof(mcr.model)
# dof(mcr::MeanRatioModel) = sum(1 ./ map(mcr.wfn[mcr.wnm], mcr.x, mcr.y))
# dof(mcr::LinearModel) = sum(1 ./ map(mcr.wfn[mcr.wnm], mcr.x, mcr.y))
# error term
function deviance(r::LinResp)
    y = r.y
    mu = r.μ
    wts = r.wts
    v = zero(eltype(y)) + zero(eltype(y)) * zero(eltype(wts))
    if isempty(wts)
        @inbounds @simd for i in eachindex(y, mu)
            v += abs2(y[i] - mu[i])
        end
    else
        @inbounds @simd for i in eachindex(y, mu, wts)
            v += abs2(y[i] - mu[i]) * wts[i]
        end
    end
    v
end
"""
    var_err(mcr::AbstractMCRModel)

Variance of error term. For a given linear model, `y = β₀ + β₁x + ϵ`, this function returns `Var[ϵ]`, i.e. residual sum of squares.
"""
function var_err(m::LinearRegressionModel)
    deviance(m.rr) / dof(m)
end

var_err(mcr::MCRModel) = var_err(mcr.model)

std_err(m::LinearRegressionModel) = sqrt(var_err(m))
std_err(mcr::MCRModel) = sqrt(var_err(mcr))

# standard error for coef
"""
    coverror(mcr::AbstractMCRModel)

Covariance matrix of coefficients. If intercept was not fitted, the vaiance and covarience are `0`.
"""
function coverror(m::LinearRegressionModel)
    if ncoef(m) == 1
        [0; 0;; 0; var_err(m) * m.pp.invXtX]
    else
        var_err(m) .* m.pp.invXtX
    end
end

coverror(mcr::MCRModel) = coverror(mcr.model)

"""
    stderror(mcr::AbstractMCRModel)

Standard errors (se) of coefficients. If intercept was not fitted, the se is `0`.
"""
function stderror(m::LinearRegressionModel)
    ve = coverror(m)
    [sqrt(ve[i, i]) for i in axes(ve, 1)]
end

function stderror(m::DemingModel)
    m.pp isa DemingLinPred ? jackknife_stderror_estimates(coef, m) : [0, jackknife_stderror_estimate(last ∘ coef, m)]
end

stderror(mcr::MCRModel{<: LLSModel, AnalyticalSE}) = stderror(mcr.model)
stderror(mcr::MCRModel{<: PassingBablokModel, AnalyticalSE}) = stderror(mcr.model)
stderror(mcr::MCRModel{<: DemingModel, AnalyticalSE}) = (@warn "Analytical standard error not available, use jackknife method instead"; stderror(mcr.model))
function stderror(mcr::MCRModel{<: DemingModel, <: JackknifeSE}) 
    mcr.model.pp isa DemingLinPred ? jackknife_stderror(mcr.stderror, mcr.model.pp.β) : [0, jackknife_stderror(mcr.stderror, mcr.model.pp.β)]
end
stderror(mcr::MCRModel{<: PassingBablokModel, <: JackknifeSE}) = jackknife_stderror(mcr.stderror, mcr.model.pp.β)
function stderror(mcr::MCRModel{<: LLSModel, <: JackknifeSE})
    mcr.model.pp isa LinPred ? jackknife_stderror(mcr.stderror, mcr.model.pp.β) : [0, jackknife_stderror(mcr.stderror, mcr.model.pp.β)]
end

"""
    ci_coef(mcr::AbstractMCRModel; α = 0.05)

Confidence intervals (ci) of coeffeicients for a given type I error rate. If intercept was not fitted, the ci is `0`.
"""
function ci_coef(m::LinearRegressionModel; α = 0.05)
    r = quantile(TDist(dof(m)), 1 - α / 2)
    map(coef(m), stderror(m)) do c, d 
        c .+ [-r, r] .* d
    end
end

function ci_coef(m::PassingBablokModel; α = 0.05)
    n = length(m.pp.x)
    c = quantile(Normal(), 1 - α / 2) * sqrt(n * (n - 1) * (2n + 5) / 18)
    N = length(m.pp.β1s)
    m1 = convert(Int, (N - c) ÷ 2)
    m2 = N - m1 + 1
    β1 = [m.pp.β1s[m1 + m.pp.K], m.pp.β1s[m2 + m.pp.K]]
    [[median(m.rr.y .- last(β1) .* m.pp.x), median(m.rr.y .- first(β1) * m.pp.x)], β1]
end

function ci_coef(mcr::MCRModel{<: LinearRegressionModel}; α = 0.05) 
    r = quantile(TDist(dof(mcr)), 1 - α / 2)
    map(coef(mcr), stderror(mcr)) do c, d 
        c .+ [-r, r] .* d
    end
end

ci_coef(mcr::MCRModel{<: PassingBablokModel, AnalyticalSE}; α = 0.05) = ci_coef(mcr.model; α)

# standard error for joinedcoef
"""
    stderror_joinedcoef(mcr::AbstractMCRModel)

Standard errors (se) of predicted y(s) considering only joined errors of both coefficients. This function is mainly used for plotting confidence band.
"""
function stderror_joinedcoef(m::DemingModel, x)
    first(jackknife_stderror_estimates(y -> predict(y, [x]), m))
end

function stderror_joinedcoef(m::DemingModel, x::AbstractVector)
    jackknife_stderror_estimates(y -> predict(y, x), m)
end

function stderror_joinedcoef(m::LinearRegressionModel, x)
    sqrt([1, x]'coverror(m) * [1, x])
end

function stderror_joinedcoef(m::LinearRegressionModel, x::AbstractVector)
    cv = coverror(m)
    map(x) do a
        sqrt([1, a]'cv * [1, a])
    end
end

stderror_joinedcoef(mcr::MCRModel{<: LinearRegressionModel, AnalyticalSE}, x) = stderror_joinedcoef(mcr.model, x)
stderror_joinedcoef(mcr::MCRModel{<: DemingModel, AnalyticalSE}, x) = (@warn "Analytical standard error not available, use jackknife method instead"; stderror_joinedcoef(mcr.model, x))
stderror_joinedcoef(mcr::MCRModel{<: DemingModel, <: JackknifeSE}, x) = jackknife_stderror_joinedcoef(mcr.stderror, mcr.model.pp.β, x)
stderror_joinedcoef(mcr::MCRModel{<: LinearRegressionModel, <: JackknifeSE}, x) = jackknife_stderror_joinedcoef(mcr.stderror, mcr.model.pp.β, x)

"""
    stderror_joinedcoef(mcr::AbstractMCRModel)

Confidence band of predicted y(s) considering only joined errors of both coefficients for a given type I error rate. This function is mainly used for plotting confidence band.
"""
function ci_joinedcoef(m::LinearRegressionModel, x; α = 0.05)
    r = quantile(TDist(dof(m)), 1 - α / 2)
    y = first(predict(m, [x]))
    y .+ [-r, r] .* stderror_joinedcoef(m, x)
end

function ci_joinedcoef(m::LinearRegressionModel, x::AbstractVector; α = 0.05)
    r = quantile(TDist(dof(m)), 1 - α / 2)
    y = predict(m, x)
    map(y, stderror_joinedcoef(m, x)) do a, b 
        a .+ [-r, r] .* b
    end
end

function ci_joinedcoef(m::PassingBablokModel, x; α = 0.05)
    α, β = ci_coef(m; α)
    @. α + β * x
end

function ci_joinedcoef(m::PassingBablokModel, x::AbstractVector; α = 0.05)
    α, β = ci_coef(m; α)
    map(x) do a 
        @. α + β * a
    end
end

function ci_joinedcoef(mcr::MCRModel{<: LinearRegressionModel}, x; α = 0.05) 
    r = quantile(TDist(dof(mcr)), 1 - α / 2)
    y = first(predict(mcr.model, [x]))
    y .+ [-r, r] .* stderror_joinedcoef(mcr, x)
end

function ci_joinedcoef(mcr::MCRModel{<: LinearRegressionModel}, x::AbstractVector; α = 0.05) 
    r = quantile(TDist(dof(mcr)), 1 - α / 2)
    y = predict(mcr.model, x)
    map(y, stderror_joinedcoef(mcr, x)) do a, b 
        a .+ [-r, r] .* b
    end
end

ci_joinedcoef(mcr::MCRModel{<: PassingBablokModel}; α = 0.05) = ci_coef(mcr.model; α)

"""
"""
function jackknife_stderror(semethod::JackknifeSE{T}, β::T) where T
    n = length(semethod.βs)
    ne = β * n 
    ei = map(semethod.βs) do b 
        ne - (n - 1) * b
    end
    map(zip(ei...)) do e
        std(e) ./ sqrt(n)
    end
end

function jackknife_stderror(semethod::JackknifeSE{T}, β::T) where {T <: Vector}
    n = length(semethod.βs)
    ne = β .* n 
    ei = map(semethod.βs) do b 
        @. ne - (n - 1) * b
    end
    std(ei) / sqrt(n)
end

"""
"""
function jackknife_stderror_joinedcoef(semethod::JackknifeSE{T}, β::T, x::AbstractVector) where {T <: Vector}
    n = length(semethod.βs)
    ne = predict(β, x) .* n 
    ei = map(semethod.βs) do b 
        je = predict(b, x)
        @. ne - (n - 1) * je 
    end
    map(zip(ei...)) do e
        std(e) ./ sqrt(n)
    end
end

function jackknife_stderror_joinedcoef(semethod::JackknifeSE{T}, β::T, x) where {T <: Vector}
    n = length(semethod.βs)
    ne = predict(β, x) * n 
    ei = map(semethod.βs) do b 
        ne - (n - 1) * predict(b, x)
    end
    std(ei) / sqrt(n)
end

### Deming 
"""
    jackknife_stderror_estimate(f, mcr)

Calculating standard error of given estimate (`f(mcr)`) by jackknife method.
"""
function jackknife_stderror_estimate(estimator::Function, m)
    n = length(m.pp.x)
    ne = estimator(m) * n
    ei = map(eachindex(m.pp.x)) do i
        id = setdiff(eachindex(m.pp.x), i)
        ne - (n - 1) * estimator(fit(m, m.pp.x[id], m.rr.y[id]))
    end
    std(ei) / sqrt(n)
end

"""
    jackknife_stderror_estimates(f, mcr)

Calculating standard errors of multiple estimates (`f(mcr)`) by jackknife method.
"""
function jackknife_stderror_estimates(estimator::Function, m)
    n = length(m.pp.x)
    ne = estimator(m) .* n
    ei = map(eachindex(m.pp.x)) do i
        id = setdiff(eachindex(m.pp.x), i)
        je = estimator(fit(m, m.pp.x[id], m.rr.y[id]))
        @. ne - (n - 1) * je
    end
    map(zip(ei...)) do e
        std(e) ./ sqrt(n)
    end
end

# bootstrap