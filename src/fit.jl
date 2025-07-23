
"""
    fit(MeanRatioModel, x, y)

Construct a `MeanRatioModel`. It is equivalent to fitting an ordinary linear model without intercept using weight "1/x²".

The algorithm ignores data that will generate invalid ratios, i.e. `+∞` and `-∞`.

# Arguments
* `x`: predictor data
* `y`: response data 
"""
function fit(
        ::Type{<: MeanRatioModel},
        x::FPVector, 
        y::FPVector
    )
    id = findall(!=(0), x)
    x = x[id]
    y = y[id]
    β1 = @p map(/, y, x) mean
    MeanRatioModel(LinResp(y, β1 .* x, 1 ./ x .^ 2), RatioPred(x, β1, 1 / length(x)))
end

"""
    fit(RatioMeanModel, x, y)

Construct a `RatioMeanModel`. It is equivalent to fitting an ordinary linear model without intercept using weight "1/x".

# Arguments
* `x`: predictor data
* `y`: response data 
"""
function fit(
        ::Type{<: RatioMeanModel},
        x::FPVector, 
        y::FPVector
    )
    β1 = mapreduce(mean, /, (y, x))
    RatioMeanModel(LinResp(collect(y), β1 .* x, 1 ./ x), RatioPred(collect(x), β1, 1 / sum(x)))
end

"""
    LinearModel(
        x, y; 
        fit_intercept = true, 
        wfn = nothing
    )

Construct a `LinearModel`.

# Arguments
* `x`: predictor data
* `y`: response data 

# Keyword Arguments
* `fit_intercept`: whether fitting intercept or not 
* `wfn`: nothing or a weight function 
"""
function fit(
        ::Type{<: LinearModel}, 
        x::FPVector, 
        y::FPVector;
        fit_intercept = true, 
        wfn = nothing
    )
    _fit(LinearModel, fit_intercept ? [ones(length(y)) x] : x, y; wfn, wts = getwts(wfn, x, y))
end

function _fit(
        ::Type{<: LinearModel}, 
        x::FPVector, 
        y::FPVector;
        wfn = nothing,
        wts::FPVector = similar(y, 0)
    )
    if isempty(wts)
        v = x'x 
        u = x'y
    else
        v = zero(eltype(x)) + zero(eltype(x)) * zero(eltype(wts))
        u = zero(eltype(x)) + zero(eltype(y)) + zero(eltype(x)) * zero(eltype(y)) * zero(eltype(wts))
        @inbounds @simd for i in eachindex(x, y, wts)
            v += x[i] ^ 2 * wts[i]
            u += x[i] * y[i] * wts[i]
        end
    end
    invXtX = 1 / v
    β1 = invXtX * u
    LinearModel(
                    LinResp(collect(y), β1 .* x, wts),
                    RatioPred(collect(x), β1, invXtX),
                    wfn
                )
end

function _fit(
        ::Type{<: LinearModel}, 
        X::AbstractMatrix{<: FP}, 
        y::FPVector;
        wfn = nothing,
        wts::FPVector = similar(y, 0)
    )
    if isempty(wts)
        invXtX = pinv(X'X)
        β = invXtX * X'y
    else
        W = diagm(wts)
        invXtX = pinv(X'W * X)
        β = invXtX * X'W * y
    end
    LinearModel(
                    LinResp(collect(y), X * β, wts),
                    LinPred(X[:, end], β, invXtX),
                    wfn
                )
end

"""
    fit(DemingModel,
        x, y; 
        λ = 1, 
        wfn = nothing,
        fit_intercept = true, 
        fit_λ = true, 
        id = nothing, 
        maxiter = (isnothing(wfn) || wfn == const_weight) ? 0 : 100, 
        atol = 1e-4, 
        rtol = 1e-4
    )

Construct a `DemingModel`.

# Arguments
* `x`: predictor data
* `y`: response data 

# Keyword Arguments
* `λ`: initial value of the ratio of x variance to y variance
* `wfn`: nothing or a weight function 
* `fit_intercept`: whether fitting intercept or not 
* `fit_λ`: whether fitting the ratio of x variance to y variance (λ) using repeated measure data (`id`), and update the value during iteration.
If repeated measure data is not available, the algorithm will set `λ = 1` initially.
* `id`: nothing or a vector of integer where each integer repressents a experimental or individual sample (not measurement duplicate)
* `atol`: absolute tolerance of estimate diffrence between iterations
* `rtol`: relative tolerance of estimate diffrence between iterations
* `maxiter`: maximal number of iterations
"""
function fit(
        ::Type{<: DemingModel},
        x::FPVector, 
        y::FPVector;
        λ = 1, 
        wfn = nothing,
        fit_intercept = true, 
        fit_λ = true, 
        id = nothing, 
        maxiter = 100,
        atol = 1e-6, 
        rtol = 1e-6, 
    )
    fit!(DemingModel(
                    LinResp(collect(y), similar(y, 0), similar(y, 0)), 
                    fit_intercept ? DemingLinPred(collect(x), [0, 0], similar(x, 0), convert(eltype(x), λ)) : DemingRatioPred(collect(x), 0, similar(x, 0), convert(eltype(x), λ)), 
                    fit_λ, 
                    maxiter, atol, rtol, id, wfn
                    )
        )
end

function fit!(m::DemingModel)
    id = m.id
    x = m.pp.x
    y = m.rr.y
    if m.maxiter < 2
        if !isnothing(id)
            vx = @p x group(id) map(sqerr) collect
            vy = @p y group(id) map(sqerr) collect
            x = @p x group(id) map(mean) collect
            y = @p y group(id) map(mean) collect
        end
        m.rr.wts = getwts(m.wfn, x, y)
        if !isnothing(id) && m.fit_λ
            m.pp.λ = isempty(m.rr.wts) ? sum(vx) / sum(vy) : vx'm.rr.wts / vy'm.rr.wts
        end
        installbeta_deming!(m)
    else
        it = 0
        β1o = 0
        if !isnothing(id)
            vx = @p x group(id) map(sqerr) collect
            vy = @p y group(id) map(sqerr) collect
            x = @p x group(id) map(mean) collect
            y = @p y group(id) map(mean) collect
        end
        if !isnothing(id) && m.fit_λ
            #m = x
            #λ = sum(x -> x ^ 2, rvx ./ m) / sum(x -> x ^ 2, rvy ./ m)
            m.pp.λ = sum(vx) / sum(vy)
        end
        m.rr.wts = similar(y, 0)

        while it < m.maxiter
            it += 1
            installbeta_deming!(m)
            β1 = last(coef(m))
            if abs(β1 - β1o) < m.atol && abs(β1 - β1o) / (β1 + β1o) * 2 < m.rtol
                break
            end
            β1o = β1
            di = y .- predict(m, x)
            x̂ = @. max(x + m.pp.λ * β1 * di / (1 + m.pp.λ * β1 ^ 2), 0)
            ŷ = @. max(y - di / (1 + m.pp.λ * β1 ^ 2), 0)
            m.rr.wts = getwts(m.wfn, x̂, ŷ)
            if isnothing(id) && m.fit_λ
                m.pp.λ = (1 / β1) ^ 2
            elseif m.fit_λ
                m.pp.λ = isempty(m.rr.wts) ? sum(vx) / sum(vy) : vx'm.rr.wts / vy'm.rr.wts
            end
        end
        if it == m.maxiter
            @warn "β1 does not converge. Try setting larger `maxiter`."
        end
    end
    β1 = last(coef(m))
    di = y .- predict(m, x)
    m.rr.μ = @. y - di / (1 + m.pp.λ * β1 ^ 2)
    m.pp.μ = @. x + m.pp.λ * β1 * di / (1 + m.pp.λ * β1 ^ 2)
    m
end

"""
    PassingBablokModel(
        x, y; 
        τ = 1, 
        maxiter = 1000
    )

Construct a `PassingBablokModel`.

# Arguments
* `x`: predictor data
* `y`: response data 

# Keyword Arguments
* `τ`: Kendall rank correlation coefficient
* `maxiter`: maximal number of iterations
"""
function fit(
        ::Type{<: PassingBablokModel},
        x::FPVector, 
        y::FPVector;
        τ = 1.0,
        maxiter = 1000
    )
    y′ = τ < 0 ? -y : y
    s = map(combinations(collect(zip(x, y′)), 2)) do ((xi, yi), (xj, yj)) 
        s = (yi - yj) / (xi - xj)
        isnan(s) ? NaN : 
        s > 0 ? s : 
        s < 0 ? s - 1 : 
        (xi - xj > 0) ? s : 
        s - 1
    end
    filter!(!isnan, s)
    sort!(s)
    negid = findlast(<(0), s)
    if !isnothing(negid)
        s[begin:negid] .+= 1
    end
    fit!(PassingBablokModel(
                            LinResp(collect(y), similar(y, 0), similar(y, 0)), 
                            PBPred(collect(x), [0, 0], s, negid, 0), 
                            τ, maxiter
                            )
            )
end

function fit!(m::PassingBablokModel)
    s = m.pp.β1s
    n = length(s)
    if m.maxiter < 2
        k = count(<(0), s) ÷ 2
        β1 = isodd(n) ? s[(n + 1) ÷ 2 + k] : sqrt(s[n ÷ 2 + k] * s[n ÷ 2 + 1 + k])
    else
        it = 0
        β1o = 1
        ks = Int[]
        βs = eltype(m.pp.x)[]
        id = nothing
        while it < m.maxiter
            it += 1
            k = count(<(-β1o - 1), s) + count(==(-β1o - 1), s) ÷ 2
            β1 = isodd(n) ? s[(n + 1) ÷ 2 + k] : sqrt(s[n ÷ 2 + k] * s[n ÷ 2 + 1 + k])
            id = findfirst(==(k), ks)
            if !isnothing(id)
                push!(βs, β1)
                break
            else
                push!(ks, k)
                push!(βs, β1)
                β1o = β1
            end
        end
        if isnothing(id)
            @warn "β1 does not converge. Try setting larger `maxiter`."
            k = m.pp.negid ÷ 2
            β1 = isodd(n) ? s[(n + 1) ÷ 2 + k] : sqrt(s[n ÷ 2 + k] * s[n ÷ 2 + 1 + k])
        else
            mi = length(ks) + 1 - id
            k = ks[id]
            β1 = iseven(mi) ? βs[id + mi ÷ 2] : sqrt(βs[id + mi ÷ 2] * βs[id + mi ÷ 2 + 1])
        end
    end
    β1 = m.τ < 0 ? -β1 : β1
    β0 = median(m.rr.y .- β1 .* m.pp.x)
    m.pp.β = [β0, β1]
    m.rr.μ = predict(m, m.pp.x)
    m
end

function installbeta_deming!(m::DemingModel{L, <: DemingLinPred}) where L
    x = m.pp.x
    y = m.rr.y 
    wts = m.rr.wts
    λ = m.pp.λ
    if isempty(wts)
        x̄ = mean(x)
        ȳ = mean(y)
        vx = var(x)
        vy = var(y)
        vxy = cov([x y])[begin, begin + 1]
    else
        x̄ = mean(x, weights(wts))
        ȳ = mean(y, weights(wts))
        vx = var(x, weights(wts))
        vy = var(y, weights(wts))
        vxy = cov([x y], weights(wts))[begin, begin + 1]
    end
    β1 = (λ * vy - vx + sqrt((vx - λ * vy) ^ 2 + 4 * λ * vxy ^ 2)) / (2 * λ * vxy)
    β0 = ȳ - β1 * x̄
    m.pp.β = [β0, β1]
    m
end

function installbeta_deming!(m::DemingModel{L, <: DemingRatioPred}) where L
    x = m.pp.x
    y = m.rr.y 
    wts = m.rr.wts
    λ = m.pp.λ
    if isempty(wts)
        vx = x'x
        vy = y'y 
        vxy = x'y 
    else
        vx = x' * (x .* wts)
        vy = y' * (y .* wts)
        vxy = x' * (y .* wts)
    end
    m.pp.β = (λ * vy - vx + sqrt((vx - λ * vy) ^ 2 + 4 * λ * vxy ^ 2)) / (2 * λ * vxy)
    m
end

function fit(m::T, x, y) where {T <: LLSModel}
    fit(T, x, y)
end

function fit(m::T, x, y) where {T <: LinearModel}
    fit(T, x, y; fit_intercept = m.pp isa LinPred, wfn = m.wfn)
end

function fit(m::T, x, y) where {T <: DemingModel}
    fit(T, x, y; fit_intercept = m.pp isa DemingLinPred, wfn = m.wfn, fit_λ = m.fit_λ, id = m.id, maxiter = m.maxiter, atol = m.atol, rtol = m.rtol)
end

function fit(m::T, x, y) where {T <: PassingBablokModel}
    fit(T, x, y; maxiter = m.maxiter, τ = m.τ)
end

# refit function
# function fit(mcr::MCRModel, x, y)
#     MCRModel(fit(mcr.model, x, y);
#         practicalx = mcr.range.practical.x, 
#         practicaly = mcr.range.practical.y, 
#         theoreticalx = mcr.range.theoretical.x, 
#         theoreticaly = mcr.range.theoretical.y, 
#         wnm = mcr.wnm
#     )
# end