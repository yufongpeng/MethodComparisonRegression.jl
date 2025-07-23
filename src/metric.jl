predict(m::LinearRegressionModel, x) = predict(m.pp.β, x)
# predict(m::LinearRegressionModel, x::FPVector) = predict(m.pp.β, x)
predict(β::Vector, x) = last(β) * x + first(β)
predict(β::Vector, x::FPVector) = last(β) .* x .+ first(β)
predict(β::Number, x) = β * x
predict(β::Number, x::FPVector) = β .* x

invpredict(m::LinearRegressionModel, x) = invpredict(m.pp.β, x)
# invpredict(m::LinearRegressionModel, x::FPVector) = invpredict(m.pp.β, x)
invpredict(β::Vector, x) = (x - first(β)) / last(β)
invpredict(β::Vector, x::FPVector) = (x .- first(β)) ./ last(β)
invpredict(β::Number, x) = x / β
invpredict(β::Number, x::FPVector) = x ./ β

# deal only with output
predict(mcr::MCRModel, xs::AbstractVector) = map(xs) do x
    x in mcr.range.theoretical.x || return NaN
    yr = mcr.range.theoretical.y
    # x ≪ xr && return coef(mcr)[2] < 0 ? tyr.last : tyr.first
    # xr ≪ x && return coef(mcr)[2] < 0 ? tyr.first : tyr.last
    y = predict(mcr.model, x)
    y ≪ yr && return yr.first
    yr ≪ y && return yr.last
    return y
end

invpredict(mcr::MCRModel, xs::AbstractVector) = map(xs) do x
    x in mcr.range.theoretical.y || return NaN
    yr = mcr.range.theoretical.x
    y = invpredict(mcr.model, x)
    y ≪ yr && return yr.first
    yr ≪ y && return yr.last
    return y
end

# (mcr::AbstractMCRModel{true})(x) = x == 0 ? 0 : max(mcr.model.β0 + coef(mcr)[2] * x, 0)
# (mcr::AbstractMCRModel{false})(x) = x == 0 ? 0 : max((x - mcr.model.β0) / coef(mcr)[2], 0)
square(x) = x ^ 2
sqerr(x, y) = (x - y) ^ 2
aberr(x, y) = abs(x - y)
ref_first(x, y) = x 
ref_last(x, y) = y 
ref_mean(x, y) = x / 2 + y / 2
# std (ref), var (ref), iqr, mad (ref), 
extrema_range(x) = -reduce(-, extrema(x))
square_extrema_range(x) = reduce(-, extrema(x)) ^ 2
square_iqr(x) = iqr(x) ^ 2

# err relative to ref 


# zero_variation(var_fn, x, args...; kwargs...) = length(x) == 1 ? 0 : var_fn(x, args...; kwargs...)

function _rdev_counter(err_fn, ref_fn, x, μ)
    x == μ && return (1, 0)
    μ <= 0 && return (0, 0)
    (1, err_fn(x, μ) / ref_fn(x, μ))
end
function _rdev(err_fn, ref_fn, x, μ)
    x == μ && return 0
    μ = max(μ, 0)
    err_fn(x, μ) / ref_fn(x, μ)
end
function ref_est(mcr::MCRModel)
    if mcr.reference == :x
        ref_first
    elseif mcr.reference == :y 
        ref_last
    else
        ref_mean 
    end
end
# skip /0
"""
    mape(ŷ, y; err_fn = aberr, ref_fn = ref_mean)

Mean absolute percentage error relative to baseline. For each element, if the mean is less than baseline, this function skip this element. 
"""
function mape(ŷ, y; err_fn = aberr, ref_fn = ref_mean)
    n = 0
    s = 0
    for (x̂, x) in zip(ŷ, y)
        a, b = _rdev_counter(err_fn, ref_fn, x̂, x)
        n += a 
        s += b
    end
    s / n
end
mape(mcr::MCRModel, ŷ, y) = mape(ŷ, y; err_fn = aberr, ref_fn = ref_est(mcr))
"""
    mae(ŷ, y; err_fn = aberr, ref_fn = ref_mean)

Mean absolute error.
"""
mae(ŷ, y; err_fn = aberr, ref_fn = ref_mean) = mean(x -> err_fn(first(x), last(x)), zip(ŷ, y))
mae(mcr::MCRModel, ŷ, y) = mae(ŷ, y; err_fn = aberr, ref_fn = ref_est(mcr))
# x/0 -> inf
"""
    meape(ŷ, y; err_fn = aberr, ref_fn = ref_mean)

Median absolute percentage error relative to baseline. For each element, if the mean is less than baseline, the percentage error becomes infinite. 
"""
meape(ŷ, y; err_fn = aberr, ref_fn = ref_mean) = median(_rdev(err_fn, ref_fn, x...) for x in zip(ŷ, y))
meape(mcr::MCRModel, ŷ, y) = meape(ŷ, y; err_fn = aberr, ref_fn = ref_est(mcr))
# skip /0
"""
    meae(ŷ, y; err_fn = aberr, ref_fn = ref_mean)

Median absolute error.
"""
meae(ŷ, y; err_fn = aberr, ref_fn = ref_mean) = median(err_fn(first(x), last(x)) for x in zip(ŷ, y))
meae(mcr::MCRModel, ŷ, y) = meae(ŷ, y; err_fn = aberr, ref_fn = ref_est(mcr))
"""
    mpe(ŷ, y; err_fn = -, ref_fn = ref_mean)

Mean percentage error relative to baseline. For each element, if the mean is less than baseline, this function skip this element. 
"""               
function mpe(ŷ, y; err_fn = -, ref_fn = ref_mean)
    n = 0
    s = 0
    for (x̂, x) in zip(ŷ, y)
        a, b = _rdev_counter(err_fn, ref_fn, x̂, x)
        n += a 
        s += b
    end
    s / n
end
mpe(mcr::MCRModel, ŷ, y) = mpe(ŷ, y; err_fn = -, ref_fn = ref_est(mcr))
"""
    me(ŷ, y; err_fn = -, ref_fn = ref_mean)

Mean error.
"""
me(ŷ, y; err_fn = -, ref_fn = ref_mean) = mean(x -> err_fn(first(x), last(x)), zip(ŷ, y))
me(mcr::MCRModel, ŷ, y) = me(ŷ, y; err_fn = -, ref_fn = ref_est(mcr))
# x/0 -> +/-inf
"""
    mepe(ŷ, y; err_fn = -, ref_fn = ref_mean)

Median percentage error relative to baseline. For each element, if the mean is less than baseline, the percentage error becomes positive or negative infinite. 
"""  
mepe(ŷ, y; err_fn = -, ref_fn = ref_mean) = median(_rdev(err_fn, ref_fn, x...) for x in zip(ŷ, y))
mepe(mcr::MCRModel, ŷ, y) = mepe(ŷ, y; err_fn = -, ref_fn = ref_est(mcr))
"""
    mee(ŷ, y; err_fn = -, ref_fn = ref_mean

Median error.
"""
mee(ŷ, y; err_fn = -, ref_fn = ref_mean) = median(err_fn(first(x), last(x)) for x in zip(ŷ, y))
mee(mcr::MCRModel, ŷ, y) = mee(ŷ, y; err_fn = -, ref_fn = ref_est(mcr))
# skip /0
"""
    mspe(ŷ, y; err_fn = sqerr, ref_fn = ref_mean)

Mean squared percentage error relative to baseline. For each element, if the mean is less than baseline, this function skip this element. 
"""
function mspe(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_mean)
    n = 0
    s = 0
    for (x̂, x) in zip(ŷ, y)
        a, b = _rdev_counter(err_fn, ref_fn, x̂, x)
        n += a 
        s += b
    end
    s / n
end
mspe(mcr::MCRModel, ŷ, y) = mspe(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_est(mcr))
# skip /0
"""
    rmspe(ŷ, y; err_fn = sqerr, ref_fn = ref_mean)

Root mean squared percentage error relative to baseline. For each element, if the mean is less than baseline, this function skip this element. 
"""
function rmspe(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_mean)
    n = 0
    s = 0
    for (x̂, x) in zip(ŷ, y)
        a, b = _rdev_counter(err_fn, ref_fn, x̂, x)
        n += a 
        s += b
    end
    sqrt(s / n)
end
rmspe(mcr::MCRModel, ŷ, y) = rmspe(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_est(mcr))
"""
    mse(ŷ, y; err_fn = sqerr, ref_fn = ref_mean)

Mean squared error.
"""
mse(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_mean) = mean(x -> err_fn(first(x), last(x)), zip(ŷ, y))
mse(mcr::MCRModel, ŷ, y) = mse(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_est(mcr))
"""
    rmse(ŷ, y; err_fn = sqerr, ref_fn = ref_mean)

Root mean squared error.
"""
rmse(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_mean) = sqrt(mean(x -> err_fn(first(x), last(x)), zip(ŷ, y)))
rmse(mcr::MCRModel, ŷ, y) = rmse(ŷ, y; err_fn = sqerr, ref_fn = square ∘ ref_est(mcr))