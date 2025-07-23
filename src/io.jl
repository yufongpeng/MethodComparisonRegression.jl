repr_formula(mcr::MCRModel; digits = nothing, sigdigits = 4, base = 10) = repr_formula(mcr.model; digits, sigdigits, base)
repr_formula(m::LinearRegressionModel; digits = nothing, sigdigits = 4, base = 10) = repr_formula(m.pp; digits, sigdigits, base)
function repr_formula(predictor::Union{LinPred, DemingLinPred, PBPred}; digits = nothing, sigdigits = 4, base = 10)
    string("y = ", 
        round(first(predictor.β); digits, sigdigits, base), 
        last(predictor.β) > 0 ? " + " : " - ", abs(round(last(predictor.β); digits, sigdigits, base)), 
        "x"
    )
end

function repr_formula(predictor::Union{RatioPred, DemingRatioPred}; digits = nothing, sigdigits = 4, base = 10)
    string("y = ",
        round(predictor.β; digits, sigdigits, base), 
        "x"
    )
end

repr_r2(mcr::MCRModel; digits = nothing, sigdigits = 4, base = 10) = repr_r2(mcr.model; digits, sigdigits, base)
repr_r2(m::LinearRegressionModel; digits = nothing, sigdigits = 4, base = 10) = "R² = $(round(cor(m.pp.x, m.rr.y) ^ 2; digits, sigdigits, base))"

function Base.show(io::IO, range::Range)
    print(io, "⟦Practical⟧ x ∈ ", range.practical.x, ", y ∈ ", range.practical.y, " | ⟦Theoretical⟧ x ∈ ", range.theoretical.y, ", y ∈ ", range.theoretical.y)
end

function Base.show(io::IO, ::MIME"text/plain", range::Range)
    print(io, typeof(range), ":\n ")
    print(io, range)
end

function Base.show(io::IO, mcr::MCRModel)
    print(io, "⟦Model⟧ ", repr_formula(mcr))
end

function Base.show(io::IO, ::MIME"text/plain", mcr::MCRModel)
    print(io, "Method Comparison Regression model:\n ")
    print(io, mcr.model, "\n ")
    print(io, "x: ", mcr.model.pp.x, "\n ")
    print(io, "y: ", mcr.model.rr.y, "\n ")
    print(io, "range: ", mcr.range, "\n ")
    print(io, "weight: ", mcr.wnm, "\n ")
    print(io, "conversion: ", mcr.conversion == :x ? "y -> x\n " : mcr.conversion == :y ? "x -> y\n " : "\n ")
    print(io, "reference: ", mcr.reference == :x ? "x" : mcr.reference == :y ? "y" : "")
end

function Base.show(io::IO, estimator::MeanRatioModel)
    print(io, "⟦Mean of Ratios⟧ ", repr_formula(estimator))
end

function Base.show(io::IO, ::MIME"text/plain", estimator::MeanRatioModel)
    print(io, "Mean of Ratios:\n ")
    print(io, "x: ", estimator.pp.x, "\n ")
    print(io, "y: ", estimator.rr.y, "\n ")
    print(io, "predictor: ", repr_formula(estimator))
end

function Base.show(io::IO, estimator::RatioMeanModel)
    print(io, "⟦Ratio of Means⟧ ", repr_formula(estimator))
end

function Base.show(io::IO, ::MIME"text/plain", estimator::RatioMeanModel)
    print(io, "Ratio of Means:\n ")
    print(io, "x: ", estimator.pp.x, "\n ")
    print(io, "y: ", estimator.rr.y, "\n ")
    print(io, "predictor: ", repr_formula(estimator))
end

function Base.show(io::IO, estimator::LinearModel)
    print(io, "⟦Linear Regression⟧ ", repr_formula(estimator))
end

function Base.show(io::IO, ::MIME"text/plain", estimator::LinearModel)
    print(io, "Linear Regression:\n ")
    print(io, "x: ", estimator.pp.x, "\n ")
    print(io, "y: ", estimator.rr.y, "\n ")
    print(io, "wts: ", estimator.rr.wts, "\n ")
    print(io, "predictor: ", repr_formula(estimator))
end

function Base.show(io::IO, estimator::DemingModel)
    print(io, "⟦Deming Regression⟧ ", repr_formula(estimator))
end

function Base.show(io::IO, ::MIME"text/plain", estimator::DemingModel)
    print(io, "Deming Regression:\n ")
    print(io, "x: ", estimator.pp.x, "\n ")
    print(io, "y: ", estimator.rr.y, "\n ")
    print(io, "wts: ", estimator.rr.wts, "\n ")
    print(io, "λ: ", estimator.pp.λ, "\n ")
    print(io, "predictor: ", repr_formula(estimator))
end

function Base.show(io::IO, estimator::PassingBablokModel)
    print(io, "⟦Passing-Bablok Regression⟧ ", repr_formula(estimator))
end

function Base.show(io::IO, ::MIME"text/plain", estimator::PassingBablokModel)
    print(io, "Passing-Bablok Regression:\n ")
    print(io, "x: ", estimator.pp.x, "\n ")
    print(io, "y: ", estimator.rr.y, "\n ")
    print(io, "predictor: ", repr_formula(estimator))
end