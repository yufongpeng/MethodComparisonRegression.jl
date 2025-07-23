module MethodComparisonRegression

using Reexport, Combinatorics, LinearAlgebra, MLStyle, Plots, Distributions, Intervals
@reexport using TypedTables, DataPipes, SplitApplyCombine, StatsBase, Statistics
import Base: in, show
import StatsBase: predict, stderror, coef, fit, fit!
import StatsAPI: RegressionModel
import Distributions: dof
export MCRModel, MeanRatioModel, RatioMeanModel, LinearModel, DemingModel, PassingBablokModel,
        const_weight, invpredict, mae, mape, meae, meape, me, mpe, mee, mepe, mse, mspe, rmspe, rmse,
        BlandAltmanPlot, ScatterPlot, @ri_str
        
const FP = AbstractFloat
const FPVector{T <: FP} = AbstractArray{T, 1}

include("utils.jl")
include("predictor.jl")
include("model.jl")
include("fit.jl")
include("stats.jl")
include("metric.jl")
include("plot.jl")
include("io.jl")

end
