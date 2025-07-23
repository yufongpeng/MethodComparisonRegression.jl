function ScatterPlot(mcr::MCRModel; reg = true, ci = true, rel = nothing, α = 0.05, r2 = true, formula = true, kwargs...)
    xlim = extrema(mcr.model.pp.x)
    xlim = xlim .+ reduce(-, xlim) .* [0.1, -0.1]
    xlim = (max(first(xlim), mcr.range.practical.x.first), min(last(xlim), mcr.range.practical.x.last))
    ylim = extrema(mcr.model.rr.y)
    ylim = ylim .+ reduce(-, ylim) .* [0.1, -0.1]
    ylim = (max(first(ylim), mcr.range.practical.y.first), min(last(ylim), mcr.range.practical.y.last))
    
    xlim2 = invpredict(mcr.model, collect(ylim))
    if coef(mcr)[2] < 0 
        xlim2 = [last(xlim2), first(xlim2)]
    end
    # ylim2 = predict(mcr, collect(xlim))
    xlim2[begin] = isnan(xlim2[begin]) ? mcr.range.practical.x.first : xlim2[begin]
    xlim2[end] = isnan(xlim2[end]) ? mcr.range.practical.x.last : xlim2[end]
    xlim2[begin] = isinf(xlim2[begin]) ? first(xlim) : xlim2[begin]
    xlim2[end] = isinf(xlim2[end]) ? last(xlim) : xlim2[end]
    xlim = [min(first(xlim), first(xlim2)), max(last(xlim), last(xlim2))]
    xlim[1] = max(xlim[1], mcr.range.practical.x.first)
    xlim[2] = min(xlim[2], mcr.range.practical.x.last)

    ylim = predict(mcr.model, xlim)

    reg = reg ? (xlim, ylim) : nothing 
    rel = isnothing(rel) ? rel : (xlim, rel.(xlim))
    if ci 
        xs = LinRange(xlim..., 100)
        ci = (xs, ci_joinedcoef(mcr, xs; α))
    else
        ci = nothing 
    end
    formula = formula ? repr_formula(mcr) : nothing
    r2 = r2 ? repr_r2(mcr) : nothing
    ScatterPlot(mcr.model.pp.x, mcr.model.rr.y; reg, ci, rel, α, r2, formula, kwargs...)
end

function ScatterPlot(x, y; reg = nothing, ci = nothing, rel = nothing, α = 0.05, r2 = nothing, formula = nothing, kwargs...)
    plot_kwargs = Dict{Symbol, Any}()
    point_kwargs = Dict{Symbol, Any}()
    reg_kwargs = Dict{Symbol, Any}()
    rel_kwargs = Dict{Symbol, Any}()
    ci_kwargs = Dict{Symbol, Any}()
    ann_kwargs = Dict{Symbol, Any}()
    for (k, v) in kwargs
        k = string(k)
        if startswith(k, "point_")
            push!(point_kwargs, Symbol(replace(k, r"^point_" => "")) => v)
        elseif startswith(k, "reg_")
            push!(reg_kwargs, Symbol(replace(k, r"^reg_" => "")) => v)
        elseif startswith(k, "rel_")
            push!(reg_kwargs, Symbol(replace(k, r"^reg_" => "")) => v)
        elseif startswith(k, "ci_")
            push!(ci_kwargs, Symbol(replace(k, r"^ci_" => "")) => v)
        elseif startswith(k, "ann_")
            push!(ann_kwargs, Symbol(replace(k, r"^ann_" => "")) => v)
        else
            push!(plot_kwargs, Symbol(replace(k, r"^plot_" => "")) => v)
        end
    end

    if isnothing(reg)
        xlim = extrema(x)
        ylim = extrema(y)
    else
        xlim, ylim = reg 
    end

    get!(point_kwargs, :label, false)
    get!(point_kwargs, :color, :deepskyblue)
    get!(point_kwargs, :markersize, 3)
    get!(ci_kwargs, :color, :chartreuse)
    get!(ci_kwargs, :alpha, 0.2)
    get!(ci_kwargs, :label, false)
    get!(reg_kwargs, :label, false)
    get!(reg_kwargs, :color, :green)
    get!(rel_kwargs, :label, false)
    get!(rel_kwargs, :color, :darkred)
    get!(rel_kwargs, :linestyle, :dash)
    get!(ann_kwargs, :x, first(xlim))
    get!(ann_kwargs, :y, last(ylim))
    get!(ann_kwargs, :styles, (:match, ))
    get!(ann_kwargs, :txt, "")
    if ann_kwargs[:txt] isa Tuple
        ann_txt_f = first(ann_kwargs[:txt])
        ann_txt_l = last(ann_kwargs[:txt])
    else
        ann_txt_f = "" 
        ann_txt_l = ann_kwargs[:txt]
    end
    if isnothing(r2) && isnothing(formula)
        ann_kwargs[:txt] = (join(filter!(!isempty, [ann_txt_f, ann_txt_l]), "\n"), 10, :black, :left, ann_kwargs[:styles]...)
    elseif isnothing(formula)
        ann_kwargs[:txt] = (join(filter!(!isempty, [ann_txt_f, r2, ann_txt_l]), "\n"), 10, :black, :left, ann_kwargs[:styles]...)
    elseif isnothing(r2)
        ann_kwargs[:txt] = (join(filter!(!isempty, [ann_txt_f, formula, ann_txt_l]), "\n"), 10, :black, :left, ann_kwargs[:styles]...)
    else
        ann_kwargs[:txt] = (join(filter!(!isempty, [ann_txt_f, formula, r2, ann_txt_l]), "\n"), 10, :black, :left, ann_kwargs[:styles]...)
    end
    get!(plot_kwargs, :title, "Scatter plot")
    get!(plot_kwargs, :xlabel, "x")
    get!(plot_kwargs, :ylabel, "y")
    
    if isnothing(ci)
        plot()
    else
        xs, ci = ci
        plot(xs, last.(ci); fillrange = first.(ci), ci_kwargs...)
    end
    scatter!(x, y; point_kwargs...)
    if !isnothing(reg)
        plot!(xlim, ylim; reg_kwargs...)
    end
    if !isnothing(rel)
        xlim2, ylim2 = rel 
        plot!(xlim2, ylim2; rel_kwargs...)
    end
    if !isempty(ann_kwargs[:txt])
        annotate!(ann_kwargs[:x], ann_kwargs[:y], ann_kwargs[:txt])
    end
    plot!(; plot_kwargs...)
end

function BlandAltmanPlot(mcr::MCRModel; kwargs...)
    if mcr.conversion == :y
        y = mcr.model.rr.y
        x = predict(mcr, mcr.model.pp.x)
    elseif mcr.conversion == :x
        x = mcr.model.pp.x
        y = invpredict(mcr, mcr.model.rr.y)
    else
        y = mcr.model.rr.y
        x = mcr.model.pp.x
    end
    if mcr.reference == :y 
        p = y 
        p̂ = x 
        m = y 
    elseif mcr.reference == :x 
        p = x 
        p̂ = y 
        m = x 
    elseif mcr.conversion == :y
        p = y 
        p̂ = x
        m = mean.(collect(zip(p̂, p))) 
    elseif mcr.conversion == :x
        p = x 
        p̂ = y 
        m = mean.(collect(zip(p̂, p))) 
    else
        p = y 
        p̂ = x 
        m = mean.(collect(zip(p̂, p))) 
    end
    BlandAltmanPlot(p̂, p, m; kwargs...)
end

function BlandAltmanPlot(p̂, p, m; pct = true, α = 0.05, loa = true, ci = true, maxdiff = 20, reg = false, regci = true, kwargs...)
    plot_kwargs = Dict{Symbol, Any}()
    point_kwargs = Dict{Symbol, Any}()
    mean_kwargs = Dict{Symbol, Any}()
    loa_kwargs = Dict{Symbol, Any}()
    ci_kwargs = Dict{Symbol, Any}()
    maxdiff_kwargs = Dict{Symbol, Any}()
    reg_kwargs = Dict{Symbol, Any}()
    regci_kwargs = Dict{Symbol, Any}()
    for (k, v) in kwargs
        k = string(k)
        if startswith(k, "point_")
            push!(point_kwargs, Symbol(replace(k, r"^point_" => "")) => v)
        elseif startswith(k, "mean_")
            push!(mean_kwargs, Symbol(replace(k, r"^mean_" => "")) => v)
        elseif startswith(k, "loa_")
            push!(loa_kwargs, Symbol(replace(k, r"^loa_" => "")) => v)
        elseif startswith(k, "ci_")
            push!(ci_kwargs, Symbol(replace(k, r"^ci_" => "")) => v)
        elseif startswith(k, "maxdiff_")
            push!(maxdiff_kwargs, Symbol(replace(k, r"^maxdiff_" => "")) => v)
        elseif startswith(k, "reg_")
            push!(reg_kwargs, Symbol(replace(k, r"^reg_" => "")) => v)
        else
            push!(plot_kwargs, Symbol(replace(k, r"^plot_" => "")) => v)
        end
    end    
    Δ = p̂ .- p
    Δ = pct ? Δ ./ m .* 100 : Δ
    s = std(Δ)
    av = mean(Δ)
    q = quantile(Normal(), 1 - α / 2)
    uloa = av + q * s
    lloa = av - q * s
    x = extrema(m) .+ reduce(-, extrema(m)) .* [0.1, -0.1]

    get!(point_kwargs, :label, false)
    get!(point_kwargs, :color, :deepskyblue)
    get!(mean_kwargs, :label, "Mean")
    get!(mean_kwargs, :color, :blue)
    get!(loa_kwargs, :label, "LoA")
    get!(loa_kwargs, :color, :green)
    get!(loa_kwargs, :linestyle, :dashdotdot)
    get!(loa_kwargs, :alpha, [1, 0, 1, 1])
    get!(maxdiff_kwargs, :label, "Max.Diff.")
    get!(maxdiff_kwargs, :color, :red)
    get!(maxdiff_kwargs, :linestyle, :dash)
    get!(maxdiff_kwargs, :alpha, [1, 0, 1, 1])
    get!(ci_kwargs, :label, false)
    get!(ci_kwargs, :color, :darkgreen)
    get!(ci_kwargs, :x, :right)
    get!(regci_kwargs, :color, :gray50)
    get!(regci_kwargs, :alpha, 0.2)
    get!(regci_kwargs, :label, false)
    get!(reg_kwargs, :color, :gray20)
    get!(reg_kwargs, :label, false)
    get!(plot_kwargs, :title, "Bland-Altman plot")
    get!(plot_kwargs, :xlabel, "Reference")
    get!(plot_kwargs, :ylabel, pct ? "% Difference" : "Difference")
    get!(plot_kwargs, :legend, :outertopright)

    n = length(m)
    scatter(m, Δ; point_kwargs...)
    plot!(x, [av, av]; mean_kwargs...)
    if loa
        plot!([x; x], [lloa, lloa, uloa, uloa]; loa_kwargs...)
    end
    if ci
        #v0 = s / sqrt(n)
        v = s * sqrt(1 / n + q ^ 2 / 2 / (n - 1))
        r = quantile(TDist(n - 1), 1 - α / 2)
        xbar = pop!(ci_kwargs, :x)
        xbar = xbar == :right ? last(x) * 0.5 + maximum(m) * 0.5 : xbar == :left ? first(x) * 0.5 + minimum(m) * 0.5 : xbar
        plot!([xbar], [uloa]; yerror = r * v, ci_kwargs...)
        plot!([xbar], [av]; yerror = r * s / sqrt(n), ci_kwargs...)
        plot!([xbar], [lloa]; yerror = r * v, ci_kwargs...)
    end
    if !isnothing(maxdiff) || !isa(maxdiff, Bool) || !maxdiff
        if !isa(maxdiff, Number)
            if pct 
                maxdiff = 20
            else
                maxdiff = mean(m) * 0.2
            end
        end
        plot!([x; x], [-maxdiff, -maxdiff, maxdiff, maxdiff]; maxdiff_kwargs...)
    end
    if reg 
        m = fit(LinearModel, m, Δ)
        xs = LinRange(x..., 100)
        ci = ci_joinedcoef(m, xs; α)
        regci && plot!(xs, last.(ci); fillrange = first.(ci), regci_kwargs...)
        plot!(x, predict(m, x); reg_kwargs...)
    end
    plot!(; plot_kwargs...)
end