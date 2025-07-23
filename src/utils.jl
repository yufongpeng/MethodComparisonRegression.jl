
"""
    const_weight(x, y) = 1

Constant weight function
"""
const_weight(x, y) = 1.0

"""
    WFN::Dict{String, Function}

The default dictionary that maps names of weight function to the actual functions
"""
const WFN = Dict{String, Function}(
    "1"     => const_weight,
    "1/√x"  => (x, y) -> 1/sqrt(x),
    "1/x"   => (x, y) -> 1/x,
    "1/x²"  => (x, y) -> 1/x^2,
    "1/√y"  => (x, y) -> 1/sqrt(y),
    "1/y"   => (x, y) -> 1/y,
    "1/y²"  => (x, y) -> 1/y^2,
    "1/√(x+y)"  => (x, y) -> 1/sqrt(x+y),
    "1/(x+y)"   => (x, y) -> 1/(x+y),
    "1/(x+y)²"  => (x, y) -> 1/(x+y)^2
)

getwts(wfn, x, y) = map(wfn, x, y)
getwts(wfn::Nothing, x, y) = similar(y, 0)

"""
    @ri_str -> Interval{Float64}

Create a `Interval{Float64}` by mathematical real interval notation.

# Examples
```julia
julia> r = ri"[4, 7)"
Interval{Float64, Closed, Open}(4.0, 7.0)

julia> 4 in r
true

julia> 7 in r
false
```
"""
macro ri_str(expr)
    return parse(Interval{Float64}, expr)
end

"""
    Range

A collection of reasonable x and y ranges. Each fields are namedtuple with two attributes `x` and `y`.

# Fields
* `practical`: interval for x and y that the values possibly occur under practical condition, i.e. instrument, analytical methods, etc.
* `theoretical`: interval for x and y that the values are valid in the physical world.
"""
struct Range
    practical::@NamedTuple{x::Interval, y::Interval}
    theoretical::@NamedTuple{x::Interval, y::Interval}
end

# practical_xrange = ri"(0, Inf)", practical_yrange = ri"(0, Inf)", theoretical_xrange = ri"[0, Inf)", theoretical_yrange = ri"[0, Inf)",
