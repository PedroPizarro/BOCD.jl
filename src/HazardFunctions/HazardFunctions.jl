module HazardFunctions

include("./constanthazard.jl")

precompile(ConstantHazard, (Float64,))
precompile(evaluate_hazard, (ConstantHazard,))

export evaluate_hazard

export AbstractHazardFunctions
       ConstantHazard

end