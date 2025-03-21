include("./abstracthazardfunction.jl")


"""
 λ::Float64 -> scale parameter. When λ is bigger, the probability of a changepoint happening is low. 
"""
struct ConstantHazard <: AbstractHazardFunction
  λ::Float64

  function ConstantHazard(;λ::Float64)::ConstantHazard
    new(λ)
  end
end

"""
 λ::Float64 -> scale parameter
"""
@inline function evaluate_hazard(hyperparameter::ConstantHazard)::Float64
  return 1/hyperparameter.λ
end