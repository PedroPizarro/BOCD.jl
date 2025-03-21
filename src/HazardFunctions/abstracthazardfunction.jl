abstract type AbstractHazardFunction end

function evaluate_hazard(::AbstractHazardFunction)
  throw(NotImplementedError())
end