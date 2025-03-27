module BOCD

include("./ConjugateModels/ConjugateModels.jl")
import .ConjugateModels as conjugateModel

include("./HazardFunctions/HazardFunctions.jl")
import .HazardFunctions as hazard

mutable struct BayesianOnlineChangePointDetection
  hazard_function::hazard.AbstractHazardFunction
  distribution::conjugateModel.AbstractConjugateModel
  t::Int64
  beliefs::Matrix{Float64}
  runLength::Vector{Int64}

  function BayesianOnlineChangePointDetection(hazard::hazard.AbstractHazardFunction, distribution::conjugateModel.AbstractConjugateModel, currentRunLengthSize::Int64, firstDataChangePointPrior::Float64)::BayesianOnlineChangePointDetection
    return new(hazard, distribution, currentRunLengthSize, [firstDataChangePointPrior 0.0], [])
  end
end

"""
Considers that a changepoint occurred right before the first datum.

The run-length is 0 and the first data changepoint prior is 1.
"""
function model(hazard::hazard.AbstractHazardFunction, distribution::conjugateModel.AbstractConjugateModel)::BayesianOnlineChangePointDetection
  # Part 1: Initialize
  return BayesianOnlineChangePointDetection(hazard, distribution, 0, 1.0)
end

#TODO: implement the recursive algorithm inside the for loop
#TODO: implement the log probability
function evaluate_datum!(model::BayesianOnlineChangePointDetection, x::Float64)
  _expand_belief_matrix!(model)

  # Part 3: Evaluate Predictive Probability (πₜ in the algorithm)
  πₜ = conjugateModel.evaluate_likelihood(x, model.distribution)

  # Calculate Hazard (h) based on run length
  H = hazard.evaluate_hazard(model.hazard_function)

  # Part 4: Calculate Growth Probability (beliefs for change not happening)
  model.beliefs[2:model.t + 2, 2] .= model.beliefs[1:model.t + 1, 1] .* πₜ .* (1 - H)

  # Part 5: Calculate Changepoint Probabilities (beliefs for change happening)
  model.beliefs[1, 2] = sum(model.beliefs[1:model.t + 1, 1] .* πₜ * H)

  # Part 6: Calculate Evidence
  evidence = sum(model.beliefs[:, 2])

  # Normalize the beliefs (as in Algorithm 1)
  model.beliefs[:, 2] .= model.beliefs[:, 2] ./ evidence

  # Update distribution parameters
  conjugateModel.update_runLength_hyperparameters!(x, model.distribution)

  # Update internal state
  _shift_belief_matrix!(model)

  # Increment time step (t)
  model.t += 1
  return nothing
end

function find_changepoints(model::BayesianOnlineChangePointDetection)
  max_val = maximum(model.beliefs[:,1])

  # Gets the index of the most probable (maximum belief) of rₜ
  append!(model.runLength, findall(x -> x == max_val, model.beliefs[:,1]))
end

function get_changepoints(model::BayesianOnlineChangePointDetection)::Vector{Int64}
  return findall(all.(x -> x < 0, diff(model.runLength)))
end

function _expand_belief_matrix!(model::BayesianOnlineChangePointDetection)
  model.beliefs = vcat(model.beliefs, [0.0 0.0])  # Append new row for the next step

  return nothing
end

function _shift_belief_matrix!(model::BayesianOnlineChangePointDetection)
  model.beliefs[:, 1] .= model.beliefs[:, 2]
  model.beliefs[:, 2] .= 0.0  # Reset the changepoint probabilities

  return nothing
end


precompile(BayesianOnlineChangePointDetection, (hazard.AbstractHazardFunction, conjugateModel.AbstractConjugateModel))
precompile(model, (hazard.AbstractHazardFunction, conjugateModel.AbstractConjugateModel))

precompile(evaluate_datum!, (BayesianOnlineChangePointDetection, Float64))
precompile(find_changepoints, (BayesianOnlineChangePointDetection, ))
precompile(get_changepoints, (BayesianOnlineChangePointDetection, ))


# Export module constructor and functions
export BayesianOnlineChangePointDetection,
model,
evaluate_datum!,
get_changepoints

# Export inner modules
export conjugateModel
export hazard

end
