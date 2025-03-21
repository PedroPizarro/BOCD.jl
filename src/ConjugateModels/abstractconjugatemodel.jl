abstract type AbstractConjugateModel end

function reset_hyperparameters!(::AbstractConjugateModel)
  throw(NotImplementedError())
end

function update_runLength_hyperparameters!(x::Float64, ::AbstractConjugateModel)
  throw(NotImplementedError())
end

function evaluate_likelihood(x::Float64, ::AbstractConjugateModel)
  throw(NotImplementedError())
end