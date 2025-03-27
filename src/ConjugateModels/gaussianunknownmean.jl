import Distributions as ds

"""
  Likelihood: Normal with unknown μ and known variance σ²
  
  Prior: Normal
  
  Posterior predictive (marginal distribution): Normal
  
  μ::Float64 -> models the unknown mean: mean of the normal distribution 

  σ²::Float64 -> models the unknown mean: variance of the normal distribution

  σ²ₓ::Float64 -> models the data known variance
"""
mutable struct GaussianUnknownMean <: AbstractConjugateModel
  μ₀::Float64
  σ²₀::Float64
  σ²ₓ::Float64

  μₜ₊₁::Vector{Float64}
  σ²ₜ₊₁::Vector{Float64}

  function GaussianUnknownMean(;μ::Float64, σ²::Float64, σ²ₓ::Float64)
      new(μ, σ², σ²ₓ, [μ], [σ²])
  end
end

function reset_hyperparameters!(hyperparameter::GaussianUnknownMean)
  hyperparameter.μₜ₊₁ = copy(hyperparameter.μ₀)
  hyperparameter.σ²ₜ₊₁ = copy(hyperparameter.σ²₀)  
end

"""
Update all run length hypotheses based on new data at time t.
"""
function update_runLength_hyperparameters!(x::Float64, hyperparameter::GaussianUnknownMean)

  # Please refer to "Conjugate Bayesian analysis of the Gaussian distribution" article from
  # "https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf", Section 3 pages 6-10., eq. (20, 24)

  σ²ₜ₊₁::Vector{Float64} = vcat(hyperparameter.σ²₀, (1 ./ ( (1 ./ hyperparameter.σ²ₜ₊₁) .+ (1 / hyperparameter.σ²ₓ))))
  μₜ₊₁::Vector{Float64} = vcat(hyperparameter.μ₀, hyperparameter.σ²ₜ₊₁ .* ((hyperparameter.μ₀ / hyperparameter.σ²₀) .+ (x ./ hyperparameter.σ²ₓ)))

  hyperparameter.μₜ₊₁ = μₜ₊₁
  hyperparameter.σ²ₜ₊₁ = σ²ₜ₊₁
end

function evaluate_likelihood(x::Float64, hyperparameter::GaussianUnknownMean)::Vector{Float64}
  #TODO: make this 

  μ =  hyperparameter.μₜ₊₁
  σ = sqrt.(hyperparameter.σ²ₜ₊₁) 

  # Normal distribution evaluated at the "x" datum.
  πₜ = ds.pdf.(ds.Normal.(μ, σ), x)

  return πₜ
end
