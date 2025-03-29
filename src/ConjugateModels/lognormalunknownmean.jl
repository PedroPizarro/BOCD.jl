import Distributions as ds

"""
  Likelihood: Lognormal with unknown μ (mean of log-transformed data) and known precision τ² of log-transformed data
  
  Prior: Normal for μ
  
  Posterior predictive (marginal distribution): Normal
  
  μ::Float64 -> models the unknown mean: mean of the normal distribution 

  τ²::Float64 -> models the unknown mean: precision of the normal distribution

  τ²ₓ::Float64 -> models the data known precision
"""
mutable struct LogNormalUnknownMean <: AbstractConjugateModel
  μ₀::Float64
  τ²₀::Float64
  τ²ₓ::Float64

  μₜ₊₁::Vector{Float64}
  τ²ₜ₊₁::Vector{Float64}

  function LogNormalUnknownMean(;μ::Float64, τ²::Float64, τ²ₓ::Float64)
      new(μ, τ², τ²ₓ, [μ], [τ²])
  end
end

function reset_hyperparameters!(hyperparameter::LogNormalUnknownMean)
  hyperparameter.μₜ₊₁ = copy(hyperparameter.μ₀)
  hyperparameter.τ²ₜ₊₁ = copy(hyperparameter.τ²₀)  
end

"""
Update all run length hypotheses based on new data at time t.
"""
function update_runLength_hyperparameters!(x::Float64, hyperparameter::LogNormalUnknownMean)

  # Please refer to "Conjugate Bayesian analysis of the Gaussian distribution" article from
  # "https://www.johndcook.com/CompendiumOfConjugatePriors.pdf", Section 2.12 pages 21-22.

  y = log(x)

  τ²ₜ₊₁::Vector{Float64} = vcat(hyperparameter.τ²₀,  hyperparameter.τ²ₜ₊₁ .+ hyperparameter.τ²ₓ)
  μₜ₊₁::Vector{Float64} = vcat(hyperparameter.μ₀,  ((hyperparameter.μₜ₊₁ .* τ²ₜ₊₁[1:end-1]) .+ (y * hyperparameter.τ²ₓ)) ./ ( hyperparameter.τ²ₜ₊₁ .+ hyperparameter.τ²ₓ))
  
  hyperparameter.τ²ₜ₊₁ =  τ²ₜ₊₁
  hyperparameter.μₜ₊₁ = μₜ₊₁
end

function evaluate_likelihood(x::Float64, hyperparameter::LogNormalUnknownMean)::Vector{Float64}
  μ =  hyperparameter.μₜ₊₁
  σ = sqrt.((1 ./hyperparameter.τ²ₜ₊₁) .+ (1 /hyperparameter.τ²ₓ)) 


  # Normal distribution evaluated at the "x" datum.
  πₜ = ds.pdf.(ds.LogNormal.(μ, σ), x)

  return πₜ
end
