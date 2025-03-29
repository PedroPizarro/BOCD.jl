import Distributions as ds

"""
  Likelihood: Normal with unknown μ and unknown precision τ
  
  Prior: Normal-gamma
  
  Posterior predictive (marginal distribution): StudentT
  
  μ::Float64 -> models the unknown mean: mean of the normal distribution

  τ::Float64 -> models the unknown mean: precision of the normal distribution

  α::Float64 -> models the unknown precision: shape parameter of the gamma distribution

  β::Float64 -> models the unknown precision: rate parameter of the gamma distribution
"""
mutable struct NormalUnknownMeanUnknownPrecision <: AbstractConjugateModel
  μ₀::Float64
  τ₀::Float64
  α₀::Float64 
  β₀::Float64

  μₜ₊₁::Vector{Float64}
  τₜ₊₁::Vector{Float64}
  αₜ₊₁::Vector{Float64}
  βₜ₊₁::Vector{Float64}

  function NormalUnknownMeanUnknownPrecision(;μ::Float64, τ::Float64, α::Float64, β::Float64)
      new(μ, τ, α, β, [μ], [τ], [α], [β])
  end
end

function reset_hyperparameters!(hyperparameter::NormalUnknownMeanUnknownPrecision)
  hyperparameter.μₜ₊₁ = copy(hyperparameter.μ₀)
  hyperparameter.τₜ₊₁ = copy(hyperparameter.τ₀)  
  hyperparameter.αₜ₊₁ = copy(hyperparameter.α₀)
  hyperparameter.βₜ₊₁ = copy(hyperparameter.β₀)
end

"""
Update all run length hypotheses based on new data at time t.
"""
function update_runLength_hyperparameters!(x::Float64, hyperparameter::NormalUnknownMeanUnknownPrecision)

  # Please refer to "Conjugate Bayesian analysis of the Gaussian distribution" article from
  # "https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf", Section 3 pages 6-10., eq. (104,86,102,101)

  βₜ₊₁::Vector{Float64} = vcat(hyperparameter.β₀, hyperparameter.βₜ₊₁ .+ (hyperparameter.τₜ₊₁ .* (x .- hyperparameter.μₜ₊₁).^2) ./ (2 .* (hyperparameter.τₜ₊₁ .+ 1)))
  μₜ₊₁::Vector{Float64} = vcat(hyperparameter.μ₀, ((hyperparameter.τₜ₊₁ .* hyperparameter.μₜ₊₁) .+ x) ./ (hyperparameter.τₜ₊₁ .+ 1))  
  τₜ₊₁::Vector{Float64} = vcat(hyperparameter.τ₀, hyperparameter.τₜ₊₁ .+ 1)
  αₜ₊₁::Vector{Float64} = vcat(hyperparameter.α₀, hyperparameter.αₜ₊₁ .+ 0.5)

  hyperparameter.βₜ₊₁ = βₜ₊₁
  hyperparameter.μₜ₊₁ = μₜ₊₁
  hyperparameter.τₜ₊₁ = τₜ₊₁
  hyperparameter.αₜ₊₁ = αₜ₊₁
end

function evaluate_likelihood(x::Float64, hyperparameter::NormalUnknownMeanUnknownPrecision)::Vector{Float64}
  μ =  hyperparameter.μₜ₊₁
  σ = sqrt.(hyperparameter.βₜ₊₁ .* (hyperparameter.τₜ₊₁ .+ 1) ./ (hyperparameter.αₜ₊₁ .* hyperparameter.τₜ₊₁)) 

  # Dislocated and scaled Student-t distribution evaluated at the "x" datum.
  πₜ = ds.pdf.(ds.TDist.(2 .* hyperparameter.αₜ₊₁), ((x .- μ) ./ σ)) ./ σ

  return πₜ
end