import Distributions as ds

"""
  Likelihood: Normal with unknown μ and known variance τ²
  
  Prior: Normal
  
  Posterior predictive (marginal distribution): Normal
  
  μ::Float64 -> models the unknown mean: mean of the normal distribution 

  τ²::Float64 -> models the unknown mean: precision of the normal distribution

  τ²ₓ::Float64 -> models the data known precision
"""
mutable struct NormalUnknownMean <: AbstractConjugateModel
  μ₀::Float64
  τ²₀::Float64
  τ²ₓ::Float64

  μₜ₊₁::Vector{Float64}
  τ²ₜ₊₁::Vector{Float64}

  function NormalUnknownMean(;μ::Float64, τ²::Float64, τ²ₓ::Float64)
      new(μ, τ², τ²ₓ, [μ], [τ²])
  end
end

function reset_hyperparameters!(hyperparameter::NormalUnknownMean)
  hyperparameter.μₜ₊₁ = copy(hyperparameter.μ₀)
  hyperparameter.τ²ₜ₊₁ = copy(hyperparameter.τ²₀)  
end

"""
Update all run length hypotheses based on new data at time t.
"""
function update_runLength_hyperparameters!(x::Float64, hyperparameter::NormalUnknownMean)

  # Please refer to "Conjugate Bayesian analysis of the Gaussian distribution" article from
  # "https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf", Section 3 pages 6-10.

  τ²ₜ₊₁::Vector{Float64} = vcat(hyperparameter.τ²₀,  hyperparameter.τ²ₜ₊₁ .+ hyperparameter.τ²ₓ)
  μₜ₊₁::Vector{Float64} = vcat(hyperparameter.μ₀,  ((hyperparameter.μₜ₊₁ .* τ²ₜ₊₁[1:end-1]) .+ (x * hyperparameter.τ²ₓ)) ./ ( hyperparameter.τ²ₜ₊₁ .+ hyperparameter.τ²ₓ))
  
  hyperparameter.τ²ₜ₊₁ =  τ²ₜ₊₁
  hyperparameter.μₜ₊₁ = μₜ₊₁
  println("hyperparameter.τ²ₜ₊₁[end]: $(hyperparameter.τ²ₜ₊₁[end])\n")
  println("hyperparameter.μₜ₊₁[end]: $(hyperparameter.μₜ₊₁[end])\n")
end

function evaluate_likelihood(x::Float64, hyperparameter::NormalUnknownMean)::Vector{Float64}
  μ =  hyperparameter.μₜ₊₁
  σ = sqrt.((1 ./hyperparameter.τ²ₜ₊₁) .+ (1 /hyperparameter.τ²ₓ)) 

  println("μ: $(μ)\n")
  println("σ: $(σ)\n")
  println("ds.logpdf.(ds.Normal.(μ, σ),x): $(ds.logpdf.(ds.Normal.(μ, σ),x))\n")

  # Normal distribution evaluated at the "x" datum.
  πₜ = ds.pdf.(ds.Normal.(μ, σ), x)

  return πₜ
end
