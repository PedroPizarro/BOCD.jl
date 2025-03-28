module ConjugateModels

include("./abstractconjugatemodel.jl")
include("./gaussianunknownmeanunknownprecision.jl")
include("./gaussianunknownmean.jl")


precompile(GaussianUnknownMeanUnknownPrecision, (Float64, Float64, Float64, Float64))
precompile(evaluate_likelihood, (Float64, GaussianUnknownMeanUnknownPrecision))
precompile(update_runLength_hyperparameters!, (Float64, GaussianUnknownMeanUnknownPrecision))

precompile(GaussianUnknownMean, (Float64, Float64, Float64))
precompile(evaluate_likelihood, (Float64, GaussianUnknownMean))
precompile(update_runLength_hyperparameters!, (Float64, GaussianUnknownMean))


export reset_hyperparameters!,
       update_runLength_hyperparameters!,
       evaluate_likelihood

export AbstractConjugateModel,
       GaussianUnknownMeanUnknownPrecision,
       GaussianUnknownMean

end