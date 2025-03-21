module ConjugateModels

include("./gaussianunknownmeanunknownprecision.jl")


precompile(GaussianUnknownMeanUnknownPrecision, (Float64, Float64, Float64, Float64))
precompile(evaluate_likelihood, (Float64, GaussianUnknownMeanUnknownPrecision))
precompile(update_runLength_hyperparameters!, (Float64, GaussianUnknownMeanUnknownPrecision))



export reset_hyperparameters!,
       update_runLength_hyperparameters!,
       evaluate_likelihood

export AbstractConjugateModel,
       GaussianUnknownMeanUnknownPrecision

end