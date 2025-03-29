module ConjugateModels

include("./abstractconjugatemodel.jl")
include("./normalunknownmeanunknownprecision.jl")
include("./normalunknownmean.jl")
include("./lognormalunknownmean.jl")


precompile(NormalUnknownMeanUnknownPrecision, (Float64, Float64, Float64, Float64))
precompile(evaluate_likelihood, (Float64, NormalUnknownMeanUnknownPrecision))
precompile(update_runLength_hyperparameters!, (Float64, NormalUnknownMeanUnknownPrecision))

precompile(NormalUnknownMean, (Float64, Float64, Float64))
precompile(evaluate_likelihood, (Float64, NormalUnknownMean))
precompile(update_runLength_hyperparameters!, (Float64, NormalUnknownMean))

precompile(LogNormalUnknownMean, (Float64, Float64, Float64))
precompile(evaluate_likelihood, (Float64, LogNormalUnknownMean))
precompile(update_runLength_hyperparameters!, (Float64, LogNormalUnknownMean))


export reset_hyperparameters!,
       update_runLength_hyperparameters!,
       evaluate_likelihood

export AbstractConjugateModel,
       NormalUnknownMeanUnknownPrecision,
       NormalUnknownMean,
       LogNormalUnknownMean

end