# Bayesian Online Changepoint Detection (BOCD)

This is a Julia implementation of the algorithm from Adams & MacKay's 2007 paper, "Bayesian Online Changepoint Detection".

Arxiv link: https://arxiv.org/abs/0710.3742

![Status][status-image]
![Latest][latest-image]
![License][license-image]

## Usage
```julia
# Import package directly from git repo.
using Pkg; Pkg.add(url="https://github.com/PedroPizarro/BOCD.jl")
import BOCD as bocd

# Choose and initialize the Hazard distribution with its hyperparameter
# Look at HazardFunctions directory files for all existing Hazard models
hazardFunction = bocd.hazard.ConstantHazard(λ = 500.0)

# Choose and initialize the conjugate model with its hyperparameters. 
# Look at ConjugateModels directory files for all existing conjugate models
conjugateModel = bocd.conjugateModel.GaussianUnknownMeanUnknownPrecision(μ = 0.0, τ = 60.0, α = 10.0, β = 100.0)

# Creates the BOCD model with the defined Hazard and conjugate model distributions
bocdModel = bocd.model(hazardFunction, conjugateModel)

# Iteratively loops through your data
# Note: Change 'your data' to a valid data vector
for newDatum in your_data
   bocd.evaluate_datum!(bocdModel, newDatum)
   bocd.evaluate_possibleChangepoints(bocdModel)
end

# Find the indices in your data where a changepoint has occurred, given your data and previously defined model parameters.
indices = bocd.get_changepoints(bocdModel)
```

## Contributing and extending the implementation
If there is the need to add another conjugate model, the user has to:
- Create a new file in the `ConjugateModels` directory, naming the file after the conjugate model.
- Implement the `AbstractConjugateModel` methods for the new conjugate model
- Include the file in `ConjugateModes.jl`, add the `precompile()` directive for its functions, and export them.


For Hazard distributions, the process is similar, but it applies to the `HazardFunctions` directory and its corresponding module files.


Please be sure to validate your new added conjugate model implementation and Hazard distribution, along with citing an appropriate reference. Once validated, you may proceed with committing the changes and submitting a pull request (PR) to the `develop` branch.


## Acknowledgments
This code was heavily inspired by other BOCD implementations in Python:
- https://github.com/y-bar/bocd
- https://github.com/gwgundersen/bocd


[status-image]: https://img.shields.io/badge/status-Active-brightgreen?style=flat
[latest-image]: https://img.shields.io/badge/release-0.2.0-blue?style=flat
[license-image]: https://img.shields.io/badge/license-LGPLv3-lightgrey?style=flat
