# Gaussian Process Models

Gaussian processes (GP) are a useful model for random signals where a given level of correlation is expected. In TABASCAL, GPs are used extensively for signal modelling. This ranges from positional deviations from a fiducial trajectory, to complex-valued signals on antennas or baselines.

A GP is fully defined through its mean and covariance functions. Given no prior knowledge of the actual signal value and only its covariance, a prior distribution over the signal is easily defined as

$$\boldsymbol{y} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{C}),$$

where $\boldsymbol{y}$ is the signal, $\mathcal{N}$ represents the Normal/Gaussian distribution, $\boldsymbol{0}$ is the zero vector/function, and $\boldsymbol{C}$ is the covariance matrix/function.

In TABASCAL, our forward model is, in general, a non-linear model. Therefore the standard analytical form to get the posterior distrbution over the signal(s) cannot be used, and the GP enters as a prior that the fit has to carry rather than as something that can be conditioned on in closed form.

## Fourier-domain models

Every signal GP in TABASCAL is modelled in the Fourier domain. For most cases the covariance function used to model the signal is kept stationary. This means that the same covariance function does not change depending on the location and is only dependent on the distance between locations. For this case, the signal can be modelled in the Fourier-domain more efficiently. The covariance function of the signal is then given by its power spectrum. Additionally, the covariance matrix is now diagonal in this space with the power spectrum representing the variance.

That leaves the parameters uncorrelated under the prior, and it costs a fast Fourier transform per evaluation rather than a factorisation of a dense covariance matrix — which is also why nothing here builds one. The last model that did, the Gaussian process gain `gains:GPGains`, was removed in #129; the components that remain — {class}`~tabascal.components.rfi_signal.ComplexRFIVarAnt`, {class}`~tabascal.components.rfi_signal.ComplexRFIConstAnt` and {class}`~tabascal.components.ast_vis.GPVisAst` — are all Fourier-domain.
