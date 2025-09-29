# Gaussian Process Models

Gaussian processes (GP) are a useful model for random signals where a given level of correlation is expected. In TABASCAL, GPs are used extensively for signal modelling. This ranges from positional deviations from a fiducial trajectory, to complex-valued signals on antennas or baselines.

A GP is fully defined through its mean and covariance functions. Given no prior knowledge of the actual signal value and only its covariance, a prior distribution over the signal is easily defined as

$$\boldsymbol{y} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{C}),$$

where $\boldsymbol{y}$ is the signal, $\mathcal{N}$ represents the Normal/Gaussian distribution, $\boldsymbol{0}$ is the zero vector/function, and $\boldsymbol{C}$ is the covariance matrix/function.

In TABASCAL, our forward model is, in general, a non-linear model. Therefore the standard analytical form to get the posterior distrbution over the signal(s) cannot be used. The standard form can be seen as a form of probabilistic interpolation though. Therefore, we can use the estimator for the mean to interpolate from a coarse grain solution to a required sampling resolution. This is useful for reducing the parameter space and also the correlation between parameters.  

<!-- For reference, this would be

$$\boldsymbol{y}(t_b) \sim \mathcal{N} \left( \boldsymbol{C}_{t_bt_a} \boldsymbol{C}_{t_at_a}^{-1} \boldsymbol{y}(t_a),  \boldsymbol{C}_{t_bt_b} - \boldsymbol{C}_{t_bt_a} \boldsymbol{C}_{t_at_a}^{-1} \boldsymbol{C}_{t_bt_a} \right)$$ -->

## Signal-domain models

When modelling the our signal in its own space, i.e. the signal domain, we can use a select number of locations, $\boldsymbol{t}_a$, where we fit for the signal values, $\boldsymbol{y}_a$. From these, we can use the standard GP mean estimator to interpolate the signal to the required locations $\boldsymbol{t}_b$ and get interpolated signal values, $\boldsymbol{y}_b$. The interpolation formula is

$$\boldsymbol{y}_b = \boldsymbol{C}_{ba} \boldsymbol{C}_{aa}^{-1} \boldsymbol{y}_a,$$

where $\boldsymbol{C}_{aa}$ is the covariance matrix formed by evaluating the covariance function between all combinations of positions in $\boldsymbol{t}_a$ and $\boldsymbol{C}_{ba}$ is the covariance matrix formed by evaluating the covariance function between all combinations of the locations $\boldsymbol{t}_b$ and $\boldsymbol{t}_a$.

## Fourier-domain models

For most cases the covariance function used to model the signal is kept stationary. This means that the same covariance function does not change depending on the location and is only dependent on the distance between locations. For this case, The signal can be modelled in the Fourier-domain more efficiently. The covariance function of the signal is then given by its power spectrum. Additionally, the covariance matrix is now diagonal in this space with the power spectrum representing the variance.