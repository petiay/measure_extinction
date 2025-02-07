.. _model_capabilities:

==================
Model Capabilities
==================

The `MEModel` is specific to the needs of fitting observed data with a 
dust extinguished model.  This model provides all the model parameters,
functions to predict the SED based on the input model grid, and functions
to fit the data (minimizer and sampler), and plotting functions.

Parameters
**********

Each parameter is gives as a `MEParameter` object and each has the following properties.

* `value`: The value
* `unc`: Uncertainty in the fitting parameter
* `bounds`: The min/max value allowed (default=[None, None])
* `pprior`: The Gaussian prior as [mean, sigma] (default=None)
* `fixed`: True/False if the parameter is fixed in the fitting (default=False)

The parameters in the model are listed below.  Some have priors by default, some are 
fixed by default, all have default bounds set (expect `norm`).

* Stellar

    * `velocity`: velocity of the star in km/s
    * `logTeff`: log10 of the surface effective temperature
    * `logg`: log10 of the surface gravity
    * `logZ`: log10 of the stellar metallicity divided by the solar metallicity
    * `vturb`: stellar atmosphere turbulent velocity 
    * `windamp`: wind amplitude, only affects 1 micron and longer
    * `windalpha`: powerlaw of the wind emission

* Dust

    * `A(V)`: dust column as the V band extinction
    * `R(V)`: average dust size as R(V)=A(V)/E(B-V)
    * `C2`: slope value of FM90 parameterization
    * `B3`: 2175 A bump amplitude of FM90 parameterization
    * `C4`: fuv-rise amplitude of FM90 parameterization
    * `xo`: 2175 A bump center wavelength in inverse microns
    * `gamma`: 2175 A bump width in inverse microns

* Gas

    * `vel_MW`: velocity in km/s of the MW Ly-alpha absorption
    * `logHI_MW`: log10 of the column density of MW HI Ly-alpha absorption
    * `vel_exgal`: velocity in km/s of the external galaxy Ly-alpha absorption
    * `log10HI_exgal`: log10 of the column density of external galaxy HI Ly-alpha absorption

* Fitting

    * `norm`: The model normalization that matches the observed data on average.

Other info
**********

In addition to the above parameters, `MEModel` also includes ancillary info.

* `paramnames`: names of the parameters as a list

* `g23_all_ext`: Instead of the usual FM90 for the UV and G23 for the longer wavelengths, 
  use G23 for all wavelengths.  (default=False)

* `approx_band_ext`: Approximate the band extinction using the pivot wavelength of the band
  instead of applying the extinction to a low resolution model spectrum and then integrating
  over the band response functions. (default=False)

* `exclude_regions`: Regions to exclude from the fitting.  Given as a list with 
  elements that are 2 element lists of inverse wavelengths with astropy units of inverse microns.

* `lnp_bignum`: Value of a very low probability.  Not all fitters handle `-np.inf` well. 
  (default=-np.inf)

Functions
*********

There are `MEModel` functions that provide calculation of the model data, access to parameters,
plotting, and fitting.

* General

    * `parameters()`: parameter values as a numpy array
    * `parameters_to_fit()`: non-fixed parameter values as a numpy array
    * `fit_to_parameters(fit_params)`: non-fixed parameters to `MEModel` parameter values
    * `pprint_paramters()`: Print parameter names, values, and (if present) uncertainties to
       the screen

* Model SED

    * `stellar_sed(moddata)`: Compute the stellar SED using the model grid and the stellar
      parameters.
    * `dust_extinguished_sed(moddata, sed)`: Compute the dust extinguished SED using the 
      sed returned from `stellar_sed`, the input model grid, and the dust parameters.
    * `hi_abs_sed(moddata, sed)`: Add the gas HI Ly-alpha absorption to the sed returned from
      `dust_extinguished_sed` using the model data and the gas parameters.

* Fitting

    * `check_param_limits()`: Check that all the parameter values are within their bounds
    * `set_weights(obsdata)`: Set the weights based on the observed data uncertainties.  Set
       weights to zero in all exclude regions.  Weights are stored as the member dictonary 
       variable `weights`.
    * `set_initial_norm(obsdata, moddata)`: Set the `norm` value based on the model parameters
      and the observed data.
    * `lnlike()`: Compute the natural log of the likelihood assuming Gaussian uncertainties.
    * `lnprior()`: Compute the natural log of the priors.
    * `lnprob()`: Compute the natural log of the probability = sum of lnlike and lnprior.
    * `fit_minimizer()`: Determine the best fit by minimizing the -1 * lnprob value using 
      a scipy optimizer.  Returns a model with the best fit parameters.
    * `fit_sampler()`: Determine the nD probability function using the `emcee` Monte Carlo 
      Markov Chain sampler.  Returns a model with the p50 parameters and uncertainties based
      on the sample.  Also returns the samples and the sampler object.

* Plotting

    * `plot()`: Plot the data, stellar model, and extinguished stellar model in the top panel
      and the residuals to the fit in the bottom (smaller) panel.
    * `plot_sampler_chains()`: Plot the sampler chains in a multi-panel plot.
    * `plot_sampler_corner()`: Plot the 2D distributions of the samples using the starndard
      "corner" plot.
