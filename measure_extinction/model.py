import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import G23
from dust_extinction.shapes import _curve_F99_method


class MEParameter(object):
    """
    Provide parameter info in a flexible format.
    Inspired by astropy modeling.
    """
    def __init__(self, value=0.0, bounds=(False, False), prior=None, fixed=False):
        self.value = value
        # square bounds supported by giving (min, max) as the bounds
        self.bounds = bounds
        # currently only Gaussian priors supports by giving (mean, sigma) as the prior
        self.prior = prior
        self.fixed = fixed


class MEModel(object):
    """
    Model
        Provide all the parameters for the measure_extinction fitting.
    Inspired by astropy modeling.
    """

    # fmt: off
    paramnames = ["logTeff", "logg", "logZ", "velocity",
                  "Av", "Rv", "C2", "B3", "C4", "xo", "gamma",
                  "vel_MW", "logHI_MW", "vel_exgal", "logHI_exgal"]
    # fmt: on
    nparams = len(paramnames)

    # stellar
    logTeff = MEParameter(value=4.0, bounds=(0.0, 10.0))
    logg = MEParameter(value=3.0, bounds=(0.0, 10.0))
    logZ = MEParameter(value=0.0, bounds=(-1.0, 1.0))
    velocity = MEParameter(value=0.0, bounds=[-1000.0, 1000.0], fixed=True)  # km/s

    # dust - values, bounds, and priors based on VCG04 and FM07 MW samples (expect Av)
    Av = MEParameter(value=0.5, bounds=(0.0, 100.0))
    Rv = MEParameter(value=3.0, bounds=(2.0, 6.0), prior=(3.0, 0.4))
    C2 = MEParameter(value=0.73, bounds=(-0.1, 5.0), prior=(0.73, 0.25))
    B3 = MEParameter(value=3.6, bounds=(-1.0, 8.0), prior=(3.6, 0.6))
    C4 = MEParameter(value=0.4, bounds=(-0.5, 1.5), prior=(0.4, 0.2))
    xo = MEParameter(value=4.59, bounds=(4.5, 4.9), prior=(4.59, 0.02))
    gamma = MEParameter(value=0.89, bounds=(0.4, 1.7), prior=(0.89, 0.08))

    # gas
    vel_MW = MEParameter(value=0.0, bounds=(-300.0, 300.0))  # km/s
    logHI_MW = MEParameter(value=20.0, bounds=(16.0, 24.0))
    vel_exgal = MEParameter(value=0.0, bounds=(-300.0, 1000.0), fixed=True)  # km/s
    logHI_exgal = MEParameter(value=16.0, bounds=(16.0, 24.0), fixed=True)

    #  bad regions are defined as those were we know the models do not work
    #  or the data is bad
    exclude_regions = [
        [8.23 - 0.1, 8.23 + 0.1],  # geocoronal line
        [8.7, 10.0],  # bad data from STIS
        [3.55, 3.6],
        [3.80, 3.90],
        [4.15, 4.3],
        [6.4, 6.6],
        [7.1, 7.3],
        [7.45, 7.55],
        [7.65, 7.75],
        [7.9, 7.95],
        [8.05, 8.1],
    ] / u.micron

    # some fitters don't like inf, can be changed here
    lnp_bignnum = -np.inf

    def __init__(self, modinfo=None):
        """
        Initialize the object, optionally using the min/max of the input model info
        to set the value and bounds on the stellar parameters

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra
        """
        if modinfo is not None:
            self.logTeff.bounds = (modinfo.temps_min, modinfo.temps_max)
            self.logTeff.value = np.average(self.logTeff.bounds)
            self.logg.bounds = (modinfo.gravs_min, modinfo.gravs_max)
            self.logg.value = np.average(self.logg.bounds)
            self.logZ.bounds = (modinfo.mets_min, modinfo.mets_max)
            self.logZ.value = np.average(self.logZ.bounds)

    def parameters(self):
        """
        Give all the parameters values in a vector (fixed or not).

        Returns
        -------
        params : np array
            parameters values
        """
        vals = []
        for cname in self.paramnames:
            vals.append(getattr(self, cname).value)
        return np.array(vals)

    def parameters_to_fit(self):
        """
        Give the non-fixed parameters values in a vector.  Needed for most fitters/samplers.

        Returns
        -------
        params : np array
            non-fixed parameters values
        """
        vals = []
        for cname in self.paramnames:
            if not getattr(self, cname).fixed:
                vals.append(getattr(self, cname).value)
        return np.array(vals)

    def fit_to_parameters(self, fit_params):
        """
        Set the parameter values based on a vector of the non-fixed values.
        Needed for most fitters/samplers.

        Parameters
        ----------
        fit_params : np array
            non-fixed parameters values
        """
        i = 0
        for cname in self.paramnames:
            if not getattr(self, cname).fixed:
                setattr(self, cname, fit_params[i])
                i += 1

    def check_param_limits(self):
        """
        Check the parameters are within the parameter bounds
        """
        for cname in self.paramnames:
            pval = getattr(self, cname).value
            pbounds = getattr(self, cname).bounds
            if (pval < pbounds[0]) | (pval > pbounds[1]):
                raise ValueError(
                    f"{cname} = {pval} is outside of the bounds ({pbounds[0]}, {pbounds[1]})"
                )

    def fit_weights(self, obsdata):
        """
        Compute the weight to be used for fitting.
        Observed data for the base weights 1/unc (expected by fitters).
        Weights in regions known to have data issues or that the models do not include
        are set to zero (e.g., stellar wind lines, geocoronal Ly-alpha)

        Parameters
        ----------
        obsdata : StarData object
            observed data for a reddened star
        """
        self.weights = {}
        for cspec in list(obsdata.data.keys()):
            # base weights
            self.weights[cspec] = np.full(len(obsdata.data[cspec].fluxes), 0.0)
            gvals = obsdata.data[cspec].npts > 0
            self.weights[cspec][gvals] = 1.0 / obsdata.data[cspec].uncs[gvals].value

            x = 1.0 / obsdata.data[cspec].waves
            for cexreg in self.exclude_regions:
                self.weights[cspec][
                    np.logical_and(x >= cexreg[0], x <= cexreg[1])
                ] = 0.0

    def stellar_sed(self, moddata):
        """
        Compute the stellar SED from the model parameters

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra

        Returns
        -------
        sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        # compute the distance between model params and grid points
        #    probably a better way using a kdtree
        dist2 = (
            (self.logTeff.value - moddata.temps) ** 2 / moddata.temps_width2
            + (self.logg.value - moddata.gravs) ** 2 / moddata.gravs_width2
            + (self.logZ.value - moddata.mets) ** 2 / moddata.mets_width2
        )
        sindxs = np.argsort(dist2)
        gsindxs = sindxs[0 : moddata.n_nearest]

        # generate model SED form nearest neighbors
        #   should handle the case where dist2 has an element that is zero
        #   i.e., one of the precomputed models exactly matches the request
        if np.sum(dist2[gsindxs]) > 0:
            weights = 1.0 / np.sqrt(dist2[gsindxs])
        else:
            weights = np.full(len(gsindxs), 1.0)
        weights /= np.sum(weights)

        sed = {}
        for cspec in moddata.fluxes.keys():
            # dot product does the multiplication and sum
            sed[cspec] = np.dot(weights, moddata.fluxes[cspec][gsindxs, :])

            sed[cspec][sed[cspec] == 0] = np.nan
            # shift spectrum if velocity given
            if self.velocity is not None:
                cwaves = moddata.waves[cspec]
                sed[cspec] = np.interp(
                    cwaves, (1.0 + self.velocity.value / 2.998e5) * cwaves, sed[cspec]
                )

        return sed

    def dust_extinguished_sed(self, moddata, sed, fit_range="all"):
        """
        Dust extinguished sed given the extinction parameters

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra

        sed : dict
            fluxes for each spectral piece

        fit_range : string, optional
            keyword to toggle SED fitting to be done with G23 only or
            to also include curve_F99_method

        Returns
        -------
        extinguished sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        g23mod = G23(Rv=self.Rv.value)

        # create the extinguished sed
        ext_sed = {}
        if fit_range.lower() == "g23":
            for cspec in moddata.fluxes.keys():
                shifted_waves = (1.0 - self.velocity.value / 2.998e5) * self.waves[
                    cspec
                ]
                axav = g23mod(shifted_waves)
                ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * self.Av))

        elif fit_range.lower() == "all":
            optnir_axav_x = np.flip(1.0 / (np.arange(0.35, 30.0, 0.1) * u.micron))
            optnir_axav_y = g23mod(optnir_axav_x)

            # updated F04 C1-C2 correlation
            C1 = 2.18 - 2.91 * self.C2.value

            for cspec in moddata.fluxes.keys():
                # get the dust extinguished SED (account for the
                #  systemic velocity of the galaxy [opposite regular sense])
                shifted_waves = (1.0 - self.velocity.value / 2.998e5) * moddata.waves[
                    cspec
                ]

                # convert to 1/micron as _curve_F99_method does not do this (as of Nov 2024)
                with u.add_enabled_equivalencies(u.spectral()):
                    shifted_waves_imicron = u.Quantity(
                        shifted_waves, 1.0 / u.micron, dtype=np.float64
                    )

                axav = _curve_F99_method(
                    shifted_waves_imicron.value,
                    self.Rv.value,
                    C1,
                    self.C2.value,
                    self.B3.value,
                    self.C4.value,
                    xo=self.xo.value,
                    gamma=self.gamma.value,
                    optnir_axav_x=optnir_axav_x.value,
                    optnir_axav_y=optnir_axav_y,
                    fm90_version="B3",
                )

                ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * self.Av.value))
        else:
            raise ValueError(
                "Incorrect input for fit_range argument in dust_extinguished_sed(). Available options are: g23, all"
            )

        # update the BAND fluxes by integrating the reddened MODEL_FULL spectrum
        if "BAND" in moddata.fluxes.keys():
            band_sed = np.zeros(moddata.n_bands)
            for k, cband in enumerate(moddata.band_names):
                gvals = np.isfinite(ext_sed["MODEL_FULL_LOWRES"])
                iwave = (1.0 - self.velocity.value / 2.998e5) * moddata.waves[
                    "MODEL_FULL_LOWRES"
                ][gvals]
                iflux = ext_sed["MODEL_FULL_LOWRES"][gvals]
                iresp = moddata.band_resp[cband](iwave)
                inttop = np.trapezoid(iwave * iresp * iflux, iwave)
                intbot = np.trapezoid(iwave * iresp, iwave)
                band_sed[k] = inttop / intbot
            ext_sed["BAND"] = band_sed

        return ext_sed

    def hi_abs_sed(self, moddata, sed):
        """
        HI abs sed given the HI columns

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra

        sed : dict
            fluxes for each spectral piece

        Returns
        -------
        hi absorbed sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        # wavelengths of HI lines
        #     only use Ly-alpha right now - others useful later
        h_lines = (
            np.array(
                [
                    1215.0,
                    1025.0,
                    972.0,
                    949.0,
                    937.0,
                    930.0,
                    926.0,
                    923.0,
                    920,
                    919.0,
                    918.0,
                ]
            )
            * u.angstrom
        )
        # width overwhich to compute the HI abs
        h_width = 100.0 * u.angstrom

        hi_sed = {}
        for cspec in moddata.fluxes.keys():
            hi_sed[cspec] = np.copy(sed[cspec])
            (indxs,) = np.where(
                np.absolute((moddata.waves[cspec] - h_lines[0]) <= h_width)
            )
            if len(indxs) > 0:
                logHI_vals = [self.logHI_MW.value, self.logHI_exgal.value]
                for i, cvel in enumerate([self.vel_MW.value, self.vel_exgal.value]):
                    # compute the Ly-alpha abs: from Bohlin et al. (197?)
                    abs_wave = (1.0 + (cvel / 3e5)) * h_lines[0].to(u.micron).value
                    phi = 4.26e-20 / (
                        (
                            1e4
                            * (
                                moddata.waves[cspec][indxs].to(u.micron).value
                                - abs_wave
                            )
                        )
                        ** 2
                        + 6.04e-10
                    )

                    nhi = 10 ** logHI_vals[i]
                    hi_sed[cspec][indxs] = hi_sed[cspec][indxs] * np.exp(
                        -1.0 * nhi * phi
                    )

        return hi_sed

    def lnlike(self, obsdata, modeldata, fit_range="all"):
        """
        Compute the natural log of the likelihood that the data
        fits the model.

        Parameters
        ----------
        obsdata : StarData object
            observed data for a reddened star

        moddata : ModelData object
            all the information about the model spectra

        fit_range : string, optional
            keyword to toggle SED fitting to be done with G23 only or
            to also include curve_F99_method

        Returns
        -------
        lnp : float
            natural log of the likelihood
        """
        # intrinsic sed
        modsed = self.stellar_sed(modeldata)

        # dust extinguished sed
        ext_modsed = self.dust_extinguished_sed(modeldata, modsed, fit_range=fit_range)

        # hi absorbed (ly-alpha) sed
        hi_ext_modsed = self.hi_abs_sed(modeldata, ext_modsed)

        # compute the normalization factors for the model and observed data
        # model data normalized to the observations using the ratio
        #   weighted average of the averages of each type of data (photometry or specific spectrum)
        #   allows for all the data to contribute to the normalization
        #   weighting by number of points in each type of data to achieve the highest S/N in
        #     the normalization
        norm_mod = []
        norm_dat = []
        norm_npts = []
        for cspec in obsdata.data.keys():
            gvals = (self.weights[cspec] > 0) & (np.isfinite(hi_ext_modsed[cspec]))
            norm_npts.append(np.sum(gvals))
            norm_mod.append(np.average(hi_ext_modsed[cspec][gvals]))
            norm_dat.append(np.average(obsdata.data[cspec].fluxes[gvals].value))
        norm_model = np.average(norm_mod, weights=norm_npts)
        norm_data = np.average(norm_dat, weights=norm_npts)

        lnl = 0.0
        for cspec in obsdata.data.keys():
            try:
                gvals = (self.weights[cspec] > 0) & (np.isfinite(hi_ext_modsed[cspec]))
            except ValueError:
                raise ValueError(
                    "Oops! The model data and reddened star data did not match.\n Hint: Make sure that the BAND name in the .dat files match."
                )
            chiarr = np.square(
                (
                    (
                        obsdata.data[cspec].fluxes[gvals].value
                        - (hi_ext_modsed[cspec][gvals] * (norm_data / norm_model))
                    )
                    * self.weights[cspec][gvals]
                )
            )
            lnl += -0.5 * np.sum(chiarr)

        return lnl

    def lnprior(self):
        """
        Compute the natural log of the priors.
        Only Gaussian priors currently supported.

        Returns
        -------
        lnp : float
            natural log of the prior
        """
        # make sure the parameters are within the limits
        # and compute the ln(prior)
        lnp = 0.0
        for cname in self.paramnames:
            param = getattr(self, cname)
            if not param.fixed:
                pval = param.value
                pbounds = param.bounds
                pprior = param.prior
                if (pval < pbounds[0]) | (pval > pbounds[1]):
                    return self.lnp_bignum
                if pprior is not None:
                    lnp += -0.5 * ((pval - pprior[0]) / pprior[1]) ** 2

        return lnp

    def lnprob(self, obsdata, modeldata, fit_range="all"):
        """
        Compute the natural log of the probability

        Parameters
        ----------
        obsdata : StarData object
            observed data for a reddened star

        moddata : ModelData object
            all the information about the model spectra
        """
        lnp = self.lnprior()
        if lnp == self.lnp_bignnum:
            return lnp
        else:
            return lnp + self.lnlike(obsdata, modeldata, fit_range=fit_range)
