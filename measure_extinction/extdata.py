from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import numpy as np
from astropy.io import fits
import astropy.units as u

from dust_extinction.parameter_averages import F04

__all__ = ["ExtData", "AverageExtData"]


# globals
# possible datasets (also extension names in saved FITS file)
_poss_datasources = ["BAND", "IUE", "FUSE", "STIS", "SpeX_SXD", "SpeX_LXD", "IRS"]


def _rebin(a, rebin_fac):
    """
    Hack code to rebin a 1d array
    """
    new_len = int(a.shape[0] / rebin_fac)
    new_a = np.full((new_len), 0.0)
    for i in range(new_len):
        new_a[i] = np.mean(a[i * rebin_fac : ((i + 1) * rebin_fac) - 1])
    return new_a


def _flux_unc_as_mags(fluxes, uncs):
    """
    Provide the flux uncertainties in magnitudes accounting for the
    case where (fluxes-uncs) is negative
    """
    uncs_mag = np.empty(len(fluxes))

    # fluxes-uncs case
    (indxs,) = np.where(fluxes - uncs <= 0)
    if len(indxs) > 0:
        uncs_mag[indxs] = -2.5 * np.log10(fluxes[indxs] / (fluxes[indxs] + uncs[indxs]))

    # normal case
    (indxs,) = np.where(fluxes - uncs > 0)
    if len(indxs) > 0:
        uncs_mag[indxs] = -2.5 * np.log10(
            (fluxes[indxs] - uncs[indxs]) / (fluxes[indxs] + uncs[indxs])
        )

    return uncs_mag


def _hierarch_keywords(names):
    """
    Prepend the 'HIERARCH ' string to all keywords > 8 characters
    Avoids FITS VerifyWarning.

    Parameters
    ----------
    names : list
        keywords

    Returns
    -------
    new_names : list
        keywords with HIERARCH prepended as apprpriate
    """
    new_names = []
    for cname in names:
        if len(cname) >= 8:
            new_names.append(f"HIERARCH {cname}")
        else:
            new_names.append(cname)

    return new_names


def AverageExtData(extdatas, alav=None):
    """
    Generate the average extinction curve from a list of ExtData objects
    """
    aveext = ExtData()
    if alav is None:
        aveext.type = extdatas[0].type
    else:
        aveext.type = "alav"

    gkeys = list(extdatas[0].waves.keys())
    gkeys.remove("BAND")
    for src in gkeys:
        aveext.waves[src] = extdatas[0].waves[src]
        n_waves = len(aveext.waves[src])
        aveext.exts[src] = np.zeros(n_waves)
        aveext.uncs[src] = np.zeros(n_waves)
        aveext.npts[src] = np.zeros(n_waves)
        for cext in extdatas:
            if src in cext.waves.keys():
                (gindxs,) = np.where(cext.npts[src] > 0)
                y = cext.exts[src][gindxs]
                yu = cext.uncs[src][gindxs]
                if alav is not None:
                    y = (y / float(cext.columns["AV"][0])) + 1.0
                    yu /= float(cext.columns["AV"][0])
                aveext.exts[src][gindxs] += y
                aveext.uncs[src][gindxs] += np.square(yu)
                aveext.npts[src][gindxs] += cext.npts[src][gindxs]
        (gindxs,) = np.where(aveext.npts[src] > 0)
        aveext.exts[src][gindxs] /= aveext.npts[src][gindxs]
        aveext.uncs[src][gindxs] /= aveext.npts[src][gindxs]
        aveext.uncs[src][gindxs] = (
            np.sqrt(aveext.uncs[src][gindxs]) / aveext.npts[src][gindxs]
        )

    # do the photometric band data separately
    src = "BAND"
    # possible band waves and number of curves with those waves
    pwaves = {}
    pwaves_ext = {}
    pwaves_uncs = {}
    pwaves_npts = {}
    pwaves_names = {}
    for cext in extdatas:
        for k, cwave in enumerate(cext.waves[src]):
            y = cext.exts[src][k]
            yu = cext.uncs[src][k]
            if alav is not None:
                y = (y / float(cext.columns["AV"][0])) + 1.0
                yu /= float(cext.columns["AV"][0])
            cwavev = cwave.to(u.micron).value
            cband = cext.names[src][k]
            if cband in pwaves.keys():
                pwaves_ext[cband] += y
                pwaves_uncs[cband] += np.square(yu)
                pwaves_npts[cband] += 1.0
            else:
                pwaves[cband] = cwavev
                pwaves_ext[cband] = y
                pwaves_uncs[cband] = np.square(yu)
                pwaves_npts[cband] = 1.0
                pwaves_names[cband] = cext.names[src][k]

    aveext.waves[src] = np.array(list(pwaves.values())) * u.micron
    aveext.exts[src] = np.array(list(pwaves_ext.values())) / np.array(
        list(pwaves_npts.values())
    )
    aveext.uncs[src] = np.sqrt(np.array(list(pwaves_uncs.values()))) / np.array(
        list(pwaves_npts.values())
    )
    aveext.npts[src] = np.array(list(pwaves_npts.values()))
    aveext.names[src] = np.array(list(pwaves_names))

    return aveext


class ExtData:
    """
    Extinction for a single line-of-sight

    Atributes:

    type : string
        extinction curve type (e.g., elx or alax)

    type_rel_band : string
        band name for relative extinction measurement (x in elx)

    red_file : string
        reddened star filename

    comp_file : string
        comparison star filename

    columns : list of tuples of column measurements
        measurements are A(V), R(V), N(HI), etc.
        tuples are measurement, uncertainty

    waves : dict of key:wavelengths
        key is BANDS, IUE, IRS, etc.

    ext : dict of key:E(lambda-v) measurements

    uncs : dict of key:E(lambda-v) measurement uncertainties

    npts : dict of key:number of measurements at each wavelength

    names : dict of key:names of names of each wavelength (if photometric bands)

    fm90 : list of FM90 parameters tuples
        tuples are measurement, uncertainty
    """

    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : string, optional [default=None]
            Full filename to a save extinction curve
        """
        self.type = ""
        self.type_rel_band = ""
        self.red_file = ""
        self.comp_file = ""
        self.columns = {}
        self.fm90 = {}
        self.waves = {}
        self.exts = {}
        self.uncs = {}
        self.npts = {}
        self.names = {}

        if filename is not None:
            self.read(filename)

    def calc_elx_bands(self, red, comp, rel_band="V"):
        """
        Calculate the E(lambda-X) for the photometric band data

        Separate from the spectral case as the bands in common must
        be found. In addition, some of the photometric observations are
        reported as colors (e.g., B-V) with uncertainties on those colors.
        As colors are what is needed for the extinction curve, we want to
        work in those colors to preserve the inheritly lower uncertainties.

        Parameters
        ----------
        red : :class:StarData
            Observed data for the reddened star

        comp : :class:StarData
            Observed data for the comparison star

        rel_band : string
            Band to use for relative extinction measurement
            default = "V"

        Returns
        -------
        updates self.(waves, exts, uncs, npts, names)['BANDS']
        """
        # reference band
        red_rel_band = red.data["BAND"].get_band_mag(rel_band)
        comp_rel_band = comp.data["BAND"].get_band_mag(rel_band)
        # possible bands for the band extinction curve
        poss_bands = red.data["BAND"].get_poss_bands()

        waves = []
        exts = []
        uncs = []
        npts = []
        names = []
        for pband_name in poss_bands.keys():
            red_mag = red.data["BAND"].get_band_mag(pband_name)
            comp_mag = comp.data["BAND"].get_band_mag(pband_name)
            if (red_mag is not None) & (comp_mag is not None):
                ext = (red_mag[0] - red_rel_band[0]) - (comp_mag[0] - comp_rel_band[0])
                unc = np.sqrt(
                    red_mag[1] ** 2
                    + red_rel_band[1] ** 2
                    + comp_mag[1] ** 2
                    + comp_rel_band[1] ** 2
                )
                waves.append(red.data["BAND"].band_waves[pband_name])
                exts.append(ext)
                uncs.append(unc)
                npts.append(1)
                names.append(pband_name)

        if len(waves) > 0:
            self.waves["BAND"] = np.array(waves) * u.micron
            self.exts["BAND"] = np.array(exts)
            self.uncs["BAND"] = np.array(uncs)
            self.npts["BAND"] = np.array(npts)
            self.names["BAND"] = np.array(names)

    def calc_elx_spectra(self, red, comp, src, rel_band="V"):
        """
        Calculate the E(lambda-X) for the spectroscopic data

        Parameters
        ----------
        red : :class:StarData
            Observed data for the reddened star

        star : :class:StarData
            Observed data for the comparison star

        src : string
            data source (see global _poss_datasources)

        rel_band : string
            Band to use for relative extinction measurement
            default = "V"

        Returns
        -------
        updates self.(waves, exts, uncs, npts)[src]
        """
        if (src in red.data.keys()) & (src in comp.data.keys()):
            # check that the wavelenth grids are identical
            delt_wave = red.data[src].waves - comp.data[src].waves
            if np.sum(np.absolute(delt_wave)) > 0.01 * u.micron:
                warnings.warn("wavelength grids not equal for %s" % src, UserWarning)
            else:
                # reference band
                red_rel_band = red.data["BAND"].get_band_mag(rel_band)
                comp_rel_band = comp.data["BAND"].get_band_mag(rel_band)

                # setup the needed variables
                self.waves[src] = red.data[src].waves
                n_waves = len(self.waves[src])
                self.exts[src] = np.zeros(n_waves)
                self.uncs[src] = np.zeros(n_waves)
                self.npts[src] = np.zeros(n_waves)

                # only compute the extinction for good, positive fluxes
                (indxs,) = np.where(
                    (red.data[src].npts > 0)
                    & (comp.data[src].npts > 0)
                    & (red.data[src].fluxes.value > 0)
                    & (comp.data[src].fluxes.value > 0)
                )
                self.exts[src][indxs] = -2.5 * np.log10(
                    red.data[src].fluxes[indxs] / comp.data[src].fluxes[indxs]
                ) + (comp_rel_band[0] - red_rel_band[0])
                self.uncs[src][indxs] = np.sqrt(
                    np.square(
                        _flux_unc_as_mags(
                            red.data[src].fluxes[indxs], red.data[src].uncs[indxs]
                        )
                    )
                    + np.square(
                        _flux_unc_as_mags(
                            comp.data[src].fluxes[indxs], comp.data[src].uncs[indxs]
                        )
                    )
                    + np.square(red_rel_band[1])
                    + np.square(comp_rel_band[1])
                )
                self.npts[src][indxs] = np.full(len(indxs), 1)

    def calc_elx(self, redstar, compstar, rel_band="V"):
        """
        Calculate the E(lambda-X) basic extinction measurement

        Parameters
        ----------
        redstar : :class:StarData
            Observed data for the reddened star

        compstar : :class:StarData
            Observed data for the comparison star

        rel_band : string
            Band to use for relative extinction measurement
            default = "V"

        Returns
        -------
        updates self.(waves, exts, uncs, npts, names)
        """
        self.type = "elx"
        self.type_rel_band = rel_band
        self.red_file = redstar.file
        self.comp_file = compstar.file
        for cursrc in _poss_datasources:
            if cursrc == "BAND":
                self.calc_elx_bands(redstar, compstar, rel_band=rel_band)
            else:
                self.calc_elx_spectra(redstar, compstar, cursrc, rel_band=rel_band)

    def trans_elv_elvebv(self):
        """
        Transform E(lambda-V) to E(lambda -V)/E(B-V) by
        normalizing by E(lambda-B).

        Parameters
        ----------

        Returns
        -------
        Updates self.(exts, uncs)
        """
        if self.type_rel_band != "V":
            warnings.warn("attempt to normalize a non-elv curve with ebv", UserWarning)
        else:
            # determine the index for the B band
            dwaves = np.absolute(self.waves["BAND"] - 0.438 * u.micron)
            sindxs = np.argsort(dwaves)
            bindx = sindxs[0]
            if dwaves[bindx] > 0.02 * u.micron:
                warnings.warn("no B band measurement in E(l-V)", UserWarning)
            else:
                # normalize each portion of the extinction curve
                ebv = self.exts["BAND"][bindx]
                for curname in self.exts.keys():
                    self.exts[curname] /= ebv
                    self.uncs[curname] /= ebv
                self.type = "elvebv"

    def trans_elv_alav(self, av=None, akav=0.112):
        """
        Transform E(lambda-V) to A(lambda)/A(V) by normalizing to
        A(V) and adding 1. Default is to calculate A(V) from the
        input elx curve. If A(V) value passed, use that instead.

        Parameters
        ----------
        av : float [default = None]
            value of A(V) to use - otherwise calculate it

        akav : float  [default = 0.112]
           Value of A(K)/A(V)
           default is from Rieke & Lebofsky (1985)
           van de Hulst No. 15 curve has A(K)/A(V) = 0.0885

        Returns
        -------
        Updates self.(exts, uncs)
        """
        if self.type_rel_band != "V":
            warnings.warn("attempt to normalize a non-E(lambda-V) curve with A(V)", UserWarning)
        else:
            if av is None:
                # compute A(V) from E(K-V)
                dwaves = np.absolute(self.waves["BAND"] - 2.19 * u.micron)
                sindxs = np.argsort(dwaves)
                kindx = sindxs[0]
                if dwaves[kindx] > 0.02 * u.micron:
                    warnings.warn("no K band measurement in E(lambda-V)", UserWarning)
                else:
                    ekv = self.exts["BAND"][kindx]
                    av = ekv / (akav - 1)
                    self.columns["AV"] = av

            for curname in self.exts.keys():
                self.exts[curname] = (self.exts[curname]/av) + 1
                self.uncs[curname] /= av
            # update the extinction curve type
            self.type = "alav"


    def get_fitdata(
        self,
        req_datasources,
        remove_uvwind_region=False,
        remove_lya_region=False,
        remove_irsblue=False,
    ):
        """
        Get the data to use in fitting the extinction curve

        Parameters
        ----------
        req_datasources : list of str
            list of data sources (e.g., ['IUE', 'BANDS'])

        remove_uvwind_region : boolean, optional (default=False)
            remove the UV wind regions from the returned data

        remove_lya_region : boolean, optional (default=False)
            remove the Ly-alpha regions from the returned data

        remove_irsblue : boolean, optional (default=False)
            remove the IRS blue photometry from the returned data

        Returns
        -------
        (wave, y, y_unc) : tuple of arrays
            wave is wavelength in microns
            y is extinction (no units)
            y_unc is uncertainty on y (no units)
        """
        xdata = []
        ydata = []
        uncdata = []
        nptsdata = []
        for cursrc in req_datasources:
            if cursrc in self.waves.keys():
                if (cursrc == "BAND") & remove_irsblue:
                    ibloc = np.logical_and(
                        14.0 * u.micron <= self.waves[cursrc],
                        self.waves[cursrc] < 16.0 * u.micron,
                    )
                    self.npts[cursrc][ibloc] = 0
                xdata.append(self.waves[cursrc].to(u.micron).value)
                ydata.append(self.exts[cursrc])
                uncdata.append(self.uncs[cursrc])
                nptsdata.append(self.npts[cursrc])
        wave = np.concatenate(xdata) * u.micron
        y = np.concatenate(ydata)
        unc = np.concatenate(uncdata)
        npts = np.concatenate(nptsdata)

        # remove uv wind line regions
        x = wave.to(1.0 / u.micron, equivalencies=u.spectral())
        if remove_uvwind_region:
            npts[np.logical_and(6.4 / u.micron <= x, x < 6.6 / u.micron)] = 0
            npts[np.logical_and(7.1 / u.micron <= x, x < 7.3 / u.micron)] = 0

        # remove Lya line region
        if remove_lya_region:
            npts[np.logical_and(8.0 / u.micron <= x, x < 8.475 / u.micron)] = 0

        # sort the data
        # at the same time, remove points with no data
        (gindxs,) = np.where(npts > 0)
        sindxs = np.argsort(x[gindxs])
        gindxs = gindxs[sindxs]
        wave = wave[gindxs]
        y = y[gindxs]
        unc = unc[gindxs]

        return (wave, y, unc)

    def save(
        self,
        ext_filename,
        column_info=None,
        fm90_best_params=None,
        fm90_per_params=None,
        p92_best_params=None,
        p92_per_params=None,
    ):
        """
        Save the extinction curve to a FITS file

        Parameters
        ----------
        filename : string
            Full filename to save extinction curve

        column_info : dict
            dictionary with information about the dust column
            for example: {'ebv': 0.1, 'rv': 4.2, 'av': 0.42}

        fm90_best_params : tuple of 2 float vectors
           parameter names and best fit values for the FM90 fit

        fm90_per_params : tuple of 2 float vectors
           parameter names and (p50, +unc, -unc) values for the FM90 fit

        p92_best_params : tuple of 2 float vectors
           parameter names and best fit values for the P92 fit

        p92_per_params : tuple of 2 float vectors
           parameter names and (p50, +unc, -unc) values for the P92 fit
        """
        # generate the primary header
        pheader = fits.Header()
        hname = ["EXTTYPE", "EXTBAND", "R_FILE", "C_FILE"]
        hcomment = [
            "Type of ext curve (options: elx, elxebv, alax)",
            "Band name for relative extinction measurement",
            "Reddened Star File",
            "Comparison Star File",
        ]
        hval = [self.type, self.type_rel_band, self.red_file, self.comp_file]

        ext_col_info = {
            "ebv": ("EBV", "E(B-V)"),
            "ebvunc": ("EBV_UNC", "E(B-V) uncertainty"),
            "av": ("AV", "A(V)"),
            "avunc": ("AV_UNC", "A(V) uncertainty"),
            "rv": ("RV", "R(V)"),
            "rvunc": ("RV_UNC", "R(V) uncertainty"),
        }
        if column_info is not None:
            for ckey in column_info.keys():
                if ckey in ext_col_info.keys():
                    keyname, desc = ext_col_info[ckey]
                    hname.append(keyname)
                    hcomment.append(desc)
                    hval.append(column_info[ckey])
                else:
                    print(ckey + " not supported for saving extcurves")

        # FM90 best fit parameters
        if fm90_best_params is not None:
            hname = np.concatenate((hname, fm90_best_params[0]))
            hval = np.concatenate((hval, fm90_best_params[1]))
            fm90_comment = [pname + ": FM90 parameter" for pname in fm90_best_params[0]]
            hcomment = np.concatenate((hcomment, fm90_comment))

        # FM90 p50 +unc -unc fit parameters
        if fm90_per_params is not None:
            # p50 values
            hname = np.concatenate(
                (hname, _hierarch_keywords([f"{cp}_p50" for cp in fm90_per_params[0]]))
            )
            hval = np.concatenate((hval, [cv[0] for cv in fm90_per_params[1]]))
            fm90_comment = [
                pname + ": FM90 p50 parameter" for pname in fm90_per_params[0]
            ]
            hcomment = np.concatenate((hcomment, fm90_comment))

            # +unc values
            hname = np.concatenate(
                (hname, _hierarch_keywords([f"{cp}_punc" for cp in fm90_per_params[0]]))
            )
            hval = np.concatenate((hval, [cv[1] for cv in fm90_per_params[1]]))
            fm90_comment = [
                pname + ": FM90 punc parameter" for pname in fm90_per_params[0]
            ]
            hcomment = np.concatenate((hcomment, fm90_comment))

            # -unc values
            hname = np.concatenate(
                (hname, _hierarch_keywords([f"{cp}_munc" for cp in fm90_per_params[0]]))
            )
            hval = np.concatenate((hval, [cv[2] for cv in fm90_per_params[1]]))
            fm90_comment = [
                pname + ": FM90 munc parameter" for pname in fm90_per_params[0]
            ]
            hcomment = np.concatenate((hcomment, fm90_comment))

        # P92 best fit parameters
        if p92_best_params is not None:
            hname = np.concatenate((hname, _hierarch_keywords(p92_best_params[0])))
            hval = np.concatenate((hval, p92_best_params[1]))
            p92_comment = [pname + ": P92 parameter" for pname in p92_best_params[0]]
            hcomment = np.concatenate((hcomment, p92_comment))

        # P92 p50 +unc -unc fit parameters
        if p92_per_params is not None:
            # p50 values
            hname = np.concatenate(
                (hname, _hierarch_keywords([f"{cp}_p50" for cp in p92_per_params[0]]))
            )
            hval = np.concatenate((hval, [cv[0] for cv in p92_per_params[1]]))
            p92_comment = [pname + ": P92 p50 parameter" for pname in p92_per_params[0]]
            hcomment = np.concatenate((hcomment, p92_comment))

            # +unc values
            hname = np.concatenate(
                (hname, _hierarch_keywords([f"{cp}_punc" for cp in p92_per_params[0]]))
            )
            hval = np.concatenate((hval, [cv[1] for cv in p92_per_params[1]]))
            p92_comment = [
                pname + ": P92 punc parameter" for pname in p92_per_params[0]
            ]
            hcomment = np.concatenate((hcomment, p92_comment))

            # -unc values
            hname = np.concatenate(
                (hname, _hierarch_keywords([f"{cp}_munc" for cp in p92_per_params[0]]))
            )
            hval = np.concatenate((hval, [cv[2] for cv in p92_per_params[1]]))
            p92_comment = [
                pname + ": P92 munc parameter" for pname in p92_per_params[0]
            ]
            hcomment = np.concatenate((hcomment, p92_comment))

        # P92 emcee results
        if hasattr(self, "p92_emcee_param_names"):
            print(self.p92_emcee_param_names)
            exit()

        # other possible header keywords
        #   setup to populate if info passed (TBD)
        #         'LOGT','LOGT_UNC','LOGG','LOGG_UNC','LOGZ','LOGZ_UNC',
        #         'AV','AV_unc','RV','RV_unc',
        #         'FMC2','FMC2U','FMC3','FMC3U','FMC4','FMC4U',
        #         'FMx0','FMx0U','FMgam','FMgamU',
        #         'LOGHI','LOGHI_U','LOGHIMW','LHIMW_U',
        #         'NHIAV','NHIAV_U','NHIEBV','NHIEBV_U'

        for k in range(len(hname)):
            pheader.set(hname[k], hval[k], hcomment[k])

        pheader.add_comment("Created with measure_extinction package")
        pheader.add_comment("https://github.com/karllark/measure_extinction")
        phdu = fits.PrimaryHDU(header=pheader)

        hdulist = fits.HDUList([phdu])

        # write the portions of the extinction curve from each dataset
        # individual extensions so that the full info is perserved
        for curname in self.exts.keys():
            col1 = fits.Column(name="WAVELENGTH", format="E", array=self.waves[curname])
            col2 = fits.Column(name="EXT", format="E", array=self.exts[curname])
            col3 = fits.Column(name="UNC", format="E", array=self.uncs[curname])
            col4 = fits.Column(name="NPTS", format="E", array=self.npts[curname])
            if curname == "BAND":
                col5 = fits.Column(name="NAME", format="A20", array=self.names[curname])
                cols = fits.ColDefs([col1, col2, col3, col4, col5])
            else:
                cols = fits.ColDefs([col1, col2, col3, col4])
            tbhdu = fits.BinTableHDU.from_columns(cols)
            tbhdu.header.set(
                "EXTNAME", "%sEXT" % curname, "%s based extinction" % curname
            )
            hdulist.append(tbhdu)

        hdulist.writeto(ext_filename, overwrite=True)

    def read(self, ext_filename):
        """
        Read in a saved extinction curve from a FITS file

        Parameters
        ----------
        filename : string
            Full filename to a save extinction curve
        """
        # read in the FITS file
        hdulist = fits.open(ext_filename)

        # get the list of extension names
        extnames = [hdulist[i].name for i in range(len(hdulist))]

        # the extinction curve itself
        for curname in _poss_datasources:
            curext = "%sEXT" % curname
            if curext in extnames:
                self.waves[curname] = hdulist[curext].data["WAVELENGTH"] * u.micron
                self.exts[curname] = hdulist[curext].data["EXT"]
                if "UNC" in hdulist[curext].data.columns.names:
                    self.uncs[curname] = hdulist[curext].data["UNC"]
                else:
                    self.uncs[curname] = hdulist[curext].data["EXT_UNC"]
                if "NPTS" in hdulist[curext].data.columns.names:
                    self.npts[curname] = hdulist[curext].data["NPTS"]
                else:
                    self.npts[curname] = np.full(len(self.waves[curname]), 1)
                if "NAME" in hdulist[curext].data.columns.names:
                    self.names[curname] = hdulist[curext].data["NAME"]

        # get the parameters of the extinction curve
        pheader = hdulist[0].header
        self.type = pheader.get("EXTTYPE")
        self.type_rel_band = pheader.get("EXTBAND")
        if self.type_rel_band is None:
            self.type_rel_band = "V"
        self.red_file = pheader.get("R_FILE")
        self.comp_file = pheader.get("C_FILE")

        column_keys = ["AV", "EBV", "RV", "LOGHI", "LOGHIMW", "NHIAV"]
        for curkey in column_keys:
            if pheader.get(curkey):
                if pheader.get("%s_UNC" % curkey):
                    tunc = float(pheader.get("%s_UNC" % curkey))
                elif pheader.get("%s_PUNC" % curkey):
                    tunc = 0.5 * (
                        float(pheader.get("%s_PUNC" % curkey))
                        + float(pheader.get("%s_MUNC" % curkey))
                    )
                else:
                    tunc = 0.0

                self.columns[curkey] = (float(pheader.get(curkey)), tunc)

        # get the columns p50 +unc -unc fit parameters if they exist
        if pheader.get("AV_p50"):
            self.columns_p50_fit = {}
            for bkey in column_keys:
                if pheader.get(f"{bkey}_p50"):
                    val = float(pheader.get(f"{bkey}_p50"))
                    punc = float(pheader.get(f"{bkey}_punc"))
                    munc = float(pheader.get(f"{bkey}_munc"))
                    self.columns_p50_fit[bkey] = (val, punc, munc)

        # get FM90 parameters if they exist
        FM90_keys = ["C1", "C2", "C3", "C4", "XO", "GAMMA"]
        if pheader.get("C2"):
            self.fm90_best_fit = {}
            for curkey in FM90_keys:
                if pheader.get(curkey):
                    self.fm90_best_fit[curkey] = float(pheader.get("%s" % curkey))
            # for completeness, populate C1 using from the FM07 relationship
            # if not already present
            if "C1" not in self.fm90_best_fit.keys():
                self.fm90_best_fit["C1"] = (
                    2.09 - 2.84 * self.fm90_best_fit["C2"][0],
                    2.84 * self.fm90_best_fit["C2"][1],
                )

        # get the FM90 p50 +unc -unc fit parameters if they exist
        if pheader.get("C2_p50"):
            self.fm90_p50_fit = {}
            for bkey in FM90_keys:
                if pheader.get(f"{bkey}_p50"):
                    val = float(pheader.get(f"{bkey}_p50"))
                    punc = float(pheader.get(f"{bkey}_punc"))
                    munc = float(pheader.get(f"{bkey}_munc"))
                    self.fm90_p50_fit[bkey] = (val, punc, munc)

        # get P92 best fit parameters if they exist
        P92_mkeys = ["BKG", "FUV", "NUV", "SIL1", "SIL2", "FIR"]
        P92_types = ["AMP", "LAMBDA", "WIDTH", "B", "N"]
        if pheader.get("BKG_amp"):
            self.p92_best_fit = {}
            for curmkey in P92_mkeys:
                for curtype in P92_types:
                    curkey = "%s_%s" % (curmkey, curtype)
                    if pheader.get(curkey):
                        self.p92_best_fit[curkey] = float(pheader.get("%s" % curkey))

        # get the P92 p50 +unc -unc fit parameters if they exist
        if pheader.get("BKG_amp_p50"):
            self.p92_p50_fit = {}
            for curmkey in P92_mkeys:
                for curtype in P92_types:
                    bkey = "%s_%s" % (curmkey, curtype)
                    if pheader.get(f"{bkey}_p50"):
                        val = float(pheader.get(f"{bkey}_p50"))
                        punc = float(pheader.get(f"{bkey}_punc"))
                        munc = float(pheader.get(f"{bkey}_munc"))
                        self.p92_p50_fit[bkey] = (val, punc, munc)

    def _get_ext_ytitle(self, ytype=None):
        """
        Format the extinction type nicely for plotting

        Returns
        -------
        ptype : string
            Latex formatted string for plotting
        """
        if not ytype:
            ytype = self.type
        relband = self.type_rel_band.replace("_", "")
        if ytype == "elx":
            return fr"$E(\lambda - {relband})$"
        elif ytype == "alax":
            return fr"$A(\lambda)/A({relband})$"
        elif ytype == "elv":
            return fr"$E(\lambda - V)$"
        elif ytype == "elvebv":
            return r"$E(\lambda - V)/E(B - V)$"
        elif ytype == "alav":
            return r"$A(\lambda)/A(V)$"
        else:
            return "%s (not found)" % ytype

    def plot(
        self,
        pltax,
        color=None,
        alpha=None,
        alax=False,
        normval=1.0,
        yoffset=0.0,
        rebin_fac=None,
        annotate_key=None,
        annotate_wave_range=None,
        annotate_rotation=0.0,
        annotate_yoffset=0.0,
        annotate_text=None,
        legend_key=None,
        legend_label=None,
        fontsize=None,
    ):
        """
        Plot an extinction curve

        Parameters
        ----------
        pltax : matplotlib plot object

        alax : boolean [False]
            convert from E(lambda-X) using A(X), if necessary
            plot A(lambda)/A(X)

        yoffset : float
            additive offset for the data

        rebin_fac : int
            factor by which to rebin spectra

        color : matplotlib color
            color for all the data plotted

        alpha : float
            transparency value (0=transparent, 1=opaque)

        annotate_key : string
            annotate the spectrum using the given data key

        legend_key : string
            legend the spectrum using the given data key

        legend_label : string
            label to use for legend

        fontsize : int
            fontsize for plot
        """
        if alax:
            # compute A(V) if it is not available
            if not "AV" in self.columns.keys():
                self.trans_elv_alav()
            av = float(self.columns["AV"])
            if self.type_rel_band != "V": # not sure if this works (where is RV given?)
                # use F04 model to convert AV to AX
                rv = float(self.columns["RV"][0])
                emod = F04(rv)
                (indx,) = np.where(self.type_rel_band == self.names["BAND"])
                axav = emod(self.waves["BAND"][indx[0]])
            else:
                axav = 1.0
            ax = axav * av

        for curtype in self.waves.keys():
            # replace extinction values by NaNs for wavelength regions that need to be excluded from the plot
            self.exts[curtype][self.npts[curtype] == 0] = np.nan
            x = self.waves[curtype].to(u.micron).value
            y = self.exts[curtype]
            yu = self.uncs[curtype]

            if alax and self.type == "elx":
                # convert E(lambda-X) to A(lambda)/A(X)
                y = (y / ax) + 1.0
                yu /= ax

            y = y / normval + yoffset
            yu = yu / normval

            # if curtype == legend_key:
            #     if legend_label is None:
            #         legval = self.red_file
            #     else:
            #         legval = legend_label
            # else:
            #     legval = None

            if curtype == "BAND":
                # plot band data as points with errorbars
                pltax.errorbar(
                    x, y, yerr=yu, fmt="o", color=color, alpha=alpha, mfc="white"
                )
            else:
                if rebin_fac is not None:
                    x = _rebin(x, rebin_fac)
                    y = _rebin(y, rebin_fac)

                pltax.plot(x, y, "-", color=color, alpha=alpha)

            if curtype == annotate_key:
                (ann_indxs,) = np.where(
                    (x >= annotate_wave_range[0].value)
                    & (x <= annotate_wave_range[1].value)
                )
                ann_val = np.nanmedian(y[ann_indxs])
                ann_val += (annotate_yoffset,)
                ann_xval = 0.5 * np.sum(annotate_wave_range.value)
                pltax.text(
                    ann_xval,
                    ann_val,
                    annotate_text,
                    horizontalalignment="right",
                    rotation=annotate_rotation,
                    fontsize=10,
                )
