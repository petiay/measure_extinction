from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings

import numpy as np
from astropy.io import fits

__all__ = ["ExtData"]


# globals
# possible datasets (also extension names in saved FITS file)
_poss_datasources = ['BAND', 'IUE', 'STIS', 'IRS']


def _flux_unc_as_mags(fluxes, uncs):
    """
    Provide the flux uncertainties in magnitudes accounting for the
    case where (fluxes-uncs) is negative
    """
    uncs_mag = np.empty(len(fluxes))

    # fluxes-uncs case
    indxs, = np.where(fluxes - uncs <= 0)
    if len(indxs) > 0:
        uncs_mag[indxs] = -2.5*np.log10(fluxes[indxs]
                                        / (fluxes[indxs] + uncs[indxs]))

    # normal case
    indxs, = np.where(fluxes - uncs > 0)
    if len(indxs) > 0:
        uncs_mag[indxs] = -2.5*np.log10((fluxes[indxs] - uncs[indxs])
                                        / (fluxes[indxs] + uncs[indxs]))

    return uncs_mag


class ExtData():
    """
    Extinction for a single line-of-sight

    Atributes:

    type : string
        extinction curve type (e.g., elv or alav)

    red_file : string
        reddened star filename

    comp_file : string
        comparison star filename

    columns : list of tuples of column measurements
        measurements are A(V), R(V), N(HI), etc.
        tuples are measurement, uncertainty

    waves : dict of key:wavelengths
    x : dict of key:wavenumbers
    ext : dict of key:E(lambda-v) measurements
    uncs : dict of key:E(lambda-v) measurement uncertainties
    npts : number of mesurements at each wavelength
        key is BANDS, IUE, IRS, etc.

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
        self.type = ''
        self.red_file = ''
        self.comp_file = ''
        self.columns = []
        self.fm90 = []
        self.waves = {}
        self.x = {}
        self.exts = {}
        self.uncs = {}
        self.npts = {}

        if filename is not None:
            self.read_ext_data(filename)

    def calc_elv_bands(self, red, comp):
        """
        Calculate the E(lambda-V) for the photometric band data

        Separate from the spectral case as the bands in common must
        be found.  In addition, some of the photometric observations are
        reported as colors (e.g., B-V) with uncertainies on those colors.
        As colors are what is needed for the extinction cruve, we want to
        work in those colors to preserve the inheritly lower uncertainties.

        Parameters
        ----------
        red : :class:StarData
            Observed data for the reddened star

        comp : :class:StarData
            Observed data for the comparison star

        Returns
        -------
        Updated self.(waves, x, exts, uncs)['BANDS']
        """
        # reference band
        red_V = red.data['BAND'].get_band_mag('V')
        comp_V = comp.data['BAND'].get_band_mag('V')

        # possible bands for the band extinction curve
        poss_bands = red.data['BAND'].get_poss_bands()

        waves = []
        exts = []
        uncs = []
        npts = []
        for pband_name in poss_bands.keys():
            red_mag = red.data['BAND'].get_band_mag(pband_name)
            comp_mag = comp.data['BAND'].get_band_mag(pband_name)
            if (red_mag is not None) & (comp_mag is not None):
                ext = (red_mag[0] - red_V[0]) - (comp_mag[0] - comp_V[0])
                unc = np.sqrt(red_mag[1]**2 + red_V[1]**2
                              + comp_mag[1]**2 + comp_V[1]**2)
                waves.append(red.data['BAND'].band_waves[pband_name])
                exts.append(ext)
                uncs.append(unc)
                npts.append(1)

        if len(waves) > 0:
            self.waves['BAND'] = np.array(waves)
            self.exts['BAND'] = np.array(exts)
            self.uncs['BAND'] = np.array(uncs)
            self.npts['BAND'] = np.array(npts)

    def calc_elv_spectra(self, red, comp, src):
        """
        Calculate the E(lambda-V) for the spectroscopic data

        Parameters
        ----------
        red : :class:StarData
            Observed data for the reddened star

        star : :class:StarData
            Observed data for the comparison star

        src : string
            data source (see global _poss_datasources)

        Returns
        -------
        Updated self.(waves, x, exts, uncs)[src]
        """
        if ((src in red.data.keys())
                & (src in red.data.keys())):
            # check that the wavelenth grids are identical
            delt_wave = red.data[src].waves - comp.data[src].waves
            if np.sum(np.absolute(delt_wave)) > 0.01:
                warnings.warn("wavelength grids not equal for %s" % src,
                              UserWarning)
            else:
                # reference band
                red_V = red.data['BAND'].get_band_mag('V')
                comp_V = comp.data['BAND'].get_band_mag('V')

                # setup the needed variables
                self.waves[src] = red.data[src].waves
                n_waves = len(self.waves[src])
                self.exts[src] = np.zeros(n_waves)
                self.uncs[src] = np.zeros(n_waves)
                self.npts[src] = np.zeros(n_waves)

                # only compute the extinction for good, positive fluxes
                indxs, = np.where((red.data[src].npts > 0)
                                  & (comp.data[src].npts > 0)
                                  & (red.data[src].fluxes > 0)
                                  & (comp.data[src].fluxes > 0))
                self.exts[src][indxs] = \
                    (-2.5*np.log10(red.data[src].fluxes[indxs]
                     / comp.data[src].fluxes[indxs])
                     + (comp_V[0] - red_V[0]))
                self.uncs[src] = np.sqrt(
                    np.square(_flux_unc_as_mags(red.data[src].fluxes[indxs],
                                                red.data[src].uncs[indxs]))
                    + np.square(_flux_unc_as_mags(comp.data[src].fluxes[indxs],
                                                  comp.data[src].uncs[indxs]))
                    + np.square(red_V[1])
                    + np.square(comp_V[1]))
                self.npts[src][indxs] = np.full(len(indxs), 1)

    def calc_elv(self, redstar, compstar):
        """
        Calculate the E(lambda-V) basic extinction measurement

        Parameters
        ----------
        redstar : :class:StarData
            Observed data for the reddened star

        compstar : :class:StarData
            Observed data for the comparison star

        Returns
        -------
        Updated self.ext_(waves, x, exts, uncs)
        """
        self.type = 'elv'
        for cursrc in _poss_datasources:
            if cursrc == 'BAND':
                self.calc_elv_bands(redstar, compstar)
            else:
                self.calc_elv_spectra(redstar, compstar, cursrc)

    def read_ext_data(self, ext_filename):
        """
        Read in a saved extinction curve

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
                self.waves[curname] = hdulist[curext].data['WAVELENGTH']
                if 'X' in hdulist[curext].data.columns.names:
                    self.x[curname] = hdulist[curext].data['X']
                self.exts[curname] = hdulist[curext].data['EXT']
                if 'UNC' in hdulist[curext].data.columns.names:
                    self.uncs[curname] = hdulist[curext].data['UNC']
                else:
                    self.uncs[curname] = hdulist[curext].data['EXT_UNC']
                if 'NPTS' in hdulist[curext].data.columns.names:
                    self.npts[curname] = hdulist[curext].data['NPTS']
                else:
                    self.npts[curname] = np.full(len(self.waves[curname]), 1)

        # get the parameters of the extiinction curve
        pheader = hdulist[0].header
        self.type = pheader.get('EXTTYPE')

        column_keys = ['AV', 'EBV', 'RV', 'LOGHI', 'LOGHIMW', 'NHIAV']
        for curkey in column_keys:
            if pheader.get(curkey):
                self.columns[curkey] = (float(pheader.get(curkey)),
                                        float(pheader.get('%s_UNC' % curkey)))

        if pheader.get('FMC2'):
            FM90_keys = ['C1', 'C2', 'C3', 'C4', 'x0', 'gam']
            self.fm90 = {}
            for curkey in FM90_keys:
                if pheader.get(curkey):
                    self.fm90[curkey] = (float(pheader.get('FM%s' % curkey)),
                                         float(pheader.get('FM%sU' % curkey)))
            # for completeness, populate C1 using from the FM07 relationship
            # if not already present
            if 'C1' not in self.fm90.keys():
                self.fm90['C1'] = (2.09 - 2.84*self.fm90['C2'][0],
                                   2.84*self.fm90['C2'][1])
