from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import string
import math
import warnings

import numpy as np
from scipy.io.idl import readsav
from astropy.io import fits

from stardata import StarData

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

    def calc_elv_bands(self, redstar, compstar):
        """
        Calculate the E(lambda-V) for the photometric band data

        Parameters
        ----------
        redstar : :class:StarData
            Observed data for the reddened star

        compstar : :class:StarData
            Observed data for the comparison star

        Returns
        -------
        Updated self.ext_(waves, x, exts, uncs)['BANDS']
        """
        pass

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
        self.type = 'elv'
        if ((src in red.data.keys())
                & (src in red.data.keys())):
            # check that the wavelenth grids are identical
            delt_wave = red.data[src].waves - comp.data[src].waves
            if np.sum(np.absolute(delt_wave)) > 0.01:
                warnings.warn("wavelength grids not equal for %s" % src,
                              UserWarning)
            else:
                # extinction is referenced to V band to
                # remove unknown or uncertain distances
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
        for cursrc in _poss_datasources:
            if cursrc == 'BAND':
                pass
            else:
                self.calc_elv_spectra(redstar, compstar, cursrc)

    def calc_ext_elvebv(self, reddened_star, model_fluxes_bands,
                        model_fluxes_spectra, ebv):

        #n_bands = self.data['BANDS'].n_flat_bands

        # need the aves of the optical/nir photometry for normalization
        _data_ave = np.average(reddened_star.data['BANDS'].flat_bands_fluxes)
        _model_ave = np.average(model_fluxes_bands)

        vindxs = np.argsort(abs(reddened_star.data['BANDS'].flat_bands_waves
                                - 0.545))
        vk = vindxs[0]

        bindxs = np.argsort(abs(reddened_star.data['BANDS'].flat_bands_waves
                                - 0.438))
        bk = bindxs[0]

        const_norm = 2.5*math.log10((_model_ave/_data_ave)
                                    * reddened_star.data['BANDS'].flat_bands_fluxes[vk]
                                    / model_fluxes_bands[vk])

        self.ext_type = 'elvebv'
        self.reddened_file = reddened_star.file

        self.ext_waves['BANDS'] = reddened_star.data['BANDS'].flat_bands_waves
        self.ext_x['BANDS'] = 1.0/reddened_star.data['BANDS'].flat_bands_waves
        self.ext_curve['BANDS'] = ((_model_ave/_data_ave)
                                   * (reddened_star.data['BANDS'].flat_bands_fluxes
                                   / model_fluxes_bands))
        self.ext_curve['BANDS'] = (-2.5*np.log10(self.ext_curve['BANDS'])
                                   + const_norm)
        self.ext_curve['BANDS'] /= ebv

        # sort to provide a clean spectroscopic extinction curve
        sindxs_good, = np.where(reddened_star.data['STIS'].npts > 0)

        self.ext_waves['STIS'] = reddened_star.data['STIS'].waves[sindxs_good]
        self.ext_x['STIS'] = 1.0/reddened_star.data['STIS'].waves[sindxs_good]
        self.ext_curve['STIS'] = ((_model_ave/_data_ave)
                                  * (reddened_star.data['STIS'].flux[sindxs_good]
                                     / model_fluxes_spectra))
        self.ext_curve['STIS'] = (-2.5*np.log10(self.ext_curve['STIS'])
                                  + const_norm)
        self.ext_curve['STIS'] /= ebv

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

    def read_ext_data_idlsave(self, ext_filename):
        spec_dict = readsav(ext_filename)

        indxs, = np.where(np.isfinite(spec_dict['realcurv']))
        self.ext_waves['STIS'] = spec_dict['wavein'][indxs]*1e-4
        self.ext_curve['STIS'] = spec_dict['realcurv'][indxs]

        indxs, = np.where(spec_dict['xcurv'] > 0.)
        self.ext_waves['MODEL'] = 1.0/spec_dict['xcurv'][indxs]
        self.ext_curve['MODEL'] = spec_dict['bestcurv'][indxs]

    def ext_uncs(self, mcmc_samples):
        # pick 100 of the samples to compute the covariance matrix
        nsamp = 10
        delta = mcmc_samples.shape[0]/nsamp
        indxs = np.arange(1,mcmc_samples.shape[0],delta)
        nsamp = len(indxs)

        # compute the model extinction curve for the picked samples
        sed_x = np.concatenate([self.ext_x['BANDS'],self.ext_x['STIS']])
        n_waves = len(sed_x)
        ext_curves = np.zeros((n_waves, nsamp))
        for i in range(nsamp):
            k = indxs[i]
            params = mcmc_samples[k,3:10]
            alav = f99.f99(params[1], sed_x, c2=params[2], c3=params[3],
                           c4=params[4], x0=params[5], gamma=params[6])
            ext_curves[:,i] = (alav - 1)*params[1]

        # compute the average extinction curve
        ave_ext = np.mean(ext_curves,axis=1)

        # compute the covariace matrix
        self.covar_matrix = np.zeros((n_waves,n_waves))
        for i in range(n_waves):
            for j in range(n_waves):
                for k in range(nsamp):
                    self.covar_matrix[i, j] += (ext_curves[i, k] - ave_ext[i]) \
                                              * (ext_curves[j, k] - ave_ext[j])
        self.covar_matrix /= (nsamp-1)

        # extract the diagonal - the straight uncertainties
        self.ext_curve_uncs_from_covar = np.sqrt(np.diagonal(self.covar_matrix))

        # compute the correlation matrix
        self.corr_matrix = np.array(self.covar_matrix)
        for i in range(n_waves):
            for j in range(n_waves):
                self.corr_matrix[i,j] /= (self.ext_curve_uncs_from_covar[i]
                                          *self.ext_curve_uncs_from_covar[j])

        # now break the straight uncertainties apart to make it easy to save
        self.ext_curve_uncs = {}
        self.ext_curve_uncs['BANDS'] = \
                    self.ext_curve_uncs_from_covar[0:len(self.ext_x['BANDS'])]
        self.ext_curve_uncs['STIS'] = \
                    self.ext_curve_uncs_from_covar[len(self.ext_x['BANDS']):]

        #print(self.covar_matrix)
        #print(self.corr_matrix)

        #print(1.0/sed_x)
        #print(ave_ext)
        #print(self.ext_curve_uncs)

        #exit()

    def save_ext_data(self, ext_filename, fit_param, fit_param_uncs):

        # pack for fit parameters
        fit_param_packed = np.zeros(2*len(fit_param))
        for k in range(len(fit_param)):
            fit_param_packed[2*k] = fit_param[k]
            fit_param_packed[2*k+1] = fit_param_uncs[k]

        # write a small primary header
        pheader = fits.Header()
        hname = ['EXTTYPE','IUEDATA','FUSEDATA','OPTDATA','NIRDATA','IRSDATA',
                 'R_FILE','C_FILE',
                 'LOGT','LOGT_UNC','LOGG','LOGG_UNC','LOGZ','LOGZ_UNC',
                 'AV','AV_unc','RV','RV_unc',
                 'FMC2','FMC2U','FMC3','FMC3U','FMC4','FMC4U','FMx0','FMx0U',
                 'FMgam','FMgamU',
                 'LOGHI','LOGHI_U','LOGHIMW','LHIMW_U',
                 'EBV','EBV_UNC',
                 'NHIAV','NHIAV_U','NHIEBV','NHIEBV_U']
        hval = np.concatenate([[self.ext_type, 1, 0, 0, 0, 0,
                                self.reddened_file, 'TLUSTY MODELS'],
                               fit_param_packed])
        hcomment = ['Type of ext curve - E(lambda-V), A(lambda)/A(V)',
                    'Positive if IUE portion of extinction curve exists',
                    'Positive if FUSE portion of extinction curve exists',
                    'Positive if optical portion of extinction curve exists',
                    'Positive if NIR portion of extinction curve exists',
                    'Positive if IRS extinction exists',
                    'Data File of Reddened Star',
                    'Data File of Comparison Star',
                    'log(Teff)','log(Teff) uncertainty',
                    'log(g)','log(g) uncertainty',
                    'log(Z)','log(Z) uncertainty',
                    'A(V)','A(V) uncertainty',
                    'R(V)','R(V) uncertainty',
                    'FM90 C2 parameter ','FM90 C2 parameter uncertainty',
                    'FM90 C3 parameter ','FM90 C3 parameter uncertainty',
                    'FM90 C4 parameter ','FM90 C4 parameter uncertainty',
                    'FM90 x0 parameter ','FM90 x0 parameter uncertainty',
                    'FM90 gamma parameter ','FM90 gamma parameter uncertainty',
                    'log(HI)','log(HI) uncertainty',
                    'logMW(HI)','logMW(HI) uncertainty',
                    'E(B-V)','E(B-V) uncertainty',
                    'N(HI)/A(V)','N(HI)/A(V) uncertainty',
                    'N(HI)/E(B-V)','N(HI)/E(B-V) uncertainty']

        for k in range(len(hname)):
            pheader.set(hname[k],hval[k],hcomment[k])

        pheader.add_comment('Extinction curve written  by extbymodel.py')
        pheader.add_comment('Extinction curve created with')
        pheader.add_comment('programs written by Karl D. Gordon (kgordon@stsci.edu)')
        phdu = fits.PrimaryHDU(header=pheader)

        hdulist = fits.HDUList([phdu])

        # write the BAND portion of the extinction curve
        col1 = fits.Column(name='WAVELENGTH', format='E',
                           array=self.ext_waves['BANDS'])
        col2 = fits.Column(name='X', format='E', array=self.ext_x['BANDS'])
        col3 = fits.Column(name='EXT', format='E',
                           array=self.ext_curve['BANDS'])
        col4 = fits.Column(name='UNC', format='E',
                           array=self.ext_curve_uncs['BANDS'])
        cols = fits.ColDefs([col1, col2, col3, col4])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set('EXTNAME', 'BANDEXT', 'Photometric Band Extinction')
        hdulist.append(tbhdu)

        # write the STIS portion of the extinction curve
        col1 = fits.Column(name='WAVELENGTH', format='E',
                           array=self.ext_waves['STIS'])
        col2 = fits.Column(name='X', format='E', array=self.ext_x['STIS'])
        col3 = fits.Column(name='EXT', format='E', array=self.ext_curve['STIS'])
        col4 = fits.Column(name='UNC', format='E',
                           array=self.ext_curve_uncs['STIS'])
        cols = fits.ColDefs([col1, col2, col3, col4])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set('EXTNAME', 'IUEEXT', 'IUE Ultraviolet Extinction')
        hdulist.append(tbhdu)

        hdulist.writeto(ext_filename, clobber=True)

        # output the covariance matrix
        hdu = fits.PrimaryHDU(self.covar_matrix)
        hdu.writeto(string.replace(ext_filename,'.fits','_covar.fits'),
                    clobber=True)
