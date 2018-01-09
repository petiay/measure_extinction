from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import string
import math

import numpy as np
from scipy.io.idl import readsav
from astropy.io import fits

from stardata import StarData

__all__ = ["ExtData"]


class ExtData():
    """
    Extinction for a single line-of-sight

    Atributes:

    ext_type : string
        extinction curve type (e.g., elv or alav)

    red_file : string
        reddened star filename

    comp_file : string
        comparison star filename

    columns : list of tuples of column measurements
        measurements are A(V), R(V), N(HI), etc.
        tuples are measurement, uncertainty

    ext_waves : dict of key:wavelengths
    ext_x : dict of key:wavenumbers
    ext_curve : dict of key:E(lambda-v) measurements
    ext_curve_uncs : dict of key:E(lambda-v) measurement uncertainties
        key is BANDS, IUE, IRS, etc.

    fm90 : list of FM90 parameters tuples
        tuples are measurement, uncertainty
    """
    def __init__(self):
        self.ext_type = ''
        self.red_file = ''
        self.comp_file = ''
        self.columns = []
        self.fm90 = []
        self.ext_waves = {}
        self.ext_x = {}
        self.ext_curve = {}
        self.ext_curve_uncs = {}

    def calc_elv_bands(self, red_star, comp_star):
        """
        Calculate the E(lambda-V) for the photometric band data

        Parameters
        ----------
        red_star : :class:StarData
            Observed data for the reddened star

        comp_star : :class:StarData
            Observed data for the comparison star

        Returns
        -------
        Updated self.ext_(waves, x, curve, curve_uncs)['BANDS']
        """
        pass

    def calc_elv_spectra(self, red_star, comp_star):
        """
        Calculate the E(lambda-V) for the spectroscopic data

        Parameters
        ----------
        red_star : :class:StarData
            Observed data for the reddened star

        comp_star : :class:StarData
            Observed data for the comparison star

        Returns
        -------
        Updated self.ext_(waves, x, curve, curve_uncs)['BANDS']
        """
        pass

    def calc_elv(self, red_star, comp_star):
        """
        Calculate the E(lambda-V) basic extinction measurement

        Parameters
        ----------
        red_star : :class:StarData
            Observed data for the reddened star

        comp_star : :class:StarData
            Observed data for the comparison star

        Returns
        -------
        Updated self.ext_(waves, x, curve, curve_uncs)
        """
        pass

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

        # read in the FITS file
        hdulist = fits.open(ext_filename)

        # get the list of extension names
        extnames = [hdulist[i].name for i in range(len(hdulist))]

        # the extinction curve itself
        self.ext_waves['BANDS'] = hdulist['BANDEXT'].data['WAVELENGTH']
        if 'X' in hdulist['BANDEXT'].data.columns.names:
            self.ext_x['BANDS'] = hdulist['BANDEXT'].data['X']
        self.ext_curve['BANDS'] = hdulist['BANDEXT'].data['EXT']
        self.ext_curve_uncs['BANDS'] = hdulist['BANDEXT'].data['UNC']

        if 'IUEEXT' in extnames:
            self.ext_waves['STIS'] = hdulist['IUEEXT'].data['WAVELENGTH']
            if 'X' in hdulist['IUEEXT'].data.columns.names:
                self.ext_x['STIS'] = hdulist['IUEEXT'].data['X']
            self.ext_curve['STIS'] = hdulist['IUEEXT'].data['EXT']
            self.ext_curve_uncs['STIS'] = hdulist['IUEEXT'].data['EXT_UNC']

        if 'IRSEXT' in extnames:
            self.ext_waves['IRS'] = hdulist['IRSEXT'].data['WAVELENGTH']
            if 'X' in hdulist['IRSEXT'].data.columns.names:
                self.ext_x['IRS'] = hdulist['IRSEXT'].data['X']
            self.ext_curve['IRS'] = hdulist['IRSEXT'].data['EXT']
            self.ext_curve_uncs['IRS'] = hdulist['IRSEXT'].data['EXT_UNC']

        # get the parameters of the extiinction curve
        pheader = hdulist[0].header
        self.exttype = pheader.get('EXTTYPE')
        if pheader.get('AV'):
            self.av = (float(pheader.get('AV')), float(pheader.get('AV_UNC')))
        if pheader.get('EBV'):
            self.ebv = (float(pheader.get('EBV')),
                        float(pheader.get('EBV_UNC')))
        if pheader.get('RV'):
            self.rv = (float(pheader.get('RV')),
                       float(pheader.get('RV_UNC')))
        if pheader.get('LOGHI'):
            self.loghi = (float(pheader.get('LOGHI')),
                          float(pheader.get('LOGHI_U')))
        if pheader.get('LOGHIMW'):
            self.loghimw = (float(pheader.get('LOGHIMW')),
                            float(pheader.get('LHIMW_U')))
        if pheader.get('NHIAV'):
            self.nhiav = (float(pheader.get('NHIAV')),
                          float(pheader.get('NHIAV_U')))

        if pheader.get('FMC2'):
            self.fm90 = {}
            self.fm90['C2'] = (float(pheader.get('FMC2')),
                               float(pheader.get('FMC2U')))
            self.fm90['C3'] = (float(pheader.get('FMC3')),
                               float(pheader.get('FMC3U')))
            self.fm90['C4'] = (float(pheader.get('FMC4')),
                               float(pheader.get('FMC4U')))
            self.fm90['x0'] = (float(pheader.get('FMx0')),
                               float(pheader.get('FMx0U')))
            self.fm90['gamma'] = (float(pheader.get('FMgam')),
                                  float(pheader.get('FMgamU')))
            # for completeness, populate C1 using from the FM07 relationship
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
