import numpy as np

from measure_extinction.stardata import StarData


class ModelData(object):
    """
    Provide stellar atmosphere model "observed" data given input stellar, gas,
    and dust extinction parameters.

    Parameters
    ----------
    modelfiles: string array
        set of model files to use

    path : string, optional
        path for model files

    band_names : string array, optional
        bands to use
        default = ['U', 'B', 'V', 'J', 'H', 'K']

    spectra_names : string array, optional
        origin of the spectra to use
        default = ['STIS']

    Attributes
    ----------
    n_models : int
        number of stellar atmosphere models

    temps : float array
        effective temperatures
    gravs : float array
        surface gravities
    mets : float array
        metallicities

    n_bands : int
        number of photometric bands
    band_names : string array
        names of the photometric bands
    band_fluxes : (n_models, n_bands) float array
        fluxes in the bands
    band_flux_uncs : (n_models, n_bands) float array
        flux uncertainties in the bands

    n_spectra : int
        number of different types of spectra
    spectra_names : string array
        names of the photometric bands
    spectra_fluxes : n_spectra list
        fluxes in the bands
    spectra_flux_uncs : n_spectra list
        flux uncertainties in the bands
    """
    def __init__(self, modelfiles,
                 path='./',
                 band_names=['U', 'B', 'V', 'J', 'H', 'K'],
                 spectra_names=['STIS']):

        self.n_models = len(modelfiles)

        # physical parameters of models
        self.temps = np.zeros(self.n_models)
        self.gravs = np.zeros(self.n_models)
        self.mets = np.zeros(self.n_models)

        # photometric band data
        self.n_bands = len(band_names)
        self.band_names = band_names
        self.band_fluxes = np.zeros((self.n_models, self.n_bands))
        self.band_flux_uncs = np.zeros((self.n_models, self.n_bands))

        # spectroscopic data
        self.n_spectra = len(spectra_names)
        self.spectra_names = spectra_names
        self.spectra_fluxes = {}
        self.spectra_flux_uncs = {}
        for cspec in self.spectra_names:
            self.spectra_fluxes[cspec] = None
            self.spectra_flux_uncs[cspec] = None

        # read and store the model data
        for k, cfile in enumerate(modelfiles):
            moddata = StarData(cfile, path=path)

            # photometric bands
            for i, cband in enumerate(self.band_names):
                band_flux = moddata.data['BAND'].get_band_flux(cband)
                self.band_fluxes[k, i] = band_flux[0]
                self.band_flux_uncs[k, i] = band_flux[1]

            # spectra
            for cspec in self.spectra_names:
                # initialize the spectra vectors
                if self.spectra_fluxes[cspec] is None:
                    self.spectra_fluxes[cspec] = \
                        np.zeros((self.n_models,
                                  len(moddata.data[cspec].fluxes)))
                    self.spectra_flux_uncs[cspec] = \
                        np.zeros((self.n_models,
                                  len(moddata.data[cspec].fluxes)))

                # get the spectral data
                self.spectra_fluxes[cspec][k, :] = moddata.data[cspec].fluxes
                self.spectra_flux_uncs[cspec][k, :] = moddata.data[cspec].uncs
