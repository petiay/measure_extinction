from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.units as u

__all__ = ["make_obsdata_from_model"]


def rebin_spectrum(wave, flux,
                   resolution,
                   wave_range):
    """
    Rebin spectrum to input resolution over input wavelength range
    High to lower resolution only

    wave : vector
        wavelengths of spectrum

    flux : vector
        spectrum in flux units

    resolution : float
        resolution of output spectrum

    wave_range : [float, float]
        wavelength range of output spectrum
    """
    npts = int(np.log10(wave_range[1]/wave_range[0])
               / np.log10((1.0 + 2.0*resolution)/(2.0*resolution - 1.0)))
    delta_wave_log = (np.log10(wave_range[1]) - np.log10(wave_range[0]))/npts
    wave_log10 = np.arange(np.log10(wave_range[0]),
                           np.log10(wave_range[1]) - delta_wave_log,
                           delta_wave_log)
    full_wave_min = 10**wave_log10
    full_wave_max = 10**(wave_log10 + delta_wave_log)

    full_wave = (full_wave_min + full_wave_max)/2.0
    full_flux = np.zeros((npts))
    full_npts = np.zeros((npts), dtype=int)

    for k in range(npts):
        indxs, = np.where((wave >= full_wave_min[k]) &
                          (wave < full_wave_max[k]))
        n_indxs = len(indxs)
        if n_indxs > 0:
            full_flux[k] = np.sum(flux[indxs])
            full_npts[k] = n_indxs

    # divide by the # of points to create the final rebinned spectrum
    indxs, = np.where(full_npts > 0)
    if len(indxs):
        full_flux[indxs] = full_flux[indxs]/full_npts[indxs]

    return (full_wave, full_flux, full_npts)


def make_obsdata_from_model(model_filename,
                            model_type='tlusty',
                            output_filebase=None,
                            output_path=None):
    """
    Create the necessary data files (.dat and spectra) from a
    stellar model atmosphere model to use as the unreddened
    comparsion star in the measure_extinction package

    Parameters
    ----------
    model_filename : string
        name of the file with the stellar atmosphere model spectrum

    model_type : string [default = 'tlusty']
        model type

    output_filebase: string
        base for the output files
        E.g., output_filebase.dat and output_filebase_stis.fits

    output_path: string
        path to use for output files
    """

    if output_filebase is None:
        output_filebase = '%s_standard' % (model_filename)

    if output_path is None:
        output_path = '/home/kgordon/Python_git/extstar_data/'

    allowed_model_types = ['tlusty']
    if model_type not in allowed_model_types:
        raise ValueError("%s not an allowed model type" % (model_type))

    # read in the model spectrum
    mspec = ascii.read(model_filename, format='no_header',
                       fast_reader={'exponent_style': 'D'},
                       names=['Freq', 'SFlux'])

    # error in file where the exponent 'D' is missing
    #   means that SFlux is read in as a string
    # solution is to remove the rows with the problem and replace
    #   the fortran 'D' with an 'E' and then convert to floats
    indxs = [k for k in range(len(mspec)) if 'D' not in mspec['SFlux'][k]]
    if len(indxs) > 0:
        indxs = [k for k in range(len(mspec)) if 'D' in mspec['SFlux'][k]]
        mspec = mspec[indxs]
        new_strs = [cval.replace('D', 'E') for cval in mspec['SFlux'].data]
        mspec['SFlux'] = new_strs
        mspec['SFlux'] = mspec['SFlux'].astype(np.float)

    # set the units
    mspec['Freq'].unit = u.Hz
    mspec['SFlux'].unit = u.erg/(u.s*u.cm*u.cm*u.Hz)

    # now extract the wave and flux colums
    mfreq = mspec['Freq'].quantity
    mwave = mfreq.to(u.angstrom, equivalencies=u.spectral())
    mflux = mspec['SFlux'].quantity.to(u.erg/(u.s*u.cm*u.cm*u.micron),
                                       equivalencies=u.spectral_density(mfreq))

    # rebin to R=5000 for speed
    #   use a wavelength range that spans FUSE to Spitzer IRS
    wave_r5000, flux_r5000, npts_r5000 = rebin_spectrum(mwave.value,
                                                        mflux.value,
                                                        5000,
                                                        [912., 40000.])

    fig, ax = plt.subplots(figsize=(13, 10))
    indxs, = np.where(npts_r5000 > 0)
    ax.plot(wave_r5000[indxs], flux_r5000[indxs], '-')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()


if __name__ == "__main__":
    mname = '/home/kgordon/Dust/Ext/Model_Standards_Data/BC15000g175v10.flux.gz'
    make_obsdata_from_model(mname, model_type='tlusty')
