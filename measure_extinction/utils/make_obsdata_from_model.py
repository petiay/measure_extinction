#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy.io import ascii
from astropy.table import Table, Column
import astropy.units as u
from astropy.convolution import Gaussian1DKernel, convolve
from synphot import SpectralElement
import stsynphot as STS

from measure_extinction.stardata import BandData
from measure_extinction.merge_obsspec import merge_stis_obsspec, merge_irs_obsspec
from measure_extinction.utils.mock_spectra_data import mock_stis_data

__all__ = ["make_obsdata_from_model"]


def rebin_spectrum(wave, flux, resolution, wave_range):
    """
    Rebin spectrum to input resolution over input wavelength range
    High to lower resolution only

    Parameters
    ----------
    wave : vector
        wavelengths of spectrum

    flux : vector
        spectrum in flux units

    resolution : float
        resolution of output spectrum

    wave_range : [float, float]
        wavelength range of output spectrum

    Outputs
    ------
    wave, flux, npts : tuple of vectors
        the model wavelength, flux, and npts at the requested wavelength
    """
    npts = int(
        np.log10(wave_range[1] / wave_range[0])
        / np.log10((1.0 + 2.0 * resolution) / (2.0 * resolution - 1.0))
    )

    twave = np.logspace(
        np.log10(wave_range[0]), np.log10(wave_range[1]), num=npts + 1, endpoint=True
    )
    full_wave_min = twave[0:-1]
    full_wave_max = twave[1:]
    full_wave = 0.5 * (full_wave_min + full_wave_max)

    full_flux = np.zeros((npts))
    full_npts = np.zeros((npts), dtype=int)

    for k in range(npts):
        (indxs,) = np.where((wave >= full_wave_min[k]) & (wave < full_wave_max[k]))
        n_indxs = len(indxs)
        if n_indxs > 0:
            full_flux[k] = np.sum(flux[indxs])
            full_npts[k] = n_indxs

    # divide by the # of points to create the final rebinned spectrum
    (indxs,) = np.where(full_npts > 0)
    if len(indxs):
        full_flux[indxs] = full_flux[indxs] / full_npts[indxs]

    # interpolate to fill in missing points in the rebinned spectrum
    #  e.g., the model spectrum is not computed at a high enough resolution
    #        at all the needed wavelengths
    (zindxs,) = np.where(full_npts <= 0)
    if len(zindxs):
        ifunc = interp1d(
            full_wave[indxs], full_flux[indxs], kind="linear", bounds_error=False
        )
        full_flux = ifunc(full_wave)
        full_npts[zindxs] = 1
        (nanindxs,) = np.where(~np.isfinite(full_flux))
        if len(nanindxs):
            full_flux[nanindxs] = 0.0
            full_npts[nanindxs] = 0

    return (full_wave, full_flux, full_npts)


def get_phot(mwave, mflux, band_names, band_resp_filenames):
    """
    Compute the magnitudes in the requested bands.

    Parameters
    ----------
    mwave : vector
        wavelengths of model flux

    mflux : vector
        model fluxes

    band_names: list of strings
        names of bands requested

    band_resp_filename : list of strings
        filenames of band response functions for the requested bands

    Outputs
    -------
    bandinfo : BandData object
    """
    # get a BandData object
    bdata = BandData("BAND")

    # path for non-HST band response curves
    data_path = pkg_resources.resource_filename(
        "measure_extinction", "data/Band_RespCurves/"
    )

    # compute the fluxes in each band
    for k, cband in enumerate(band_names):
        if "HST" in cband:
            bp_info = cband.split("_")
            bp = STS.band("%s,%s,%s" % (bp_info[1], bp_info[2], bp_info[3]))
            ncband = "%s_%s" % (bp_info[1], bp_info[3])
        else:
            bp = SpectralElement.from_file("%s%s" % (data_path, band_resp_filenames[k]))
            ncband = cband

        # check if the wavelength units are in microns instead of Angstroms
        #   may not work
        if max(bp.waveset) < 500 * u.Angstrom:
            print("filter wavelengths not in angstroms")
            exit()
            # a.waveset *= 1e4
        iresp = bp(mwave)
        bflux = np.sum(iresp * mflux) / np.sum(iresp)
        bflux_unc = 0.0
        bdata.band_fluxes[ncband] = (bflux, bflux_unc)

    # calculate the band magnitudes from the fluxes
    bdata.get_band_mags_from_fluxes()

    # get the band fluxes from the magnitudes
    #   partially redundant, but populates variables useful later
    bdata.get_band_fluxes()

    return bdata


def write_dat_file(
    filename,
    bandinfo,
    specinfo,
    header_info=["# obsdata created from model spectrum"],
    modelparams=None,
):
    """
    Write out a DAT file containing the photometry and pointers to
    spectroscopy files.

    Parameters
    ----------
    filename: string
        full file name of output DAT file

    bandinfo : BandData object
        contains the photometry data

    specinfo : dict of {type: filename}

    header_info: string array
        comments to add to the header

    modelparams: dict of {type: value}
        model parameters
        e.g., {'Teff': 10000.0, 'logg': 4.0, 'Z': 1, 'vturb': 2.0}
    """
    dfile = open(filename, "w")

    for hline in header_info:
        dfile.write("%s\n" % (hline))

    for cband in bandinfo.bands.keys():
        dfile.write(
            "%s = %f +/- %f\n"
            % (cband, bandinfo.bands[cband][0], bandinfo.bands[cband][1])
        )

    if specinfo is not None:
        for ckey in specinfo.keys():
            dfile.write("%s = %s\n" % (ckey, specinfo[ckey]))

    if modelparams is not None:
        for ckey in modelparams.keys():
            if isinstance(modelparams[ckey], str):
                dfile.write("%s = %s\n" % (ckey, modelparams[ckey]))
            else:
                dfile.write("%s = %f\n" % (ckey, modelparams[ckey]))

    dfile.close()


def make_obsdata_from_model(
    model_filename,
    model_type="tlusty",
    model_params=None,
    output_filebase=None,
    output_path=None,
    show_plot=False,
):
    """
    Create the necessary data files (.dat and spectra) from a
    stellar model atmosphere model to use as the unreddened
    comparsion star in the measure_extinction package

    Parameters
    ----------
    model_filename: string
        name of the file with the stellar atmosphere model spectrum

    model_type: string [default = 'tlusty']
        model type

    model_params: dict of {type: value}
        model parameters
        e.g., {'Teff': 10000.0, 'logg': 4.0, 'Z': 1, 'vturb': 2.0}

    output_filebase: string
        base for the output files
        E.g., output_filebase.dat and output_filebase_stis.fits

    output_path: string
        path to use for output files

    show_plot: boolean
        show a plot of the original and rebinned spectra/photometry
    """

    if output_filebase is None:
        output_filebase = "%s_standard" % (model_filename)

    if output_path is None:
        output_path = "/home/kgordon/Python_git/extstar_data/"

    allowed_model_types = ["tlusty"]
    if model_type not in allowed_model_types:
        raise ValueError("%s not an allowed model type" % (model_type))

    # read in the model spectrum
    mspec = ascii.read(
        model_filename,
        format="no_header",
        fast_reader={"exponent_style": "D"},
        names=["Freq", "SFlux"],
    )

    # error in file where the exponent 'D' is missing
    #   means that SFlux is read in as a string
    # solution is to remove the rows with the problem and replace
    #   the fortran 'D' with an 'E' and then convert to floats
    if mspec["SFlux"].dtype != np.float:
        indxs = [k for k in range(len(mspec)) if "D" not in mspec["SFlux"][k]]
        if len(indxs) > 0:
            indxs = [k for k in range(len(mspec)) if "D" in mspec["SFlux"][k]]
            mspec = mspec[indxs]
            new_strs = [cval.replace("D", "E") for cval in mspec["SFlux"].data]
            mspec["SFlux"] = new_strs
            mspec["SFlux"] = mspec["SFlux"].astype(np.float)

    # set the units
    mspec["Freq"].unit = u.Hz
    mspec["SFlux"].unit = u.erg / (u.s * u.cm * u.cm * u.Hz)

    # now extract the wave and flux colums
    mfreq = mspec["Freq"].quantity
    mwave = mfreq.to(u.angstrom, equivalencies=u.spectral())
    mflux = mspec["SFlux"].quantity.to(
        u.erg / (u.s * u.cm * u.cm * u.angstrom),
        equivalencies=u.spectral_density(mfreq),
    )

    # rebin to R=5000 for speed
    #   use a wavelength range that spans FUSE to Spitzer IRS
    wave_r5000, flux_r5000, npts_r5000 = rebin_spectrum(
        mwave.value, mflux.value, 5000, [912.0, 500000.0]
    )

    # save the full spectrum to a binary FITS table
    otable = Table()
    otable["WAVELENGTH"] = Column(wave_r5000, unit=u.angstrom)
    otable["FLUX"] = Column(flux_r5000, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(
        flux_r5000 * 0.0, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom)
    )
    otable["NPTS"] = Column(npts_r5000)
    otable.write(
        "%s/Models/%s_full.fits" % (output_path, output_filebase), overwrite=True
    )

    # dictionary to saye names of spectroscopic filenames
    specinfo = {}

    # create the ultraviolet HST/STIS mock observation
    stis_table = mock_stis_data(otable)

    # UV STIS obs
    rb_stis_uv = merge_stis_obsspec(stis_table[0:2], waveregion="UV")
    rb_stis_uv["SIGMA"] = rb_stis_uv["FLUX"] * 0.0
    stis_uv_file = "%s_stis_uv.fits" % (output_filebase)
    rb_stis_uv.write("%s/Models/%s" % (output_path, stis_uv_file), overwrite=True)
    specinfo["STIS"] = stis_uv_file
    # Optical STIS obs
    rb_stis_opt = merge_stis_obsspec(stis_table[2:4], waveregion="Opt")
    rb_stis_opt["SIGMA"] = rb_stis_opt["FLUX"] * 0.0
    stis_opt_file = "%s_stis_opt.fits" % (output_filebase)
    rb_stis_opt.write("%s/Models/%s" % (output_path, stis_opt_file), overwrite=True)
    specinfo["STIS_Opt"] = stis_opt_file

    # Spitzer IRS mock observation
    # Resolution approximately 100
    lrs_fwhm_pix = 5000.0 / 100.0
    g = Gaussian1DKernel(stddev=lrs_fwhm_pix / 2.355)

    # Convolve data
    nflux = convolve(otable["FLUX"].data, g)

    lrs_table = Table()
    lrs_table["WAVELENGTH"] = otable["WAVELENGTH"]
    lrs_table["FLUX"] = nflux
    lrs_table["NPTS"] = otable["NPTS"]
    lrs_table["ERROR"] = Column(np.full((len(lrs_table)), 1.0))

    rb_lrs = merge_irs_obsspec([lrs_table])
    rb_lrs["SIGMA"] = rb_lrs["FLUX"] * 0.0
    lrs_file = "%s_irs.fits" % (output_filebase)
    rb_lrs.write("%s/Models/%s" % (output_path, lrs_file), overwrite=True)
    specinfo["IRS"] = lrs_file

    # compute photometry
    # band_path = "%s/Band_RespCurves/" % output_path
    john_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
    john_fnames = ["John%s.dat" % (cband) for cband in john_bands]
    hst_bands = [
        "HST_WFC3_UVIS1_F275W",
        "HST_WFC3_UVIS1_F336W",
        "HST_WFC3_UVIS1_F475W",
        "HST_WFC3_UVIS1_F814W",
        "HST_WFC3_IR_F110W",
        "HST_WFC3_IR_F160W",
        "HST_ACS_WFC1_F475W",
        "HST_ACS_WFC1_F814W",
        "HST_WFPC2_4_F170W",
    ]
    hst_fnames = [""]
    # spitzer_bands = ['IRAC1', 'IRAC2', 'IRAC3', 'IRAC4', 'IRS15', 'MIPS24']
    # spitzer_fnames = ["{}/{}.dat".format(band_path, cband)
    #                   for cband in spitzer_bands]
    bands = john_bands + hst_bands
    band_fnames = john_fnames + hst_fnames
    # bands = john_bands
    # band_fnames = john_fnames

    bandinfo = get_phot(wave_r5000, flux_r5000, bands, band_fnames)

    # create the DAT file
    dat_filename = "%s/Models/%s.dat" % (output_path, output_filebase)
    header_info = [
        "# obsdata created from %s model atmosphere" % model_type,
        "# %s" % (output_filebase),
        "# file created by make_obsdata_from_model.py",
        "model_type = %s" % model_type,
    ]
    write_dat_file(
        dat_filename,
        bandinfo,
        specinfo,
        modelparams=model_params,
        header_info=header_info,
    )

    if show_plot:
        fig, ax = plt.subplots(figsize=(13, 10))
        # indxs, = np.where(npts_r5000 > 0)
        ax.plot(wave_r5000 * 1e-4, flux_r5000, "b-")
        ax.plot(bandinfo.waves, bandinfo.fluxes, "ro")

        (indxs,) = np.where(rb_stis_uv["NPTS"] > 0)
        ax.plot(
            rb_stis_uv["WAVELENGTH"][indxs].to(u.micron),
            rb_stis_uv["FLUX"][indxs],
            "m-",
        )
        (indxs,) = np.where(rb_stis_opt["NPTS"] > 0)
        ax.plot(
            rb_stis_opt["WAVELENGTH"][indxs].to(u.micron),
            rb_stis_opt["FLUX"][indxs],
            "g-",
        )
        (indxs,) = np.where(rb_lrs["NPTS"] > 0)
        ax.plot(rb_lrs["WAVELENGTH"][indxs].to(u.micron), rb_lrs["FLUX"][indxs], "c-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.show()


if __name__ == "__main__":
    mname = "/home/kgordon/Dust/Ext/Model_Standards_Data/BC30000g300v10.flux.gz"
    model_params = {}
    model_params["origin"] = "bstar"
    model_params["Teff"] = 30000.0
    model_params["logg"] = 3.0
    model_params["Z"] = 1.0
    model_params["vturb"] = 10.0
    make_obsdata_from_model(
        mname,
        model_type="tlusty",
        output_filebase="BC30000g300v10",
        output_path="/home/kgordon/Python_git/extstar_data",
        model_params=model_params,
        show_plot=True,
    )
