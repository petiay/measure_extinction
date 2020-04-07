#!/usr/bin/env python

import argparse
import numpy as np
import astropy.units as u
import pkg_resources
from synphot import SpectralElement, SourceSpectrum, Observation
from measure_extinction.stardata import StarData, BandData
from synphot.models import Empirical1D


# function to get photometry from a spectrum
def get_phot(spec, bands):
    """
    Compute the fluxes in the requested bands.

    Parameters
    ----------
    spec : SpecData object
        the spectrum

    bands: list of strings
        bands requested

    Outputs
    -------
    band fluxes : numpy array
        calculated band fluxes
    """

    # create a SourceSpectrum object from the spectrum, excluding the bad regions
    spectrum = SourceSpectrum(Empirical1D, points=spec.waves.to(u.Angstrom)[spec.npts!=0], lookup_table=spec.fluxes[spec.npts!=0])

    # path for band response curves
    band_path = pkg_resources.resource_filename(
        "measure_extinction", "data/Band_RespCurves/"
    )

    # dictionary linking the bands to their response curves
    bandnames = {"J":"2MASSJ", "H":"2MASSH", "K":"2MASSKs", "IRAC1":"IRAC1", "IRAC2":"IRAC2", "L":"AAOL", "M":"AAOM"}

    # define the units of the output fluxes
    funit = u.erg / (u.s * u.cm * u.cm * u.Angstrom)

    # compute the flux in each band
    fluxes = np.zeros(len(bands))
    for k, band in enumerate(bands):
        # create the bandpass (as a SpectralElement object)
        bp = SpectralElement.from_file("%s%s.dat" % (band_path, bandnames[band])) # assumes wavelengths are in Angstrom!!
        # integrate the spectrum over the bandpass
        obs = Observation(spectrum,bp,force='taper') # 'taper' bridges the gaps in the spectrum
        fluxes[k] = obs.effstim(funit).value
    return fluxes


# function to compare the photometry obtained from the spectrum with the band photometry
# and to calculate an average correction factor for the spectrum
def calc_corfac(star_phot, star_spec, bands):
    """
    Compute the correction factor for the spectrum.

    Parameters
    ----------
    star_phot : BandData object
        band data of the star

    star_spec : SpecData object
        spectral data of the star

    bands: list of strings
        bands to use in the calculation of the correction factor

    Outputs
    -------
    corfac : float
        mean correction factor for the spectrum
    """
    bands = [band for band in bands if band in star_phot.band_fluxes.keys()]
    if "IRAC1" in bands and "L" in bands:
        bands.remove("L")
    if "IRAC2" in bands and "M" in bands:
        bands.remove("M")
    if not bands:
        print("No photometric data available to scale " + star_spec.type + " spectrum!!")
        return None
    fluxes_bands = np.array([star_phot.band_fluxes[band][0] for band in bands])
    fluxes_spectra = get_phot(star_spec, bands)
    corfacs = fluxes_bands / fluxes_spectra
    return "%.3f" % np.mean(corfacs)


# function to read in the available data, calculate the correction factors and save the factors in the data file
def calc_save_corfac_spex(starname,path):
    # read in the data file (do not use the correction factors at this point)
    star_data = StarData('%s.dat' % starname.lower(), path=path, use_corfac=False)
    star_phot = star_data.data["BAND"]
    # check which spectra are available,
    # and calculate the correction factors for the spectra
    if "SpeX_SXD" in star_data.data.keys():
        star_spec_SXD = star_data.data["SpeX_SXD"]
        corfac_SXD = calc_corfac(star_phot, star_spec_SXD, ["J", "H", "K"])
    else:
        corfac_SXD = None

    if "SpeX_LXD" in star_data.data.keys():
        star_spec_LXD = star_data.data["SpeX_LXD"]
        corfac_LXD = calc_corfac(star_phot, star_spec_LXD, ["IRAC1", "IRAC2", "L", "M"])
    else:
        corfac_LXD = None

    # add the correction factors to the data file if not already in there,
    # otherwise overwrite the existing correction factors
    if "SpeX_SXD" in star_data.corfac.keys():
        datafile = open(path+'%s.dat' % starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex_SXD" in line:
                star_data.datfile_lines[ln] = "corfac_spex_SXD = " + str(corfac_SXD) + "\n"
            datafile.write(star_data.datfile_lines[ln])
        datafile.close()
    else:
         with open(path+'%s.dat' % starname.lower(), "a") as datafile:
             datafile.write("corfac_spex_SXD = "+ str(corfac_SXD) + "\n")

    if "SpeX_LXD" in star_data.corfac.keys():
        datafile = open(path+'%s.dat' % starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex_LXD" in line:
                star_data.datfile_lines[ln] = "corfac_spex_LXD = " + str(corfac_LXD) + "\n"
            datafile.write(star_data.datfile_lines[ln])
        datafile.close()
    else:
        with open(path+'%s.dat' % starname.lower(), "a") as datafile:
            datafile.write("corfac_spex_LXD = "+ str(corfac_LXD) + "\n")


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()

    parser.add_argument("starname", help="name of the star (filebase)")
    parser.add_argument("--path", help="path where data files are stored",
    default=pkg_resources.resource_filename('measure_extinction',
                                            'data/'))
    args = parser.parse_args()

    # calculate and save the SpeX correction factors
    calc_save_corfac_spex(args.starname,args.path)
