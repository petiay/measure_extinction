#!/usr/bin/env python

import argparse
import numpy as np
import astropy.units as u
import pkg_resources
from synphot import SpectralElement
from measure_extinction.stardata import StarData
from measure_extinction.stardata import BandData

# function to get photometry from a spectrum
def get_phot(wave, flux, bands, bandnames):
    """
    Compute the fluxes in the requested bands.

    Parameters
    ----------
    wave : vector
        spectrum wavelengths

    flux : vector
        spectrum fluxes

    bands: list of strings
        bands requested

    bandnames : list of strings
        filenames of band response functions for the requested bands

    Outputs
    -------
    band fluxes : numpy array
        calculated band fluxes
    """

    # create a BandData object
    bdata = BandData("BAND")

    # path for band response curves
    band_path = pkg_resources.resource_filename(
        "measure_extinction", "data/Band_RespCurves/"
    )

    fluxes = np.zeros(len(bands))

    # compute the flux in each band
    for k, band in enumerate(bands):
        bp = SpectralElement.from_file("%s%s" % (band_path, bandnames[k]))
        resp = bp(wave)
        fluxes[k] = np.sum(resp * flux) / np.sum(resp)
    return fluxes

# function to compare the photometry obtained from the spectrum with the band photometry
# and to calculate an average correction factor for the spectrum
def calc_corfac(star_phot, star_spec, bands, bandnames):
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

    bandnames : list of strings
        filenames of band response functions

    Outputs
    -------
    corfac : floats
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
    fluxes_spectra = get_phot(star_spec.waves.to(u.Angstrom), star_spec.fluxes.value, bands, bandnames)
    corfacs = fluxes_bands / fluxes_spectra
    return np.mean(corfacs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("starname", help="name of the star (filebase)")
    parser.add_argument("--path", help="path where data files are stored",
    default=pkg_resources.resource_filename('measure_extinction',
                                            'data/'))
    args = parser.parse_args()

    # read in the data file (do not use the correction factors at this point)
    star_data = StarData('%s.dat' % args.starname.lower(), path=args.path, use_corfac=False)
    star_phot = star_data.data["BAND"]

    # check which spectra are available,
    # and calculate the correction factors for the spectra
    if "SpeX_SXD" in star_data.data.keys():
        star_spec_SXD = star_data.data["SpeX_SXD"]
        corfac_SXD = calc_corfac(star_phot, star_spec_SXD, ["J", "H", "K"], ["2MASSJ.dat","2MASSH.dat","2MASSKs.dat"])
    else:
        corfac_SXD = None

    if "SpeX_LXD" in star_data.data.keys():
        star_spec_LXD = star_data.data["SpeX_LXD"]
        corfac_LXD = calc_corfac(star_phot, star_spec_LXD, ["IRAC1", "IRAC2", "L", "M"], ["IRAC1.dat", "IRAC2.dat", "AAOL.dat", "AAOM.dat"])
    else:
        corfac_LXD = None

    # add the correction factors to the data file if not already in there,
    # otherwise overwrite the existing correction factors
    if "SpeX_SXD" in star_data.corfac.keys():
        datafile = open(args.path+'%s.dat' % args.starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex_SXD" in line:
                star_data.datfile_lines[ln] = "corfac_spex_SXD = " + str(corfac_SXD) + "\n"
            datafile.write(star_data.datfile_lines[ln])
        datafile.close()
    else:
         with open(args.path+'%s.dat' % args.starname.lower(), "a") as datafile:
             datafile.write("corfac_spex_SXD = "+ str(corfac_SXD) + "\n")

    if "SpeX_LXD" in star_data.corfac.keys():
        datafile = open(args.path+'%s.dat' % args.starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex_LXD" in line:
                star_data.datfile_lines[ln] = "corfac_spex_LXD = " + str(corfac_LXD) + "\n"
            datafile.write(star_data.datfile_lines[ln])
        datafile.close()
    else:
        with open(args.path+'%s.dat' % args.starname.lower(), "a") as datafile:
            datafile.write("corfac_spex_LXD = "+ str(corfac_LXD) + "\n")
