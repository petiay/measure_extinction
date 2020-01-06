#!/usr/bin/env python

import argparse
import numpy as np
import astropy.units as u
import pkg_resources

from astropy.table import Table
from synphot import SpectralElement
from measure_extinction.stardata import StarData
from measure_extinction.stardata import BandData

# function to get photometry from a spectrum
def get_phot(wave, flux, band_names, band_resp_filenames):
    """
    Compute the fluxes in the requested bands.

    Parameters
    ----------
    wave : vector
        spectrum wavelengths

    flux : vector
        spectrum fluxes

    band_names: list of strings
        names of bands requested

    band_resp_filename : list of strings
        filenames of band response functions for the requested bands

    Outputs
    -------
    band fluxes : dict of {band: flux}
    """

    # create a BandData object
    bdata = BandData("BAND")

    # path for band response curves
    band_path = pkg_resources.resource_filename(
        "measure_extinction", "data/Band_RespCurves/"
    )

    # compute the fluxes in each band
    for k, band in enumerate(band_names):
        bp = SpectralElement.from_file("%s%s" % (band_path, band_resp_filenames[k]))
        resp = bp(wave)
        bflux = np.sum(resp * flux) / np.sum(resp)
        bflux_unc = 0.0
        bdata.band_fluxes[band] = (bflux, bflux_unc)

    return bdata.band_fluxes

# function to compare the photometry obtained from the spectrum with the band photometry
# and to calculate a correction factor
def calc_corfac(fluxes_bands, fluxes_spectra, band):
    return fluxes_bands[band][0] / fluxes_spectra[band][0]


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
    star_spec = star_data.data["SpeX"]

    # get the band photometry and the photometry from the spectrum
    fluxes_bands = star_phot.band_fluxes
    fluxes_spectra = get_phot(star_spec.waves.to(u.Angstrom), star_spec.fluxes.value, ["J","H","K"], ["JohnJ.dat","JohnH.dat","JohnK.dat"])

    # compute the correction factors for the different bands and average them
    corfac_J = calc_corfac(fluxes_bands, fluxes_spectra, "J")
    corfac_H = calc_corfac(fluxes_bands, fluxes_spectra, "H")
    corfac_K = calc_corfac(fluxes_bands, fluxes_spectra, "K")
    av_corfac = np.mean([corfac_J, corfac_H, corfac_K])

    # add this correction factor to the data file if it is not already in there,
    # otherwise overwrite the existing correction factor
    if "SpeX" in star_data.corfac.keys():
        datafile = open(args.path+'%s.dat' % args.starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex" in line:
                line = "corfac_spex = " + str(av_corfac) + "\n"
            datafile.write(line)
        datafile.close()
    else:
         with open(args.path+'%s.dat' % args.starname.lower(), "a") as datafile:
             datafile.write("corfac_spex = "+ str(av_corfac) + "\n")
