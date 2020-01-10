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
        bp = SpectralElement.from_file("%s%s" % (band_path, band_resp_filenames[k]))
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
    if not bands:
        print("No photometric data available to scale " + star_spec.type + " spectrum!!")
        return 1
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
    parser.add_argument("--split", help="Whether the input data are stored in one file (False) or split over two files (True)", action="store_true")
    args = parser.parse_args()

    # read in the data file (do not use the correction factors at this point)
    star_data = StarData('%s.dat' % args.starname.lower(), path=args.path, use_corfac=False)
    star_phot = star_data.data["BAND"]

    # check whether the data are stored in one file or split over two files,
    # and calculate the correction factor(s) for the spectrum (spectra)
    # add the correction factor(s) to the data file if not already in there,
    # otherwise overwrite the existing correction factor(s)
    if args.split:
        star_spec_short = star_data.data["SpeX_SXD"]
        star_spec_long = star_data.data["SpeX_LXD"]
        corfac_SXD = calc_corfac(star_phot, star_spec_short, ["J", "H", "K"], ["JohnJ.dat","JohnH.dat","JohnK.dat"])
        corfac_LXD = calc_corfac(star_phot, star_spec_long, ["L"], ["JohnL.dat"])
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

    else:
        star_spec = star_data.data["SpeX_full"]
        corfac = calc_corfac(star_phot, star_spec, ["J", "H", "K", "L"], ["JohnJ.dat","JohnH.dat","JohnK.dat","JohnL.dat"])
        if "SpeX_full" in star_data.corfac.keys():
            datafile = open(args.path+'%s.dat' % args.starname.lower(), "w")
            for ln, line in enumerate(star_data.datfile_lines):
                if "corfac_spex_full" in line:
                    star_data.datfile_lines[ln] = "corfac_spex_full = " + str(corfac) + "\n"
                datafile.write(star_data.datfile_lines[ln])
            datafile.close()
        else:
             with open(args.path+'%s.dat' % args.starname.lower(), "a") as datafile:
                 datafile.write("corfac_spex_full = "+ str(corfac) + "\n")
