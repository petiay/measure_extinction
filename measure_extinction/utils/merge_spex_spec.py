#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pkg_resources
import os

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_spex_obsspec


def spex_mask(starname, path):
    mask = []
    file = open(
        os.path.dirname(os.path.normpath(path)) + "/" + starname.lower() + ".dat", "r"
    )
    for line in list(file):
        if line.startswith("spex_mask"):
            mask = eval(line.split("= ")[1].strip())
    return mask
    file.close()


def merge_spex(starname, inpath, outpath):
    # check which data are available
    filename_S = "%s/%s_sxd.txt" % (inpath, starname.lower())
    filename_L = "%s/%s_lxd.txt" % (inpath, starname.lower())
    if not os.path.isfile(filename_S):
        filenames = [filename_L]
        if not os.path.isfile(filename_L):
            print("No SpeX spectra could be found for this star!")
    elif not os.path.isfile(filename_L):
        filenames = [filename_S]
    else:
        filenames = [filename_S, filename_L]

    # obtain wavelength regions that need to be masked
    mask = spex_mask(starname, outpath)

    # bin and merge the spectra
    for filename in filenames:
        table = Table.read(
            filename,
            format="ascii",
            names=["WAVELENGTH", "FLUX", "ERROR", "FLAG"],
        )
        spex_merged = merge_spex_obsspec(table, mask)
        spex_file = os.path.basename(filename).split(".")[0] + "_spex.fits"
        spex_merged.write("%s/%s" % (outpath, spex_file), overwrite=True)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--inpath",
        help="path where original SpeX ASCII files are stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Orig/NIR"),
    )
    parser.add_argument(
        "--outpath",
        help="path where merged SpeX spectra will be stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Spectra"),
    )
    args = parser.parse_args()

    # merge the spectra
    merge_spex(args.starname, args.inpath, args.outpath)
