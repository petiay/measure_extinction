#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pkg_resources
import os

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_spex_obsspec


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--inpath",
        help="path where original data files are stored",
        default=pkg_resources.resource_filename('measure_extinction',
                                            'data/Orig/NIR')
    )
    parser.add_argument(
        "--outpath",
        help="path where merged spectra will be stored",
        default=pkg_resources.resource_filename('measure_extinction',
                                                'data/Out')
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--split", help="Whether the input data are stored in one file (False) or split over two files (True)", action="store_true")
    args = parser.parse_args()

    # check whether the data are stored in one file or split over two files
    if args.split:
        sfilename_S = "%s/%s_SXD.txt" % (args.inpath, args.starname)
        sfilename_L = "%s/%s_LXD.txt" % (args.inpath, args.starname)
        if not os.path.isfile(sfilename_S):
            sfilenames = [sfilename_L]
            if not os.path.isfile(sfilename_L):
                print("No spectra could be found to merge!")
        elif not os.path.isfile(sfilename_L):
            sfilenames = [sfilename_S]
        else:
            sfilenames = [sfilename_S, sfilename_L]
    else:
        sfilenames = ["%s/%s.txt" % (args.inpath, args.starname)]

    for filename in sfilenames:
        stable = Table.read(
            filename,
            format="ascii",
            names=[
                "WAVELENGTH",
                "FLUX",
                "ERROR",
                "FLAG",
                ],
                )
        rb_stis_opt = merge_spex_obsspec(stable)
        if args.outname:
            outname = args.outname
        else:
            outname = os.path.basename(filename).split('.')[0]
        spex_file = "%s_spex.fits" % (outname)
        rb_stis_opt.write("%s/%s" % (args.outpath, spex_file), overwrite=True)
