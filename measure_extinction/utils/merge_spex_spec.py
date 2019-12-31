#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_spex_obsspec


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--path",
        help="path where merged spectra will be stored",
        default="/home/mdecleir/Documents/NIR_ext/Data/Reduced",
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    sfilename = "%s/%s.txt" % (args.path, args.starname)
    stable = Table.read(
        sfilename,
        format="ascii",
        names=[
            "WAVELENGTH",
            "FLUX",
            "ERROR",
            "FLAG",
        ],
    )
    stable = [stable]
    rb_stis_opt = merge_spex_obsspec(stable)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname
    spex_file = "%s_table.fits" % (outname)
    rb_stis_opt.write("%s/%s" % (args.path, spex_file), overwrite=True)
