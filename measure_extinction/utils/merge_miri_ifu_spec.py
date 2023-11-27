#!/usr/bin/env python

import glob
import argparse
import numpy as np
import pkg_resources

from astropy.table import QTable

from measure_extinction.merge_obsspec import merge_miri_ifu_obsspec


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--inpath",
        help="path where original data files are stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Orig"),
    )
    parser.add_argument(
        "--outpath",
        help="path where merged spectra will be stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Out"),
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    sfilename = f"{args.inpath}{args.starname}*.fits"
    print(sfilename)
    sfiles = glob.glob(sfilename)
    print(sfiles)
    stable = []
    for cfile in sfiles:
        print(cfile)
        cdata = QTable.read(cfile)
        cdata.rename_column("FLUX_ERROR", "ERROR")
        cdata["NPTS"] = np.full((len(cdata["FLUX"])), 1.0)
        cdata["NPTS"][cdata["FLUX"] == 0.0] = 0.0
        stable.append(cdata)

    rb_mrs = merge_miri_ifu_obsspec(stable)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname.lower()
    mrs_file = f"{outname}_miri_ifu.fits"
    rb_mrs.write(f"{args.outpath}/{mrs_file}", overwrite=True)
