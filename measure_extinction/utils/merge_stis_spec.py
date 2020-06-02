#!/usr/bin/env python
import argparse
import numpy as np
import pkg_resources

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_stis_obsspec


def read_stis_archive_format(filename):
    """
    Read the STIS archive format and make it a *normal* table
    """
    t1 = Table.read(filename)
    ot = Table()
    for ckey in t1.colnames:
        if len(t1[ckey].shape) == 2:
            ot[ckey] = t1[ckey].data[0, :]

    return ot


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--waveregion",
        choices=["UV", "Opt"],
        default="Opt",
        help="wavelength region to merge",
    )
    parser.add_argument(
        "--inpath",
        help="path where original data files are stored",
        default="/home/kgordon/Python_git/extstar_data/STIS_Data/Orig",
    )
    parser.add_argument(
        "--project",
        help="project name (used in path)",
        default="/",
    )
    parser.add_argument(
        "--outpath",
        help="path where merged spectra will be stored",
        default="/home/kgordon/Python_git/extstar_data/STIS_Data",
    )
    parser.add_argument(
        "--ralph", action="store_true", help="Ralph Bohlin reduced data"
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    if args.ralph:
        sfilename = "%s/%s/%s.mrg" % (args.inpath, args.waveregion, args.starname)
        stable = Table.read(
            sfilename,
            format="ascii",
            data_start=23,
            names=[
                "WAVELENGTH",
                "COUNT-RATE",
                "FLUX",
                "STAT-ERROR",
                "SYS-ERROR",
                "NPTS",
                "TIME",
                "QUAL",
            ],
        )
        stable = [stable]
    else:
        sfilename = "%s/%s/%s/%s" % (
            args.inpath,
            args.waveregion,
            args.project,
            args.starname.lower(),
        )
        t1 = read_stis_archive_format(sfilename + "10_x1d.fits")
        t2 = read_stis_archive_format(sfilename + "20_x1d.fits")
        t1.rename_column("ERROR", "STAT-ERROR")
        t2.rename_column("ERROR", "STAT-ERROR")
        t1["NPTS"] = np.full((len(t1["FLUX"])), 1.0)
        t2["NPTS"] = np.full((len(t2["FLUX"])), 1.0)
        t1["NPTS"][t1["FLUX"] == 0.0] = 0.0
        t2["NPTS"][t2["FLUX"] == 0.0] = 0.0
        stable = [t1, t2]

    rb_stis_opt = merge_stis_obsspec(stable, waveregion=args.waveregion)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname.lower()
    stis_opt_file = "%s_stis_%s.fits" % (outname, args.waveregion)
    print(stis_opt_file)
    rb_stis_opt.write("%s/%s" % (args.outpath, stis_opt_file), overwrite=True)
