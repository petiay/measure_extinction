#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from measure_extinction.stardata import StarData
from measure_extinction.utils.helpers import get_full_starfile


def plot_spec_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star for which to plot the spectrum")
    parser.add_argument("--path", help="path to data files", default=pkg_resources.resource_filename('measure_extinction', 'data/'))
    parser.add_argument("--mlam4", help="plot lambda^4*F(lambda)", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    return parser


if __name__ == "__main__":

    # commandline parser
    parser = plot_spec_parser()
    args = parser.parse_args()

    # read in the observed data of the star
    # fstarname, file_path = get_full_starfile(args.starname)
    # --> this is currently not working, because the default path in get_full_starfile is not accessible as an external user.

    starobs = StarData('%s.dat' % args.starname.lower(), path=args.path, use_corfac=True)

    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=1)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    # setup the plot
    fig, ax = plt.subplots(figsize=(13, 10))

    # plot the bands and all spectra for this star
    starobs.plot(ax, mlam4=args.mlam4)

    # finish configuring the plot
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylabel(r"$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # use the whitespace better
    fig.tight_layout()

    # plot or save to a file
    if args.png:
        fig.savefig(args.path + args.starname + "_spec.png")
    elif args.pdf:
        fig.savefig(args.path + args.starname + "_spec.pdf")
    else:
        plt.show()
