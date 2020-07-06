#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from measure_extinction.stardata import StarData


def plot_spec(starname, path, mlam4, range, pdf):
    # read in the observed data of the star
    # fstarname, file_path = get_full_starfile(args.starname)
    # --> this is currently not working, because the default path in get_full_starfile is not accessible as an external user.

    starobs = StarData("%s.dat" % starname.lower(), path=path, use_corfac=True)

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

    # plot all bands and spectra for this star
    starobs.plot(ax, mlam4=mlam4)

    # finish configuring the plot
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
    if mlam4:
        ax.set_ylabel(
            r"$F(\lambda)\ \lambda^4$ [$ergs\ cm^{-2}\ s\ \AA\ \mu m^4$]",
            fontsize=1.5 * fontsize,
        )
        outname = path + starname.lower() + "_spec_mlam4.pdf"
    else:
        ax.set_ylabel(
            r"$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]", fontsize=1.5 * fontsize
        )
        outname = path + starname.lower() + "_spec.pdf"
    ax.set_title(starname.upper(), fontsize=50)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # zoom in on region
    if range is not None:
        ax.set_xlim(range)
        # calculate the appropriate y limits
        ymin = 1
        ymax = 0
        for line in plt.gca().lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()[
                np.logical_and(x_data > range[0], x_data < range[1])
            ]
            if y_data.size != 0 and np.nanmin(y_data) < ymin:
                ymin = np.nanmin(y_data)
            if y_data.size != 0 and np.nanmax(y_data) > ymax:
                ymax = np.nanmax(y_data)
        ax.set_ylim(ymin * 0.95, ymax * 1.05)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # use the whitespace better
    fig.tight_layout()

    # plot or save to a file
    if pdf:
        fig.savefig(outname)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star for which to plot the spectrum")
    parser.add_argument(
        "--path",
        help="path to data files",
        default=pkg_resources.resource_filename("measure_extinction", "data/"),
    )
    parser.add_argument("--mlam4", help="plot lambda^4*F(lambda)", action="store_true")
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # plot the spectrum
    plot_spec(args.starname, args.path, args.mlam4, args.range, args.pdf)
