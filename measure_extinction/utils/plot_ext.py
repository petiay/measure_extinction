#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

import numpy as np
import astropy.units as u

from measure_extinction.extdata import ExtData

from dust_extinction.parameter_averages import CCM89


def irpowerlaw(x, a, alpha, c):
    return a * (x ** (-1.0 * alpha) - c)


def irpowerlaw_18(x, a, c):
    return a * (x ** (-1.8) - c)


# function to zoom in on a certain wavelength range
def zoom(ax, range):
    ax.set_xlim(range)

    # calculate the appropriate y limits
    ymin, ymax = np.inf, -np.inf
    for line in ax.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()[np.logical_and(x_data > range[0], x_data < range[1])]
        if y_data.size != 0 and np.nanmin(y_data) < ymin:
            ymin = np.nanmin(y_data)
        if y_data.size != 0 and np.nanmax(y_data) > ymax:
            ymax = np.nanmax(y_data)
        h = ymax - ymin
    ax.set_ylim(ymin - 0.05 * h, ymax + 0.05 * h)


def plot_extinction(starlist, path, alax, extmodels, powerlaw, onefig, range, pdf):
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

    # plot all curves in the same figure
    if onefig:
        # setup the plot
        fig, ax = plt.subplots(figsize=(16, 13))
        colors = plt.cm.jet(np.linspace(0, 1, len(starlist)))

        # sort the stars according to their extinction value at the longest wavelength
        max_yvals = np.zeros(len(starlist))
        max_waves = np.zeros(len(starlist))
        for i, star in enumerate(starlist):
            # read in the extinction curve data
            extdata = ExtData("%s%s_ext.fits" % (path, star))
            if alax:
                extdata.trans_elv_alav()
            # find the extinction value at the longest wavelength
            (wave, y, y_unc) = extdata.get_fitdata(
                ["BAND", "SpeX_SXD", "SpeX_LXD", "IRS"]
            )
            if range is not None:
                max_waves[i] = wave.value[wave.value < range[1]][0]
                max_yvals[i] = y[wave.value < range[1]][0]
            else:
                max_yvals[i] = y[0]
                max_waves[i] = wave.value[0]
        sort_id = np.argsort(max_yvals)
        sorted_starlist = starlist[sort_id]
        max_yvals = max_yvals[sort_id]
        max_waves = max_waves[sort_id]

        for i, star in enumerate(sorted_starlist):
            # read in the extinction curve data
            extdata = ExtData("%s%s_ext.fits" % (path, star))

            # plot the extinction curve
            extdata.plot(ax, alax=alax, color=colors[i], alpha=0.5)

            # add the name of the star
            ax.text(
                max_waves[i] * 1.1,
                max_yvals[i],
                star,
                color=colors[i],
                alpha=0.5,
                fontsize=0.7 * fontsize,
            )

        # finish configuring the plot
        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
        ax.set_ylabel(
            extdata._get_ext_ytitle(ytype=extdata.type), fontsize=1.5 * fontsize
        )
        ax.tick_params("both", length=10, width=2, which="major")
        ax.tick_params("both", length=5, width=1, which="minor")
        outname = "%sall_ext_%s.pdf" % (path, extdata.type)
        plt.margins(x=0.1)

        # zoom in on region
        if range is not None:
            zoom(ax, range)
            outname = outname.replace(".pdf", "_zoom.pdf")

        # use the whitespace better
        fig.tight_layout()

        if pdf:
            fig.savefig(outname)
        else:
            plt.show()

    else:  # plot all curves separately
        for star in starlist:
            # setup the plot
            fig, ax = plt.subplots(figsize=(13, 10))

            # read in the extinction curve data
            extdata = ExtData("%s%s_ext.fits" % (path, star))

            # plot the extinction curve
            extdata.plot(ax, alax=alax)

            # finish configuring the plot
            ax.set_yscale("linear")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
            ax.set_ylabel(
                extdata._get_ext_ytitle(ytype=extdata.type), fontsize=1.5 * fontsize
            )
            ax.tick_params("both", length=10, width=2, which="major")
            ax.tick_params("both", length=5, width=1, which="minor")
            ax.set_title(star, fontsize=50)
            outname = "%s%s_ext_%s.pdf" % (path, star, extdata.type)

            # zoom in on region
            if range is not None:
                zoom(ax, range)
                outname = outname.replace(".pdf", "_zoom.pdf")

            # plot extinction models
            if extmodels:
                x = np.arange(0.12, 3.0, 0.01) * u.micron
                Rvs = [2.0, 3.1, 4.0, 5.0]
                for cRv in Rvs:
                    if alax:
                        if extdata.type_rel_band != "V":
                            emod = CCM89(cRv)
                            (indx,) = np.where(
                                extdata.type_rel_band == extdata.names["BAND"]
                            )
                            axav = emod(extdata.waves["BAND"][indx[0]])
                        else:
                            axav = 1.0

                    t = CCM89(Rv=cRv)
                    ax.plot(
                        x,
                        t(x) / axav,
                        "k--",
                        linewidth=2,
                        label="R(V) = {:4.2f}".format(cRv),
                    )

            # plot NIR power law model
            if powerlaw:
                ftype = "BAND"
                gbool = np.all(
                    [
                        (extdata.npts[ftype] > 0),
                        (extdata.waves[ftype] > 1.0 * u.micron),
                        (extdata.waves[ftype] < 40.0 * u.micron),
                    ],
                    axis=0,
                )
                xdata = extdata.waves[ftype][gbool].value
                ydata = extdata.exts[ftype][gbool]
                func = irpowerlaw
                # func = irpowerlaw_18
                popt, pcov = curve_fit(func, xdata, ydata)
                ax.plot(
                    xdata,
                    func(xdata, *popt),
                    "-",
                    label="fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt),
                )
                # ax.plot(
                #    xdata, func(xdata, *popt), "-", label="fit: a=%5.3f, c=%5.3f" % tuple(popt)
                # )

                mod_x = np.arange(1.0, 40.0, 0.1)
                mod_y = func(mod_x, *popt)
                ax.plot(mod_x, mod_y, "--", label="A(V) = %5.2f" % (popt[0] * popt[2]))
                # ax.plot(mod_x, mod_y, "--", label="A(V) = %5.2f" % (popt[0] * popt[1]))

            # use the whitespace better
            fig.tight_layout()

            if pdf:
                fig.savefig(outname)
                plt.close()
            else:
                plt.show()


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "starlist",
        nargs="+",
        help="star name or list of star names for which to plot the extinction curve",
    )
    parser.add_argument(
        "--path",
        help="path to data files",
        default=pkg_resources.resource_filename("measure_extinction", "data/"),
    )
    parser.add_argument("--alax", help="plot A(lambda)/A(X)", action="store_true")
    parser.add_argument(
        "--extmodels", help="plot extinction curve models", action="store_true"
    )
    parser.add_argument(
        "--powerlaw", help="plot NIR powerlaw model", action="store_true"
    )
    parser.add_argument(
        "--onefig",
        help="whether or not to plot all curves in the same figure",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    plot_extinction(
        args.starlist,
        args.path,
        args.alax,
        args.extmodels,
        args.powerlaw,
        args.onefig,
        args.range,
        args.pdf,
    )
