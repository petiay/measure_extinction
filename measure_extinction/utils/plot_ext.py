#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import astropy.units as u

from measure_extinction.extdata import ExtData

from dust_extinction.parameter_averages import CCM89


def elx_powerlaw(x, a, alpha, c):
    return a * x ** -alpha - c


def alav_powerlaw(x, a, alpha):
    return a * x ** -alpha


# this function is currently not used
# def irpowerlaw_18(x, a, c):
#    return a * x ** -1.8 - c

# function to plot Milky Way extinction curve models of Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245), only possible for wavelengths between 0.1 and 3.33 micron.
def plot_extmodels(extdata, alax):
    x = np.arange(0.1, 3.33, 0.01) * u.micron
    Rvs = [2.0, 3.1, 4.0, 5.0]
    style = ["--", "-", ":", "-."]
    for i, cRv in enumerate(Rvs):
        curve = CCM89(Rv=cRv)
        if alax:
            if extdata.type_rel_band != "V":
                emod = CCM89(cRv)
                (indx,) = np.where(extdata.type_rel_band == extdata.names["BAND"])
                axav = emod(extdata.waves["BAND"][indx[0]])
            else:
                axav = 1.0
            y = curve(x) / axav
        else:
            # compute A(V)
            extdata.calc_AV()
            # convert the model curve from A(lambda)/A(V) to E(lambda-V), using the computed A(V) of the data.
            y = (curve(x) - 1) * extdata.columns["AV"]
        plt.plot(
            x,
            y,
            style[i],
            color="k",
            alpha=0.7,
            linewidth=1,
            label="R(V) = {:4.2f}".format(cRv),
        )
        plt.legend()


# function to fit and plot a NIR powerlaw model to the band data between 1 and 40 micron
def plot_powerlaw(extdata, alax):
    # retrieve the band data to be fitted
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
    # fit the data points with a powerlaw function
    if alax:
        func = alav_powerlaw
        labeltxt = r"fit: $%5.2f \lambda ^{-%5.2f}$"
    else:
        func = elx_powerlaw
        labeltxt = r"fit: $%5.2f \lambda ^{-%5.2f} - %5.2f$"
    popt, pcov = curve_fit(func, xdata, ydata)
    # plot the fitted curve
    plt.plot(
        xdata, func(xdata, *popt), "-", label=labeltxt % tuple(popt),
    )

    # plot the model line from 1 to 40 micron
    mod_x = np.arange(1.0, 40.0, 0.1)
    mod_y = func(mod_x, *popt)
    if alax:
        av = extdata.columns["AV"]
    else:
        av = popt[2]
    plt.plot(mod_x, mod_y, "--", label="A(V) = %5.2f" % av)
    plt.legend()


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
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

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
            extdata = ExtData("%s%s_ext.fits" % (path, star.lower()))

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

            # fit and plot a NIR powerlaw model if requested
            if powerlaw:
                plot_powerlaw(extdata, alax)

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

        # zoom in on region if requested
        if range is not None:
            zoom(ax, range)
            outname = outname.replace(".pdf", "_zoom.pdf")

        # overplot Milky Way extinction curve models if requested
        if extmodels:
            if alax:
                plot_extmodels(extdata, alax)
            else:
                warnings.warn(
                    "Overplotting Milky Way extinction curve models on a figure with multiple observed extinction curves in E(lambda-V) units is disabled, because the model curves in these units are different for every star, and would overload the plot. Please, do one of the following if you want to overplot Milky Way extinction curve models: 1) Use the flag --alax to plot ALL curves in A(lambda)/A(V) units, OR 2) Plot all curves separately by removing the flag --onefig.",
                    stacklevel=2,
                )

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
            extdata = ExtData("%s%s_ext.fits" % (path, star.lower()))

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
            outname = "%s%s_ext_%s.pdf" % (path, star.lower(), extdata.type)

            # zoom in on region if requested
            if range is not None:
                zoom(ax, range)
                outname = outname.replace(".pdf", "_zoom.pdf")

            # plot Milky Way extinction models if requested
            if extmodels:
                plot_extmodels(extdata, alax)

            # fit and plot a NIR powerlaw model if requested
            if powerlaw:
                plot_powerlaw(extdata, alax)

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
        "--powerlaw", help="fit and plot NIR powerlaw model", action="store_true"
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
        np.array(
            args.starlist
        ),  # convert the type of "starlist" from list to numpy array (to enable sorting later on)
        args.path,
        args.alax,
        args.extmodels,
        args.powerlaw,
        args.onefig,
        args.range,
        args.pdf,
    )
