#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from scipy.optimize import curve_fit
from measure_extinction.extdata import ExtData
from dust_extinction.parameter_averages import CCM89


def elx_powerlaw(x, a, alpha, c):
    return a * x ** -alpha - c


def alav_powerlaw(x, a, alpha):
    return a * x ** -alpha


# this function is currently not used
# def irpowerlaw_18(x, a, c):
#    return a * x ** -1.8 - c


def plot_extmodels(extdata, alax):
    """
    Plot Milky Way extinction curve models of Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245), only possible for wavelengths between 0.1 and 3.33 micron

    Parameters
    ----------
    extdata : ExtData
        Extinction data under consideration

    alax : boolean
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    Returns
    -------
    Overplots extinction curve models
    """
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
            x.value,
            y,
            style[i],
            color="k",
            alpha=0.7,
            linewidth=1,
            label="R(V) = {:4.2f}".format(cRv),
        )
        plt.legend()


def plot_powerlaw(extdata, alax):
    """
    Fit and plot a NIR powerlaw model to the band data between 1 and 40 micron

    Parameters
    ----------
    extdata : ExtData
        Extinction data under consideration

    alax : boolean
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    Returns
    -------
    Overplots a fitted NIR powerlaw model
    """
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


def zoom(ax, range):
    """
    Zoom in on the requested wavelength range by setting the axes limits to this range

    Parameters
    ----------
    ax : AxesSubplot
        Axes of plot for which new limits need to be set

    range : list of 2 floats
        Wavelength range to be plotted (in micron) - [min,max]

    Returns
    -------
    Sets the axes limits to the requested range
    """
    # set the x axis limits
    ax.set_xlim(range)

    # calculate the appropriate y axis limits
    ymin, ymax = np.inf, -np.inf
    for line in ax.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()[
            np.logical_and(x_data >= range[0], x_data <= range[1])
        ]
        if y_data.size != 0 and np.nanmin(y_data) < ymin:
            ymin = np.nanmin(y_data)
        if y_data.size != 0 and np.nanmax(y_data) > ymax:
            ymax = np.nanmax(y_data)
    h = ymax - ymin
    ax.set_ylim(ymin - 0.05 * h, ymax + 0.05 * h)


def plot_multi_extinction(
    starlist,
    path,
    alax=False,
    extmodels=False,
    powerlaw=False,
    range=None,
    spread=False,
    exclude=[],
    pdf=False,
):
    """
    Plot the extinction curves of multiple stars in the same plot

    Parameters
    ----------
    starlist : list of strings
        List of stars for which to plot the extinction curve

    path : string
        Path to the data files

    alax : boolean [default=False]
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    extmodels: boolean [default=False]
        Whether or not to overplot Milky Way extinction curve models

    powerlaw: boolean [default=False]
        Whether or not to fit and overplot a NIR powerlaw model

    range : list of 2 floats [default=None]
        Wavelength range to be plotted (in micron) - [min,max]

    spread : boolean [default=False]
        Whether or not to spread the extinction curves out by adding a vertical offset to each curve

    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., IRS)

    pdf : boolean [default=False]
        Whether or not to save the figure as a pdf file

    Returns
    -------
    Figure with extinction curves of multiple stars
    """
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

    # setup the plot
    fig, ax = plt.subplots(figsize=(15, len(starlist) * 1.25))
    colors = plt.get_cmap("tab10")

    for i, star in enumerate(starlist):
        # read in the extinction curve data
        extdata = ExtData("%s%s_ext.fits" % (path, star.lower()))

        # spread out the curves if requested
        if spread:
            yoffset = 0.3 * i
        else:
            yoffset = 0

        # determine where to add the name of the star
        # find the shortest plotted wavelength
        (waves, exts, ext_uncs) = extdata.get_fitdata(extdata.waves.keys() - exclude)
        if range is not None:
            waves = waves[waves.value >= range[0]]
        min_wave = waves[-1]
        # find out which data type corresponds with this wavelength
        for data_type in extdata.waves.keys():
            if data_type in exclude:
                continue
            used_waves = extdata.waves[data_type][extdata.npts[data_type] > 0]
            if min_wave in used_waves:
                ann_key = data_type
        ann_range = [min_wave, min_wave] * u.micron

        # plot the extinction curve
        extdata.plot(
            ax,
            color=colors(i % 10),
            alpha=0.7,
            alax=alax,
            exclude=exclude,
            yoffset=yoffset,
            annotate_key=ann_key,
            annotate_wave_range=ann_range,
            annotate_text=star,
            annotate_yoffset=0.05,
            annotate_color=colors(i % 10),
        )

        # fit and plot a NIR powerlaw model if requested
        if powerlaw:
            plot_powerlaw(extdata, alax)

    # overplot Milky Way extinction curve models if requested
    if extmodels:
        if alax:
            plot_extmodels(extdata, alax)
        else:
            warnings.warn(
                "Overplotting Milky Way extinction curve models on a figure with multiple observed extinction curves in E(lambda-V) units is disabled, because the model curves in these units are different for every star, and would overload the plot. Please, do one of the following if you want to overplot Milky Way extinction curve models: 1) Use the flag --alax to plot ALL curves in A(lambda)/A(V) units, OR 2) Plot all curves separately by removing the flag --onefig.",
                stacklevel=2,
            )

    # define the output name
    outname = "all_ext_%s.pdf" % (extdata.type)

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
    ax.set_ylabel(extdata._get_ext_ytitle(ytype=extdata.type), fontsize=1.5 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # use the whitespace better
    fig.tight_layout()

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname)
    else:
        plt.show()


def plot_extinction(
    star,
    path,
    alax=False,
    extmodels=False,
    powerlaw=False,
    range=None,
    exclude=[],
    pdf=False,
):
    """
    Plot the extinction curve of a star

    Parameters
    ----------
    star : string
        Name of the star for which to plot the extinction curve

    path : string
        Path to the data files

    alax : boolean [default=False]
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    extmodels: boolean [default=False]
        Whether or not to overplot Milky Way extinction curve models

    powerlaw: boolean [default=False]
        Whether or not to fit and overplot a NIR powerlaw model

    range : list of 2 floats [default=None]
        Wavelength range to be plotted (in micron) - [min,max]

    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., IRS)

    pdf : boolean [default=False]
        Whether or not to save the figure as a pdf file

    Returns
    -------
    Figure with extinction curve
    """

    # setup the plot
    fig, ax = plt.subplots(figsize=(13, 10))
    fontsize = 18
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    # read in and plot the extinction curve data for this star
    extdata = ExtData("%s%s_ext.fits" % (path, star.lower()))
    extdata.plot(ax, alax=alax, exclude=exclude)

    # set the title of the plot
    ax.set_title(star, fontsize=50)

    # define the output name
    outname = "%s_ext_%s.pdf" % (star.lower(), extdata.type)

    # fit and plot a NIR powerlaw model if requested
    if powerlaw:
        plot_powerlaw(extdata, alax)

    # plot Milky Way extinction models if requested
    if extmodels:
        plot_extmodels(extdata, alax)

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
    ax.set_ylabel(extdata._get_ext_ytitle(ytype=extdata.type), fontsize=1.5 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # use the whitespace better
    fig.tight_layout()

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname)
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
        help="path to the data files",
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
    parser.add_argument(
        "--spread",
        help="spread the curves out over the figure; can only be used in combination with --onefig",
        action="store_true",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="data type(s) to be excluded from the plotting",
        type=str,
        default=[],
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    if args.onefig:  # plot all curves in the same figure
        plot_multi_extinction(
            args.starlist,
            args.path,
            args.alax,
            args.extmodels,
            args.powerlaw,
            args.range,
            args.spread,
            args.exclude,
            args.pdf,
        )
    else:  # plot all curves separately
        if args.spread:
            parser.error(
                "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the curves if there is more than one curve in the same plot."
            )
        for star in args.starlist:
            plot_extinction(
                star,
                args.path,
                args.alax,
                args.extmodels,
                args.powerlaw,
                args.range,
                args.exclude,
                args.pdf,
            )
