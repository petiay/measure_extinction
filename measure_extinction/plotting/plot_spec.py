#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import astropy.units as u

from measure_extinction.stardata import StarData


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
    ax.set_ylim(ymin * 0.95, ymax * 1.05)


def plot_multi_spectra(
    starlist, path, mlam4, range, norm_range, spread, pdf, outname="all_spec.pdf"
):
    """
    Plot the observed band and spectral data of multiple stars in the same plot

    Parameters
    ----------
    starlist : numpy.ndarray
        Array of stars for which to plot the spectrum

    path : string
        Path to the data files

    mlam4 : boolean
        Whether or not to multiply the flux F(lambda) by lambda^4 to remove the Rayleigh-Jeans slope

    range : list of 2 floats
        Wavelength range to be plotted (in micron) - [min,max]

    norm_range : list of 2 floats
        Wavelength range to use to normalize the data - [min,max]

    spread : boolean
        Whether or not to spread the spectra out by adding a vertical offset to each spectrum

    pdf : boolean
        Whether or not to save the figure as a pdf file

    outname : string [default="all_spec.pdf"]
        Name for the output pdf file

    Returns
    -------
    Figure with band data points and spectra of multiple stars
    """
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
    fig, ax = plt.subplots(figsize=(15, 20))
    colors = plt.cm.jet(np.linspace(0, 1, len(starlist)))

    # sort the stars according to their flux value at the longest wavelength
    max_waves = np.zeros(len(starlist))
    max_fluxes = np.zeros(len(starlist))
    for i, star in enumerate(starlist):
        # read in the band and spectral data for this star
        starobs = StarData("%s.dat" % star.lower(), path=path, use_corfac=True)

        # find the flux at the longest wavelength
        (waves, fluxes, flux_uncs) = starobs.get_flat_data_arrays(
            ["BAND", "SpeX_SXD", "SpeX_LXD", "IRS"]
        )
        if range is not None:
            max_waves[i] = waves[waves < range[1]][-1]
            max_fluxes[i] = fluxes[waves < range[1]][-1]
        else:
            max_waves[i] = waves[-1]
            max_fluxes[i] = fluxes[-1]
    if mlam4:
        max_fluxes = max_fluxes * max_waves ** 4
    sort_id = np.argsort(max_fluxes)
    sorted_starlist = starlist[sort_id]
    max_waves = max_waves[sort_id]
    max_fluxes = max_fluxes[sort_id]

    for i, star in enumerate(sorted_starlist):
        # read in and plot all bands and spectra for this star
        starobs = StarData("%s.dat" % star.lower(), path=path, use_corfac=True)
        ylabel = r"$F(\lambda)$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}$]"
        # spread out the spectra if requested
        if spread:
            yoffset = 0.5 * i
        else:
            yoffset = 0
        starobs.plot(
            ax,
            pcolor=colors[i],
            norm_wave_range=norm_range * u.micron,
            mlam4=mlam4,
            yoffset=yoffset,
            yoffset_type="add",
        )

        # add the name of the star
        # ax.text(
        #     max_waves[i] * 1.1,
        #     max_fluxes[i] + yoffset,
        #     star,
        #     color=colors[i],
        #     alpha=0.7,
        #     fontsize=fontsize,
        # )

    # zoom in on region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    if norm_range is None:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
    if mlam4:
        ylabel = r"$F(\lambda)\ \lambda^4$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}\ \mu m^4$]"
        outname = outname.replace("spec", "spec_mlam4")
    else:
        ylabel = r"$F(\lambda)$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}$]"
    if spread:
        ylabel = ylabel + " + offset"
    ax.set_ylabel(
        ylabel, fontsize=1.5 * fontsize,
    )
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # use the whitespace better
    fig.tight_layout()

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname)
    else:
        plt.show()


def plot_spectrum(star, path, mlam4, range, pdf):
    """
    Plot the observed band and spectral data of a star

    Parameters
    ----------
    star : string
        Name of the star for which to plot the spectrum

    path : string
        Path to the data files

    mlam4 : boolean
        Whether or not to multiply the flux F(lambda) by lambda^4 to remove the Rayleigh-Jeans slope

    range : list of 2 floats
        Wavelength range to be plotted (in micron) - [min,max]

    pdf : boolean
        Whether or not to save the figure as a pdf file

    Returns
    -------
    Figure with band data points and spectrum
    """

    # setup the plot
    fig, ax = plt.subplots(figsize=(13, 10))
    fontsize = 18
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=1)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    # read in and plot all bands and spectra for this star
    starobs = StarData("%s.dat" % star.lower(), path=path, use_corfac=True)
    starobs.plot(ax, mlam4=mlam4)

    # set the title of the plot
    ax.set_title(star, fontsize=50)

    # define the output name
    outname = star.lower() + "_spec.pdf"

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
    if mlam4:
        ax.set_ylabel(
            r"$F(\lambda)\ \lambda^4$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}\ \mu m^4$]",
            fontsize=1.5 * fontsize,
        )
        outname = outname.replace("spec", "spec_mlam4")
    else:
        ax.set_ylabel(
            r"$F(\lambda)$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}$]",
            fontsize=1.5 * fontsize,
        )
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # use the whitespace better
    fig.tight_layout()

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname)
    else:
        plt.show()


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "starlist",
        nargs="+",
        help="star name or list of star names for which to plot the spectrum",
    )
    parser.add_argument(
        "--path",
        help="path to the data files",
        default=pkg_resources.resource_filename("measure_extinction", "data/"),
    )
    parser.add_argument("--mlam4", help="plot lambda^4*F(lambda)", action="store_true")
    parser.add_argument(
        "--onefig",
        help="whether or not to plot all spectra in the same figure",
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
        "--norm_range",
        nargs="+",
        help="wavelength range to use to normalize the spectrum (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread", help="spread the spectra out over the figure", action="store_true",
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    if args.onefig:
        plot_multi_spectra(
            np.array(
                args.starlist
            ),  # convert the type of "starlist" from list to numpy array (to enable sorting later on)
            args.path,
            args.mlam4,
            args.range,
            args.norm_range,
            args.spread,
            args.pdf,
        )
    else:
        for star in args.starlist:
            plot_spectrum(star, args.path, args.mlam4, args.range, args.pdf)
