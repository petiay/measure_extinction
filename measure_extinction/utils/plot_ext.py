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

def plot_extinction(starname,path,alax,extmodels,powerlaw,pdf):
    # read in the extinction curve data
    extdata = ExtData("%s%s_ext.fits" % (path,starname))

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

    # plot the extinction curve
    extdata.plot(ax, alax=alax)

    # fix the x,y plot limits
    # ax.set_xlim(ax.get_xlim())
    # ax.set_xlim(0.1, 2.5)
    ax.set_ylim(ax.get_ylim())

    # finish configuring the plot
    ax.set_yscale("linear")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.5 * fontsize)
    if alax:
        ytype = 'alax'
    else:
        ytype = extdata.type
    ax.set_ylabel(extdata._get_ext_ytitle(ytype=ytype), fontsize=1.5 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")
    ax.set_title(starname, fontsize=50)

    # plot extinction models
    if extmodels:
        x = np.arange(0.12, 3.0, 0.01) * u.micron
        Rvs = [2.0, 3.1, 4.0, 5.0]
        for cRv in Rvs:
            if alax:
                if extdata.type_rel_band != "V":
                    emod = CCM89(cRv)
                    indx, = np.where(extdata.type_rel_band == extdata.names["BAND"])
                    axav = emod(extdata.waves["BAND"][indx[0]])
                else:
                    axav = 1.0

            t = CCM89(Rv=cRv)
            ax.plot(
                x, t(x) / axav, "k--", linewidth=2, label="R(V) = {:4.2f}".format(cRv)
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
    #ax.legend()
    fig.tight_layout()

    # plot or save to a file
    if pdf:
        fig.savefig("%s%s_ext.pdf" % (path,starname))
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star for which to plot the extinction curve")
    parser.add_argument("--path", help="path to data files", default=pkg_resources.resource_filename('measure_extinction', 'data/'))
    parser.add_argument("--alax", help="plot A(lambda)/A(X)", action="store_true")
    parser.add_argument(
        "--extmodels", help="plot extinction curve models", action="store_true"
    )
    parser.add_argument(
        "--powerlaw", help="plot NIR powerlaw model", action="store_true"
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    plot_extinction(args.starname,args.path,args.alax,args.extmodels,args.powerlaw,args.pdf)
