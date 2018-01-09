from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from extdata import ExtData

__all__ = ["plot_extdata"]


def _get_ext_ytitle(exttype):
    """
    Format the extinction type nicely for plotting

    Parameters
    ----------
    exttype : string
        type of extinction curve (e.g., elv, alav, elvebv)

    Returns
    -------
    ptype : string
        Latex formated string for plotting
    """
    if exttype == 'elv':
        return '$E(\lambda - V)$'
    elif exttype == 'elvebv':
        return '$E(\lambda - V)/E(B - V)$'
    elif exttype == 'alav':
        return '$A(\lambda)/A(V)$'
    else:
        return "%s (not found)" % exttype


def plot_extdata(ax, extdata, fontsize=18):
    """
    Plot an extinction curve

    Parameters
    ----------
    ax : matplotlib plot object

    extdata : ExtData object
        contains the measured extinction curve

    fontsize : float, optional [default=18]
        base font size
    """
    for curtype in extdata.waves.keys():
        gindxs, = np.where(extdata.npts[curtype] > 0)
        if len(gindxs) < 20:
            # plot small number of points (usually BANDS data) as
            # points with errorbars
            ax.errorbar(extdata.waves[curtype][gindxs],
                        extdata.exts[curtype][gindxs],
                        yerr=extdata.uncs[curtype][gindxs],
                        fmt='o')
        else:
            ax.plot(extdata.waves[curtype][gindxs],
                    extdata.exts[curtype][gindxs],
                    '-')

    # finish configuring the plot
    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$ [$\mu m$]', fontsize=1.3*fontsize)
    ax.set_ylabel(_get_ext_ytitle(extdata.type),
                  fontsize=1.3*fontsize)
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("extfile", help="file with saved extinction curve")
    parser.add_argument("--path", help="base path to extinction curves",
                        default="./")
    parser.add_argument("--png", help="save figure as a png file",
                        action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file",
                        action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()

    # read in the observed data for both stars
    extdata = ExtData('%s/%s' % (args.path, args.extfile))

    # plotting setup for easier to read plots
    fontsize = 18
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=1)
    matplotlib.rc('axes', linewidth=2)
    matplotlib.rc('xtick.major', width=2)
    matplotlib.rc('xtick.minor', width=2)
    matplotlib.rc('ytick.major', width=2)
    matplotlib.rc('ytick.minor', width=2)

    # setup the plot
    fig, ax = plt.subplots(figsize=(10, 13))

    # plot the bands and all spectra for this star
    plot_extdata(ax, extdata)

    # use the whitespace better
    fig.tight_layout()

    # plot or save to a file
    save_str = '_ext'
    if args.png:
        fig.savefig(args.starname.replace('.dat', save_str+'.png'))
    elif args.eps:
        fig.savefig(args.starname.replace('.dat', save_str+'.eps'))
    elif args.pdf:
        fig.savefig(args.starname.replace('.dat', save_str+'.pdf'))
    else:
        plt.show()
