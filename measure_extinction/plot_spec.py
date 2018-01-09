from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from stardata import StarData


def plot_obsdata(ax, starobs):
    """
    Plot all the data for a star (bands and spectra)

    Parameters
    ----------
    ax : matplotlib plot object

    starobs : StarData object
        contains all the observations
    """
    # plot the bands and all spectra for this star
    for curtype in starobs.data.keys():
        gindxs, = np.where(starobs.data[curtype].npts > 0)
        print(curtype, len(gindxs))
        if len(gindxs) < 20:
            # plot small number of points (usually BANDS data) as
            # points with errorbars
            ax.errorbar(starobs.data[curtype].waves[gindxs],
                        starobs.data[curtype].fluxes[gindxs],
                        yerr=starobs.data[curtype].uncs[gindxs],
                        fmt='o')
        else:
            ax.plot(starobs.data[curtype].waves[gindxs],
                    starobs.data[curtype].fluxes[gindxs],
                    '-')

    # finish configuring the plot
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$ [$\mu m$]', fontsize=1.3*fontsize)
    ax.set_ylabel('$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]',
                  fontsize=1.3*fontsize)
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star")
    parser.add_argument("--png", help="save figure as a png file",
                        action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file",
                        action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()

    # read in the observed data on the star
    starobs = StarData('DAT_files/%s.dat' % args.starname,
                       path='/home/kgordon/Dust/Ext/')

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
    plot_obsdata(ax, starobs)

    # use the whitespace better
    fig.tight_layout()

    # plot or save to a file
    save_str = '_spec'
    if args.png:
        fig.savefig(args.starname.replace('.dat', save_str+'.png'))
    elif args.eps:
        fig.savefig(args.starname.replace('.dat', save_str+'.eps'))
    elif args.pdf:
        fig.savefig(args.starname.replace('.dat', save_str+'.pdf'))
    else:
        plt.show()
