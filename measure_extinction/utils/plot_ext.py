from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import astropy.units as u

from measure_extinction.extdata import ExtData

from dust_extinction.parameter_averages import CCM89

if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("extfile", help="file with saved extinction curve")
    parser.add_argument("--path", help="base path to extinction curves",
                        default="./")
    parser.add_argument("--alav", help="plot A(lambda)/A(V)",
                        action="store_true")
    parser.add_argument("--extmodels", help="plot extinction curve models",
                        action="store_true")
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
    fig, ax = plt.subplots(figsize=(13, 10))

    # plot the bands and all spectra for this star
    extdata.plot_ext(ax, alav=args.alav)

    # fix the x,y plot limits
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # finish configuring the plot
    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$ [$\mu m$]', fontsize=1.3*fontsize)
    if args.alav:
        ytype = 'alav'
    else:
        ytype = 'elv'
    ax.set_ylabel(extdata._get_ext_ytitle(ytype),
                  fontsize=1.3*fontsize)
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')

    # plot extinctionm models if asked
    if args.extmodels:
        x = np.arange(0.12, 3.0, 0.01)*u.micron
        Rvs = [2.0, 3.1, 4.0, 5.0]
        for cRv in Rvs:
            t = CCM89(Rv=cRv)
            ax.plot(x, t(x), 'k--', linewidth=2)

    # use the whitespace better
    fig.tight_layout()

    # plot or save to a file
    save_str = '_ext'
    if args.png:
        fig.savefig(args.extfile.replace('.fits', save_str+'.png'))
    elif args.eps:
        fig.savefig(args.extfile.replace('.fits', save_str+'.eps'))
    elif args.pdf:
        fig.savefig(args.extfile.replace('.fits', save_str+'.pdf'))
    else:
        plt.show()
