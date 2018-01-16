from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import matplotlib

from extdata import ExtData


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
    extdata.plot_extdata(ax)

    # finish configuring the plot
    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$ [$\mu m$]', fontsize=1.3*fontsize)
    ax.set_ylabel(extdata._get_ext_ytitle(extdata.type),
                  fontsize=1.3*fontsize)
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')

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
