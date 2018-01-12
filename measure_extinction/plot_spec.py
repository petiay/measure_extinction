from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import matplotlib

from stardata import StarData


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
    starobs.plot_obsdata(ax, fontsize=fontsize)

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
