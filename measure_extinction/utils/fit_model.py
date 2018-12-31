#!/usr/bin/env python
# fit a stellar model atmosphere model + dust extinction model to
# observed spectra and photometry

import glob
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl

from measure_extinction.modeldata import ModelData


def fit_model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to star files",
                        default='./')
    return parser


if __name__ == '__main__':
    # commandline parser
    parser = fit_model_parser()
    args = parser.parse_args()
    args.path = '/home/kgordon/Python_git/extstar_data/Models/'

    # get the observed reddened star data
    # TBD

    # get just the filenames
    tlusty_models_fullpath = glob.glob(args.path + 'tlusty_*v2.dat')
    tlusty_models = [tfile[tfile.rfind('/')+1: len(tfile)]
                     for tfile in tlusty_models_fullpath]

    # get the models with just the reddened star band data and spectra
    modinfo = ModelData(tlusty_models[0:100],
                        path=args.path)

    # parameters
    # [logT, logg, logZ,
    #  Av, Rv, C2, C3, C4, x0, gamma]
    params = [4.3, 4.0, 0.0,
              1.0, 3.1, 0.679, 2.991, 0.319, 4.592, 0.922]

    # intrinsic sed
    modsed = modinfo.get_stellar_sed(params[0:3])

    # extinguished sed
    ext_modsed = modinfo.get_dust_extinguished_sed(params[3:10], modsed)

    # plot the SEDs

    # plotting setup for easier to read plots
    fontsize = 18
    font = {'size': fontsize}
    mpl.rc('font', **font)
    mpl.rc('lines', linewidth=1)
    mpl.rc('axes', linewidth=2)
    mpl.rc('xtick.major', width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.major', width=2)
    mpl.rc('ytick.minor', width=2)

    # setup the plot
    fig, ax = plt.subplots(figsize=(13, 10))

    # plot the bands and all spectra for this star
    for cspec in modinfo.fluxes.keys():
        if cspec == 'BAND':
            ptype = 'ko'
        else:
            ptype = '-'

        ax.plot(modinfo.waves[cspec], modsed[cspec], ptype,
                label=cspec)
        ax.plot(modinfo.waves[cspec], ext_modsed[cspec], ptype,
                label=cspec)

    # finish configuring the plot
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$ [$\mu m$]', fontsize=1.3*fontsize)
    ax.set_ylabel('$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]',
                  fontsize=1.3*fontsize)
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')

    ax.legend()

    # use the whitespace better
    fig.tight_layout()

    plt.show()
