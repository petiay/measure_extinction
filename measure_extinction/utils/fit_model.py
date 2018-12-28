#!/usr/bin/env python
# fit a stellar model atmosphere model + dust extinction model to
# observed spectra and photometry

import glob

import argparse

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
    tlusty_models_fullpath = glob.glob(args.path + 'tlusty_*.dat')
    tlusty_models = [tfile[tfile.rfind('/')+1: len(tfile)]
                     for tfile in tlusty_models_fullpath]

    # get the models with just the reddened star band data and spectra
    modinfo = ModelData(tlusty_models[0:10],
                        path=args.path)
