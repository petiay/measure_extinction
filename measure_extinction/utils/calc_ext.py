#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import matplotlib.pyplot as plt
import matplotlib

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("redstarname", help="name of reddened star")
    parser.add_argument("compstarname", help="name of comparison star")
    parser.add_argument("--path", help="path to data files", default=pkg_resources.resource_filename('measure_extinction', 'data/'))
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # read in the observed data for both stars
    redstarobs = StarData("%s.dat" % args.redstarname, path=args.path)
    compstarobs = StarData("%s.dat" % args.compstarname, path=args.path)

    # calculate the extinction curve
    extdata = ExtData()
    extdata.calc_elx(redstarobs, compstarobs)

    extdata.save(args.path+"%s_ext.fits" % args.redstarname)
