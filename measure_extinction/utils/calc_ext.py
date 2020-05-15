#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData


def calc_extinction(redstarname, compstarname, path):
    # read in the observed data for both stars
    redstarobs = StarData("%s.dat" % redstarname, path=path)
    compstarobs = StarData("%s.dat" % compstarname, path=path)

    # calculate the extinction curve
    extdata = ExtData()
    extdata.calc_elx(redstarobs, compstarobs)

    extdata.save(path + "%s_ext.fits" % redstarname)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("redstarname", help="name of reddened star")
    parser.add_argument("compstarname", help="name of comparison star")
    parser.add_argument(
        "--path",
        help="path to data files",
        default=pkg_resources.resource_filename("measure_extinction", "data/"),
    )
    args = parser.parse_args()

    calc_extinction(args.redstarname, args.compstarname, args.path)
