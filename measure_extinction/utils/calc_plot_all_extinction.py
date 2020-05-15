# This script is intended to automate the calculation and plotting of the extinction curves for all stars.

from measure_extinction.utils.calc_ext import calc_extinction
from measure_extinction.utils.plot_ext import plot_extinction

import argparse
import pkg_resources
import pandas as pd

# commandline parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    help="path to the data files",
    default=pkg_resources.resource_filename("measure_extinction", "data/"),
)
parser.add_argument("--alax", help="plot A(lambda)/A(X)", action="store_true")
parser.add_argument(
    "--extmodels", help="plot extinction curve models", action="store_true"
)
parser.add_argument("--powerlaw", help="plot NIR powerlaw model", action="store_true")
args = parser.parse_args()

# read the list of stars for which to measure the extinction curve
table = pd.read_table(args.path + "red-comp.list", comment="#")
stars = table["reddened"]

# calculate and plot the extinction curve for every stars
for i, star in enumerate(stars):
    print("reddened star: ", star, "/ comparison star:", table["comparison"][i])
    calc_extinction(star, table["comparison"][i], args.path)
    plot_extinction(star, args.path, args.alax, args.extmodels, args.powerlaw, pdf=True)
