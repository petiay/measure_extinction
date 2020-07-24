# This script is intended to automate the different steps to create SpeX spectra for all stars:
# 	- "merging" and rebinning the SpeX spectra
# 	- scaling the SpeX spectra
# 	- plotting the spectra

from measure_extinction.utils.merge_spex_spec import merge_spex
from measure_extinction.utils.scale_spex_spec import calc_save_corfac_spex
from measure_extinction.utils.plot_spec import plot_spec

import argparse
import pkg_resources
import glob
import os

if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpath",
        help="path where original SpeX ASCII files are stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Orig/NIR"),
    )
    parser.add_argument(
        "--spex_path",
        help="path where merged SpeX spectra will be stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Spectra"),
    )
    parser.add_argument("--mlam4", help="plot lambda^4*F(lambda)", action="store_true")
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted (in micron)",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    # collect the star names in the input directory
    stars = []
    for filename in glob.glob(args.inpath + "/*.txt"):
        starname = os.path.basename(filename).split("_")[0]
        if starname not in stars:
            stars.append(starname)

    # do the different steps for all the stars
    for star in stars:
        print(star.upper())
        merge_spex(star, args.inpath, args.spex_path, outname=None)
        calc_save_corfac_spex(
            star, os.path.dirname(os.path.normpath(args.spex_path)) + "/"
        )
        plot_spec(
            star,
            os.path.dirname(os.path.normpath(args.spex_path)) + "/",
            args.mlam4,
            args.range,
            pdf=True,
        )
