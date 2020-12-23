# This script is intended to automate the calculation and plotting of the extinction curves for all stars.
import argparse
import pkg_resources
import pandas as pd

from measure_extinction.utils.calc_ext import calc_extinction
from measure_extinction.plotting.plot_ext import plot_multi_extinction, plot_extinction


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="path to the data files",
        default=pkg_resources.resource_filename("measure_extinction", "data/"),
    )
    parser.add_argument("--alax", help="plot A(lambda)/A(X)", action="store_true")
    parser.add_argument(
        "--average", help="plot the average extinction curve", action="store_true"
    )
    parser.add_argument(
        "--extmodels", help="plot extinction curve models", action="store_true"
    )
    parser.add_argument(
        "--powerlaw", help="plot NIR powerlaw model", action="store_true"
    )
    parser.add_argument("--HI_lines", help="indicate the HI-lines", action="store_true")
    parser.add_argument(
        "--onefig",
        help="whether or not to plot all curves in the same figure",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread", help="spread the curves out over the figure", action="store_true"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="data type(s) to be excluded from the plot",
        type=str,
        default=[],
    )
    args = parser.parse_args()

    # read the list of stars for which to measure and plot the extinction curve
    table = pd.read_table(args.path + "red-comp.list", comment="#")
    redstars = table["reddened"]
    compstars = table["comparison"]
    starpair_list = []

    # calculate and plot the extinction curve for every star
    for redstar, compstar in zip(redstars, compstars):
        print("reddened star: ", redstar, "/ comparison star:", compstar)
        calc_extinction(redstar, compstar, args.path)
        # create the starpair_list
        starpair_list.append(redstar + "_" + compstar)

    if args.onefig:  # plot all curves in the same figure
        plot_multi_extinction(
            starpair_list,
            args.path,
            args.alax,
            args.average,
            args.extmodels,
            args.powerlaw,
            args.HI_lines,
            args.range,
            args.spread,
            args.exclude,
            pdf=True,
        )
    else:  # plot all curves separately
        if args.spread:
            parser.error(
                "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the curves if there is more than one curve in the same plot."
            )
        if args.average:
            parser.error(
                "The flag --average can only be used in combination with the flag --onefig. It only makes sense to add the average extinction curve to a plot with multiple curves."
            )
        for redstar, compstar in zip(redstars, compstars):
            plot_extinction(
                redstar + "_" + compstar,
                args.path,
                args.alax,
                args.extmodels,
                args.powerlaw,
                args.HI_lines,
                args.range,
                args.exclude,
                pdf=True,
            )
