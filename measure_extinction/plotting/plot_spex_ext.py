#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

from measure_extinction.plotting.plot_ext import plot_multi_extinction


def plot_extinction_curves():
    # define the path and the names of the stars (first the main sequence stars and then the giant stars, sorted by spectral type from B9 to O0)
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    stars = [
        "HD034921",
        "HD013338",
        "HD037023",
        "HD037020",
        "HD052721",
        "HD037022",
        "HD038087",
        "HD014422",
        "HD294264",
        "HD206773",
        "HD037061",
        "HD029309",
        "HD283809",
        "HD156247",
        "HD029647",
        "HD014250",
        "BD+56d524",
        "HD166734",
        "HD183143",
        "HD185418",
        "HD014956",
        "HD229238",
        "HD017505",
        "HD204827",
        "HD192660",
    ]

    # plot the extinction curves
    plot_multi_extinction(
        stars, path, alax=True, range=[0.75, 6], spread=True, exclude=["IRS"], pdf=True,
    )


if __name__ == "__main__":
    plot_extinction_curves()
