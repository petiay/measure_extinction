#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

from measure_extinction.plotting.plot_ext import plot_multi_extinction


def plot_extinction_curves():
    # define the path and the names of the star pairs in the format "reddenedstarname_comparisonstarname" (first the main sequence stars and then the giant stars, sorted by spectral type from B9 to O0)
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    starpair_list = [
        "HD034921_HD036512",
        "HD013338_HD031726",
        "HD037023_HD034816",
        "HD037020_HD034816",
        "HD052721_HD003360",
        "HD037022_HD047839",
        "HD038087_HD034759",
        "HD014422_HD031726",
        "HD294264_HD032630",
        "HD206773_HD036512",
        "HD037061_HD031726",
        "HD029309_HD003360",
        "HD283809_HD032630",
        "HD156247_HD034759",
        "HD029647_HD078316",
        "HD014250_HD031726",
        "BD+56d524_HD031726",
        "HD166734_HD047839",
        "HD183143_HD078316",
        "HD185418_HD034816",
        "HD014956_HD051283",
        "HD229238_HD204172",
        "HD017505_HD047839",
        "HD204827_HD036512",
        "HD192660_HD204172",
    ]

    # plot the extinction curves
    plot_multi_extinction(
        starpair_list,
        path,
        alax=True,
        range=[0.75, 6],
        spread=True,
        exclude=["IRS"],
        pdf=True,
    )


if __name__ == "__main__":
    plot_extinction_curves()
