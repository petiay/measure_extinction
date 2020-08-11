#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

from measure_extinction.plotting.plot_spec import plot_multi_spectra


def plot_comp_spectra():
    # define the path and the names of the comparison stars (first the main sequence stars and then the giant stars, sorted by spectral type from B9 to O0)
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    stars = [
        "HD034759",
        "HD032630",
        "HD042560",
        "HD031726",
        "HD003360",
        "HD034816",
        "HD036512",
        "HD214680",
        "HD047839",
        "HD164794",
        "HD078316",
        "HD051283",
        "HD091316",
        "HD204172",
        "HD188209",
    ]

    # plot the spectra
    plot_multi_spectra(
        stars,
        path,
        mlam4=True,
        range=[0.75, 6],
        norm_range=[1, 1.1],
        spread=True,
        exclude=["IRS"],
        pdf=True,
        outname="comp_stars.pdf",
    )


def plot_red_spectra():
    # define the path and the names of the reddened stars (first the main sequence stars and then the giant stars, sorted by spectral type from B9 to O0)
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    stars = [
        "HD038087",
        "HD156247",
        "HD294264",
        "HD283809",
        "HD052721",
        "HD029309",
        "HD013338",
        "HD014422",
        "BD+56d524",
        "HD037061",
        "HD014250",
        "HD037023",
        "HD185418",
        "HD037020",
        "HD206773",
        "HD204827",
        "HD034921",
        "HD166734",
        "HD037022",
        "HD017505",
        "HD029647",
        "HD183143",
        "HD014956",
        "HD229238",
        "HD192660",
    ]

    # plot the spectra
    plot_multi_spectra(
        stars,
        path,
        mlam4=True,
        range=[0.75, 6],
        norm_range=[1, 1.1],
        spread=True,
        exclude=["IRS", "STIS_Opt"],
        pdf=True,
        outname="red_stars.pdf",
    )


if __name__ == "__main__":
    plot_comp_spectra()
    plot_red_spectra()
