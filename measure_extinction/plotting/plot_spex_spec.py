#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pkg_resources
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import astropy.units as u

from measure_extinction.plotting.plot_spec import plot_multi_spectra


def plot_comp_spectra():
    # define the path and the names of the comparison stars
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    stars = [
        "HD164794",
        "HD047839",
        "HD214680",
        "HD036512",
        "HD188209",
        "HD204172",
        "HD034816",
        "HD091316",
        "HD003360",
        "HD031726",
        "HD051283",
        "HD042560",
        "HD032630",
        "HD034759",
        "HD078316",
    ]

    # plot the spectra
    plot_multi_spectra(
        np.array(stars),
        path,
        mlam4=True,
        range=[0.75, 6],
        pdf=True,
        outname="comp_stars.pdf",
    )


if __name__ == "__main__":
    plot_comp_spectra()
