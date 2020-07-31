import pkg_resources
import os
import numpy as np
import warnings

from measure_extinction.utils.plot_spec import plot_spectra


def test_plot_spectra():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    starlist = ["HD229238", "HD204172"]

    # plot the spectrum with the default settings
    plot_spectra(
        np.array(
            starlist
        ),  # convert the type of "starlist" from list to numpy array (to enable sorting later on)
        data_path,
        mlam4=False,
        onefig=False,
        range=None,
        pdf=True,
    )

    # check if the expected pdf file has been created
    for star in starlist:
        if not os.path.isfile(data_path + star + "_spec.pdf"):
            warnings.warn("No pdf file has been created,", stacklevel=2)
