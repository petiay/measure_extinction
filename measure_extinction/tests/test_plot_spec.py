import pkg_resources
import os
import numpy as np
import warnings

from measure_extinction.utils.plot_spec import plot_spectra


def test_plot_spectra():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    starlist = ["HD229238", "HD204172"]

    # plot the spectra with the default settings
    plot_spectra(
        np.array(starlist), data_path, mlam4=False, onefig=False, range=None, pdf=True,
    )

    # plot the spectra as lambda^4 * F(lambda) (i.e. mlam4=True)
    plot_spectra(
        np.array(starlist), data_path, mlam4=True, onefig=False, range=None, pdf=True,
    )

    # plot the spectra and zoom in on a specific wavelength region (i.e. range=[0.7,6])
    plot_spectra(
        np.array(starlist),
        data_path,
        mlam4=False,
        onefig=False,
        range=[0.7, 6],
        pdf=True,
    )

    # check if the expected pdf files have been created
    for star in starlist:
        if not os.path.isfile(data_path + star + "_spec.pdf"):
            warnings.warn(
                "Plotting the spectra with the default settings has failed,",
                stacklevel=2,
            )

        if not os.path.isfile(data_path + star + "_spec_mlam4.pdf"):
            warnings.warn(
                "Plotting the spectra in lambda^4*F(lambda) has failed,", stacklevel=2
            )

        if not os.path.isfile(data_path + star + "_spec_zoom.pdf"):
            warnings.warn(
                "Plotting the spectra in a specific wavelength range has failed,",
                stacklevel=2,
            )
