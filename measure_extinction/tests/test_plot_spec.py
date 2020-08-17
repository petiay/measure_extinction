import pkg_resources
import os
import warnings

from measure_extinction.plotting.plot_spec import plot_multi_spectra, plot_spectrum


def test_plot_spectra():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")
    starlist = ["HD229238", "HD204172"]

    # plot the spectra in separate figures
    for star in starlist:
        # with the default settings
        plot_spectrum(star, data_path, pdf=True)

        # as lambda^4 * F(lambda) (i.e. mlam4=True)
        plot_spectrum(star, data_path, mlam4=True, pdf=True)

        # zoomed in on a specific wavelength region (i.e. range=[0.7,6])
        plot_spectrum(star, data_path, range=[0.7, 6], pdf=True)

        # check if the expected pdf files were created
        if not os.path.isfile(data_path + star + "_spec.pdf"):
            warnings.warn(
                "Plotting the spectrum for star "
                + star
                + " with the default settings has failed.",
                stacklevel=2,
            )

        if not os.path.isfile(data_path + star + "_spec_mlam4.pdf"):
            warnings.warn(
                "Plotting the spectrum for star "
                + star
                + " in lambda^4*F(lambda) has failed.",
                stacklevel=2,
            )

        if not os.path.isfile(data_path + star + "_spec_zoom.pdf"):
            warnings.warn(
                "Plotting the spectrum for star "
                + star
                + " in a specific wavelength range has failed.",
                stacklevel=2,
            )

    # plot the two spectra in one figure
    # with the default settings
    plot_multi_spectra(starlist, data_path, pdf=True)

    # as lambda^4 * F(lambda) (i.e. mlam4=True)
    plot_multi_spectra(starlist, data_path, mlam4=True, pdf=True)

    # zoomed in on a specific wavelength region (i.e. range=[0.7,6])
    plot_multi_spectra(starlist, data_path, range=[0.7, 6], pdf=True)

    # check if the expected pdf files were created
    if not os.path.isfile(data_path + "all_spec.pdf"):
        warnings.warn(
            "Plotting the spectra of both stars in one figure with the default settings has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + "all_spec_mlam4.pdf"):
        warnings.warn(
            "Plotting the spectra of both stars in one figure in lambda^4*F(lambda) has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + "all_spec_zoom.pdf"):
        warnings.warn(
            "Plotting the spectra of both stars in one figure in a specific wavelength range has failed.",
            stacklevel=2,
        )
