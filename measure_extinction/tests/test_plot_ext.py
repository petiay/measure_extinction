import pkg_resources
import os
import warnings

from measure_extinction.plotting.plot_ext import plot_multi_extinction, plot_extinction


def test_plot_extinction():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")
    star = "HD229238"

    # plot the extinction curve
    # with the default settings
    plot_extinction(star, data_path, pdf=True)

    # as A(lambda)/A(V) (i.e. alax=True)
    plot_extinction(star, data_path, alax=True, pdf=True)

    # zoomed in on a specific wavelength region (i.e. range=[0.7,6])
    plot_extinction(star, data_path, range=[0.7, 6], pdf=True)

    # test several other plotting options
    # note: verifying the existence of the pdf file with the plot is not sufficient to ensure that the different plotting options work as expected. However, these tests at least make sure that the corresponding functions run without errors.

    # with band data excluded from the plot
    plot_extinction(star, data_path, exclude="BAND", pdf=True)

    # with HI-lines indicated
    plot_extinction(star, data_path, HI_lines=True, pdf=True)

    # in one figure, although this is actually only relevant if there is more than one curve.
    stars = ["HD229238"]
    plot_multi_extinction(stars, data_path, pdf=True)
    plot_multi_extinction(stars, data_path, alax=True, pdf=True)
    plot_multi_extinction(stars, data_path, range=[0.7, 6], pdf=True)
    plot_multi_extinction(stars, data_path, exclude="BAND", pdf=True)
    plot_multi_extinction(stars, data_path, HI_lines=True, pdf=True)

    # check if the expected pdf files were created
    if not os.path.isfile(data_path + star + "_ext_elx.pdf"):
        warnings.warn(
            "Plotting the extinction curve of star "
            + star
            + " with the default settings has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + star + "_ext_alav.pdf"):
        warnings.warn(
            "Plotting the extinction curve of star "
            + star
            + " in A(lambda)/A(V) has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + star + "_ext_elx_zoom.pdf"):
        warnings.warn(
            "Plotting the extintion curve of star "
            + star
            + " in a specific wavelength range has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + "all_ext_elx.pdf"):
        warnings.warn(
            "Plotting the extintion curve of star "
            + star
            + "in one figure with the default settings has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + "all_ext_alav.pdf"):
        warnings.warn(
            "Plotting the extinction curve of star "
            + star
            + " in one figure in A(lambda)/A(V) has failed.",
            stacklevel=2,
        )

    if not os.path.isfile(data_path + "all_ext_elx_zoom.pdf"):
        warnings.warn(
            "Plotting the extintion curve of star "
            + star
            + " in one figure in a specific wavelength range has failed.",
            stacklevel=2,
        )
