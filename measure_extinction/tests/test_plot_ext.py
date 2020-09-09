import pkg_resources
import os

from measure_extinction.plotting.plot_ext import plot_multi_extinction, plot_extinction


def test_plot_extinction():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")
    starpair = "HD229238_HD204172"

    # plot the extinction curve
    # with the default settings
    plot_extinction(starpair, data_path, pdf=True)

    # as A(lambda)/A(V) (i.e. alax=True)
    plot_extinction(starpair, data_path, alax=True, pdf=True)

    # zoomed in on a specific wavelength region (i.e. range=[0.7,6])
    plot_extinction(starpair, data_path, range=[0.7, 6], pdf=True)

    # test several other plotting options
    # note: verifying the existence of the pdf file with the plot is not sufficient to ensure that the different plotting options work as expected. However, these tests at least make sure that the corresponding functions run without errors.

    # with band data excluded from the plot
    plot_extinction(starpair, data_path, exclude="BAND", pdf=True)

    # with HI-lines indicated
    plot_extinction(starpair, data_path, HI_lines=True, pdf=True)

    # in one figure, although this is actually only relevant if there is more than one curve.
    starpair_list = ["HD229238_HD204172"]
    plot_multi_extinction(starpair_list, data_path, pdf=True)
    plot_multi_extinction(starpair_list, data_path, alax=True, pdf=True)
    plot_multi_extinction(starpair_list, data_path, range=[0.7, 6], pdf=True)
    plot_multi_extinction(starpair_list, data_path, exclude="BAND", pdf=True)
    plot_multi_extinction(starpair_list, data_path, HI_lines=True, pdf=True)

    # check if the expected pdf files were created
    message = (
        "Plotting the extinction curve of reddened star "
        + starpair.split("_")[0]
        + " with comparison star "
        + starpair.split("_")[1]
        + ", "
    )

    assert os.path.isfile(data_path + starpair.lower() + "_ext_elx.pdf"), (
        message + "with the default settings, has failed."
    )

    assert os.path.isfile(data_path + starpair.lower() + "_ext_alav.pdf"), (
        message + "in A(lambda)/A(V), has failed."
    )

    assert os.path.isfile(data_path + starpair.lower() + "_ext_elx_zoom.pdf"), (
        message + "in a specific wavelength range, has failed."
    )

    assert os.path.isfile(data_path + "all_ext_elx.pdf"), (
        message + "in one figure, with the default settings, has failed."
    )

    assert os.path.isfile(data_path + "all_ext_alav.pdf"), (
        message + "in one figure, in A(lambda)/A(V), has failed."
    )

    assert os.path.isfile(data_path + "all_ext_elx_zoom.pdf"), (
        message + "in one figure, in a specific wavelength range, has failed."
    )
