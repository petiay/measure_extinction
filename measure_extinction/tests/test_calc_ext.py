import pkg_resources
import os

from measure_extinction.utils.calc_ext import calc_extinction, calc_ave_ext


def test_calc_extinction():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    redstarname = "HD229238"
    compstarname = "HD204172"

    # calculate the extinction curve
    calc_extinction(redstarname, compstarname, data_path, savepath=data_path)

    # check if an extinction curve has been calculated and saved to a fits file
    assert os.path.isfile(
        data_path + "%s_%s_ext.fits" % (redstarname.lower(), compstarname.lower())
    ), (
        "No FITS file has been created with the extinction curve of reddened star "
        + redstarname
        + " with comparison star "
        + compstarname
    )


def test_calc_ave_ext():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    # list the same starpair 3 times so that an average curve will be calculated (it needs at least 3 sightlines)
    starpair_list = ["HD229238_HD204172", "HD229238_HD204172", "HD229238_HD204172"]

    # calculate the average extinction curve
    # this is actually not very useful if there is only one starpair, but at least the function will be run
    calc_ave_ext(starpair_list, data_path)

    # check if the average extinction curve has been calculated and saved to a fits file
    assert os.path.isfile(
        data_path + "average_ext.fits"
    ), "No FITS file has been created with the average extinction curve"
