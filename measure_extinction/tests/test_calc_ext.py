import pkg_resources
import os
import warnings

from measure_extinction.utils.calc_ext import calc_extinction


def test_calc_extinction():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    redstarname = "HD229238"
    compstarname = "HD204172"

    # calculate the extinction curve
    calc_extinction(redstarname, compstarname, data_path)

    # check if an extinction curve has been calculated and saved to a fits file
    if not os.path.isfile(data_path + redstarname + "_ext.fits"):
        warnings.warn(
            "No FITS file with the extinction curve has been created for star "
            + redstarname,
            stacklevel=2,
        )
