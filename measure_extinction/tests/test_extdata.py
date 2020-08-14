import pkg_resources

import astropy.units as u
import numpy as np

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData


def test_calc_ext():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    # read in the observed data of the stars
    redstar = StarData("hd229238.dat", path=data_path)
    compstar = StarData("hd204172.dat", path=data_path)

    # calculate the extinction curve
    ext = ExtData()
    ext.calc_elx(redstar, compstar)

    # test that the quantities have units (or not as appropriate)
    for cursrc in ext.waves.keys():
        assert isinstance(ext.waves[cursrc], u.Quantity)
        assert not isinstance(ext.exts[cursrc], u.Quantity)
        assert not isinstance(ext.uncs[cursrc], u.Quantity)
        assert not isinstance(ext.npts[cursrc], u.Quantity)

    # check that the wavelengths can be converted to microns
    for cursrc in ext.waves.keys():
        twave = ext.waves[cursrc].to(u.micron)
        assert twave.unit == u.micron


def test_get_fitdata():

    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    # read in the observed data of the stars
    redstar = StarData("hd229238.dat", path=data_path)
    compstar = StarData("hd204172.dat", path=data_path)

    # calculate the extinction curve
    ext = ExtData()
    ext.calc_elx(redstar, compstar)

    # once wavelenth units saved, update FITS file and use this line instead
    # of the 4 lines above

    # ext = ExtData(filename=data_path + "hd283809_hd064802_ext.fits")

    wave, y, unc = ext.get_fitdata(
        ["BAND", "IUE"], remove_uvwind_region=True, remove_lya_region=True
    )

    # fitting routines often cannot handle units, make sure none are present
    for cursrc in ext.waves.keys():
        assert isinstance(wave, u.Quantity)
        assert not isinstance(y, u.Quantity)
        assert not isinstance(unc, u.Quantity)


def test_calc_AV_RV():
    # get the location of the data files
    data_path = pkg_resources.resource_filename("measure_extinction", "data/")

    # read in the observed data of the stars
    redstar = StarData("hd229238.dat", path=data_path)
    compstar = StarData("hd204172.dat", path=data_path)

    # calculate the extinction curve
    ext = ExtData()
    ext.calc_elx(redstar, compstar)

    # calculate A(V)
    ext.calc_AV()
    np.testing.assert_almost_equal(ext.columns["AV"], 2.5902545306388625)

    # calculate R(V)
    ext.calc_RV()
    np.testing.assert_almost_equal(ext.columns["RV"], 2.643116867998838)
