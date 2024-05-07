import numpy as np
import astropy.units as u
from measure_extinction.extdata import ExtData


def test_calc_ebv():
    text = ExtData()
    text.waves["BAND"] = np.array([0.438, 0.555]) * u.micron
    text.exts["BAND"] = np.array([0.5, 0.0])
    text.uncs["BAND"] = np.array([0.03, 0.0])

    text.calc_EBV()
    ebv = text.columns["EBV"]
    assert isinstance(ebv, tuple)
    assert ebv[0] == 0.5
    assert ebv[1] == 0.03


def test_calc_av():
    text = ExtData()
    text.waves["BAND"] = np.array([0.555, 2.19]) * u.micron
    text.exts["BAND"] = np.array([0.0, -0.5])
    text.uncs["BAND"] = np.array([0.0, 0.02])

    text.calc_AV()
    av = text.columns["AV"]
    assert isinstance(av, tuple)
    assert np.isclose(av[0], 0.55866, atol=1e-3)
    assert np.isclose(av[1], 0.02235, atol=1e-3)


def test_calc_rv_fromelv():
    text = ExtData()
    text.waves["BAND"] = np.array([0.438, 0.555, 2.19]) * u.micron
    text.exts["BAND"] = np.array([0.5, 0.0, -1.5])
    text.uncs["BAND"] = np.array([0.03, 0.0, 0.02])

    text.calc_RV()
    rv = text.columns["RV"]
    assert isinstance(rv, tuple)
    assert np.isclose(rv[0], 3.351955, atol=1e-3)
    assert np.isclose(rv[1], 0.206023, atol=1e-3)


def test_calc_rv_fromebvav():
    text = ExtData()
    text.columns["EBV"] = (0.5, 0.03)
    text.columns["AV"] = (3.1 * 0.5, 0.04)

    text.calc_RV()
    rv = text.columns["RV"]
    assert isinstance(rv, tuple)
    assert np.isclose(rv[0], 3.1, atol=1e-3)
    assert np.isclose(rv[1], 0.20247, atol=1e-3)


def test_calc_rv_fromebvav_pmunc():
    text = ExtData()
    text.columns["EBV"] = (0.5, 0.02, 0.04)
    text.columns["AV"] = (3.1 * 0.5, 0.05, 0.03)

    text.calc_RV()
    rv = text.columns["RV"]
    assert isinstance(rv, tuple)
    assert np.isclose(rv[0], 3.1, atol=1e-3)
    assert np.isclose(rv[1], 0.20247, atol=1e-3)
