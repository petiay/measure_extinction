import numpy as np
import astropy.units as u

from measure_extinction.stardata import SpecData
from measure_extinction.extdata import ExtData
from measure_extinction.merge_obsspec import _wavegrid


def test_spec_rebin_constres():
    spec = SpecData("TEST")

    # fake some simple data
    # flux values alternate between 1 and 2
    tres = 1000
    # start at 1.001 to get an even number of points
    # needed as the a and b arrays have to be equal
    twaves, tmp, tmp = _wavegrid(tres, [1.001, 10.0])
    spec.waves = twaves * u.micron
    n_test = len(twaves)
    a = np.full((n_test // 2), 1.0)
    b = np.full((n_test // 2), 2.0)
    spec.fluxes = np.empty((a.size + b.size), dtype=a.dtype)
    spec.fluxes[0::2] = a
    spec.fluxes[1::2] = b
    spec.fluxes *= u.erg / ((u.cm**2) * u.s * u.angstrom)
    spec.uncs = spec.fluxes * 0.05
    spec.npts = np.full((n_test), 1)

    # rebin it using the SpecData member function
    gres = tres / 2.0
    rwaverange = [1.0, 10.0] * u.micron
    spec.rebin_constres(rwaverange, gres)

    ndelt = spec.waves[1:] - spec.waves[0:-1]
    nwave = 0.5 * (spec.waves[1:] + spec.waves[0:-1])
    nres = nwave / ndelt

    # not quite the same, but very close
    np.testing.assert_allclose(nres, np.full(len(nres), gres), rtol=0.001)

    # test all but the end elements as edge effects mean they will be different
    np.testing.assert_equal(spec.fluxes[1:-1].value, np.full(len(nres) - 1, 1.2))


def test_ext_rebin_constres():
    ext = ExtData()

    # fake some simple data
    # ext values alternate between 1 and 2
    tres = 1000
    # start at 1.001 to get an even number of points
    # needed as the a and b arrays have to be equal
    twaves, tmp, tmp = _wavegrid(tres, [1.001, 10.0])
    ext.waves["TEST"] = twaves * u.micron
    n_test = len(twaves)
    a = np.full((n_test // 2), 1.0)
    b = np.full((n_test // 2), 2.0)
    ext.exts["TEST"] = np.empty((a.size + b.size), dtype=a.dtype)
    ext.exts["TEST"][0::2] = a
    ext.exts["TEST"][1::2] = b
    ext.uncs["TEST"] = ext.exts["TEST"] * 0.05
    ext.npts["TEST"] = np.full((n_test), 1)

    # rebin it using the ExtData member function
    gres = tres / 2.0
    rwaverange = [1.0, 10.0] * u.micron
    ext.rebin_constres("TEST", rwaverange, gres)

    ndelt = ext.waves["TEST"][1:] - ext.waves["TEST"][0:-1]
    nwave = 0.5 * (ext.waves["TEST"][1:] + ext.waves["TEST"][0:-1])
    nres = nwave / ndelt

    # not quite the same, but very close
    np.testing.assert_allclose(nres, np.full(len(nres), gres), rtol=0.001)

    # test all but the end elements as edge effects mean they will be different
    # weighted average with fractional weights gives 1.2 (equal weights would give 1.5)
    np.testing.assert_equal(ext.exts["TEST"][1:-1], np.full(len(nres) - 1, 1.2))
