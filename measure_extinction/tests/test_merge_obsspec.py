import numpy as np
import astropy.units as u
from astropy.table import QTable

from measure_extinction.merge_obsspec import fluxunit, _wavegrid, merge_miri_lrs_obsspec


def _check_genmerge(wave_range, resolution):

    wave1_info = _wavegrid(resolution, wave_range.value)
    wave1 = wave1_info[0] * wave_range.unit
    nwaves = len(wave1)
    itable1 = QTable()
    itable1["WAVELENGTH"] = wave1
    itable1["FLUX"] = np.full(nwaves, 1.0) * fluxunit
    itable1["ERROR"] = 0.01 * itable1["FLUX"]
    itable1["NPTS"] = np.full(nwaves, 1)

    itable2 = QTable()
    itable2["WAVELENGTH"] = wave1
    itable2["FLUX"] = np.full(nwaves, 2.0) * fluxunit
    # uncertainties are 1/2 of the first table
    # same amplitude uncertainties = equal weighting
    itable2["ERROR"] = 0.005 * itable2["FLUX"]
    itable2["NPTS"] = np.full(nwaves, 1)

    # merge into standard format
    otable = merge_miri_lrs_obsspec([itable1, itable2])

    # check standard format
    for ckey in ["WAVELENGTH", "FLUX", "SIGMA", "NPTS"]:
        assert ckey in otable.keys()

    nowaves = len(otable["WAVELENGTH"])
    np.testing.assert_allclose(otable["FLUX"], np.full(nowaves, 1.5))


def test_miri_lrs():

    # generate mock spectra
    wave_range = [0.4, 15.0] * u.micron
    resolution = 150.0
    _check_genmerge(wave_range, resolution)


def test_miri_mrs():

    # generate mock spectra
    wave_range = [4.5, 32.0] * u.micron
    resolution = 4000.
    _check_genmerge(wave_range, resolution)


if __name__ == "__main__":
    test_miri_lrs()
