import numpy as np
import astropy.units as u
from astropy.table import QTable

from measure_extinction.merge_obsspec import (
    fluxunit,
    _wavegrid,
    merge_iue_obsspec,
    merge_gen_obsspec,
)

# still need to add merging of STIS spectroscopy
#  more complicated due to UV/optical options


def _check_genmerge(wave_range, resolution, mergefunc, iue=False):

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
    if iue:
        otable = mergefunc([itable1, itable2])
    else:
        otable = mergefunc([itable1, itable2], wave_range, resolution)

    # check standard format
    for ckey in ["WAVELENGTH", "FLUX", "SIGMA", "NPTS"]:
        assert ckey in otable.keys()

    nowaves = len(otable["WAVELENGTH"])
    np.testing.assert_allclose(otable["FLUX"], np.full(nowaves, 1.5))


def test_iue():
    wave_range = [1000.0, 3400.0] * u.angstrom
    resolution = 1000.0
    _check_genmerge(wave_range, resolution, merge_iue_obsspec, iue=True)


# only need to test one of the "generic" merges as all the 
# rest work the same
def test_irs():
    wave_range = [5.0, 40.0] * u.micron
    resolution = 150.0
    _check_genmerge(wave_range, resolution, merge_gen_obsspec)
