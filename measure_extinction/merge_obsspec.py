from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table, Column
import astropy.units as u

__all__ = ["merge_iue_obsspec", "merge_stis_obsspec", "merge_irs_obsspec",
           "merge_spex_obsspec"]


def _wavegrid(resolution, wave_range):
    """
    Define a wavelength grid at a specified resolution given
    the min/max as input

    Parameters
    ----------
    resolution : float
        resolution of grid

    wave_range : [float, float]
        min/max of grid

    Returns
    -------
    wave_info : tuple [waves, waves_bin_min, waves_bin_max]
        wavelength grid center, min, max wavelengths
    """
    npts = int(
        np.log10(wave_range[1] / wave_range[0])
        / np.log10((1.0 + 2.0 * resolution) / (2.0 * resolution - 1.0))
    )
    delta_wave_log = (np.log10(wave_range[1]) - np.log10(wave_range[0])) / npts
    wave_log10 = np.arange(
        np.log10(wave_range[0]),
        np.log10(wave_range[1]) - delta_wave_log,
        delta_wave_log,
    )
    full_wave_min = 10 ** wave_log10
    full_wave_max = 10 ** (wave_log10 + delta_wave_log)

    full_wave = (full_wave_min + full_wave_max) / 2.0

    return (full_wave, full_wave_min, full_wave_max)


def merge_iue_obsspec(obstables, output_resolution=500):
    """
    Merge one or more IUE 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IUE spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [1000.0, 3400.0] * u.angstrom

    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)
    for ctable in obstables:
        # may want to add in the SYS-ERROR, but need to be careful
        # to propagate it correctly, SYS-ERROR will not reduce with
        # multiple spectra or measurements in a wavelength bin
        cuncs = ctable["STAT-ERROR"].data
        cwaves = ctable["WAVELENGTH"].data
        cfluxes = ctable["FLUX"].data
        cnpts = ctable["NPTS"].data
        for k in range(n_waves):
            indxs, = np.where(
                (cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]) & (cnpts > 0)
            )
            if len(indxs) > 0:
                weights = 1.0 / np.square(cuncs[indxs])
                full_flux[k] += np.sum(weights * cfluxes[indxs])
                full_unc[k] += np.sum(weights)
                full_npts[k] += len(indxs)

    # divide by the net weights
    indxs, = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(1.0 / full_unc[indxs])

    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(full_unc, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["NPTS"] = Column(full_npts)

    return otable


def merge_stis_obsspec(obstables, waveregion="UV", output_resolution=1000):
    """
    Merge one or more STIS 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed STIS spectra
        usually the result of reading tables

    waveregion : string [default = 'UV']
        wavelength region of spectra
        possibilities are 'UV', 'Opt'

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    if waveregion == "UV":
        wave_range = [1000.0, 3400.0] * u.angstrom
    elif waveregion == "Opt":
        wave_range = [2900.0, 10250.0] * u.angstrom

    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)
    for ctable in obstables:
        # may want to add in the SYS-ERROR, but need to be careful
        # to propagate it correctly, SYS-ERROR will not reduce with
        # multiple spectra or measurements in a wavelength bin
        cuncs = ctable["STAT-ERROR"].data
        cwaves = ctable["WAVELENGTH"].data
        cfluxes = ctable["FLUX"].data
        cnpts = ctable["NPTS"].data
        for k in range(n_waves):
            indxs, = np.where(
                (cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]) & (cnpts > 0)
            )
            if len(indxs) > 0:
                weights = 1.0 / np.square(cuncs[indxs])
                full_flux[k] += np.sum(weights * cfluxes[indxs])
                full_unc[k] += np.sum(weights)
                full_npts[k] += len(indxs)

    # divide by the net weights
    indxs, = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(1.0 / full_unc[indxs])

    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(full_unc, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["NPTS"] = Column(full_npts)

    return otable


def merge_irs_obsspec(obstables, output_resolution=150):
    """
    Merge one or more Spitzer IRS 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IRS spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [5.0, 40.0] * u.micron

    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)
    for ctable in obstables:
        cuncs = ctable["ERROR"].data
        cwaves = ctable["WAVELENGTH"].data
        cfluxes = ctable["FLUX"].data
        cnpts = ctable["NPTS"].data
        for k in range(n_waves):
            indxs, = np.where(
                (cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]) & (cnpts > 0)
            )
            if len(indxs) > 0:
                weights = 1.0 / np.square(cuncs[indxs])
                full_flux[k] += np.sum(weights * cfluxes[indxs])
                full_unc[k] += np.sum(weights)
                full_npts[k] += len(indxs)

    # divide by the net weights
    indxs, = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(1.0 / full_unc[indxs])

    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(full_unc, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["NPTS"] = Column(full_npts)

    return otable


def merge_spex_obsspec(obstable, output_resolution=2000):
    """
    Merge one or more IRTF SpeX 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstable : astropy Table object
        table containing the observed SpeX spectrum
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    waves = obstable["WAVELENGTH"].data * 1e4
    fluxes = obstable["FLUX"].data
    uncs = obstable["ERROR"].data
    npts = np.full((len(obstable["FLUX"])), 1.0)
    # take out data points that were flagged by Spextool
    npts[obstable["FLAG"] == 1.0] = 0
    # take out data points with NaN fluxes
    npts[np.isnan(fluxes)] = 0
    # take out data points with low SNR
    npts[np.less(fluxes/uncs, 10, where=~np.isnan(fluxes/uncs))] = 0
    # take out wavelength regions affected by the atmosphere
    npts[np.logical_and(2.5e4<waves,waves<2.9e4)] = 0

    # determine the wavelength range and calculate the wavelength grid
    if np.max(waves) < 25000:
        wave_range = [0.8, 2.45] * u.micron
    else:
        wave_range = [2.4, 5.5] * u.micron

    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    # create empty arrays
    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)

    # fill the arrays
    for k in range(n_waves):
        indxs, = np.where(
            (waves >= full_wave_min[k]) & (waves < full_wave_max[k]) & (npts > 0)
        )
        if len(indxs) > 0:
            weights = 1.0 / np.square(uncs[indxs])
            full_flux[k] = np.sum(weights * fluxes[indxs])
            full_unc[k] = np.sum(weights)
            full_npts[k] = len(indxs)

    # divide by the net weights
    indxs, = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(1.0 / full_unc[indxs])

    # create the output table
    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(full_unc, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["NPTS"] = Column(full_npts)

    return otable
