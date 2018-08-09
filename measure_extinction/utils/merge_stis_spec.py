from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_stis_obsspec

if __name__ == "__main__":

    path = '/home/kgordon/Python_git/extstar_data/STIS/'
    sfilename = "%s/Orig/%s" % (path, 'hd99872.mrg')

    stable = Table.read(sfilename, format='ascii',
                        data_start=23,
                        names=['WAVELENGTH', 'COUNT-RATE', 'FLUX',
                               'STAT-ERROR', 'SYS-ERROR', 'NPTS',
                               'TIME', 'QUAL'])

    rb_stis_opt = merge_stis_obsspec([stable], waveregion='Opt')
    stis_opt_file = "hd99872_stis_opt.fits"
    rb_stis_opt.write("%s/%s" % (path, stis_opt_file),
                      overwrite=True)
