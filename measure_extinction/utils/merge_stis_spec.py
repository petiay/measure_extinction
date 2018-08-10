from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_stis_obsspec

if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument("--waveregion", choices=['UV', 'Opt'], default='Opt',
                        help="wavelength region to merge")
    parser.add_argument(
        "--path",
        help="path where merged spectra will be stored",
        default="/home/kgordon/Python_git/extstar_data/STIS_Data/")
    parser.add_argument("--png", help="save figure as a png file",
                        action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file",
                        action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()

    sfilename = "%s/Orig/%s/%s.mrg" % (args.path, args.waveregion,
                                       args.starname)

    stable = Table.read(sfilename, format='ascii',
                        data_start=23,
                        names=['WAVELENGTH', 'COUNT-RATE', 'FLUX',
                               'STAT-ERROR', 'SYS-ERROR', 'NPTS',
                               'TIME', 'QUAL'])

    rb_stis_opt = merge_stis_obsspec([stable], waveregion=args.waveregion)
    stis_opt_file = "%s_stis_%s.fits" % (args.starname, args.waveregion)
    rb_stis_opt.write("%s/%s" % (args.path, stis_opt_file),
                      overwrite=True)
