import os

__all__ = ["get_full_starfile"]


def get_full_starfile(starname):
    """
    Get full path to the DAT file for the specified star
    """
    if "dat" not in starname:
        fstarname = "%s.dat" % starname
    else:
        fstarname = starname

    if not os.path.isfile(fstarname):
        # not an installable package yet
        # def_path = pkg_resources.resource_filename(
        #    'extstar_data', '')
        # def_path = "/home/kgordon/Python_git/extstar_data/"
        def_path = "/Users/pyanchulova/Documents/extstar_data/"
        if not os.path.isfile("{}/DAT_files/{}".format(def_path, fstarname)):
            print("{} file not found".format(fstarname))
        else:
            fstarname = "DAT_files/{}".format(fstarname)
    else:
        def_path = "./"

    return fstarname, def_path
