#!/usr/bin/env python

import glob

from measure_extinction.utils.make_obsdata_from_model import make_obsdata_from_model


def decode_params_pre2022(filename):
    """
    Decode the tlusty filenames for the model parameters
    """
    model_params = {}

    slashpos = filename.rfind("/")
    periodpos = filename.rfind(".flux")

    # translation of metallicity code
    met_code = {"C": 2.0, "G": 1.0, "L": 0.5, "S": 0.2, "T": 0.1}
    met_char = filename[slashpos + 1 : slashpos + 2]
    T_start = slashpos + 1
    if met_char == "B":
        met_char = filename[slashpos + 2 : slashpos + 3]
        T_start += 1
        model_params["origin"] = "Bstar"
    else:
        model_params["origin"] = "Ostar"
    model_params["Z"] = met_code[met_char]

    gpos = filename.find("g", slashpos)
    vpos = filename.find("v", slashpos)

    model_params["Teff"] = float(filename[T_start + 1 : gpos])
    model_params["logg"] = float(filename[gpos + 1 : vpos]) * 0.01
    model_params["vturb"] = float(filename[vpos + 1 : periodpos])

    return model_params


def decode_params(filename):
    """
    Decode the tlusty filenames for the model parameters
    """
    model_params = {}

    slashpos = filename.rfind("/")
    periodpos = filename.rfind(".spec")

    zpos = filename.find("z", slashpos)
    tpos = filename.find("t", slashpos)
    gpos = filename.find("g", slashpos)
    vpos = filename.find("v", slashpos)

    model_params["Z"] = float(filename[zpos + 1 : tpos]) * 0.01
    model_params["Teff"] = float(filename[tpos + 1 : gpos])
    model_params["logg"] = float(filename[gpos + 1 : vpos]) * 0.01
    model_params["vturb"] = float(filename[vpos + 1 : periodpos])

    return model_params


if __name__ == "__main__":
    tlusty_models = glob.glob("/home/kgordon/Python/extstar_data/Models/Tlusty_2023/*v5.spec.gz")

    for cfname in tlusty_models:
        # parse the filename to get the model parameters
        model_params = decode_params(cfname)

        # get the base filename for the output files
        slashpos = cfname.rfind("/")
        periodpos = cfname.rfind(".spec")

        basename = "tlusty_{}".format(cfname[slashpos + 1 : periodpos])

        print(cfname)
        print(basename)
        print(model_params)

        make_obsdata_from_model(
            cfname,
            model_type="tlusty",
            output_filebase=basename,
            output_path="/home/kgordon/Python/extstar_data",
            model_params=model_params,
        )
