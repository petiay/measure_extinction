from matplotlib import pyplot as plt
import pkg_resources
from synphot import SpectralElement


def plot_bandpasses(bands):
    # create the figure
    plt.figure(figsize=(15, 5))

    # plot all response curves
    for band in bands:
        bp = SpectralElement.from_file("%s%s.dat" % (band_path, band))
        if "MIPS" in band:
            wavelengths = bp.waveset * 10**4
        else:
            wavelengths = bp.waveset
        plt.plot(wavelengths, bp(bp.waveset), label=band)

    plt.xlabel(r"$\lambda$ [$\AA$]", size=15)
    plt.ylabel("transmission", size=15)
    plt.legend(ncol=2, loc=1)
    plt.show()


if __name__ == "__main__":
    # path for band response curves
    band_path = pkg_resources.resource_filename(
        "measure_extinction", "data/Band_RespCurves/"
    )

    # define the different bandpasses
    bands = [
        "JohnU",
        "JohnB",
        "JohnV",
        "JohnR",
        "JohnI",
        "JohnJ",
        "2MASSJ",
        "JohnH",
        "2MASSH",
        "JohnK",
        "JohnKs",
        "2MASSKs",
        "WISE1",
        "AAOL",
        "IRAC1",
        "AAOLprime",
        "IRAC2",
        "WISE2",
        "AAOM",
    ]
    plot_bandpasses(bands)
