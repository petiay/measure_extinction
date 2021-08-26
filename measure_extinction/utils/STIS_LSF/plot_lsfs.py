import argparse

import matplotlib.pyplot as plt
from astropy.table import Table


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fig, ax = plt.subplots(ncols=3, figsize=(16, 8))

    # resolutions in A/pixel
    g140l_res = 0.5831004076162168
    g230l_res = 1.548239955089159

    tags = ["G140L_1200", "G140L_1500"]
    for i, ctag in enumerate(tags):
        a = Table.read(
            f"data/LSF_{ctag}.txt", format="ascii.commented_header", header_start=-1
        )

        # slits = ["52x0.1", "52x0.2", "52x0.5", "52x2.0"]
        slits = ["52x2.0"]
        for cslit in slits:
            ax[0].plot(g140l_res * a["Rel_pixel"], a[cslit], label=ctag)
        ax[0].legend()
    ax[0].set_xlabel(r"$\Delta\lambda$ [$\AA$]")
    ax[0].set_xlim(-20 * g140l_res, 20 * g140l_res)

    tags = ["G230L_1700", "G230L_2400"]
    for i, ctag in enumerate(tags):
        a = Table.read(
            f"data/LSF_{ctag}.txt", format="ascii.commented_header", header_start=-1
        )

        # slits = ["52x0.1", "52x0.2", "52x0.5", "52x2.0"]
        slits = ["52x2.0"]
        for cslit in slits:
            ax[1].plot(g230l_res * a["Rel_pixel"], a[cslit], label=ctag)
        ax[1].legend()
    ax[1].set_xlabel(r"$\Delta\lambda$ [$\AA$]")
    ax[1].set_xlim(-20 * g230l_res, 20 * g230l_res)

    tags = ["G430L_3200", "G430L_5500", "G750L_7000"]
    for i, ctag in enumerate(tags):
        a = Table.read(
            f"data/LSF_{ctag}.txt", format="ascii.commented_header", header_start=-1
        )

        # slits = ["52x0.1", "52x0.2", "52x0.5", "52x2.0"]
        slits = ["52x2.0"]
        for cslit in slits:
            ax[2].plot(a["Rel_pixel"], a[cslit], label=ctag)
        ax[2].legend()
    ax[2].set_xlabel(r"$\Delta\lambda$ [pixel]")
    ax[2].set_xlim(-20, 20)

    fig.tight_layout()

    if args.png:
        fig.savefig("stis_lsfs.png")
    else:
        plt.show()
