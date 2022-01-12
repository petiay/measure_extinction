
=================
Code Capabilities
=================

The code is built around two classes.
StarData is for photometry and spectrosocpy of a single star.
ExtData is for calculating and storing the extinction curve for a single
sightline.

For both classes, the data is stored by the source of origin of the data.  Possible
origins are band photometry ("BAND") or spectroscopy from the
International Ultraviolet Explorer ("IUE"),
Far-Ultraviolet Spectroscopic Explorer ("FUSE"),
Hubble Space Telescope Imaging Spectrograph ("STIS"),
NASA's Infrared Telescope SpeX Instrument ("SpeX_SXD", "SpeX_LXD"),
and Spitzer Infrared Spectrograph ("IRS")
The strings in "" give the dictionary key.

This package assumes the spectral data from a specific source is one the
same wavelength grid for all stars/sightlines.
This simplifies the extinction calculations.
Code to do this is done for some data sources can be found in
`measure_extinction.merge_obsspec`.

StarData
========

The data stored as part of `measure_extinction.stardata.StarData`
is the photometry and spectroscopy for a
single star, each dataset stored in a class itself.
The photometry is stored in the dedicated BandData class to
provide specific capabilities needed for handling photometric data.
The spectroscopy is stored in SpecData class.

Member Functions
----------------

BandData
========

Member Functions
----------------

SpecData
========

Member Functions
----------------

ExtData
=======

The data that is stored as part of `measure_extinction.extdata.ExtData`
is the extinction curve from each source and associated details.
Specfic times stored include (where src = data source),

* type: type of extinction measurement (e.g., elx, elxebv, alax)
* type_rel_band: photometric band that is used for the relative extinction measurement (usually V band)
* waves[src]: wavelengths
* exts[src]: extinction versus wavelength
* uncs[src]: uncertainty in the extinction versus wavelength
* npts[src]: number of measurements at each wavelength
* names[src]: names of the photometric bands (only present for names["BAND"])
* columns: dictionary of gas or dust column measurements (e.g., A(V), R(V), N(HI))
* red_file: filename of the reddened star
* comp_file: filename of the comparison star

Member Functions
----------------

There are a number of member functions for ExtData.

* calc_elx: calculate E(lambda-X) for all data sources present in both reddened and comparision StarData objects
* calc_elx_bands: calculate E(lambda-X) for the "BAND" data
* calc_elx_spectra: calculate E(lambda-X) for a single spectral data source
* calc_EBV: determine E(B-V) based on E(lambda-V) "BAND" data
* calc_AV: determine A(V) based on E(lambda-V) "BAND" data
* calc_RV: determine R(V) using calc_EBV and calc_AV as needed
* trans_elv_elvebv: calculate E(lambda-V)/E(B-V) based on E(lambda-V) and E(B-V)
* trans_elv_alav: calculate A(lambda)/A(V) based on E(lambda-V) and A(V)
* rebin_constres: rebin one data source extinction data to a constant resolution
* get_fitdata: gets a tuple of vectors with (wavelengths, exts, uncs) that is useful for fitting
* save: save the extinction curve to a FITS file
* read: read an extinction curve from a FITS file
* plot: plot the extinction curve given a matplotlib plot object
