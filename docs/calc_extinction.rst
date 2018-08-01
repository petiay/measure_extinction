======================
Calculating Extinction
======================

Background
----------

Extinction is generally measured by comparing two stars that are
identical in their physical properties, differing only in that one
is seen through more dust than the other.
Finding two stars with identical properties is straightforward
as it just means finding two stars with the same 2D spectral types
(temperature and luminosity) and the metallicities.
As a result, measuring extinction is a simple matter of dividing the
the observations of the two stars and a small amount of math to convert
it to magnitudes.  Of course, there are a few details with which to be
concerned.

The basic measurement is given the magnitude excess relative to a
reference wavelength measurement.
This use of a reference wavelength is that the distance to a star is rarely
known to high enough accuracy to directly measure the extinction at a
single wavelength.
Thus, the basic measurement gives the relative extinction between two
wavelengths.
Usually, the V band measurement is used at the reference.
Thus, the dust extinction at :math:`x` wavelength is:

.. math ::
  E(x - V) = m(x - V)_r - m(x - V)_c

where :math:`m(x - V)` is the difference in magnitudes between the flux at
:math:`x` wavelength and the V band, :math:`r` refers to the
reddened star, and :math:`c` refers to the comparison star.

This extinction measurement can be normalized allowing comparison with
extinction along other lines-of-sight.
The normalization is often done to :math:`E(B-V)` as this is easily measured.
But notice that this is a normalization to a differential measurement.

Converting the :math:`E(x-V)` differential measurement to the :math:`A(x)`
absolute measurement requires knowledge of the the absolute extinction at the
reference wavelength, V band in this case.
Determining :math:`A(V)` requires measuring :math:`E(x-V)` at the longest wavelength
possible and extrapolating with a reference shape to infinite wavelength
as :math:`A(inf) = 0`.
Usually, the longest wavelength measured is the K band and the extrapolation
to infinite wavelength is on the order of 10%
(`Whittet, van Breda, & Glass 1976 <https://ui.adsabs.harvard.edu//#abs/1976MNRAS.177..625W/abstract>`_;
`Fitzpatrick & Massa 2009 <https://ui.adsabs.harvard.edu//#abs/2009ApJ...699.1209F/abstract>`_).
With a measurement of :math:`A(V)` then an absolute normalized extinction
measurement is possible using

.. math::
  A(x)/A(V) = [E(x - V) + A(V)]/A(V)

With a measurement of :math:`E(B-V)` and the extrapolated measurement of
:math:`A(V)`, then the total-to-selective extinction can be computed as
this is :math:`R(V) = A(V)/E(B-V)`.  :math:`R(V)` is diagnostic of the
average behavior of dust extinction as a function of wavelength
(`Cardelli, Clayton, & Mathis 1989 <https://ui.adsabs.harvard.edu//#abs/1989ApJ...345..245C/abstract>`_;
`Valencic et al. 2004 <https://ui.adsabs.harvard.edu//#abs/2004ApJ...616..912V/abstract>`_;
`Fitzpatrick & Massa 2007 <https://ui.adsabs.harvard.edu//#abs/2007ApJ...663..320F/abstract>`_;
`Gordon et al. 2009 <https://ui.adsabs.harvard.edu//#abs/2009ApJ...705.1320G/abstract>`_).
The :math:`R(V)` dependent relationship for the average extinction behavior
does not give the full picture as extinction curves in the Magellanic Clouds
strongly deviate from this relationship
(`Gordon et al. 2003 <https://ui.adsabs.harvard.edu//#abs/2003ApJ...594..279G/abstract>`_).

Terminology Summary
^^^^^^^^^^^^^^^^^^^

* :math:`m(x - V)` = magnitude difference between :math:`x` wavelength and V band
* :math:`E(x - V)` = extinction excess between :math:`x` wavelength and V band
* :math:`A(V)` = V band extinction
* :math:`R(V) = A(V)/E(B-V)` = total-to-selective extinction

Example
-------

Spectra
^^^^^^^

HD 283809 and HD 64802 are two stars with similar spectral types have
observed ultraviolet spectra from IUE or HST/STIS and optical/NIR photometry
from ground-based observations.  These two stars differ in that HD 283809
is seen through a large column of dust and HD 64802 is seen through
very little.

The DAT file for both stars is given in the data directory for this
package along with the STIS (HD 283809) and IUE (HD 64802) spectra.  Using
these files, the spectra of both stars can be plotted by reading in the
observed data using the StarData object and then calling the member function
plot_obs.

.. code-block:: python

   from measure_extinction.stardata import StarData
   starobs = StarData(dat_filename)

   fig, ax = plt.subplots()
   starobs.plot_obs(ax)

The spectra for both stars are plotted using those data files.  Which star
is reddened is clear as it has a non-stellar slope for a early type star
and clearly shows the 2175 A absorption feature.

.. plot::

   import pkg_resources

   from measure_extinction.stardata import StarData

   # get the location of the data files
   data_path = pkg_resources.resource_filename('measure_extinction',
                                               'data/')

   # read in the observed data on the star
   redstar = StarData('hd283809.dat', path=data_path)
   compstar = StarData('hd064802.dat', path=data_path)

   fig, ax = plt.subplots()

   # plot the bands and all spectra for both stars
   redstar.plot_obs(ax, pcolor='r')
   compstar.plot_obs(ax, pcolor='b')

   # finish configuring the plot
   ax.set_title('HD 283809 (reddened) & HD 64802 (comparion)')
   ax.set_yscale('log')
   ax.set_xscale('log')
   ax.set_ylim(1e-17, 1e-9)
   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel('$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]')
   ax.tick_params('both', length=10, width=2, which='major')
   ax.tick_params('both', length=5, width=1, which='minor')

   # use the whitespace better
   fig.tight_layout()

   plt.show()

Extinction
^^^^^^^^^^

Measuring the extinction is done by reading in observed data for both
stars in to StarData objects and then using an ExtData object and the
calc_elv member function.

.. code-block:: python

   from measure_extinction.stardata import StarData
   from measure_extinction.extdata import ExtData

   redstar = StarData(red_dat_filename)
   compstar = StarData(comp_dat_filename)

   extdata = ExtData()
   extdata.calc_elv(redstar, compstar)

   fig, ax = plt.subplots()
   extdata.plot_ext(ax)

.. plot::

   import pkg_resources

   from measure_extinction.stardata import StarData
   from measure_extinction.extdata import ExtData

   # get the location of the data files
   data_path = pkg_resources.resource_filename('measure_extinction',
                                               'data/')

   # read in the observed data on the star
   redstar = StarData('hd283809.dat', path=data_path)
   compstar = StarData('hd064802.dat', path=data_path)

   # calculate the extinction curve
   extdata = ExtData()
   extdata.calc_elv(redstar, compstar)

   fig, ax = plt.subplots()

   # plot the bands and all spectra for this star
   extdata.plot_ext(ax)

   # finish configuring the plot
   ax.set_title('HD 283809/HD 64802 extinction')
   ax.set_xscale('log')
   #ax.set_ylim(1e-17, 1e-9)
   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel('$E(\lambda - V)$ [mag]')
   ax.tick_params('both', length=10, width=2, which='major')
   ax.tick_params('both', length=5, width=1, which='minor')

   # use the whitespace better
   fig.tight_layout()

   plt.show()

Normalization
^^^^^^^^^^^^^

Derive A(V).  Transform E(l-V) to A(l)/A(V).  Calculate R(V).

Comparison to Models
^^^^^^^^^^^^^^^^^^^^

Show comparisons to existing R(V) dependent models using dust_extinction.
