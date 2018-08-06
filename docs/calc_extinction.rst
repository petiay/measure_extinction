.. |Av| replace:: :math:`A(V)`
.. |Ebv| replace:: :math:`E(B-V)`
.. |Elv| replace:: :math:`E(\lambda-V)`

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
it to magnitudes.  Using observations taken with the same instrumentation
for both the reddened and comparison star means that only good relative
calibration is needed.  Instead of using observations of a star with little
to no reddening, a model stellar atmosphere can be used
('Fitzpatrick & Massa 2005 <https://ui.adsabs.harvard.edu//#abs/2005AJ....130.1127F/abstract>'_).
This can provide a better match, but it comes at the expense of requiring
good absolute calibration.

The basic measurement is given the magnitude excess relative to a
reference wavelength measurement.
This use of a reference wavelength is that the distance to a star is rarely
known to high enough accuracy to directly measure the extinction at a
single wavelength.
Thus, the basic measurement gives the relative extinction between two
wavelengths.
Usually, the V band measurement is used at the reference.
Thus, the dust extinction at :math:`\lambda` wavelength is:

.. math ::
  E(\lambda - V) = m(\lambda - V)_r - m(\lambda - V)_c

where :math:`m(\lambda - V)` is the difference in magnitudes between the flux at
:math:`\lambda` and the V band, :math:`r` refers to the
reddened star, and :math:`c` refers to the comparison star.

This extinction measurement can be normalized allowing comparison with
extinction along other lines-of-sight.
The normalization is often done to |Ebv| as this is easily measured.
But notice that this is a normalization to a differential measurement.

Converting the :math:`E(\lambda-V)` differential measurement to the
:math:`A(\lambda)`
absolute measurement requires knowledge of the the absolute extinction at the
reference wavelength, V band in this case.
Determining |Av| requires measuring :math:`E(\lambda-V)` at
the longest wavelength
possible and extrapolating with a reference shape to infinite wavelength
as :math:`A(inf) = 0`.
Usually, the longest wavelength measured is the K band and the extrapolation
to infinite wavelength is on the order of 10%
(`Whittet, van Breda, & Glass 1976 <https://ui.adsabs.harvard.edu//#abs/1976MNRAS.177..625W/abstract>`_;
`Fitzpatrick & Massa 2009 <https://ui.adsabs.harvard.edu//#abs/2009ApJ...699.1209F/abstract>`_).
With a measurement of |Av| then an absolute normalized extinction
measurement is possible using

.. math::
  A(\lambda)/A(V) = [E(\lambda - V) + A(V)]/A(V) = E(\lambda - V)/A(V) + 1

With a measurement of |Ebv| and the extrapolated measurement of
|Av|, then the total-to-selective extinction can be computed as
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

* :math:`m(\lambda - V)` = magnitude difference between :math:`\lambda` wavelength and V band
* :math:`E(\lambda - V)` = extinction excess between :math:`\lambda` wavelength and V band
* |Av| = V band extinction
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
package along with the STIS (HD 283809) and IUE (HD 64802) spectra.
For details of the format of these files, see :ref:`data_formats`.
Using these files, the spectra of both stars can be plotted by reading in the
observed data using the :class:`~measure_extinction.stardata.StarData` object
and then calling its member function plot_obs.

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

   # start the plotting
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
stars in to :class:`~measure_extinction.stardata.StarData` objects and
then using an :class:`~measure_extinction.extdata.ExtData` object and its
calc_elv member function.  The calc_elv function ratios the reddened to
comparison star relative to V band and coverts the results to magnitudes
resulting in :math:`E(\lambda - V)`.  The plot can then be shown using the
memebr function plot_ext.

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

   # start the plotting
   fig, ax = plt.subplots()

   # plot the bands and all spectra for this star
   extdata.plot_ext(ax)

   # finish configuring the plot
   ax.set_title('HD 283809/HD 64802 extinction')
   ax.set_xscale('log')
   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel('$E(\lambda - V)$ [mag]')
   ax.tick_params('both', length=10, width=2, which='major')
   ax.tick_params('both', length=5, width=1, which='minor')

   # use the whitespace better
   fig.tight_layout()

   plt.show()

Normalization
^^^^^^^^^^^^^

One common normalization is to divide by :math:`E(\lambda-V)`.  As long as
both the data used for the reddened and comparison stars inlcude B and V
measurements, :math:`E(\lambda-V)` has already been calculated.  The
:class:`~measure_extinction.extdata.ExtData` member function trans_elv_elvebv
performs this normalization while checking that the B band measurement
exists.

.. code-block:: python

   extdata.trans_elv_elvebv()

.. plot::

   import pkg_resources

   import numpy as np

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

   # divide by the E(B-V)
   extdata.trans_elv_elvebv()

   # start the plotting
   fig, ax = plt.subplots()

   # plot the bands and all spectra for this star
   extdata.plot_ext(ax)

   # finish configuring the plot
   ax.set_title('HD 283809/HD 64802 extinction')
   ax.set_xscale('log')
   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel('$E(\lambda - V)/E(B-V)$')
   ax.tick_params('both', length=10, width=2, which='major')
   ax.tick_params('both', length=5, width=1, which='minor')

   # use the whitespace better
   fig.tight_layout()

   plt.show()

Another common normalization is by |Av|.  This provides an absolute
normalization instead of the differential normalization provide by
|Ebv|.  In order to determine |Av|, the |Elv| curve is extrapolated to
infinite wavelength as :math:`A(inf) = 0`, thus :math:`E(inf - V) = -A(V)`.
In general, the longest wavelength easy to measure is K band so
:math:`E(K - V)` is often the measurement to extrapolated.
To do this extrapolation, a functional form the extinction curve at the
longest wavelengths must be assumed.
One choice is to assume the near-/mid-IR extinction curve from
`Rieke & Lebofsky 1985 <https://ui.adsabs.harvard.edu//#abs/1985ApJ...288..618R/abstract>`_.
The value for the K band extinction is give in Table 3 of this reference as
:math:`A(K)/A(V) = 0.112`.

.. math::
   A(K)/A(V) = E(K-V)/A(V) + 1

   0.112 = E(K-V)/A(V) + 1

   A(V) = E(K-V)/(0.112 - 1)

   A(V) = -1.126 E(K-V)

The :class:`~measure_extinction.extdata.ExtData` member function trans_elv_alav
performs this normalization.  Other choices for :math:`A(K)/A(V)` can be used
by setting the parameter `akav` in this member function.

.. code-block:: python

   # value from Rieke & Lebofsky (1985)
   extdata.trans_elv_alav(akav=0.112)

   # use value for van de Hulst No. 15 curve instead
   extdata.trans_elv_alav(akav=0.0885)

.. plot::

   import pkg_resources
   import copy

   import numpy as np

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

   # make a copy for use later
   extdata2 = copy.deepcopy(extdata)

   # divide by the A(V) derived with two different A(K)/A(V) assumptions
   extdata.trans_elv_alav(akav=0.112)
   extdata2.trans_elv_alav(akav=0.0885)

   # start the plotting
   fig, ax = plt.subplots()

   # plot the bands and all spectra for this star
   extdata.plot_ext(ax, color='b')
   extdata2.plot_ext(ax, color='g')

   # finish configuring the plot
   ax.set_title('HD 283809/HD 64802 extinction')
   ax.set_xscale('log')
   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel('$A(\lambda)/A(V)$')
   ax.tick_params('both', length=10, width=2, which='major')
   ax.tick_params('both', length=5, width=1, which='minor')

   # custom legend
   from matplotlib.lines import Line2D
   custom_lines = [Line2D([0], [0], color='b', lw=4),
                   Line2D([0], [0], color='g', lw=4)]
   ax.legend(custom_lines, ['A(K)/A(V): Reike & Lebofsky (1985)',
                            'A(K)/A(V): van de Hulst No. 15'])

   # use the whitespace better
   fig.tight_layout()

   plt.show()

Comparison to Models
^^^^^^^^^^^^^^^^^^^^

Compute R(V).

Show comparisons to existing R(V) dependent models using dust_extinction.
