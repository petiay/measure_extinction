.. _model standards:

===========================
Stellar Models as Standards
===========================

Background
----------

The usual method for measuring dust extinction is to use observations
of two stars with identical spectral types, one that is reddened and one
that is unreddened/lightly reddened.  The unreddened/lightly reddened star
is termed the standard meaning what a star of that spectral type looks like
without foreground dust.  The main pro in using observations for
the standard
is then the resulting extinction measurement only depends on the relative
calibration of the instrument used.  The main con is that getting a "perfect"
match of spectral types is rare and, hence, spectral mismatches impact the
accuracy of the resulting extinction measurements.

Alternatively, stellar atmosphere models can be used for the unreddened
standard stars.  This means that better spectral matching is possible, but
with the cost of now being dependent on the instrumental calibration of the
reddened star observation.  A good reference for using models as standards
is
`Fitzpatrick & Massa (2005) <https://ui.adsabs.harvard.edu/abs/2005AJ....130.1127F/abstract>`_.

In summary:

Using observed standards:

- pro: sensitive to relative calibration only
- pro: empirical matching of stellar physics
- con: requires observations of the reddened and unreddened stars
- con: spectral mismatch between the two stars

Using stellar models for standards:

- pro: better spectral matches
- pro: only need to observe the reddened star
- con: models are approximate at some level (can be missing lines/physics)
- con: dependent on the absolute calibration

Model Fitting
-------------

When using a model as the standard, fitting the observed data can be done to 
determine the stellar and dust extinction parameters.  This is done by using
a grid of stellar atmosphere models with a model for the dust extinction curve.
The model of the dust extinction curve that is often used is a combination of
a FM90 parameterization for the ultraviolet and and a R(V) dependent model for the 
longer wavelengths that are joined with carefully chosen splines.

Fitting is supported through the `ModelData` and `MEModel` classes.  The `ModelData`
class stores the stellar atmosphere mocked data.  The `MEModel` class has all the 
model parameters and functions to compute dust extinguished model data, fit with 
a minimizer or a sampler observed data, and plot the resulting fits including
diagnostic plots.

.. toctree::
   :maxdepth: 2

   Model Details <model_capabilities.rst>
   Model Data <model_data.rst>
