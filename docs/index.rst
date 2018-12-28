##################
Measure Extinction
##################

``measure_extinction`` is a python package to measure extinction
due to dust absorbing photons or scattering photons out of
the line-of-sight.
Extinction applies to the case for a star seen behind a foreground
screen of dust.
This package provides the tools to measure dust extinction curves using
observations of two effectively identical stars, differing only in that one
is seen through more dust than the other.

Models of the wavelength dependence of dust extinction can be found in the
`dust_extinction package <http://dust-extinction.readthedocs.io/>`_.
More complex geometries of stars and dust include additional radiative
transfer effects resulting in attenuation.
Models of the wavelength dependence of dust attenuation can be found in the
`dust_attenuation package <http://dust-attenuation.readthedocs.io/>`_.

This package is developed in the
`astropy affiliated package <http://www.astropy.org/affiliated/>`_
template.

User Documentation
==================

.. toctree::
   :maxdepth: 2

   Extinction Calculation <calc_extinction.rst>
   Observed Data Formats <data_formats.rst>
   Plotting <plotting.rst>
   Stellar Models as Standards <model_standards.rst>

Reporting Issues
================

If you have found a bug in ``measure_extinction`` please report it by creating a
new issue on the ``measure_extinction`` `GitHub issue tracker
<https://github.com/karllark/measure_extinction/issues>`_.

Please include an example that demonstrates the issue sufficiently so that
the developers can reproduce and fix the problem. You may also be asked to
provide information about your operating system and a full Python
stack trace.  The developers will walk you through obtaining a stack
trace if it is necessary.

Contributing
============

Like the `Astropy`_ project, ``measure_extinction`` is made both by and for its
users.  We accept contributions at all levels, spanning the gamut from
fixing a typo in the documentation to developing a major new feature.
We welcome contributors who will abide by the `Python Software
Foundation Code of Conduct
<https://www.python.org/psf/codeofconduct/>`_.

``measure_extinction`` follows the same workflow and coding guidelines as
`Astropy`_.  The following pages will help you get started with
contributing fixes, code, or documentation (no git or GitHub
experience necessary):

* `How to make a code contribution <http://astropy.readthedocs.io/en/stable/development/workflow/development_workflow.html>`_

* `Coding Guidelines <http://docs.astropy.io/en/latest/development/codeguide.html>`_

* `Try the development version <http://astropy.readthedocs.io/en/stable/development/workflow/get_devel_version.html>`_

* `Developer Documentation <http://docs.astropy.org/en/latest/#developer-documentation>`_


For the complete list of contributors please see the `dust_extinction
contributors page on Github
<https://github.com/karllark/measure_extinction/graphs/contributors>`_.

Reference API
=============

.. automodapi:: measure_extinction.stardata

.. automodapi:: measure_extinction.extdata

.. automodapi:: measure_extinction.merge_obsspec
