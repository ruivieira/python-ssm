[![Build Status](https://travis-ci.org/ruivieira/python-ssm.svg?branch=master)](https://travis-ci.org/ruivieira/python-ssm)
[![PyPI version](https://badge.fury.io/py/pssm.svg)](https://badge.fury.io/py/pssm)
[![Downloads](http://pepy.tech/badge/pssm)](http://pepy.tech/project/pssm)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ruivieira/python-ssm/master?filepath=docs%2Fstate_space_models.ipynb)
# Python state-space models (`pssm`)

A Python package for state-space models. Basic usage in the [example notebooks](docs/state_space_models.ipynb).

# Features

  - [ ] Dynamic Generalised Linear Models
     - [x] Normal DLM
     - [x] Poisson DLM
     - [x] Binomial DLM
     - [x] `iterator`
  - [ ] Model composition
     - [x] Locally constant
     - [x] Locally linear
     - [x] Cyclic Fourier
     - [x] ARMA(*p*)
  - [ ] Multivariate
     - [x] Composite DGLMs
     - [ ] Multivariate Gaussian
  - [ ] Filters
     - [x] Kalman filter 