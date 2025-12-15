# Cataclysmic Variable Peak Analysis using GP (Draft)

**Status:** Draft — work in progress  

This repository contains Python scripts developed to determine the times of individual maxima in the light curves of variable stars. 
It was written for cataclysmic variable **TCP J10240289+4808512**, but can be adapted for other stars with similar light curve data.

## Overview

The time moments of individual maxima are determined using **Gaussian Process (GP) regression**, applied separately to each peak. This approach is adopted because the light curves are noisy, irregularly sampled, and contain gaps. Moreover, the asymmetric shapes of the maxima make them unsuitable for fitting with Gaussian profiles or low-order polynomials, while the GP model can be applied without assuming a fixed analytical form.

For each maximum, the time of maximum is taken as the position of the maximum of the GP predictive mean. 
The uncertainty is therefore estimated using **posterior sampling**: multiple realisations of the light curve are drawn from the GP posterior, the time of maximum is determined for each realisation, and the dispersion of this set is adopted as the uncertainty of the moment of maximum.

Since GP fitting requires observational error estimates, and such errors are not available for all light curve points, heuristics are applied to estimate them. These heuristics are based on the scatter of points around smooth curves, providing rough error estimates for the GP.

## References

Gaussian Process regression has been previously applied to light curves fitting, i.g:  
  - Castro, N et al. 2018, AJ, 155, 16 "Uncertain Classification of Variable Stars: Handling Observational GAPS and Noise"
  - Trevisan, P. et al. 2023, ApJ, 950, 103 "Sparse Logistic Regression for RR Lyrae versus Binaries Classification"

