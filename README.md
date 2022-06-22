# temspGANs

## Goal
Simulation on the Gaussian Random Field (GRF) to verify the generality of transformer-based GANs

## Simulation experiment 
$\mathbf{X}(s) \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma(\theta))$, 
where each individuals in the corvariance matrix, $\Sigma(\theta)_{ij} = \sigma^2 exp(-\frac{||s1 - s2||}{\alpha})$,
could be estimated by Maximum Likelihood Estimation (MLE)
