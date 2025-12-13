# Biscale Spectral Convolution

Biscale Spectral Convolution is an extension of standard [Spectral Convolution](https://arxiv.org/abs/1506.03767) used in [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) that allows to process inputs discretised on grids with different resolutions. This is done by keeping two separate channels for distinct resolutions that interact only through a Fourier kernel like in the picture below.

<img src="supplementary/biscale_spectral convolution.png" width="400" height="auto" alt="Fourier kernel of bispectral convolution.">