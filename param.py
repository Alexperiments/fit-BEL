import math
import config
import numpy as np
from scipy.optimize import curve_fit


def calc_mass(luminosity, sigma=None, fwhm=None, relation='VP06'):
    """ This function calculate the central BH mass starting from a measure of
        the line width and an estimate of the continuum luminosity near the line,
        which can be retrieved from a single epoch spectrum.

        Parameters
        ----------
        luminosity: float or array_like
            invariant luminosity at a certain
            wavelength (e.g. 1350 \AA for VP06).
            It needs to be in Log10 scale.
        sigma: float or array_like
            line dispersion of the line. Default value is None.
        fwhm: float or array_like
            FWHM of the line. Default value is None.
        relation: str
            single epoch scale-relation. Default value is "VP06".

        Returns
        -------
        out: float or array_like
            mass estimate in Log10 scale, in unit of solar masses. """

    assert (sigma is not None) or (fwhm is not None), "The function needs a measure of the line width."
    assert not ((sigma is not None) and (fwhm is not None)), "Choose one between sigma and FWHM."
    assert sigma > 0
    if relation == 'VP06':
        if sigma is not None:
            return 2 * np.log10(sigma / 1000) + 0.53 * (luminosity - 44) + 6.73
        elif fwhm is not None:
            return 2 * np.log10(fwhm / 1000) + 0.53 * (luminosity - 44) + 6.66


def calc_line_dispersion(A, mean1, sigma1, B=0, mean2=0, sigma2=1, C=0, mean3=0, sigma3=1):
    area = (A * sigma1 + B * sigma2 + C * sigma3)
    integral = np.sqrt(2 * math.pi) * area
    centroid = (A * mean1 * sigma1 + B * mean2 * sigma2 + C * mean3 * sigma3) / area
    variance = ((mean1 ** 2 + sigma1 ** 2) * A * sigma1 + (mean2 ** 2 + sigma2 ** 2) * B * sigma2 + (
                mean3 ** 2 + sigma3 ** 2) * C * sigma3) / area - centroid * centroid
    return integral, np.sqrt(variance)


def calc_fwhm(x, ym):
    max_y = max(ym)  # Find the maximum y value
    xs = x[ym >= max_y / 2.]
    return max(xs) - min(xs)


def flux_to_lum(flux, dl):
    return np.log10(flux) - 17 + np.log10(4 * math.pi) + 2 * np.log10(dl * 3.086) + 48


def calc_edd_ratio(lamL1350, mass):
    Lbol = lamL1350 + np.log10(config.BOLOMETRIC_CORRECTION)
    Ledd = mass + 38 + np.log10(1.26)
    return Lbol - Ledd


def gaussian_model(x, A, mean1, sigma1, B=0, mean2=0, sigma2=1, C=0, mean3=0, sigma3=1):
    return (A * np.exp(-np.power((x - mean1) / sigma1, 2.) / 2.) +
            B * np.exp(-np.power((x - mean2) / sigma2, 2.) / 2.) +
            C * np.exp(-np.power((x - mean3) / sigma3, 2.) / 2.))


init_guess1 = [1, 1549, 30]
init_guess2 = [1, 1549, 30, 1, 1549, 30]
init_guess3 = [1, 1549, 30, 1, 1549, 30, 1, 1549, 30]
bounds1 = [[0, 0, 0], [np.inf, np.inf, np.inf]]
bounds2 = [2 * [0, 0, 0], 2 * [np.inf, np.inf, np.inf]]
bounds3 = [3 * [0, 0, 0], 3 * [np.inf, np.inf, np.inf]]

if __name__ == '__main__':

    wl = np.arange(1000)
    sigma = np.ones
    flux = gaussian_model(wl, 1, 10, 500)
    init_guess = [1, 10, 500]
    bounds = [[0,0,0], [np.inf, np.inf, np.inf]]
    par1, pcov1 = curve_fit(gaussian_model, wl, flux, p0=init_guess1, bounds=bounds, sigma=1, maxfev = 500000)
