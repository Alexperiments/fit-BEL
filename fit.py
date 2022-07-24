from scipy.optimize import curve_fit
import math
import numpy as np
import config


def basic_gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean) / sigma)**2 / 2.)


def gaussian_model(x, *pars):
    model = 0
    for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3]):
        model += basic_gaussian(x, a, mean, sigma)
    return model


def gaussian_pre_fit(n_components):
    init_guess = config.FIT_GAUSSIAN_X0 * n_components
    bounds = [[0, 0, 0] * n_components, [np.inf, np.inf, np.inf] * n_components]
    return init_guess, bounds


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


def fit(wl, fl, ivar, n_components, mode='gaussian_mixture'):
    if mode == 'gaussian_mixture':
        x0, bounds = gaussian_pre_fit(n_components)
        model = gaussian_model
    else:
        raise ValueError("Fit mode not recognised.")
    return curve_fit(model, wl, fl, p0=x0, bounds=bounds, sigma=np.sqrt(1 / ivar), maxfev=500000)


def extract_param_gaussians(par):
    area, line_disp = calc_line_dispersion(*par)
    x_bin = np.linspace(config.TRIM_INTERVALS[0], config.TRIM_INTERVALS[-1], num=10000)
    y_gaus = gaussian_model(x_bin, *par)
    fwhm = calc_fwhm(x_bin, y_gaus) * 299792 / 1549

    return line_disp, fwhm, area


if __name__ == '__main__':
    from Spectrum import Spectrum
    import matplotlib.pyplot as plt

    file_path = 'examples/sample.fits'
    redshift = 3
    obj = Spectrum(file_path, redshift=redshift)
    wl, fl = obj.get_spectrum()
    ivar = obj.get_ivar()

    print(wl)
    print(fl)
    print(ivar)