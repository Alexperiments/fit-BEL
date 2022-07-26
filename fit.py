from scipy.optimize import curve_fit
import math
import numpy as np
import config


def vectorize_parameters(*pars):
    new_pars = []
    for par in pars:
        new_pars.append(np.expand_dims(np.array(par), 0))
    return new_pars


def basic_gaussian(x, a, mean, sigma):
    x, a, mean, sigma = vectorize_parameters(x, a, mean, sigma)
    gaus = a * np.exp(-((x.T - mean) / sigma) ** 2 / 2.)
    if gaus.shape[1] == 1: gaus = gaus[:, 0]
    return gaus


def gaussian_model(x, *pars):
    model = 0
    for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3]):
        model += basic_gaussian(x, a, mean, sigma)
    return model


def gaussian_pre_fit(n_components):
    init_guess = config.FIT_GAUSSIAN_X0 * n_components
    bounds = [[0, 0, 0] * n_components, [np.inf, np.inf, np.inf] * n_components]
    return init_guess, bounds


def calc_line_dispersion(*pars):
    area = sum([a * sigma for a, sigma in zip(pars[::3], pars[2::3])])
    integral = np.sqrt(2 * math.pi) * area
    centroid = sum([a * mean * sigma for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3])]) / area
    variance = sum([(mean ** 2 + sigma ** 2) * a * sigma for a, mean, sigma in
                    zip(pars[::3], pars[1::3], pars[2::3])]) / area - centroid * centroid
    return integral, np.sqrt(variance)


def calc_fwhm(x, ym):
    axis = ym.ndim - 1
    max_y = np.max(ym, axis=axis)
    if axis == 1: max_y = max_y[:, None]
    half_max = max_y / 2.
    left = np.argmax(ym >= half_max, axis=axis)
    right = np.argmax(ym[::-1] >= half_max, axis=axis)
    return x[::-1][right] - x[left]


def extract_param_gaussians(par):
    area, line_disp = calc_line_dispersion(*par)
    x_bin = np.linspace(config.TRIM_INTERVALS[0], config.TRIM_INTERVALS[-1], num=10000)
    y_gaus = gaussian_model(x_bin, *par)
    fwhm = calc_fwhm(x_bin, y_gaus)
    return line_disp, fwhm, area


def fit(wl, fl, ivar, n_components, mode='gaussian_mixture'):
    if mode == 'gaussian_mixture':
        x0, bounds = gaussian_pre_fit(n_components)
        model = gaussian_model
    else:
        raise ValueError("Fit mode not recognised.")
    return curve_fit(model, wl, fl, p0=x0, bounds=bounds, sigma=np.sqrt(1 / ivar), maxfev=500000)


if __name__ == '__main__':
    from Spectrum import Spectrum
    import matplotlib.pyplot as plt

    file_path = 'examples/sample.fits'
    redshift = 3
    obj = Spectrum(file_path, redshift=redshift)
    wl, fl = obj.get_spectrum()
    ivar = obj.get_ivar()

    pars, pcov = fit(wl, fl, ivar, 1)