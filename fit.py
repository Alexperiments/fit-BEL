from scipy.optimize import curve_fit
import math
import numpy as np
import config
from abc import abstractmethod


class BasicModel:
    def __init__(self):
        self.x_bin = np.linspace(config.TRIM_INTERVALS[0], config.TRIM_INTERVALS[-1], num=10000)

    @abstractmethod
    def base_model(self, x, *pars):
        pass

    @abstractmethod
    def composed_model(self, *pars):
        pass

    @abstractmethod
    def pre_fit(self, n_components):
        pass

    @abstractmethod
    def calc_line_params(self, pars):
        pass

    @abstractmethod
    def calc_line_dispersion(self):
        pass

    def calc_fwhm(self, x, ym):
        axis = ym.ndim - 1
        max_y = np.max(ym, axis=axis)
        if axis == 1: max_y = max_y[:, None]
        half_max = max_y / 2.
        left = np.argmax(ym >= half_max, axis=axis)
        right = np.argmax(ym[::-1] >= half_max, axis=axis)
        return x[::-1][right] - x[left]

    def vectorize_parameters(self, *pars):
        new_pars = []
        for par in pars:
            new_pars.append(np.expand_dims(np.array(par), 0))
        return new_pars

    def fit(self, wl, fl, ivar, n_components):
        x0, bounds = self.pre_fit(n_components)
        model = self.composed_model
        pars, _ = curve_fit(model, wl, fl, p0=x0, bounds=bounds, sigma=np.sqrt(1 / ivar), maxfev=500000)
        return pars

    def calc_line(self, pars):
        return self.composed_model(self.x_bin, *pars)


class Gaussians(BasicModel):
    def base_model(self, x, *pars):
        x, a, mean, sigma = self.vectorize_parameters(x, *pars)
        gaus = a * np.exp(-((x.T - mean) / sigma) ** 2 / 2.)
        if gaus.shape[1] == 1: gaus = gaus[:, 0]
        return gaus

    def composed_model(self, x, *pars):
        model = 0
        for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3]):
            model += self.base_model(x, a, mean, sigma)
        return model

    def pre_fit(self, n_components):
        init_guess = config.FIT_GAUSSIAN_X0 * n_components
        bounds = [[0, 0, 0] * n_components, [np.inf, np.inf, np.inf] * n_components]
        return init_guess, bounds

    def calc_line_params(self, pars):
        integral, dispersion = self.calc_line_dispersion(*pars)
        x_bin = np.linspace(config.TRIM_INTERVALS[0], config.TRIM_INTERVALS[-1], num=10000)
        y_gaus = self.composed_model(x_bin, *pars)
        fwhm = self.calc_fwhm(x_bin, y_gaus)
        return dispersion, fwhm, integral

    def fit_ensamble(self, wl, fl, ivar, n_components, n_tries):
        length = len(wl)
        pars_list = []
        fl_mocks = np.random.normal(fl, ivar, size=(n_tries, length))
        for i in range(n_tries):
            pars = self.fit(wl, fl_mocks[i], ivar, n_components)
            pars_list.append(pars)
        return pars_list

    def calc_line_dispersion(self, *pars):
        area = sum([a * sigma for a, sigma in zip(pars[::3], pars[2::3])])
        integral = np.sqrt(2 * math.pi) * area
        centroid = sum([a * mean * sigma for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3])]) / area
        variance = sum([(mean ** 2 + sigma ** 2) * a * sigma for a, mean, sigma in
                        zip(pars[::3], pars[1::3], pars[2::3])]) / area - centroid * centroid
        return integral, np.sqrt(variance)


def set_model(model):
    if model=='gaussians' or model=='gaussian':
        return Gaussians()
    else:
        raise "This model is not implemented yet."


if __name__ == '__main__':
    from Spectrum import Spectrum
    from time import time

    file_path = 'examples/sample.fits'
    redshift = 3
    obj = Spectrum(file_path, redshift=redshift)
    wl = obj.wavelength
    fl = obj.flux
    ivar = obj.ivar

    model = Gaussians()

    t0 = time()
    pars_list = model.fit_ensamble(wl, fl, ivar, 1, n_tries=100)
    print(time()-t0)
    print(pars_list.shape)