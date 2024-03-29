import math
import config
import numpy as np
from abc import abstractmethod
from scipy.optimize import curve_fit


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
    def calc_line_dispersion(self, pars):
        pass

    @staticmethod
    def calc_fwhm(x, ym):
        axis = ym.ndim - 1
        max_y = np.max(ym, axis=axis)
        if axis == 1: max_y = max_y[:, None]
        half_max = max_y / 2.
        left = np.argmax(ym >= half_max, axis=axis)
        right = np.argmax(ym[::-1] >= half_max, axis=axis)
        return x[::-1][right] - x[left]

    @staticmethod
    def vectorize_parameters(*pars):
        new_pars = []
        for par in pars:
            new_pars.append(np.expand_dims(np.array(par), 0))
        return new_pars

    def fit(self, wl, fl, ivar, n_components):
        x0, bounds = self.pre_fit(n_components)
        model = self.composed_model
        pars, pcov = curve_fit(model, wl, fl, p0=x0, bounds=bounds, sigma=np.sqrt(1 / ivar), maxfev=500000)
        return pars, pcov

    def fit_ensamble(self, wl, fl_mocks, ivar, n_components, n_tries):
        pars_list = []
        for i in range(n_tries):
            pars = self.fit(wl, fl_mocks[i], ivar, n_components)
            pars_list.append(pars)
        return pars_list

    def calc_line(self, pars):
        return self.composed_model(self.x_bin, *pars)


class Gaussians(BasicModel):
    def __init__(self):
        super().__init__()
        self.x0 = [1, config.LINE_CENTROID, 1]

    def base_model(self, x, *pars):
        x, a, mean, sigma = self.vectorize_parameters(x, *pars)
        gaus = a * np.exp(-((x.T - mean) / sigma) ** 2 / 2.)
        if gaus.shape[1] == 1: gaus = gaus[:, 0]
        return gaus

    def composed_model(self, x, *pars_list):
        models = []
        dim = np.array(pars_list).ndim
        if dim == 1: pars_list = [pars_list]
        for pars in pars_list:
            model = 0
            for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3]):
                model += self.base_model(x, a, mean, sigma)
            models.append(model)
        if dim == 1: models = models[0]
        return np.array(models)

    def pre_fit(self, n_components):
        init_guess = self.x0 * n_components
        bounds = [[0, 0, 0] * n_components, [np.inf, np.inf, np.inf] * n_components]
        return init_guess, bounds

    def calc_line_params(self, pars):
        integral, dispersion = self.calc_line_dispersion(pars)
        x_bin = np.linspace(config.TRIM_INTERVALS[0], config.TRIM_INTERVALS[-1], num=10000)
        y_gaus = self.composed_model(x_bin, *pars)
        fwhm = self.calc_fwhm(x_bin, y_gaus)
        return dispersion, fwhm, integral

    def calc_line_dispersion(self, pars_list):
        integral_list = []
        dispersion_list = []
        dim = np.array(pars_list).ndim
        if dim == 1: pars_list = [pars_list]
        for pars in pars_list:
            area = sum([a * sigma for a, sigma in zip(pars[::3], pars[2::3])])
            integral = np.sqrt(2 * math.pi) * area
            centroid = sum([a * mean * sigma for a, mean, sigma in zip(pars[::3], pars[1::3], pars[2::3])]) / area
            variance = sum([(mean ** 2 + sigma ** 2) * a * sigma for a, mean, sigma in
                            zip(pars[::3], pars[1::3], pars[2::3])]) / area - centroid * centroid
            integral_list.append(integral)
            dispersion_list.append(np.sqrt(variance))
        if dim == 1:
            integral_list = integral_list[0]
            dispersion_list = dispersion_list[0]
        return integral_list, dispersion_list


def set_model(model):
    if model == 'gaussians' or model == 'gaussian':
        return Gaussians()
    else:
        raise "This model is not implemented yet."


def continuum(wl, flux):
    continuum_mask = (
            ((wl >= config.CONTINUUM_INTERVALS[0]) &
             (wl < config.CONTINUUM_INTERVALS[1])) |
            ((wl >= config.CONTINUUM_INTERVALS[2]) &
             (wl < config.CONTINUUM_INTERVALS[3]))
    )
    wl = wl[continuum_mask]
    flux = flux[continuum_mask]
    m, q = np.polyfit(wl, flux, 1)
    return m, q


def continuum_ensamble(wl, fl_mocks):
    m_list, q_list = [], []
    for i in range(fl_mocks.shape[0]):
        m, q = continuum(wl, fl_mocks[i])
        m_list.append(m)
        q_list.append(q)
    return m_list, q_list


if __name__ == '__main__':
    from Spectrum import Spectrum

    file_path = 'examples/sample.fits'
    redshift = 3
    obj = Spectrum(file_path, redshift=redshift)
    wl = obj.wavelength
    fl = obj.flux
    ivar = obj.ivar

    model = Gaussians()

    length = len(wl)
    n_tries = 100
    fl_mocks = np.random.normal(fl, ivar, size=(n_tries, length))

    pars_list = model.fit_ensamble(wl, fl_mocks, ivar, 2, n_tries)
    print(len(pars_list))
    line_pars = model.calc_line_params(pars_list)
    print(line_pars)
