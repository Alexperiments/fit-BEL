import math
import numpy as np
import config
import utils


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

    if (sigma is None) and (fwhm is None):
        raise "The function needs a measure of the line width."
    if (sigma is not None) and (fwhm is not None):
        raise "Choose one between sigma and FWHM."
    if relation == 'VP06':
        if sigma is not None:
            return 2 * np.log10(sigma / 1000) + 0.53 * (luminosity - 44) + 6.73
        elif fwhm is not None:
            return 2 * np.log10(fwhm / 1000) + 0.53 * (luminosity - 44) + 6.66


def calc_flux_from_continuum(m, q, lam=1350):
    return lam * m + q


def flux_to_lum(flux, dl):
    return np.log10(flux) + config.FLUX_LOG_UNITS + np.log10(4 * math.pi) + 2 * np.log10(dl * 3.086) + 48


def calc_edd_ratio(lamL1350, mass):
    Lbol = lamL1350 + np.log10(config.BOLOMETRIC_CORRECTION)
    Ledd = mass + 38 + np.log10(1.26)
    return Lbol - Ledd


def calc_mock_pars(best_pars, cov, n_tries):
    return np.random.multivariate_normal(best_pars, cov, size=n_tries)


def calc_params(spectrum_dict, redshift, fit_model):
    d = {}
    pars = spectrum_dict['fit_pars']
    continuum_flux = calc_flux_from_continuum(spectrum_dict['m'], spectrum_dict['q'],
                                              lam=config.CONTINUUM_LUMINOSITY_LAMBDA)
    line_disp, fwhm, area = fit_model.calc_line_params(pars)
    dl = utils.ned_calc(redshift)
    d['lineLuminosity'] = flux_to_lum(area, dl)
    d['FWHM'] = fwhm * 299792 / config.LINE_CENTROID
    d['lineDispersion'] = line_disp * 299792 / config.LINE_CENTROID
    d['continuumInvarLuminosity'] = flux_to_lum(continuum_flux, dl) + np.log10(1350)
    d['bolLuminosity'] = d['continuumInvarLuminosity'] + np.log10(config.BOLOMETRIC_CORRECTION)
    d['sigmaMass'] = calc_mass(d['continuumInvarLuminosity'], sigma=d['lineDispersion'])
    d['fwhmMass'] = calc_mass(d['continuumInvarLuminosity'], fwhm=d['FWHM'])
    d['sigmaEddRatio'] = calc_edd_ratio(d['continuumInvarLuminosity'], d['sigmaMass'])
    d['fwhmEddRatio'] = calc_edd_ratio(d['continuumInvarLuminosity'], d['fwhmMass'])
    return d


def calc_errors(spectrum_dict, redshift, fit_model, d):
    pars_list = spectrum_dict['fit_pars_list']
    d['continuumFluxErr'] = np.log10(spectrum_dict['continuumFluxErr'])
    dl = utils.ned_calc(redshift)

    # parameters standard deviation
    line_disp, fwhm, area = fit_model.calc_line_params(pars_list)
    d['lineLuminosityErr'] = np.std(flux_to_lum(area, dl))
    d['FWHMErr'] = np.std(fwhm) * 299792 / config.LINE_CENTROID
    d['lineDispersionErr'] = np.std(line_disp) * 299792 / config.LINE_CENTROID
    d['continuumInvarLuminosityErr'] = d['continuumFluxErr']
    d['bolLuminosityErr'] = d['continuumInvarLuminosityErr']
    d['sigmaMassErr'] = np.std(calc_mass(d['continuumInvarLuminosity'], sigma=d['lineDispersion']))
    d['fwhmMassErr'] = np.std(calc_mass(d['continuumInvarLuminosity'], fwhm=d['FWHM']))
    d['sigmaEddRatioErr'] = np.std(calc_edd_ratio(d['continuumInvarLuminosity'], d['sigmaMass']))
    d['fwhmEddRatioErr'] = np.std(calc_edd_ratio(d['continuumInvarLuminosity'], d['fwhmMass']))
    return d


if __name__ == '__main__':
    N_samples = 10

    dl = utils.ned_calc(3)

    A = np.random.normal(1, 0.1, N_samples)
    mean = np.random.normal(1500, 4, N_samples)
    sigma = np.random.normal(20, 1, N_samples)
    lumin = np.random.normal(46, 0.1, N_samples)
    f1350 = np.random.normal(1e-17, 1e-19, N_samples)
    x_bin = np.arange(1000, 2000)

    rand = np.random.normal(x_bin, x_bin, size=(100, 1000))
    print(rand.shape)
