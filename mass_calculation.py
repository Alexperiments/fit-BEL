import numpy as np


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
            return 2 * np.log10(sigma/1000) + 0.53 * (luminosity - 44) + 6.73
        elif fwhm is not None:
            return 2 * np.log10(fwhm/1000) + 0.53 * (luminosity - 44) + 6.66


if __name__ == '__main__':
    sigmas = np.random.uniform(2000, 4000, 1000)
    luminosities = np.random.uniform(45, 47, 1000)
    print(calc_mass(luminosity=luminosities))
