import numpy as np


def ned_calc(z, H0=70, Omega_m=0.3, Omega_vac=0.7):
    '''Script basato sul NED cosmology calculator, per stimare DL e V(Gpc)'''
    # initialize constants
    WM = Omega_m  # Omega(matter)
    WV = Omega_vac  # Omega(vacuum) or lambda
    WR = 0.  # Omega(radiation)
    WK = 0.  # Omega curvaturve = 1-Omega(total)
    c = 299792.458  # velocity of light in km/sec
    h = H0 / 100.
    WR = 4.165E-5 / (h * h)  # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1 - WM - WR - WV
    az = 1.0 / (1 + 1.0 * z)
    age = 0.

    n = 1000  # number of points in integrals
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = np.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        age = age + 1. / adot
    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = np.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DTT = DTT + 1. / adot
        DCMR = DCMR + 1. / (a * adot)
    DCMR = (1. - az) * DCMR / n

    # tangential comoving distance
    x = np.sqrt(abs(WK)) * DCMR
    if WK > 0:
        ratio = 0.5 * (np.exp(x) - np.exp(-x)) / x
    else:
        ratio = np.sin(x) / x
    DCMT = ratio * DCMR
    DA = az * DCMT
    DL = DA / (az * az)
    DL_Mpc = (c / H0) * DL
    return DL_Mpc