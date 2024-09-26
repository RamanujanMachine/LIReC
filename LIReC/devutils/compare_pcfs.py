from LIReC.lib.pcf import PCF
from scipy.optimize import curve_fit
from gmpy2.gmpy2 import mpq, mpfr, log, gcd
from mpmath import mp
import numpy as np


def foo(x, a, b, c, d):
    """The function to match the curve to"""
    return a * x * np.log(x) + b * x + c * np.log(x) + d


def get_fit(x: list[mpq], y: list[mpq], cutoff: mpq):
    """
    Calculates the fit to the input data. If the lead value is zero,
    assumes it is zero and calculate the other values with that assumption,
    until finding a leading value that is not zero.
    Gets x and y, the data to fit to, and cutoff, which is the closeness to zero needed to skip the lead value
    Returns a list of four elements, which are the coefficients to foo
    """
    popt, _ = curve_fit(foo, x, y)
    if abs(popt[0]) < cutoff:
        popt, _ = curve_fit(lambda x, b, c, d: foo(x, 0, b, c, d), x, y)
        if abs(popt[0]) < cutoff:
            popt, _ = curve_fit(lambda x, c, d: foo(x, 0, 0, c, d), x, y)
            if abs(popt[0]) < cutoff:
                popt, _ = curve_fit(lambda x, d: foo(x, 0, 0, 0, d), x, y)
                if abs(popt[0]) < cutoff:
                    return curve_fit(foo, x, y)[0]
                else:
                    return np.append([0, 0, 0], popt)
            else:
                return np.append([0, 0], popt)
        else:
            return np.append([0], popt)
    else:
        return popt


def get_measure(L: mpq, p: int, q: int):
    """
    Gets the delta, like Uri's code
    Gets the limit, and p and q which should be the numerator and denominator respectively, calculated to half of L's depth
    """
    if q + p == 0 or p == 0 or q == 0:
        return [-1010, -1010]
    q = abs(q)
    p = abs(p)

    qGCD = mpfr(q / gcd(p, q), precision=10000)
    numerator = -(log(abs(L - mpq(p, q))))

    FRDelta = (numerator / log(qGCD)) - 1
    return FRDelta


def get_pcf_params(pcf: PCF, iterations=10000, num_points=50, L=None, until_fr_decided=False):
    """
    Gets characteristics of the inputed PCF
    Paramaters:
    pcf: Lirec PCF
    iterations: the depth to go before calculating the characteristics
    num_points: the number of points to use when matching a function to the PCF
    L: the limit, if already known
    until_fr_decided: if True, continues until the fr is decided

    Returns the limit, delta, convergence rates, and if the PCF has fr or not, or undecided.
    Convergence rates are coefficients of the function a * x * log(x) + b * x + c * log(x) + d.
    The further forward in the list, the more the coefficients affect the outcome.
    """
    if pcf.depth != 0:
        raise ValueError("Depth of PCF should be 0")

    points_max = iterations // 2 if L is None else iterations
    interval = points_max // num_points
    convergence_thresold = interval / 2000

    points = []
    calculated_vals_fr = []
    decided_fr: None | bool = None

    i = interval
    while i < points_max or until_fr_decided:
        pcf.eval(depth=i, force_fr=False, precision=0)
        points.append(pcf.value)
        if decided_fr is None:
            calculated_vals_fr.append(
                mpfr(log(gcd(pcf.mat[0][1], pcf.mat[1][1])) / i +
                     PCF([1], [1]).a_coeffs[0] * (-log(i) + 1))
            )

            if len(calculated_vals_fr) >= 3 and abs(calculated_vals_fr[-2] - calculated_vals_fr[-1]) > abs(
                    calculated_vals_fr[-2] - calculated_vals_fr[-3]):
                decided_fr = False
                until_fr_decided = False

            if len(calculated_vals_fr) >= 2 and abs(
                    calculated_vals_fr[-2] - calculated_vals_fr[-1]) < convergence_thresold:
                decided_fr = True
                until_fr_decided = False
        i += interval

    midpoint = (pcf.mat[0][1], pcf.mat[1][1])

    if L is None:  # if L is not known, calculate only halfway through the iterations so that L is significantly far away from the points
        pcf.eval(depth=iterations, force_fr=False, precision=0)
        L = pcf.value

    delta = get_measure(L, midpoint[0], midpoint[1])

    x = np.array(range(interval, points_max, interval))
    slopes = np.vectorize(lambda x: log(abs(x)))(L - np.array(points))
    convergence_rates = get_fit(x, slopes, 10 ** -(log(iterations) / log(10)))
    return L, delta, convergence_rates, decided_fr


def compare_pcfs(a: PCF, b: PCF, iterations=10000, ignore_pslq=False):
    """
    Compares PCFs to see if they are similiar in terms of their attributes.
    Gets the two PCFs to compare, a and b. Variable iterations is the depth to check.
    Return None, and the two arrays that are the results of get_pcf_params if the PCFs are not similiar.
    If they are similiar, returns a dictionary containing the pslq of the limits and of the convergence rates, and the two arrays.
    """
    L_a, delta_a, conv_rate_a, fr_a = get_pcf_params(a, iterations=iterations)
    L_b, delta_b, conv_rate_b, fr_b = res_b = get_pcf_params(b, iterations=iterations)
    L_a_mpmath = mp.mpf(L_a.numerator) / L_a.denominator
    L_b_mpmath = mp.mpf(L_b.numerator) / L_b.denominator
    pslq_res = mp.pslq([1, L_a_mpmath, L_b_mpmath, L_a_mpmath * L_b_mpmath], tol=0.00000001)
    # comparing limits
    if not ignore_pslq and (pslq_res is None or max([abs(i) for i in pslq_res]) > 100):
        return None, [L_a, delta_a, conv_rate_a, fr_a], [L_b, delta_b, conv_rate_b, fr_b]
    # comparing factorial reductions
    if fr_a is not None and fr_b is not None and fr_a != fr_b:
        return None, [L_a, delta_a, conv_rate_a, fr_a], [L_b, delta_b, conv_rate_b, fr_b]

    # comparing deltas
    if abs(delta_a - delta_b) > 10**-(log(iterations) / log(100)):
        return None, [L_a, delta_a, conv_rate_a, fr_a], [L_b, delta_b, conv_rate_b, fr_b]

    # comparing convergence rates
    for conv_a, conv_b in zip(conv_rate_a, conv_rate_b):
        if conv_a == conv_b == 0:
            continue
        conv_pslq = mp.pslq([1, conv_a, conv_b, conv_a * conv_b], tol=10**-(mp.log(iterations) / mp.log(100)))
        if pslq_res is not None and max([abs(i) for i in pslq_res]) < 10 **(log(iterations) / log(100)):
            break
        else:
            return None, [L_a, delta_a, conv_rate_a, fr_a], [L_b, delta_b, conv_rate_b, fr_b]
    return {"limit_pslq": pslq_res, "conv_pslq": conv_pslq}, [L_a, delta_a, conv_rate_a, fr_a], [L_b, delta_b, conv_rate_b, fr_b]