from collections import Counter
from enum import Enum
from functools import reduce
from itertools import chain, combinations, combinations_with_replacement, count, takewhile
from operator import mul, add, and_
from re import match, subn
from sympy import sympify, Symbol
from traceback import format_exc
from typing import List
import mpmath as mp

MIN_PSLQ_DPS = 15
PRECISION_RATIO = 0.75

import sys
try: # just disable it
    sys.set_int_max_str_digits(0)
except:
    pass

class Confidence(Enum):
    No = 0
    Minimal = 1
    Moderate = 2
    High = 3
    Extreme = 4
    Theorem = 5

class PreciseConstant:
    value: mp.mpf
    precision: int
    symbol: str
    
    def __init__(self, value, precision, symbol=None):
        self.precision = int(precision)
        # these values are intended to feed into PSLQ later, and if we try to initialize them with
        # less precision than the minimum with which PSLQ can work, it will cause problems
        with mp.workdps(max(self.precision, MIN_PSLQ_DPS)):
            self.value = mp.mpf(str(value))
        self.symbol = symbol
    
    def to_json(self) -> dict:
        with mp.workdps(self.precision):
            d = {'value': str(self.value), 'precision': self.precision}
            if self.symbol:
                d['symbol'] = str(self.symbol)
            return d

def _latexify(name: str) -> str:
    '''
    formats the given constant's name in LaTeX format.
    
    (this would be in calculator.Constants, but that would cause a circular dependency...
     TODO put calculator.Constants to its own file)
    '''
    if name == 'e':
        return name
    
    if name[0] == 'e':
        exp = _latexify(name[1:])
        if len(exp) > 1:
            exp = f'{{{exp}}}'
        return f'e^{exp}'
    
    if 'cbrt' in name:
        name = f'root3of{name.replace("cbrt","")}'
    
    root = match(r'root(\d+)of(\w+)', name)
    if root and root[0] == name:
        return fr'\sqrt[{root[1]}]{{{_latexify(root[2])}}}'
    
    groups = match(r'([A-Za-z]*)(_?)([A-Za-z]*)(\w*)', name)
    if groups[0] != name:
        raise ValueError(f'cannot latexify {name}')
    
    greek_letters = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta ', 'theta', 'iota', 'kappa', 'lambda', 'mu  ', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
    special_funcs = ['log', 'ln', 'sqrt', 'zeta', 'Gamma']
    special_funcs_parentheses = ['zeta', 'Gamma']
    
    first = groups[1].replace('prime', "'").replace('dot','.')
    second = groups[3]
    if second.lower() in greek_letters:
        second = f'\\{second}'
    second += groups[4]
    second = second.replace('dot','.')
    
    if (groups[2] and len(second) > 1) or first in special_funcs:
        second = f'({second})' if first in special_funcs_parentheses else f'{{{second}}}'
    if first.lower() in greek_letters or first in special_funcs:
        first = f'\\{first}'
    
    return f'{first}{groups[2]}{second}'

class PolyPSLQRelation:
    constants: List[PreciseConstant]
    degree: int
    order: int
    coeffs: List[int]
    isolate: int or str or None # or Symbol
    include_isolated: bool
    latex_mode: bool
    _confidence: Confidence or None
    ranges: List[mp.mpf]
    
    def __fix_isolate(self):
        # need isolate as a str eventually
        if isinstance(self.isolate, bool):
            return # do nothing! keep it as is!
        if isinstance(self.isolate, int):
            self.isolate = (self.constants[self.isolate].symbol or f'c{self.isolate}') if self.isolate < len(self.constants) else None
        try: # will become a Symbol (or stay None if is None)
            self.isolate = sympify(self.isolate)
        except:
            try: # try to force it into a symbol instead...
                self.isolate = Symbol(self.isolate)
            except:
                pass
    
    def __fix_symbols(self):
        if self.latex_mode:
            for c in self.constants:
                try:
                    c.symbol = Symbol(_latexify(str(c.symbol)))
                except:
                    # either it's not in the named constant format, or it's
                    # already latexified. either way, nothing to do
                    pass
    
    def __init__(self, consts, degree, order, coeffs, isolate=None, include_isolated=True, latex_mode=False, confidence=None, ranges=None):
        self.constants = consts
        self.degree = degree
        self.order = order
        self.coeffs = coeffs
        self.isolate = isolate
        self.include_isolated = include_isolated
        self.latex_mode = latex_mode
        self._confidence = confidence
        self.ranges = ranges or [mp.mpf('1.25'), mp.mpf('1.5'), mp.mpf(2), mp.mpf(3)]
    
    def __str__(self):
        exponents = get_exponents(self.degree, self.order, len(self.constants))
        for i,c in enumerate(self.constants): # verify symbols
            if not c.symbol:
                c.symbol = f'c{i}'
            if not isinstance(c.symbol, Symbol):
                c.symbol = Symbol(c.symbol)
        
        self.__fix_isolate()
        self.__fix_symbols()
        monoms = [reduce(mul, (c.symbol**exp[i] for i,c in enumerate(self.constants)), self.coeffs[j]) for j, exp in enumerate(exponents)]
        expr = sympify(reduce(add, monoms, 0))
        res = None
        if self.isolate not in expr.free_symbols or not expr.is_Add: # checking is_Add just in case...
            res = f'{expr} = 0 ({self.precision})'
        else:
            # expect expr to be Add of Muls or Pows, so args will give the terms
            # so now the relation is (-num) + (denom) * isolate = 0, or isolate = num/denom!
            num = reduce(add, [-t for t in expr.args if self.isolate not in t.free_symbols], 0)
            denom = reduce(add, [t/self.isolate for t in expr.args if self.isolate in t.free_symbols], 0)
            # this will not be perfect if isolate appears with an exponent! will also be weird if either num or denom is 0
            res = (fr'\frac{{{num}}}{{{denom}}}' if self.latex_mode else f'{num/denom}') + f' ({self.precision})' 
            res = (f'{self.isolate} = ' if self.include_isolated else '') + res
        # finally perform latex_mode substitution for exponents if necessary
        return subn('\*\*(\w+)', '**{\\1}', res)[0] if self.latex_mode else res

    @property
    def precision(self):
        return poly_eval(poly_get(self.constants, get_exponents(self.degree, self.order, len(self.constants))),
                         self.coeffs, [c.precision for c in self.constants])
    
    @property
    def precision_binary(self):
        return poly_eval(poly_get(self.constants, get_exponents(self.degree, self.order, len(self.constants))),
                         self.coeffs, [c.precision for c in self.constants], 2)
    
    @property
    def roi(self):
        substance = len([x for x in self.coeffs if x]) + sum(x.bit_length() for x in self.coeffs)
        return self.precision_binary / substance
    
    @property
    def confidence(self):
        if self._confidence:
            return self._confidence
        roi = self.roi
        if roi < self.ranges[0]:
            return Confidence.No
        if roi < self.ranges[1]:
            return Confidence.Minimal
        if roi < self.ranges[2]:
            return Confidence.Moderate
        if roi < self.ranges[3]:
            return Confidence.High
        return Confidence.Extreme
    
    def to_json(self) -> dict:
        return {'constants' : [c.to_json() for c in self.constants], 'degree': self.degree, 'order': self.order, 'coeffs': self.coeffs }

def cond_print(verbose, m):
    if verbose:
        print(m)

def get_exponents(degree, order, total_consts):
    if degree == None or order == None:
        raise Exception('degree and order cannot be None')
    return [c for c in map(Counter, chain.from_iterable(combinations_with_replacement(range(total_consts), i) for i in range(degree + 1)))
            if not any(i for i in c.values() if i > order)]

def poly_get(consts, exponents):
    mp.mp.dps = max(min(c.precision for c in consts), MIN_PSLQ_DPS) # must be at least 15!
    values = [c.value for c in consts]
    return [reduce(mul, (values[i] ** exp[i] for i in range(len(values))), mp.mpf(1)) for exp in exponents]

def poly_eval(poly, coeffs, precisions, base=10):
    with mp.workdps(max(precisions) + 10):
        min_prec = mp.floor(max(min(precisions), MIN_PSLQ_DPS) * mp.log(10, base))
        return int(min(mp.floor(-mp.log(abs(mp.fdot(poly, coeffs)), base)), min_prec))

def poly_verify(consts, degree = None, order = None, relation = None, full_relation = None, exponents = None):
    if full_relation:
        degree = full_relation[0]
        order = full_relation[1]
        relation = full_relation[2:]
    if not exponents:
        exponents = get_exponents(degree, order, len(consts))
    poly = poly_get(consts, get_exponents(degree, order, len(consts)))
    return poly_eval(poly, relation, [c.precision for c in consts]) if poly else None

def poly_check(consts, degree = None, order = None, exponents = None, test_prec = 15, min_roi = 2, verbose = False):
    if not exponents:
        exponents = get_exponents(degree, order, len(consts))
    try:
        poly = poly_get(consts, exponents)
        if poly:
            precs = [c.precision for c in consts]
            true_min = min(precs)
            tol = None
            if true_min < MIN_PSLQ_DPS: # otherwise let pslq automatically set tol
                tol_offset = mp.floor(mp.log10(abs(consts[precs.index(true_min)].value))) - max(mp.floor(mp.log10(abs(x))) for x in poly)
                tol = mp.mpf(10)**((tol_offset - min(11,int(true_min))) * PRECISION_RATIO)
            with mp.workdps(max(test_prec, MIN_PSLQ_DPS)): # intentionally low-resolution to quickly try something basic...
                res = pslq(poly, tol, mp.inf, mp.inf, verbose)
            # don't know why, but when testing PSLQ on random vectors, the expected amount of total digits in the
            # result is approximately the same as the working precision. this is the justification for calculating roi
            if res:
                res = PolyPSLQRelation(consts, degree, order, res)
                roi = res.roi
                if roi >= min_roi:
                    return res
                elif verbose:
                    print(f'GARBAGE RELATION FOUND. roi={roi} < min_roi={min_roi}')
    except ValueError:
        # one of the constants has too small precision, or one constant
        # is small enough that another constant is smaller than its precision.
        # eitherway there's no relation to be found here!
        # TODO for now assuming that the latter case (one constant is smaller than another constant's precision) doesn't happen,
        # solve this later by normalizing everything or something
        cond_print(verbose, f'poly_check failed, no relation will be returned. {format_exc()}')
    return None

def compress_relation(result, consts, exponents, degree, order, verbose=False):
    # will need to use later, so evaluating into lists
    cond_print(verbose, f'Original relation is {result}')
    
    consts = list(consts) # shallow copy to preserve the original list
    
    nonzero = [pair for pair in zip(exponents, result) if pair[1]] # first remove obvious common factors (if any exist)
    common = reduce(and_, [c for c,_ in nonzero])
    # could have used dict here but need Counters for computing common, and Counter can't be a key
    result = [([x for test_e,x in nonzero if e + common == test_e] + [0])[0] for e in exponents]
    
    indices_per_var = [[i for i, e in enumerate(exponents) if e[j]] for j in range(len(consts))]
    redundant_vars = [i for i, e in enumerate(indices_per_var) if not any(result[j] for j in e)]
    redundant_coeffs = set()
    for redundant_var in sorted(redundant_vars, reverse=True): # remove redundant variables
        cond_print(verbose, f'Removing redundant variable #{redundant_var}')
        redundant_coeffs |= set(indices_per_var[redundant_var])
        del consts[redundant_var]
    
    # remove redundant degrees and orders
    indices_per_degree = [[i for i, e in enumerate(exponents) if sum(e.values()) == j] for j in range(degree + 1)]
    redundant_degrees = [i for i, e in enumerate(indices_per_degree) if not any(result[j] for j in e)]
    redundant_degrees = list(takewhile(lambda x: sum(x) == degree, enumerate(sorted(redundant_degrees, reverse=True))))
    if redundant_degrees:
        degree = redundant_degrees[-1][1] - 1
    redundant_coeffs.update(*indices_per_degree[degree+1:])
    
    indices_per_order = [[i for i, e in enumerate(exponents) if max(e.values(), default=0) == j] for j in range(order+1)]
    redundant_orders = [i for i, e in enumerate(indices_per_order) if not any(result[j] for j in e)]
    redundant_orders = list(takewhile(lambda x: sum(x) == order, enumerate(sorted(redundant_orders, reverse=True))))
    if redundant_orders:
        order = redundant_orders[-1][1] - 1
    redundant_coeffs.update(*indices_per_order[order+1:])
    
    cond_print(verbose, f'True degree and order are {degree, order}')
    for i in sorted(redundant_coeffs, reverse=True):
        del result[i]
    
    cond_print(verbose, f'Compressed relation is {result}')

    return PolyPSLQRelation(consts, degree, order, result)

def combination_is_old(consts, degree, order, other_relations): # if the combination is old, returns a relation "as proof". else returns None
    return ([r for r in other_relations
             if {c.symbol for c in r.constants} <= {c.symbol for c in consts} and r.degree <= degree and r.order <= order] + [None])[0]

def check_subrelations(relation: PolyPSLQRelation, test_prec=15, min_roi=2, extra_relations=None, verbose=False):
    subrelations = []
    extra_relations = extra_relations if extra_relations else []
    for i in range(1, len(relation.constants)):
        exponents = get_exponents(relation.degree, relation.order, i)
        for subset in combinations(relation.constants, i):
            if combination_is_old(subset, relation.degree, relation.order, subrelations + extra_relations):
                continue
            subresult = poly_check(subset, relation.degree, relation.order, exponents, test_prec, min_roi, verbose)
            if subresult:
                subresult = compress_relation(subresult.coeffs, list(subset), exponents, relation.degree, relation.order)
                if subresult.precision >= PRECISION_RATIO * min(c.precision for c in subresult.constants):
                    subrelations += [subresult]
    # sometimes the relation can have no constants due to precision issues! this should catch it
    # also need to independently check if true_prec is significant when compared to the constants themselves,
    # since test_prec is typically much lower than the maximum reasonable precision
    return subrelations if subrelations else [] if not relation.constants else [relation]

def check_consts(consts: List[PreciseConstant], degree=2, order=1, test_prec=15, min_roi=2, strict=False, verbose=False):
    min_order, max_order = 1, order # first "binary search" over the orders
    result, true_prec = [], -1
    while max_order > min_order:
        avg_order = (min_order + max_order) // 2
        exponents = get_exponents(degree, avg_order, len(consts)) # need later for compress_relation
        rel = poly_check(consts, degree, avg_order, exponents, test_prec, min_roi, verbose)
        if rel:
            result, true_prec = rel.coeffs, rel.precision
            relation = compress_relation(result, consts, exponents, degree, avg_order)
            consts, degree, max_order = relation.constants, relation.degree, relation.order
        else:
            min_order = avg_order + 1
    # now min_order == max_order == avg_order, and either the minimal order has been found, or no relation exists
    exponents = get_exponents(degree, order, len(consts))
    rel = poly_check(consts, degree, min_order, exponents, test_prec, min_roi, verbose)
    if not rel:
        return []
    result, true_prec = rel.coeffs, rel.precision
    if verbose:
        with mp.workdps(5):
            print(f'Found relation with precision ratio {true_prec / test_prec}')
    # now must check subrelations! PSLQ is only guaranteed to return a small norm,
    # but not guaranteed to return a 1-dimensional relation, see for example pslq([1,2,3])
    res = compress_relation(result, consts, exponents, degree, order)
    return [res] if strict else check_subrelations(res, test_prec, min_roi, [], verbose)


# ===== COPYPASTED FROM mpmath\identification.py WITH MODIFICATIONS =====
# Intended to allow PSLQ to run with unbounded steps/maxcoeff

"""
Implements the PSLQ algorithm for integer relation detection,
and derivative algorithms for constant recognition.
"""
from mpmath.libmp.backend import xrange
from mpmath.libmp import int_types, sqrt_fixed

# round to nearest integer (can be done more elegantly...)
def round_fixed(x, prec):
    return ((x + (1<<(prec-1))) >> prec) << prec

class IdentificationMethods(object):
    pass

ctx = mp.mp
def pslq(x, tol=None, maxcoeff=1000, maxsteps=100, verbose=False):
    r"""
    Given a vector of real numbers `x = [x_0, x_1, ..., x_n]`, ``pslq(x)``
    uses the PSLQ algorithm to find a list of integers
    `[c_0, c_1, ..., c_n]` such that

    .. math ::

        |c_1 x_1 + c_2 x_2 + ... + c_n x_n| < \mathrm{tol}

    and such that `\max |c_k| < \mathrm{maxcoeff}`. If no such vector
    exists, :func:`~mpmath.pslq` returns ``None``. The tolerance defaults to
    3/4 of the working precision.

    **Examples**

    Find rational approximations for `\pi`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> pslq([-1, pi], tol=0.01)
        [22, 7]
        >>> pslq([-1, pi], tol=0.001)
        [355, 113]
        >>> mpf(22)/7; mpf(355)/113; +pi
        3.14285714285714
        3.14159292035398
        3.14159265358979

    Pi is not a rational number with denominator less than 1000::

        >>> pslq([-1, pi])
        >>>

    To within the standard precision, it can however be approximated
    by at least one rational number with denominator less than `10^{12}`::

        >>> p, q = pslq([-1, pi], maxcoeff=10**12)
        >>> print(p); print(q)
        238410049439
        75888275702
        >>> mpf(p)/q
        3.14159265358979

    The PSLQ algorithm can be applied to long vectors. For example,
    we can investigate the rational (in)dependence of integer square
    roots::

        >>> mp.dps = 30
        >>> pslq([sqrt(n) for n in range(2, 5+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 6+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 8+1)])
        [2, 0, 0, 0, 0, 0, -1]

    **Machin formulas**

    A famous formula for `\pi` is Machin's,

    .. math ::

        \frac{\pi}{4} = 4 \operatorname{acot} 5 - \operatorname{acot} 239

    There are actually infinitely many formulas of this type. Two
    others are

    .. math ::

        \frac{\pi}{4} = \operatorname{acot} 1

        \frac{\pi}{4} = 12 \operatorname{acot} 49 + 32 \operatorname{acot} 57
            + 5 \operatorname{acot} 239 + 12 \operatorname{acot} 110443

    We can easily verify the formulas using the PSLQ algorithm::

        >>> mp.dps = 30
        >>> pslq([pi/4, acot(1)])
        [1, -1]
        >>> pslq([pi/4, acot(5), acot(239)])
        [1, -4, 1]
        >>> pslq([pi/4, acot(49), acot(57), acot(239), acot(110443)])
        [1, -12, -32, 5, -12]

    We could try to generate a custom Machin-like formula by running
    the PSLQ algorithm with a few inverse cotangent values, for example
    acot(2), acot(3) ... acot(10). Unfortunately, there is a linear
    dependence among these values, resulting in only that dependence
    being detected, with a zero coefficient for `\pi`::

        >>> pslq([pi] + [acot(n) for n in range(2,11)])
        [0, 1, -1, 0, 0, 0, -1, 0, 0, 0]

    We get better luck by removing linearly dependent terms::

        >>> pslq([pi] + [acot(n) for n in range(2,11) if n not in (3, 5)])
        [1, -8, 0, 0, 4, 0, 0, 0]

    In other words, we found the following formula::

        >>> 8*acot(2) - 4*acot(7)
        3.14159265358979323846264338328
        >>> +pi
        3.14159265358979323846264338328

    **Algorithm**

    This is a fairly direct translation to Python of the pseudocode given by
    David Bailey, "The PSLQ Integer Relation Algorithm":
    http://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html

    The present implementation uses fixed-point instead of floating-point
    arithmetic, since this is significantly (about 7x) faster.
    """

    n = len(x)
    if n < 2:
        raise ValueError("n cannot be less than 2")

    # At too low precision, the algorithm becomes meaningless
    prec = ctx.prec
    if prec < 53:
        raise ValueError("prec cannot be less than 53")

    if verbose and prec // n < 5:
        print("Warning: precision for PSLQ may be too low")

    if tol is None:
        tol = ctx.mpf(2)**-int(prec*PRECISION_RATIO)
    else:
        tol = ctx.convert(tol)

    prec += 60

    if verbose:
        print("PSLQ using prec %i and tol %s" % (prec, ctx.nstr(tol)))

    tol = ctx.to_fixed(tol, prec)
    assert tol

    # Convert to fixed-point numbers. The dummy None is added so we can
    # use 1-based indexing. (This just allows us to be consistent with
    # Bailey's indexing. The algorithm is 100 lines long, so debugging
    # a single wrong index can be painful.)
    x = [None] + [ctx.to_fixed(ctx.mpf(xk), prec) for xk in x]

    # Sanity check on magnitudes
    minx = min(abs(xx) for xx in x[1:])
    if not minx:
        raise ValueError("PSLQ requires a vector of nonzero numbers")
    if minx < tol//100:
        if verbose:
            print("STOPPING: (one number is too small)")
        return None

    g = sqrt_fixed((4<<prec)//3, prec)
    # This matrix should be used to test whether precision is exhausted, but
    # this implementation doesn't do that! So we can just comment it out.
    #A = [[0]*(n+1) for i in xrange(n+1)]
    B = [[0]*(n+1) for i in xrange(n+1)] # redundant cells to allow 1-based indexing
    H = [[0]*(n+1) for i in xrange(n+1)]
    # Initialization
    # step 1
    temp = 1 << prec
    for i in xrange(1, n+1):
        B[i][i] = temp
    # step 2
    s = [0]*(n+1)
    for k in xrange(n, 0, -1):
        if k < n:
            s[k] = s[k+1]
        s[k] = s[k] + ((x[k]*x[k]) >> prec)
    for k in xrange(1, n+1):
        s[k] = sqrt_fixed(s[k], prec)
    t = s[1]
    y = x[:]
    for k in xrange(1, n+1):
        y[k] = (x[k] << prec) // t
        s[k] = (s[k] << prec) // t
    # step 3
    for i in xrange(1, n+1):
        if i <= n-1 and s[i]:
            H[i][i] = (s[i+1] << prec) // s[i]
        
        for j in xrange(1, i):
            sjj1 = s[j]*s[j+1]
            if sjj1:
                H[i][j] = ((-y[i]*y[j]) << prec) // sjj1
    # step 4
    for i in xrange(2, n+1):
        for j in xrange(i-1, 0, -1):
            #t = floor(H[i][j]/H[j][j] + 0.5)
            if H[j][j]:
                t = round_fixed((H[i][j] << prec)//H[j][j], prec)
            else:
                #t = 0
                continue
            y[j] = y[j] + (t*y[i] >> prec)
            for k in xrange(1, j+1):
                H[i][k] = H[i][k] - (t*H[j][k] >> prec)
            for k in xrange(1, n+1):
                #A[i][k] = A[i][k] - (t*A[j][k] >> prec)
                B[k][j] = B[k][j] + (t*B[k][i] >> prec)
    # Main algorithm
    global_best_err = mp.inf
    for REP in count():
        # Step 1
        m = -1
        szmax = -1
        for i in xrange(1, n):
            sz = (g**i * abs(H[i][i])) >> (prec*(i-1))
            if sz > szmax:
                m = i
                szmax = sz
        # Step 2
        y[m], y[m+1] = y[m+1], y[m]
        H[m], H[m+1] = H[m+1], H[m]
        #A[m], A[m+1] = A[m+1], A[m]
        for i in xrange(1,n+1): B[i][m], B[i][m+1] = B[i][m+1], B[i][m]
        # Step 3
        if m <= n - 2:
            t0 = sqrt_fixed((H[m][m]*H[m][m] + H[m][m+1]*H[m][m+1])>>prec, prec)
            # A zero element probably indicates that the precision has
            # been exhausted. XXX: this could be spurious, due to
            # using fixed-point arithmetic
            if not t0:
                break
            t1 = (H[m][m] << prec) // t0
            t2 = (H[m][m+1] << prec) // t0
            for i in xrange(m, n+1):
                t3 = H[i][m]
                t4 = H[i][m+1]
                H[i][m] = (t1*t3+t2*t4) >> prec
                H[i][m+1] = (-t2*t3+t1*t4) >> prec
        # Step 4
        for i in xrange(m+1, n+1):
            for j in xrange(min(i-1, m+1), 0, -1):
                try:
                    t = round_fixed((H[i][j] << prec)//H[j][j], prec)
                # Precision probably exhausted
                except ZeroDivisionError:
                    break
                y[j] = y[j] + ((t*y[i]) >> prec)
                for k in xrange(1, j+1):
                    H[i][k] = H[i][k] - (t*H[j][k] >> prec)
                for k in xrange(1, n+1):
                    #A[i][k] = A[i][k] - (t*A[j][k] >> prec)
                    B[k][j] = B[k][j] + (t*B[k][i] >> prec)
        # Until a relation is found, the error typically decreases
        # slowly (e.g. a factor 1-10) with each step TODO: we could
        # compare err from two successive iterations. If there is a
        # large drop (several orders of magnitude), that indicates a
        # "high quality" relation was detected. Reporting this to
        # the user somehow might be useful.
        best_err = mp.inf
        for i in xrange(1, n+1):
            err = abs(y[i])
            # Maybe we are done?
            if err < tol:
                # We are done if the coefficients are acceptable
                vec = [int(round_fixed(B[j][i], prec) >> prec) for j in \
                xrange(1,n+1)]
                if all(abs(v)<maxcoeff for v in vec):
                    if verbose:
                        print("FOUND relation at iter %i/%i, error: %s" % \
                            (REP, -1 if mp.isinf(maxsteps) else maxsteps, ctx.nstr(err / ctx.mpf(2)**prec, 1)))
                    return vec
            best_err = min(err, best_err)
        # test error: if much bigger than global_best_err, trigger failsafe
        if best_err < global_best_err:
            global_best_err = best_err
        elif best_err * best_err > (global_best_err << prec) and REP > 100: # initial anti-failsafe grace period
            if verbose:
                print("BAD ERROR FAILSAFE TRIGGERED")
            break
        # Calculate a lower bound for the norm. We could do this
        # more exactly (using the Euclidean norm) but there is probably
        # no practical benefit.
        recnorm = max(abs(h) for hh in H for h in hh)
        if recnorm:
            norm = (((1 << (2*prec)) // recnorm) >> prec) // 100
        else:
            norm = ctx.inf
        if verbose:
            print("%i/%i:  Error: %8s   Norm: %s" % \
                (REP, -1 if mp.isinf(maxsteps) else maxsteps, ctx.nstr(best_err / ctx.mpf(2)**prec, 1), norm))
        if norm >= maxcoeff or REP >= maxsteps:
            break
    if verbose:
        print("CANCELLING after step %i/%i." % (REP, -1 if mp.isinf(maxsteps) else maxsteps))
        print("Could not find an integer relation. Norm bound: %s" % norm)
    return None

def findpoly(x, n=1, **kwargs):
    r"""
    ``findpoly(x, n)`` returns the coefficients of an integer
    polynomial `P` of degree at most `n` such that `P(x) \approx 0`.
    If no polynomial having `x` as a root can be found,
    :func:`~mpmath.findpoly` returns ``None``.

    :func:`~mpmath.findpoly` works by successively calling :func:`~mpmath.pslq` with
    the vectors `[1, x]`, `[1, x, x^2]`, `[1, x, x^2, x^3]`, ...,
    `[1, x, x^2, .., x^n]` as input. Keyword arguments given to
    :func:`~mpmath.findpoly` are forwarded verbatim to :func:`~mpmath.pslq`. In
    particular, you can specify a tolerance for `P(x)` with ``tol``
    and a maximum permitted coefficient size with ``maxcoeff``.

    For large values of `n`, it is recommended to run :func:`~mpmath.findpoly`
    at high precision; preferably 50 digits or more.

    **Examples**

    By default (degree `n = 1`), :func:`~mpmath.findpoly` simply finds a linear
    polynomial with a rational root::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> findpoly(0.7)
        [-10, 7]

    The generated coefficient list is valid input to ``polyval`` and
    ``polyroots``::

        >>> nprint(polyval(findpoly(phi, 2), phi), 1)
        -2.0e-16
        >>> for r in polyroots(findpoly(phi, 2)):
        ...     print(r)
        ...
        -0.618033988749895
        1.61803398874989

    Numbers of the form `m + n \sqrt p` for integers `(m, n, p)` are
    solutions to quadratic equations. As we find here, `1+\sqrt 2`
    is a root of the polynomial `x^2 - 2x - 1`::

        >>> findpoly(1+sqrt(2), 2)
        [1, -2, -1]
        >>> findroot(lambda x: x**2 - 2*x - 1, 1)
        2.4142135623731

    Despite only containing square roots, the following number results
    in a polynomial of degree 4::

        >>> findpoly(sqrt(2)+sqrt(3), 4)
        [1, 0, -10, 0, 1]

    In fact, `x^4 - 10x^2 + 1` is the *minimal polynomial* of
    `r = \sqrt 2 + \sqrt 3`, meaning that a rational polynomial of
    lower degree having `r` as a root does not exist. Given sufficient
    precision, :func:`~mpmath.findpoly` will usually find the correct
    minimal polynomial of a given algebraic number.

    **Non-algebraic numbers**

    If :func:`~mpmath.findpoly` fails to find a polynomial with given
    coefficient size and tolerance constraints, that means no such
    polynomial exists.

    We can verify that `\pi` is not an algebraic number of degree 3 with
    coefficients less than 1000::

        >>> mp.dps = 15
        >>> findpoly(pi, 3)
        >>>

    It is always possible to find an algebraic approximation of a number
    using one (or several) of the following methods:

        1. Increasing the permitted degree
        2. Allowing larger coefficients
        3. Reducing the tolerance

    One example of each method is shown below::

        >>> mp.dps = 15
        >>> findpoly(pi, 4)
        [95, -545, 863, -183, -298]
        >>> findpoly(pi, 3, maxcoeff=10000)
        [836, -1734, -2658, -457]
        >>> findpoly(pi, 3, tol=1e-7)
        [-4, 22, -29, -2]

    It is unknown whether Euler's constant is transcendental (or even
    irrational). We can use :func:`~mpmath.findpoly` to check that if is
    an algebraic number, its minimal polynomial must have degree
    at least 7 and a coefficient of magnitude at least 1000000::

        >>> mp.dps = 200
        >>> findpoly(euler, 6, maxcoeff=10**6, tol=1e-100, maxsteps=1000)
        >>>

    Note that the high precision and strict tolerance is necessary
    for such high-degree runs, since otherwise unwanted low-accuracy
    approximations will be detected. It may also be necessary to set
    maxsteps high to prevent a premature exit (before the coefficient
    bound has been reached). Running with ``verbose=True`` to get an
    idea what is happening can be useful.
    """
    x = ctx.mpf(x)
    if n < 1:
        raise ValueError("n cannot be less than 1")
    if x == 0:
        return [1, 0]
    xs = [ctx.mpf(1)]
    for i in range(1,n+1):
        xs.append(x**i)
        a = pslq(xs, **kwargs)
        if a is not None:
            return a[::-1]

def fracgcd(p, q):
    x, y = p, q
    while y:
        x, y = y, x % y
    if x != 1:
        p //= x
        q //= x
    if q == 1:
        return p
    return p, q

def pslqstring(r, constants):
    q = r[0]
    r = r[1:]
    s = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p,q)
            cs = constants[i][1]
            if cs == '1':
                cs = ''
            else:
                cs = '*' + cs
            if isinstance(z, int_types):
                if z > 0: term = str(z) + cs
                else:     term = ("(%s)" % z) + cs
            else:
                term = ("(%s/%s)" % z) + cs
            s.append(term)
    s = ' + '.join(s)
    if '+' in s or '*' in s:
        s = '(' + s + ')'
    return s or '0'

def prodstring(r, constants):
    q = r[0]
    r = r[1:]
    num = []
    den = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p,q)
            cs = constants[i][1]
            if isinstance(z, int_types):
                if abs(z) == 1: t = cs
                else:           t = '%s**%s' % (cs, abs(z))
                ([num,den][z<0]).append(t)
            else:
                t = '%s**(%s/%s)' % (cs, abs(z[0]), z[1])
                ([num,den][z[0]<0]).append(t)
    num = '*'.join(num)
    den = '*'.join(den)
    if num and den: return "(%s)/(%s)" % (num, den)
    if num: return num
    if den: return "1/(%s)" % den

def quadraticstring(t,a,b,c):
    if c < 0:
        a,b,c = -a,-b,-c
    u1 = (-b+ctx.sqrt(b**2-4*a*c))/(2*c)
    u2 = (-b-ctx.sqrt(b**2-4*a*c))/(2*c)
    if abs(u1-t) < abs(u2-t):
        if b:  s = '((%s+sqrt(%s))/%s)' % (-b,b**2-4*a*c,2*c)
        else:  s = '(sqrt(%s)/%s)' % (-4*a*c,2*c)
    else:
        if b:  s = '((%s-sqrt(%s))/%s)' % (-b,b**2-4*a*c,2*c)
        else:  s = '(-sqrt(%s)/%s)' % (-4*a*c,2*c)
    return s

# Transformation y = f(x,c), with inverse function x = f(y,c)
# The third entry indicates whether the transformation is
# redundant when c = 1
transforms = [
  (lambda ctx,x,c: x*c, '$y/$c', 0),
  (lambda ctx,x,c: x/c, '$c*$y', 1),
  (lambda ctx,x,c: c/x, '$c/$y', 0),
  (lambda ctx,x,c: (x*c)**2, 'sqrt($y)/$c', 0),
  (lambda ctx,x,c: (x/c)**2, '$c*sqrt($y)', 1),
  (lambda ctx,x,c: (c/x)**2, '$c/sqrt($y)', 0),
  (lambda ctx,x,c: c*x**2, 'sqrt($y)/sqrt($c)', 1),
  (lambda ctx,x,c: x**2/c, 'sqrt($c)*sqrt($y)', 1),
  (lambda ctx,x,c: c/x**2, 'sqrt($c)/sqrt($y)', 1),
  (lambda ctx,x,c: ctx.sqrt(x*c), '$y**2/$c', 0),
  (lambda ctx,x,c: ctx.sqrt(x/c), '$c*$y**2', 1),
  (lambda ctx,x,c: ctx.sqrt(c/x), '$c/$y**2', 0),
  (lambda ctx,x,c: c*ctx.sqrt(x), '$y**2/$c**2', 1),
  (lambda ctx,x,c: ctx.sqrt(x)/c, '$c**2*$y**2', 1),
  (lambda ctx,x,c: c/ctx.sqrt(x), '$c**2/$y**2', 1),
  (lambda ctx,x,c: ctx.exp(x*c), 'log($y)/$c', 0),
  (lambda ctx,x,c: ctx.exp(x/c), '$c*log($y)', 1),
  (lambda ctx,x,c: ctx.exp(c/x), '$c/log($y)', 0),
  (lambda ctx,x,c: c*ctx.exp(x), 'log($y/$c)', 1),
  (lambda ctx,x,c: ctx.exp(x)/c, 'log($c*$y)', 1),
  (lambda ctx,x,c: c/ctx.exp(x), 'log($c/$y)', 0),
  (lambda ctx,x,c: ctx.ln(x*c), 'exp($y)/$c', 0),
  (lambda ctx,x,c: ctx.ln(x/c), '$c*exp($y)', 1),
  (lambda ctx,x,c: ctx.ln(c/x), '$c/exp($y)', 0),
  (lambda ctx,x,c: c*ctx.ln(x), 'exp($y/$c)', 1),
  (lambda ctx,x,c: ctx.ln(x)/c, 'exp($c*$y)', 1),
  (lambda ctx,x,c: c/ctx.ln(x), 'exp($c/$y)', 0),
]

def identify(x, constants=[], full=False, **kwargs):
    r"""
    Given a real number `x`, ``identify(x)`` attempts to find an exact
    formula for `x`. This formula is returned as a string. If no match
    is found, ``None`` is returned. With ``full=True``, a list of
    matching formulas is returned.

    As a simple example, :func:`~mpmath.identify` will find an algebraic
    formula for the golden ratio::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> identify(phi)
        '((1+sqrt(5))/2)'

    :func:`~mpmath.identify` can identify simple algebraic numbers and simple
    combinations of given base constants, as well as certain basic
    transformations thereof. More specifically, :func:`~mpmath.identify`
    looks for the following:

        1. Fractions
        2. Quadratic algebraic numbers
        3. Rational linear combinations of the base constants
        4. Any of the above after first transforming `x` into `f(x)` where
           `f(x)` is `1/x`, `\sqrt x`, `x^2`, `\log x` or `\exp x`, either
           directly or with `x` or `f(x)` multiplied or divided by one of
           the base constants
        5. Products of fractional powers of the base constants and
           small integers

    Base constants can be given as a list of strings representing mpmath
    expressions (:func:`~mpmath.identify` will ``eval`` the strings to numerical
    values and use the original strings for the output), or as a dict of
    formula:value pairs.

    In order not to produce spurious results, :func:`~mpmath.identify` should
    be used with high precision; preferably 50 digits or more.

    **Examples**

    Simple identifications can be performed safely at standard
    precision. Here the default recognition of rational, algebraic,
    and exp/log of algebraic numbers is demonstrated::

        >>> mp.dps = 15
        >>> identify(0.22222222222222222)
        '(2/9)'
        >>> identify(1.9662210973805663)
        'sqrt(((24+sqrt(48))/8))'
        >>> identify(4.1132503787829275)
        'exp((sqrt(8)/2))'
        >>> identify(0.881373587019543)
        'log(((2+sqrt(8))/2))'

    By default, :func:`~mpmath.identify` does not recognize `\pi`. At standard
    precision it finds a not too useful approximation. At slightly
    increased precision, this approximation is no longer accurate
    enough and :func:`~mpmath.identify` more correctly returns ``None``::

        >>> identify(pi)
        '(2**(176/117)*3**(20/117)*5**(35/39))/(7**(92/117))'
        >>> mp.dps = 30
        >>> identify(pi)
        >>>

    Numbers such as `\pi`, and simple combinations of user-defined
    constants, can be identified if they are provided explicitly::

        >>> identify(3*pi-2*e, ['pi', 'e'])
        '(3*pi + (-2)*e)'

    Here is an example using a dict of constants. Note that the
    constants need not be "atomic"; :func:`~mpmath.identify` can just
    as well express the given number in terms of expressions
    given by formulas::

        >>> identify(pi+e, {'a':pi+2, 'b':2*e})
        '((-2) + 1*a + (1/2)*b)'

    Next, we attempt some identifications with a set of base constants.
    It is necessary to increase the precision a bit.

        >>> mp.dps = 50
        >>> base = ['sqrt(2)','pi','log(2)']
        >>> identify(0.25, base)
        '(1/4)'
        >>> identify(3*pi + 2*sqrt(2) + 5*log(2)/7, base)
        '(2*sqrt(2) + 3*pi + (5/7)*log(2))'
        >>> identify(exp(pi+2), base)
        'exp((2 + 1*pi))'
        >>> identify(1/(3+sqrt(2)), base)
        '((3/7) + (-1/7)*sqrt(2))'
        >>> identify(sqrt(2)/(3*pi+4), base)
        'sqrt(2)/(4 + 3*pi)'
        >>> identify(5**(mpf(1)/3)*pi*log(2)**2, base)
        '5**(1/3)*pi*log(2)**2'

    An example of an erroneous solution being found when too low
    precision is used::

        >>> mp.dps = 15
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '((11/25) + (-158/75)*pi + (76/75)*e + (44/15)*sqrt(2))'
        >>> mp.dps = 50
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '1/(3*pi + (-4)*e + 2*sqrt(2))'

    **Finding approximate solutions**

    The tolerance ``tol`` defaults to 3/4 of the working precision.
    Lowering the tolerance is useful for finding approximate matches.
    We can for example try to generate approximations for pi::

        >>> mp.dps = 15
        >>> identify(pi, tol=1e-2)
        '(22/7)'
        >>> identify(pi, tol=1e-3)
        '(355/113)'
        >>> identify(pi, tol=1e-10)
        '(5**(339/269))/(2**(64/269)*3**(13/269)*7**(92/269))'

    With ``full=True``, and by supplying a few base constants,
    ``identify`` can generate almost endless lists of approximations
    for any number (the output below has been truncated to show only
    the first few)::

        >>> for p in identify(pi, ['e', 'catalan'], tol=1e-5, full=True):
        ...     print(p)
        ...  # doctest: +ELLIPSIS
        e/log((6 + (-4/3)*e))
        (3**3*5*e*catalan**2)/(2*7**2)
        sqrt(((-13) + 1*e + 22*catalan))
        log(((-6) + 24*e + 4*catalan)/e)
        exp(catalan*((-1/5) + (8/15)*e))
        catalan*(6 + (-6)*e + 15*catalan)
        sqrt((5 + 26*e + (-3)*catalan))/e
        e*sqrt(((-27) + 2*e + 25*catalan))
        log(((-1) + (-11)*e + 59*catalan))
        ((3/20) + (21/20)*e + (3/20)*catalan)
        ...

    The numerical values are roughly as close to `\pi` as permitted by the
    specified tolerance:

        >>> e/log(6-4*e/3)
        3.14157719846001
        >>> 135*e*catalan**2/98
        3.14166950419369
        >>> sqrt(e-13+22*catalan)
        3.14158000062992
        >>> log(24*e-6+4*catalan)-1
        3.14158791577159

    **Symbolic processing**

    The output formula can be evaluated as a Python expression.
    Note however that if fractions (like '2/3') are present in
    the formula, Python's :func:`~mpmath.eval()` may erroneously perform
    integer division. Note also that the output is not necessarily
    in the algebraically simplest form::

        >>> identify(sqrt(2))
        '(sqrt(8)/2)'

    As a solution to both problems, consider using SymPy's
    :func:`~mpmath.sympify` to convert the formula into a symbolic expression.
    SymPy can be used to pretty-print or further simplify the formula
    symbolically::

        >>> from sympy import sympify # doctest: +SKIP
        >>> sympify(identify(sqrt(2))) # doctest: +SKIP
        2**(1/2)

    Sometimes :func:`~mpmath.identify` can simplify an expression further than
    a symbolic algorithm::

        >>> from sympy import simplify # doctest: +SKIP
        >>> x = sympify('-1/(-3/2+(1/2)*5**(1/2))*(3/2-1/2*5**(1/2))**(1/2)') # doctest: +SKIP
        >>> x # doctest: +SKIP
        (3/2 - 5**(1/2)/2)**(-1/2)
        >>> x = simplify(x) # doctest: +SKIP
        >>> x # doctest: +SKIP
        2/(6 - 2*5**(1/2))**(1/2)
        >>> mp.dps = 30 # doctest: +SKIP
        >>> x = sympify(identify(x.evalf(30))) # doctest: +SKIP
        >>> x # doctest: +SKIP
        1/2 + 5**(1/2)/2

    (In fact, this functionality is available directly in SymPy as the
    function :func:`~mpmath.nsimplify`, which is essentially a wrapper for
    :func:`~mpmath.identify`.)

    **Miscellaneous issues and limitations**

    The input `x` must be a real number. All base constants must be
    positive real numbers and must not be rationals or rational linear
    combinations of each other.

    The worst-case computation time grows quickly with the number of
    base constants. Already with 3 or 4 base constants,
    :func:`~mpmath.identify` may require several seconds to finish. To search
    for relations among a large number of constants, you should
    consider using :func:`~mpmath.pslq` directly.

    The extended transformations are applied to x, not the constants
    separately. As a result, ``identify`` will for example be able to
    recognize ``exp(2*pi+3)`` with ``pi`` given as a base constant, but
    not ``2*exp(pi)+3``. It will be able to recognize the latter if
    ``exp(pi)`` is given explicitly as a base constant.

    """

    solutions = []
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    full = kwargs['full'] if 'full' in kwargs else False
    def addsolution(s):
        if verbose: print("Found: ", s)
        solutions.append(s)

    x = ctx.mpf(x)

    # Further along, x will be assumed positive
    if x == 0:
        if full: return ['0']
        else:    return '0'
    if x < 0:
        sol = identify(-x, constants, full, **kwargs)
        if sol is None:
            return sol
        if full:
            return ["-(%s)"%s for s in sol]
        else:
            return "-(%s)" % sol

    if 'tol' in kwargs and kwargs['tol']:
        kwargs['tol'] = ctx.mpf(kwargs['tol'])
    else:
        kwargs['tol'] = ctx.eps**0.7
    M = kwargs['maxcoeff'] if 'maxcoeff' in kwargs else 1000

    if constants:
        if isinstance(constants, dict):
            constants = [(ctx.mpf(v), name) for (name, v) in sorted(constants.items())]
        else:
            namespace = dict((name, getattr(ctx,name)) for name in dir(ctx))
            constants = [(eval(p, namespace), p) for p in constants]
    else:
        constants = []

    # We always want to find at least rational terms
    if 1 not in [value for (name, value) in constants]:
        constants = [(ctx.mpf(1), '1')] + constants

    # PSLQ with simple algebraic and functional transformations
    for ft, ftn, red in transforms:
        for c, cn in constants:
            if red and cn == '1':
                continue
            t = ft(ctx,x,c)
            # Prevent exponential transforms from wreaking havoc
            if abs(t) > M**2 or abs(t) < kwargs['tol']:
                continue
            # Linear combination of base constants
            r = pslq([t] + [a[0] for a in constants], **kwargs)
            s = None
            if r is not None and max(abs(uw) for uw in r) <= M and r[0]:
                s = pslqstring(r, constants)
            # Quadratic algebraic numbers
            else:
                q = pslq([ctx.one, t, t**2], **kwargs)
                if q is not None and len(q) == 3 and q[2]:
                    aa, bb, cc = q
                    if max(abs(aa),abs(bb),abs(cc)) <= M:
                        s = quadraticstring(t,aa,bb,cc)
            if s:
                if cn == '1' and ('/$c' in ftn):
                    s = ftn.replace('$y', s).replace('/$c', '')
                else:
                    s = ftn.replace('$y', s).replace('$c', cn)
                addsolution(s)
                if not full: return solutions[0]

            if verbose:
                print(".")

    # Check for a direct multiplicative formula
    if x != 1:
        # Allow fractional powers of fractions
        ilogs = [2,3,5,7]
        # Watch out for existing fractional powers of fractions
        logs = []
        for a, s in constants:
            if not sum(bool(findpoly(ctx.ln(a)/ctx.ln(i),1)) for i in ilogs):
                logs.append((ctx.ln(a), s))
        logs = [(ctx.ln(i),str(i)) for i in ilogs] + logs
        r = pslq([ctx.ln(x)] + [a[0] for a in logs], **kwargs)
        if r is not None and max(abs(uw) for uw in r) <= M and r[0]:
            addsolution(prodstring(r, logs))
            if not full: return solutions[0]

    if full:
        return sorted(solutions, key=len)
    else:
        return None

IdentificationMethods.pslq = pslq
IdentificationMethods.findpoly = findpoly
IdentificationMethods.identify = identify
