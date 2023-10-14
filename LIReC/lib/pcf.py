from __future__ import annotations
from enum import Enum
from functools import reduce
import gmpy2
from gmpy2 import mpz, xmpz, mpq
import mpmath as mp
from sympy import Poly, gcd as sgcd, cancel
from sympy.abc import n
from time import time
from typing import List, Tuple, Callable
from LIReC.lib.pslq_utils import poly_check, PreciseConstant, MIN_PSLQ_DPS
CanonicalForm = Tuple[List[int], List[int]]

def _poly_eval(poly: List, n):
    # current fastest method, poly must be coefficients in increasing order of exponent
    # P.S.: if you're curious and don't feel like looking it up, the difference between
    # mpz and xmpz is that xmpz is mutable, so in-place operations are faster
    c = xmpz(1)
    res = xmpz(0)
    for coeff in poly:
        res += coeff * c
        c *= n
    return mpz(res)

CALC_JUMP = 256
REDUCE_JUMP = 128
LOG_CALC_JUMP = 7
LOG_REDUCE_JUMP = 6
FR_THRESHOLD = 0.1
MAX_PREC = mpz(99999)
    
class IllegalPCFException(Exception):
    pass

class NoFRException(Exception):
    pass

class ContinuedFraction:

    class Util:
    
        @staticmethod
        def mult(A: List[List], B: List[List]):
            # yes it's faster to manually multiply than use numpy for instance!
            return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]
        
        @staticmethod
        def div_mat(mat, x) -> List[List]:
            return [[mat[0][0] // x, mat[0][1] // x],[mat[1][0] // x, mat[1][1] // x]]
        
        @staticmethod
        def as_mpf(q: mpq) -> mp.mpf:
            return mp.mpf(q.numerator) / mp.mpf(q.denominator)
        
        @staticmethod
        def combine(self: ContinuedFraction, mats, force: bool = False):
            # technically it's not necessary to return mats since everything is in-place
            # operations, but just in case one day a non-in-place operation is added...
            orig_len = len(mats)
            while len(mats) > 1 and (force or mats[-1][1] >= mats[-2][1]):
                mat1 = mats.pop()
                mat2 = mats.pop()
                mats += [(ContinuedFraction.Util.mult(mat1[0], mat2[0]), mat1[1] + mat2[1])]
            return self._end_combine(mats, orig_len, force)
    
    a: Callable[[int], Any]
    b: Callable[[int], Any]
    mat: List[List]
    depth: int
    true_value: mpq or None
    
    def __init__(self: ContinuedFraction, a: Callable[[int], Any], b: Callable[[int], Any], mat: List[int] or None = None, depth: int = 0):
        self.a_func = a
        self.b_func = b
        self.mat = [mat[0:2], mat[2:4]] if mat else [[self.a_func(0), 1], [1, 0]]
        self.depth = depth
        self.true_value = None
        self.eval_defaults = {
            'depth': 2 ** 13, # at this depth, calculation of one PCF is expected to take about 3 seconds, depending on your machine
            'precision': 50,
            'timeout_sec': 0,
            'timeout_check_freq': 1024,
            'no_exception': False
        }
    
    @property
    def value(self: ContinuedFraction) -> mpq:
        return mpq(self.mat[0][0], self.mat[0][1])
    
    @property
    def precision(self: ContinuedFraction) -> gmpy2.mpfr:
        to_compare = self.true_value if self.true_value != None else mpq(self.mat[1][0], self.mat[1][1])
        return gmpy2.floor(-gmpy2.log10(abs(self.value - to_compare))) if all([self.mat[0][1], self.mat[1][1]]) else MAX_PREC
    
    def _pre_eval(self):
        pass
    
    def _end_combine(self, mats, orig_len, force=False):
        return mats, None
    
    def _end_depth(self, extras_list, kwargs):
        return kwargs
    
    def _end_eval(self, value, prec, extras_list, kwargs):
        return self
    
    def eval(self: ContinuedFraction, **kwargs) -> ContinuedFraction or Exception:
        '''
        Approximate the value of this PCF. Will first calculate to an initial depth,
        and then will procedurally double the depth until the desired precision is obtained.
        
        Accepted kwargs: (others will be ignored)
            'depth': Minimal depth to calculate to, defaults to 8192
            'precision': Minimal precision to obtain, defaults to 50
            'force_fr': Ensure the result has FR if possible (AKA keep going if INDETERMINATE_FR), defaults to True
            'timeout_sec': If nonzero, halt calculation after this many seconds and return whatever you got, no matter what. Defaults to 0
            'timeout_check_freq': Only check for timeout every this many iterations. Defaults to 1024
            'no_exception': Return exceptions (or ContinuedFraction.Result if possible) instead of raising them (see below). Defaults to False
        
        Exceptions:
            NoFRException: The PCF doesn't converge.
            IllegalPCFException: The PCF has natural roots.
        '''
        # P.S.: The code here is in fact similar to enumerators.FREnumerator.check_for_fr, but
        # the analysis we're doing here is both more delicate (as we allow numerically-indeterminate PCFs),
        # and also less redundant (since we also want the value of the PCF instead of just discarding it for instance)
        self._pre_eval()
        mp.mp.dps = 100 # temporarily, to let the precision calculation work, will probably be increased later
        kwargs = {**self.eval_defaults, **kwargs}
        extras_list = []
        start = time()
        mats = [(self.mat, self.depth)]
        while self.depth < kwargs['depth']:
            self.depth += 1
            mats, extra = ContinuedFraction.Util.combine(self, mats + [([[self.a_func(self.depth), self.b_func(self.depth)], [1, 0]], 1)])
            if extra:
                extras_list += [extra]
            if kwargs['timeout_sec'] and self.depth % kwargs['timeout_check_freq'] == 0 and time() - start > kwargs['timeout_sec']:
                break
            if self.depth == kwargs['depth']:
                self.mat = ContinuedFraction.Util.combine(self, mats, True)[0][0][0]
                
                prec = self.precision # check precision
                if prec.is_infinite():
                    ex = IllegalPCFException('continuant denominator zero')
                    #if kwargs['no_exception']:
                    #    return ex
                    raise ex
                if prec < kwargs['precision']:
                    kwargs['depth'] *= 2
                    continue
                
                kwargs = self._end_depth(extras_list, kwargs)
                if isinstance(kwargs, Exception):
                    #if kwargs['no_exception']:
                    #    return ex
                    raise ex
        
        self.mat = ContinuedFraction.Util.combine(self, mats, True)[0][0][0]
        mp.mp.dps = max(100, self.precision)
        value = ContinuedFraction.Util.as_mpf(self.value)
        if mp.almosteq(0, value):
            self.true_value = 0
        
        prec = self.precision
        if prec.is_infinite():
            ex = IllegalPCFException('continuant denominator zero')
            if not kwargs['no_exception']:
                raise ex
        
        rational, _ = poly_check([PreciseConstant(value, prec)], 1, 1, test_prec = MIN_PSLQ_DPS)
        if rational:
            self.true_value = mpq(rational[0], -rational[1])
        
        return self._end_eval(value, prec, extras_list, kwargs)


class PCF(ContinuedFraction):
    '''
    a polynomial continued fraction, represented by two Polys a, b:
    a0 + b1 / (a1 + b2 / (a2 + b3 / (...)))
    yes, this is the reverse of wikipedia's convention (i.e. https://en.wikipedia.org/wiki/Generalized_continued_fraction)
    '''

    class Convergence(Enum):
        ZERO_DENOM = 0 # now considered an illegal PCF
        NO_FR = 1 # now considered an illegal PCF
        INDETERMINATE_FR = 2
        FR = 3
        RATIONAL = 4
    
    a: Poly
    b: Poly
    
    def _pre_eval(self):
        self.a_coeffs = [mpz(x) for x in reversed(self.a.all_coeffs())]
        self.b_coeffs = [mpz(x) for x in reversed(self.b.all_coeffs())]
    
    def _end_depth(self, extras_list, kwargs):
        if kwargs['force_fr']: # check convergence
            convergence = self.check_convergence(extras_list)
            if convergence == PCF.Convergence.NO_FR:
                return NoFRException()
            if convergence == PCF.Convergence.INDETERMINATE_FR:
                kwargs['depth'] *= 2
        return kwargs
    
    def _end_eval(self, value, prec, extras_list, kwargs):
        self.convergence = self.check_convergence(extras_list)
        if self.convergence == PCF.Convergence.NO_FR and not kwargs['no_exception'] and kwargs['force_fr']:
            raise NoFRException()
        
        return self

    def _end_combine(self, mats, orig_len, force=False):
        if force or orig_len - len(mats) > LOG_REDUCE_JUMP:
            gcd = gmpy2.gcd(*[x for row in mats[-1][0] for x in row])
            mats[-1] = (ContinuedFraction.Util.div_mat(mats[-1][0], gcd), mats[-1][1])
        if force or orig_len - len(mats) > LOG_CALC_JUMP:
            return mats, gmpy2.log(gmpy2.gcd(*mats[-1][0][0])) / self.depth + (len(self.a_coeffs) - 1) * (1 - gmpy2.log(self.depth))
        return mats, None
    
    def __init__(self: PCF, a: Poly or List[int], b: Poly or List[int], mat: List[int] or None = None, depth: int = 0, auto_deflate: bool = True) -> None:
        '''
        a_coeffs, b_coeffs: lists of integers from the largest power to the smallest power.
        '''
        self.a = a if isinstance(a, Poly) else Poly(a, n)
        self.b = b if isinstance(b, Poly) else Poly(b, n)
        if auto_deflate:
            self.deflate()
        self._pre_eval()
        super().__init__(lambda n: _poly_eval(self.a_coeffs, n), lambda n: _poly_eval(self.b_coeffs, n), mat, depth)
        self.eval_defaults['force_fr'] = True

    def moving_canonical_form(self: PCF) -> Tuple(Poly, Poly):
        # Should always be real roots. (TODO modify if not!)
        roots = [r for r in self.b.all_roots() if r.is_real]

        # In case b is a constant (has no roots) we still want an to have the canonical roots
        # => (the smallest root to be in (-1,0] )
        # If some of the roots are irrational, it makes the coefficients look ugly, so I decided not to move them.
        # ground_roots is a dict {root:power_of_root} while real_roots is a list of all of the roots including multiplicity
        if len(roots) == 0:
            roots = self.a.real_roots()
            if len(roots) == 0 or len(roots) != sum(self.a.ground_roots().values()):
                return self.a, self.b

        largest_root = max(roots)
        # We want the largest root to be in (-1,0].
        return self.b.compose(Poly(n + largest_root)), self.a.compose(Poly(n + largest_root))

    def inflating_canonical_form(self: PCF) -> Tuple(Poly, Poly):
        top = self.b
        bot = self.a * self.a.compose(Poly(n - 1))
        gcd = sgcd(top, bot)
        return Poly(cancel(top / gcd), n), Poly(cancel(bot / gcd), n)

    def get_canonical_form(self: PCF) -> Tuple(Poly, Poly):
        top, bot = self.inflating_canonical_form()
        return PCF(bot.all_coeffs(), top.all_coeffs()).moving_canonical_form()

    def get_canonical_form_string(self: PCF) -> str:
        a, b = self.get_canonical_form()
        return str(b / a)

    def __str__(self: PCF) -> str:
        return f'a: {self.a.all_coeffs()}\t|\tb: {self.b.all_coeffs()}'

    def is_inflation(self: PCF) -> bool:
        return sgcd(self.b, self.a * self.a.compose(Poly(n - 1))) != 1

    def deflate(self: PCF) -> None:
        deflated: bool = True
        while deflated: # keep going so long as something cancels out
            deflated = False
            a_factors = [factor_tuple[0] for factor_tuple in self.a.factor_list()[1]]
            b_factors = [factor_tuple[0] for factor_tuple in self.b.factor_list()[1]]
            for factor in a_factors:
                if factor in b_factors and factor.compose(Poly(n-1)) in b_factors:
                    self.a = Poly(cancel(self.a / factor), n) # n must stay in the constructor because these polynomials can end up being constant!
                    self.b = Poly(cancel(self.b / (factor * factor.compose(Poly(n - 1)))), n)
                    deflated = True
    
    @staticmethod
    def from_canonical_form(canonical_form: CanonicalForm) -> PCF:
        '''
        Receive the canonical form of a pcf (an := 1 ; bn := bn / (an*a(n+1)))
        and return a pcf of this canonical form.
        Notice there may be many pcfs that fit the same canonical form, this returns just one of them.
        TODO: add link to the doc which explains this
        '''
        a = Poly(canonical_form[1], n).compose(Poly(n + 1))
        b = Poly(canonical_form[0], n) * a
        return PCF(a.all_coeffs(), b.all_coeffs())
    
    def check_convergence(self: ContinuedFraction, fr_list) -> PCF.Convergence:
        if self.true_value != None:
            return PCF.Convergence.RATIONAL
        
        if any(abs(fr_list[i + 1] - fr_list[i]) < FR_THRESHOLD for i in range(len(fr_list) - 1)):
            return PCF.Convergence.FR
        
        if any(abs(fr_list[i + 1] - fr_list[i + 2]) > abs(fr_list[i] - fr_list[i + 1]) for i in range(len(fr_list) - 2)):
            return PCF.Convergence.NO_FR
        
        return PCF.Convergence.INDETERMINATE_FR
