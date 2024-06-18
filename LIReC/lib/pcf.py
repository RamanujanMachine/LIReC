from __future__ import annotations
from collections import Counter
from enum import Enum
from functools import reduce
import gmpy2
from gmpy2 import mpz, mpfr, xmpz, mpq
import mpmath as mp
from operator import floordiv
from sympy import Poly, gcd, cancel, re, floor, factorint, RootOf, limit_seq
from sympy.abc import n
from time import time
from typing import List, Tuple, Callable
from LIReC.lib.pslq_utils import poly_check, PreciseConstant, MIN_PSLQ_DPS
CanonicalForm = Tuple[List[int], List[int]]

def _poly_eval(poly: List, n):
    # current fastest method, poly must be coefficients in decreasing order of exponent
    # P.S.: if you're curious and don't feel like looking it up, the difference between
    # mpz and xmpz is that xmpz is mutable, so in-place operations are faster
    res = xmpz(0)
    for coeff in poly:
        res = res * n + coeff
    return mpz(res)

def _floor_roots(poly: Poly):
    roots = poly.all_roots()
    for r in roots:
        if isinstance(r, RootOf):
            r = r.n(5) # basic eval attempt
            split = str(r).split('e+')
            if len(split) > 1:
                r = r.n(int(split[1]) + 3) # evaluate enough digits
        yield floor(re(r))

def _laurent(ex, terms=2):
    res, p = [], 2 # don't care for laurents that are too big
    while terms:
        coeff = limit_seq(ex / n**p)
        if coeff:
            if coeff.is_infinite:
                res += [[1, p + 1]]
                break
            terms -= 1
            res += [[coeff, p]]
            ex = cancel(ex - coeff * n**p) # slightly more efficient than simplify
        if not ex:
            break
        p -= 1
    return res

FR_THRESHOLD = 0.1
MAX_PREC = mpfr(99999)
    
class IllegalPCFException(Exception):
    pass

class NoFRException(Exception):
    pass

class GCF:

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
        def combine(self: GCF, mats, kwargs, force: bool = False):
            # technically it's not necessary to return mats since everything is in-place
            # operations, but just in case one day a non-in-place operation is added...
            orig_len = len(mats)
            while len(mats) > 1 and (force or mats[-1][1] >= mats[-2][1]):
                mat2 = mats.pop()
                mat1 = mats.pop()
                mats += [(GCF.Util.mult(mat1[0], mat2[0]), mat1[1] + mat2[1])]
            return self._end_combine(mats, orig_len, kwargs, force)
    
    a: Callable[[int], Any]
    b: Callable[[int], Any]
    mat: List[List]
    depth: int
    true_value: mpq or None
    
    def __init__(self: GCF, a: Callable[[int], Any], b: Callable[[int], Any], mat: List[int] or None = None, init_depth: int = 0, **kwargs):
        self.a_func = a
        self.b_func = b
        # this choice of initial matrix will compute the value of the semi-canonical form instead! intentional!
        a0 = self.a_func(0)
        self.mat = [mat[0:2], mat[2:4]] if mat else [[1, a0], [0, a0]]
        self.depth = init_depth
        self.true_value = None
        self.eval_defaults = {
            'depth': 2 ** 10,
            'precision': -gmpy2.inf(),
            'timeout_sec': 0,
            'timeout_check_freq': 2 ** 10,
            'no_exception': False,
            'rational_test': True,
            'max_depth': mp.inf
        }
        self.eval_defaults = {**self.eval_defaults, **kwargs}
    
    @property
    def value_rational_unreduced(self: GCF) -> Tuple(mpz, mpz):
        return self.mat[0][1], self.mat[1][1] # mpq automatically reduces gcd, sometimes don't want that
    
    @property
    def value_rational(self: GCF) -> mpq:
        return mpq(*self.value_rational_unreduced)
    
    @property
    def value(self: GCF) -> mp.mpf:
        return mp.mpf(self.value_rational.numerator) / mp.mpf(self.value_rational.denominator)
    
    @property
    def precision(self: GCF) -> gmpy2.mpfr:
        to_compare = self.true_value
        if not to_compare:
            try:
                to_compare = mpq(self.mat[0][0], self.mat[1][0])
            except:
                to_compare = self.mat[0][0] / self.mat[1][0]
        
        return gmpy2.floor(-gmpy2.log10(abs(self.value_rational - to_compare))) if all(self.mat[1]) else MAX_PREC
    
    def _pre_eval(self):
        pass
    
    def _end_depth(self, extras_list, kwargs):
        return kwargs
    
    def _end_eval(self, val, prec, extras_list, kwargs):
        return self
    
    def _end_combine(self, mats, orig_len, kwargs, force=False):
        return mats, None
    
    def eval(self: GCF, **kwargs) -> GCF or Exception:
        '''
        Approximate the value of this PCF. Will first calculate to an initial depth,
        and then will procedurally double the depth until the desired precision is obtained.
        
        Accepted kwargs: (others will be ignored)
            'depth': Minimal depth to calculate to, defaults to 8192
            'precision': Minimal precision to obtain, defaults to 50
            'force_fr': Ensure the result has FR if possible (AKA keep going if INDETERMINATE_FR), defaults to True
            'timeout_sec': If nonzero, halt calculation after this many seconds and return whatever you got, no matter what. Defaults to 0
            'timeout_check_freq': Only check for timeout every this many iterations. Defaults to 1024
            'no_exception': Return exceptions (or GCF.Result if possible) instead of raising them (see below). Defaults to False
            'rational_test': Whether or not to attempt a PSLQ run to identify the limit value as a rational value. Defaults to True.
        
        Exceptions:
            NoFRException: The PCF doesn't converge.
            IllegalPCFException: The PCF has natural roots.
        '''
        # P.S.: The code here is in fact similar to enumerators.FREnumerator.check_for_fr, but
        # the analysis we're doing here is both more delicate (as we allow numerically-indeterminate PCFs),
        # and also less redundant (since we also want the value of the PCF instead of just discarding it for instance)
        self.true_value = None
        self._pre_eval()
        mp.mp.dps = 100 # temporarily, to let the precision calculation work, will probably be increased later
        kwargs = {**self.eval_defaults, **kwargs}
        extras_list = []
        start = time()
        mats = [(self.mat, self.depth)]
        while self.depth < kwargs['depth']:
            self.depth += 1
            mats, extra = GCF.Util.combine(self, mats + [([[0, self.b_func(self.depth)], [1, self.a_func(self.depth)]], 1)], kwargs)
            if extra:
                extras_list += [extra]
            if kwargs['timeout_sec'] and self.depth % kwargs['timeout_check_freq'] == 0 and time() - start > kwargs['timeout_sec']:
                break
            if self.depth == kwargs['depth']:
                self.mat = GCF.Util.combine(self, mats, kwargs, True)[0][0][0]
                
                prec = self.precision # check precision # TODO add check for negative precision?
                if prec.is_infinite():
                    ex = IllegalPCFException('continuant denominator zero')
                    #if kwargs['no_exception']:
                    #    return ex
                    raise ex
                if prec < kwargs['precision'] and kwargs['depth'] < kwargs['max_depth']:
                    kwargs['depth'] = min(2 * kwargs['depth'], kwargs['max_depth'])
                    continue
                
                kwargs = self._end_depth(extras_list, kwargs)
                if isinstance(kwargs, Exception):
                    #if kwargs['no_exception']:
                    #    return kwargs
                    raise kwargs
        
        self.mat = GCF.Util.combine(self, mats, kwargs, True)[0][0][0]
        mp.mp.dps = max(100, self.precision)
        val = self.value
        if mp.almosteq(0, val):
            self.true_value = 0
        
        prec = self.precision
        if prec.is_infinite():
            ex = IllegalPCFException('continuant denominator zero')
            if not kwargs['no_exception']:
                raise ex
        
        if prec > 0 and kwargs['rational_test']:
            rational = poly_check([PreciseConstant(val, prec)], 1, 1, test_prec = MIN_PSLQ_DPS)
            if rational:
                self.true_value = mpq(rational.coeffs[0], -rational.coeffs[1])
        
        return self._end_eval(val, prec, extras_list, kwargs)


class PCF(GCF):
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
        self.a_coeffs = [mpz(x) for x in self.a.all_coeffs()]
        self.b_coeffs = [mpz(x) for x in self.b.all_coeffs()]
    
    def _end_depth(self, extras_list, kwargs):
        if kwargs['force_fr']: # check convergence
            convergence = self.check_convergence(extras_list)
            if convergence == PCF.Convergence.NO_FR:
                return NoFRException()
            if convergence == PCF.Convergence.INDETERMINATE_FR:
                kwargs['depth'] *= 2
        return kwargs
    
    def _end_eval(self, val, prec, extras_list, kwargs):
        self.convergence = self.check_convergence(extras_list)
        if self.convergence == PCF.Convergence.NO_FR and not kwargs['no_exception'] and kwargs['force_fr']:
            raise NoFRException()
        
        return self

    def _end_combine(self, mats, orig_len, kwargs, force=False):
        log_reduce_jump = kwargs['log_reduce_jump']
        if log_reduce_jump and (force or orig_len - len(mats) > log_reduce_jump):
            g = gmpy2.gcd(*[x for row in mats[-1][0] for x in row])
            mats[-1] = (GCF.Util.div_mat(mats[-1][0], g), mats[-1][1])
        
        log_calc_jump = kwargs['log_calc_jump']
        if log_calc_jump and (force or orig_len - len(mats) > log_calc_jump):
            return mats, gmpy2.log(gmpy2.gcd(*mats[-1][0][0])) / self.depth + (len(self.a_coeffs) - 1) * (1 - gmpy2.log(self.depth))
        return mats, None
    
    def __init__(self: PCF, a: Poly or List[int], b: Poly or List[int], mat: List[int] or None = None, init_depth: int = 0, auto_deflate: bool = True, **kwargs) -> None:
        '''
        a_coeffs, b_coeffs: lists of integers from the largest power to the smallest power.
        '''
        self.a = Poly(a, n)
        self.b = Poly(b, n)
        if self.a == 0 or self.b == 0:
            raise Exception('neither polynomial can be 0')
        if auto_deflate:
            self.deflate()
        # canonize!
        self.a, self.b = PCF.canonical_form_to_a_b(self.canonical_form())
        self._pre_eval()
        super().__init__(lambda n: _poly_eval(self.a_coeffs, n), lambda n: _poly_eval(self.b_coeffs, n), mat, init_depth, **kwargs)
        self.eval_defaults['force_fr'] = kwargs.get('force_fr', False)
        self.eval_defaults['log_calc_jump'] = kwargs.get('log_calc_jump', 7)
        self.eval_defaults['log_reduce_jump'] = kwargs.get('log_reduce_jump', 6)

    def semi_canonical_form(self: PCF) -> Tuple(Poly, Poly):
        top = self.b # inflate everything by 1/an, so partial denominators become constant 1
        bot = self.a * self.a.compose(Poly(n - 1))
        g = gcd(top, bot)
        return Poly(cancel(top / g), n), Poly(cancel(bot / g), n)

    def canonical_form(self: PCF) -> Tuple(Poly, Poly):
        top, bot = self.semi_canonical_form() # start with the semi-canonical form (partial denominator series is constant 1)
        
        # the largest real part of the roots of top should be in (-1,0].
        # If top is constant, use the roots of bot instead
        roots = list(_floor_roots(top)) + list(_floor_roots(bot))

        # If both are constants, just leave them as is
        if not roots:
            return top, bot
        
        # after the shift, real part of every root must be less than 1, with at least one in the range [0,1)
        largest_root = max(roots)
        return top.compose(Poly(n + largest_root)), bot.compose(Poly(n + largest_root))

    def canonical_form_string(self: PCF) -> str:
        top, bot = self.canonical_form()
        return str(top / bot)

    def __str__(self: PCF) -> str:
        return f'PCF[{self.a.expr}, {self.b.expr}]'

    def is_inflation(self: PCF) -> bool:
        return gcd(self.b, self.a * self.a.compose(Poly(n - 1))) != 1

    def deflate(self: PCF) -> None:
        def better_factor_list(p: Poly):
            factors = p.factor_list()
            return Counter(factorint(factors[0])), Counter({k:v for k,v in factors[1]})
        
        a_scalars, a_subpolys = better_factor_list(self.a)
        b_scalars, b_subpolys = better_factor_list(self.b)
        # first reduce common scalars
        common = {k:min(a_scalars[k],b_scalars[k]//2) for k in a_scalars}
        self.a = reduce(floordiv, [k ** common[k] for k in common], self.a)
        self.b = reduce(floordiv, [k ** (2 * common[k]) for k in common], self.b)
        # then reduce polynomial factors, must do iteratively instead of with comprehension!
        changed = True
        while changed:
            changed = False
            for k in a_subpolys:
                exp = min(a_subpolys[k],b_subpolys[k],b_subpolys[k.compose(Poly(n - 1))])
                if exp > 0:
                    self.a //= k ** exp
                    self.b //= (k * k.compose(Poly(n-1))) ** exp
                    _, a_subpolys = better_factor_list(self.a)
                    _, b_subpolys = better_factor_list(self.b)
                    changed = True
                    break
    
    @staticmethod
    def canonical_form_to_a_b(canonical_form: CanonicalForm):
        top, bot = canonical_form
        a = Poly(bot, n).compose(Poly(n + 1)) # inflate everything by bot(n+1)
        b = Poly(top, n) * a # then what remains of top/bot is top/bot * bot*bot(n+1), or just top*bot(n+1)
        return a, b # the end result is PCF[bot(n+1),top*bot(n+1)], from here deflate and we're done!
    
    @staticmethod
    def from_canonical_form(canonical_form: CanonicalForm) -> PCF:
        '''
        Receive the canonical form of a pcf (an := 1 ; bn := top/bot)
        and return a pcf of this canonical form.
        Notice there may be many pcfs that fit the same canonical form, this returns just one of them.
        TODO: add link to the doc which explains this
        '''
        return PCF(*PCF.canonical_form_to_a_b(canonical_form))
    
    def check_convergence(self: GCF, fr_list) -> PCF.Convergence:
        if self.true_value != None:
            return PCF.Convergence.RATIONAL
        
        if any(abs(fr_list[i + 1] - fr_list[i]) < FR_THRESHOLD for i in range(len(fr_list) - 1)):
            return PCF.Convergence.FR
        
        if any(abs(fr_list[i + 1] - fr_list[i + 2]) > abs(fr_list[i] - fr_list[i + 1]) for i in range(len(fr_list) - 2)):
            return PCF.Convergence.NO_FR
        
        return PCF.Convergence.INDETERMINATE_FR
    
    def predict_error(self: PCF, depth: int) -> mp.mpf:
        top, bot = self.canonical_form()
        laurent = _laurent(1+4*top/bot)
        if laurent[0] == [1, 0]: # factorial convergence
            return mp.factorial(depth) ** laurent[1][1]
        if laurent[0][1] == 0: # exponential convergence
            sqrtC = mp.sqrt(laurent[0][0])
            return abs((1+sqrtC)/(1-sqrtC)) ** -depth
        if laurent[0][1] == 1: # esqrt convergence
            return mp.exp(-4 * mp.sqrt(depth / laurent[0][0]))
        return None # no known convergence formula here!
    
    def predict_precision(self: PCF, depth: int) -> mp.mpf:
        return mp.floor(-mp.log10(self.predict_error(depth)))
    
    def predict_depth(self: PCF, precision: mp.mpf, is_error: bool = False) -> mp.mpf:
        error = mp.mpf(precision)
        if not is_error:
            error = 10 ** -error
        
        top, bot = self.canonical_form()
        laurent = _laurent(1+4*top/bot)
        if laurent[0] == [1, 0]: # factorial convergence
            # wikipedia's asymptotic formula for inverse factrorial, usually slightly pessimistic in practice so it's good enough
            temp = mp.ln(mp.root(error, laurent[1][1]) / mp.sqrt(2 * mp.pi))
            return mp.ceil(mp.mpf(1) / 2 + temp / mp.lambertw(mp.exp(-1) * temp))
        if laurent[0][1] == 0: # exponential convergence
            sqrtC = mp.sqrt(laurent[0][0])
            return mp.ceil(-mp.ln(error) / mp.ln(abs((1+sqrtC)/(1-sqrtC))))
        if laurent[0][1] == 1: # esqrt convergence
            return mp.ceil(mp.ln(error) ** 2 * laurent[0][0] / 16)
        return None # no known convergence formula here!
