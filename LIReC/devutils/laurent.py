from LIReC.db.access import db
from LIReC.db.models import *
from LIReC.lib.pcf import *
from LIReC.devutils.compare_pcfs import get_pcf_params

from sympy import limit_seq, simplify
# if we wanted we could speed this up by not using sympy and explicitly calculating everything! this is all rational functions after all
def laurent(ex, terms=2):
    res, p = 0, 0
    while limit_seq(ex / n**p).is_infinite:
        p += 1
    while terms:
        extra = limit_seq(ex / n**p) * n**p
        if extra:
            terms -= 1
            res += extra
            ex = simplify(ex - extra)
        if not ex:
            return str(res)
        p -= 1
    p += 1
    order = f'1/n**{-p}' if p<-1 else str(n**p)
    return f'{res} + o({order})'

# implement this in PCF class later?
def speed(self: PCF, terms=2):
    P, Q = self.canonical_form()
    return laurent(1 + 4*P/Q, terms)
