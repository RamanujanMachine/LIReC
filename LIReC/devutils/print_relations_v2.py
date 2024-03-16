# https://jeltef.github.io/PyLaTeX/current/pylatex/pylatex.section.html?highlight=section#module-pylatex.section
# https://stackoverflow.com/questions/66779798/latex-pylatex-math-equation-i-want-to-add-equations-to-a-table-using-python-eg
# https://en.wikibooks.org/wiki/LaTeX/Mathematics#Introducing_dots_in_formulas

from functools import reduce
from pylatex import Document, Section, NoEscape, Package
from operator import mul
import os
from LIReC.db.access import db
from LIReC.db.models import *
from LIReC.lib.pcf import *
from LIReC.lib.pslq_utils import *

# need to keep hold of the original db constant, but pslq_utils doesn't care for that so this is separate here
class ExtendedConstant(PreciseConstant):
    orig: Constant
    ext: None or PcfCanonicalConstant or NamedConstant # or anything else "inheriting" from Constant
    related: List[NamedConstant]
    
    def __init__(self, value, precision, orig, related=[], symbol=None):
        self.orig = orig
        self.related = related
        super().__init__(value, precision, symbol)
    
    @staticmethod
    def from_db(const: Constant, symbol = None):
        return ExtendedConstant(const.value, const.precision, const, Symbol(symbol if symbol!=None else f'c_{const.const_id}'))

class Explained:
    pcf: PcfCanonicalConstant
    explained_by: List[NamedConstant] or None
    
    def __init__(self: Explained, pcf: PcfCanonicalConstant, explained_by: List[NamedConstant] or NamedConstant or None = None):
        self.pcf = pcf
        self.const_id = pcf.const_id
        if not explained_by or isinstance(explained_by, list):
            self.explained_by = explained_by
        else:
            self.explained_by = [explained_by]

def extend(const, nameds, pcfs):
    ext = nameds.get(const.const_id, None)
    if ext:
        const.ext = ext
        const.symbol = ext.name
        return const
    ext = pcfs.get(const.const_id, None)
    if ext:
        const.ext = ext
        const.symbol = str(const.symbol).replace('c','p')
    return const

def to_db_format(relation: PolyPSLQRelation, consts=None) -> models.Relation:
    res = models.Relation()
    res.relation_type = ALGORITHM_NAME
    res.precision = relation.precision
    res.details = [relation.degree, relation.order] + relation.coeffs
    if consts:
        symbols = [str(c.symbol)[2:] for c in relation.constants]
        res.constants = [c for c in consts if c.const_id in symbols]
    else:
        res.constants = [c.orig for c in relation.constants] # inner constants need to be ExtendedConstant, else this fails
    return res

def from_db_format(relation: Relation, consts: List[ExtendedConstant]) -> PolyPSLQRelation:
    # the symbols don't matter much, just gonna keep them unique within the relation itself
    # the orig.const_id will decide anyway
    return PolyPSLQRelation(consts, relation.details[0], relation.details[1], relation.details[2:])

def all_consts(extended=False):
    # faster to query everything at once! and it's safer when committing
    consts = {c.const_id:ExtendedConstant.from_db(c, f'c{i}') for i,c in enumerate(db.session.query(Constant)) if c.value}
    if extended:
        nameds = {n.const_id:n for n in db.session.query(NamedConstant)}
        pcfs = {p.const_id:p for p in db.session.query(PcfCanonicalConstant)}
        consts = {c:extend(consts[c], nameds, pcfs) for c in consts}
    return consts

def all_relations(consts=None, include_vague=False) -> List[PolyPSLQRelation]:
    consts = consts or all_consts()
    rels = {r.relation_id:r for r in db.session.query(Relation)}
    rels = [(rels[relation_id], [consts[p[0]] for p in g]) for relation_id, g in groupby(db.session.query(constant_in_relation_table), lambda p:p[1])]
    rels_real = [from_db_format(*x) for x in rels if x[0].relation_type!='VAGUE']
    if include_vague:
        rels_vague = [from_db_format(*x) for x in rels if x[0].relation_type=='VAGUE']
        return rels_real, rels_vague
    return rels_real

def to_tex(expr):
    return str(expr).replace('**','^').replace('*','')

def as_factors_str(poly, despace=False):
    factors = poly.factor_list()
    prefix = ''
    leading = factors[0]
    if leading < 0:
        prefix = '-' + (r'\!' if despace else '')
        leading = -leading
    if leading != 1:
        prefix += str(leading)
    return prefix + to_tex(reduce(mul, [p[0].expr**p[1] for p in factors[1]], 1))

def despace(n):
    if n < 0:
        return f'-\\!{-n}'
    return str(n)

def pcf_to_tex(pcf):
    # oddly enough only need to despace b
    return f'{pcf.a(0)}+\\cfrac{{{despace(pcf.b(1))}}}{{'\
           + f'{pcf.a(1)}+\\cfrac{{{despace(pcf.b(2))}}}{{'\
           + f'{pcf.a(2)}+\\cfrac{{{despace(pcf.b(3))}}}{{'\
           + f'\\ddots+\\cfrac{{{as_factors_str(pcf.b, True)}}}{{{as_factors_str(pcf.a)}}}}}}}}}'

doc = Document('basic')
doc.packages.append(Package('amsmath'))

pcf = PCF(3*n*n+5*n-3, -n*n*(n+2)*(2*n-3))
with doc.create(Section('LIReC relations',numbering=False)):
 doc.append(NoEscape(r'$$\mathrm{PCF}[3n^2+5n-3,-n^2(n+2)(2n-3)]=\frac{8}{15\zeta(2)-16\pi+22}$$'))
 doc.append(NoEscape(rf'$$\mathrm{{PCF}}[3n^2+5n-3,-n^2(n+2)(2n-3)]\stackrel{{?}}{{=}}{pcf_to_tex(pcf)}$$'))

doc.generate_tex()
try:
 doc.generate_pdf(clean_tex=False)
 os.startfile('basic.pdf')
except:
 print('PyLaTeX could not find a LaTeX compiler, or something else went wrong')
