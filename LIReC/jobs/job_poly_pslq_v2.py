'''
Finds polynomial relations between constants in LIReC, using PSLQ.

Configured as such:
'degree' + 'order':
    Two integers. All relations are structured like multivariate polynomials over the constants and CFs,
    of degree 'degree' with a maximum exponent of 'order'. For example, a 2-variable polynomial of
    degree 2 and order 1 will be of the form a+bx+cy+dxy (note the lack of x^2 and y^2), and
    a 4-variable polynomial of degree (3,1) will be of the form:
        a + bx+cy+dz+ew + fxy+gxz+hxw+iyz+jyw+kzw + lxyz+mxyw+nxzw+oyzw
    Note here the lack of any single variable with an exponent greater than 1, and also the lack of xyzw.
'min_precision':
    Only use constants with this much digital precision (everything else is ignored).
'testing_precision' + 'min_roi':
    Parameters passed to our modified PSLQ, see pslq_utils.poly_check for more information.
    (if testing_precision is absent, min_precision is used instead)
'bulk':
    If present, instead of testing all constants at once in the query phase,
    only 'bulk' constants are added at a time to test for relations.
'filters':
    A dictionary that specifies which kinds of constants to use to look for relations.
    If present, no anti-relation logging can happen. Currently supported:
    'PcfCanonical': 'balanced_only' filters to only PCFs of balanced degrees if set to True.
'''
import mpmath as mp
from itertools import groupby, product
from logging import getLogger
from logging.config import fileConfig
from os import getpid
from sympy import Symbol
from time import time
from traceback import format_exc
from typing import List, Dict
from LIReC.db.access import db
from LIReC.db import models
from LIReC.lib.pslq_utils import *

EXECUTE_NEEDS_ARGS = True

ALGORITHM_NAME = 'POLYNOMIAL_PSLQ'
UNRELATION_NAME = 'NO_PSLQ'
LOGGER_NAME = 'job_logger'
EXTENSION_TYPES = ['PowerOf', 'Derived', 'PcfCanonical', 'Named'] # ordered by increasing priority

# need to keep hold of the original db constant, but pslq_utils doesn't care for that so this is separate here
class DualConstant(PreciseConstant):
    orig: models.Constant
    
    def __init__(self, value, precision, orig, symbol=None):
        self.orig = orig
        super().__init__(value, precision, symbol)
    
    @staticmethod
    def from_db(const: models.Constant):
        return DualConstant(const.value, const.precision, const, Symbol(f'C_{const.const_id}'))

def get_const_class(const_type):
    name = const_type + 'Constant'
    if name not in models.__dict__:
        raise ValueError(f'Unknown constant type {const_type}')
    return models.__dict__[name]

def is_workable(x, prec, roi):
    return x and roi * abs(mp.log10(x)) < prec # first testing if x is zero to avoid computing log10(x)

def lowest_priority(consts: List[DualConstant], priorities: Dict[str, int]):
    return sorted(consts, key=lambda c: (-priorities[c.orig.const_id], c.orig.time_added))[-1]

def to_db_format(relation: PolyPSLQRelation, consts=None) -> models.Relation:
    res = models.Relation()
    res.relation_type = ALGORITHM_NAME
    res.precision = relation.precision
    res.details = [relation.degree, relation.order] + relation.coeffs
    if consts:
        symbols = [str(c.symbol)[2:] for c in relation.constants]
        res.constants = [c for c in consts if c.const_id in symbols]
    else:
        res.constants = [c.orig for c in relation.constants] # inner constants need to be DualConstant, else this fails
    return res

def run_query(degree=2, order=1, min_precision=50, min_roi=2, testing_precision=None, bulk=10, filters=None):
    fileConfig('LIReC/logging.config', defaults={'log_filename': 'poly_pslq_main'})
    testing_precision = testing_precision if testing_precision else min_precision
    consts = [[c] for c in db.session.query(models.Constant).filter(models.Constant.precision >= min_precision).order_by(models.Constant.const_id)]
    for i, const_type in enumerate(EXTENSION_TYPES):
        const_class = get_const_class(const_type)
        exts = db.session.query(const_class)
        if const_type == 'PcfCanonical': # no rationals please
            exts = exts.filter(models.PcfCanonicalConstant.convergence != models.PcfConvergence.RATIONAL.value)
        exts = exts.all() # evaluate once then reuse
        consts = [c + [e for e in exts if e.const_id == c[0].const_id] for c in consts]
        if filters and filters.get(const_type, {}):
            if const_type == 'PcfCanonical' and filters[const_type].get('balanced_only', False):
                consts = [c for c in consts if len(c[-1].P) == len(c[-1].Q)]
        consts = [([c[0], i] if isinstance(c[-1], const_class) else c) for c in consts] # don't need the extension data after filtering, and only need the top priority extension type
    
    # enforce constants that have extensions! also enforce "workable numbers" which are not "too large" nor "too small" (in particular nonzero)
    # also don't really care about symbolic representation for the constants here, just need them to be unique
    priorities = {c[0].const_id : c[1] for c in consts if len(c) > 1}
    consts = [DualConstant.from_db(c[0]) for c in consts if len(c) > 1 and is_workable(c[0].value, testing_precision, min_roi)]
    
    relations = db.relations()
    while True: # remove constants that you know are related
        redundant = combination_is_old(consts, degree, order, relations)
        if redundant:
            to_remove = lowest_priority(redundant.constants, priorities)
            consts = [c for c in consts if c.symbol != to_remove.symbol]
        else: # found all redundant constants!
            break
    
    testing_consts = []
    refill = True
    relations = []
    getLogger(LOGGER_NAME).info(f'QUERY BEGIN - {len(consts)} constants, {bulk} constants at a time')
    getLogger(LOGGER_NAME).info(f'QUERY BEGIN - all constants have {min_precision} precision or more')
    getLogger(LOGGER_NAME).info(f'QUERY BEGIN - searching for relations with degree {degree} and order {order}')
    getLogger(LOGGER_NAME).info(f'QUERY BEGIN - PSLQ will be run with a precision of {testing_precision}')
    getLogger(LOGGER_NAME).info(f"QUERY BEGIN - R.O.I on PSLQ result's norm is required to be at least {min_roi}")
    while not (refill and not consts): # keep going until you want to refill but can't
        if refill:
            testing_consts += consts[:bulk]
            consts = consts[bulk:]
            refill = False
        getLogger(LOGGER_NAME).debug(f'testing {len(testing_consts)} constants. {len(consts)} remain in reserve')
        new_rels = check_consts(testing_consts, degree, order, testing_precision, min_roi)
        if new_rels:
            getLogger(LOGGER_NAME).info(f'found {len(new_rels)} relations:')
            for r in new_rels:
                getLogger(LOGGER_NAME).debug(str(r))
            relations += new_rels
            testing_consts = list(set(testing_consts) - {lowest_priority(r.constants, priorities) for r in new_rels})
        else:
            refill = True
    
    # TODO log unrelation
    
    getLogger(LOGGER_NAME).info(f'QUERY DONE - found {len(relations)} preliminary relations, leaving {len(testing_consts)} constants unrelated')
    getLogger(LOGGER_NAME).info(f'QUERY DONE - subjobs will now attempt to find transitive relations')
    db.session.add_all([to_db_format(r) for r in relations])
    db.session.commit()
    return relations
    # TODO investigate randomness! on catalan+22 pcfs, sometimes finds 57 relations, sometimes finds 59

def execute_job(query_data, degree=2, order=1, min_precision=50, min_roi=2, testing_precision=None, bulk=10, filters=None, manual=False):
    # actually faster to manually query everything at once!
    try:
        fileConfig('LIReC/logging.config', defaults={'log_filename': 'analyze_pcfs' if manual else f'poly_pslq_subjob_{getpid()}'})
        testing_precision = testing_precision if testing_precision else min_precision
        relations = db.relations()
        new_relations = []
        total = len(query_data)*len(relations)
        getLogger(LOGGER_NAME).info(f'BEGIN - searching {total} pairs of relations for transitive relations:')
        last_percent = 0
        for i, (r1, r2) in enumerate(product(query_data, relations)):
            dummy_rel = PolyPSLQRelation(r1.constants + [c for c in r2.constants if c.symbol not in [c.symbol for c in r1.constants]],
                                         max(r1.degree, r2.degree), max(r1.order, r2.order), []) # coeffs don't matter!
            new_rels = [r for r in check_subrelations(dummy_rel, testing_precision, min_roi, relations + new_relations) if r.coeffs]
            if new_rels:
                getLogger(LOGGER_NAME).info(f'found {len(new_rels)} relations:')
                for r in new_rels:
                    getLogger(LOGGER_NAME).debug(str(r))
            if (total >= 100) and (100*i > total*last_percent): # not gonna bother if total is too small, this should be fast enough then
                last_percent += 1
                getLogger(LOGGER_NAME).debug(f'{last_percent}% done...')
            new_relations += new_rels
        getLogger(LOGGER_NAME).info(f'DONE - found {len(new_relations)} transitive relations')
        return new_relations
    except:
        getLogger(LOGGER_NAME).error(f'exception while executing job: {format_exc()}')
        

def summarize_results(results):
    fileConfig('LIReC/logging.config', defaults={'log_filename': 'poly_pslq_main'})
    relations = db.relations()
    new_relations = []
    for result in results:
        new_relations += [r for r in result if not combination_is_old(r.constants, r.degree, r.order, relations+new_relations)]
    all_consts = db.session.query(models.Constant).all() # need to "internally refresh" the constants so they get committed right
    db.session.add_all([to_db_format(r, all_consts) for r in new_relations])
    db.session.commit()
    getLogger(LOGGER_NAME).info(f'JOB DONE - subjobs found {len(new_relations)} relations in total')
