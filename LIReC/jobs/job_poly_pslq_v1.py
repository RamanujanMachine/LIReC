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
from itertools import combinations, product
import logging
from math import ceil
from os import getpid
from sqlalchemy import or_
from sqlalchemy.sql.expression import func
from time import time
from traceback import format_exc
from LIReC.db import models
from LIReC.db.access import db
from LIReC.lib.pslq_utils import *
from LIReC.jobs.job_poly_pslq_v2 import DualConstant, to_db_format # just gonna borrow this

EXECUTE_NEEDS_ARGS = True
KEEP_UNSPLIT = True
SEND_INDEX = True

ALGORITHM_NAME = 'POLYNOMIAL_PSLQ'
BULK_SIZE = 500
BULK_TYPES = {'PcfCanonical'}
SUPPORTED_TYPES = ['Named', 'PcfCanonical', 'Derived', 'PowerOf']
DEFAULT_CONST_COUNT = 1
DEFAULT_DEGREE = 2
DEFAULT_ORDER = 1
ADDON_FAMILY = 'addon'

FILTERS = [
        models.Constant.precision.isnot(None)
        #or_(models.Cf.scanned_algo == None, ~models.Cf.scanned_algo.has_key(ALGORITHM_NAME)) # TODO USE scan_history TABLE!!!
        ]

def get_filters(filters, const_type):
    filter_list = list(FILTERS) # copy!
    if 'global' in filters:
        global_filters = filters['global']
        if 'min_precision' in global_filters:
            filter_list += [models.Constant.precision >= global_filters['min_precision']]
    
    if const_type == 'PcfCanonical':
        filter_list += [models.PcfCanonicalConstant.convergence != models.PcfConvergence.RATIONAL.value]
        if filters['PcfCanonical'].get('balanced_only', False):
            filter_list += [func.cardinality(models.PcfCanonicalConstant.P) == func.cardinality(models.PcfCanonicalConstant.Q)]
    
    return filter_list 

def get_const_class(const_type):
    name = const_type + 'Constant'
    if name not in models.__dict__:
        raise ValueError(f'Unknown constant type {const_type}')
    
    return models.__dict__[name]

def get_consts(const_type, filters):
    if const_type == 'Named': # Constant first intentionally! don't need extra details, but want to filter still
        precision = filters.get('global', {}).get('min_precision', None)
        named = db.constants
        return [DualConstant.from_db(c) for c in db._get_all(models.Constant) if c.const_id in named and c.precision is not None and (precision is None or c.precision >= precision)]

def relation_is_new(consts, degree, order, other_relations):
    return not any(r for r in other_relations
                   if {c.const_id for c in r.constants} <= {c.orig.const_id for c in consts} and r.details[0] <= degree and r.details[1] <= order)

def add_addons(consts, const_type, filters, all_addons):
    if not all_addons:
        return consts
    addons = filters.get(const_type, {}).get('addons', [])
    return consts + [DualConstant.from_db(a.base) for a in all_addons if a.args['name'] in addons]

def run_query(filters=None, degree=None, order=None, bulk=None):
    if not filters:
        return []
    bulk_types = set(filters.keys()) & BULK_TYPES
    if not bulk_types:
        return []
    bulk = bulk if bulk else BULK_SIZE
    logging.info(f'Querying constants... may take a while')
    # TODO replace with _get_all and pythonic filtering
    results = {}
    if 'PcfCanonical' in bulk_types:
        precision = filters.get('global', {}).get('min_precision', None)
        pcfs = [p.const_id for p in db.cfs]
        consts = [c for c in db._get_all(models.Constant) if c.const_id in pcfs and c.precision is not None and (precision is None or c.precision >= precision)]
        from random import shuffle
        shuffle(consts)
        results['PcfCanonical'] = consts[:bulk]
    
    # apparently postgresql is really slow with the order_by(random) part,
    # but on 1000 CFs it only takes 1 second, which imo is worth it since
    # that allows us more variety in testing the CFs...
    # TODO what to do if results is unintentionally empty?
    logging.info(f'Query done, batch size is {sum(len(results[k]) for k in results)}')
    return results

def execute_job(query_data, filters=None, degree=None, order=None, bulk=None, manual=False):
    try: # whole thing must be wrapped so it gets logged
        #configure_logger('analyze_pcfs' if manual else f'pslq_const_worker_{getpid()}')
        i, total_cores, query_data = query_data # SEND_INDEX = True guarantees this
        global_filters = filters.get('global', {})
        filters.pop('global', 0) # instead of del so we can silently dispose of global even if it doesn't exist
        if not filters:
            logging.error('No filters found! Aborting...')
            return 0 # this shouldn't happen unless pool_handler changes, so just in case...
        keys = filters.keys()
        for const_type in keys:
            if const_type not in SUPPORTED_TYPES:
                msg = f'Unsupported filter type {const_type} will be ignored! Must be one of {SUPPORTED_TYPES}.'
                print(msg)
                logging.warn(msg)
                del filters[const_type]
            elif 'count' not in filters[const_type]:
                filters[const_type]['count'] = DEFAULT_CONST_COUNT
        total_consts = sum(c['count'] for c in filters.values())
        degree = degree if degree else DEFAULT_DEGREE
        order = order if order else DEFAULT_ORDER
        logging.info(f'Checking against {total_consts} constants at a time, subdivided into {({k : filters[k]["count"] for k in filters})}, using degree-{degree} relations')
        if degree > total_consts * order:
            degree = total_consts * order
            logging.info(f'redundant degree detected! reducing to {degree}')
        
        # need to "internally refresh" the constants so they get committed right
        all_consts = db._get_all(models.Constant)
        subsets = []
        for const_type in filters:
            if const_type in query_data:
                subsets += [(const_type, [DualConstant.from_db(c) for c in all_consts if c.const_id in query_data[const_type]])]
            else:
                subsets += [(const_type, get_consts(const_type, {**filters, 'global':global_filters}))]
        
        addons = None
        if any(c for c in filters if 'addons' in filters[c]):
            addons = db.session.query(models.DerivedConstant).filter(models.DerivedConstant.family == ADDON_FAMILY).all()
        subsets = [list(combinations(add_addons(x, const_type, filters, addons), filters[const_type]['count'])) for const_type, x in subsets]
        total_options = reduce(mul, [len(x) for x in subsets])
        first, last = ceil((total_options * i) / total_cores), ceil((total_options * (i + 1)) / total_cores)
        i = 0
        
        # TODO mass query the many-to-many table! the first call to relation_is_new takes too long!
        old_relations = db.relations()# db.session.query(models.Relation).filter(models.Relation.relation_type==ALGORITHM_NAME).all()
        orig_size = len(old_relations)
        test_prec = global_filters.get('min_precision', 15)
        # even if the commented code were to be uncommented and implemented for
        # the scan_history table, this loop still can't be turned into list comprehension
        # because finding new relations depends on the new relations we found so far!
        print_index, PRINT_DELAY = 0, 10
        for consts in product(*subsets):
            if i >= last:
                break
            i += 1
            if i < first:
                continue
            consts = [c for t in consts for c in t] # need to flatten...
            print_msg = f'checking consts: {[c.orig.const_id for c in consts]}'
            if print_index >= PRINT_DELAY:
                print_index = 0
                logging.info(print_msg)
            else:
                print_index += 1
                logging.debug(print_msg)
            if not combination_is_old(consts, degree, order, old_relations):
                # some leeway with the extra 10 precision
                new_relations = [r for r in check_consts(consts, degree, order, test_prec) if r.precision > PRECISION_RATIO * min(c.precision for c in r.constants) - 10]
                if new_relations:
                    logging.info(f'Found relation(s) on constants {[c.orig.const_id for c in consts]}!')
                    try_count = 1
                    while try_count < 3:
                        try:
                            db.session.add_all([to_db_format(r) for r in new_relations])
                            db.session.commit()
                            old_relations += new_relations
                            break
                        except:
                            db.session.rollback()
                            #db.session.close()
                            #db = access.LIReC_DB()
                            if try_count == 1:
                                logging.warn('Failed to commit once, trying again.')
                            else:
                                logging.error(f'Could not commit relation(s): {format_exc()}')
                        try_count += 1
            #for cf in consts:
            #    if not cf.scanned_algo:
            #        cf.scanned_algo = dict()
            #    cf.scanned_algo[ALGORITHM_NAME] = int(time())
            #db.session.add_all(consts)
        logging.info(f'finished - found {len(old_relations) - orig_size} results')
        db.session.close()
        
        logging.info('Commit done')
        
        return len(old_relations) - orig_size
    except:
        logging.error(f'Exception in execute job: {format_exc()}')
        # TODO "SSL connection has been closed unexpectedly" is a problem...
        # this is just a bandaid fix to make sure the system doesn't shit itself,
        # but we should instead figure out a way to be resistant to this and keep working normally.
        # right now this will just cause the search job to restart itself without
        # knowing where to return to. not great
        db.session.rollback()
        # not returning anything so summarize_results can see the error

def summarize_results(results):
    if any(r for r in results if r==None):
        logging.warn(f'At least one of the workers had an exception! Check logs')
    logging.info(f'In total found {sum(r for r in results if r)} relations')
