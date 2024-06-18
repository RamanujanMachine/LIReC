from __future__ import annotations
import logging
from collections import namedtuple
from decimal import Decimal, getcontext
from functools import reduce
from itertools import combinations
from sympy import Symbol, parse_expr
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
from pickle import dump, load # better than json because it supports a lot more. not human readable but we don't care
from psycopg2.errors import UniqueViolation
from time import time
from typing import Tuple, List, Dict, Generator
from LIReC.db import models
from LIReC.lib.calculator import Universal
from LIReC.lib.pcf import *
from LIReC.lib.pslq_utils import *

# the default itertools.groupby assumes the input is sorted according to the key.
# this doesn't assume that and is faster than groupby(sorted(...)), though it requires
# the output of key to be hashable...
def groupby(iterable, key=lambda x:x):
    from collections import defaultdict
    d = defaultdict(list)
    for item in iterable:
        d[key(item)].append(item)
    return d.items()

# need to keep hold of the original db constant, but pslq_utils doesn't care for that so this is separate here
class DualConstant(PreciseConstant):
    orig: models.Constant
    
    def __init__(self, value, precision, orig, symbol=None):
        self.orig = orig
        super().__init__(value, precision, symbol)
    
    @staticmethod
    def from_db(const: models.Constant):
        return DualConstant(const.value, const.precision, const, Symbol(f'C_{const.const_id}'))

class DBConnection:
    host: str
    port: int
    user: str
    passwd: str
    db_name: str
    
    def __init__(self):
        self.host = 'database-1.c1keieal025m.us-east-2.rds.amazonaws.com'
        self.port = 5432
        self.user = 'spectator_public'
        self.passwd = 'helloworld123'
        self.db_name = 'lirec-main'
    
    def __str__(self):
        return f'postgresql://{self.user}:{self.passwd}@{self.host}:{self.port}/{self.db_name}'

class LIReC_DB:
    def __init__(self):
        self.session = None
        self.auto_pcf = {
            'depth': 8192,
            'precision': 50,
            'force_fr': True,
            'timeout_sec': 60,
            'timeout_check_freq': 1000,
            'no_exception': False
        }
        self.use_cache = True
        self.cache_path = str(Path(os.getcwd()) / 'lirec.cache')
        self.auto_recache_delay_sec = 60*60*24*7 # exactly one week
        self.cache = None
        self.cached_tables = [models.Constant, models.NamedConstant, models.PcfCanonicalConstant, models.constant_in_relation_table, models.Relation]
        self.reconnect()
    
    def _redownload(self):
        self.cache = {str(table):self.session.query(table).all() for table in self.cached_tables}
        with open(self.cache_path, 'wb') as f:
            dump(self.cache, f)
    
    def _recache(self):
        with open(self.cache_path, 'rb') as f:
            self.cache = load(f)
    
    def _get_all(self, table):
        if self.use_cache and table in self.cached_tables:
            if not os.path.exists(self.cache_path) or (self.auto_recache_delay_sec and time() - os.path.getmtime(self.cache_path) > self.auto_recache_delay_sec):
                self._redownload()
            if not self.cache:
                self._recache()
            return self.cache[str(table)]
        return self.session.query(table).all()
    
    def _clear_cache(self):
        self.cache = None # will recache next time
    
    def _remove_cache(self):
        os.remove(self.cache_path) # will redownload the cache next time
        self._clear_cache()
    
    def reconnect(self):
        if self.session:
            self.session.rollback()
            self.session.close()
        # TODO can be unit-tested with mock-alchemy or alchemy-mock...
        self.session = sessionmaker(create_engine(str(connection), echo=False))()
        
    @property
    def constants(self):
        return self._get_all(models.NamedConstant)

    @property
    def names(self):
        return sorted(c.name for c in self.constants)

    @property
    def names_with_descriptions(self):
        return [(c.name, c.description) for c in self.constants]

    @property
    def cfs(self):
        return self._get_all(models.PcfCanonicalConstant)

    def describe(self, name):
        return next((d for c, d in self.names_with_descriptions if c == name), None)
    
    def relations_with(self, name, max_degree=2, max_order=1):
        const = next((c for c in self.constants if c.name == name), None)
        if not const:
            return []
        const = const.const_id
        pcfs = {c.const_id:c for c in self.cfs}
        names = {c.const_id:c.name for c in self.constants}
        relations = self.relations()
        rels = [r for r in relations if const in [c.orig.const_id for c in r.constants]
                                        and r.degree <= max_degree
                                        and r.order <= max_order]
        for r in rels:
            for c in r.constants:
                if c.orig.const_id in names:
                    c.symbol = names[c.orig.const_id]
                else:
                    pcf = pcfs[c.orig.const_id]
                    r.isolate = c.symbol = str(PCF.from_canonical_form((pcf.P, pcf.Q)))
        return rels

    def add_pcf_canonical(self, pcf: PCF, minimalist=False) -> models.PcfCanonicalConstant:
        # TODO implement add_pcf_canonicals that uploads multiple at a time
        const = models.PcfCanonicalConstant()
        const.base = models.Constant()
        self.session.add(Universal.fill_pcf_canonical(const, pcf, minimalist))
        
        # yes, commit and check error is better than preemptively checking if unique and then adding,
        # since the latter is two SQL commands instead of one, which breaks on "multithreading" for example
        # and also should be generally slower
        # also this can't be turned into a kind of INSERT pcf ON CONFLICT DO NOTHING statement
        # since this needs the base Constant to be added first so it gains its const_id...
        # TODO investigate if triggers can make something like ON CONFLICT DO NOTHING work anyway,
        # possibly will help with the previous TODO... maybe something like:  https://stackoverflow.com/questions/46105982/postgres-trigger-function-on-conflict-update-another-table
        try:
            self.session.commit()
            return pcf
        except Exception as e:
            self.session.rollback()
            raise e

    def add_pcf(self, pcf: PCF, minimalist=False) -> None:
        """
        Expect PCF object.
        raises IntegrityError if pcf already exists in LIReC.
        raises NoFRException if calculate is True and the pcf doesn't converge
        raises IllegalPCFException if the pcf has natural roots or if its b_n has irrational roots.
        """
        if any(r for r in pcf.a.real_roots() if r.is_integer and r > 0):
            raise IllegalPCFException('Natural root in partial denominator ensures divergence.')
        if any(r for r in pcf.b.real_roots() if r.is_integer and r > 0):
            raise IllegalPCFException('Natural root in partial numerator ensures trivial convergence to a rational number.')
        if any(r for r in pcf.b.all_roots() if not r.is_rational):
            raise IllegalPCFException('Irrational or Complex roots in partial numerator are not allowed.')
        #calculation = LIReC_DB.calc_pcf(pcf, depth) if depth else None
        # By default the coefs are sympy.core.numbers.Integer but sql need them to be integers
        return self.add_pcf_canonical(pcf.eval(**self.auto_pcf), minimalist)
    
    def add_pcfs(self, pcfs: Generator[PCF, None, None], minimalist=False) -> Tuple[List[models.PcfCanonicalConstant], Dict[str, List[PCF]]]:
        """
        Expects a list of PCF objects.
        """
        successful = []
        unsuccessful = {'Already exist': [], 'No FR': [], 'Illegal': []}
        for pcf in pcfs:
            try:
                successful.append(self.add_pcf(pcf, minimalist))
            except IntegrityError as e:
                if not isinstance(e.orig, UniqueViolation):
                    raise e # otherwise already in LIReC
                unsuccessful['Already exist'] += [pcf]
            except NoFRException:
                unsuccessful['No FR'] += [pcf]
            except IllegalPCFException:
                unsuccessful['Illegal'] += [pcf]
        return successful, unsuccessful
    
    def add_pcfs_silent(self, pcfs: Generator[PCF, None, None], minimalist=False) -> None:
        """
        Expects a list of PCF objects. Doesn't return which PCFs were successfully or unsuccessfully added.
        """
        for pcf in pcfs:
            try:
                self.add_pcf(pcf, minimalist)
            except IntegrityError as e:
                if not isinstance(e.orig, UniqueViolation):
                    raise e # otherwise already in LIReC
            except NoFRException:
                pass
            except IllegalPCFException:
                pass
    
    def canonical_forms(self) -> List[CanonicalForm]: # TODO fix the canonical forms in the db!
        return [[[int(coef) for coef in pcf.P], [int(coef) for coef in pcf.Q]] for pcf in self.cfs]
    
    def canonize(self, first, second=None) -> PCF or None:
        pcf = PCF(first, second) if second else first
        top, bot = pcf.canonical_form()
        top, bot = [[int(coef) for coef in top.all_coeffs()], [int(coef) for coef in bot.all_coeffs()]]
        if [top, bot] in self.canonical_forms():
            return PCF.from_canonical_form((top, bot))
    
    def relations_native(self, more=False) -> List[PolyPSLQRelation]:
        consts = {c.const_id:DualConstant.from_db(c) for c in self._get_all(models.Constant) if c.value}
        rels = {r.relation_id:r for r in self._get_all(models.Relation) if more or r.relation_type == 'POLYNOMIAL_PSLQ'}
        return [[rels[relation_id], [consts[p[0]] for p in g]] for relation_id, g in groupby(self._get_all(models.constant_in_relation_table), lambda p:p[1]) if relation_id in rels]
        
    def relations(self) -> List[PolyPSLQRelation]:
        return [PolyPSLQRelation(x[1], x[0].details[0], x[0].details[1], x[0].details[2:]) for x in self.relations_native()]
    
    def get_actual_pcfs(self) -> List[PCF]:
        """
        return a list of PCFs from the DB
        """
        return [PCF.from_canonical_form(c) for c in self.canonical_forms()]

    def identify(self, values, degree=2, order=1, min_prec=None, min_roi=2, isolate=0, strict=False, wide_search=False, see_also=False, verbose=False):
        if not values: # SETUP - organize values
            return []
        numbers, named, pcfs = {}, {}, {}
        for i,v in enumerate(values): # first iteration: detect expressions
            if isinstance(v, PCF):
                pcfs[i] = v
            elif isinstance(v, PreciseConstant):
                numbers[i] = v
            else:
                if not isinstance(v, str):
                    try:
                        numbers[i] = PreciseConstant(*v)
                        continue
                    except: # if can't unpack v (or PreciseConstant can't work with it), try something else
                        pass
                d = {}
                exec('from sympy import Symbol, Integer, Float', d)
                as_expr = parse_expr(str(v), global_dict=d) # sympy has its own predefined names via sympify... don't want!
                if as_expr.free_symbols:
                    named[i] = as_expr
                else:
                    if isinstance(v, float):
                        cond_print(verbose, "Warning: Python's default float type suffers from rounding errors and limited precision! Try inputting values as string or mpmath.mpf (or pslq_utils.PreciseConstant) for better results.")
                    try:
                        numbers[i] = PreciseConstant(v, max(min_prec or 0, len(str(v).replace('.','').rstrip('0'))), f'c{i}')
                    except: # no free symbols but cannot turn into mpf means it involves only sympy constants. need min_prec later
                        named[i] = as_expr # TODO do we ever get here now?
        
        if not min_prec:
            min_prec = min(v.precision for v in numbers.values()) if numbers else 50
            cond_print(verbose, f'Notice: No minimal precision given, assuming {min_prec} accurate decimal digits')
        if min_prec < MIN_PSLQ_DPS: # too low for PSLQ to work in the usual way!
            cond_print(verbose, 'Notice: Precision too low. Switching to manual tolerance mode. Might get too many results.')
        
        include_isolated = (len(values) > 1) # auto isolate one value!
        if isolate != None and not (isolate is False):
            if len(named) > 1:
                cond_print(verbose, f'Notice: More than one named constant (or expression involving named constants) was given! Will isolate for {list(named)[0]}.')
            if order > 1:
                cond_print(verbose, f'Notice: isolating when order > 1 can give weird results.')
        
        # try to autocalc pcfs, ignore pcfs that don't converge or converge too slowly
        for i in pcfs:
            if pcfs[i].depth == 0 or pcfs[i].precision < min_prec: # can accept PCFs that were already calculated, but that's up to you...
                predict = pcfs[i].predict_depth(min_prec)
                if not predict or predict > 2 ** 20:
                    cond_print(verbose, f'{pcfs[i]} either doesn\'t converge, or converges too slowly. Will be ignored!')
                else:
                    cond_print(verbose, f'Autocalculating {pcfs[i]}...')
                    pcfs[i] = pcfs[i].eval(depth=predict,precision=min_prec)
                    numbers[i] = PreciseConstant(pcfs[i].value, pcfs[i].precision, f'c{i}')
        
        res = None
        if not strict: # STEP 1 - try to PSLQ the numbers alone
            res = check_consts(list(numbers.values()), degree, order, min_prec, min_roi, False, verbose)
            if res:
                cond_print(verbose, 'Found relation(s) between the given numbers without using the named constants!')
            elif not named and not wide_search:
                cond_print(verbose, 'No named constants were given, and the given numbers have no relation. Consider running with wide_search=True to search with all named constants.')
                return []
        
        if not res: # STEP 2 - add named constants to the mix
            if named or wide_search:
                cond_print(verbose, 'Querying database...')
                names = [c for c in self._get_all(models.NamedConstant) if c.base.value and c.base.precision >= MIN_PSLQ_DPS]
                for k in named:
                    if named[k].free_symbols:
                        found = [c.base.precision for c in names if Symbol(c.name) in named[k].free_symbols]
                        if not found:
                            raise Exception(f'Named constant {named[k]} not found in the database! Did you misspell it?')
                    precision = min(found) if named[k].free_symbols else len(str(named[k]).replace('.','').rstrip('0'))
                    mp.mp.dps = max(precision, MIN_PSLQ_DPS)
                    value = named[k].subs({c.name:mp.mpf(str(c.base.value)) for c in names}).evalf(mp.mp.dps) # sympy can ignore unnecessary variables
                    numbers[k] = PreciseConstant(value, precision, f'({named[k]})')
                cond_print(verbose, 'Query done. Finding relations...')
            numbers = [numbers[k] for k in sorted(numbers)] # can flatten now, and everything will be in the original order still
            res = check_consts(numbers, degree, order, min_prec, min_roi, strict, verbose)
        
        extra = None
        if not res and wide_search: # STEP 3 - wide search
            try: # if it's a generator, convert it first
                wide_search = set(wide_search)
            except:
                pass
            consts = [PreciseConstant(c.base.value, c.base.precision, c.name) for c in names]
            sizes = sorted(wide_search) if isinstance(wide_search, set) else [1] # no more unbounded searches!
            original_symbols = {n.symbol for n in numbers}
            cond_print(verbose, 'Wide search beginning (PSLQ verbose prints are suppressed during the wide search!)')
            for i in sizes:
                cond_print(verbose, f'Searching subsets of size {i}')
                for subset in combinations(consts, i):
                    if any(c for c in subset if c.symbol=='C_10'): # TODO temporarily ignore champernowne, it's causing too many false positives
                        continue
                    to_test = numbers + list(subset)
                    min_prec = min(v.precision for v in to_test)
                    res = check_consts(to_test, degree, order, min_prec, min_roi, False, False) # too much printing!
                    res = [r for r in res if {c.symbol for c in r.constants} & original_symbols]
                    if res:
                        if see_also:
                            extra = self.relations_with(subset[0].symbol, degree, order)
                        break
                if res:
                    break
        
        if isolate != None and not (isolate is False): # need to differentiate between 0 and False
            isolate = named[0].symbol if (isolate is True) and named else isolate
            for r in res:
                r.isolate = isolate
                r.include_isolated = include_isolated
        return [res, extra] if extra else res

connection = DBConnection()
db = LIReC_DB()
