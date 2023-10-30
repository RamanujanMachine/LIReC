from __future__ import annotations
import logging
from collections import namedtuple
from decimal import Decimal, getcontext
from functools import reduce
from sympy import sympify
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
            'depth': 10000,
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
        self.cached_tables = [models.Constant, models.NamedConstant]#, models.PcfCanonicalConstant] # for now not caching pcfs!
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
        match = [d for c,d in self.names_with_descriptions if c==name]
        return match[0] if match else None

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
    
    @staticmethod
    def parse_cf_to_lists(cf: models.PcfCanonicalConstant) -> CanonicalForm:
        return [int(coef) for coef in cf.P], [int(coef) for coef in cf.Q]
    
    def canonical_forms(self) -> List[CanonicalForm]:
        return [LIReC_DB.parse_cf_to_lists(pcf) for pcf in self.cfs]
    
    def get_actual_pcfs(self) -> List[PCF]:
        """
        return a list of PCFs from the DB
        """
        return [PCF.from_canonical_form(c) for c in self.canonical_forms()]

    def get_original_pcfs(self) -> List[PCF]:
        return [PCF(cf.original_a, cf.original_b) for cf in self.cfs if cf.original_a and cf.original_b]

    def identify(self, values, degree=2, order=1, min_prec=None, max_prec=None, isolate=False, wide_search=False, verbose=False):
        if not values:
            return []
        numbers, named = [], []
        for i,v in enumerate(values): # first iteration: detect expressions
            if isinstance(v, PreciseConstant):
                numbers += [v]
            else:
                as_expr = sympify(str(v))
                if as_expr.free_symbols:
                    named += [as_expr]
                else:
                    if isinstance(v, float):
                        cond_print(verbose, "Warning: Python's default float type suffers from rounding errors and limited precision! Try inputting values as string or mpmath.mpf (or pslq_utils.PreciseConstant) for better results.")
                    try:
                        numbers += [PreciseConstant(v, len(str(v).replace('.','').rstrip('0')), f'c{i}')]
                    except: # no free symbols but cannot turn into mpf means it involves only sympy constants. need min_prec later
                        named += [as_expr]
        
        if not min_prec:
            min_prec = min([MIN_PSLQ_DPS] + [v.precision for v in numbers])
            cond_print(verbose, f'Notice: No minimal precision given, assuming {min_prec} accurate decimal digits')
        if min_prec < MIN_PSLQ_DPS: # too low for PSLQ to work in the usual way!
            cond_print(verbose, 'Notice: Precision too low. Switching to manual tolerance mode. Might get too many results.')
        max_prec = max_prec if max_prec else min_prec * 2
        
        if isolate:
            if not named:
                cond_print(verbose, "Warning: no named constants given! Don't know what to isolate for!")
                isolate = False
            elif len(named) > 1:
                cond_print(verbose, f'Notice: More than one named constant (or expression involving named constants) was given! Will isolate for {named[0]}.')
            if order > 1:
                cond_print(verbose, f'Notice: isolating when order > 1 can give weird results.')
        
        # step 1 - try to PSLQ the numbers alone
        res = check_consts(numbers, degree=degree, order=order)
        if res:
            cond_print(verbose, 'Found relation(s) between the given numbers without using the named constants!')
            return res
        
        if not named and not wide_search:
            cond_print(verbose, 'No named constants were given, and the given numbers have no relation. Run with wide_search=True to search with all named constants.')
            return []
        
        # step 2 - add named constants to the mix
        cond_print(verbose, 'Querying database...')
        names = [c for c in self._get_all(models.NamedConstant) if c.base.value and c.base.precision >= MIN_PSLQ_DPS]
        if named:
            for expr in named:
                precision = min(c.base.precision for c in names if sympify(c.name) in expr.free_symbols) if expr.free_symbols else len(str(expr).replace('.','').rstrip('0'))
                mp.mp.dps = max(precision, MIN_PSLQ_DPS)
                value = expr.subs({c.name:mp.mpf(str(c.base.value)) for c in names}).evalf(mp.mp.dps) # sympy can ignore unnecessary variables
                numbers += [PreciseConstant(value, precision, f'({expr})')]
        else:
            cond_print(verbose, 'Warning: Currently the wide search is not very efficient. This may take a while...')
            named = [PreciseConstant(c.base.value, c.base.precision, c.name) for c in names]
        cond_print(verbose, 'Query done. Finding relations...')
        
        min_prec = min(v.precision for v in numbers)
        res = check_consts(numbers, degree=degree, order=order)
        if isolate and named:
            for r in res:
                r.isolate = f'({named[0]})'
        return res

connection = DBConnection()
db = LIReC_DB()
