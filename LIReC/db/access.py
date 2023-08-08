from __future__ import annotations
import logging
from collections import namedtuple
from decimal import Decimal, getcontext
from functools import reduce
from sympy import Poly
from sympy.core.numbers import Integer, Rational
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker
from psycopg2.errors import UniqueViolation
from typing import Tuple, List, Dict, Generator
from LIReC.db.config import get_connection_string, auto_pcf_config
from LIReC.db import models
from LIReC.lib.calculator import Universal
from LIReC.lib.pcf import *
from LIReC.lib.pslq_utils import *

class LIReC_DB:
    def __init__(self):
        logging.debug("Trying to connect to database")
        self._engine = create_engine(get_connection_string(), echo=False)
        Session = sessionmaker(bind=self._engine)
        self.session = Session() # TODO can be unit-tested with mock-alchemy or alchemy-mock...
        logging.debug("Connected to database")

    @property
    def constants(self):
        return self.session.query(models.NamedConstant).order_by(models.NamedConstant.const_id)

    @property
    def names(self):
        return sorted(c.name for c in self.constants)

    @property
    def cfs(self):
        return self.session.query(models.PcfCanonicalConstant).order_by(models.PcfCanonicalConstant.const_id)

    def add_pcf_canonical(self, pcf: PCF, calculation: PCFCalc or None = None) -> models.PcfCanonicalConstant:
        # TODO implement add_pcf_canonicals that uploads multiple at a time
        const = models.PcfCanonicalConstant()
        const.base = models.Constant()
        self.session.add(Universal.fill_pcf_canonical(const, pcf, calculation))
        
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

    def add_pcf(self, pcf: PCF) -> None:
        """
        Expect PCF object.
        raises IntegrityError if pcf already exists in LIReC.
        raises NoFRException if calculate is True and the pcf doesn't converge
        raises IllegalPCFException if the pcf has natural roots or if its b_n has irrational roots.
        """
        if any(r for r in pcf.a.real_roots() if isinstance(r, Integer) and r > 0):
            raise PCFCalc.IllegalPCFException('Natural root in partial denominator ensures divergence.')
        if any(r for r in pcf.b.real_roots() if isinstance(r, Integer) and r > 0):
            raise PCFCalc.IllegalPCFException('Natural root in partial numerator ensures trivial convergence to a rational number.')
        if any(r for r in pcf.b.all_roots() if not isinstance(r, Rational)):
            raise PCFCalc.IllegalPCFException('Irrational or Complex roots in partial numerator are not allowed.')
        #calculation = LIReC_DB.calc_pcf(pcf, depth) if depth else None
        # By default the coefs are sympy.core.numbers.Integer but sql need them to be integers
        return self.add_pcf_canonical(pcf, PCFCalc(pcf).run(**auto_pcf_config))
    
    def add_pcfs(self, pcfs: Generator[PCF, None, None]) -> Tuple[List[models.PcfCanonicalConstant], Dict[str, List[PCF]]]:
        """
        Expects a list of PCF objects.
        """
        successful = []
        unsuccessful = {'Already exist': [], 'No FR': [], 'Illegal': []}
        for pcf in pcfs:
            try:
                successful.append(self.add_pcf(pcf))
            except IntegrityError as e:
                if not isinstance(e.orig, UniqueViolation):
                    raise e # otherwise already in LIReC
                unsuccessful['Already exist'] += [pcf]
            except PCFCalc.NoFRException:
                unsuccessful['No FR'] += [pcf]
            except PCFCalc.IllegalPCFException:
                unsuccessful['Illegal'] += [pcf]
        return successful, unsuccessful
    
    def add_pcfs_silent(self, pcfs: Generator[PCF, None, None]) -> None:
        """
        Expects a list of PCF objects. Doesn't return which PCFs were successfully or unsuccessfully added.
        """
        for pcf in pcfs:
            try:
                self.add_pcf(pcf)
            except IntegrityError as e:
                if not isinstance(e.orig, UniqueViolation):
                    raise e # otherwise already in LIReC
            except PCFCalc.NoFRException:
                pass
            except PCFCalc.IllegalPCFException:
                pass
    
    @staticmethod
    def parse_cf_to_lists(cf: models.PcfCanonicalConstant) -> CanonicalForm:
        return [int(coef) for coef in cf.P], [int(coef) for coef in cf.Q]
    
    def get_canonical_forms(self) -> List[CanonicalForm]:
        return [LIReC_DB.parse_cf_to_lists(pcf) for pcf in self.cfs.all()]
    
    def get_actual_pcfs(self) -> List[PCF]:
        """
        return a list of PCFs from the DB
        """
        return [PCF.from_canonical_form(c) for c in self.get_canonical_forms()]

    def get_original_pcfs(self) -> List[PCF]:
        return [PCF(cf.original_a, cf.original_b) for cf in self.cfs if cf.original_a and cf.original_b]

    def identify(self, values, names=None, degree=2, order=1, min_prec=None, max_prec=None, verbose=False):
        if not min_prec:
            min_prec = min(v.precision if isinstance(v, PreciseConstant) else len(str(v).replace('.','').rstrip('0')) for v in values)
            cond_print(verbose, f'Notice: No minimal precision given, assuming {min_prec} accurate decimal digits')
        if min_prec < 15: # too low for PSLQ to work!
            cond_print(verbose, 'Error: Precision too low! Must be at least 15')
            return None
        max_prec = max_prec if max_prec else min_prec * 2
        
        cond_print(any(isinstance(v, float) for v in values) and verbose, "Warning: Python's default float type suffers from rounding errors and limited precision! Try inputting values as string or mpmath.mpf (or pslq_utils.PreciseConstant) for better results.")
        values = [v if isinstance(v, PreciseConstant) else PreciseConstant(v, min_prec) for v in values]
        
        res = check_consts(values, degree=degree, order=order)
        if res:
            cond_print(verbose, 'Found relation(s) between the given values without using the hint(s)!')
            return res
        
        cond_print(verbose, 'Querying database...')
        extras = self.session.query(models.NamedConstant).join(models.Constant).filter(models.Constant.value != None).all()
        cond_print(verbose, 'Query done. Finding relations...')
        filtered = extras
        if names:
            filtered = [c for c in extras if c.name in names]
        else: # TODO
            cond_print(verbose, 'Warning: Currently the code is not very efficient when not given a names array as a hint. This may take a while ...')
        
        if verbose:
            for name in names:
                if name not in [c.name for c in extras]:
                    print(f'Warning: Named constant {name} not found! Will be ignored.')
        if not filtered:
            filtered = extras
        
        values += [PreciseConstant(c.base.value, c.base.precision, c.name) for c in filtered]
        min_prec = min(min_prec, min(c.base.precision for c in filtered if c.base.precision >= 15))
        return check_consts(values, degree=degree, order=order)
