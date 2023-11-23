from __future__ import annotations
from functools import reduce
from itertools import groupby
from operator import add
from sympy import Poly, Symbol
from typing import List
import sys
from LIReC.db.access import db
from LIReC.db.models import *
from LIReC.lib.pslq_utils import get_exponents

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

def expand(const, nameds, pcfs):
    res = nameds.get(const.const_id, pcfs.get(const.const_id, None))
    if not res:
        print(f'const {const.const_id} has no extension!')
    return res or const

def main():
    keep_going = True# len(sys.argv) > 1
    print(f'printing relations one at a time in descending order of precision{"" if keep_going else ", press enter to print the next"}')
    n = Symbol('n')
    consts = {c.const_id:c for c in db.session.query(Constant) if c.value}
    rels = {r.relation_id:r for r in db.session.query(Relation)}
    nameds = {n.const_id:n for n in db.session.query(NamedConstant)}
    pcfs = {p.const_id:Explained(p) for p in db.session.query(PcfCanonicalConstant)}
    rels = [[rels[relation_id], [expand(consts[p[0]], nameds, pcfs) for p in g]] for relation_id, g in groupby(db.session.query(constant_in_relation_table), lambda p:p[1])]
    rels_vague = [x for x in rels if x[0].relation_type=='VAGUE']
    rels = [x for x in rels if x[0].relation_type!='VAGUE']
    print(f'query done')
    for rel,constants in rels: # first explain pcfs using vague relations
        explained = {i for i,c in enumerate(constants) if isinstance(c, Explained)}
        if len(explained) == 1:
            others = [constants[i] for i in set(range(len(constants)))-explained]
            to_update = constants[list(explained)[0]]
            if not to_update.explained_by or len(others) < len(to_update.explained_by):
                to_update.explained_by = others
    
    for rel,constants in rels: # now print everything!
        exponents = get_exponents(*rel.details[:2], len(constants))
        monoms = [reduce(add, (f'*c{i}**{exp[i]}' for i in range(len(constants))), f'{rel.details[2:][j]}') for j, exp in enumerate(exponents)]
        poly = Poly(reduce(add, ['+'+monom for monom in monoms], ''), n)
        #if poly.degree()==0:
        #    print(f'WARNING: bad relation detected! check relation with id {rel.relation_id}')
        #    continue
        toprint = f'\r\npoly: {poly.expr}, precision: {rel.precision}' + ', consts: {\r\n'
        for const in constants:
            if isinstance(const, NamedConstant):
                toprint += f'    {const.name} : {const.description}'
            elif isinstance(const, Explained):
                toprint += f'    P: {Poly(const.pcf.P, n).expr}, Q: {Poly(const.pcf.Q, n).expr}'
                if const.explained_by:
                    toprint += f', related to: {[n.name if isinstance(n, NamedConstant) else n.const_id for n in const.explained_by]}'
                const=const.pcf
            else:
                print(f'WARNING: constant with uuid {const.const_id} has no known extension!')
            const=const.base
            toprint += f', precision: {const.precision}, value: {str(const.value)[:50]}...'
            if const.source_notes:
                toprint += f', source: {const.source_ref.alias} ({const.source_notes})'
            toprint += '\r\n'
        
        print(toprint + '} (uuids: ' + f'{[str(c.const_id) for c in constants]})')
        
        ids = {c.const_id for c in constants}
        if any(v for v,consts in rels_vague if ids == {c.const_id for c in consts}):
            print('This is exactly predicted by a vague relation!')
        if not keep_going:
            input()

if __name__ == '__main__':
    main()
