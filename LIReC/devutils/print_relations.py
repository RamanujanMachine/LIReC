from functools import reduce
from operator import add
from sympy import Poly, Symbol
import sys
from LIReC.db.access import db
from LIReC.db.models import *
from LIReC.lib.pslq_utils import get_exponents

def main():
    keep_going = len(sys.argv) > 1
    print(f'printing relations one at a time in descending order of precision{"" if keep_going else ", press enter to print the next"}')
    n = Symbol('n')
    rels = db.session.query(Relation).order_by(Relation.precision.desc()).all()
    rels_vague = [r for r in rels if r.relation_type=='VAGUE']
    rels = [r for r in rels if r.relation_type!='VAGUE']
    nameds = db.session.query(NamedConstant).all()
    pcfs = db.session.query(PcfCanonicalConstant).all()
    for rel in rels:
        exponents = get_exponents(*rel.details[:2], len(rel.constants))
        monoms = [reduce(add, (f'*c{i}**{exp[i]}' for i in range(len(rel.constants))), f'{rel.details[2:][j]}') for j, exp in enumerate(exponents)]
        poly = Poly(reduce(add, ['+'+monom for monom in monoms], ''))
        toprint = f'\r\npoly: {poly.expr}, precision: {rel.precision}' + ', consts: {\r\n'
        for const in rel.constants:
            named = [n for n in nameds if n.const_id == const.const_id]
            if named:
                toprint += f'    {named[0].name} : {named[0].description}'
            else:
                pcf = [p for p in pcfs if p.const_id == const.const_id]
                if pcf:
                    pcf = pcf[0]
                    toprint += f'    P: {Poly(pcf.P, n).expr}, Q: {Poly(pcf.Q, n).expr}'
                else:
                    print(f'constant with uuid {const.const_id} has no known extension')
            toprint += f', precision: {const.precision}, value: {str(const.value)[:50]}...'
            if const.source_notes:
                toprint += f', source: {const.source_ref.alias} ({const.source_notes})'
            toprint += '\r\n'
        
        print(toprint + '} (uuids: ' + f'{[str(c.const_id) for c in rel.constants]})')
        
        ids = {c.const_id for c in rel.constants}
        if any(v for v in rels_vague if ids == {c.const_id for c in v.constants}):
            print('This is exactly predicted by a vague relation!')
        if not keep_going:
            input()

if __name__ == '__main__':
    main()
