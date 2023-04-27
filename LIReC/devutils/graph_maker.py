from LIReC.lib.db_access import LIReC_DB
from LIReC.lib.models import *

def make_graph(const_name):
    db = LIReC_DB()
    rels = db.session.query(Relation).order_by(Relation.precision.desc()).all()
    nameds = db.session.query(NamedConstant).all()
    special_id = [c for c in nameds if c.name == const_name][0].const_id
    special_rels = [r for r in rels if special_id in [c.const_id for c in r.constants]]
    while True:
        participating = {c.const_id for rel in special_rels for c in rel.constants}
        transitive = [r for r in rels if {c.const_id for c in r.constants} & participating]
        if len(transitive) > len(special_rels):
            special_rels = transitive
        else:
            break
    participating = list({c.const_id for rel in special_rels for c in rel.constants})
    edges = sorted(sorted(participating.index(c.const_id) for c in r.constants) for r in special_rels)
    print(edges)
    print('index of ' + const_name + ' is ' + str(participating.index(special_id)))

def main():
    make_graph('C')

if __name__ == '__main__':
    main()
