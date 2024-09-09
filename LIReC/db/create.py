from decimal import Decimal
from os import system
from LIReC.db.access import connection, db
from LIReC.db.models import NamedConstant, Constant
from LIReC.lib.calculator import Constants, Universal

if __name__ == '__main__':
    oldname = connection.db_name
    connection.db_name = 'postgres'
    system(f'psql {connection} < LIReC/db/create.sql')
    connection.db_name = oldname
    
    precision = 16000
    print(f'Using {precision} digits of precision')
    Constants.set_precision(precision)
    for const in Constants.__dict__.keys():
        if const[0] == '_' or const == 'set_precision':
            continue
        print(f'Adding named constant {const}')
        if 'CAUTION' in Constants.__dict__[const].__get__(0).__doc__: Constants.set_precision(precision // 4)
        else: Constants.set_precision(precision)
        db.session.add(Universal.calc_named(const, None, True, True))
    
    from mpmath import zeta
    
    for x in [2, 4, 5, 6, 7]:
        named_const = NamedConstant()
        named_const.base = Constant()
        named_const.base.precision = precision
        named_const.base.value = Decimal(str(zeta(x)))
        named_const.name = f'zeta{x}'
        named_const.description = f'zeta({x})'
        db.session.add(named_const)
    
    db.session.commit()
    db.session.close()
