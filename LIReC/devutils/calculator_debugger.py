from LIReC.lib.calculator import *
from LIReC.db.access import db

PREFIX_LINK = 'OEIS link: '
PREFIX_URL = 'org/A'
MIN_PRECISION = 20

# TODO add a name filter to each function here so only names in the given list are tested (no list = test everything)

def vs_oeis():
    # TODO some of the logic here is copypasted from within Universal.calc_named, and that part of the logic should get separated sometime
    precision = 16000
    for const in Constants.__dict__.keys():
        if const[0] == '_' or const == 'set_precision':
            continue
        const_func = Constants.__dict__[const].__get__(0)
        if 'CAUTION' in const_func.__doc__: Constants.set_precision(precision // 4)
        else: Constants.set_precision(precision)
        calculated = Universal.calc_named(const)
        if calculated and PREFIX_LINK in const_func.__doc__:
            i = const_func.__doc__.index(PREFIX_LINK)
            url = const_func.__doc__[i + len(PREFIX_LINK) : i + const_func.__doc__[i:].index('\n')]
            url = f'{url}/b{url[url.index(PREFIX_URL) + len(PREFIX_URL) : ]}.txt'
            oeis = Universal.read_oeis(urlopen(url))
            if oeis[1] >= MIN_PRECISION:
                oeis_value = oeis[0]
                print(const, mp.nstr(mp.mpf(str(calculated.base.value)) - mp.mpf(oeis_value)), oeis[1])

def vs_db():
    names = db.constants
    for named in names:
        if named.base.value and named.name in Constants.__dict__:
            Universal.set_precision(named.base.precision)
            expected = Universal.calc_named(named.name, force=True)
            print(named.name, mp.nstr(mp.mpf(str(named.base.value)) - mp.mpf(str(expected.base.value))), named.base.precision)
