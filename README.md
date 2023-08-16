## Installation

As of now, for use with EC2, [Python3.8.10](https://www.python.org/downloads/release/python-3810/) (and no later) is required. For local usage, later versions are also OK. 
```commandline
pip install git+https://github.com/RamanujanMachine/LIReC.git
```

### Access pcfs
```python
from LIReC.db.access import db
pcfs = db.get_actual_pcfs()
```
Each pcf is of PCF class which is defined in lib\pcf.py

To access the properties of each pcf:
```python
pcf = pcfs[0]
pcf.a # Polynomial of a_n
pcf.b # Polynomial of b_n

pcf.a.all_coeffs() # List of the coefficients of a_n
pcf.b.all_coeffs() # Same for b_n
```

### Numeric Identification
```python
from LIReC.db.access import db
db.identify(...)
```
The `db.identify` function allows one to check one (or more) numeric values against the known "famous" constants in the database for the existence of polynomial relations. The inputs are:
1. `values`: List that contains any combination of the following:
   - Numeric values. It is not recommended to use python's native `float` type as it cannot reach high precision. Instead, use `str` (or `decimal.Decimal` or `mpmath.mpf`) for better results.
   - Strings that represent famous constants. For a full list of possible names try `db.names`. Also try `db.names_with_descriptions` for a full list of names along with a short description of each name, or `db.describe` to conveniently fish out the description of any one name in the database (if it exists).
   - `Sympy` expressions that involve one or more famous constants. Can be inputted either in string form or as `sympy.Expr` form.
3. `degree`: The degree of the relation. Each relation is defined as the coefficients of a multivariate polynomial, where each monomial (in multi-index notation) is of the form a_alpha \* x \*\* alpha. Then, the degree is the maximal allowed L1 norm on alpha. Defaults to 2.
4. `order`: The order of the relation, which is the maximal allowed L-infinity norm on each alpha (see degree). Defaults to 1.
5. `min_prec`: The minimal digital precision expected of the numbers in `values`. Can be omitted, in which case it will be inferred from `values`.
6. `isolate`: If set to `True`, will take the first recognized constant from `names` and isolate it as a function of all other constants in the relations it participates in. Defaults to `False`.
7. `verbose`: If set to `True`, will print various messages regarding the input/output and progress. Defaults to `False`.

The result of calling `identify` is a list of `pslq_util.PolyPSLQRelation` objects. These can be casted into a string that represents the relation and its estimated precision. For instance, try this code snippet:
```python
from LIReC.db.access import db
import mpmath as mp
mp.mp.dps=400
results = db.identify([(100-mp.zeta(3))/(23+5*mp.zeta(3)), 'Zeta3']) # first run should take a few seconds to query the db...
print([str(x) for x in results])
results = db.identify([mp.mpf(1)/3 - mp.zeta(3), 'Zeta3']) # this works
print([str(x) for x in results])
results = db.identify([1/3 - mp.zeta(3), 'Zeta3']) # this doesn't! don't let bad floats pollute your numbers!
print([str(x) for x in results])
```
