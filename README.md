## Installation

As of now, for use with EC2, [Python3.8.10](https://www.python.org/downloads/release/python-3810/) (and no later) is required. For local usage, later versions are also OK. 
```commandline
pip install git+https://github.com/RamanujanMachine/LIReC.git
```

### Access pcfs
```python
from LIReC.db.access import LIReC_DB
pcfs = LIReC_DB().get_actual_pcfs()
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
from LIReC.db.access import LIReC_DB
db = LIReC_DB()
db.identify(...)
```
The `LIReC_DB.identify` function allows one to check one (or more) numeric values against the known "famous" constants in the database for the existence of polynomial relations. The inputs are:
1. `values`: List of numbers. It is not recommended to use python's native `float` type as it cannot reach high precision. Instead, use `str` (or `decimal.Decimal` or `mpmath.mpf`) for better results.
2. `names`: List of names of famous constants that one or more of the values in `values` are suspected to be related to. For a full list of possible names try `LIReC_DB.names`.
3. `degree`: The degree of the relation. Each relation is defined as the coefficients of a multivariate polynomial, where each monomial (in multi-index notation) is of the form a_alpha \* x \*\* alpha. Then, the degree is the maximal allowed L1 norm on alpha. Defaults to 2.
4. `order`: The order of the relation, which is the maximal allowed L-infinity norm on each alpha (see degree). Defaults to 1.
5. `min_prec`: The minimal digital precision expected of the numbers in `values`. Can be omitted, in which case it will be inferred from `values`.
6. `verbose`: If set to `True`, will print various messages regarding the input/output and progress. Defaults to `False`.

The result of calling `identify` is a list of `pslq_util.PolyPSLQRelation` objects. These can be casted into a string that represents the relation and its estimated precision. For instance, try this code snippet:
```python
from LIReC.db.access import LIReC_DB
db = LIReC_DB()
import mpmath as mp
mp.mp.dps=400
results = db.identify([(100-mp.zeta(3))/(23+5*mp.zeta(3))], ['Zeta3'])
print([str(x) for x in results])
```
