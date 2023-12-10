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
The `db.identify` function allows one to check one (or more) numeric values against the known "famous" constants in the database for the existence of polynomial relations. This is done in three steps:
- First, any decimal expressions within `values` (see below) are tested against each other for relation. This can be skipped by inputting `first_step=False` (see below).
- If nothing was found yet, all expressions in `values` are tested together for a relation.
- Finally, if still nothing was found and `wide_search` is truthy (see below), executes an iterative search against all named constants in the database, stopping at the first relation found. 

The inputs are:
1. `values`: List that contains any combination of the following:
   - Numeric values. It is not recommended to use python's native `float` type as it cannot reach high precision. Instead, use `str` (or `decimal.Decimal` or `mpmath.mpf`) for better results. Each such value will be given a name `c{i}` where `i` is its position in `values`, and these are assumed to all be accurate up to the least accurate value among them (unless `min_prec` is specified, see below). If this is not the case, input less digits or explicitly specify `min_prec`.
   - Strings that represent famous constants. For a full list of possible names try `db.names`. Also try `db.names_with_descriptions` for a full list of names along with a short description of each name, or `db.describe` to conveniently fish out the description of any one name in the database (if it exists).
   - `Sympy` expressions that involve one or more famous constants. Can be inputted either in string form or as `sympy.Expr` form.
2. `degree`: The maximal degree of the relation. Each relation is defined as the coefficients of a multivariate polynomial, where each monomial (in multi-index notation) is of the form a_alpha \* x \*\* alpha. Then, the degree is the L1 norm on alpha. Defaults to 2.
3. `order`: The maximal order of the relation, which is the L-infinity norm on each alpha (see degree). Defaults to 1.
4. `min_prec`: The minimal digital precision expected of the numbers in `values`. Can be omitted, in which case it will be inferred from `values`, see above.
5. `min_roi`: Given a vector of random numbers, the total amount of digits in an integer relation on them is usually about the same as the working precision. To differentiate between such "garbage" cases and potentially substantive results, any relation found will be required to have `min_roi` times less digits than the working precision to be returned. Defaults to `2`.
6. `isolate`: Modifies the way results will be printed:
   - If set to `False`, results will be printed in the form `expr = 0 (precision)`, where `precision` specifies how many digits the computer knows for sure satisfy the equality. This is the default option.
   - If set to `True`, will take the first recognized named constant and isolate it as a function of all other constants in the relations it participates in, resulting in the form `const = expr (precision)`.
   - If set to a number `i`, will similarly isolate `values[i]` as a function of everything else.
7. `first_step`: Whether or not to perform the first step of identification (see above). Defaults to `True`.
8. `wide_search`: If truthy, enables the wide search (see above). Also, if this is a `list` (or any kind of collection) of positive integers, the search is limited to subsets of sizes contained in `wide_search`. Do note that the wide search may take a long time, so Ctrl+C it if you have to. Defaults to `False`.
9. `verbose`: If set to `True`, will print various messages regarding the input/output and progress. Defaults to `False`.

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
