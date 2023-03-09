## Installation

As of now, for use with EC2, [Python3.8.10](https://www.python.org/downloads/release/python-3810/) (and no later) is required. For local usage, later versions are also OK. 
```commandline
pip install git+https://github.com/RamanujanMachine/LIReC.git
```

### Access pcfs
```python
from LIReC.lib.db_access import LIReC_DB
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