from setuptools import setup

setup(
    name="LIReC",
    version="0.0.1",
    # due to Amazon EC2 issues, we probably can't go beyond Python3.9
    # (Python3.10.8 seems to have issues with pip). As such, I don't care if
    # "this is an outdated version of python" until one of 2 things happens:
    #     1. We no longer need EC2
    #     2. Someone manages a working EC2 instance with a later python version
    # Even then, as of now numba (including llvmlite) and ortools do not support python3.11, see respectively:
    # https://github.com/numba/numba/issues/8304 ; https://github.com/google/or-tools/issues/3515
    # Because of this, upgrading to Python3.11 will also require these packages to update.
    python_requires=">=3.8.10",
    description="Library of Integer RElations and Constants",
    packages=['LIReC', 'LIReC.lib', 'LIReC.jobs', "LIReC.su"],
    install_requires=[
        'sqlalchemy>=2.0.5',
        'sympy>=1.5.1',
        'psycopg2>=2.8.6',
        'mpmath>=1.2.1',
        'gmpy2>=2.1.5',
    ]
)
