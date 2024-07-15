from setuptools import setup

setup(
    name="LIReC",
    version="0.0.6",
    # due to Amazon EC2 issues, we probably can't go beyond Python3.9
    # (Python3.10.8 seems to have issues with pip). As such, I don't care if
    # "this is an outdated version of python" until one of 2 things happens:
    #     1. We no longer need EC2
    #     2. Someone manages a working EC2 instance with a later python version
    python_requires=">=3.8.10",
    description="Library of Integer RElations and Constants",
    packages=['LIReC', 'LIReC.lib', 'LIReC.jobs', "LIReC.db"], # devutils folder intentionally omitted
    package_data={'': ['logging.config']}, # manually include extra files
    install_requires=[
        'psycopg2>=2.8.6',
        'sqlalchemy>=2.0.5',
        
        'gmpy2>=2.1.5',
        'mpmath>=1.2.1',
        'sympy>=1.5.1'
    ]
)
