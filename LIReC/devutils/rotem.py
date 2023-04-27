# All of our calculations require high precision. We use mpmath, and set the precision here.
import mpmath
import math
from mpmath import mpf
mpmath.mp.dps = 20_000

analytical_cmfs = {
    'deg1': {
        'funcs_generator': lambda coefs: (
            lambda x,y: coefs[0] + coefs[1]*(x + y), 
            lambda x,y: coefs[2] + coefs[3]*(x - y)
        ),
        'n_coefs': 4
    },
    
    'deg2': {
        'funcs_generator': lambda coefs: (
            lambda x,y: 
                  (2*coefs[1]+coefs[2]) * (coefs[1]+coefs[2]) - coefs[3]*coefs[0] \
                  - coefs[3] * ( (2*coefs[1]+coefs[2]) * (x+y) + (coefs[1]+coefs[2]) * (2*x+y) ) \
                  + coefs[3]**2 * (2*x**2 + 2*x*y + y**2), 
            lambda x,y: 
                  coefs[3] * (coefs[0] + coefs[2]*x + coefs[1]*y - coefs[3] * (2*x**2 - 2*x*y + y**2)) 
        ),
        'n_coefs': 4
    },
    
    'deg3_1': {
        'funcs_generator': lambda coefs: (
            lambda x,y: 
                -( (coefs[0] + coefs[1]*(x+y) ) * (coefs[0]*(x + 2*y) + coefs[1]*(x**2 + x*y + y**2)) ), 
            lambda x,y: 
                (coefs[0] + coefs[1]*(-x + y)) * (coefs[0]*(x - 2*y) - coefs[1]*(x**2 - x*y + y**2 ))
        ),
        'n_coefs': 2
    },
    'deg3_2': {
        'funcs_generator': lambda coefs: (
            lambda x,y: 
                -(x + y)*(coefs[0]**2 + 2*coefs[1]**2 * (x**2 + x*y + y**2)), 
            lambda x,y: 
                (x - y)*(coefs[0]**2 + 2*coefs[1]**2 * (x**2 - x*y + y**2))
        ),
        'n_coefs': 2
    },
    'deg3_3': {
        'funcs_generator': lambda coefs: (
            lambda x,y: 
                (x + y)*(coefs[0]**2 - coefs[0]*coefs[1]*(x - y) - 2*coefs[1]**2 * (x**2 + x*y + y**2)), 
            lambda x,y: 
                (coefs[0] + coefs[1]*(x - y)) * (3*coefs[0]*(x - y) + 2*coefs[1]*(x**2 - x*y + y**2))
        ),
        'n_coefs': 2
    },
}

def get_mx(f, f_bar):
    return lambda x,y: mpmath.matrix([
        [0                    ,    f(x,0)*f_bar(x,0) - f(0,0)*f_bar(0,0)],
        [1                    ,    f(x,y) - f_bar(x+1,y)                ]
    ]).T

def get_my(f, f_bar):
    return lambda x,y: mpmath.matrix([
        [f_bar(x,y)           ,    f(x,0)*f_bar(x,0) - f(0,0)*f_bar(0,0)],
        [1                    ,    f(x,y)                               ]
    ]).T

def get_delta_no_limit(lattice_cell, metadata='', location={}):
    try:
        p_n_1 = int(lattice_cell[0,0])
        q_n_1 = int(lattice_cell[0,1])
        p_n = int(lattice_cell[1,0])
        q_n = int(lattice_cell[1,1])
        if q_n == -1 or q_n_1 == -1:
            return mpf(-1)
        gcd_n_1 = math.gcd(int(p_n_1), int(q_n_1))
        gcd_n = math.gcd(int(p_n), int(q_n))  
        p_n_1 = p_n_1 // gcd_n_1
        p_n = p_n // gcd_n
        q_n_1 = q_n_1 // gcd_n_1
        q_n = q_n // gcd_n
        delta = (mpmath.log(abs(q_n_1)) - mpmath.log(abs(p_n * q_n_1 - q_n * p_n_1))) / mpmath.log(abs(q_n))
        return delta
    except Exception as e:
        return mpf(-1)

# ONLY PRINT UNKNOWN LIMTIS
import itertools
from mpmath import mpf, ln, e, nstr, pi, sqrt, zeta
import math
from LIReC.lib.db_access import LIReC_DB
from LIReC.lib.models import NamedConstant

possible_constants = [ln(2), e, e**2, e**0.5, pi, zeta(3), zeta(2),
                      mpf(2)**(1/mpf(3)), mpf(2)**(2/mpf(3)), mpf(5)**(1/mpf(3)), mpf(5)**(2/mpf(3)),  
                      sqrt(2), sqrt(3), sqrt(5), sqrt(6), sqrt(7), sqrt(8)]
all_named = LIReC_DB().session.query(NamedConstant).all()
possible_constants += [mpf(str(c.base.value)) for c in all_named if c.base.value and c.base.value < 1000] # ramanujan's almost integer is causing trouble...

dept = 500

for pcf_family in analytical_cmfs:
    for coefs in itertools.product(*[range(5) for _ in range(analytical_cmfs[pcf_family]['n_coefs'])]):
        
        #print(f"trying {coefs}")
        
        # step 0 - skip known families
        if pcf_family == 'deg1':
            if coefs[0] == coefs[2] == 0:
                continue
            if coefs[0] == 0 and coefs[1] == coefs[2] == 1:
                continue
        
        # step 1 - calculate over main diag
        f, f_bar = analytical_cmfs[pcf_family]['funcs_generator'](coefs)
        mx = get_mx(f, f_bar)
        my = get_my(f, f_bar)

        pcf = mpmath.eye(2)
        for n in range(1, dept):
            pcf = mx(n,n+1) @ my(n,n) @ pcf

        if pcf[0,1] == 0:
            continue
            
        
        # step 2 - approximate limit and delta
        delta = get_delta_no_limit(pcf)
        num_limit = pcf[0,0] / pcf[0,1]
        
        # step 3 - keep only positive deltas
        if delta >0:
            skip = False
            
            # step 4 - try to discard known constants
            for const_id, const in enumerate(possible_constants):
                pslq_res = mpmath.pslq([1,const,-num_limit,-num_limit * const],tol=1e-20)
                if pslq_res is not None:
                    # print(pslq_res, const_id, nstr(const, 20))
                    skip = True
                    break
                    
            if skip == True:
                continue
            
            # step 5 - print matches
            print(mpmath.nstr(num_limit, 100))
            print(f'{pcf_family}; c={coefs}, delta[{dept}]: {mpmath.nstr(delta, 10)}\n\n')

# PRINT EVERYTHING
def get_delta(lattice_cell, limit):
    try:
        p = lattice_cell[0,0]
        q = lattice_cell[0,1]
        if q == -1:
            return mpf(-1)

        gcd = mpmath.mpf(math.gcd(int(p), int(q)))
        if gcd == 0:
            return mpf(-1)
        reduced_q = mpmath.absmin(q / gcd)
        error = mpmath.log10(mpmath.fabs((mpf(p)/mpf(q)) - limit))

        return - 1 - (error / mpmath.log10(reduced_q))

    except Exception as e:
        return mpf(-1)

import itertools
from mpmath import mpf, ln, e, nstr, pi, sqrt, zeta
import math

possible_constants = [
    (ln(2), 'ln(2)'),
    (e, 'e'),
    (e**2, 'e**2'),
    (e**0.5, 'e**0.5'),
    (pi, 'pi'),
    (zeta(3), 'zeta(3)'),
    (zeta(2), 'zeta(2)'),
    (mpf(2)**(1/mpf(3)), '2^(1/3)'),
    (mpf(2)**(2/mpf(3)), '2^(2/3)'),
    (mpf(5)**(1/mpf(3)), '5^(1/3)'),
    (mpf(5)**(2/mpf(3)), '5^(2/3)'),
    (mpf(7)**(1/mpf(3)), '7^(1/3)'),
    (mpf(14)**(1/mpf(3)), '14^(1/3)'),
    (sqrt(2), 'sqrt(2)'),
    (sqrt(3), 'sqrt(3)'),
    (sqrt(5), 'sqrt(5)'),
    (sqrt(6), 'sqrt(6)'),
    (sqrt(7), 'sqrt(7)'),
    (sqrt(8), 'sqrt(8)'),
    (sqrt(8), 'sqrt(21)')]
possible_constants += [(mpf(str(c.base.value)), c.name) for c in all_named if c.base.value and c.base.value < 1000] # see above

dept = 500

for pcf_family in analytical_cmfs:
    for coefs in itertools.product(*[range(5) for _ in range(analytical_cmfs[pcf_family]['n_coefs'])]):
        
        # step 0 - edit const list for known familes
        if pcf_family == 'deg1' and coefs[0] == coefs[2] == 0 and coefs[3] != 0:
            constants_to_scan = [(ln(1+mpf(coefs[1])/mpf(coefs[3])), f'ln(1+{coefs[1]}/{coefs[3]})')]
        
        elif pcf_family == 'deg1' and coefs[0] == 0 and coefs[1] == coefs[2] == 1 and coefs[3] != 0:
            constants_to_scan = [(mpf(coefs[3]+1)**(1/mpf(coefs[3])), f'(1+{coefs[3]})^(1/{coefs[3]})')]
        
        else:
            constants_to_scan = possible_constants
            
        # step 1 - calculate over main diag
        f, f_bar = analytical_cmfs[pcf_family]['funcs_generator'](coefs)
        mx = get_mx(f, f_bar)
        my = get_my(f, f_bar)

        pcf = mpmath.eye(2)
        for n in range(1, dept):
            pcf = mx(n,n+1) @ my(n,n) @ pcf

        if pcf[0,1] == 0:
            continue
            
        
        # step 2 - approximate limit
        num_limit = pcf[0,0] / pcf[0,1]
        if num_limit == 0:
            continue 
            
        for const, const_name in constants_to_scan:
            pslq_res = mpmath.pslq(
                [
                    1,
                    const,
                    -num_limit,
                    -num_limit * const
                ],
                tol=1e-20
            )
            if pslq_res is not None:
                break
        
        # step 3 - find delta
        if pslq_res is not None:
            fitted_limit = (pslq_res[0] + pslq_res[1]*const) / (pslq_res[2] + pslq_res[3]*const)
            delta = get_delta(pcf, fitted_limit)
            
            if delta < 0:
                continue
                
            print(f'{pcf_family}; c={coefs}, ' + \
                  f'limit=({pslq_res[0]}+{pslq_res[1]}*{const_name})/({pslq_res[2]}+{pslq_res[3]}*{const_name}), ' + \
                  f'delta[{dept}]: {mpmath.nstr(delta, 10)}\n\n')
        
        else:
            delta = get_delta_no_limit(pcf)
            
            if delta < 0:
                continue
                
            print(f'{pcf_family}; c={coefs}, ' + \
                  f'limit unknown, ' + \
                  f'delta[{dept}]: {mpmath.nstr(delta, 10)}')
            print(mpmath.nstr(num_limit, 100), '\n\n')

# Plot 𝛿 vs 𝑛
# If there is an unexpected / not smooth behaviour, this usually points to a bug

coefs = (0, 1, 2, 3)

# step 1 - calculate over main diag
f, f_bar = analytical_cmfs['deg1']['funcs_generator'](coefs)
mx = get_mx(f, f_bar)
my = get_my(f, f_bar)

pcfs = [mpmath.eye(2)]
for n in range(1, dept):
    pcfs.append(mx(n,n+1) @ my(n,n) @ pcfs[-1])

# step 2 - approximate limit
num_limit = pcfs[-1][0,0] / pcfs[-1][0,1]

const = 2**(1/mpf(3))

pslq_res = mpmath.pslq(
    [
        1,
        const,
        -num_limit,
        -num_limit * const
    ],
    tol=1e-20
)

fitted_limit = (pslq_res[0] + pslq_res[1]*const) / (pslq_res[2] + pslq_res[3]*const)
deltas = []
ns_sampled = []

sampling_distance = 5
# the first few elements are usually trash
for i in range(sampling_distance, len(pcfs)+1, sampling_distance):
    pcf = pcfs[i-1]
    ns_sampled.append(i)
    deltas.append(get_delta(pcf, fitted_limit))
    
from matplotlib import pyplot as plt
plt.figure()
plt.plot(ns_sampled, deltas, '.')
plt.xlabel('$n$')
plt.ylabel('$\delta$')