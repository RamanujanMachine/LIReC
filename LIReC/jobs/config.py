configuration = {
    'jobs_to_run': [
        ('poly_pslq_v1', {
            'args': { 'degree': 3, 'order': 2, 'bulk': 1000, 'first': 0, 'last': 330, 'filters': {
                'global': { 'min_precision': 75 },
                'PcfCanonical': { 'count': 1, 'balanced_only': False, 'addons': ['delta'] },
                'Named': { 'count': 1, 'names' : ['delta', 'alpha'] }
                }
            },
            'run_async': False,
            'async_cores': 1
        }),#,
        #'poly_pslq_v2', {
        #    'args': {
        #        'degree': 2, 'order': 1,
        #        'min_precision': 50, 'min_roi': 2,
        #        'testing_precision': 15, # replaces min_precision when fed to pslq
        #        'bulk': 10, # if testing lots of constants, can instead limit discovery to 'bulk' constants at a time until no relevant relation is found, then another 'bulk' constants are added
        #        #'filters': { # the existence of filters disables antirelation logging
        #        #    'PcfCanonical': { 'balanced_only': True }
        #        #}
        #    },
        #    'run_async': True,
        #    'iterations': 1
    ]
}
