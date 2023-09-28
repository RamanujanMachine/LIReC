configuration = {
    'pool_size': 10,
    'jobs_to_run': [
        ('poly_pslq', {
            'args': {
                'degree': 2, 'order': 1,
                'min_precision': 50, 'min_roi': 2,
                'testing_precision': 16, # replaces min_precision when fed to pslq
                'bulk': 10, # if testing lots of constants, can instead limit discovery to 'bulk' constants at a time until no relevant relation is found, then another 'bulk' constants are added
                #'filters': { # the existence of filters disables antirelation logging
                #    'PcfCanonical': { 'balanced_only': True }
                #}
            },
            'run_async': True,
            'iterations': 1
        })
    ]
}
