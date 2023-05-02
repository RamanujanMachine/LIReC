configuration = {
    'pool_size': 10,
    'jobs_to_run': [
        ('poly_pslq', {
            'args': { 'degree': 2, 'order': 1, 'bulk': 1000, 'filters': {
                'global': { 'min_precision': 50 },
                'PcfCanonical': { 'count': 2, 'balanced_only': True },
                'Named': { 'count': 2 }
                }
            },
            'run_async': True,
            'cooldown': 30,
            'no_work_timeout': 60
        })
    ]
}
