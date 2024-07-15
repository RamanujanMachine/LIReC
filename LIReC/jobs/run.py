'''
'''
import signal
#import numpy as np
import os
import sys
from LIReC.jobs.config import configuration
from LIReC.lib.pool import WorkerPool

MOD_PATH = 'LIReC.jobs.job_%s'

def main() -> None:
    os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)
    worker_pool = WorkerPool()
    results = worker_pool.start([(MOD_PATH % name, config) for name, config in configuration['jobs_to_run']])

    for module_path, timings in results:
        print('-------------------------------------')
        if timings:
            print(f'module {module_path} running times:')
            print(f'min time: {min(timings)}')
            print(f'max time: {max(timings)}')
            #print(f'median time: {np.median(timings)}')
            #print(f'average time: {np.average(timings)}')
        else:
            print(f"module {module_path} didn't run! check logs")
        print('-------------------------------------')

if __name__ == '__main__':
    main()
