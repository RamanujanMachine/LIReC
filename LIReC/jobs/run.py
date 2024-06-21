'''
'''
import signal
#import numpy as np
import os
import sys
from LIReC.jobs.config import configuration
from LIReC.lib.pool import WorkerPool
from LIReC.lib.logger import *

LOGGER_NAME = 'job_logger'
MOD_PATH = 'LIReC.jobs.job_%s'

def main() -> None:
    os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)
    with open('pid.txt', 'w') as pid_file:
        pid_file.writelines([str(os.getpid()), os.linesep])
    worker_pool = WorkerPool(configuration['pool_size'])
    #signal.signal(signal.SIGINT, lambda sig, frame: worker_pool.stop())
    results = worker_pool.start([(MOD_PATH % name, config) for name, config in configuration['jobs_to_run']])
    configure_logger('main')

    for module_path, timings in results:
        logger.info('-------------------------------------')
        if timings:
            logger.info(f'module {module_path} running times:')
            logger.info(f'min time: {min(timings)}')
            logger.info(f'max time: {max(timings)}')
            #logger.info(f'median time: {np.median(timings)}')
            #logger.info(f'average time: {np.average(timings)}')
        else:
            logger.info(f"module {module_path} didn't run! check logs")
        logger.info('-------------------------------------')
        

def stop() -> None:
    print('stopping')
    with open('pid.txt', 'r') as pid_file:
        lines = pid_file.readlines()
    os.kill(int(lines[0].strip()), signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Only commands are start and stop')
        exit(1)

    if sys.argv[1] == 'stop':
        stop()
    elif sys.argv[1] == 'start':
        main()
    else:
        print('Only commands are start and stop')
        exit(1)
