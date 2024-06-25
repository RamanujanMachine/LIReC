'''how this works:
Every job is run (at most) 'iterations' amount of times, defaults to infinity.
Every name must correspond to a file in jobs folder of the format 'job_*.py'.
In each file, the simpler case is when it doesn't have the function 'run_query',
and then it runs 'execute_job' with 'args' and finishes.
Otherwise it runs 'run_query' with 'args' and then 'execute_job' with its results,
and finally 'summarize_results' with the results of that.
Also, in the latter mode, if 'execute_job' needs access to the configuration,
setting 'EXECUTE_NEEDS_ARGS' to True in the file and specifying
all of the configuration as parameters after 'query_data' does the trick.
'''
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from math import inf, ceil
from multiprocessing import Pool, Manager, Value
from multiprocessing.managers import ValueProxy
from os import cpu_count
from queue import Queue
from time import sleep, time
from traceback import format_exc
from typing import Tuple, Dict, Any
from types import ModuleType
from LIReC.lib.logger import *

NO_CRASH = True

@dataclass
class Message:
    is_kill_message: bool
    module_id: str
    parameters: list

    @staticmethod
    def get_kill_message() -> Message:
        return Message(True, '', [])

    @staticmethod
    def get_execution_message(module_id: str, parameters: list) -> Message:
        return Message(False, module_id, parameters)


def _import(module_id):
    return import_module(module_id[module_id.index('@')+1:])

class WorkerPool:
    manager: Manager
    running: int
    job_queue: Queue
    pool: Pool
    result_queues: Dict[str, Queue]
    main_jobs: int

    def __init__(self: WorkerPool, pool_size: int = 0) -> None:
        configure_logger('pool')
        self.manager = Manager()
        self.running = self.manager.Value('i', 0)
        self.job_queue = self.manager.Queue()
        self.pool = Pool(pool_size) if pool_size else Pool() # default is cpu_count()
        self.result_queues = {}
        self.main_jobs = 0

    def stop(self: WorkerPool) -> None:
        self.running.value = 0

    def start(self: WorkerPool, modules: Dict[str, Any]): # -> module_id, timings
        self.main_jobs = len(modules)
        self.running.value = 1
        results = []

        for i, (module_path, module_config) in enumerate(modules):
            module_id = f'{i}@{module_path}'
            self.result_queues[module_id] = self.manager.Queue()
            result_queue = self.result_queues[module_id] # must be initialized first?
            results.append(self.pool.apply_async(
                WorkerPool.run_job,
                (self.running, self.job_queue, result_queue, module_id, module_config)
            ))
        
        self.read_queue()

        return [result.get() for result in results]

    def read_queue(self: WorkerPool) -> None:
        while self.main_jobs != 0:
            while self.job_queue.empty():
                sleep(2)
            message = self.job_queue.get()

            if message.is_kill_message:
                self.main_jobs -= 1
                logger.info('Killed')
            else:
                self.pool.apply_async(
                    WorkerPool.run_sub_job,
                    (message.module_id, message.parameters),
                    callback=lambda result: self.result_queues[message.module_id].put(result)
                )
        self.pool.close()
        self.pool.join()

    @staticmethod
    def run_module(module: ModuleType, module_id: str, job_queue: Queue, result_queue: Queue, run_async: bool, async_cores: int, args: Dict[str, Any]) -> bool:
        try:
            if not hasattr(module, 'run_query'):
                module.execute_job(**args)
                return True
            queried_data = module.run_query(**args)
            if queried_data == -1:
                raise Exception('internal error')
            extra_args = getattr(module, 'EXECUTE_NEEDS_ARGS', False)
            keep_unsplit = getattr(module, 'KEEP_UNSPLIT', False)
            send_index = getattr(module, 'SEND_INDEX', False)
            if not run_async:
                if send_index:
                    queried_data = (-1, 1, queried_data)
                results = [module.execute_job(queried_data, **args) if extra_args else module.execute_job(queried_data)]
            else:
                async_cores = async_cores if async_cores != 0 else cpu_count()
                for i, queried_chunk in enumerate(WorkerPool.split_parameters(queried_data, async_cores, keep_unsplit)):
                    if send_index:
                        queried_chunk = (i, async_cores, queried_chunk)
                    job_queue.put(Message.get_execution_message(module_id, (queried_chunk, args) if extra_args else queried_chunk))
                results = []
                while len(results) < async_cores:
                    results.append(result_queue.get())
            module.summarize_results(results)
            return True
        except:
            logger.info(f'Error in module {module_id}: {format_exc()}')
            return False

    @staticmethod
    def run_job(running, job_queue, result_queue, module_id, module_config) -> Tuple[str, float]:
        try:
            configure_logger('pool_run_job')
            module = _import(module_id)
            args = module_config.get('args', {})
            timings = []
            iterations = module_config.get('iterations', inf)
            run_async = module_config.get('run_async', False)
            async_cores = module_config.get('async_cores', 0)
            iteration = 0
            while running.value and iteration < iterations:
                start_time = time()
                worked = WorkerPool.run_module(module, module_id, job_queue, result_queue, run_async, async_cores, args)
                if not NO_CRASH and not worked:
                    break
                if len(timings) < 30:
                    timings.append(time() - start_time)
                iteration += 1
            
            job_queue.put(Message.get_kill_message())
            return module_id, timings
        except:
            logger.info(f'Error in job {module_id}: {format_exc()}')
            return module_id, []

    @staticmethod
    def run_sub_job(module_id, parameters):
        module = _import(module_id)
        if parameters:
            result = module.execute_job(parameters[0], **parameters[1]) if getattr(module, 'EXECUTE_NEEDS_ARGS', False) else module.execute_job(parameters)
        else:
            result = module.execute_job()
        return result

    @staticmethod
    def split_parameters(parameters, pool_size, keep_unsplit=False):
        from fractions import Fraction
        def arange(start, end=None, step=1):
            if end == None:
                end = start
                start = 0
            while start < end:
                yield start
                start += step
        if isinstance(parameters, dict):
            split = {k:(WorkerPool.split_parameters(parameters[k], pool_size, keep_unsplit) if isinstance(parameters[k], list) else parameters[k]) for k in parameters}
            return [{k:split[k][i] if isinstance(split[k], list) else split[k] for k in split} for i in range(pool_size)]
        if keep_unsplit:
            return [parameters] * pool_size
        l = max(len(parameters), 1)
        chunk_size = Fraction(l, pool_size)
        return [parameters[ceil(i):ceil(i+chunk_size)] for i in arange(0, l, chunk_size)]
