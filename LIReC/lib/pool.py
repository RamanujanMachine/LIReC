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
from multiprocessing import Manager, Value, Process, Pipe
from os import cpu_count, getpid
from queue import Queue
from time import sleep, time
from traceback import format_exc
from typing import Tuple, Dict, Any
from types import ModuleType
from LIReC.lib.logger import *

NO_CRASH = True

@dataclass
class Message:
    module_id: str
    parameters: list
    
    @staticmethod
    def get_done_message() -> Message:
        return Message(None, None)
    
    @staticmethod
    def get_execution_message(module_id: str, parameters: list) -> Message:
        return Message(module_id, parameters)
    
    @property
    def is_done_message(self: Message):
        return self.module_id is None

def _import(module_id):
    return import_module(module_id[module_id.index('@')+1:])

class WorkerPool:
    manager: Manager
    job_queue: Queue
    result_queues: Dict[str, Queue]

    def start(self: WorkerPool, modules: Dict[str, Any]): # -> module_id, timings
        if len(modules) == 1 and not modules[0][1].get('run_async', False):
            module_path, module_config = modules[0]
            i = 0
            module_id = f'{i}@{module_path}'
            return [WorkerPool.run_job(None, None, None, module_id, module_config, None)]
        
        self.manager = Manager()
        self.job_queue = self.manager.Queue()
        self.log_queue = self.manager.Queue()
        self.result_queues = {}
        Process(target=print_logger, args=(self.log_queue,)).start()
        pipes = []
        for i, (module_path, module_config) in enumerate(modules):
            module_id = f'{i}@{module_path}'
            self.result_queues[module_id] = self.manager.Queue()
            result_queue = self.result_queues[module_id] # must be initialized first?
            out_pipe, in_pipe = Pipe()
            Process(
                target=WorkerPool.run_job,
                args=(self.job_queue, self.log_queue, result_queue, module_id, module_config, in_pipe)
            ).start()
            pipes += [out_pipe]
        
        jobs_left = len(modules)
        while jobs_left != 0:
            while self.job_queue.empty():
                sleep(2)
            message = self.job_queue.get()
            jobs_left -= 1
            
            if not message.is_done_message:
                result_queue = self.result_queues[message.module_id]
                Process(
                    target=WorkerPool.run_sub_job,
                    args=(message.module_id, message.parameters, self.log_queue, result_queue)
                ).start()
        
        return [p.recv() for p in pipes]

    @staticmethod
    def run_module(module: ModuleType, module_id: str, job_queue: Queue, result_queue: Queue, run_async: bool, async_cores: int, args: Dict[str, Any]) -> bool:
        if not hasattr(module, 'run_query'):
            module.execute_job(**args)
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

    @staticmethod
    def run_job(job_queue, log_queue, result_queue, module_id, module_config, in_pipe) -> Tuple[str, float]:
        try:
            configure_logger(f'query_{getpid()}', log_queue)
            module = _import(module_id)
            args = module_config.get('args', {})
            timings = []
            iterations = module_config.get('iterations', 1)
            run_async = module_config.get('run_async', False)
            async_cores = module_config.get('async_cores', 0)
            iteration = 0
            while iteration < iterations:
                start_time = time()
                WorkerPool.run_module(module, module_id, job_queue, result_queue, run_async, async_cores, args)
                if len(timings) < 30:
                    timings.append(time() - start_time)
                iteration += 1
            
            if in_pipe:
                job_queue.put(Message.get_done_message())
                in_pipe.send((module_id, timings))
            else:
                return (module_id, timings)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e # let it go through and stop everything
            logging.info(f'Error in job {module_id}: {format_exc()}')
            if in_pipe:
                in_pipe.send((module_id, []))
            else:
                return (module_id, [])

    @staticmethod
    def run_sub_job(module_id, parameters, log_queue, result_queue):
        configure_logger(f'execute_{getpid()}', log_queue)
        module = _import(module_id)
        if parameters:
            result = module.execute_job(parameters[0], **parameters[1]) if getattr(module, 'EXECUTE_NEEDS_ARGS', False) else module.execute_job(parameters)
        else:
            result = module.execute_job()
        result_queue.put(result)

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
