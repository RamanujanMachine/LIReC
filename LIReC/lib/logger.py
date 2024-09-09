import logging
import logging.handlers
import os
LOG_FORMAT = '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'

def configure_logger(name, log_queue):
    if not log_queue:
        return
    
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    root = logging.getLogger()
    
    fileHandler = logging.handlers.RotatingFileHandler(f'logs/{name}.log', 'a', 1024*1024, 50)
    fileHandler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(fileHandler)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)

def print_logger(queue):
    # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    # borrowed from ^^^ and modified to my liking
    root = logging.getLogger()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(h)
    
    while True:
        record = queue.get()
        if record is None:
            break
        logging.getLogger(record.name).handle(record)
