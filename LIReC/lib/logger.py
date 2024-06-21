logger = None # do not use before calling configure_logger!

def configure_logger(name):
    global logger
    import os, sys, logging
    try:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        for p in sys.path:
            p2 = os.path.join(p, 'LIReC/logging.config')
            if os.path.exists(p2):
                import logging.config
                logging.config.fileConfig(p2, defaults={'log_filename': name})
                logger = logging.getLogger('job_logger')
                return
        raise Exception('logging.config file not found')
    except:
        from traceback import format_exc
        print(f'ERROR WHILE CONFIGURING LOGGER: {format_exc()}')
