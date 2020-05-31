import logging
def get_module_logger(modname, filename):
    logger = logging.getLogger(modname)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    filehandler = logging.FileHandler(filename, 'a+')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    return logger