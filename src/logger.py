import logging

def get_logger():
    logger = logging.getLogger("mange_ta_main")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console output
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


logger = get_logger()
