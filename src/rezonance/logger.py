import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[90m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: cyan,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(
            f"{log_fmt}{self.format_str}{self.reset}"
        )
        return formatter.format(record)


def get_logger():
    logger = logging.getLogger("mange_ta_main")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console output
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    return logger


logger = get_logger()
