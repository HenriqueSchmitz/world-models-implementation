import logging
import coloredlogs

def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=level, logger=logger, fmt="%(asctime)s [%(levelname)s] %(message)s", isatty=True)
    logger.info("Logger initialized.")
    return logger