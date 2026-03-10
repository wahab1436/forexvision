import sys
from loguru import logger

def setup_logging(log_file="logs/forexvision.log"):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, rotation="10 MB", retention="7 days", level="DEBUG")
    return logger
