# standard dependencies
import logging
from logging.handlers import RotatingFileHandler
import os

# 3rd-party dependencies
pass

# internal dependencies
pass


log_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'app.log')

MB = 1024 * 1024

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=(25 * MB),     #    25 MB per
    backupCount=4,      #   100 MB total
)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[file_handler, stream_handler]
)

def get_logger(name=None):
    return logging.getLogger(name or __name__)
