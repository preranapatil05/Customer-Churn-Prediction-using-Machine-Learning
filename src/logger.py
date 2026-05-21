import logging
import os
from datetime import datetime

# create logs folder
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# create log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# full log path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.DEBUG,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)