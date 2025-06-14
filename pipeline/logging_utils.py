import logging
from pathlib import Path

LOG_FILE = Path('logs/process.log')

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ]
)

logger = logging.getLogger('dataset_kurator')

def log_step(step: str):
    logger.info(step)
