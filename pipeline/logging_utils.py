import logging
from pathlib import Path

LOG_FILE = Path("logs/process.log")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure a dedicated log level for the pipeline so that we can ignore
# noisy messages from other libraries such as Flask/Werkzeug.
DSK_LEVEL = logging.INFO + 5
logging.addLevelName(DSK_LEVEL, "DSK")


def dsk(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(DSK_LEVEL):
        self._log(DSK_LEVEL, message, args, **kwargs)


logging.Logger.dsk = dsk  # type: ignore[attr-defined]

# Silence Flask's request logs that pollute the process log.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Keep the root logger quiet and attach a handler only to our logger.
logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("dataset_kurator")
logger.setLevel(DSK_LEVEL)
handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
handler.setLevel(DSK_LEVEL)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False


def log_step(step: str) -> None:
    logger.dsk(step)
